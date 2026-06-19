"""Short-window walk-forward validation + overfit haircuts — the verdict harness.

Replaces the single static 2019 split with rolling-origin walk-forward, and adds the
two López de Prado overfit controls so a result survives the MANY configs we tried.

Mechanics. Each alpha trades as its OWN market-neutral sleeve (weekly, #13 buffer, 10bps
cost): the #12 legs {rev5, mom, trend, osc} + the #19 orthogonal PEAD leg. At each step we
fit sleeve weights on a TRAILING train window (with a 1-week EMBARGO so the train block's
last 5-day forward label can't leak into the test), hold them over the next TEST window,
roll forward, and STITCH every test window into ONE continuous out-of-sample curve. The
stitched curve uses every period (full power) even when folds are tiny; a single short
fold's Sharpe (1-week test = 1 return) is noise and is never a verdict.

Weight schemes (refit each fold): equal (1/N, the control), sharpe (trailing-Sharpe,
clip>=0), mvar (max-Sharpe = shrunk-Sigma^-1 mu, clip>=0). Sweeping the TRAIN length
(incl. the short 4wk/1wk and 4mo/1mo) shows how short the window can get before the refit
weights are noise.

Rigor (haircut the best config): DEFLATED SHARPE (Bailey-Lopez de Prado) discounts the
selected config's Sharpe for the number of trials and the dispersion of their Sharpes
(answers: still significant after trying N configs?); PBO via CSCV (Combinatorially
Symmetric Cross-Validation) = P(the in-sample-best config lands below the OOS median) —
the probability the selection is overfit.

CAVEAT: current-names (survivorship) universe; inter-fold reweighting cost not modeled
(weights move slowly, leg-level turnover cost dominates). Trust the SHAPE across configs,
the recent-window read, and the haircuts.

    python walkforward.py [--names N]
"""
from __future__ import annotations

import math
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yfinance as yf  # noqa: E402
from pead_drift import load_earnings  # reuse cached earnings surprises  # noqa: E402

BENCHMARK = "SPY"
COST_RT = 0.0010
N_CANDIDATES = 220
MIN_BARS = 2500
REBAL, ENTER, EXIT = 5, 0.2, 0.4     # weekly + #13 buffer
PPY = 252 / REBAL                    # ~50.4 weekly periods/yr
INPLAY_W = 63
RECENT_WK = 156                      # ~3y "is it working now" window
EMBARGO = 1                          # weeks dropped from train tail (5-day fwd-label overlap)


# ------------------------- dependency-free stats helpers -------------------------
def _phi(x):                          # standard normal CDF
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _phi_inv(p):                      # inverse normal CDF (Acklam)
    if p <= 0:
        return -math.inf
    if p >= 1:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def per_period_sharpe(s):
    s = np.asarray(s, float); s = s[~np.isnan(s)]
    if len(s) < 8 or s.std(ddof=1) == 0:
        return float("nan")
    return s.mean() / s.std(ddof=1)


def ann_sharpe(s):
    s = np.asarray(s, float); s = s[~np.isnan(s)]
    if len(s) < 8 or s.std(ddof=1) == 0:
        return float("nan"), float("nan"), float("nan")
    sh = (s.mean() / s.std(ddof=1)) * np.sqrt(PPY)
    ann = ((1 + s.mean()) ** PPY - 1) * 100
    return sh, ann, (s > 0).mean() * 100


def deflated_sharpe(sel_ret, all_pp_sharpes, n_trials):
    """Bailey & Lopez de Prado (2014). Returns (DSR prob, SR_hat pp, SR0 haircut pp)."""
    r = np.asarray(sel_ret, float); r = r[~np.isnan(r)]
    T = len(r)
    sd = r.std(ddof=1)
    if T < 12 or sd == 0:
        return float("nan"), float("nan"), float("nan")
    sr = r.mean() / sd
    dev = r - r.mean()
    sd0 = math.sqrt((dev**2).mean())
    skew = (dev**3).mean() / sd0**3
    kurt = (dev**4).mean() / sd0**4                      # non-excess
    trials = np.asarray([x for x in all_pp_sharpes if not np.isnan(x)], float)
    var_sr = trials.var(ddof=1) if len(trials) > 1 else 0.0
    gamma = 0.5772156649015329
    N = max(2, int(n_trials))
    sr0 = math.sqrt(var_sr) * ((1 - gamma) * _phi_inv(1 - 1.0/N) + gamma * _phi_inv(1 - 1.0/(N*math.e)))
    denom = math.sqrt(max(1e-12, 1 - skew*sr + (kurt - 1)/4.0 * sr*sr))
    dsr = _phi((sr - sr0) * math.sqrt(T - 1) / denom)
    return dsr, sr, sr0


def pbo_cscv(M, S=12):
    """Probability of Backtest Overfitting via CSCV. M = T x N per-period returns."""
    T, N = M.shape
    if N < 2 or T < 2 * S:
        return float("nan"), 0
    blocks = np.array_split(np.arange(T), S)
    lam = []
    for comb in combinations(range(S), S // 2):
        is_idx = np.concatenate([blocks[b] for b in comb])
        oos_idx = np.concatenate([blocks[b] for b in range(S) if b not in comb])
        IS, OOS = M[is_idx], M[oos_idx]
        is_sh = IS.mean(0) / (IS.std(0, ddof=1) + 1e-12)
        oos_sh = OOS.mean(0) / (OOS.std(0, ddof=1) + 1e-12)
        n_star = int(np.argmax(is_sh))
        rank = oos_sh.argsort().argsort()[n_star] + 1     # 1..N ascending (N=best OOS)
        w = rank / (N + 1.0)
        lam.append(math.log(w / (1 - w)))
    lam = np.array(lam)
    return float((lam < 0).mean()), len(lam)


# ------------------------- data / sleeves -------------------------
def fetch_batch(symbols, period, interval):
    raw = yf.download(symbols, period=period, interval=interval,
                      auto_adjust=False, progress=False, group_by="ticker", threads=True)
    out = {}
    for s in symbols:
        try:
            d = raw[s].dropna(how="all").copy()
        except Exception:
            continue
        if len(d):
            if getattr(d.index, "tz", None) is not None:
                d.index = d.index.tz_localize(None)
            out[s] = d
    return out


def rsi(close, n):
    ch = close.diff()
    up = ch.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-ch).clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50)


def zrow(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)


def sleeve_returns(score, closes):
    fwd = closes.shift(-REBAL) / closes - 1
    dates = closes.index
    prev_l, prev_s = [], []
    idx, vals = [], []
    for p in range(252, len(dates) - REBAL, REBAL):
        sc = score.iloc[p].dropna()
        fz = fwd.iloc[p]
        sc = sc.loc[sc.index[fz.reindex(sc.index).notna()]]
        if len(sc) < 25:
            continue
        q = max(3, len(sc) // 5)
        pct = sc.rank(pct=True)
        keep_l = [x for x in prev_l if x in pct.index and pct[x] >= 1 - EXIT]
        add_l = [x for x in pct.sort_values(ascending=False).index if pct[x] >= 1 - ENTER and x not in keep_l]
        longs = (keep_l + add_l)[:q]
        keep_s = [x for x in prev_s if x in pct.index and pct[x] <= EXIT]
        add_s = [x for x in pct.sort_values().index if pct[x] <= ENTER and x not in keep_s]
        shorts = (keep_s + add_s)[:q]
        lr, sr = float(fz[longs].mean()), float(fz[shorts].mean())
        turn = (len(set(longs) ^ set(prev_l)) + len(set(shorts) ^ set(prev_s))) / (2 * q) if prev_l else 1.0
        prev_l, prev_s = longs, shorts
        idx.append(dates[p]); vals.append((lr - sr) - turn * COST_RT)
    return pd.Series(vals, index=idx)


def fit_weights(train, scheme):
    n = train.shape[1]
    mu = train.mean().values
    sd = train.std(ddof=1).values
    if scheme == "equal":
        return np.full(n, 1.0 / n)
    if scheme == "sharpe":
        s = np.where(sd > 0, mu / np.where(sd > 0, sd, 1.0), 0.0)
        s = np.clip(s, 0.0, None)
        return s / s.sum() if s.sum() > 0 else np.full(n, 1.0 / n)
    # mvar: max-Sharpe weights w ~ Sigma^-1 mu, with diagonal shrinkage, long-only
    cov = np.cov(train.values, rowvar=False)
    if cov.shape == ():
        cov = cov.reshape(1, 1)
    delta = 0.3
    cov_sh = (1 - delta) * cov + delta * np.diag(np.diag(cov))
    try:
        w = np.linalg.solve(cov_sh + 1e-8 * np.eye(n), mu)
    except np.linalg.LinAlgError:
        return np.full(n, 1.0 / n)
    w = np.clip(w, 0.0, None)
    return w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)


def walkforward(L, train_w, test_w, scheme, embargo=EMBARGO):
    vals, idx = [], []
    i = train_w
    while i < len(L):
        train = L.iloc[i - train_w:i - embargo] if embargo and train_w - embargo >= 4 else L.iloc[i - train_w:i]
        w = fit_weights(train, scheme)
        te = L.iloc[i:i + test_w]
        if len(te):
            for d, r in zip(te.index, te.values @ w):
                idx.append(d); vals.append(r)
        i += test_w
    return pd.Series(vals, index=idx)


def main():
    import argparse
    global N_CANDIDATES
    ap = argparse.ArgumentParser(description="Short-window walk-forward + overfit haircuts")
    ap.add_argument("--names", type=int, default=N_CANDIDATES)
    args = ap.parse_args()
    N_CANDIDATES = args.names

    from stock_symbols_1243 import STOCK_SYMBOLS
    cands = [s for s in dict.fromkeys(STOCK_SYMBOLS) if s != BENCHMARK][:N_CANDIDATES]
    print(f"downloading {len(cands)} + {BENCHMARK} (20y daily) ...", flush=True)
    data = {}
    for k in range(0, len(cands), 110):
        data.update(fetch_batch(cands[k:k + 110], "20y", "1d"))
    spy = fetch_batch([BENCHMARK], "20y", "1d").get(BENCHMARK)
    keep = {s: d for s, d in data.items() if len(d) >= MIN_BARS and float(d["Close"].iloc[-1]) > 5}
    closes = pd.DataFrame({s: d["Close"].astype(float) for s, d in keep.items()})
    print(f"universe: {closes.shape[1]} names, {closes.index[0].date()} -> {closes.index[-1].date()}")

    legs = {
        "rev5": zrow(-closes.pct_change(5, fill_method=None)),
        "mom": zrow(closes.shift(21) / closes.shift(252) - 1),
        "trend": zrow(closes / closes.rolling(200).mean() - 1),
        "osc": zrow(-closes.apply(lambda c: rsi(c, 2))),
    }
    earnings = load_earnings(list(closes.columns), use_cache=True)
    dvals = closes.index.values
    sm = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
    for sym, evs in earnings.items():
        if sym not in sm.columns:
            continue
        col = sm.columns.get_loc(sym)
        for dstr, sp in evs:
            e = int(np.searchsorted(dvals, np.datetime64(pd.Timestamp(dstr).normalize()), side="right"))
            if 0 <= e < len(dvals):
                sm.iat[e, col] = sp
    legs["pead"] = zrow(sm.ffill(limit=INPLAY_W))

    L = pd.DataFrame({k: sleeve_returns(v, closes) for k, v in legs.items()}).dropna()
    print(f"aligned sleeve weeks: {len(L)}  ({L.index[0].date()} -> {L.index[-1].date()})")
    print("per-sleeve full-sample Sharpe: " + "  ".join(f"{k} {ann_sharpe(L[k])[0]:.2f}" for k in L.columns))
    base_sh, base_ann, base_pos = ann_sharpe(L.mean(axis=1))
    print(f"\nbaseline (static equal-weight 5 sleeves, whole period): Sharpe {base_sh:.2f}  "
          f"ann {base_ann:+.1f}%  %+ {base_pos:.0f}  | recent-3y Sharpe {ann_sharpe(L.mean(axis=1).iloc[-RECENT_WK:])[0]:.2f}")

    configs = [
        ("4wk -> 1wk", 4, 1), ("8wk -> 1wk", 8, 1),
        ("17wk(4mo) -> 4wk(1mo)", 17, 4), ("26wk(6mo) -> 4wk", 26, 4),
        ("52wk(1y) -> 4wk", 52, 4), ("104wk(2y) -> 13wk(1q)", 104, 13),
    ]
    print(f"\n{'walk-forward config':<24}{'scheme':<8}{'OOSwk':>7}{'OOSshrp':>9}{'OOSann%':>9}{'%+':>6}{'RECENT3y':>10}")
    trials = {}                                   # (label,scheme) -> oos series
    for label, tw, sw in configs:
        for scheme in ("equal", "sharpe", "mvar"):
            oos = walkforward(L, tw, sw, scheme)
            trials[(label, scheme)] = oos
            sh, ann, pos = ann_sharpe(oos)
            rec = ann_sharpe(oos.iloc[-RECENT_WK:])[0] if len(oos) > RECENT_WK else float("nan")
            print(f"{label:<24}{scheme:<8}{len(oos):>7}{sh:>9.2f}{ann:>9.1f}{pos:>6.0f}{rec:>10.2f}")

    # ------------------------- RIGOR: deflate the best config -------------------------
    pp = {k: per_period_sharpe(v) for k, v in trials.items()}
    best_key = max(pp, key=lambda k: pp[k] if not np.isnan(pp[k]) else -9)
    dsr, sr_pp, sr0 = deflated_sharpe(trials[best_key].values, list(pp.values()), n_trials=len(trials))
    # PBO across all trial OOS curves, aligned on common dates
    common = None
    for v in trials.values():
        common = v.index if common is None else common.intersection(v.index)
    M = np.column_stack([trials[k].reindex(common).values for k in trials])
    mask = ~np.isnan(M).any(axis=1)
    pbo, ncomb = pbo_cscv(M[mask], S=12)
    best_ann = ann_sharpe(trials[best_key].values)[0]
    print("\n================ OVERFIT HAIRCUTS ================")
    print(f"  trials evaluated (N): {len(trials)}   |   best config: {best_key[0]} / {best_key[1]} "
          f"(OOS Sharpe {best_ann:.2f})")
    print(f"  Deflated Sharpe: per-period SR {sr_pp:.3f} vs selection-haircut SR0 {sr0:.3f}  ->  "
          f"DSR = {dsr:.2f}  ({'PASS >0.95' if dsr > 0.95 else 'significant' if dsr > 0.90 else 'NOT significant after haircut'})")
    print(f"  PBO (CSCV, {ncomb} combos over {int(mask.sum())} aligned wks): {pbo:.2f}  "
          f"({'low overfit risk' if pbo < 0.25 else 'moderate' if pbo < 0.5 else 'HIGH overfit risk'})")
    print("\nReads: OOSshrp = stitched out-of-sample Sharpe (full power, the trustworthy number).")
    print("Compare 'sharpe'/'mvar' vs 'equal' per train length: if they only beat equal at >=17-26wk,")
    print("short windows are too noisy to estimate weights. DSR>0.95 = survives the config search;")
    print("PBO<0.25 = selecting the best config is not overfit. RECENT3y = is it working now.")


if __name__ == "__main__":
    main()
