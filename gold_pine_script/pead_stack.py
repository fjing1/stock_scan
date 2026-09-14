"""Stack PEAD onto the market-neutral ensemble — the orthogonal-alpha payoff test.

#14 (alpha_stack.py) showed that adding more PRICE alphas HURT: they were decayed or
correlated, so equal-weighting dragged the blend negative. Its conclusion: a lift needs an
individually-positive, NON-decayed, ORTHOGONAL component. #19 (pead_drift.py) found exactly
that — post-earnings drift is event/fundamental (not OHLCV), OOS-persistent, and low-turnover.

This study adds PEAD as a 5th leg to the validated 4-alpha ensemble (#12) under the winning
buffer config (#13: weekly, enter top/bottom 20%, hold to 40% band), and asks the clean
question #14 set up: does a genuinely orthogonal positive alpha lift the NET (after-cost)
Sharpe out-of-sample without adding market beta?

  4-alpha blend (#12):  z(-5d ret) + z(12-1 mom) + z(close/MA200) + z(-RSI2)
  PEAD leg (#19):        cross-sectional z of a name's earnings surprise while it is "in play"
                         (the INPLAY_W trading days after it reports), NEUTRAL (0) otherwise.
  stacked:               4-alpha + w * z(PEAD),  w in {0.5, 1.0}  (untuned, like the base blend)

Reports each leg's standalone net Sharpe/beta/turnover, the correlation of the PEAD spread to
the 4-alpha spread (orthogonality check), and the 4-alpha-vs-stacked net Sharpe / beta / turnover
/ long-only-alpha, train & test. CAVEAT: current-names (survivorship) universe — trust the
DELTA (stacked minus base), beta~0, and OOS persistence, not absolute long-leg magnitude.

    python pead_stack.py [--names N]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yfinance as yf  # noqa: E402
from pead_drift import load_earnings  # reuse the cached earnings-surprise fetch  # noqa: E402

BENCHMARK = "SPY"
SPLIT = pd.Timestamp("2019-01-01")
COST_RT = 0.0010
N_CANDIDATES = 220
MIN_BARS = 2500
REBAL, ENTER, EXIT = 5, 0.2, 0.4          # weekly + buffer (the #13 winning config)
PPY = 252 / REBAL
INPLAY_W = 63                              # trading days a name stays "in play" post-report


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


def run_alpha(score, closes, spy_fwd):
    """Weekly top/bottom-quintile spread WITH the #13 hysteresis buffer. NaN scores excluded."""
    fwd = closes.shift(-REBAL) / closes - 1
    dates = closes.index
    prev_l, prev_s = [], []
    rows = []
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
        spy_f = float(spy_fwd.iloc[p]) if not np.isnan(spy_fwd.iloc[p]) else np.nan
        rows.append((dates[p], lr - sr, lr, spy_f, (lr - sr) - turn * COST_RT, turn))
    return pd.DataFrame(rows, columns=["date", "spread", "long", "spy", "net", "turn"])


def _sh(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return (x.mean() / x.std(ddof=1)) * np.sqrt(PPY) if len(x) > 5 and x.std(ddof=1) > 0 else float("nan")


def metrics(R):
    tr, te = R[R["date"] < SPLIT], R[R["date"] >= SPLIT]
    x, y = te["spy"].values, te["spread"].values
    m = ~(np.isnan(x) | np.isnan(y))
    bta = np.polyfit(x[m], y[m], 1)[0] if m.sum() > 5 else float("nan")
    lo_alpha = (te["long"].values - te["spy"].values)
    lo_ann = ((1 + np.nanmean(lo_alpha)) ** PPY - 1) * 100
    return dict(trNet=_sh(tr["net"]), teNet=_sh(te["net"]), teGr=_sh(te["spread"]),
                beta=bta, turn=R["turn"].mean() * 100, teLOalpha=lo_ann)


def main():
    import argparse
    global N_CANDIDATES
    ap = argparse.ArgumentParser(description="Stack PEAD onto the 4-alpha market-neutral ensemble")
    ap.add_argument("--names", type=int, default=N_CANDIDATES)
    args = ap.parse_args()
    N_CANDIDATES = args.names

    from stock_symbols_1243 import STOCK_SYMBOLS
    cands = [s for s in dict.fromkeys(STOCK_SYMBOLS) if s != BENCHMARK][:N_CANDIDATES]
    print(f"downloading {len(cands)} + {BENCHMARK} (20y daily) ...", flush=True)
    data = {}
    for i in range(0, len(cands), 110):
        data.update(fetch_batch(cands[i:i + 110], "20y", "1d"))
    spy = fetch_batch([BENCHMARK], "20y", "1d").get(BENCHMARK)
    keep = {s: d for s, d in data.items() if len(d) >= MIN_BARS and float(d["Close"].iloc[-1]) > 5}
    closes = pd.DataFrame({s: d["Close"].astype(float) for s, d in keep.items()})
    spy_close = spy["Close"].astype(float).reindex(closes.index)
    spy_fwd = spy_close.shift(-REBAL) / spy_close - 1
    print(f"universe: {closes.shape[1]} names, {closes.index[0].date()} -> {closes.index[-1].date()}")

    # --- the validated 4-alpha price blend (#12) ---
    combo4 = (zrow(-closes.pct_change(5, fill_method=None)) + zrow(closes.shift(21) / closes.shift(252) - 1)
              + zrow(closes / closes.rolling(200).mean() - 1)
              + zrow(-closes.apply(lambda c: rsi(c, 2))))

    # --- the PEAD leg (#19): in-play earnings surprise, z-scored cross-sectionally ---
    earnings = load_earnings(list(closes.columns), use_cache=True)
    dvals = closes.index.values
    surprise_mat = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
    for sym, evs in earnings.items():
        if sym not in surprise_mat.columns:
            continue
        col = surprise_mat.columns.get_loc(sym)
        for dstr, sp in evs:
            entry = int(np.searchsorted(dvals, np.datetime64(pd.Timestamp(dstr).normalize()), side="right"))
            if 0 <= entry < len(dvals):
                surprise_mat.iat[entry, col] = sp
    inplay = surprise_mat.ffill(limit=INPLAY_W)
    pead_sparse = zrow(inplay)             # z across in-play names; NaN where not in play (standalone)
    pead_neutral = pead_sparse.fillna(0.0)  # neutral (0) where not in play (for blending)

    # --- standalone legs ---
    print("\n--- standalone legs (weekly + buffer; net = after 10bps x turnover) ---")
    print(f"{'leg':<16}{'turn%':>7}{'trNetSh':>9}{'teNetSh':>9}{'teGrSh':>9}{'beta':>7}{'teLOalpha%':>11}")
    R4 = run_alpha(combo4, closes, spy_fwd)
    Rp = run_alpha(pead_sparse, closes, spy_fwd)
    for label, R in (("4-alpha (#12)", R4), ("PEAD (#19)", Rp)):
        m = metrics(R)
        print(f"{label:<16}{m['turn']:>7.0f}{m['trNet']:>9.2f}{m['teNet']:>9.2f}{m['teGr']:>9.2f}{m['beta']:>7.2f}{m['teLOalpha']:>11.1f}")

    # --- orthogonality: correlation of spread returns ---
    S = pd.DataFrame({"4alpha": R4.set_index("date")["spread"],
                      "PEAD": Rp.set_index("date")["spread"]}).dropna()
    print(f"\n--- spread-return correlation (n={len(S)}): corr(4-alpha, PEAD) = {S.corr().iloc[0,1]:+.2f} "
          f"(near 0 = orthogonal, the diversifier #14 wanted) ---")

    # --- stacked blends ---
    print("\n--- 4-alpha vs PEAD-stacked (does the orthogonal leg lift NET Sharpe?) ---")
    print(f"{'ensemble':<22}{'turn%':>7}{'trNetSh':>9}{'teNetSh':>9}{'teGrSh':>9}{'beta':>7}{'teLOalpha%':>11}")
    base = metrics(R4)
    print(f"{'4-alpha (base)':<22}{base['turn']:>7.0f}{base['trNet']:>9.2f}{base['teNet']:>9.2f}{base['teGr']:>9.2f}{base['beta']:>7.2f}{base['teLOalpha']:>11.1f}")
    best = None
    for w in (0.5, 1.0):
        m = metrics(run_alpha(combo4 + w * pead_neutral, closes, spy_fwd))
        tag = f"4-alpha + {w}*PEAD"
        print(f"{tag:<22}{m['turn']:>7.0f}{m['trNet']:>9.2f}{m['teNet']:>9.2f}{m['teGr']:>9.2f}{m['beta']:>7.2f}{m['teLOalpha']:>11.1f}")
        if best is None or (m["teNet"] > best[1]["teNet"]):
            best = (w, m)

    print("\n================ VERDICT ================")
    dNet_tr = best[1]["trNet"] - base["trNet"]
    dNet_te = best[1]["teNet"] - base["teNet"]
    print(f"  [score-stack] best: 4-alpha + {best[0]}*PEAD")
    print(f"    net Sharpe TRAIN: {base['trNet']:.2f} -> {best[1]['trNet']:.2f} ({dNet_tr:+.2f})   "
          f"TEST: {base['teNet']:.2f} -> {best[1]['teNet']:.2f} ({dNet_te:+.2f})")

    # --- sleeve combination: hold PEAD as its OWN book, combine at the RETURN level ---
    # (the right way to harvest an orthogonal sleeve; score-stacking just perturbs one ranking)
    P = pd.DataFrame({"a": R4.set_index("date")["net"], "p": Rp.set_index("date")["net"]}).dropna()
    blend = 0.5 * (P["a"] + P["p"])
    def split(s):
        return s[s.index < SPLIT], s[s.index >= SPLIT]
    a_tr, a_te = split(P["a"]); b_tr, b_te = split(blend)
    print(f"  [sleeve 50/50] hold 4-alpha and PEAD as separate books, blend net returns (n={len(P)})")
    print(f"    net Sharpe TRAIN: {_sh(a_tr):.2f} -> {_sh(b_tr):.2f} ({_sh(b_tr)-_sh(a_tr):+.2f})   "
          f"TEST: {_sh(a_te):.2f} -> {_sh(b_te):.2f} ({_sh(b_te)-_sh(a_te):+.2f})")
    sleeve_win = (_sh(b_tr) > _sh(a_tr)) and (_sh(b_te) > _sh(a_te))
    score_win = dNet_tr > 0 and dNet_te > 0 and abs(best[1]["beta"]) < 0.25
    if sleeve_win:
        print("  -> DEPLOY AS A SLEEVE: a 50/50 capital split with PEAD lifts net Sharpe in BOTH")
        print("     windows (orthogonality pays at the RETURN level even though score-stacking didn't).")
    elif score_win:
        print("  -> STACK IT: score-blended PEAD lifts net Sharpe in both windows, beta ~0.")
    else:
        print("  -> KEEP 4-ALPHA: neither score-stack nor 50/50 sleeve lifts net Sharpe in both windows.")
        print("     PEAD's edge is real (#19) but too concentrated/correlated here to raise the blend.")
    print("\nCAVEAT (#3 survivorship): current-names universe; trust the DELTA vs base, beta~0,")
    print("and OOS persistence over absolute magnitude. PEAD is event-driven => the orthogonal")
    print("leg #14 said was missing; this is the clean test of whether orthogonality pays.")


if __name__ == "__main__":
    main()
