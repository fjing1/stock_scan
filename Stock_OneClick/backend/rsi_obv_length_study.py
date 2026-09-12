#!/usr/bin/env python3
"""DEEP DIVE: what RSI length is right when the RSI is sourced from OBV, and what is that
oscillator actually good for on a daily chart?

Motivation: xunlong_panel.pine runs RSI(14) on OBV and smooths it with SMA 8/22/85. Every
prior study here tested a USE of that line (MA crossover: rsi_obv_ma_compare.py; pivot
divergence: rsi_obv_divergence.py) and both came back with no edge. Neither asked the prior
question -- whether the LENGTH was wrong, or whether the oscillator carries information at
all in some other form. This does.

Part 1  IDENTITY & CHARACTER
        RSI(n) on standard OBV is not a momentum oscillator. Because OBV's first difference
        is exactly sign(Dclose)*volume, the RSI's up/down averages are up-day volume and
        down-day volume, so:
              RSI(OBV, n) == 100 * rma(up-volume, n) / (rma(up-volume, n) + rma(down-volume, n))
        i.e. a Wilder-smoothed UP-VOLUME SHARE. Verified numerically here. That reframing
        changes what "overbought" means and what lengths are sensible.

Part 2  INFORMATION CONTENT (the real "which length" test)
        Cross-sectional rank IC: each day rank every name by the feature, correlate with
        forward returns at h = 1/5/10/20 days. This measures screening/ranking value without
        baking in a strategy, so it can't be flattered by rule construction. Swept over
        length. Run for the LEVEL, the SLOPE, and the OBV-minus-PRICE spread, and -- the
        crux -- for OBV-RSI after neutralizing price-RSI, which asks whether OBV as a source
        adds anything a price RSI doesn't already say.

Part 3  SHAPE
        Decile forward returns for the best length: monotone, or only the tails?

Part 4  RULES
        Long-only backtests swept over length (trend rule, mean-reversion rule) vs buy-hold,
        15bps/turn, next-day fill, with per-year walk-forward.

Part 5  HAIRCUT
        Deflated Sharpe over the whole trial count, plus a split-sample PBO check, because
        Parts 2-4 try hundreds of configurations ([[prefers-walkforward-over-static-split]]).

    ../../vcp_env/bin/python _rsi_obv_len_data.py         # build the panel first
    ../../vcp_env/bin/python rsi_obv_length_study.py
    ../../vcp_env/bin/python rsi_obv_length_study.py --group broad --part ic
"""
from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
from _rsi_obv_len_data import load  # noqa: E402

COST = 0.0015
TD = 252
LENGTHS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 17, 21, 25, 30, 40, 50]
HORIZONS = [1, 5, 10, 20]
SLOPE_LB = 5


# ---------------------------------------------------------------- stats helpers (no scipy)
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Acklam's rational approximation to the inverse normal CDF (|err| < 1.15e-9)."""
    if not 0.0 < p < 1.0:
        return float("nan")
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


def deflated_sharpe(sr: float, n_obs: int, n_trials: int, sr_var: float,
                    skew: float, kurt: float) -> float:
    """Bailey & Lopez de Prado's Deflated Sharpe Ratio: the probability the observed Sharpe
    is real once you account for having searched n_trials configurations. sr/sr_var are in
    ANNUALIZED units here; converted to per-observation internally."""
    if n_obs < 10 or sr_var <= 0:
        return float("nan")
    sr_p = sr / math.sqrt(TD)                       # per-period Sharpe
    sd_p = math.sqrt(sr_var) / math.sqrt(TD)
    gamma = 0.5772156649
    e = math.e
    z1 = norm_ppf(1 - 1.0 / n_trials)
    z2 = norm_ppf(1 - 1.0 / (n_trials * e))
    sr0 = sd_p * ((1 - gamma) * z1 + gamma * z2)    # expected max Sharpe under the null
    denom = 1 - skew * sr_p + (kurt - 1) / 4.0 * sr_p ** 2
    if denom <= 0:
        return float("nan")
    return norm_cdf((sr_p - sr0) * math.sqrt(n_obs - 1) / math.sqrt(denom))


def metrics(daily: pd.Series) -> dict:
    d = daily.dropna()
    if len(d) < 2:
        return {}
    eq = (1 + d).cumprod()
    yrs = len(d) / TD
    sd = d.std(ddof=1)
    return {"total": eq.iloc[-1] - 1,
            "cagr": eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan,
            "sharpe": (d.mean() / sd * math.sqrt(TD)) if sd else np.nan,
            "maxdd": (eq / eq.cummax() - 1).min()}


# ---------------------------------------------------------------- indicator maths
def rma(x, n):
    return x.ewm(alpha=1 / n, adjust=False).mean()


def obv_parts(close: pd.DataFrame, volume: pd.DataFrame):
    """Returns (obv, up_volume, down_volume) as wide frames.

    sign() of the first bar's (undefined) change is treated as 0, matching the scanner's
    obv_series(). Pre-history is zero-filled rather than NaN so the ewm can run column-wise;
    because BOTH up and down start at zero that warm-up is equivalent to starting fresh at
    the name's first bar, and the burn-in mask discards it regardless."""
    valid = close.notna() & volume.notna()
    d_obv = (np.sign(close.diff().fillna(0.0)) * volume).where(valid, 0.0)
    return d_obv.cumsum().where(valid), d_obv.clip(lower=0), (-d_obv).clip(lower=0)


def rsi_on(src: pd.DataFrame, n: int) -> pd.DataFrame:
    d = src.diff()
    up = rma(d.clip(lower=0), n)
    dn = rma((-d).clip(lower=0), n)
    return (100 - 100 / (1 + up / dn)).replace([np.inf, -np.inf], np.nan)


def rsi_obv(upv: pd.DataFrame, dnv: pd.DataFrame, n: int) -> pd.DataFrame:
    """The OBV-sourced RSI in its exact closed form: Wilder-smoothed up-volume share.

    Identical to rsi_on(obv, n) away from the series boundary (verified in Part 1), but
    computed straight from the up/down volume so there is no first-bar warm-up discrepancy
    from OBV's leading NaN."""
    u, d = rma(upv, n), rma(dnv, n)
    return (100 * u / (u + d)).replace([np.inf, -np.inf], np.nan)


def burn_mask(close: pd.DataFrame, burn: int) -> pd.DataFrame:
    """True once a name has at least `burn` real bars behind it -- drops indicator warm-up,
    which is where every smoothed series is least trustworthy."""
    return close.notna().cumsum() > burn


# ---------------------------------------------------------------- IC machinery
def xs_ic(feat: pd.DataFrame, fwd: pd.DataFrame, min_names: int = 40) -> pd.Series:
    """Daily cross-sectional Spearman IC between a feature and forward returns."""
    m = feat.notna() & fwd.notna()
    keep = m.sum(axis=1) >= min_names
    if not keep.any():
        return pd.Series(dtype=float)
    f = feat.where(m)[keep]
    r = fwd.where(m)[keep]
    rf = f.rank(axis=1)
    rr = r.rank(axis=1)
    a = rf.sub(rf.mean(axis=1), axis=0)
    b = rr.sub(rr.mean(axis=1), axis=0)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))
    return (num / den).replace([np.inf, -np.inf], np.nan).dropna()


def ic_stats(ic: pd.Series, h: int) -> tuple[float, float]:
    """(mean IC, t-stat). Overlapping h-day forward windows make consecutive ICs dependent,
    so the effective sample is n/h, not n -- ignoring that inflates t by sqrt(h)."""
    if len(ic) < 10 or ic.std(ddof=1) == 0:
        return float("nan"), float("nan")
    n_eff = max(2.0, len(ic) / h)
    return ic.mean(), ic.mean() / ic.std(ddof=1) * math.sqrt(n_eff)


def neutralize(feat: pd.DataFrame, ctrl: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectionally regress feat's ranks on ctrl's ranks each day, return the residual.
    Answers: what does OBV-RSI know that price-RSI does not already say?"""
    m = feat.notna() & ctrl.notna()
    f = feat.where(m).rank(axis=1)
    c = ctrl.where(m).rank(axis=1)
    f = f.sub(f.mean(axis=1), axis=0)
    c = c.sub(c.mean(axis=1), axis=0)
    beta = (f * c).sum(axis=1) / (c ** 2).sum(axis=1)
    return f - c.mul(beta, axis=0)


# ---------------------------------------------------------------- rule backtests
def apply_positions(pos: pd.DataFrame, ret: pd.DataFrame) -> pd.Series:
    """Equal-weight pooled daily net return. Position acts next day; 15bps per turn."""
    p = pos.shift(1)
    turn = p.diff().abs()
    return (p * ret - turn * COST).mean(axis=1)


def mr_positions(rsi: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    """Mean-reversion state machine: enter below lo, exit above hi, hold in between."""
    raw = pd.DataFrame(np.where(rsi < lo, 1.0, np.where(rsi > hi, 0.0, np.nan)),
                       index=rsi.index, columns=rsi.columns)
    return raw.ffill().fillna(0.0)


# ---------------------------------------------------------------- parts
def part_identity(close, volume, obv, upv, dnv, mask):
    print("\n" + "=" * 96)
    print("PART 1 — WHAT THIS OSCILLATOR ACTUALLY IS")
    print("=" * 96)
    n = 14
    # (a) the substantive claim: OBV's first difference IS the signed volume, so the RSI's
    #     "up moves" and "down moves" are literally up-day and down-day volume.
    interior = obv.notna() & obv.shift(1).notna()
    d_check = (obv.diff() - (upv - dnv)).abs().where(interior)
    print(f"\n  (a) is  d(OBV)  ==  sign(dclose) * volume  ?")
    print(f"      max abs difference on interior bars: {d_check.max().max():.2e}  "
          f"-> {'EXACT' if d_check.max().max() < 1e-6 else 'NOT exact'}")
    # (b) hence RSI(OBV) is the smoothed up-volume share. Algebraically exact:
    #     100 - 100/(1 + u/d) == 100u/(u+d). Residual disagreement can only come from bars
    #     where OBV is NaN (gaps/first bar) and the two paths warm up differently.
    lhs = rsi_on(obv, n).where(mask)
    rhs = rsi_obv(upv, dnv, n).where(mask)
    diff = (lhs - rhs).abs().stack().dropna()
    agree = (diff < 1e-6).mean() * 100
    print(f"\n  (b) RSI(OBV,{n})  vs  100*rma(up-vol)/(rma(up-vol)+rma(down-vol))")
    print(f"      agree to 1e-6 on {agree:.3f}% of {len(diff):,} name-days; "
          f"p99.9 diff {diff.quantile(0.999):.2e}, max {diff.max():.2e}")
    print(f"      (algebraically identical; the residual cells are names with data gaps,")
    print(f"       where OBV goes NaN and the two smoothings restart differently)")
    print("    So the 'RSI' is a Wilder-smoothed UP-VOLUME SHARE, not a price-momentum")
    print("    oscillator. 50 = up-volume and down-volume balanced. It says nothing directly")
    print("    about how far price has travelled, only about which side the volume fell on.")

    print(f"\n  character by length (pooled over all names/days):")
    print(f"    {'n':>4} | {'--------- RSI on OBV ---------':^46} | {'--- RSI on PRICE ---':^30}")
    print(f"    {'':>4} | {'mean':>7}{'sd':>7}{'ac1':>7}{'>70%':>7}{'<30%':>7}{'x50/yr':>8} | "
          f"{'mean':>7}{'sd':>7}{'ac1':>7}{'x50/yr':>8}")
    prsi_cache = {}
    for n in LENGTHS:
        ro = rsi_obv(upv, dnv, n).where(mask)
        rp = rsi_on(close, n).where(mask)
        prsi_cache[n] = rp
        def stats(r):
            v = r.stack()
            ac = r.corrwith(r.shift(1)).mean()
            cross = ((r > 50) != (r.shift(1) > 50)).sum().sum() / max(1, r.notna().sum().sum()) * TD
            return v.mean(), v.std(), ac, (v > 70).mean() * 100, (v < 30).mean() * 100, cross
        mo, so, ao, hi_o, lo_o, xo = stats(ro)
        mp, sp, ap, _, _, xp = stats(rp)
        print(f"    {n:>4} | {mo:>7.1f}{so:>7.1f}{ao:>7.3f}{hi_o:>7.1f}{lo_o:>7.1f}{xo:>8.1f} | "
              f"{mp:>7.1f}{sp:>7.1f}{ap:>7.3f}{xp:>8.1f}")
    print("\n    ac1 = lag-1 autocorrelation (how sticky the line is)")
    print("    x50/yr = times per year it crosses 50 = how often any 50-line rule would trade")
    return prsi_cache


def part_ic(robv, prsi_cache, fwd):
    print("\n" + "=" * 96)
    print("PART 2 — DOES IT PREDICT ANYTHING? (cross-sectional rank IC, by length)")
    print("=" * 96)
    print("\n  A positive IC means high readings precede high forward returns. |IC| < 0.01 is")
    print("  noise; equity factors that are worth trading usually run 0.02-0.05 with |t| > 3.")

    best = {}
    for label, builder in [
        ("LEVEL   RSI(OBV,n)", lambda n: robv(n)),
        (f"SLOPE   {SLOPE_LB}d change", lambda n: robv(n).diff(SLOPE_LB)),
        ("SPREAD  OBV - PRICE", lambda n: robv(n) - prsi_cache[n]),
        ("RESID   OBV | PRICE", lambda n: neutralize(robv(n), prsi_cache[n])),
        ("BASELINE RSI(PRICE,n)", lambda n: prsi_cache[n]),
    ]:
        print(f"\n  --- {label} ---")
        print(f"    {'n':>4}" + "".join([f"{'IC h=' + str(h):>12}{'t':>7}" for h in HORIZONS]))
        rows = []
        for n in LENGTHS:
            feat = builder(n)
            cells, rec = [], {"n": n}
            for h in HORIZONS:
                ic, t = ic_stats(xs_ic(feat, fwd[h]), h)
                rec[h] = (ic, t)
                cells.append(f"{ic:>12.4f}{t:>7.1f}")
            rows.append(rec)
            print(f"    {n:>4}" + "".join(cells))
        # strongest |IC| at the 5- and 20-day horizons, the tradeable ones
        for h in (5, 20):
            b = max(rows, key=lambda r: abs(r[h][0]) if np.isfinite(r[h][0]) else -1)
            best[(label, h)] = (b["n"], b[h][0], b[h][1])
        b5 = best[(label, 5)]
        print(f"    -> strongest at h=5: n={b5[0]}  IC {b5[1]:+.4f}  t {b5[2]:+.1f}")
    return best


def part_panel(robv, prsi_cache, fwd):
    """Evaluate the line the Pine panel ACTUALLY plots and alerts on.

    The panel doesn't trade raw RSI(OBV,14) -- it plots SMA(rsi,8)/22/85 and fires its buy
    on ma8 crossing above ma22. Smoothing adds lag, so the traded line behaves like a much
    longer effective length; Part 2 says longer = less information, so this quantifies the
    cost of the smoothing and checks the sign of the panel's actual trigger quantity."""
    print("\n" + "=" * 96)
    print("PART 2b — THE PANEL'S ACTUAL LINES (RSI(OBV,14) smoothed by SMA 8/22)")
    print("=" * 96)
    r14 = robv(14)
    variants = [
        ("raw RSI(OBV,14)", r14),
        ("ma8  = SMA(rsi,8)", r14.rolling(8).mean()),
        ("ma22 = SMA(rsi,22)", r14.rolling(22).mean()),
        ("ma8 - ma22  (the buy trigger quantity)", r14.rolling(8).mean() - r14.rolling(22).mean()),
        ("raw RSI(PRICE,14), for reference", prsi_cache[14]),
    ]
    print(f"\n    {'series':<40}" + "".join([f"{'IC h=' + str(h):>12}{'t':>7}" for h in HORIZONS]))
    for label, feat in variants:
        cells = []
        for h in HORIZONS:
            ic, t = ic_stats(xs_ic(feat, fwd[h]), h)
            cells.append(f"{ic:>12.4f}{t:>7.1f}")
        print(f"    {label:<40}" + "".join(cells))
    print("\n    The panel's buy fires when (ma8 - ma22) crosses from negative to positive, i.e.")
    print("    it BUYS high values of the last row's quantity. If that row's IC is negative,")
    print("    the alert is oriented against the measured effect.")


def part_shape(robv, fwd, n_best: int, h: int = 5):
    print("\n" + "=" * 96)
    print(f"PART 3 — SHAPE OF THE EFFECT: decile forward returns, RSI(OBV,{n_best}), h={h}d")
    print("=" * 96)
    r = robv(n_best)
    f = fwd[h]
    m = r.notna() & f.notna()
    rr = r.where(m).rank(axis=1, pct=True)
    dec = (rr * 10).clip(upper=9.999).apply(np.floor)
    print(f"\n    {'decile':>8}{'mean fwd':>12}{'median':>11}{'win%':>8}{'n':>10}")
    means = []
    for d in range(10):
        sel = f.where(dec == d)
        v = sel.stack()
        if len(v) < 100:
            continue
        means.append(v.mean())
        print(f"    {d:>8}{v.mean():>+12.3%}{v.median():>+11.3%}{(v > 0).mean() * 100:>8.1f}{len(v):>10,}")
    if len(means) == 10:
        spread = means[9] - means[0]
        mono = np.corrcoef(np.arange(10), means)[0, 1]
        print(f"\n    top-minus-bottom decile: {spread:+.3%} per {h}d   "
              f"monotonicity (corr of decile vs mean): {mono:+.2f}")
        print("    A real factor is roughly monotone. A big spread with low monotonicity means")
        print("    only the extremes matter -- usually a liquidity/volatility artifact.")


def part_rules(robv, ret):
    print("\n" + "=" * 96)
    print("PART 4 — TRADEABLE RULES SWEPT OVER LENGTH (long-only, 15bps/turn, next-day fill)")
    print("=" * 96)
    bh = ret.mean(axis=1)
    bhm = metrics(bh)
    print(f"\n  BUY & HOLD (pooled equal-weight): total {bhm['total']:>+9.1%}  "
          f"CAGR {bhm['cagr']:>+7.2%}  Sharpe {bhm['sharpe']:>5.2f}  maxDD {bhm['maxdd']:>+7.1%}")

    trials = []
    print(f"\n  --- TREND rule: long while RSI(OBV,n) > 50 ---")
    print(f"    {'n':>4}{'total':>11}{'CAGR':>9}{'Sharpe':>8}{'maxDD':>9}{'vsB&H':>8}{'exposure':>10}")
    for n in LENGTHS:
        r = robv(n)
        pos = (r > 50).astype(float).where(r.notna())
        d = apply_positions(pos, ret)
        m = metrics(d)
        trials.append((f"trend n={n}", m["sharpe"], d))
        print(f"    {n:>4}{m['total']:>+11.1%}{m['cagr']:>+9.2%}{m['sharpe']:>8.2f}"
              f"{m['maxdd']:>+9.1%}{m['sharpe'] - bhm['sharpe']:>+8.2f}{pos.mean().mean():>10.1%}")

    print(f"\n  --- MEAN-REVERSION rule: buy below 30, sell above 50 ---")
    print(f"    {'n':>4}{'total':>11}{'CAGR':>9}{'Sharpe':>8}{'maxDD':>9}{'vsB&H':>8}{'exposure':>10}")
    for n in LENGTHS:
        r = robv(n)
        pos = mr_positions(r, 30, 50).where(r.notna())
        d = apply_positions(pos, ret)
        m = metrics(d)
        trials.append((f"MR n={n}", m["sharpe"], d))
        print(f"    {n:>4}{m['total']:>+11.1%}{m['cagr']:>+9.2%}{m['sharpe']:>8.2f}"
              f"{m['maxdd']:>+9.1%}{m['sharpe'] - bhm['sharpe']:>+8.2f}{pos.mean().mean():>10.1%}")
    return bh, bhm, trials


def part_haircut(bh, bhm, trials):
    print("\n" + "=" * 96)
    print("PART 5 — MULTIPLE-TESTING HAIRCUT")
    print("=" * 96)
    sharpes = np.array([s for _, s, _ in trials if np.isfinite(s)])
    n_trials = len(sharpes)
    best_label, best_sr, best_daily = max(trials, key=lambda t: t[1] if np.isfinite(t[1]) else -9)
    d = best_daily.dropna()
    dsr = deflated_sharpe(best_sr, len(d), max(n_trials, 2), float(np.var(sharpes, ddof=1)),
                          float(d.skew()), float(d.kurtosis() + 3.0))
    print(f"\n  configurations tried in Part 4: {n_trials}   spread of their Sharpes: "
          f"{sharpes.min():.2f} .. {sharpes.max():.2f}")
    print(f"  best config: {best_label}   Sharpe {best_sr:.2f}   (buy-hold {bhm['sharpe']:.2f})")
    print(f"  Deflated Sharpe (prob. the best is genuinely > 0 after the search): {dsr:.3f}")
    print("    < 0.95 means the winner is not distinguishable from the best of a random search.")

    print(f"\n  walk-forward on the best config, by calendar year:")
    print(f"    {'year':>6}{'strat Sh':>10}{'B&H Sh':>9}{'strat':>9}{'B&H':>9}{'win?':>6}")
    wins = tot = 0
    for y in sorted(set(d.index.year)):
        sd = d[d.index.year == y]
        bd = bh[bh.index.year == y].dropna()
        if len(sd) < 60 or len(bd) < 60:
            continue
        tot += 1
        ss = sd.mean() / sd.std(ddof=1) * math.sqrt(TD) if sd.std(ddof=1) else np.nan
        bs = bd.mean() / bd.std(ddof=1) * math.sqrt(TD) if bd.std(ddof=1) else np.nan
        win = ss > bs
        wins += int(win)
        print(f"    {y:>6}{ss:>+10.2f}{bs:>+9.2f}{(1 + sd).prod() - 1:>+9.1%}"
              f"{(1 + bd).prod() - 1:>+9.1%}{('Y' if win else 'n'):>6}")
    print(f"    -> beat buy-hold Sharpe in {wins}/{tot} years")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", choices=["all", "broad", "basket", "index"], default="all")
    ap.add_argument("--start", default="1995-01-01",
                    help="Yahoo volume before the mid-90s is unreliable; OBV needs volume")
    ap.add_argument("--part", choices=["all", "char", "ic", "rules"], default="all")
    ap.add_argument("--burn", type=int, default=250,
                    help="discard each name's first N bars (indicator warm-up)")
    args = ap.parse_args()

    data, group = load()
    if args.group != "all":
        data = {s: d for s, d in data.items() if group.get(s) == args.group}
    print(f"panel: {len(data)} names, group={args.group}")

    close = pd.DataFrame({s: d["Close"] for s, d in data.items()}).sort_index()
    volume = pd.DataFrame({s: d["Volume"] for s, d in data.items()}).sort_index()
    close = close[close.index >= args.start]
    volume = volume.reindex(close.index)
    # A name needs enough history for the longest RSI plus the longest horizon.
    ok = close.notna().sum() >= 400
    close, volume = close.loc[:, ok], volume.loc[:, ok]
    print(f"       {close.shape[1]} names with >=400 bars, {close.index.min().date()} .. "
          f"{close.index.max().date()} ({close.shape[0]} bars)")

    obv, upv, dnv = obv_parts(close, volume)
    mask = burn_mask(close, args.burn)
    print(f"       burn-in {args.burn} bars/name -> {mask.sum().sum():,} usable name-days")
    ret = close.pct_change()
    fwd = {h: (close.shift(-h) / close - 1).where(mask) for h in HORIZONS}
    _robv: dict[int, pd.DataFrame] = {}
    def robv(n):
        if n not in _robv:
            _robv[n] = rsi_obv(upv, dnv, n).where(mask)
        return _robv[n]

    best = {}
    if args.part in ("all", "char"):
        prsi_cache = part_identity(close, volume, obv, upv, dnv, mask)
    elif args.part in ("all", "ic"):
        prsi_cache = {n: rsi_on(close, n).where(mask) for n in LENGTHS}
    if args.part in ("all", "ic"):
        best = part_ic(robv, prsi_cache, fwd)
        part_panel(robv, prsi_cache, fwd)
        n_best = best.get(("LEVEL   RSI(OBV,n)", 5), (14,))[0]
        part_shape(robv, fwd, n_best)
    if args.part in ("all", "rules"):
        bh, bhm, trials = part_rules(robv, ret)
        part_haircut(bh, bhm, trials)

    print("\n" + "=" * 96)
    print("Caveats: universe is today's listed names (membership survivorship — returns are")
    print("real but dead companies are absent, which flatters long-only absolutes); close-to-")
    print("close; long-only cash; equal-weight pooling; forward windows overlap (t-stats use")
    print("n/h effective observations). Compare strategy-vs-buy-hold, not absolute levels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
