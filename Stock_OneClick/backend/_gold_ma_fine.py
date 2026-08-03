"""
_gold_ma_fine.py — fine sensitivity of the gold trend filter around the SMA(500) baseline.

The point of a fine sweep is NOT to find the best of 480/490/500/510/520. Picking the in-sample
winner from five neighbouring lengths is the overfitting trap in miniature: with ~26 independent
years, the standard error on any one length's Sharpe is ~0.2, so a 0.02 gap between neighbours is
noise by construction.

What a fine sweep CAN tell you:
  1. Is the response smooth and flat (robust) or jagged (the parameter is fitting wiggles)?
  2. Do neighbouring lengths ever DISAGREE about the current regime? If 480 and 520 give the same
     answer 99% of days, the choice is operationally irrelevant and the debate is moot.
  3. Does the ranking hold across subperiods, or does each regime prefer a different length?
  4. Is the difference statistically distinguishable at all? (Spoiler: no.)

Baseline is SMA(500) with the +/-2% hysteresis band, i.e. gold_system.GOLD_TREND_*.

Run: ../../vcp_env/bin/python _gold_ma_fine.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import gold_system as G

TD = 252
RNG = np.random.default_rng(31337)
FOCUS = [480, 490, 500, 510, 520]
BASE = 500


def state_series(g: pd.Series, n: int, band: float) -> pd.Series:
    """Banded regime state (hysteresis): 1 = uptrend, 0 = downtrend, held inside the band."""
    ma = g.rolling(n).mean()
    st = pd.Series(np.nan, index=g.index)
    st[g > ma * (1 + band)] = 1.0
    st[g < ma * (1 - band)] = 0.0
    return st.ffill().fillna(1.0)


def metrics(g, r, idx, rf_d, n, band):
    st = state_series(g, n, band).shift(1).reindex(idx).fillna(1.0).astype(bool)
    curve = (1 + r.where(st, rf_d)).cumprod()
    ret = curve.pct_change().dropna()
    yrs = len(ret) / TD
    cagr = curve.iloc[-1] ** (1 / yrs) - 1
    vol = ret.std(ddof=1) * np.sqrt(TD)
    mdd = ((curve / curve.cummax()) - 1).min()
    flips = int((st.astype(int).diff().abs() == 1).sum())
    return {"n": n, "cagr": cagr * 100, "vol": vol * 100, "maxdd": mdd * 100,
            "sharpe": (cagr - G.RISK_FREE) / vol, "inmkt": st.mean() * 100,
            "flips": flips, "state": st, "curve": curve}


def main():
    g = G.load_panel()["GC=F"]["Close"].dropna()
    r = g.pct_change()
    idx = r.dropna().index
    r = r.loc[idx]
    rf_d = (1 + G.RISK_FREE) ** (1 / TD) - 1
    band = G.GOLD_TREND_BAND
    yrs = len(idx) / TD
    print(f"gold daily {idx[0].date()} -> {idx[-1].date()}  ({yrs:.0f} independent years)")
    print(f"baseline: SMA({BASE}) with a +/-{band*100:.0f}% hysteresis band\n")

    bhc = (1 + r).cumprod()
    bret = bhc.pct_change().dropna()
    bcagr = bhc.iloc[-1] ** (1 / (len(bret) / TD)) - 1
    bvol = bret.std(ddof=1) * np.sqrt(TD)
    print(f"buy-and-hold: CAGR {bcagr*100:.2f}%  MaxDD {((bhc/bhc.cummax())-1).min()*100:.1f}%  "
          f"Sharpe {(bcagr-G.RISK_FREE)/bvol:.3f}\n")

    # ---- 1. the five lengths asked for ------------------------------------ #
    print("=== 1. THE FIVE LENGTHS, banded (deltas vs the SMA500 baseline) ===")
    res = {n: metrics(g, r, idx, rf_d, n, band) for n in FOCUS}
    b = res[BASE]
    print(f"  {'n':>5}{'CAGR':>8}{'vol':>7}{'MaxDD':>9}{'Sharpe':>8}{'dSharpe':>9}"
          f"{'inMkt':>7}{'flips':>7}")
    for n in FOCUS:
        m = res[n]
        d = m["sharpe"] - b["sharpe"]
        star = "  <- baseline" if n == BASE else ""
        print(f"  {n:>5}{m['cagr']:>7.2f}%{m['vol']:>6.1f}%{m['maxdd']:>8.1f}%"
              f"{m['sharpe']:>8.3f}{d:>+9.3f}{m['inmkt']:>6.0f}%{m['flips']:>7}{star}")
    sh = np.array([res[n]["sharpe"] for n in FOCUS])
    print(f"\n  spread across these five: {sh.max()-sh.min():.3f} Sharpe "
          f"(min {sh.min():.3f} at n={FOCUS[int(sh.argmin())]}, "
          f"max {sh.max():.3f} at n={FOCUS[int(sh.argmax())]})")

    # ---- 2. is the response smooth? --------------------------------------- #
    print("\n=== 2. SHAPE OF THE RESPONSE (5-day steps, 440-580) — flat or jagged? ===")
    fine = [metrics(g, r, idx, rf_d, n, band) for n in range(440, 581, 5)]
    fd = pd.DataFrame([{k: v for k, v in m.items() if k not in ("state", "curve")}
                       for m in fine])
    for _, row in fd.iterrows():
        bar = "#" * int(round((row.sharpe - 0.40) / 0.005))
        mark = "  *" if int(row.n) in FOCUS else ""
        print(f"  {int(row.n):>4}d  {row.sharpe:.3f}  {bar}{mark}")
    diffs = fd.sharpe.diff().abs().dropna()
    print(f"\n  mean |change| between adjacent 5-day steps: {diffs.mean():.4f}")
    print(f"  max  |change| between adjacent 5-day steps: {diffs.max():.4f} "
          f"(at n={int(fd.n.iloc[int(diffs.idxmax())])})")
    print(f"  full range 440-580: {fd.sharpe.min():.3f} to {fd.sharpe.max():.3f}")

    # ---- 3. do they ever DISAGREE in practice? ---------------------------- #
    print("\n=== 3. OPERATIONAL AGREEMENT — do these lengths ever say different things? ===")
    states = pd.DataFrame({n: state_series(g, n, band) for n in FOCUS}).dropna()
    agree = (states.nunique(axis=1) == 1)
    print(f"  days where all five agree: {int(agree.sum())} of {len(states)} "
          f"({agree.mean()*100:.1f}%)")
    dis = states[~agree]
    if len(dis):
        runs, prev, start = [], None, None
        for d in dis.index:
            if prev is None or (d - prev).days > 5:
                if start is not None:
                    runs.append((start, prev))
                start = d
            prev = d
        runs.append((start, prev))
        print(f"  disagreement episodes: {len(runs)}")
        for a, bb in runs[:8]:
            print(f"    {a.date()} -> {bb.date()}  ({(bb-a).days} days)")
    print(f"  480 vs 520 pairwise agreement: "
          f"{(states[480]==states[520]).mean()*100:.1f}% of days")

    # ---- 4. subperiod stability ------------------------------------------- #
    print("\n=== 4. WHICH LENGTH WINS IN EACH SUBPERIOD? (agreement = robust) ===")
    spans = {"2000-2011 bull": ("2000-01-01", "2011-08-31"),
             "2011-2015 bear": ("2011-09-01", "2015-12-31"),
             "2016-2026 bull": ("2016-01-01", "2026-12-31")}
    for lbl, (a, bb) in spans.items():
        sub = idx[(idx >= a) & (idx <= bb)]
        if len(sub) < TD:
            continue
        out = {}
        for n in range(440, 581, 20):
            st = state_series(g, n, band).shift(1).reindex(sub).fillna(1.0).astype(bool)
            x = r.loc[sub].where(st, rf_d)
            sd = x.std(ddof=1)
            if sd > 0:
                out[n] = (x.mean() - rf_d) / sd * np.sqrt(TD)
        if out:
            bn = max(out, key=out.get)
            print(f"  {lbl:<16} best n={bn:>3}d (Sharpe {out[bn]:+.2f})   "
                  f"n=500 -> {out.get(500, float('nan')):+.2f}   "
                  f"range across lengths {min(out.values()):+.2f} to {max(out.values()):+.2f}")

    # ---- 5. is any difference significant? -------------------------------- #
    print("\n=== 5. BOOTSTRAP: is any of 480/490/510/520 distinguishable from 500? ===")
    years = sorted({d.year for d in idx})
    s500 = res[BASE]["state"]
    for n in [x for x in FOCUS if x != BASE]:
        sn = res[n]["state"]
        d = []
        for _ in range(1500):
            pick = RNG.choice(years, size=len(years), replace=True)
            sel = np.concatenate([np.where(idx.year == y)[0] for y in pick])
            x = r.iloc[sel].where(s500.iloc[sel], rf_d)
            y = r.iloc[sel].where(sn.iloc[sel], rf_d)
            xs, ys = x.std(ddof=1), y.std(ddof=1)
            if xs > 0 and ys > 0:
                d.append((((y.mean() - rf_d) / ys) - ((x.mean() - rf_d) / xs)) * np.sqrt(TD))
        d = np.array(d)
        print(f"  n={n} minus n=500:  mean {d.mean():+.3f}  SE {d.std(ddof=1):.3f}  "
              f"95% CI [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}]  "
              f"P(better) {(d>0).mean()*100:3.0f}%")

    # ---- 6. current reading ----------------------------------------------- #
    print("\n=== 6. CURRENT READING FOR EACH LENGTH ===")
    px = float(g.iloc[-1])
    print(f"  gold {px:,.2f} as of {g.index[-1].date()}")
    print(f"  {'n':>5}{'SMA':>11}{'lower band':>12}{'ext%':>8}{'state':>11}{'flip price':>12}")
    for n in FOCUS:
        ma = float(g.rolling(n).mean().iloc[-1])
        lo = ma * (1 - band)
        st = "UPTREND" if px > ma * (1 + band) else ("DOWNTREND" if px < lo else "IN BAND")
        print(f"  {n:>5}{ma:>11,.0f}{lo:>12,.0f}{(px/ma-1)*100:>7.1f}%{st:>11}"
              f"{lo:>11,.0f} ({(lo/px-1)*100:+.1f}%)")


if __name__ == "__main__":
    main()
