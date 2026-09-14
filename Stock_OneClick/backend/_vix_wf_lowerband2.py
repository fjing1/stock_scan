"""
_vix_wf_lowerband2.py — follow-ups to _vix_wf_lowerband.py that decide the verdict.

A. Recent-window test: is the lower-band edge still alive? Excess + rotation p computed
   INSIDE 2010-2026 and 2015-2026 (baseline recomputed inside the window too).
B. Honest de-risk rule: walk-forward selection of (n, k, K) on prior data only, plus a
   rotation test on the FULL-SAMPLE in-sample rule so we know how much of its Sharpe gain
   is just "some mask that avoided 8% of days".
C. Control: does the BAND add anything over the plain level (VIX < MA10, VIX 1y-percentile
   low, VIX z-score low)? If the band is just a proxy for "VIX is low", say so.

Run: ../../vcp_env/bin/python _vix_wf_lowerband2.py
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

import _vix_data
from _vix_wf_lowerband import panel, rot_p, rot_all, stars, episodes

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(4242)
HZ = [1, 3, 5]


def sub_excess(d, mask, h, win):
    """Excess + rotation p computed ENTIRELY inside the window `win` (baseline too)."""
    fwd = d[f"g{h}"].values[win]
    m = mask[win]
    v = ~np.isnan(fwd)
    sel = m & v
    if sel.sum() < 25:
        return int(sel.sum()), np.nan, np.nan, np.nan
    cond, base = fwd[sel].mean(), fwd[v].mean()
    return int(sel.sum()), cond - base, rot_p(m, fwd, cond), base


def main():
    d = panel()
    N = len(d)
    yr = np.asarray(d.index.year)
    windows = [("FULL 1990-2026", np.ones(N, bool)),
               ("2010-2026", yr >= 2010),
               ("2015-2026", yr >= 2015),
               ("2020-2026", yr >= 2020)]

    print("=" * 112)
    print("A. IS IT STILL ALIVE?  excess + rotation p recomputed INSIDE each window")
    print("=" * 112)
    for n, k in [(10, 2.0), (20, 2.0), (10, 1.5), (15, 1.5), (20, 1.5), (10, 1.0)]:
        mask = d[f"L{n}_{k}"].values
        print(f"\n  BB({n},{k})")
        print(f"  {'window':<16}{'n':>6}{'win-base g5':>13}   " +
              "".join(f"{'D'+str(h)+' exc   p':>22}" for h in HZ))
        for wl, w in windows:
            cells, n5, base5 = [], 0, np.nan
            for h in HZ:
                cnt, exc, p, base = sub_excess(d, mask, h, w)
                if h == 5:
                    n5, base5 = cnt, base
                cells.append(("n<25 incon." if cnt < 25 else
                              f"{exc*100:+7.2f}%  p={p:.3f} {stars(p)}").rjust(22))
            print(f"  {wl:<16}{n5:>6}{(base5*100 if base5==base5 else float('nan')):>12.2f}%   " +
                  "".join(cells))

    print("\n" + "=" * 112)
    print("B1. DE-RISK RULE — rotation test on the rule itself (is the Sharpe gain more than a")
    print("    random 8%-of-days mask would give?)  1000 circular rotations of the signal.")
    print("=" * 112)
    ret = (d.spx.shift(-1) / d.spx - 1.0).fillna(0.0).values

    def derisk(mask, K):
        out = np.zeros(N, dtype=bool)
        for i in np.flatnonzero(mask):
            out[i + 1: i + 1 + K] = True
        return out

    def stats(r):
        c = (1 + r).prod() ** (252 / len(r)) - 1
        v = r.std(ddof=0) * math.sqrt(252)
        eq = (1 + r).cumprod()
        return c, (c / v if v else np.nan), (eq / np.maximum.accumulate(eq) - 1).min()

    for wl, w in [("FULL 1990-2026", np.ones(N, bool)), ("2010-2026", yr >= 2010)]:
        rr = ret[w]
        bc, bs, bd = stats(rr)
        print(f"\n  {wl}: buy-hold CAGR {bc*100:.2f}%  Sharpe {bs:.2f}  maxDD {bd*100:.1f}%")
        for n, k, K in [(10, 2.0, 3), (10, 2.0, 5), (20, 2.0, 3), (10, 1.5, 5)]:
            mask = d[f"L{n}_{k}"].values
            out = derisk(mask, K)
            c, s, dd = stats(np.where(out, 0.0, ret)[w])
            null_s, null_c = [], []
            offs = RNG.integers(1, N, size=1000)
            for off in offs:
                o2 = derisk(np.roll(mask, off), K)
                c2, s2, _ = stats(np.where(o2, 0.0, ret)[w])
                null_s.append(s2)
                null_c.append(c2)
            null_s, null_c = np.array(null_s), np.array(null_c)
            ps = float((null_s >= s).mean())
            pc = float((null_c >= c).mean())
            print(f"    flat {K}d after BB({n},{k}): CAGR {c*100:>6.2f}% (rot-null median "
                  f"{np.median(null_c)*100:>6.2f}%, p={pc:.3f})   Sharpe {s:.2f} (null median "
                  f"{np.median(null_s):.2f}, p={ps:.3f})   out {out[w].mean()*100:.1f}% of days")

    print("\n" + "=" * 112)
    print("B2. WALK-FORWARD de-risk rule: (n,k,K) picked each year on PRIOR data only (max Sharpe),")
    print("    applied in that year. No config is ever chosen with knowledge of its own year.")
    print("=" * 112)
    grid = [(n, k, K) for n in (5, 10, 15, 20, 30) for k in (1.0, 1.5, 2.0) for K in (3, 5, 10)]
    years = sorted(set(yr))
    start = years[0] + 8
    oos = np.zeros(N)
    oos_out = np.zeros(N, bool)
    picks = []
    for y in years:
        if y < start:
            continue
        tr = yr < y
        best, bsh = None, -np.inf
        for n, k, K in grid:
            mask = d[f"L{n}_{k}"].values & tr
            if mask.sum() < 25:
                continue
            o = derisk(mask, K)
            r = np.where(o, 0.0, ret)[tr]
            s = r.mean() / (r.std(ddof=0) or 1e-9)
            if s > bsh:
                bsh, best = s, (n, k, K)
        if best is None:
            continue
        n, k, K = best
        te = yr == y
        # signals only from data available at the time (mask itself is causal)
        o = derisk(d[f"L{n}_{k}"].values, K)
        oos[te] = np.where(o, 0.0, ret)[te]
        oos_out[te] = o[te]
        picks.append((y, best))
    live = yr >= start
    o_, b_ = oos[live], ret[live]
    c1, s1, d1 = stats(o_)
    c0, s0, d0 = stats(b_)
    print(f"  OOS {d.index[live][0].date()} -> {d.index[live][-1].date()}  n={int(live.sum()):,}")
    print(f"  {'walk-forward de-risk':<24}CAGR {c1*100:>6.2f}%  Sharpe {s1:.2f}  maxDD {d1*100:>6.1f}%  "
          f"out {oos_out[live].mean()*100:.1f}% of days")
    print(f"  {'buy & hold':<24}CAGR {c0*100:>6.2f}%  Sharpe {s0:.2f}  maxDD {d0*100:>6.1f}%")
    print(f"  avoided days mean {b_[oos_out[live]].mean()*100:+.4f}%/day vs all-days "
          f"{b_.mean()*100:+.4f}%/day  (n={int(oos_out[live].sum())})")
    from collections import Counter
    print(f"  configs chosen across {len(picks)} OOS years: {Counter(p for _, p in picks).most_common(5)}")
    # split the OOS record in half
    half = np.flatnonzero(live)
    mid = half[len(half) // 2]
    for lbl, sl in [("OOS first half", live & (np.arange(N) <= mid)),
                    ("OOS second half", live & (np.arange(N) > mid))]:
        ca, sa, _ = stats(oos[sl])
        cb, sb, _ = stats(ret[sl])
        print(f"  {lbl} {d.index[sl][0].date()}..{d.index[sl][-1].date()}: rule CAGR {ca*100:>6.2f}% "
              f"Sh {sa:.2f}  vs buy-hold {cb*100:>6.2f}% Sh {sb:.2f}")

    print("\n" + "=" * 112)
    print("C. DOES THE BAND ADD ANYTHING OVER THE PLAIN LEVEL?  g5 excess, full sample")
    print("=" * 112)
    fwd = d["g5"].values
    v = ~np.isnan(fwd)
    base = fwd[v].mean()
    below10 = d["L10_1.5"].values
    below10b = d["L10_2.0"].values
    cands = [
        ("VIX < MA10 (any)", (d.stretch < 0).values),
        ("VIX < MA10 by >5%", (d.stretch <= -0.05).values),
        ("VIX 1y percentile < 20%", (d.vix_pct1y < 0.20).values),
        ("VIX 1y percentile < 10%", (d.vix_pct1y < 0.10).values),
        ("VIX z1y < -1", (d.vix_z1y < -1).values),
        ("BB(10,1.5) below", below10),
        ("BB(10,2.0) below", below10b),
        ("BB(10,1.5) below & VIX pct1y<20%", below10 & (d.vix_pct1y < 0.20).values),
        ("BB(10,1.5) below & VIX pct1y>=20%", below10 & (d.vix_pct1y >= 0.20).values),
        ("VIX pct1y<20% & NOT below band", (d.vix_pct1y < 0.20).values & ~below10),
        ("VIX<MA10-5% & NOT below band", (d.stretch <= -0.05).values & ~below10),
    ]
    print(f"  {'condition':<38}{'n':>7}{'g5 mean':>10}{'excess':>9}{'p':>8}{'win%':>8}")
    for lbl, m in cands:
        m = m & ~np.isnan(d.vix_pct1y.values) if "pct1y" in lbl else m
        sel = m & v
        if sel.sum() < 25:
            print(f"  {lbl:<38}{int(sel.sum()):>7}   n<25 — inconclusive")
            continue
        cond = fwd[sel].mean()
        p = rot_p(m, fwd, cond)
        print(f"  {lbl:<38}{int(sel.sum()):>7}{cond*100:>9.2f}%{(cond-base)*100:>8.2f}%"
              f"{p:>8.3f}{(fwd[sel]>0).mean()*100:>7.1f}%")
    print(f"  {'ALL DAYS (baseline)':<38}{int(v.sum()):>7}{base*100:>9.2f}%{0:>8.2f}%{'':>8}"
          f"{(fwd[v]>0).mean()*100:>7.1f}%")

    # current reading
    last = d.iloc[-1]
    print("\n" + "=" * 112)
    print(f"CURRENT READING {d.index[-1].date()}: VIX {last.vix:.2f}  MA10 {last.ma10:.2f}  "
          f"stretch {last.stretch*100:+.1f}%  BB(10,2) lo {last['bb10_2.0_lo']:.2f}  "
          f"below={bool(d['L10_2.0'].iloc[-1])}  BB(10,1.5) lo={last['bb10_1.5_lo']:.2f} "
          f"below={bool(d['L10_1.5'].iloc[-1])}")
    print("=" * 112)


if __name__ == "__main__":
    main()
