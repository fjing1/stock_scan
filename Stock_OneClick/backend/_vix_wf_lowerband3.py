"""
_vix_wf_lowerband3.py — the last two gaps.

D. Persistence properly: BB(10,2.0) episodes are 1.2 days long, so "day 2..5" is unanswerable
   there (n=22). Re-run the ENTRY-vs-STATE question on BB(10,1.5) and BB(10,1.0), which have
   long enough streaks, and add a "days since entry" decay curve.
E. Walk-forward de-risk rule, honest scoring: deflated Sharpe for the 45 configs searched,
   vol-matched CAGR (the rule sits in cash 33% of the time, so raw CAGR is not the fair
   comparison), and a decade-by-decade record.
F. Recent-window win-rate/median gap (robust stats, not just the mean) for 2010-2026.

Run: ../../vcp_env/bin/python _vix_wf_lowerband3.py
"""
from __future__ import annotations

import math
import warnings
from collections import Counter

import numpy as np

from _vix_wf_lowerband import panel, rot_p, stars, streak_day, episodes, cell

# NOTE: cannot `from _vix_ma10_bb_research import deflated_sharpe` — that file has a
# pre-existing SyntaxError at line 397 (`last = d.iloc[-1]    print(...)`) and I was told not
# to edit it. Verbatim copy of its deflated_sharpe / _phi / _phi_inv below.


def _phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _phi_inv(p):
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
    dd = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return ((((( c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -((((( c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def deflated_sharpe(sel_pp_sharpe, all_pp_sharpes, n_obs):
    trials = max(len(all_pp_sharpes), 2)
    sd = float(np.std(all_pp_sharpes, ddof=1)) or 1e-9
    emc = 0.5772156649
    e_max = sd * ((1 - emc) * _phi_inv(1 - 1.0 / trials) + emc * _phi_inv(1 - 1.0 / (trials * math.e)))
    return _phi((sel_pp_sharpe - e_max) * math.sqrt(max(n_obs - 1, 1)))

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(777)
HZ = [1, 3, 5, 10]


def main():
    d = panel()
    N = len(d)
    yr = np.asarray(d.index.year)
    ret = (d.spx.shift(-1) / d.spx - 1.0).fillna(0.0).values

    print("=" * 116)
    print("D. ENTRY vs STATE, on signals with long enough streaks to actually answer it")
    print("=" * 116)
    for n, k in [(10, 1.5), (10, 1.0), (20, 1.5)]:
        mask = d[f"L{n}_{k}"].values
        sd_ = streak_day(mask)
        ep = episodes(mask)
        print(f"\n  BB({n},{k})  {int(mask.sum())} days / {ep} episodes  "
              f"mean length {mask.sum()/ep:.1f}d  max streak {sd_.max()}d")
        print(f"  {'group':<30}{'n':>6}   " + "".join(f"{'D'+str(h)+' exc  p':>21}" for h in HZ))
        groups = [("day 1 only (ENTRY)", sd_ == 1), ("day 2", sd_ == 2), ("day 3", sd_ == 3),
                  ("day 4", sd_ == 4), ("day 5", sd_ == 5), ("day 6+", sd_ >= 6),
                  ("day >=2 (continuation)", sd_ >= 2), ("ALL days below (STATE)", mask)]
        for lbl, g in groups:
            c = int(g.sum())
            if c < 25:
                print(f"  {lbl:<30}{c:>6}   n<25 — inconclusive")
                continue
            cells = []
            for h in HZ:
                cc = cell(d, g, h)
                cells.append((f"{cc['exc']*100:+7.2f}% p={cc['p']:.3f}{stars(cc['p'])}"
                              if cc["n"] >= 25 else "n<25").rjust(21))
            print(f"  {lbl:<30}{c:>6}   " + "".join(cells))

    print("\n" + "=" * 116)
    print("E. WALK-FORWARD DE-RISK, honest scoring")
    print("=" * 116)

    def derisk(mask, K):
        out = np.zeros(N, dtype=bool)
        for i in np.flatnonzero(mask):
            out[i + 1: i + 1 + K] = True
        return out

    def stats(r):
        c = (1 + r).prod() ** (252 / len(r)) - 1
        v = r.std(ddof=0) * math.sqrt(252)
        eq = (1 + r).cumprod()
        return c, v, (c / v if v else np.nan), (eq / np.maximum.accumulate(eq) - 1).min()

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
            m = d[f"L{n}_{k}"].values & tr
            if m.sum() < 25:
                continue
            r = np.where(derisk(m, K), 0.0, ret)[tr]
            s = r.mean() / (r.std(ddof=0) or 1e-9)
            if s > bsh:
                bsh, best = s, (n, k, K)
        if best is None:
            continue
        n, k, K = best
        te = yr == y
        o = derisk(d[f"L{n}_{k}"].values, K)
        oos[te] = np.where(o, 0.0, ret)[te]
        oos_out[te] = o[te]
        picks.append((y, best))
    live = yr >= start
    o_, b_ = oos[live], ret[live]
    c1, v1, s1, dd1 = stats(o_)
    c0, v0, s0, dd0 = stats(b_)
    print(f"  OOS {d.index[live][0].date()} -> {d.index[live][-1].date()}  n={int(live.sum()):,}"
          f"   out of market {oos_out[live].mean()*100:.1f}% of days")
    print(f"  {'':<26}{'CAGR':>9}{'vol':>8}{'Sharpe':>8}{'maxDD':>9}")
    print(f"  {'walk-forward de-risk':<26}{c1*100:>8.2f}%{v1*100:>7.1f}%{s1:>8.2f}{dd1*100:>8.1f}%")
    print(f"  {'buy & hold':<26}{c0*100:>8.2f}%{v0*100:>7.1f}%{s0:>8.2f}{dd0*100:>8.1f}%")
    lev = v0 / v1
    cl, vl, sl, ddl = stats(o_ * lev)
    print(f"  {'rule levered to BH vol':<26}{cl*100:>8.2f}%{vl*100:>7.1f}%{sl:>8.2f}{ddl*100:>8.1f}%"
          f"   (leverage {lev:.2f}x, financing cost ignored -> optimistic)")
    all_sh = []
    for n, k, K in grid:
        m = d[f"L{n}_{k}"].values
        if m.sum() < 25:
            continue
        r = np.where(derisk(m, K), 0.0, ret)
        all_sh.append(r.mean() / (r.std(ddof=0) or 1e-9))
    dsr = deflated_sharpe(o_.mean() / (o_.std(ddof=0) or 1e-9), all_sh, len(o_))
    print(f"  deflated Sharpe P(true Sharpe > 0) after {len(all_sh)} configs searched: {dsr:.3f}")
    print(f"  configs chosen: {Counter(p for _, p in picks).most_common(4)}")
    print(f"\n  {'decade':<12}{'rule CAGR':>11}{'BH CAGR':>10}{'rule Sh':>9}{'BH Sh':>8}{'%out':>7}")
    for nm, y0, y1 in [("1998-1999", 1998, 1999), ("2000s", 2000, 2009),
                       ("2010s", 2010, 2019), ("2020s", 2020, 2099)]:
        sl_ = live & (yr >= y0) & (yr <= y1)
        if sl_.sum() < 100:
            continue
        ca, va, sa, _ = stats(oos[sl_])
        cb, vb, sb, _ = stats(ret[sl_])
        print(f"  {nm:<12}{ca*100:>10.2f}%{cb*100:>9.2f}%{sa:>9.2f}{sb:>8.2f}"
              f"{oos_out[sl_].mean()*100:>6.1f}%")

    print("\n" + "=" * 116)
    print("F. RECENT WINDOW robust stats (2010-2026): is the win-rate / median gap still there?")
    print("=" * 116)
    w = yr >= 2010
    fwd = d["g5"].values
    for n, k in [(10, 2.0), (20, 2.0), (10, 1.5)]:
        m = d[f"L{n}_{k}"].values
        for wl, ww in [("FULL", np.ones(N, bool)), ("2010-2026", w)]:
            f = fwd[ww]
            mm = m[ww]
            v = ~np.isnan(f)
            sel = mm & v
            if sel.sum() < 25:
                print(f"  BB({n},{k}) {wl:<10} n={int(sel.sum())} — inconclusive")
                continue
            a, b = f[sel], f[v]
            # rotation p on the win rate inside the window
            nn = len(f)
            null = []
            for off in RNG.integers(1, nn, size=3000):
                q = np.roll(mm, off) & v
                if q.sum() >= 25:
                    null.append((f[q] > 0).mean())
            null = np.array(null)
            b0 = null.mean()
            pwin = float((np.abs(null - b0) >= abs((a > 0).mean() - b0)).mean())
            print(f"  BB({n},{k}) {wl:<10} n={len(a):>4}  win {(a>0).mean()*100:>5.1f}% vs base "
                  f"{(b>0).mean()*100:>5.1f}% (gap {((a>0).mean()-(b>0).mean())*100:+5.1f}pp, p={pwin:.3f})"
                  f"   median {np.median(a)*100:+6.2f}% vs {np.median(b)*100:+5.2f}%"
                  f"   mean {a.mean()*100:+6.2f}% vs {b.mean()*100:+5.2f}%")


if __name__ == "__main__":
    main()
