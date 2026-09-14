"""
_vix_wf_verify_snooping_covidflip_c.py — part 3.

  T13 the drop-one-quarter jackknife the claim cites: is 2020Q1's influence unusual, or is that
      just what a 25-observation quarter does inside an n=~180 window? Full jackknife distribution.
  T14 a NON-date-selective robustness stat: win-rate excess by era, COVID INCLUDED, all horizons.
      If the 2020s win-rate excess is positive with COVID in the sample, the "tail not decay"
      reading has support that does not require deleting anything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _vix_data

MIN_N = 25
d = _vix_data.add_features(_vix_data.load())
idx = d.index
N = len(d)
g5 = d.g5.values
v5 = ~np.isnan(g5)
SIG = {"stretch>=+10%": (d.stretch >= 0.10).values,
       "stretch>=+20%": (d.stretch >= 0.20).values,
       "BB(10,2.0) above": d["bb10_2.0_above"].values,
       "BB(10,2.0) re-entry": d["bb10_2.0_reentry"].values}
is20s = np.asarray(idx.year >= 2020)
ERAS = [("1990s", np.asarray(idx.year <= 1999)),
        ("2000s", np.asarray((idx.year >= 2000) & (idx.year <= 2009))),
        ("2010s", np.asarray((idx.year >= 2010) & (idx.year <= 2019))),
        ("2020s", is20s)]


def exc(mask, keep, fwd=None):
    fwd = g5 if fwd is None else fwd
    vv = ~np.isnan(fwd)
    k = keep & vv
    s = mask & k
    if s.sum() < MIN_N:
        return np.nan, int(s.sum())
    return fwd[s].mean() - fwd[k].mean(), int(s.sum())


print("=" * 108)
print("T13 — drop-one-QUARTER jackknife over 2015-2026: is 2020Q1 an outlier influence?")
print("=" * 108)
win = np.asarray(idx.year >= 2015)
q = idx.to_period("Q")
qs = sorted(set(q[win]))
for s, m in SIG.items():
    base, nb = exc(m, win)
    recs = []
    for qq in qs:
        dropq = np.asarray(q != qq)
        e, n = exc(m, win & dropq)
        nsig = int((m & win & np.asarray(q == qq) & v5).sum())
        if not np.isnan(e):
            recs.append((e - base, str(qq), nsig))
    recs.sort(reverse=True)
    infl = np.array([r[0] for r in recs])
    q1 = [r for r in recs if r[1] == "2020Q1"]
    rk = recs.index(q1[0]) + 1 if q1 else -1
    print(f"\n  {s}: 2015-2026 excess {base*100:+.3f}pp (n={nb}); {len(recs)} quarters jackknifed")
    print(f"    2020Q1 influence {q1[0][0]*100:+.3f}pp ({q1[0][2]} signal days) — rank {rk}/{len(recs)}")
    print(f"    next 4 most influential: " +
          ", ".join(f"{r[1]} {r[0]*100:+.3f}pp(n{r[2]})" for r in recs[1:5]))
    print(f"    |influence| mean {np.abs(infl).mean()*100:.3f}pp, 90th pct "
          f"{np.percentile(np.abs(infl),90)*100:.3f}pp, max-negative {infl.min()*100:+.3f}pp "
          f"({recs[-1][1]})")
    # how many quarters must be dropped to flip 2020s alone positive?
    k20 = is20s
    b20, _ = exc(m, k20)
    dropped, cur = [], b20
    remaining = np.ones(N, bool)
    while cur < 0 and len(dropped) < 6:
        best, bq = -9, None
        for qq in sorted(set(q[is20s])):
            if str(qq) in dropped:
                continue
            trial = remaining & np.asarray(q != qq)
            e, n = exc(m, k20 & trial)
            if not np.isnan(e) and e > best:
                best, bq = e, str(qq)
        if bq is None:
            break
        dropped.append(bq)
        remaining = remaining & np.asarray(q.astype(str) != bq)
        cur = best
    print(f"    quarters that must be deleted to make the 2020s block positive: "
          f"{dropped} -> {cur*100:+.3f}pp")

print("\n" + "=" * 108)
print("T14 — WIN-RATE excess (no date deletion, COVID INCLUDED), by era and horizon")
print("=" * 108)
print("  win-rate on signal days MINUS win-rate on all same-sample days, in pp.\n")
for s, m in SIG.items():
    print(f"  {s}")
    print(f"    {'era':<8}" + "".join(f"{'D'+str(h):>16}" for h in (1, 3, 5, 10, 21)))
    for en, ek in ERAS:
        cells = []
        for h in (1, 3, 5, 10, 21):
            fwd = d[f"g{h}"].values
            vv = ~np.isnan(fwd)
            k = ek & vv
            sel = m & k
            if sel.sum() < MIN_N:
                cells.append(f"{'n<25':>16}")
                continue
            w = (fwd[sel] > 0).mean() - (fwd[k] > 0).mean()
            cells.append(f"{w*100:+8.2f}(n{int(sel.sum())})".rjust(16))
        print(f"    {en:<8}" + "".join(cells))
    print()

print("=" * 108)
print("T15 — 2020s block (COVID INCLUDED) mean excess across ALL horizons: is D5 special?")
print("=" * 108)
for s, m in SIG.items():
    cells = []
    for h in (1, 3, 5, 10, 21):
        fwd = d[f"g{h}"].values
        e, n = exc(m, is20s, fwd)
        cells.append(f"{e*100:+8.3f}(n{n})".rjust(16))
    print(f"  {s:<22}" + "".join(cells))
print("\ndone.")
