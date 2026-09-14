"""
_vix_wf_verify_mechanism_covidflip2.py -- part 2.

Part 1 established: the claim's arithmetic is exactly right, and the COVID window really is the
single most influential 5-month block (1st pctile of 106 sliding excisions).

Part 2 asks the mechanism question properly: once you control the CONFOUND rather than the
CALENDAR, does the 2020s block still look like the other decades?

  P1  Symmetric outlier treatment: trimmed / winsorized excess by era. If "COVID is an outlier"
      is the whole story, a symmetric trim applied to EVERY era should equalise them.
  P2  Symmetric crisis excision WITHIN the 2020s: the block also contains the Apr-2025 tariff
      VIX spike, whose fat RIGHT tail is what makes the ex-COVID number positive. Excise every
      VIX>40 episode of the 2020s the same way COVID was excised.
  P3  The simplest competing variable: index MEAN REVERSION. Residualise g5 on trailing SPX
      return (and VIX level) inside each era, then re-measure the signal's excess on residuals.
  P4  Matched-control test: pair each signal day with same-era non-signal days at the same
      trailing-drawdown / VIX-level cell.
  P5  Rotation p-values for the honest, non-calendar controls.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import _vix_data

N_ROT = 5000
RNG = np.random.default_rng(777)
FWD = "g5"
ERAS = [("1990s", 1990, 1999), ("2000s", 2000, 2009), ("2010s", 2010, 2019), ("2020s", 2020, 2026)]
COVID0, COVID1 = pd.Timestamp("2020-02-01"), pd.Timestamp("2020-06-30")


def rotation_pvalue(mask, fwd, observed, n_rot=N_ROT):
    n = len(mask)
    if n < 10 or mask.sum() == 0:
        return float("nan")
    valid = ~np.isnan(fwd)
    offs = RNG.integers(1, n, size=n_rot)
    null = np.empty(n_rot)
    for i, off in enumerate(offs):
        m = np.roll(mask, off) & valid
        null[i] = fwd[m].mean() if m.sum() else np.nan
    null = null[~np.isnan(null)]
    base = np.nanmean(null)
    return float((np.abs(null - base) >= abs(observed - base)).mean())


def stars(p):
    if p is None or np.isnan(p):
        return "   "
    return "***" if p < .01 else ("** " if p < .05 else ("*  " if p < .10 else "   "))


d = _vix_data.add_features(_vix_data.load())
d["tr5"] = d.spx / d.spx.shift(5) - 1.0
d["tr10"] = d.spx / d.spx.shift(10) - 1.0
d["dd63"] = d.spx / d.spx.rolling(63).max() - 1.0

SIGNALS = {
    "stretch>=+10%": d.stretch >= 0.10,
    "stretch>=+20%": d.stretch >= 0.20,
    "BB(10,2) above": d["bb10_2.0_above"].astype(bool),
    "BB(10,2) re-entry": d["bb10_2.0_reentry"].astype(bool),
}


def exc_on(frame, sig, col=FWD, transform=None):
    f = frame[col].copy()
    ok = f.notna()
    if transform is not None:
        f = transform(f, ok)
    m = sig.reindex(frame.index).fillna(False).astype(bool) & ok
    if m.sum() == 0:
        return np.nan, 0
    return (f[m].mean() - f[ok].mean()) * 100, int(m.sum())


# ================================================================= P1 trimming
print("=" * 104)
print("### P1. SYMMETRIC OUTLIER TREATMENT applied to EVERY era (no calendar surgery at all).")
print("     If 'a handful of crash observations' is the whole story, trimming should equalise eras.")
print("     Winsorise g5 at the era's own p/1-p quantiles, then recompute excess.")
for q in (0.01, 0.025, 0.05):
    print(f"\n  winsorise at {q:.1%}/{1-q:.1%}:")
    print(f"    {'signal':<20}" + "".join(f"{e[0]:>17}" for e in ERAS))
    for name, sig in SIGNALS.items():
        cells = []
        for _, y0, y1 in ERAS:
            sub = d[(d.index.year >= y0) & (d.index.year <= y1)]

            def tf(f, ok, q=q):
                lo, hi = f[ok].quantile(q), f[ok].quantile(1 - q)
                return f.clip(lo, hi)

            e, n = exc_on(sub, sig, transform=tf)
            cells.append(f"{e:>+8.3f}p n{n:<4}" if n >= 25 else f"{'n<25':>9} n{n:<4}")
        print(f"    {name:<20}" + "".join(f"{c:>17}" for c in cells))

print("\n  10% two-sided TRIMMED mean (drop tails entirely, every era):")
print(f"    {'signal':<20}" + "".join(f"{e[0]:>17}" for e in ERAS))
for name, sig in SIGNALS.items():
    cells = []
    for _, y0, y1 in ERAS:
        sub = d[(d.index.year >= y0) & (d.index.year <= y1)]
        f, ok = sub[FWD], sub[FWD].notna()
        lo, hi = f[ok].quantile(.05), f[ok].quantile(.95)
        keep = ok & (f >= lo) & (f <= hi)
        m = sig.reindex(sub.index).fillna(False).astype(bool) & keep
        e = (f[m].mean() - f[keep].mean()) * 100 if m.sum() else np.nan
        cells.append(f"{e:>+8.3f}p n{int(m.sum()):<4}" if m.sum() >= 25 else f"{'n<25':>9} n{int(m.sum()):<4}")
    print(f"    {name:<20}" + "".join(f"{c:>17}" for c in cells))

# ================================================================= P2 both crises
print("\n" + "=" * 104)
print("### P2. SYMMETRY WITHIN THE 2020s. The block has TWO VIX>40 episodes, not one.")
pk20 = d[(d.index.year >= 2020)]
spikes = pk20[pk20.vix >= 40]
runs = []
cur = [spikes.index[0]]
for t in spikes.index[1:]:
    if (t - cur[-1]).days <= 30:
        cur.append(t)
    else:
        runs.append(cur); cur = [t]
runs.append(cur)
for r in runs:
    seg = d.loc[r[0]:r[-1]]
    print(f"     VIX>=40 episode {r[0].date()}..{r[-1].date()}  ({len(r)} days, peak {seg.vix.max():.1f})")

W2020 = (d.index >= COVID0) & (d.index <= COVID1)
W2025 = (d.index >= pd.Timestamp("2025-03-01")) & (d.index <= pd.Timestamp("2025-07-31"))
d20 = d[d.index.year >= 2020]
variants = {
    "2020s raw": d20,
    "  ex-COVID only (the claim)": d20[~((d20.index >= COVID0) & (d20.index <= COVID1))],
    "  ex-Apr2025 only": d20[~((d20.index >= pd.Timestamp("2025-03-01")) & (d20.index <= pd.Timestamp("2025-07-31")))],
    "  ex BOTH VIX>40 episodes": d20[~(((d20.index >= COVID0) & (d20.index <= COVID1)) |
                                       ((d20.index >= pd.Timestamp("2025-03-01")) & (d20.index <= pd.Timestamp("2025-07-31"))))],
}
print(f"\n    {'frame':<30}" + "".join(f"{k:>21}" for k in SIGNALS))
for lab, fr in variants.items():
    cells = []
    for name, sig in SIGNALS.items():
        e, n = exc_on(fr, sig)
        cells.append(f"{e:>+8.3f}p n{n:<4}")
    print(f"    {lab:<30}" + "".join(f"{c:>21}" for c in cells))
print("     (Apr-2025 window = the 2020s' OTHER VIX>40 episode, matched 5-month length.)")

print("\n     rotation p for the 'ex BOTH' variant (5000 rolls, within-frame):")
fr = variants["  ex BOTH VIX>40 episodes"]
for name, sig in SIGNALS.items():
    f = fr[FWD]; ok = f.notna()
    m = sig.reindex(fr.index).fillna(False).astype(bool) & ok
    if m.sum() < 25:
        print(f"       {name:<20} n={int(m.sum())} INCONCLUSIVE (n<25)"); continue
    e = (f[m].mean() - f[ok].mean()) * 100
    p = rotation_pvalue(sig.reindex(fr.index).fillna(False).astype(bool).values,
                        f.values.astype(float), f[m].mean())
    print(f"       {name:<20} n={int(m.sum()):>4}  excess {e:>+7.3f}p  p={p:.3f} {stars(p)}")

# ================================================================= P3 residualise
print("\n" + "=" * 104)
print("### P3. THE SIMPLER VARIABLE: index mean reversion. Residualise g5 on trailing SPX move.")
print("     Within each era, fit g5 ~ a + b1*tr5 + b2*tr10 (+ b3*log(vix)) on ALL rows of that era,")
print("     then measure the signal's mean RESIDUAL excess. If VIX-stretch is only a proxy for")
print("     'the index just fell', the residual excess collapses toward zero everywhere.")


def resid_excess(sub, sig, cols):
    f = sub[FWD]
    X = sub[cols].copy()
    ok = f.notna() & X.notna().all(axis=1)
    if ok.sum() < 50:
        return np.nan, 0
    A = np.column_stack([np.ones(ok.sum())] + [X[c][ok].values for c in cols])
    y = f[ok].values
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = pd.Series(y - A @ beta, index=f[ok].index)
    m = sig.reindex(r.index).fillna(False).astype(bool)
    if m.sum() == 0:
        return np.nan, 0
    return (r[m].mean() - r.mean()) * 100, int(m.sum())


for cols, lab in ((["tr5"], "control tr5"),
                  (["tr5", "tr10"], "control tr5+tr10"),
                  (["tr5", "tr10", "dd63"], "control tr5+tr10+dd63"),
                  (["tr5", "tr10", "dd63", "vix"], "control tr5+tr10+dd63+VIX level")):
    print(f"\n  {lab}:")
    print(f"    {'signal':<20}" + "".join(f"{e[0]:>17}" for e in ERAS))
    for name, sig in SIGNALS.items():
        cells = []
        for _, y0, y1 in ERAS:
            sub = d[(d.index.year >= y0) & (d.index.year <= y1)]
            e, n = resid_excess(sub, sig, cols)
            cells.append(f"{e:>+8.3f}p n{n:<4}" if n >= 25 else f"{'n<25':>9} n{n:<4}")
        print(f"    {name:<20}" + "".join(f"{c:>17}" for c in cells))

print("\n  Same residualisation, but on the FULL 1990-2026 sample (one fit), to see whether the")
print("  headline signals survive the mean-reversion control at all:")
for cols, lab in ((["tr5"], "tr5"), (["tr5", "tr10", "dd63"], "tr5+tr10+dd63"),
                  (["tr5", "tr10", "dd63", "vix"], "tr5+tr10+dd63+VIX")):
    row = []
    for name, sig in SIGNALS.items():
        e, n = resid_excess(d, sig, cols)
        row.append(f"{name} {e:>+7.3f}p n={n}")
    print(f"    [{lab:<16}] " + " | ".join(row))
row = []
for name, sig in SIGNALS.items():
    e, n = exc_on(d, sig)
    row.append(f"{name} {e:>+7.3f}p n={n}")
print(f"    [{'raw (no control)':<16}] " + " | ".join(row))

# ================================================================= P4 matched control
print("\n" + "=" * 104)
print("### P4. MATCHED-CONTROL: compare each signal day only against same-era days in the SAME")
print("     (trailing-5d-return x VIX-level) cell. Removes the composition effect entirely.")
tr_edges = [-1, -.08, -.05, -.03, -.015, 0, .015, 1]
vx_edges = [0, 15, 18, 22, 27, 35, 1e9]
d["cell_tr"] = pd.cut(d.tr5, tr_edges)
d["cell_vx"] = pd.cut(d.vix, vx_edges)
print(f"    {'signal':<20}" + "".join(f"{e[0]:>17}" for e in ERAS))
for name, sig in SIGNALS.items():
    cells = []
    for _, y0, y1 in ERAS:
        sub = d[(d.index.year >= y0) & (d.index.year <= y1)].copy()
        ok = sub[FWD].notna() & sub.cell_tr.notna() & sub.cell_vx.notna()
        sub = sub[ok]
        sub["sig"] = sig.reindex(sub.index).fillna(False).astype(bool)
        parts, wts = [], []
        for key, g in sub.groupby(["cell_tr", "cell_vx"], observed=True):
            s, ns = g[g.sig], g[~g.sig]
            if len(s) == 0 or len(ns) < 3:
                continue
            parts.append(s[FWD].mean() - ns[FWD].mean()); wts.append(len(s))
        if not wts or sum(wts) < 25:
            cells.append(f"{'n<25':>9} n{int(sum(wts)):<4}"); continue
        e = float(np.average(parts, weights=wts)) * 100
        cells.append(f"{e:>+8.3f}p n{int(sum(wts)):<4}")
    print(f"    {name:<20}" + "".join(f"{c:>17}" for c in cells))
print("     (matched n < era n because cells with <3 controls are dropped -- reported honestly.)")

# ================================================================= P5 the honest 2020s
print("\n" + "=" * 104)
print("### P5. THE HONEST 2020s NUMBER: non-calendar VIX>=40 excision, with rotation p, all eras")
d_lvl = d[d.vix < 40]
for name, sig in SIGNALS.items():
    row = []
    for nm, y0, y1 in ERAS:
        sub = d_lvl[(d_lvl.index.year >= y0) & (d_lvl.index.year <= y1)]
        f = sub[FWD]; ok = f.notna()
        m = sig.reindex(sub.index).fillna(False).astype(bool) & ok
        if m.sum() < 25:
            row.append(f"{nm} n<25"); continue
        e = (f[m].mean() - f[ok].mean()) * 100
        p = rotation_pvalue(sig.reindex(sub.index).fillna(False).astype(bool).values,
                            f.values.astype(float), f[m].mean())
        row.append(f"{nm} {e:>+7.3f}p p={p:.3f} n={int(m.sum())}")
    print(f"    {name:<20} " + " | ".join(row))

print("\n### P5b. Does the 'flip' survive if COVID is excised but the 2020s is judged against the")
print("     SAME surgery applied to history (VIX>=40 dropped everywhere, incl. inside 2020s)?")
print("     ex-COVID-2020s vs ex-VIX40-2020s, side by side:")
d20ex = d20[~((d20.index >= COVID0) & (d20.index <= COVID1))]
d20lv = d20[d20.vix < 40]
for name, sig in SIGNALS.items():
    a, na = exc_on(d20ex, sig)
    b, nb = exc_on(d20lv, sig)
    c, nc = exc_on(variants["  ex BOTH VIX>40 episodes"], sig)
    print(f"    {name:<20} ex-COVID {a:>+7.3f}p n={na:<4} | ex-VIX40 {b:>+7.3f}p n={nb:<4} "
          f"| ex-both-episodes {c:>+7.3f}p n={nc:<4}")

print("\nDONE")
