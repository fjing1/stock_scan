"""
_vix_wf_verify_mechanism_lvlredundancy.py — self-contained (a sibling process overwrote the
module I was importing helpers from, so everything needed is inlined here).

CLAIM UNDER TEST (lower-band analysis, mechanism lens):
    bb10_2.0_below's D5 excess (~-0.58%, n~159) is NOT redundant with the VIX LEVEL --
    it retains ~84% under vix_pct1y vigintile FE and ~70% under a level+width+rv+momentum
    ladder, and is strongest in the MID level tercile.

Attack plan (mechanism / confounding):
  A  reproduce the headline and the claim's own control ladder
  B  push the LEVEL control much harder than the claim did:
       B1 year FE + level + width (level control that also removes the era in which the
          level occurred -- a 1994 VIX of 12 is not a 2017 VIX of 12)
       B2 1:20 nearest-neighbour matching on the ABSOLUTE VIX level inside a +/-252-session
          time caliper  (non-parametric, no functional form)
       B3 same, matching on vix_pct1y
       B4 count-matched single-variable rivals (is any plain "VIX is low" flag as good?)
  C  formal test of the claimed MID-TERCILE heterogeneity (is T2 really > T1?)
  D  MECHANISM: the band reduces algebraically to zz = (vix - ma10)/sd10 < -2.  Is the effect
     a smooth function of zz (band framing is decoration) or a threshold jump?  Plus a 10x10
     interacted (stretch x width) grid FE -- the band's literal ingredients.
  E  the label: "complacency" (low VIX) vs "post-spike collapse" (VIX falling off a spike)
  F  2010-2026 re-run of D/E

Rules: excess over the SAME-SAMPLE unconditional mean; circular-rotation p-values; g* only;
n<25 flagged inconclusive.

Run: ../../vcp_env/bin/python _vix_wf_verify_mechanism_lvlredundancy.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import _vix_data

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20260910)
N_ROT = 5000


def stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "   "
    return "***" if p < 0.01 else ("** " if p < 0.05 else ("*  " if p < 0.10 else "   "))


def rot_p_simple(mask, fwd, observed):
    n = len(mask)
    valid = ~np.isnan(fwd)
    null = np.empty(N_ROT)
    for i, off in enumerate(RNG.integers(1, n, size=N_ROT)):
        m = np.roll(mask, off) & valid
        null[i] = fwd[m].mean() if m.sum() else np.nan
    null = null[~np.isnan(null)]
    if not len(null):
        return np.nan
    b = null.mean()
    return float((np.abs(null - b) >= abs(observed - b)).mean())


def excess(y, mask, label=None, n_min=25):
    sel = mask & ~np.isnan(y)
    n = int(sel.sum())
    if n < n_min:
        if label:
            print(f"  {label:<58}   n={n:<5} INCONCLUSIVE (n<25)")
        return n, np.nan, np.nan
    cond = y[sel].mean()
    base = y[~np.isnan(y)].mean()
    p = rot_p_simple(mask, y, cond)
    if label:
        print(f"  {label:<58}{(cond-base)*100:>+8.3f}%   p={p:5.3f} {stars(p)}  n={n}")
    return n, cond - base, p


def fe_matrix(n, fe_list=(), extra=None):
    cols = [np.ones(n)]
    for codes in fe_list:
        codes = np.asarray(codes)
        for k in np.unique(codes)[1:]:
            cols.append((codes == k).astype(float))
    if extra is not None:
        cols.extend([np.asarray(c, float) for c in extra])
    return np.column_stack(cols)


def _basis(X, tol=1e-9):
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, s > tol * s.max()]


def fwl_coef(y_res, dvec, Q):
    dr = dvec - Q @ (Q.T @ dvec)
    den = float(dr @ dr)
    return float(dr @ y_res) / den if den > 1e-10 else np.nan


def controlled(y, X, mask, label, verbose=True):
    Q = _basis(X)
    y_res = y - Q @ (Q.T @ y)
    d = mask.astype(float)
    obs = fwl_coef(y_res, d, Q)
    null = np.empty(N_ROT)
    for i, off in enumerate(RNG.integers(1, len(d), size=N_ROT)):
        null[i] = fwl_coef(y_res, np.roll(d, off), Q)
    null = null[~np.isnan(null)]
    b = null.mean()
    p = float((np.abs(null - b) >= abs(obs - b)).mean())
    if verbose:
        print(f"  {label:<58}{obs*100:>+8.3f}%   p={p:5.3f} {stars(p)}  (k={Q.shape[1]})")
    return obs, p


def nn_match(y, sig, keyvals, caliper, kneigh, n):
    """mean(signal g5 - mean of kneigh nearest non-signal days by |key| within +/-caliper)."""
    idx = np.arange(n)
    pool = ~sig
    out = []
    for i in idx[sig]:
        if np.isnan(y[i]):
            continue
        lo, hi = max(0, i - caliper), min(n, i + caliper + 1)
        cand = idx[lo:hi][pool[lo:hi]]
        cand = cand[~np.isnan(y[cand])]
        if len(cand) < kneigh:
            continue
        pick = cand[np.argsort(np.abs(keyvals[cand] - keyvals[i]))[:kneigh]]
        out.append(y[i] - y[pick].mean())
    return np.array(out)


def main():
    d = _vix_data.add_features(_vix_data.load())
    rr = d.spx.pct_change()
    d["rv20"] = rr.rolling(20).std() * np.sqrt(252)
    d["r5"] = d.spx / d.spx.shift(5) - 1.0
    d["r20"] = d.spx / d.spx.shift(20) - 1.0
    d["zz"] = (d.vix - d.ma10) / d.vix.rolling(10).std(ddof=0)
    d["spx_ma10"] = d.spx / d.spx.rolling(10).mean() - 1.0
    d["vix_vs200"] = d.vix / d.vix.rolling(200).mean() - 1.0
    d["spike_ratio"] = d.vix.rolling(20).max() / d.vix
    d["vix_c1"] = d.vix.pct_change()
    d["spx_c1"] = d.spx.pct_change()

    cols = ["g5", "vix", "vix_pct1y", "stretch", "bb10_width", "bb10_2.0_below", "zz",
            "rv20", "r5", "r20", "spx_ma10", "vix_vs200", "spike_ratio", "vix_c1", "spx_c1"]
    a = d[cols].dropna().copy()
    a = a[a.g5.notna()]
    y = a.g5.values
    base = y.mean()
    below = a["bb10_2.0_below"].to_numpy(bool)
    n = len(a)
    k = int(below.sum())
    Qd = lambda s, kk=10: pd.qcut(a[s], kk, labels=False, duplicates="drop").values
    lvl20, wid, rv, r20q, r5q = Qd("vix_pct1y", 20), Qd("bb10_width"), Qd("rv20"), Qd("r20"), Qd("r5")
    yr = a.index.year.values

    print("=" * 100)
    print(f"FRAME n={n:,}  {a.index[0].date()} -> {a.index[-1].date()}   base g5 {base*100:+.3f}%"
          f"   below-band n={k}   (zz<-2 mismatch: {int((below != (a.zz.values < -2)).sum())})")
    print("=" * 100)

    print("\nA. REPRODUCE THE CLAIM'S OWN LADDER")
    raw, _ = controlled(y, fe_matrix(n), below, "no control (= raw excess)")
    o1, _ = controlled(y, fe_matrix(n, [lvl20]), below, "vix_pct1y VIGINTILE FE   [claim 84%]")
    o2, _ = controlled(y, fe_matrix(n, [lvl20, wid, rv, r20q, r5q]), below,
                       "level+width+rv+r20+r5 FE  [claim 70%]")
    print(f"    retention: level FE {o1/raw*100:.0f}%   full ladder {o2/raw*100:.0f}%")

    print("\nB. HARDER LEVEL CONTROLS THE CLAIM DID NOT RUN")
    o3, _ = controlled(y, fe_matrix(n, [yr]), below, "B1a YEAR FE alone")
    o4, p4 = controlled(y, fe_matrix(n, [yr, lvl20]), below, "B1b YEAR + vix_pct1y vigintile FE")
    o5, p5 = controlled(y, fe_matrix(n, [yr, lvl20, wid]), below, "B1c YEAR + level + width FE")
    o6, p6 = controlled(y, fe_matrix(n, [yr, lvl20, wid, rv, r20q, r5q]), below,
                        "B1d YEAR + full ladder FE")
    print(f"    retention: yr+level {o4/raw*100:.0f}%   yr+level+width {o5/raw*100:.0f}%   "
          f"yr+full {o6/raw*100:.0f}%")

    print()
    for key, cal, kk, lab in [("vix", 252, 20, "B2 NN match on ABS VIX, +/-252d, k=20"),
                              ("vix", 126, 10, "B2b NN match on ABS VIX, +/-126d, k=10"),
                              ("vix_pct1y", 252, 20, "B3 NN match on vix_pct1y, +/-252d, k=20")]:
        diffs = nn_match(y, below, a[key].values, cal, kk, n)
        null = np.empty(800)
        for b in range(800):
            m = np.roll(below, RNG.integers(1, n))
            dd = nn_match(y, m, a[key].values, cal, kk, n)
            null[b] = dd.mean() if len(dd) else np.nan
        null = null[~np.isnan(null)]
        bb = null.mean()
        pm = float((np.abs(null - bb) >= abs(diffs.mean() - bb)).mean())
        print(f"  {lab:<58}{diffs.mean()*100:>+8.3f}%   p={pm:5.3f} {stars(pm)}  "
              f"n={len(diffs)}  retention {diffs.mean()/raw*100:.0f}%")

    print("\n  B4 count-matched single-variable 'VIX is low' rivals (same n as the claim)")
    def bottom_k(col, kk=k):
        v = a[col].values
        return v <= np.sort(v)[kk - 1]
    excess(y, below, "the claim: bb10_2.0_below")
    for col, lab in [("vix", "lowest ABSOLUTE VIX"), ("vix_pct1y", "lowest vix_pct1y"),
                     ("vix_vs200", "lowest VIX vs its own 200d mean"),
                     ("stretch", "most-negative stretch (VIX vs ma10)"),
                     ("bb10_width", "tightest bb10_width")]:
        m = bottom_k(col)
        nn_, ex, p = excess(y, m, f"    count-matched {lab}")
        print(f"        overlap with claim: {int((m & below).sum())}/{k}")

    print("\nC. IS THE MID-TERCILE HETEROGENEITY REAL?")
    ter = pd.qcut(a.vix_pct1y, 3, labels=False).values
    inner = {}
    for t in range(3):
        sub = ter == t
        m = below & sub
        nn_ = int((m & ~np.isnan(y)).sum())
        bin_ = y[sub & ~np.isnan(y)].mean()
        if nn_ < 25:
            print(f"  tercile {t+1}: below n={nn_} INCONCLUSIVE (n<25)")
            inner[t] = np.nan
            continue
        inner[t] = y[m].mean() - bin_
        print(f"  tercile {t+1}: n={nn_:<4} within-tercile excess {inner[t]*100:+.3f}%")
    diff = inner[1] - inner[0]
    # rotation null for the DIFFERENCE between tercile-2 and tercile-1 within-excess
    null = np.empty(N_ROT)
    for i, off in enumerate(RNG.integers(1, n, size=N_ROT)):
        m = np.roll(below, off)
        vals = []
        for t in (0, 1):
            sub = ter == t
            mm = m & sub & ~np.isnan(y)
            vals.append(y[mm].mean() - y[sub & ~np.isnan(y)].mean() if mm.sum() >= 10 else np.nan)
        null[i] = vals[1] - vals[0]
    null = null[~np.isnan(null)]
    bb = null.mean()
    pdiff = float((np.abs(null - bb) >= abs(diff - bb)).mean())
    print(f"  T2 - T1 = {diff*100:+.3f}%   rotation p = {pdiff:.3f} {stars(pdiff)}"
          f"   -> heterogeneity {'IS' if pdiff < 0.05 else 'is NOT'} established")

    print("\nD. MECHANISM: the band is a threshold on zz = (vix-ma10)/sd10")
    edges = [-np.inf, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (a.zz.values >= lo) & (a.zz.values < hi)
        nn_ = int((m & ~np.isnan(y)).sum())
        lab = f"  zz in [{lo:>5.1f}, {hi:>5.1f})"
        if nn_ < 25:
            print(f"{lab:<32} n={nn_:<5} INCONCLUSIVE (n<25)")
        else:
            print(f"{lab:<32} n={nn_:<5} excess {(y[m].mean()-base)*100:+7.3f}%")
    print()
    zzv = a.zz.values
    for lo, hi, lab in [(-np.inf, -2.5, "D-RD deep below      zz < -2.5"),
                        (-2.50, -2.00, "D-RD just below      zz in [-2.5,-2.0)"),
                        (-2.00, -1.75, "D-RD just ABOVE      zz in [-2.0,-1.75)"),
                        (-1.75, -1.50, "D-RD                 zz in [-1.75,-1.5)")]:
        excess(y, (zzv >= lo) & (zzv < hi), lab)
    controlled(y, fe_matrix(n, [], [zzv]), below, "D1 dummy | zz LINEAR")
    controlled(y, fe_matrix(n, [], [zzv, zzv**2, zzv**3]), below, "D2 dummy | zz cubic")
    st = Qd("stretch")
    controlled(y, fe_matrix(n, [st * 10 + wid]), below, "D3 dummy | stretch x width 10x10 GRID FE")
    controlled(y, fe_matrix(n, [Qd("spx_ma10")]), below, "D4 dummy | SPX vs 10dma decile FE")
    controlled(y, fe_matrix(n, [Qd("vix_c1", 20), Qd("spx_c1", 20)]), below,
               "D5 dummy | 1d VIX chg + 1d SPX ret vigintile FE")

    print("\nE. LABEL CHECK: 'complacency' (low VIX) vs 'post-spike collapse'")
    for lab, m in [("VIX < 15  (true complacency)", below & (a.vix.values < 15)),
                   ("VIX 15-25", below & (a.vix.values >= 15) & (a.vix.values < 25)),
                   ("VIX >= 25 (post-spike)", below & (a.vix.values >= 25)),
                   ("spike_ratio < 1.25 (no recent spike)", below & (a.spike_ratio.values < 1.25)),
                   ("spike_ratio >= 1.25 (>=25% off a 20d VIX high)",
                    below & (a.spike_ratio.values >= 1.25))]:
        excess(y, m, "  " + lab)
    tot = (y[below] - base).sum()
    for lab, m in [("VIX<15", below & (a.vix.values < 15)),
                   ("VIX 15-25", below & (a.vix.values >= 15) & (a.vix.values < 25)),
                   ("VIX>=25", below & (a.vix.values >= 25))]:
        print(f"    {lab:<12} n={int(m.sum()):<4} share of total excess dollars "
              f"{(y[m]-base).sum()/tot*100:6.1f}%")

    print("\nF. 2010-2026 ONLY (baseline recomputed inside the window)")
    win = a.index.year >= 2010
    yw, bw, aw = y[win], below[win], a[win]
    print(f"  window n={int(win.sum()):,}  baseline {np.nanmean(yw)*100:+.3f}%  "
          f"below n={int(bw.sum())}")
    for lab, m in [("all below-band", bw),
                   ("  VIX < 15", bw & (aw.vix.values < 15)),
                   ("  VIX >= 15", bw & (aw.vix.values >= 15)),
                   ("  spike_ratio >= 1.25", bw & (aw.spike_ratio.values >= 1.25))]:
        excess(yw, m, lab)


if __name__ == "__main__":
    main()
