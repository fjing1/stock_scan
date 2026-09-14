"""
_vix_wf_verify_mechanism_upperband2.py — follow-ups that decide the remaining ambiguities in
the upper-band redundancy claim.

Round 1 killed my power/over-control attack (absorption only 11.6%, MDE ~0.24%, injection test
detects a true +0.30% at p=.007, ATT == FE-OLS, reverse Frisch-Waugh is asymmetric). What is left:

  I.  Is the SHRINKAGE itself significant?  Paired SE of (raw coef - controlled coef).
  J.  Overlap accounting: how many band days actually ARE "VIX high" / "index just fell" days?
  K.  Round 1's only counter-evidence was the disjoint cell "band ONLY = +0.26%, p=.054".
      Is that a real residual or an artifact of controlling with ONE coarse cut?  Re-run it
      with progressively finer joint matching.
  L.  Era stability of the CONFOUND itself. "Explained by X" is weak if X is also dead.
  M.  Nested model comparison: does the band dummy add anything to a level+reversal model?

Run: ../../vcp_env/bin/python _vix_wf_verify_mechanism_upperband2.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _vix_data

RNG = np.random.default_rng(20260911)
N_ROT = 5000
SIG = "bb10_2.0_above"
ERAS = [("1990s", 1990, 1999), ("2000s", 2000, 2009), ("2010s", 2010, 2019), ("2020s", 2020, 2099)]


def stars(p):
    return "***" if p < 0.01 else ("** " if p < 0.05 else ("*  " if p < 0.10 else "   "))


def fe_matrix(n, fe_list):
    cols = [np.ones(n)]
    for codes in fe_list:
        codes = np.asarray(codes)
        for k in np.unique(codes)[1:]:
            cols.append((codes == k).astype(float))
    return np.column_stack(cols)


def hac(u, L=10):
    s = float(u @ u)
    for l in range(1, L + 1):
        s += 2.0 * (1.0 - l / (L + 1.0)) * float(u[l:] @ u[:-l])
    return s


def fwl(y, X, dv):
    Q, _ = np.linalg.qr(X)
    y_res = y - Q @ (Q.T @ y)
    dr = dv - Q @ (Q.T @ dv)
    den = float(dr @ dr)
    coef = float(dr @ y_res) / den
    e = y_res - coef * dr
    return coef, dr * e, den, dr


def rot_p_mean(mask, y):
    obs = y[mask].mean()
    null = np.array([y[np.roll(mask, o)].mean()
                     for o in RNG.integers(1, len(mask), size=N_ROT)])
    b = null.mean()
    return float((np.abs(null - b) >= abs(obs - b)).mean())


def matched(mask, bins, y):
    num = den = 0.0
    used = 0
    for b in np.unique(bins):
        s = bins == b
        t, c = s & mask, s & ~mask
        if t.sum() == 0 or c.sum() == 0:
            continue
        num += t.sum() * (y[t].mean() - y[c].mean())
        den += t.sum()
        used += 1
    return (num / den if den else np.nan), int(den), used


def rot_p_matched(mask, bins, y, obs):
    null = np.array([matched(np.roll(mask, o), bins, y)[0]
                     for o in RNG.integers(1, len(mask), size=N_ROT)])
    null = null[~np.isnan(null)]
    b = null.mean()
    return float((np.abs(null - b) >= abs(obs - b)).mean())


def main():
    d = _vix_data.add_features(_vix_data.load())
    d["rv20"] = d.spx.pct_change().rolling(20).std() * np.sqrt(252)
    d["spx_r20"] = d.spx / d.spx.shift(20) - 1.0
    d["spx_r5"] = d.spx / d.spx.shift(5) - 1.0
    d["vrp"] = d.vix / 100.0 - d.rv20
    keep = ["g5", "vix_pct1y", "bb10_width", "rv20", "spx_r20", "spx_r5", "vrp", SIG]
    a = d[keep].dropna().copy()
    y = a.g5.values
    n = len(a)
    above = a[SIG].astype(bool).values
    yr = a.index.year.values
    q = lambda s, k: pd.qcut(a[s], k, labels=False, duplicates="drop").values
    lvl20, r5d = q("vix_pct1y", 20), q("spx_r5", 10)
    FULL = [lvl20, q("bb10_width", 10), q("rv20", 10), q("spx_r20", 10), r5d]

    # ---------------------------------------------------------------- I
    print("=" * 100)
    print("I. IS THE SHRINKAGE ITSELF SIGNIFICANT?  paired HAC test of (raw coef - controlled coef)")
    print("=" * 100)
    dv = above.astype(float)
    c0, u0, den0, _ = fwl(y, fe_matrix(n, []), dv)
    print(f"  {'control':<40}{'raw':>9}{'ctrl':>9}{'drop':>9}{'HAC se(drop)':>14}{'t':>7}{'p':>8}")
    print("  " + "-" * 96)
    for lbl, fes in (("vix_pct1y vigintile FE", [lvl20]),
                     ("SPX 5d-return decile FE", [r5d]),
                     ("EVERYTHING FE", FULL)):
        c1, u1, den1, _ = fwl(y, fe_matrix(n, fes), dv)
        # influence-function difference: the two estimators share the same rows, so difference
        # out the scores and HAC the paired series (this is the SE of the DROP, not of a level)
        ud = u0 / den0 - u1 / den1
        se = float(np.sqrt(max(hac(ud), 0.0)))
        drop = c0 - c1
        t = drop / se if se else np.nan
        from math import erfc, sqrt
        p = erfc(abs(t) / sqrt(2.0)) if se else np.nan
        print(f"  {lbl:<40}{c0*100:>+8.3f}%{c1*100:>+8.3f}%{drop*100:>+8.3f}%"
              f"{se*100:>13.3f}%{t:>7.2f}{p:>8.4f}{stars(p)}")
    print("\n  If the drop is significant, the controls really did explain the effect;")
    print("  if not, 'falls to 14% of raw' is just estimator wobble.")

    # ---------------------------------------------------------------- J
    print()
    print("=" * 100)
    print("J. OVERLAP ACCOUNTING — is the band literally the same days as the simple signals?")
    print("=" * 100)
    k = int(above.sum())
    r5 = a.spx_r5.values
    lv = a.vix_pct1y.values
    fall = r5 <= np.sort(r5)[k - 1]
    hi = lv >= np.sort(lv)[-k]
    print(f"  band days n={k}")
    print(f"    also in the {k} worst SPX-5d days : {int((above & fall).sum())} "
          f"({(above & fall).sum()/k*100:.0f}%)   corr={np.corrcoef(dv, fall.astype(float))[0,1]:+.3f}")
    print(f"    also in the {k} highest vix_pct1y : {int((above & hi).sum())} "
          f"({(above & hi).sum()/k*100:.0f}%)   corr={np.corrcoef(dv, hi.astype(float))[0,1]:+.3f}")
    print(f"  mean vix_pct1y  on band days {lv[above].mean():.3f} vs all days {lv.mean():.3f}")
    print(f"  mean SPX 5d ret on band days {r5[above].mean()*100:+.2f}% vs all days {r5.mean()*100:+.2f}%")
    print(f"  corr(band, vix_pct1y) = {np.corrcoef(dv, lv)[0,1]:+.3f}    "
          f"corr(band, spx_r5) = {np.corrcoef(dv, r5)[0,1]:+.3f}")

    # ---------------------------------------------------------------- K
    print()
    print("=" * 100)
    print("K. THE 'band ONLY = +0.26%' RESIDUAL — real, or an artifact of a single coarse cut?")
    print("   Same question asked with progressively finer joint (level x reversal) matching.")
    print("=" * 100)
    print(f"  {'control resolution':<52}{'band residual':>15}{'n':>7}{'cells':>7}{'rot p':>8}")
    print("  " + "-" * 92)
    grids = [("one coarse cut: not-in-worst-518-SPX5d (round 1)", None),
             ("2 x 2 (level median x SPX5d median)", q("vix_pct1y", 2) * 2 + q("spx_r5", 2)),
             ("3 x 3 terciles", q("vix_pct1y", 3) * 3 + q("spx_r5", 3)),
             ("5 x 5 quintiles", q("vix_pct1y", 5) * 5 + q("spx_r5", 5)),
             ("8 x 8 octiles", q("vix_pct1y", 8) * 8 + q("spx_r5", 8)),
             ("10 x 10 deciles", q("vix_pct1y", 10) * 10 + q("spx_r5", 10))]
    for lbl, b in grids:
        if b is None:
            m = above & ~fall
            e = y[m].mean() - y.mean()
            print(f"  {lbl:<52}{e*100:>+14.3f}%{int(m.sum()):>7}{2:>7}"
                  f"{rot_p_mean(m, y):>8.3f}{stars(rot_p_mean(m, y))}")
            continue
        md, nt, nb = matched(above, b, y)
        p = rot_p_matched(above, b, y, md)
        print(f"  {lbl:<52}{md*100:>+14.3f}%{nt:>7}{nb:>7}{p:>8.3f}{stars(p)}")

    # ---------------------------------------------------------------- L
    print()
    print("=" * 100)
    print("L. ERA STABILITY OF THE CONFOUND ITSELF — 'explained by X' is weak if X is also dead")
    print("=" * 100)
    print(f"  {'era':<10}{'n band':>8}{'band exc':>11}{'n SPX5d-low':>13}{'SPX5d-low exc':>15}"
          f"{'n VIXhi':>9}{'VIXhi exc':>12}")
    print("  " + "-" * 92)
    for nm, y0, y1 in ERAS:
        s = (yr >= y0) & (yr <= y1)
        b = y[s].mean()
        cells = []
        for m in (above, fall, hi):
            mm = m & s
            cells.append((int(mm.sum()), (y[mm].mean() - b) if mm.sum() >= 25 else np.nan))
        print(f"  {nm:<10}{cells[0][0]:>8}{cells[0][1]*100:>+10.3f}%{cells[1][0]:>13}"
              f"{cells[1][1]*100:>+14.3f}%{cells[2][0]:>9}{cells[2][1]*100:>+11.3f}%")
    print("  (era baselines differ; each excess is vs that era's own unconditional g5 mean)")

    # ---------------------------------------------------------------- M
    print()
    print("=" * 100)
    print("M. NESTED MODEL — does the band dummy add explanatory power over level + reversal?")
    print("=" * 100)
    def r2(X):
        Q, _ = np.linalg.qr(X)
        yr_ = y - Q @ (Q.T @ y)
        return 1.0 - float(yr_ @ yr_) / float(((y - y.mean()) @ (y - y.mean())))
    Xb = fe_matrix(n, [lvl20, r5d])
    Xbb = np.column_stack([Xb, dv])
    Xband = fe_matrix(n, [above.astype(int)])
    print(f"  R^2  level(20) + SPX5d(10) FE only            : {r2(Xb)*100:.4f}%")
    print(f"  R^2  + band dummy                             : {r2(Xbb)*100:.4f}%")
    print(f"  R^2  band dummy alone                         : {r2(Xband)*100:.4f}%")
    print(f"  incremental R^2 of the band over level+reversal: {(r2(Xbb)-r2(Xb))*100:.4f}%")
    print(f"  incremental R^2 of level+reversal over band    : {(r2(Xbb)-r2(Xband))*100:.4f}%")


if __name__ == "__main__":
    main()
