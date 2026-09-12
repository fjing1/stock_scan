"""
_vix_wf_verify_mechanism_lowerband2.py — round 2 of the adversarial mechanism audit.

Round 1 result: the claim REPLICATES exactly (-0.486% under level-vigintile FE, p=.018;
-0.407% under the claim's full ladder, p=.046) and my two omitted mechanism controls
(vol-normalised SPX overboughtness, raw VIX 5d collapse) only shave it to -0.367% (p=.072).
So "redundant with the VIX LEVEL" is NOT the right refutation.

But round 1 turned up two things worth pushing on:
  * the dose-response on the underlying continuous statistic z10=(vix-ma10)/sd10 is
    NON-MONOTONE: z10<-1.5 -> -0.338% (p=.002, n=916); z10<-2.0 -> -0.486% (p=.018, n=159);
    z10<-2.25 -> +0.032% (p=.919, n=57).  The most extreme third of "below band" days has NO
    effect at all.
  * a smooth cubic in z10 (plus level FE) reduces the dummy to -0.198% (p=.322), and a
    regression-discontinuity at -2.0 with a [-2.85,-1.25] window gives -0.098% (p=.587).

Round 2 therefore asks:
  K. what is the actual SHAPE of E[g5 | z10] after level FE?  jump or slope?
  L. proper local-linear RD at the -2.0 cutoff across bandwidths (+ HAC t as a second opinion)
  M. saturated 2-WAY cell FE (level x SPX-overbought, level x VIX-collapse, level x
     how-far-VIX-has-fallen-from-its-60d-peak) — a much stronger control than additive FE
  N. genuinely out-of-sample era split: does the coefficient exist after 2010?
  O. the shallower cut (z10<-1.5) head-to-head against the 2.0 band
  P. how large is the search space this cell was picked from (multiplicity)?

Run: ../../vcp_env/bin/python _vix_wf_verify_mechanism_lowerband2.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _vix_data

RNG = np.random.default_rng(20260915)
PCT = 100.0
N_ROT = 20000


def stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "   "
    return "***" if p < 0.01 else ("** " if p < 0.05 else ("*  " if p < 0.10 else "   "))


def basis(X, tol=1e-9):
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, s > tol * s.max()]


def fe_matrix(n, fe_list=(), extra=None):
    cols = [np.ones(n)]
    for codes in fe_list:
        codes = np.asarray(codes)
        for k in np.unique(codes)[1:]:
            cols.append((codes == k).astype(float))
    if extra is not None:
        cols.extend([np.asarray(c, float) for c in extra])
    return np.column_stack(cols)


def rotation_all(y_res, dv, Q):
    n = len(dv)
    fd = np.conj(np.fft.rfft(dv))
    num = np.fft.irfft(fd * np.fft.rfft(y_res), n)
    sq = np.zeros(n)
    for j in range(Q.shape[1]):
        a = np.fft.irfft(fd * np.fft.rfft(Q[:, j]), n)
        sq += a * a
    den = float(dv @ dv) - sq
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-8, num / den, np.nan)


def fe_test(y, X, mask, label, quiet=False):
    Q = basis(X)
    y_res = y - Q @ (Q.T @ y)
    coef = rotation_all(y_res, np.asarray(mask, float), Q)
    obs, null = coef[0], coef[1:][~np.isnan(coef[1:])]
    b = null.mean()
    p = float((np.abs(null - b) >= abs(obs - b)).mean())
    if not quiet:
        print(f"  {label:<62}{obs*PCT:>+9.3f}%  p={p:5.3f} {stars(p)}  (k={Q.shape[1]})")
    return obs, p


def nw_t(y, X, j, lags=5):
    """Newey-West t-stat on coefficient j — a second opinion that is not a rotation test."""
    XtX = np.linalg.pinv(X.T @ X)
    b = XtX @ (X.T @ y)
    e = y - X @ b
    n, k = X.shape
    S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        A = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
        S += w * (A + A.T)
    V = XtX @ S @ XtX
    return float(b[j]), float(b[j] / np.sqrt(max(V[j, j], 1e-30)))


def main():
    d = _vix_data.add_features(_vix_data.load())
    sx = d.spx.dropna()
    rr = sx.pct_change(fill_method=None)
    px = lambda s: s.reindex(d.index)
    d["rv20"] = px(rr.rolling(20).std() * np.sqrt(252))
    d["spx_r5"] = px(sx / sx.shift(5) - 1.0)
    d["spx_r20"] = px(sx / sx.shift(20) - 1.0)
    for n in (10, 20):
        d[f"spx_z{n}"] = px((sx - sx.rolling(n).mean()) / sx.rolling(n).std(ddof=0))
    d["vix_chg5"] = d.vix / d.vix.shift(5) - 1.0
    # how far has VIX fallen from its own recent peak? (the "post-spike vol crush" story)
    d["vix_off60"] = d.vix / d.vix.rolling(60).max() - 1.0
    d["vix_off20"] = d.vix / d.vix.rolling(20).max() - 1.0
    d["z10"] = (d.vix - d.vix.rolling(10).mean()) / d.vix.rolling(10).std(ddof=0)

    keep = ["g5", "vix_pct1y", "bb10_width", "rv20", "spx_r5", "spx_r20",
            "spx_z10", "spx_z20", "vix_chg5", "vix_off60", "vix_off20", "z10",
            "bb10_2.0_below"]
    a = d[keep].dropna().copy()
    y = a.g5.values
    n = len(a)
    below = a["bb10_2.0_below"].astype(bool).values
    z = a.z10.values
    QC = lambda s, k: pd.qcut(a[s], k, labels=False, duplicates="drop").values
    lvl20, lvl10 = QC("vix_pct1y", 20), QC("vix_pct1y", 10)
    print("=" * 100)
    print(f"FRAME {n:,} rows {a.index[0].date()} -> {a.index[-1].date()}  "
          f"base g5 {y.mean()*PCT:+.3f}%  below n={below.sum()}")
    print("=" * 100)

    # ============================================================ K. shape of E[g5|z10]
    print("\n" + "=" * 100)
    print("K. SHAPE — level-FE-residualised mean g5 by z10 bin.  Is there a JUMP at -2.0?")
    print("=" * 100)
    Q = basis(fe_matrix(n, [lvl20]))
    yres = y - Q @ (Q.T @ y)          # residual: mean 0 within each level vigintile
    edges = [-2.85, -2.4, -2.25, -2.1, -2.0, -1.9, -1.75, -1.6, -1.45, -1.3, -1.15, -1.0,
             -0.75, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    print(f"  {'z10 bin':<20}{'n':>7}{'resid mean g5':>16}   {'(cum n below cut)':>18}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (z >= lo) & (z < hi)
        tag = "  <== BAND CUT at -2.0" if abs(hi + 2.0) < 1e-9 else ""
        flag = " (n<25)" if m.sum() < 25 else ""
        print(f"  [{lo:+.2f},{hi:+.2f}){m.sum():>7}{yres[m].mean()*PCT:>+15.3f}%"
              f"{flag}{tag}")
    print("\n  the same, coarse, either side of the cut:")
    for lab, m in (("z10 < -2.25            ", z < -2.25),
                   ("z10 in [-2.25,-2.00)   ", (z >= -2.25) & (z < -2.0)),
                   ("z10 in [-2.00,-1.75)   ", (z >= -2.0) & (z < -1.75)),
                   ("z10 in [-1.75,-1.50)   ", (z >= -1.75) & (z < -1.5)),
                   ("z10 in [-1.50,-1.00)   ", (z >= -1.5) & (z < -1.0))):
        f = " INCONCLUSIVE (n<25)" if m.sum() < 25 else ""
        print(f"    {lab} n={m.sum():>5}  resid mean {yres[m].mean()*PCT:+7.3f}%{f}")

    # ============================================================ L. local-linear RD
    print("\n" + "=" * 100)
    print("L. LOCAL-LINEAR RD AT z10 = -2.0  (y ~ 1 + (z+2) + below + below*(z+2), within +-h)")
    print("=" * 100)
    print(f"  {'h':>6}{'nL':>7}{'nR':>7}{'jump':>11}{'rot p':>9}{'NW t(5)':>10}")
    for h in (0.25, 0.35, 0.45, 0.60, 0.80, 1.00):
        w = np.abs(z + 2.0) <= h
        zl, yl, bl = z[w] + 2.0, y[w], below[w].astype(float)
        nL, nR = int(bl.sum()), int((1 - bl).sum())
        if min(nL, nR) < 25:
            print(f"  {h:>6.2f}{nL:>7}{nR:>7}    INCONCLUSIVE (a side has n<25)")
            continue
        X = np.column_stack([np.ones(w.sum()), zl, bl, bl * zl])
        b, t = nw_t(yl, X, 2)
        # rotation null inside the window
        Xc = np.column_stack([np.ones(w.sum()), zl])
        Qw = basis(Xc)
        yw = yl - Qw @ (Qw.T @ yl)
        cf = rotation_all(yw, bl, Qw)
        nu = cf[1:][~np.isnan(cf[1:])]
        bb = nu.mean()
        pr = float((np.abs(nu - bb) >= abs(cf[0] - bb)).mean())
        print(f"  {h:>6.2f}{nL:>7}{nR:>7}{b*PCT:>+10.3f}%{pr:>9.3f}{t:>10.2f}")
    print("\n  (a real threshold effect should show a stable jump across bandwidths;")
    print("   a smooth slope in z10 mis-modelled as a step will not)")

    # ============================================================ M. saturated 2-way cells
    print("\n" + "=" * 100)
    print("M. SATURATED 2-WAY CELL FE (level x mechanism) — much stronger than additive FE")
    print("=" * 100)
    fe_test(y, fe_matrix(n, [lvl20]), below, "level vigintile FE (baseline, = the claim)")
    for col, lab in (("spx_z10", "SPX Bollinger z(10d) — index overbought"),
                     ("spx_z20", "SPX Bollinger z(20d)"),
                     ("vix_chg5", "VIX 5d % change"),
                     ("vix_off60", "VIX vs its 60d max — post-spike vol crush"),
                     ("vix_off20", "VIX vs its 20d max"),
                     ("spx_r5", "SPX 5d return"),
                     ("bb10_width", "band width")):
        c10 = QC(col, 10)
        cell = lvl10 * 10 + c10
        fe_test(y, fe_matrix(n, [cell]), below, f"CELL FE: level-decile x {lab}")
    cell3 = lvl10 * 100 + QC("spx_z10", 10) * 10 + QC("vix_chg5", 10)
    fe_test(y, fe_matrix(n, [cell3]), below, "CELL FE: level x SPX-z10 x VIX-chg5 (3-way)")

    # ============================================================ N. era split
    print("\n" + "=" * 100)
    print("N. TRUE ERA SPLIT — does the coefficient exist in the second half of the sample?")
    print("=" * 100)
    Xlad = fe_matrix(n, [lvl20, QC("bb10_width", 10), QC("rv20", 10),
                         QC("spx_r20", 10), QC("spx_r5", 10),
                         QC("spx_z10", 10), QC("vix_chg5", 10)])
    yrs = a.index.year.values
    for lab, sel in (("1991-2009 (first half)", yrs < 2010),
                     ("2010-2026 (second half)", yrs >= 2010),
                     ("1991-2004", yrs < 2005),
                     ("2005-2026", yrs >= 2005),
                     ("2013-2026 (last ~half of below days? check n)", yrs >= 2013)):
        s = sel
        nb = int(below[s].sum())
        ys, bs = y[s], below[s]
        raw = ys[bs].mean() - ys[~bs].mean() if nb else np.nan
        if nb < 25:
            print(f"  {lab:<45} below n={nb:>4}   raw {raw*PCT:+7.3f}%   INCONCLUSIVE (n<25)")
            continue
        lv = pd.qcut(a.vix_pct1y.values[s], 20, labels=False, duplicates="drop")
        o1, p1 = fe_test(ys, fe_matrix(int(s.sum()), [lv]), bs, "", quiet=True)
        print(f"  {lab:<45} below n={nb:>4}   raw {raw*PCT:+7.3f}%   "
              f"levelFE {o1*PCT:+7.3f}% p={p1:5.3f} {stars(p1)}")

    # ============================================================ O. shallow vs deep cut
    print("\n" + "=" * 100)
    print("O. WHICH CUT IS THE REAL SIGNAL?  head-to-head, both dummies in one model")
    print("=" * 100)
    shallow = (z < -1.5) & ~below          # in [-2.0,-1.5) only
    deep = below
    X = fe_matrix(n, [lvl20], extra=[shallow.astype(float)])
    fe_test(y, X, deep, "below-2.0  |  level FE + [-2.0<=z<-1.5] dummy")
    X2 = fe_matrix(n, [lvl20], extra=[deep.astype(float)])
    fe_test(y, X2, shallow, "[-2.0<=z<-1.5]  |  level FE + below-2.0 dummy")
    print(f"    n: below-2.0 = {deep.sum()}, [-2.0,-1.5) = {shallow.sum()}")
    print("\n  and the nested-tail decomposition of the 159 below-band days:")
    inner = below & (z >= -2.25)
    outer = below & (z < -2.25)
    for lab, m in (("z in [-2.25,-2.00)", inner), ("z < -2.25 (deepest)", outer)):
        f = "  INCONCLUSIVE (n<25)" if m.sum() < 25 else ""
        o, p = fe_test(y, fe_matrix(n, [lvl20]), m, "", quiet=True)
        print(f"    {lab:<24} n={m.sum():>4}  levelFE coef {o*PCT:+7.3f}%  p={p:5.3f} "
              f"{stars(p)}{f}")

    # ============================================================ P. multiplicity
    print("\n" + "=" * 100)
    print("P. SEARCH SPACE — every band/side/horizon cell in the shared feature set, level FE")
    print("=" * 100)
    hits = []
    for nn in (10, 20):
        for k in (1.5, 2.0, 2.5):
            for side in ("below", "above", "reentry", "exit_lo"):
                col = f"bb{nn}_{k}_{side}"
                m = d[col].reindex(a.index).astype(bool).values
                for h in (1, 3, 5, 10, 21):
                    yy = d[f"g{h}"].reindex(a.index).values
                    ok = ~np.isnan(yy)
                    if m[ok].sum() < 25:
                        continue
                    lv = pd.qcut(a.vix_pct1y.values[ok], 20, labels=False, duplicates="drop")
                    o, p = fe_test(yy[ok], fe_matrix(int(ok.sum()), [lv]), m[ok], "", quiet=True)
                    hits.append((col, h, int(m[ok].sum()), o, p))
    hits_df = pd.DataFrame(hits, columns=["signal", "h", "n", "coef", "p"])
    tot = len(hits_df)
    sig5 = int((hits_df.p < 0.05).sum())
    sig1 = int((hits_df.p < 0.01).sum())
    print(f"  {tot} interpretable cells tested (n>=25).  p<.05 in {sig5} "
          f"({sig5/tot*100:.0f}%, expected 5% by chance = {tot*0.05:.1f});  p<.01 in {sig1} "
          f"(expected {tot*0.01:.1f})")
    print("  most significant 12:")
    print(hits_df.assign(coef=lambda t: (t.coef * PCT).round(3))
                 .sort_values("p").head(12).to_string(index=False))
    rank = int((hits_df.sort_values("p").reset_index(drop=True)
                .query("signal=='bb10_2.0_below' and h==5").index[0])) + 1
    print(f"\n  bb10_2.0_below @ D5 ranks {rank} of {tot} by p-value.  "
          f"Sidak-adjusted p for the BEST cell would need p < {1-(1-0.05)**(1/tot):.4f}.")
    bonf = hits_df.p.min() * tot
    print(f"  Bonferroni over the {tot} cells: best raw p={hits_df.p.min():.4f} -> "
          f"{min(bonf,1.0):.3f};  bb10_2.0_below@D5 raw p="
          f"{float(hits_df.query('signal==\"bb10_2.0_below\" and h==5').p.iloc[0]):.3f} -> "
          f"{min(float(hits_df.query('signal==\"bb10_2.0_below\" and h==5').p.iloc[0])*tot,1.0):.3f}")


if __name__ == "__main__":
    main()
