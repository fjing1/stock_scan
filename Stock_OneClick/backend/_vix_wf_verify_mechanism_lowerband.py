"""
_vix_wf_verify_mechanism_lowerband.py — ADVERSARIAL mechanism/confounding audit of the claim:

  "bb10_2.0_below's D5 excess (-0.58%, n=159) is NOT redundant with VIX level; it keeps 84%
   under vix_pct1y vigintile FE and 70% under a level+width+realized-vol+momentum ladder,
   and is strongest in the MID level tercile."

The prior work controlled for: vix_pct1y (vigintile), bb10_width, rv20, spx_r20, spx_r5, VRP,
stretch decile, raw-vix decile.  It did NOT control for the two variables that are the most
mechanically-adjacent competing explanations:

  (1) SPX MEAN REVERSION / OVERBOUGHT, measured the way the VIX band measures VIX --
      vol-normalised distance of SPX from its own short MA (SPX's own Bollinger z-score).
      A VIX collapse under its lower band is near-mechanically an SPX melt-up above its own
      upper band.  Raw 5d/20d RETURN deciles are NOT the same control: they are not
      normalised by the prevailing dispersion, and the band is.
  (2) "VIX JUST CRASHED" -- the 5d/10d % change in VIX.  The band is a threshold on a
      normalised VIX drop; the simple un-normalised drop is the parsimonious rival.

Plus: calendar FE (year / decade / month / dow), longer-horizon level (vix vs its 200d MA),
trend state (SPX vs 200MA, drawdown from 252d high), a regression-discontinuity test of
whether -2.0 sd is a real threshold or just a slice of a monotone slope, a formal test of the
claimed MID-tercile heterogeneity, leave-one-year-out, and the full band-parameter grid
(multiplicity).

Inference: exact FULL circular-rotation null (all n-1 offsets, computed by FFT) of the
Frisch-Waugh dummy coefficient -- same estimator as the claim, but the null uses every
rotation instead of 5,000 random ones.

Run: ../../vcp_env/bin/python _vix_wf_verify_mechanism_lowerband.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _vix_data

RNG = np.random.default_rng(20260914)
PCT = 100.0


def stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "   "
    return "***" if p < 0.01 else ("** " if p < 0.05 else ("*  " if p < 0.10 else "   "))


# ----------------------------------------------------------------- linear algebra
def basis(X, tol=1e-9):
    """Orthonormal basis for col(X), rank-safe (handles collinear FE dummies)."""
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


def rotation_all(y_res, d, Q):
    """coef[s] = FWL coefficient of roll(d, s) on y_res given design basis Q, for EVERY s.
    Exact, via circular cross-correlation.  coef[0] is the observed value."""
    n = len(d)
    fd = np.conj(np.fft.rfft(d))
    num = np.fft.irfft(fd * np.fft.rfft(y_res), n)
    sq = np.zeros(n)
    for j in range(Q.shape[1]):
        a = np.fft.irfft(fd * np.fft.rfft(Q[:, j]), n)
        sq += a * a
    den = float(d @ d) - sq
    with np.errstate(invalid="ignore", divide="ignore"):
        coef = np.where(den > 1e-8, num / den, np.nan)
    return coef


def fe_test(y, X, mask, label, quiet=False):
    Q = basis(X)
    y_res = y - Q @ (Q.T @ y)
    d = np.asarray(mask, float)
    coef = rotation_all(y_res, d, Q)
    obs = coef[0]
    null = coef[1:]
    null = null[~np.isnan(null)]
    base = null.mean()
    p = float((np.abs(null - base) >= abs(obs - base)).mean())
    if not quiet:
        print(f"  {label:<62}{obs*PCT:>+9.3f}%  p={p:5.3f} {stars(p)}  (k={Q.shape[1]})")
    return obs, p


def main():
    pd.set_option("display.width", 200)
    d = _vix_data.add_features(_vix_data.load())

    # ---------------- competing-mechanism features
    # NOTE: the panel has 2 rows (2026-05-25, 2026-09-07 — US cash holidays) where ^VIX printed
    # but ^GSPC did not. Computing rolling windows on the raw series propagates those NaNs
    # through 200/252 bars and silently truncates the frame; compute SPX rollings on the
    # NaN-free SPX series and reindex back.
    sx = d.spx.dropna()
    rr = sx.pct_change(fill_method=None)

    def px(s):
        return s.reindex(d.index)

    d["rv20"] = px(rr.rolling(20).std() * np.sqrt(252))
    d["rv5"] = px(rr.rolling(5).std() * np.sqrt(252))
    d["spx_r5"] = px(sx / sx.shift(5) - 1.0)
    d["spx_r20"] = px(sx / sx.shift(20) - 1.0)
    d["vrp"] = d.vix / 100.0 - d.rv20
    # (1) SPX overbought, measured EXACTLY as the VIX band measures VIX
    for n in (10, 20):
        d[f"spx_z{n}"] = px((sx - sx.rolling(n).mean()) / sx.rolling(n).std(ddof=0))
    # (2) VIX just crashed
    d["vix_chg5"] = d.vix / d.vix.shift(5) - 1.0
    d["vix_chg10"] = d.vix / d.vix.shift(10) - 1.0
    # longer-horizon level / trend state
    d["vix_rel200"] = d.vix / d.vix.rolling(200).mean() - 1.0
    d["dd252"] = px(sx / sx.rolling(252).max() - 1.0)
    d["spx_rel200"] = px(sx / sx.rolling(200).mean() - 1.0)
    # the band statistic itself
    d["z10"] = (d.vix - d.vix.rolling(10).mean()) / d.vix.rolling(10).std(ddof=0)
    d["z20"] = (d.vix - d.vix.rolling(20).mean()) / d.vix.rolling(20).std(ddof=0)

    keep = ["g1", "g3", "g5", "g10", "g21", "vix", "vix_pct1y", "stretch", "bb10_width",
            "rv20", "rv5", "spx_r5", "spx_r20", "vrp", "spx_z10", "spx_z20",
            "vix_chg5", "vix_chg10", "vix_rel200", "dd252", "spx_rel200", "z10", "z20",
            "bb10_2.0_below", "bb10_2.0_above"]
    a = d[keep].copy()
    a = a[a[[c for c in keep if c not in ("g10", "g21")]].notna().all(axis=1)]
    y = a.g5.values
    below = a["bb10_2.0_below"].astype(bool).values
    n = len(a)
    base = y.mean()
    print("=" * 100)
    print(f"FRAME  {n:,} rows  {a.index[0].date()} -> {a.index[-1].date()}   "
          f"unconditional g5 = {base*PCT:+.3f}%   below-band n = {below.sum()}")
    print("=" * 100)
    raw_exc = y[below].mean() - base
    raw_coef = y[below].mean() - y[~below].mean()   # this is what an OLS dummy coef estimates
    print(f"raw below-band D5 excess vs unconditional = {raw_exc*PCT:+.3f}%   "
          f"| OLS-dummy version (vs complement) = {raw_coef*PCT:+.3f}%")
    print(f"   (claim quotes -0.58% raw on n=159; here n={below.sum()}.  Note the claim's "
          f"controlled numbers\n    are OLS dummy coefficients, so the like-for-like "
          f"denominator is {raw_coef*PCT:+.3f}%, not {raw_exc*PCT:+.3f}%.)")
    # sanity: rotation on an intercept-only design must reproduce the OLS dummy coefficient
    o0, p0 = fe_test(y, fe_matrix(n), below, "no control (reproduce raw dummy coef)")
    assert abs(o0 - raw_coef) < 1e-10, (o0, raw_coef)

    # sanity check FFT rotation against a brute-force FWL at a couple of offsets
    Xc = fe_matrix(n, [pd.qcut(a.vix_pct1y, 20, labels=False, duplicates="drop").values])
    Qc = basis(Xc)
    yr = y - Qc @ (Qc.T @ y)
    cf = rotation_all(yr, below.astype(float), Qc)
    for off in (1, 137, 4001):
        v = np.roll(below.astype(float), off)
        vr = v - Qc @ (Qc.T @ v)
        brute = float(vr @ yr) / float(vr @ vr)
        assert abs(brute - cf[off]) < 1e-9, (off, brute, cf[off])
    print("  [FFT rotation verified against brute-force FWL at 3 offsets]")

    QC = lambda s, k: pd.qcut(a[s], k, labels=False, duplicates="drop").values
    lvl20 = QC("vix_pct1y", 20)
    wid = QC("bb10_width", 10)
    rv = QC("rv20", 10)
    r20 = QC("spx_r20", 10)
    r5 = QC("spx_r5", 10)

    # =============================================================== A. replicate the claim
    print("\n" + "=" * 100)
    print("A. REPLICATE THE CLAIM'S OWN LADDER (should recover -0.487% and -0.406%)")
    print("=" * 100)
    fe_test(y, fe_matrix(n, [lvl20]), below, "vix_pct1y VIGINTILE FE  [claim: -0.487%, p=.020]")
    fe_test(y, fe_matrix(n, [lvl20, wid, rv, r20, r5]), below,
            "level+width+rv20+r20+r5 FE   [claim: -0.406%, p=.039]")

    # =============================================================== B. the omitted controls
    print("\n" + "=" * 100)
    print("B. CONTROLS THE CLAIM OMITTED — one at a time (decile FE unless noted)")
    print("=" * 100)
    singles = [
        ("spx_z10", 10, "SPX Bollinger z(10d)  <-- index overbought, vol-normalised"),
        ("spx_z20", 10, "SPX Bollinger z(20d)"),
        ("vix_chg5", 10, "VIX 5d % change      <-- 'VIX just crashed'"),
        ("vix_chg10", 10, "VIX 10d % change"),
        ("vix_rel200", 10, "VIX vs its own 200d MA (longer-horizon level)"),
        ("rv5", 10, "SPX realised vol 5d"),
        ("dd252", 10, "SPX drawdown from 252d high"),
        ("spx_rel200", 10, "SPX vs its 200d MA (trend state)"),
        ("vrp", 10, "VIX minus realised vol (VRP)"),
    ]
    for col, k, lab in singles:
        fe_test(y, fe_matrix(n, [QC(col, k)]), below, lab)
    yr_c = a.index.year.values
    dec_c = (a.index.year.values // 10) * 10
    mo_c = a.index.month.values
    dow_c = a.index.dayofweek.values
    fe_test(y, fe_matrix(n, [dec_c]), below, "DECADE fixed effects")
    fe_test(y, fe_matrix(n, [yr_c]), below, "YEAR fixed effects")
    fe_test(y, fe_matrix(n, [mo_c]), below, "MONTH-of-year fixed effects")
    fe_test(y, fe_matrix(n, [dow_c]), below, "day-of-week fixed effects")

    # =============================================================== C. joint kill test
    print("\n" + "=" * 100)
    print("C. JOINT — claim's ladder PLUS the omitted mechanism controls")
    print("=" * 100)
    z10c, z20c, vc5, vc10 = QC("spx_z10", 10), QC("spx_z20", 10), QC("vix_chg5", 10), QC("vix_chg10", 10)
    fe_test(y, fe_matrix(n, [lvl20, z10c]), below, "level + SPX-z10 FE")
    fe_test(y, fe_matrix(n, [lvl20, vc5]), below, "level + VIX-5d-change FE")
    fe_test(y, fe_matrix(n, [lvl20, z10c, vc5]), below, "level + SPX-z10 + VIX-5d-change FE")
    fe_test(y, fe_matrix(n, [lvl20, wid, rv, r20, r5, z10c]), below,
            "claim ladder + SPX-z10 FE")
    fe_test(y, fe_matrix(n, [lvl20, wid, rv, r20, r5, vc5]), below,
            "claim ladder + VIX-5d-change FE")
    fe_test(y, fe_matrix(n, [lvl20, wid, rv, r20, r5, z10c, vc5]), below,
            "claim ladder + SPX-z10 + VIX-5d-change FE")
    fe_test(y, fe_matrix(n, [lvl20, wid, rv, r20, r5, z10c, vc5, yr_c]), below,
            "EVERYTHING + YEAR FE")
    # finer versions (20 bins) of the two omitted controls
    fe_test(y, fe_matrix(n, [QC("spx_z10", 20)]), below, "SPX-z10 VIGINTILE FE alone")
    fe_test(y, fe_matrix(n, [QC("vix_chg5", 20)]), below, "VIX-5d-change VIGINTILE FE alone")
    fe_test(y, fe_matrix(n, [lvl20, QC("spx_z10", 20), QC("vix_chg5", 20)]), below,
            "level20 + SPX-z10(20) + VIX-chg5(20) FE")

    # =============================================================== D. rival simple signals
    print("\n" + "=" * 100)
    print("D. DO SIMPLER RIVALS DO THE SAME JOB? (raw D5 excess, no controls)")
    print("=" * 100)
    rivals = {
        "bb10_2.0_below (the claim)": below,
        "stretch < -6%": (a.stretch < -0.06).values,
        "stretch < -8%": (a.stretch < -0.08).values,
        "stretch < -10%": (a.stretch < -0.10).values,
        "VIX 5d change < -12%": (a.vix_chg5 < -0.12).values,
        "VIX 5d change < -15%": (a.vix_chg5 < -0.15).values,
        "VIX 5d change < -20%": (a.vix_chg5 < -0.20).values,
        "SPX z10 > +1.5 (index overbought)": (a.spx_z10 > 1.5).values,
        "SPX z10 > +2.0": (a.spx_z10 > 2.0).values,
        "SPX 5d return > +2%": (a.spx_r5 > 0.02).values,
        "SPX 5d return > +3%": (a.spx_r5 > 0.03).values,
    }
    for lab, m in rivals.items():
        if m.sum() < 25:
            print(f"  {lab:<40} n={m.sum():>5}   INCONCLUSIVE (n<25)")
            continue
        o, p = fe_test(y, fe_matrix(n), m, f"{lab}  (n={m.sum()})", quiet=True)
        print(f"  {lab:<40} n={m.sum():>5}   excess {o*PCT:+7.3f}%  p={p:5.3f} {stars(p)}")

    print("\n  overlap of below-band with each rival (share of the 159 below days also in rival):")
    for lab, m in rivals.items():
        if lab.startswith("bb10"):
            continue
        print(f"    {lab:<40} {np.mean(m[below])*PCT:5.1f}%   rival n={m.sum()}")

    # head-to-head: does below-band survive controlling for the rival DUMMY, and vice versa?
    print("\n  HEAD-TO-HEAD (both dummies in one regression, level-vigintile FE also included):")
    for lab in ("VIX 5d change < -15%", "SPX z10 > +1.5 (index overbought)", "stretch < -8%",
                "SPX 5d return > +2%"):
        rm = rivals[lab].astype(float)
        X = fe_matrix(n, [lvl20], extra=[rm])
        o, p = fe_test(y, X, below, f"below | level FE + [{lab}]")
        X2 = fe_matrix(n, [lvl20], extra=[below.astype(float)])
        o2, p2 = fe_test(y, X2, rivals[lab], f"    [{lab}] | level FE + below")

    # =============================================================== E. threshold or slope?
    print("\n" + "=" * 100)
    print("E. IS -2.0 SD A REAL THRESHOLD, OR JUST A SLICE OF A MONOTONE SLOPE IN z10?")
    print("=" * 100)
    print(f"  note: with a 10-day window and ddof=0, z10 is bounded by 9/sqrt(10) = "
          f"{9/np.sqrt(10):.3f}; observed range {a.z10.min():.3f} .. {a.z10.max():.3f}")
    # E1: smooth polynomial in z10 as the control
    for deg in (1, 2, 3):
        ex = [a.z10.values ** k for k in range(1, deg + 1)]
        fe_test(y, fe_matrix(n, [lvl20], extra=ex), below,
                f"level FE + degree-{deg} polynomial in z10")
    # E2: regression-discontinuity — restrict to a neighbourhood of the -2 cut
    for lo in (-2.846, -3.0):
        for hi in (-1.0, -1.25, -1.5):
            w = (a.z10.values >= lo) & (a.z10.values <= hi)
            if w.sum() < 60:
                continue
            yw = y[w]
            zw = a.z10.values[w]
            bw = below[w]
            Xw = np.column_stack([np.ones(w.sum()), zw])
            ow, pw = fe_test(yw, Xw, bw, f"RD window z10 in [{lo:.2f},{hi:.2f}]  "
                                          f"n={w.sum()} below={bw.sum()}")
    # E3: placebo cut points on the same statistic
    print("\n  placebo cut points on z10 (raw excess, no control) — is -2.0 special?")
    for cut in (-1.25, -1.5, -1.75, -2.0, -2.25, -2.5):
        m = (a.z10.values < cut)
        if m.sum() < 25:
            print(f"    z10 < {cut:+.2f}   n={m.sum():>5}   INCONCLUSIVE (n<25)")
            continue
        o, p = fe_test(y, fe_matrix(n), m, "", quiet=True)
        ol, pl = fe_test(y, fe_matrix(n, [lvl20]), m, "", quiet=True)
        print(f"    z10 < {cut:+.2f}   n={m.sum():>5}   raw {o*PCT:+7.3f}% p={p:5.3f}   "
              f"| levelFE {ol*PCT:+7.3f}% p={pl:5.3f} {stars(pl)}")

    # =============================================================== F. mid-tercile claim
    print("\n" + "=" * 100)
    print("F. THE 'STRONGEST IN THE MID LEVEL TERCILE' SUB-CLAIM — formal heterogeneity test")
    print("=" * 100)
    terc = pd.qcut(a.vix_pct1y, 3, labels=False, duplicates="drop").values
    names = {0: "BOTTOM", 1: "MID", 2: "TOP"}

    def within(mask):
        out = []
        for k in (0, 1, 2):
            s = mask & (terc == k)
            out.append(y[s].mean() - y[terc == k].mean() if s.sum() else np.nan)
        return np.array(out)

    obs_w = within(below)
    N_R = 20000
    offs = RNG.integers(1, n, size=N_R)
    null_w = np.empty((N_R, 3))
    for i, off in enumerate(offs):
        null_w[i] = within(np.roll(below, off))
    for k in (0, 1, 2):
        nk = int((below & (terc == k)).sum())
        col = null_w[:, k]
        col = col[~np.isnan(col)]
        b = col.mean()
        p = float((np.abs(col - b) >= abs(obs_w[k] - b)).mean())
        flag = "  INCONCLUSIVE (n<25)" if nk < 25 else ""
        print(f"  {names[k]:<8} n={nk:>4}   within-tercile excess {obs_w[k]*PCT:+7.3f}%  "
              f"p={p:5.3f} {stars(p)}{flag}")
    for i, j in ((1, 0), (1, 2)):
        dobs = obs_w[i] - obs_w[j]
        dn = null_w[:, i] - null_w[:, j]
        dn = dn[~np.isnan(dn)]
        b = dn.mean()
        p = float((np.abs(dn - b) >= abs(dobs - b)).mean())
        print(f"  DIFFERENCE {names[i]} minus {names[j]}: {dobs*PCT:+7.3f}%  p={p:5.3f} {stars(p)}"
              f"   <- this is what 'strongest in MID' requires")
    # same thing with quintiles, to show how arbitrary the tercile cut is
    print("\n  same heterogeneity with QUINTILES of vix_pct1y (fragility of the slicing):")
    q5 = pd.qcut(a.vix_pct1y, 5, labels=False, duplicates="drop").values
    for k in range(5):
        s = below & (q5 == k)
        nk = int(s.sum())
        e = (y[s].mean() - y[q5 == k].mean()) if nk else np.nan
        tag = " INCONCLUSIVE (n<25)" if nk < 25 else ""
        print(f"    Q{k+1} n={nk:>4}  within excess {e*PCT:+7.3f}%{tag}")

    # =============================================================== G. horizon profile
    print("\n" + "=" * 100)
    print("G. HORIZON PROFILE under controls (is D5 the only one that works?)")
    print("=" * 100)
    for h in (1, 3, 5, 10, 21):
        col = a[f"g{h}"].values
        ok = ~np.isnan(col)
        yh = col[ok]
        nn = ok.sum()
        bh = below[ok]
        lv = pd.qcut(a.vix_pct1y.values[ok], 20, labels=False, duplicates="drop")
        wd = pd.qcut(a.bb10_width.values[ok], 10, labels=False, duplicates="drop")
        rvh = pd.qcut(a.rv20.values[ok], 10, labels=False, duplicates="drop")
        r20h = pd.qcut(a.spx_r20.values[ok], 10, labels=False, duplicates="drop")
        r5h = pd.qcut(a.spx_r5.values[ok], 10, labels=False, duplicates="drop")
        zh = pd.qcut(a.spx_z10.values[ok], 10, labels=False, duplicates="drop")
        vh = pd.qcut(a.vix_chg5.values[ok], 10, labels=False, duplicates="drop")
        raw = yh[bh].mean() - yh.mean()
        o1, p1 = fe_test(yh, fe_matrix(nn, [lv]), bh, "", quiet=True)
        o2, p2 = fe_test(yh, fe_matrix(nn, [lv, wd, rvh, r20h, r5h]), bh, "", quiet=True)
        o3, p3 = fe_test(yh, fe_matrix(nn, [lv, wd, rvh, r20h, r5h, zh, vh]), bh, "", quiet=True)
        print(f"  D{h:<3} n={nn:>5} below={bh.sum():>4}  raw {raw*PCT:+7.3f}%  | levelFE "
              f"{o1*PCT:+7.3f}% p={p1:5.3f}{stars(p1)} | claim-ladder {o2*PCT:+7.3f}% "
              f"p={p2:5.3f}{stars(p2)} | +mechanism {o3*PCT:+7.3f}% p={p3:5.3f}{stars(p3)}")

    # =============================================================== H. temporal stability
    print("\n" + "=" * 100)
    print("H. LEAVE-ONE-YEAR-OUT / DECADE of the FULL-MECHANISM coefficient")
    print("=" * 100)
    Xfull = fe_matrix(n, [lvl20, wid, rv, r20, r5, z10c, vc5])
    Qf = basis(Xfull)
    yrf = y - Qf @ (Qf.T @ y)

    def fwl(mv):
        v = np.asarray(mv, float)
        vr = v - Qf @ (Qf.T @ v)
        den = float(vr @ vr)
        return float(vr @ yrf) / den if den > 1e-10 else np.nan

    full = fwl(below)
    print(f"  full-sample (level+width+rv+mom+SPXz10+VIXchg5) coef {full*PCT:+.3f}%")
    jk = {}
    for yy in sorted(set(a.index.year[below])):
        m = below.copy()
        m[a.index.year.values == yy] = False
        jk[yy] = fwl(m)
    arr = np.array(list(jk.values()))
    worst = max(jk, key=lambda k: jk[k])
    print(f"  leave-one-YEAR-out: range {arr.min()*PCT:+.3f}% .. {arr.max()*PCT:+.3f}%  "
          f"median {np.median(arr)*PCT:+.3f}%  sign flips {(arr>0).sum()}/{len(arr)}")
    print(f"  most influential year {worst} (below days that year = "
          f"{int((below & (a.index.year.values==worst)).sum())}) -> coef {jk[worst]*PCT:+.3f}%")
    for dd0 in (1990, 2000, 2010, 2020):
        m = below.copy()
        sel = (a.index.year.values >= dd0) & (a.index.year.values < dd0 + 10)
        m[sel] = False
        print(f"    drop the {dd0}s ({int((below & sel).sum())} below days) -> "
              f"coef {fwl(m)*PCT:+.3f}%")
    # in-decade coefficients (with era-specific baselines already absorbed by year FE)
    print("\n  decade-by-decade below-band raw excess vs that decade's own baseline:")
    for dd0 in (1990, 2000, 2010, 2020):
        sel = (a.index.year.values >= dd0) & (a.index.year.values < dd0 + 10)
        m = below & sel
        nk = int(m.sum())
        e = y[m].mean() - y[sel].mean() if nk else np.nan
        tag = "  INCONCLUSIVE (n<25)" if nk < 25 else ""
        print(f"    {dd0}s  n={nk:>4}  excess {e*PCT:+7.3f}%   "
              f"(decade baseline {y[sel].mean()*PCT:+.3f}%){tag}")

    # =============================================================== I. multiplicity
    print("\n" + "=" * 100)
    print("I. BAND-PARAMETER GRID under the level+mechanism ladder (how special is bb10/2.0?)")
    print("=" * 100)
    Xlad = fe_matrix(n, [lvl20, wid, rv, r20, r5, z10c, vc5])
    for nn_ in (10, 20):
        for k in (1.5, 2.0, 2.5):
            col = f"bb{nn_}_{k}_below"
            m = d[col].reindex(a.index).astype(bool).values
            if m.sum() < 25:
                print(f"  {col:<20} n={m.sum():>5}  INCONCLUSIVE (n<25)")
                continue
            raw = y[m].mean() - base
            o, p = fe_test(y, Xlad, m, "", quiet=True)
            ol, pl = fe_test(y, fe_matrix(n, [lvl20]), m, "", quiet=True)
            print(f"  {col:<20} n={m.sum():>5}  raw {raw*PCT:+7.3f}%  | levelFE {ol*PCT:+7.3f}% "
                  f"p={pl:5.3f}{stars(pl)}  | +mechanism {o*PCT:+7.3f}% p={p:5.3f}{stars(p)}")

    # =============================================================== J. what actually happens
    print("\n" + "=" * 100)
    print("J. DESCRIPTIVE MECHANISM — what is a below-band day?")
    print("=" * 100)
    cmp_cols = ["vix", "vix_pct1y", "stretch", "z10", "bb10_width", "spx_z10", "spx_r5",
                "spx_r20", "vix_chg5", "rv20", "vrp", "dd252"]
    tbl = pd.DataFrame({"below-band": a.loc[below, cmp_cols].mean(),
                        "all days": a[cmp_cols].mean()})
    tbl["ratio/diff"] = tbl["below-band"] - tbl["all days"]
    print(tbl.round(4).to_string())
    fv = (d.vix.shift(-6) / d.vix.shift(-1) - 1.0).reindex(a.index).values
    okv = ~np.isnan(fv)
    print(f"\n  forward 5d VIX change (next close -> +5): below-band "
          f"{np.nanmean(fv[below & okv])*PCT:+.2f}%  vs all {np.nanmean(fv[okv])*PCT:+.2f}%  "
          f"(n={int((below & okv).sum())})")


if __name__ == "__main__":
    main()
