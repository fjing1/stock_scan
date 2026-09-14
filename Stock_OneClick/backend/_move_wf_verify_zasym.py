"""
_move_wf_verify_zasym.py — ADVERSARIAL VERIFICATION of the "tail-shape asymmetry" claim.

CLAIM UNDER TEST
  "Index z is genuinely left-asymmetric beyond drift and the asymmetry grows with horizon, but
   single-name z is symmetric beyond drift -- so the two asset classes need different shape tables,
   and P(down 2%) != P(up 2%) for single names is driven purely by drift, not by skew."

ATTACKS IMPLEMENTED (run with:  python _move_wf_verify_zasym.py <sections...>)
  0  reproduce the claimed numbers exactly (median-centred ratio, Bowley skew, raw tail COUNTS)
  1  SIMPLER EXPLANATION #1: the numerator is a SIMPLE return, the denominator a LOG-return sigma.
     r_simple = exp(r_log)-1 is a CONVEX map, so it injects POSITIVE skew proportional to sigma.
     Singles have ~2-3x the index sigma, so their genuine left skew may simply be cancelled.
     Re-run everything on z_log = log(1+r_h)/sigma_hat_h.
  2  FAKE SAMPLE SIZE: block size 63 -> 126/252/504, plus NON-OVERLAPPING subsampling (every h-th
     date), plus explicit count of independent tail EPISODES for the index.
  3  SIMPLER EXPLANATION #2: is "asset class" the right variable, or just VOLATILITY LEVEL?
     Per-symbol Bowley skew regressed on log10(median sigma); do the index series lie on the
     singles' regression line? Vol-matched cohorts.
  4  FRAGILITY: drop 2008-2009 and 2020; per-index-symbol breakdown; SPY alone.
  5  SIMPSON: asymmetry within vol quintiles and within calendar years (both groups).
  6  LOOKAHEAD: the centring median is a FULL-SAMPLE median. Redo with an expanding-window median.
  7  OOS at ALL FOUR HORIZONS: does symmetrising the index table hurt at h=10/21 where the
     asymmetry is claimed to be LARGEST? Plus a log-space-symmetric model for singles.
  8  OOS CALIBRATION by vol quintile and by year for the shipped singles table vs symmetrised
     vs log-symmetric.
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

import _move_lib as L
from _move_wf_zshape2 import (HORIZONS, THR, EmpCDF, actual_idx, build, norm_cdf_vec,
                              probs_from_cdf, qq_fit_t, robust_shape, t_unit_cdf_vec)

pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 120)
pd.set_option("display.float_format", lambda v: f"{v:10.4f}")

KA = [1.0, 1.5, 2.0, 2.5]
NREP = 2000
RNG_SEED = 7


def iqs(a):
    return (np.percentile(a, 75) - np.percentile(a, 25)) / 1.34898


def bowley(a, lo=10, hi=90):
    ql, qm, qh = np.percentile(a, [lo, 50, hi])
    return (qh + ql - 2 * qm) / (qh - ql)


# ------------------------------------------------------------------ shared state
_CACHE = {}


def data():
    if "d" not in _CACHE:
        close, idx, singles, Z, SH, FWD, sig, rank = build()
        # log-return version of z: numerator is the LOG forward return, SAME denominator sigma_h
        ZL = {}
        for h in HORIZONS:
            lg = np.log(close.shift(-h) / close)
            ZL[h] = (lg / SH[h]).where(Z[h].notna())
        _CACHE["d"] = dict(close=close, idx=idx, singles=singles, Z=Z, ZL=ZL, SH=SH, FWD=FWD,
                           sig=sig, rank=rank)
    return _CACHE["d"]


def long_form(df_z, cols, extra=None):
    """Long-form (date_ord, symbol_id, value) for one horizon/group, dropping NaNs."""
    sub = df_z[cols]
    M = sub.notna().to_numpy()
    V = sub.to_numpy(dtype=float)
    fin = M & np.isfinite(V)
    di, si = np.nonzero(fin)
    out = {"z": V[fin], "date_ord": di, "sym": si}
    if extra:
        for name, df in extra.items():
            out[name] = df[cols].to_numpy(dtype=float)[fin]
    out["n_dates_total"] = len(sub)
    return out


# ------------------------------------------------------------------ block bootstrap on a RATIO
def boot_ratio(date_ord, n_dates_total, lo_flag, hi_flag, block, nrep=NREP, seed=RNG_SEED):
    """Moving-block bootstrap over DATES of ratio = sum(lo)/sum(hi).
    Clusters by date (kills cross-sectional correlation); block >= h absorbs window overlap."""
    lo = np.bincount(date_ord, weights=lo_flag.astype(float), minlength=n_dates_total)
    hi = np.bincount(date_ord, weights=hi_flag.astype(float), minlength=n_dates_total)
    nn = np.bincount(date_ord, minlength=n_dates_total).astype(float)
    keep = nn > 0
    A = np.column_stack([lo[keep], hi[keep]])
    nd = A.shape[0]
    if nd <= block:
        return dict(nd=nd, point=np.nan, lo_n=int(A[:, 0].sum()), hi_n=int(A[:, 1].sum()))
    point = A[:, 0].sum() / max(A[:, 1].sum(), 1e-12)
    nblk = int(math.ceil(nd / block))
    st = np.arange(0, nd - block + 1)
    cs = np.vstack([np.zeros(2), np.cumsum(A, axis=0)])
    blk = cs[st + block] - cs[st]
    rng = np.random.default_rng(seed)
    tot = blk[rng.integers(0, len(st), size=(nrep, nblk))].sum(axis=1)
    rb = tot[:, 0] / np.maximum(tot[:, 1], 1e-12)
    l, u = np.percentile(rb, [2.5, 97.5])
    return dict(nd=nd, point=point, ci_lo=l, ci_hi=u, p_gt1=float((rb > 1).mean()),
                boot_med=float(np.median(rb)), lo_n=int(A[:, 0].sum()), hi_n=int(A[:, 1].sum()))


def ratio_table(zdict, groups, ks=KA, block=63, subsample_h=False, label_extra="",
                year_filter=None, horizons=HORIZONS):
    rows = []
    d = data()
    for label, cols in groups:
        for h in horizons:
            lf = long_form(zdict[h], cols)
            z, do = lf["z"], lf["date_ord"]
            years = d["close"].index.year.to_numpy()
            if year_filter is not None:
                keep = year_filter(years[do])
                z, do = z[keep], do[keep]
            if subsample_h and h > 1:
                keep = (do % h) == 0
                z, do = z[keep], do[keep]
            if z.size < 200:
                continue
            med = np.median(z)
            c = z - med
            for k in ks:
                r = boot_ratio(do, lf["n_dates_total"], c < -k, c > k, block)
                rows.append(dict(group=label, h=h, k=k, med=med, n_obs=z.size, **r))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ci"] = df.apply(lambda r: f"[{r.get('ci_lo', np.nan):.2f},{r.get('ci_hi', np.nan):.2f}]"
                        if pd.notna(r.get("ci_lo", np.nan)) else "--", axis=1)
    print(df[["group", "h", "k", "n_obs", "nd", "lo_n", "hi_n", "point", "ci", "p_gt1",
              "boot_med"]].to_string(index=False))
    return df


# ================================================================== SECTION 0
def sec0():
    d = data()
    groups = [("INDEX5", d["idx"]), ("SPY", ["SPY"]), ("SINGLES", d["singles"])]
    print("=" * 150)
    print("S0. REPRODUCTION of the claim's median-centred ratio P(z<-k)/P(z>+k), block=63, "
          f"{NREP} reps. SIMPLE-return z, exactly as claimed.")
    print("    lo_n / hi_n are the RAW TAIL COUNTS behind each ratio -- read them before "
          "believing any CI.")
    ratio_table(d["Z"], groups, block=63)
    print("\nS0b. Bowley skew (q10,q50,q90) of median-centred z, IN-SAMPLE (centring is "
          "irrelevant for Bowley).")
    rows = []
    for label, cols in groups:
        for h in HORIZONS:
            a = d["Z"][h][cols].to_numpy(dtype=float).ravel()
            a = a[np.isfinite(a)]
            rows.append(dict(group=label, h=h, n=a.size, bowley_10_90=bowley(a),
                             bowley_5_95=bowley(a, 5, 95), bowley_1_99=bowley(a, 1, 99),
                             median=np.median(a), iqr_scale=iqs(a)))
    print(pd.DataFrame(rows).to_string(index=False))


# ================================================================== SECTION 1  (simple vs log)
def sec1():
    d = data()
    groups = [("INDEX5", d["idx"]), ("SPY", ["SPY"]), ("SINGLES", d["singles"])]
    print("=" * 150)
    print("S1. ATTACK: the numerator is a SIMPLE return. exp() is convex, so simple returns carry")
    print("    MECHANICAL positive skew that grows with sigma. Singles' sigma >> index sigma.")
    print("    Same statistic on z_log = log(1+r_h)/sigma_hat_h:")
    ratio_table(d["ZL"], groups, block=63)
    print("\nS1b. Bowley skew: SIMPLE-return z vs LOG-return z, side by side.")
    rows = []
    for label, cols in groups:
        for h in HORIZONS:
            a = d["Z"][h][cols].to_numpy(dtype=float).ravel()
            a = a[np.isfinite(a)]
            b = d["ZL"][h][cols].to_numpy(dtype=float).ravel()
            b = b[np.isfinite(b)]
            rows.append(dict(group=label, h=h, n=a.size,
                             bowley_SIMPLE=bowley(a), bowley_LOG=bowley(b),
                             delta=bowley(a) - bowley(b),
                             bw5_SIMPLE=bowley(a, 5, 95), bw5_LOG=bowley(b, 5, 95),
                             med_sigma_h=np.nanmedian(d["SH"][h][cols].to_numpy(dtype=float))))
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nS1c. HOW BIG IS THE CONVEXITY ARTIFACT? Take each group's own LOG-z, SYMMETRISE it,")
    print("     map back through r=exp(z*sigma)-1 and re-measure the SIMPLE-space Bowley skew.")
    print("     If the observed SIMPLE-space skew ~ this synthetic value, the 'symmetry' is an")
    print("     artifact and the underlying shape is left-skewed.")
    rng = np.random.default_rng(11)
    rows = []
    for label, cols in groups:
        for h in HORIZONS:
            zl = d["ZL"][h][cols].to_numpy(dtype=float)
            sh = d["SH"][h][cols].to_numpy(dtype=float)
            m = np.isfinite(zl) & np.isfinite(sh)
            zl, sh = zl[m], sh[m]
            med = np.median(zl)
            u = zl - med                       # centred log shape
            sgn = rng.choice([-1.0, 1.0], size=u.size)
            u_sym = sgn * u                    # exactly symmetric by construction
            r_syn = np.expm1((u_sym + med) * sh)
            z_syn = r_syn / sh
            rows.append(dict(group=label, h=h, n=zl.size,
                             observed_simple_bowley=bowley(d["Z"][h][cols].to_numpy(float)[
                                 np.isfinite(d["Z"][h][cols].to_numpy(float))]),
                             synth_from_SYMMETRIC_log=bowley(z_syn),
                             observed_log_bowley=bowley(zl)))
    print(pd.DataFrame(rows).to_string(index=False))


# ================================================================== SECTION 2  (sample size)
def sec2():
    d = data()
    groups = [("INDEX5", d["idx"]), ("SINGLES", d["singles"])]
    print("=" * 150)
    print("S2. ATTACK: fake precision. Same statistic at block = 63 / 126 / 252 / 504 trading days.")
    for blk in (63, 126, 252, 504):
        print(f"\n--- block = {blk} trading days ---")
        ratio_table(d["Z"], groups, ks=[2.0], block=blk)
    print("\nS2b. NON-OVERLAPPING windows only (every h-th trading date), block=252.")
    ratio_table(d["Z"], groups, ks=[1.0, 2.0], block=252, subsample_h=True)
    print("\nS2c. How many INDEPENDENT EPISODES actually produce the index's left tail?")
    print("     (dates with >=1 obs in the tail, grouped into runs separated by >21 trading days)")
    for label, cols in [("INDEX5", d["idx"]), ("SINGLES", d["singles"])]:
        for h in HORIZONS:
            lf = long_form(d["Z"][h], cols)
            z, do = lf["z"], lf["date_ord"]
            med = np.median(z)
            c = z - med
            for k in (2.0,):
                for side, flag in (("LEFT", c < -k), ("RIGHT", c > k)):
                    dts = np.unique(do[flag])
                    if dts.size == 0:
                        print(f"  {label} h={h} k={k} {side}: 0 obs")
                        continue
                    gaps = np.diff(dts)
                    episodes = 1 + int((gaps > 21).sum())
                    yrs = d["close"].index.year.to_numpy()[dts]
                    top = pd.Series(yrs).value_counts().head(4).to_dict()
                    print(f"  {label} h={h:2d} k={k} {side:5s}: n_obs={int(flag.sum()):7d} "
                          f"n_dates={dts.size:5d} episodes={episodes:4d} top_years={top}")


# ================================================================== SECTION 3  (vol level)
def sec3():
    d = data()
    print("=" * 150)
    print("S3. ATTACK: is 'asset class' the real variable, or just VOLATILITY LEVEL?")
    print("    Per-symbol Bowley skew of median-centred z vs that symbol's median sigma.")
    rows = []
    allcols = d["idx"] + d["singles"]
    for h in HORIZONS:
        for s in allcols:
            a = d["Z"][h][s].to_numpy(dtype=float)
            a = a[np.isfinite(a)]
            if a.size < 500:
                continue
            sg = d["sig"][s].to_numpy(dtype=float)
            sg = np.nanmedian(sg[np.isfinite(sg)])
            rows.append(dict(h=h, sym=s, kind="INDEX" if s in d["idx"] else "SINGLE",
                             n=a.size, med_sigma=sg,
                             bowley=bowley(a), bowley_log=bowley(
                                 d["ZL"][h][s].to_numpy(dtype=float)[
                                     np.isfinite(d["ZL"][h][s].to_numpy(dtype=float))])))
    per = pd.DataFrame(rows)
    for h in HORIZONS:
        sub = per[per.h == h]
        sing = sub[sub.kind == "SINGLE"]
        x = np.log10(sing.med_sigma.to_numpy())
        y = sing.bowley.to_numpy()
        A = np.vstack([np.ones_like(x), x]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        se = math.sqrt((resid ** 2).sum() / (len(x) - 2) /
                       ((x - x.mean()) ** 2).sum())
        print(f"\n[h={h}] SINGLES: bowley = {coef[0]:+.4f} {coef[1]:+.4f}*log10(sigma)   "
              f"slope t = {coef[1] / se:+.2f}   n_sym={len(x)}   resid_sd={resid.std(ddof=1):.4f}")
        ix = sub[sub.kind == "INDEX"].copy()
        ix["pred_from_singles_line"] = coef[0] + coef[1] * np.log10(ix.med_sigma)
        ix["resid_in_sd"] = (ix.bowley - ix.pred_from_singles_line) / resid.std(ddof=1)
        print(ix[["sym", "n", "med_sigma", "bowley", "pred_from_singles_line",
                  "resid_in_sd"]].to_string(index=False))
        # same regression in LOG space (convexity removed)
        yl = sing.bowley_log.to_numpy()
        coefl, *_ = np.linalg.lstsq(A, yl, rcond=None)
        residl = yl - A @ coefl
        sel = math.sqrt((residl ** 2).sum() / (len(x) - 2) / ((x - x.mean()) ** 2).sum())
        print(f"[h={h}] SINGLES in LOG space: bowley_log = {coefl[0]:+.4f} "
              f"{coefl[1]:+.4f}*log10(sigma)  slope t = {coefl[1]/sel:+.2f}  "
              f"mean bowley_log = {yl.mean():+.4f} (median {np.median(yl):+.4f}, "
              f"frac<0 = {(yl < 0).mean():.3f})")
    print("\nS3b. VOL-MATCHED COHORTS: singles split into sigma quintiles (by symbol), "
          "block=252, k=2.")
    med_sig = {s: float(np.nanmedian(d["sig"][s].to_numpy(dtype=float)))
               for s in d["singles"]}
    ss = pd.Series(med_sig).dropna().sort_values()
    q = pd.qcut(ss, 5, labels=False)
    groups = [(f"SINGLES_sigQ{i+1}(med_sig={ss[q == i].median():.4f})",
               list(ss.index[q == i])) for i in range(5)]
    groups = [("INDEX5", d["idx"])] + groups
    ratio_table(d["Z"], groups, ks=[2.0], block=252, horizons=(1, 21))
    print("\nS3c. Same vol-matched cohorts on LOG-z (convexity removed).")
    ratio_table(d["ZL"], groups, ks=[2.0], block=252, horizons=(1, 21))


# ================================================================== SECTION 4  (fragility)
def sec4():
    d = data()
    groups = [("INDEX5", d["idx"]), ("SPY", ["SPY"]), ("SINGLES", d["singles"])]
    print("=" * 150)
    print("S4. FRAGILITY: drop 2008+2009+2020 entirely (crisis years), block=252, k=1 and 2.")
    ratio_table(d["Z"], groups, ks=[1.0, 2.0], block=252,
                year_filter=lambda y: ~np.isin(y, [2008, 2009, 2020]))
    print("\nS4b. FULL sample for comparison, block=252.")
    ratio_table(d["Z"], groups, ks=[1.0, 2.0], block=252)
    print("\nS4c. PER-INDEX-SYMBOL (the 'INDEX5' pool is 5 ~0.95-correlated series, not 5 "
          "independent assets), block=252, k=2.")
    ratio_table(d["Z"], [(s, [s]) for s in d["idx"]], ks=[2.0], block=252)
    print("\nS4d. Bowley skew with crisis years dropped.")
    yrs = d["close"].index.year.to_numpy()
    rows = []
    for label, cols in groups:
        for h in HORIZONS:
            sub = d["Z"][h][cols]
            keep = ~np.isin(yrs, [2008, 2009, 2020])
            a = sub[keep].to_numpy(dtype=float).ravel()
            a = a[np.isfinite(a)]
            b = sub.to_numpy(dtype=float).ravel()
            b = b[np.isfinite(b)]
            rows.append(dict(group=label, h=h, n_full=b.size, n_nocrisis=a.size,
                             bowley_full=bowley(b), bowley_nocrisis=bowley(a),
                             bw1_99_full=bowley(b, 1, 99), bw1_99_nocrisis=bowley(a, 1, 99)))
    print(pd.DataFrame(rows).to_string(index=False))


# ================================================================== SECTION 5  (Simpson)
def sec5():
    d = data()
    print("=" * 150)
    print("S5. SIMPSON: median-centred asymmetry WITHIN vol quintiles (own-quintile median), "
          "block=252, k=2.")
    for label, cols in (("INDEX5", d["idx"]), ("SINGLES", d["singles"])):
        rows = []
        for h in HORIZONS:
            lf = long_form(d["Z"][h], cols, extra={"rk": d["rank"], "zl": d["ZL"][h]})
            z, do, rk, zl = lf["z"], lf["date_ord"], lf["rk"], lf["zl"]
            g = np.isfinite(rk)
            z, do, rk, zl = z[g], do[g], rk[g], zl[g]
            qb = np.digitize(rk, [0.2, 0.4, 0.6, 0.8])
            for q in range(5):
                m = qb == q
                if m.sum() < 500:
                    continue
                zz = z[m] - np.median(z[m])
                r = boot_ratio(do[m], lf["n_dates_total"], zz < -2, zz > 2, 252)
                zzl = zl[m] - np.median(zl[m])
                rl = boot_ratio(do[m], lf["n_dates_total"], zzl < -2, zzl > 2, 252)
                rows.append(dict(h=h, volq=q + 1, n=int(m.sum()),
                                 ratio_simple=r["point"],
                                 ci=f"[{r.get('ci_lo', np.nan):.2f},{r.get('ci_hi', np.nan):.2f}]",
                                 p_gt1=r.get("p_gt1", np.nan),
                                 bowley_simple=bowley(z[m]),
                                 ratio_log=rl["point"], p_gt1_log=rl.get("p_gt1", np.nan),
                                 bowley_log=bowley(zl[m])))
        print(f"\n[{label}]")
        print(pd.DataFrame(rows).to_string(index=False))
    print("\nS5b. SIMPSON: SINGLES median-centred ratio (k=2) BY CALENDAR YEAR, h=1 "
          "(centred on each year's own median).")
    yrs = d["close"].index.year.to_numpy()
    for label, cols in (("SINGLES", d["singles"]), ("INDEX5", d["idx"])):
        rows = []
        lf = long_form(d["Z"][1], cols, extra={"zl": d["ZL"][1]})
        z, do, zl = lf["z"], lf["date_ord"], lf["zl"]
        yy = yrs[do]
        for y in sorted(set(yy.tolist())):
            m = yy == y
            if m.sum() < 300:
                continue
            zz = z[m] - np.median(z[m])
            zzl = zl[m] - np.median(zl[m])
            lo, hi = int((zz < -2).sum()), int((zz > 2).sum())
            lol, hil = int((zzl < -2).sum()), int((zzl > 2).sum())
            rows.append(dict(year=y, n=int(m.sum()), lo=lo, hi=hi,
                             ratio_simple=lo / max(hi, 1e-9), bowley_simple=bowley(z[m]),
                             ratio_log=lol / max(hil, 1e-9), bowley_log=bowley(zl[m])))
        t = pd.DataFrame(rows)
        print(f"\n[{label}] h=1 by year   frac_years_ratio>1 (simple) = "
              f"{(t.ratio_simple > 1).mean():.3f}   (log) = {(t.ratio_log > 1).mean():.3f}")
        print(t.to_string(index=False))


# ================================================================== SECTION 6  (lookahead)
def sec6():
    d = data()
    print("=" * 150)
    print("S6. LOOKAHEAD: the claim centres on the FULL-SAMPLE median. Redo with an "
          "EXPANDING-WINDOW median")
    print("    (each year's z centred on the median of years strictly before it), block=252.")
    yrs = d["close"].index.year.to_numpy()
    for label, cols in (("INDEX5", d["idx"]), ("SPY", ["SPY"]), ("SINGLES", d["singles"])):
        rows = []
        for h in HORIZONS:
            lf = long_form(d["Z"][h], cols)
            z, do = lf["z"], lf["date_ord"]
            yy = yrs[do]
            years = sorted(set(yy.tolist()))
            c = np.full(z.shape, np.nan)
            for y in years:
                prev = yy < y
                if prev.sum() < 1000:
                    continue
                c[yy == y] = z[yy == y] - np.median(z[prev])
            ok = np.isfinite(c)
            for k in (1.0, 2.0):
                r = boot_ratio(do[ok], lf["n_dates_total"], c[ok] < -k, c[ok] > k, 252)
                rows.append(dict(h=h, k=k, n=int(ok.sum()), point=r["point"],
                                 ci=f"[{r.get('ci_lo', np.nan):.2f},"
                                    f"{r.get('ci_hi', np.nan):.2f}]",
                                 p_gt1=r.get("p_gt1", np.nan),
                                 lo_n=r["lo_n"], hi_n=r["hi_n"]))
        print(f"\n[{label}] expanding-window centring")
        print(pd.DataFrame(rows).to_string(index=False))


# ================================================================== SECTION 7  (OOS all h)
def _stack_full(h, cols, d):
    """Long-form z, sigma_h, fwd, vol-rank, year, plus log-z, for one horizon/group."""
    z = d["Z"][h][cols]
    M = z.notna().to_numpy() & np.isfinite(z.to_numpy(dtype=float))
    out = {}
    for name, df in (("z", d["Z"][h][cols]), ("zl", d["ZL"][h][cols]), ("sh", d["SH"][h][cols]),
                     ("fwd", d["FWD"][h][cols]), ("rk", d["rank"][cols])):
        out[name] = df.to_numpy(dtype=float)[M]
    yr = np.repeat(z.index.year.to_numpy()[:, None], len(cols), axis=1)
    out["year"] = yr[M]
    g = np.all([np.isfinite(out[k]) for k in ("z", "zl", "sh", "fwd", "rk")], axis=0)
    return {k: v[g] for k, v in out.items()}


def probs_from_log_cdf(cdf_fn, sh, thr=THR):
    """Same bucket probabilities but the shape lives in LOG-return z space, so the +/-thr
    thresholds convert EXACTLY: r<=-thr <=> z_log <= log(1-thr)/sh."""
    k_dn, k_up = math.log(1 - thr) / sh, math.log(1 + thr) / sh
    F_dn, F_0, F_up = cdf_fn(k_dn), cdf_fn(np.zeros_like(sh)), cdf_fn(k_up)
    P = np.column_stack([F_dn, F_0 - F_dn, F_up - F_0, 1.0 - F_up])
    P = np.clip(P, 1e-6, None)
    return P / P.sum(axis=1, keepdims=True)


def sec7(save="/tmp/verify_zasym_oos.pkl"):
    d = data()
    print("=" * 150)
    print("S7. OOS at ALL FOUR HORIZONS. 4-bucket log loss, thr=2%, every table fit on years < Y.")
    print("    Models: emp_table (own asym shape) | emp_sym (own shape mirrored about its median)")
    print("            | log_table (asym shape in LOG-z space) | log_sym (SYMMETRIC log shape,")
    print("              i.e. 'all the apparent simple-space skew is convexity') | other_shape")
    rows = []
    groups = [("INDEX5", d["idx"]), ("SINGLES", d["singles"]), ("SPY", ["SPY"])]
    S = {}
    for label, cols in groups:
        for h in HORIZONS:
            S[(label, h)] = _stack_full(h, cols, d)
            S[(label, h)]["ai"] = actual_idx(S[(label, h)]["fwd"])
    other_cache: dict = {}
    for label, cols in groups:
        for h in HORIZONS:
            s = S[(label, h)]
            years = sorted(set(s["year"].tolist()))
            for Y in years:
                if Y - years[0] < 5:
                    continue
                tr, te = s["year"] < Y, s["year"] == Y
                if tr.sum() < 2000 or te.sum() < 100:
                    continue
                ztr, zltr = s["z"][tr], s["zl"][tr]
                shte, aite = s["sh"][te], s["ai"][te]
                clim = np.clip(L.climatology(s["ai"][tr], k=4), 1e-6, None)
                clim /= clim.sum()
                row = dict(group=label, h=h, year=Y, n_test=int(te.sum()),
                           clim_ll=L.log_loss(np.tile(clim, (te.sum(), 1)), aite))
                tf = qq_fit_t(ztr)
                nu = tf["nu"]
                med, sc = float(np.median(ztr)), iqs(ztr)
                shape = (ztr - med) / sc
                row["emp_table"] = L.log_loss(probs_from_cdf(EmpCDF(ztr, nu).cdf, shte), aite)
                esym = EmpCDF(np.concatenate([shape, -shape]), nu)
                row["emp_sym"] = L.log_loss(
                    probs_from_cdf(lambda k: esym.cdf((k - med) / sc), shte), aite)
                # ---- log-space
                tfl = qq_fit_t(zltr)
                medl, scl = float(np.median(zltr)), iqs(zltr)
                shl = (zltr - medl) / scl
                row["log_table"] = L.log_loss(
                    probs_from_log_cdf(EmpCDF(zltr, tfl["nu"]).cdf, shte), aite)
                lsym = EmpCDF(np.concatenate([shl, -shl]), tfl["nu"])
                row["log_sym"] = L.log_loss(
                    probs_from_log_cdf(lambda k: lsym.cdf((k - medl) / scl), shte), aite)
                # ---- other asset class's centred shape, own loc/scale
                other = "SINGLES" if label != "SINGLES" else "INDEX5"
                okey = (other, h, Y)
                if okey not in other_cache:
                    s3 = S[(other, h)]
                    z3 = s3["z"][s3["year"] < Y]
                    other_cache[okey] = (EmpCDF((z3 - np.median(z3)) / iqs(z3), nu)
                                         if z3.size > 2000 else None)
                oc = other_cache[okey]
                if oc is not None:
                    row["other_shape"] = L.log_loss(
                        probs_from_cdf(lambda k: oc.cdf((k - med) / sc), shte), aite)
                    osym = EmpCDF(np.concatenate([oc.a, -oc.a]), nu)
                    row["other_shape_sym"] = L.log_loss(
                        probs_from_cdf(lambda k: osym.cdf((k - med) / sc), shte), aite)
                row["normal01"] = L.log_loss(probs_from_cdf(norm_cdf_vec, shte), aite)
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_pickle(save)
    mods = ["normal01", "emp_table", "emp_sym", "log_table", "log_sym", "other_shape",
            "other_shape_sym"]
    mods = [m for m in mods if m in df.columns]
    for label in ("INDEX5", "SPY", "SINGLES"):
        dd = df[df.group == label].dropna(subset=mods)
        if dd.empty:
            continue
        print(f"\n--- [{label}] OOS skill vs train-climatology, n_years/h = "
              f"{dd.groupby('h').size().to_dict()} ---")
        sk = pd.DataFrame({m: 1 - dd.groupby("h")[m].mean() / dd.groupby("h").clim_ll.mean()
                           for m in mods})
        print(sk.to_string())
        print(f"[{label}] paired-by-test-year t of (model - emp_table); >0 => emp_table BETTER, "
              f"<0 => model better")
        tt, mg = {}, {}
        for m in mods:
            if m == "emp_table":
                continue
            g = dd.assign(x=dd[m] - dd.emp_table).groupby("h").x
            tt[m] = g.mean() / (g.std(ddof=1) / np.sqrt(g.size()))
            mg[m] = g.mean()
        print(pd.DataFrame(tt).to_string())
        print(f"[{label}] mean log-loss DIFFERENCE (model - emp_table), same sign convention")
        print(pd.DataFrame(mg).to_string())
    return df


# ================================================================== SECTION 8  (calibration)
def sec8():
    d = data()
    print("=" * 150)
    print("S8. OOS CALIBRATION of P(down_big) and P(up_big), pooled over test years, "
          "for SINGLES and INDEX5.")
    print("    Aggregate ECE AND ECE within vol quintile / within calendar year "
          "(Simpson's check).")
    for label, cols in (("SINGLES", d["singles"]), ("INDEX5", d["idx"])):
        for h in (1, 21):
            s = _stack_full(h, cols, d)
            s["ai"] = actual_idx(s["fwd"])
            years = sorted(set(s["year"].tolist()))
            acc = []
            for Y in years:
                if Y - years[0] < 5:
                    continue
                tr, te = s["year"] < Y, s["year"] == Y
                if tr.sum() < 2000 or te.sum() < 100:
                    continue
                ztr, zltr = s["z"][tr], s["zl"][tr]
                shte = s["sh"][te]
                nu = qq_fit_t(ztr)["nu"]
                med, sc = float(np.median(ztr)), iqs(ztr)
                shape = (ztr - med) / sc
                P_asym = probs_from_cdf(EmpCDF(ztr, nu).cdf, shte)
                esym = EmpCDF(np.concatenate([shape, -shape]), nu)
                P_sym = probs_from_cdf(lambda k: esym.cdf((k - med) / sc), shte)
                nul = qq_fit_t(zltr)["nu"]
                medl, scl = float(np.median(zltr)), iqs(zltr)
                shl = (zltr - medl) / scl
                lsym = EmpCDF(np.concatenate([shl, -shl]), nul)
                P_logsym = probs_from_log_cdf(lambda k: lsym.cdf((k - medl) / scl), shte)
                acc.append(pd.DataFrame(dict(
                    year=Y, rk=s["rk"][te], ai=s["ai"][te],
                    dn_asym=P_asym[:, 0], up_asym=P_asym[:, 3],
                    dn_sym=P_sym[:, 0], up_sym=P_sym[:, 3],
                    dn_logsym=P_logsym[:, 0], up_logsym=P_logsym[:, 3])))
            if not acc:
                continue
            A = pd.concat(acc, ignore_index=True)
            A["hit_dn"] = (A.ai == 0).astype(float)
            A["hit_up"] = (A.ai == 3).astype(float)
            A["volq"] = np.digitize(A.rk, [0.2, 0.4, 0.6, 0.8]) + 1
            print(f"\n=== [{label}] h={h}  n_oos={len(A):,} ===")
            summ = []
            for mdl in ("asym", "sym", "logsym"):
                summ.append(dict(model=mdl,
                                 pred_dn=A[f"dn_{mdl}"].mean(), obs_dn=A.hit_dn.mean(),
                                 pred_up=A[f"up_{mdl}"].mean(), obs_up=A.hit_up.mean(),
                                 ece_dn=L.ece(A[f"dn_{mdl}"].to_numpy(), A.hit_dn.to_numpy()),
                                 ece_up=L.ece(A[f"up_{mdl}"].to_numpy(), A.hit_up.to_numpy())))
            print("AGGREGATE mean-predicted vs observed, and ECE:")
            print(pd.DataFrame(summ).to_string(index=False))
            print("BY VOL QUINTILE: predicted-minus-observed bias (positive = over-predicts)")
            rows = []
            for q, g in A.groupby("volq"):
                r = dict(volq=q, n=len(g), obs_dn=g.hit_dn.mean(), obs_up=g.hit_up.mean())
                for mdl in ("asym", "sym", "logsym"):
                    r[f"bias_dn_{mdl}"] = g[f"dn_{mdl}"].mean() - g.hit_dn.mean()
                    r[f"bias_up_{mdl}"] = g[f"up_{mdl}"].mean() - g.hit_up.mean()
                rows.append(r)
            print(pd.DataFrame(rows).to_string(index=False))
            print("BY YEAR: predicted-minus-observed bias for down_big")
            rows = []
            for y, g in A.groupby("year"):
                r = dict(year=y, n=len(g), obs_dn=g.hit_dn.mean(), obs_up=g.hit_up.mean())
                for mdl in ("asym", "sym", "logsym"):
                    r[f"b_dn_{mdl}"] = g[f"dn_{mdl}"].mean() - g.hit_dn.mean()
                    r[f"b_up_{mdl}"] = g[f"up_{mdl}"].mean() - g.hit_up.mean()
                rows.append(r)
            t = pd.DataFrame(rows)
            print(t.to_string(index=False))
            print(f"  mean |bias_dn| across years: " +
                  "  ".join(f"{m}={t[f'b_dn_{m}'].abs().mean():.4f}"
                            for m in ("asym", "sym", "logsym")) +
                  "   |bias_up|: " +
                  "  ".join(f"{m}={t[f'b_up_{m}'].abs().mean():.4f}"
                            for m in ("asym", "sym", "logsym")))


SECTIONS = {"0": sec0, "1": sec1, "2": sec2, "3": sec3, "4": sec4, "5": sec5, "6": sec6,
            "7": sec7, "8": sec8}

if __name__ == "__main__":
    want = sys.argv[1:] or list(SECTIONS)
    for s in want:
        SECTIONS[s]()
        print()
