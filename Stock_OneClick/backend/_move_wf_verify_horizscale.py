"""
_move_wf_verify_horizscale.py — ADVERSARIAL verification of the "horizon-scaling" claim.

Claim under test (from _move_wf_volscale2/3/4.py):
  "A mean-reverting variance forecast beats plain sqrt(h) scaling out of sample at every horizon,
   and the required correction is state-dependent by 35-40% between calm and excited states; the
   recommended form removes 58% of the state-dependent scale miscalibration for single names and
   80% of the tail-probability spread for indices."

Attacks implemented, each labelled in the output:

 S0  REPLICATION of the headline OOS QLIKE numbers (flat vs ar1, indices/singles, h=21/63).
 S1  STRAWMAN BASELINE.  Every model in the original race is  sigma_h_hat = c_h * base(t) * sqrt(h)
     with c_h refit per h.  So (a) sqrt(h) is common to ALL of them and is never tested, and
     (b) "flat" is EWMA(lam=0.94), an extremely noisy per-day estimator.  Controls added:
     six fixed-window estimators, a train-selected best fixed estimator, an h-INDEPENDENT
     shrink weight, and an h-INDEPENDENT log-blend.  If a boring estimator matches ar1, the
     "mean-reverting / horizon" story is not what is doing the work.
 S1b TRUE sqrt(h) TEST: flat94_c1 = one single c for ALL h (that is literally plain sqrt(h)).
     flat94_c1 -> flat94 isolates the pure per-h LEVEL correction; flat94 -> ar1 isolates the
     h-dependent state-mapping.  Only the second is "mean reversion".
 S2  h-DEPENDENCE: ar1 (w free per h) vs shrink1w (one w for all h).  Also prints fitted w_h.
 S3  PRECISION: per-DATE clustered block bootstrap on the QLIKE gap + non-overlapping (stride-h)
     dates + effective independent sample size for the 5 index series.
 S4  FRAGILITY: drop test years 2008/2009/2020; per-year win counts and gaps; indices vs singles
     kept strictly separate throughout.
 S5  TRAIN/TEST LEAK: train rows in the last h days of year y-1 have targets realised INSIDE test
     year y.  Purge them and re-score.
 S6  THE STATE-DEPENDENCE ARTIFACT: R = sigma_fwd / sigma_ewma is sorted on a quintile of
     sigma_ewma/sigma_252 -- i.e. sorted on its own noisy DENOMINATOR.  Control = i.i.d. shuffled
     returns (zero vol clustering, identical unconditional distribution). Any Q1/Q5 spread there
     is pure errors-in-variables.
 S7  CALIBRATION SUBGROUPS (Simpson): sd(z) and P(|z|>1.5) pooled-over-years vs WITHIN each test
     year, and the 58%/80% numbers recomputed against the boring baselines.
 S8  DOES IT MATTER: end-to-end 4-bucket (+/-2%) probability log loss vs CLIMATOLOGY, OOS.

Run: ../../vcp_env/bin/python _move_wf_verify_horizscale.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import _move_data as D
import _move_lib as L

HS = [2, 3, 5, 10, 21, 42, 63]
IDX = ["SPY", "QQQ", "^GSPC", "IWM", "DIA"]
MIN_OBS, FLOOR = 500, 1e-4
THR = 0.02
FIXED = ["ew94", "ew97", "cc22", "cc63", "cc252", "yz10"]
WGRID = np.round(np.concatenate([np.arange(0.02, 1.0, 0.025), [1.0]]), 4)
AGRID = np.round(np.arange(0.0, 1.351, 0.05), 4)
FITCAP = 800_000
CRISIS = {2008, 2009, 2020}
RNG = np.random.default_rng(2026)

MODELS = ["flat94_c1", "flat94", "flat97", "cc22", "cc63", "cc252", "yz10", "flatBest",
          "ar1", "ar1own", "shrink1w", "blendFree", "blendFix", "ar1q", "blendFixq",
          "flat94_purge", "ar1_purge"]
BUCKET_MODELS = ["flat94_c1", "flat94", "cc63", "ar1", "blendFix", "ar1q", "blendFixq"]
DATE_MODELS = ["flat94", "ar1", "ar1q", "blendFix", "blendFixq", "cc63", "shrink1w"]

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 90)
pd.set_option("display.max_rows", 400)


# ------------------------------------------------------------------ small helpers
def qlike_el(fc2, f2):
    """per-observation QLIKE, x = actual^2/forecast^2 (same convention as the originals)."""
    x = f2 / fc2
    return x - np.log(x) - 1.0


def blocks_of(n, b):
    return [np.arange(s, min(s + b, n)) for s in range(0, n, b)]


def block_boot_ci(v, wts, b=252, n_boot=2000, rng=None):
    """count-weighted mean of per-date values, block bootstrap over contiguous date blocks."""
    rng = rng or np.random.default_rng(5)
    v, wts = np.asarray(v, float), np.asarray(wts, float)
    bl = blocks_of(len(v), b)
    if len(bl) < 3:
        return (np.nan, np.nan)
    out = []
    for _ in range(n_boot):
        pk = rng.integers(0, len(bl), len(bl))
        ii = np.concatenate([bl[i] for i in pk])
        w = wts[ii]
        out.append(float(np.sum(v[ii] * w) / np.sum(w)))
    return tuple(np.percentile(out, [2.5, 97.5]))


def wmean(g, col):
    return float(np.average(g[col], weights=g.n))


def main():
    t0 = time.time()
    p = D.load()
    close, high, low, opn = p["Close"], p["High"], p["Low"], p["Open"]
    keep = [s for s in close.columns if close[s].notna().sum() >= MIN_OBS and s != "^VIX"]
    keep = [s for s in IDX if s in keep] + [s for s in keep if s not in IDX]
    close, high, low, opn = close[keep], high[keep], low[keep], opn[keep]
    singles = [s for s in keep if s not in IDX]
    lr = np.log(close).diff()
    years = close.index.year.values
    yrs = sorted(set(years))
    nrow = len(close)
    print(f"panel: {len(keep)} symbols kept of {p['Close'].shape[1]} "
          f"({len(IDX)} indices + {len(singles)} singles), {nrow} rows "
          f"{close.index[0].date()} -> {close.index[-1].date()}, "
          f"{int(close.notna().sum().sum()):,} non-null closes")

    BASES = {"ew94": L.vol_ewma(close, 0.94), "ew97": L.vol_ewma(close, 0.97),
             "cc22": L.vol_cc(close, 22), "cc63": L.vol_cc(close, 63),
             "cc252": L.vol_cc(close, 252),
             "yz10": L.vol_yang_zhang(opn, high, low, close, 10)}
    state = BASES["ew94"] / BASES["cc252"]
    fwd = {h: L.realized_vol_forward(close, h) for h in HS}
    frl = {h: np.log(close.shift(-h) / close) for h in HS}
    frs = {h: close.shift(-h) / close - 1.0 for h in HS}

    rows, sdz, prm, bkt = [], [], [], []
    dsum = {}          # (grp,h,model) -> per-date sum of qlike
    dcnt = {}          # (grp,h)       -> per-date count
    dhit = {}          # (grp,h,model,q) -> per-date [hits, count]  (for tail-spread CI)

    for lab, cols in [("INDICES", IDX), ("SINGLES", singles)]:
        ci = [close.columns.get_loc(c) for c in cols]
        ncol = len(ci)
        raw = {k: np.ascontiguousarray(v.iloc[:, ci].values) for k, v in BASES.items()}
        raw["st"] = np.ascontiguousarray(state.iloc[:, ci].values)
        lrv = np.ascontiguousarray(lr.iloc[:, ci].values)
        ok = np.ones((nrow, ncol), bool)
        for k in FIXED:
            ok &= np.isfinite(raw[k]) & (raw[k] >= FLOOR)
        ok &= np.isfinite(raw["st"])
        yrw = np.repeat(years[:, None], ncol, 1)
        rrw = np.repeat(np.arange(nrow, dtype=np.int32)[:, None], ncol, 1)
        crw = np.repeat(np.arange(ncol, dtype=np.int32)[None, :], nrow, 0)

        per = {}
        for h in HS:
            F = fwd[h].iloc[:, ci].values
            RL = frl[h].iloc[:, ci].values
            RS = frs[h].iloc[:, ci].values
            m = ok & np.isfinite(F) & (F >= FLOOR) & np.isfinite(RL) & np.isfinite(RS)
            idx = np.flatnonzero(m)
            per[h] = dict(idx=idx, yy=yrw.ravel()[idx].astype(np.int16),
                          rr=rrw.ravel()[idx], cc=crw.ravel()[idx],
                          f2=(F.ravel()[idx]) ** 2, rl=RL.ravel()[idx], rs=RS.ravel()[idx])
            del F, RL, RS, m
            dcnt[(lab, h)] = np.zeros(nrow)
            for mn in DATE_MODELS:
                dsum[(lab, h, mn)] = np.zeros(nrow)
                for q in range(5):
                    dhit[(lab, h, mn, q)] = np.zeros((nrow, 2))
        print(f"  {lab}: n per h = " + ", ".join(f"h{h}:{len(per[h]['idx']):,}" for h in HS)
              + f"   [{time.time()-t0:.0f}s]")

        def gv(h, name, lo, hi):
            return raw[name].ravel()[per[h]["idx"][lo:hi]].astype(np.float64)

        for y in yrs[5:]:
            LRp = float(np.nanmean(lrv[years < y] ** 2))
            sub = lrv[years < y]
            LRsym = np.nanmean(np.where(np.isfinite(sub), sub ** 2, np.nan), axis=0)
            LRsym = np.where(np.isfinite(LRsym), LRsym, LRp)
            bd = {}
            for h in HS:
                yy = per[h]["yy"]
                i0, i1 = np.searchsorted(yy, [y, y + 1])
                if i0 < 1500 or (i1 - i0) < 100:
                    continue
                # purge bound: last train row strictly more than h trading days before year y
                r0 = int(np.flatnonzero(years == y)[0])
                ip = int(np.searchsorted(per[h]["rr"][:i0], r0 - h))
                bd[h] = (i0, i1, ip)
            if len(bd) < len(HS):
                continue

            # ---------------- PASS 1: train-only fits (shared-across-h params need all h first)
            fit = {}
            for h in HS:
                i0, i1, ip = bd[h]
                f2 = per[h]["f2"][:i0]
                n = i0
                if n > FITCAP:
                    sel = RNG.choice(n, FITCAP, replace=False)
                    sel.sort()
                else:
                    sel = slice(None)
                f2s = f2[sel]
                B = {k: gv(h, k, 0, i0)[sel] for k in FIXED}
                se2 = B["ew94"] ** 2
                sc_fixed, u_fixed = {}, {}
                for k in FIXED:
                    b2 = B[k] ** 2
                    u_fixed[k] = float(np.mean(f2s / b2))
                    sc_fixed[k] = float(np.log(u_fixed[k]) + np.mean(np.log(b2)))
                # shrink grid (pooled LR)
                scw = np.empty(len(WGRID))
                for i, w in enumerate(WGRID):
                    sh2 = np.maximum(LRp + (se2 - LRp) * w, FLOOR ** 2)
                    scw[i] = np.log(np.mean(f2s / sh2)) + np.mean(np.log(sh2))
                # shrink grid (own-symbol LR)
                LRo = LRsym[per[h]["cc"][:i0][sel]]
                scwo = np.empty(len(WGRID))
                for i, w in enumerate(WGRID):
                    sh2 = np.maximum(LRo + (se2 - LRo) * w, FLOOR ** 2)
                    scwo[i] = np.log(np.mean(f2s / sh2)) + np.mean(np.log(sh2))
                # log-blend grid  base = ew94^a * cc252^(1-a)
                le, ll = np.log(B["ew94"]), np.log(B["cc252"])
                sca = np.empty(len(AGRID))
                for i, a in enumerate(AGRID):
                    lb2 = 2.0 * (a * le + (1 - a) * ll)
                    sca[i] = np.log(np.mean(f2s * np.exp(-lb2))) + float(np.mean(lb2))
                fit[h] = dict(n=n, u_fixed=u_fixed, sc_fixed=sc_fixed, scw=scw, scwo=scwo,
                              sca=sca)
            nw = np.array([fit[h]["n"] for h in HS], float)
            c1_v = float(np.sum(nw * [fit[h]["u_fixed"]["ew94"] for h in HS]) / nw.sum())
            w_sh = float(WGRID[int(np.argmin(np.sum(nw[:, None] *
                                                    np.array([fit[h]["scw"] for h in HS]),
                                                    axis=0)))])
            a_sh = float(AGRID[int(np.argmin(np.sum(nw[:, None] *
                                                    np.array([fit[h]["sca"] for h in HS]),
                                                    axis=0)))])

            # ---------------- PASS 2: build, fit c, score OOS
            for h in HS:
                i0, i1, ip = bd[h]
                f2t, f2e = per[h]["f2"][:i0], per[h]["f2"][i0:i1]
                rlt, rle = per[h]["rl"][:i0], per[h]["rl"][i0:i1]
                rse = per[h]["rs"][i0:i1]
                rre = per[h]["rr"][i0:i1]
                Bt = {k: gv(h, k, 0, i0) for k in FIXED}
                Be = {k: gv(h, k, i0, i1) for k in FIXED}
                stt, ste = gv(h, "st", 0, i0), gv(h, "st", i0, i1)
                edges = np.percentile(stt, [20, 40, 60, 80])
                qt, qe = np.digitize(stt, edges), np.digitize(ste, edges)
                w_h = float(WGRID[int(np.argmin(fit[h]["scw"]))])
                w_ho = float(WGRID[int(np.argmin(fit[h]["scwo"]))])
                a_h = float(AGRID[int(np.argmin(fit[h]["sca"]))])
                kbest = min(FIXED, key=lambda k: fit[h]["sc_fixed"][k])
                LRot, LRoe = LRsym[per[h]["cc"][:i0]], LRsym[per[h]["cc"][i0:i1]]

                def mk(B, se2, LRo, st, q):
                    o = {"flat94": B["ew94"], "flat94_c1": B["ew94"], "flat97": B["ew97"],
                         "cc22": B["cc22"], "cc63": B["cc63"], "cc252": B["cc252"],
                         "yz10": B["yz10"], "flatBest": B[kbest]}
                    o["ar1"] = np.sqrt(np.maximum(LRp + (se2 - LRp) * w_h, FLOOR ** 2))
                    o["ar1own"] = np.sqrt(np.maximum(LRo + (se2 - LRo) * w_ho, FLOOR ** 2))
                    o["shrink1w"] = np.sqrt(np.maximum(LRp + (se2 - LRp) * w_sh, FLOOR ** 2))
                    o["blendFree"] = B["ew94"] ** a_h * B["cc252"] ** (1 - a_h)
                    o["blendFix"] = B["ew94"] ** a_sh * B["cc252"] ** (1 - a_sh)
                    return o

                bt = mk(Bt, Bt["ew94"] ** 2, LRot, stt, qt)
                be = mk(Be, Be["ew94"] ** 2, LRoe, ste, qe)
                # quintile rms corrections (train, shape only)
                for src, dst in [("ar1", "ar1q"), ("blendFix", "blendFixq")]:
                    R = f2t / np.maximum(bt[src], FLOOR) ** 2
                    cq = np.array([np.sqrt(np.mean(R[qt == k])) if (qt == k).sum() > 30 else 1.0
                                   for k in range(5)])
                    cq = cq / np.sqrt(np.mean(R))
                    bt[dst], be[dst] = bt[src] * cq[qt], be[src] * cq[qe]
                    if dst == "ar1q":
                        cq_keep = cq
                # purged variants: fit c (and w) on train rows ending h days before year y
                wp = float(WGRID[int(np.argmin([
                    np.log(np.mean(f2t[:ip] / np.maximum(LRp + (Bt["ew94"][:ip] ** 2 - LRp) * w,
                                                         FLOOR ** 2)))
                    + np.mean(np.log(np.maximum(LRp + (Bt["ew94"][:ip] ** 2 - LRp) * w,
                                                FLOOR ** 2)))
                    for w in WGRID]))])
                bt["flat94_purge"], be["flat94_purge"] = Bt["ew94"], Be["ew94"]
                bt["ar1_purge"] = np.sqrt(np.maximum(LRp + (Bt["ew94"] ** 2 - LRp) * wp,
                                                     FLOOR ** 2))
                be["ar1_purge"] = np.sqrt(np.maximum(LRp + (Be["ew94"] ** 2 - LRp) * wp,
                                                     FLOOR ** 2))

                for mn in MODELS:
                    b2t = np.maximum(bt[mn], FLOOR) ** 2
                    b2e = np.maximum(be[mn], FLOOR) ** 2
                    if mn.endswith("_purge"):
                        c2 = float(np.mean(f2t[:ip] / b2t[:ip]))
                        cz = float(np.mean(rlt[:ip] ** 2 / (h * b2t[:ip])))
                    elif mn == "flat94_c1":
                        c2, cz = c1_v, None
                    else:
                        c2 = float(np.mean(f2t / b2t))
                        cz = float(np.mean(rlt ** 2 / (h * b2t)))
                    ql = qlike_el(c2 * b2e, f2e)
                    rows.append(dict(grp=lab, h=h, y=y, model=mn, n=int(i1 - i0),
                                     QL=float(ql.mean())))
                    if mn in DATE_MODELS:
                        dsum[(lab, h, mn)] += np.bincount(rre, weights=ql, minlength=nrow)
                    if cz is None:
                        cz = float(np.mean(rlt ** 2 / (h * b2t)))
                    z = rle / np.sqrt(cz * h * b2e)
                    az = np.abs(z)
                    for k in range(5):
                        s = qe == k
                        if s.sum() < 20:
                            continue
                        sdz.append(dict(grp=lab, h=h, y=y, model=mn, q=k + 1, n=int(s.sum()),
                                        sq=float(np.mean(z[s] ** 2)),
                                        hit=float(np.mean(az[s] > 1.5)),
                                        hit2=float(np.mean(az[s] > 2.0))))
                        if mn in DATE_MODELS:
                            dhit[(lab, h, mn, k)][:, 0] += np.bincount(
                                rre[s], weights=(az[s] > 1.5).astype(float), minlength=nrow)
                            dhit[(lab, h, mn, k)][:, 1] += np.bincount(rre[s], minlength=nrow)
                    # ---- end-to-end bucket probabilities (empirical z shape from TRAIN)
                    if mn in BUCKET_MODELS:
                        zt = rlt / np.sqrt(cz * h * np.maximum(bt[mn], FLOOR) ** 2)
                        if len(zt) > 300_000:
                            zt = zt[RNG.choice(len(zt), 300_000, replace=False)]
                        zs = np.sort(zt)
                        nz = len(zs)
                        sg = np.sqrt(cz * h * b2e)
                        k1 = np.searchsorted(zs, np.log(1 - THR) / sg) / nz
                        k2 = np.searchsorted(zs, 0.0) / nz
                        k3 = np.searchsorted(zs, np.log(1 + THR) / sg) / nz
                        P = np.column_stack([k1, np.maximum(k2 - k1, 0), np.maximum(k3 - k2, 0),
                                             np.maximum(1 - k3, 0)])
                        P = np.clip(P, 1e-6, 1.0)
                        P /= P.sum(axis=1, keepdims=True)
                        ai = np.where(rse <= -THR, 0, np.where(rse <= 0, 1,
                                      np.where(rse < THR, 2, 3)))
                        at = np.where(per[h]["rs"][:i0] <= -THR, 0,
                                      np.where(per[h]["rs"][:i0] <= 0, 1,
                                      np.where(per[h]["rs"][:i0] < THR, 2, 3)))
                        clim = L.climatology(at)
                        bkt.append(dict(grp=lab, h=h, y=y, model=mn, n=int(i1 - i0),
                                        LL=L.log_loss(P, ai),
                                        LLclim=L.log_loss(np.tile(clim, (len(ai), 1)), ai),
                                        BR=L.brier_multi(P, ai)))
                prm.append(dict(grp=lab, h=h, y=y, w_h=w_h, w_sh=w_sh, w_ho=w_ho, a_h=a_h,
                                a_sh=a_sh, kbest=kbest, wp=wp,
                                phi_h=np.nan, LRsig=np.sqrt(LRp),
                                **{f"cq{k+1}": cq_keep[k] for k in range(5)}))
                dcnt[(lab, h)] += np.bincount(rre, minlength=nrow)
            if y % 5 == 0:
                print(f"    {lab} year {y} done [{time.time()-t0:.0f}s]")
        del raw, lrv, ok, yrw, rrw, crw, per
        print(f"  {lab} finished [{time.time()-t0:.0f}s]")

    A = pd.DataFrame(rows)
    S = pd.DataFrame(sdz)
    P_ = pd.DataFrame(prm)
    BK = pd.DataFrame(bkt)

    # ================================================================= S0 replication
    print("\n" + "=" * 120)
    print("S0. REPLICATION of the headline OOS QLIKE (expanding walk-forward, params train-only)")
    print("    claim: INDICES h=63 0.5705->0.4543 (-20.4%, n=25,330), h=21 -7.0%;")
    print("           SINGLES h=63 0.5929->0.3689 (-37.8%, n=948k),  h=21 -7.9%")
    print("=" * 120)
    for lab in ("INDICES", "SINGLES"):
        for h in (21, 63):
            g = A[(A.grp == lab) & (A.h == h)]
            fl = wmean(g[g.model == "flat94"], "QL")
            ar = wmean(g[g.model == "ar1"], "QL")
            n = int(g[g.model == "flat94"].n.sum())
            print(f"  {lab:8s} h={h:2d}: flat94 {fl:.4f} -> ar1 {ar:.4f} "
                  f"({100*(1-ar/fl):+.1f}%)  n={n:,}  years={g.y.nunique()}")

    # ================================================================= S1 boring controls
    print("\n" + "=" * 120)
    print("S1. STRAWMAN-BASELINE ATTACK -- OOS QLIKE, n-weighted over walk-forward test years.")
    print("    All rows are sigma_h_hat = c_h * base(t) * sqrt(h); sqrt(h) is COMMON to every row")
    print("    and c_h is refit per h, so nothing here tests sqrt(h) itself (see S1b).")
    print("    flat94 = the claim's baseline (EWMA 0.94). cc63/cc252 = boring longer windows.")
    print("=" * 120)
    show = ["flat94_c1", "flat94", "flat97", "cc22", "cc63", "cc252", "yz10", "flatBest",
            "ar1", "ar1own", "shrink1w", "blendFree", "blendFix", "ar1q", "blendFixq"]
    for lab in ("INDICES", "SINGLES"):
        g = A[A.grp == lab]
        piv = g.groupby(["h", "model"]).apply(lambda x: wmean(x, "QL"),
                                              include_groups=False).unstack()[show]
        piv["BEST"] = piv.idxmin(axis=1)
        piv["ar1_vs_cc63%"] = 100 * (1 - piv["ar1"] / piv["cc63"])
        piv["ar1q_vs_blendFix%"] = 100 * (1 - piv["ar1q"] / piv["blendFix"])
        piv["n"] = g.groupby("h").n.sum() // len(MODELS)
        print(f"\n  {lab} / OOS QLIKE (lower better):")
        print(piv.round(4).to_string())

    print("\n" + "-" * 120)
    print("S1b. TRUE sqrt(h) DECOMPOSITION.  flat94_c1 = ONE c for all h = literally plain")
    print("     sqrt(h) scaling of one per-day estimate.  Gains are cumulative left to right.")
    print("-" * 120)
    for lab in ("INDICES", "SINGLES"):
        g = A[A.grp == lab]
        t = pd.DataFrame({m: g[g.model == m].groupby("h").apply(lambda x: wmean(x, "QL"),
                                                               include_groups=False)
                          for m in ("flat94_c1", "flat94", "ar1", "cc63", "blendFix")})
        t["level_only_%"] = 100 * (1 - t.flat94 / t.flat94_c1)
        t["mean_rev_extra_%"] = 100 * (1 - t.ar1 / t.flat94)
        t["boring_cc63_%"] = 100 * (1 - t.cc63 / t.flat94_c1)
        t["ar1_total_%"] = 100 * (1 - t.ar1 / t.flat94_c1)
        print(f"\n  {lab}:")
        print(t.round(4).to_string())

    # ================================================================= S2 h-dependence
    print("\n" + "=" * 120)
    print("S2. IS THE SHRINK WEIGHT REALLY h-DEPENDENT?  ar1 fits w per h; shrink1w uses ONE w.")
    print("=" * 120)
    for lab in ("INDICES", "SINGLES"):
        g = P_[P_.grp == lab]
        t = g.groupby("h").agg(w_h_med=("w_h", "median"), w_h_min=("w_h", "min"),
                               w_h_max=("w_h", "max"), w_own_med=("w_ho", "median"),
                               a_h_med=("a_h", "median"))
        t["w_shared_med"] = g.w_sh.median()
        t["a_shared_med"] = g.a_sh.median()
        aa = A[A.grp == lab]
        t["QL_ar1"] = aa[aa.model == "ar1"].groupby("h").apply(lambda x: wmean(x, "QL"),
                                                              include_groups=False)
        t["QL_shrink1w"] = aa[aa.model == "shrink1w"].groupby("h").apply(
            lambda x: wmean(x, "QL"), include_groups=False)
        t["ar1_gain_over_1w_%"] = 100 * (1 - t.QL_ar1 / t.QL_shrink1w)
        print(f"\n  {lab} (median over walk-forward fits):")
        print(t.round(4).to_string())
        print(f"  train-selected best fixed estimator by (h): " +
              ", ".join(f"h{h}:{g[g.h==h].kbest.mode().iloc[0]}" for h in HS))

    # ================================================================= S3 precision
    print("\n" + "=" * 120)
    print("S3. PRECISION.  (a) effective independent series count for the 'indices' pool.")
    print("=" * 120)
    cr = lr[IDX].corr()
    print(cr.round(4).to_string())
    ev = np.linalg.eigvalsh(cr.values)
    print(f"  n_eff (participation ratio of eigenvalues) = "
          f"{(ev.sum()**2/ (ev**2).sum()):.2f} of {len(IDX)} series -> the 25,330 index")
    print(f"  observations are ~{5066:,} dates x ~1 independent factor, not 25k independent draws.")
    print("\n  (b) per-DATE clustered block bootstrap (252-date blocks, 2000 reps) on the")
    print("      QLIKE gap vs flat94, and the same on NON-OVERLAPPING dates (stride h).")
    for lab in ("INDICES", "SINGLES"):
        out = []
        for h in HS:
            cnt = dcnt[(lab, h)]
            v = cnt > 0
            qf = dsum[(lab, h, "flat94")][v] / cnt[v]
            row = dict(h=h, n_dates=int(v.sum()))
            for mn in ("ar1", "cc63", "blendFix", "ar1q", "shrink1w"):
                qm = dsum[(lab, h, mn)][v] / cnt[v]
                d = qf - qm
                lo, hi = block_boot_ci(d, cnt[v], 252, 2000, np.random.default_rng(7))
                row[f"{mn}_gap"] = float(np.average(d, weights=cnt[v]))
                row[f"{mn}_lo"] = lo
                row[f"{mn}_hi"] = hi
                row[f"{mn}_sig"] = "yes" if lo > 0 else ("NEG" if hi < 0 else "no")
                dn = d[::h]
                wn = cnt[v][::h]
                row[f"{mn}_gapNOV"] = float(np.average(dn, weights=wn))
                lo2, hi2 = block_boot_ci(dn, wn, max(3, 252 // h), 2000,
                                         np.random.default_rng(8))
                row[f"{mn}_sigNOV"] = "yes" if lo2 > 0 else ("NEG" if hi2 < 0 else "no")
            out.append(row)
        o = pd.DataFrame(out)
        print(f"\n  {lab} -- gap = QLIKE(flat94) - QLIKE(model), positive = model better:")
        print(o[["h", "n_dates"] + [c for c in o.columns if c.startswith("ar1_")]
                ].round(4).to_string(index=False))
        print(o[["h"] + [c for c in o.columns if c.split("_")[0] in ("cc63", "blendFix")]
                ].round(4).to_string(index=False))
        print(o[["h"] + [c for c in o.columns if c.startswith("ar1q") or
                         c.startswith("shrink1w")]].round(4).to_string(index=False))

    # ================================================================= S4 fragility
    print("\n" + "=" * 120)
    print("S4. FRAGILITY.  Per-year wins and the effect of dropping test years 2008/2009/2020.")
    print("=" * 120)
    for lab in ("INDICES", "SINGLES"):
        g = A[A.grp == lab]
        wide = g.pivot_table(index=["h", "y"], columns="model", values="QL")
        nn = g.pivot_table(index=["h", "y"], columns="model", values="n")["flat94"]
        out = []
        for h in HS:
            w = wide.loc[h]
            n_ = nn.loc[h]
            nc = [yy for yy in w.index if yy not in CRISIS]
            r = dict(h=h, years=len(w),
                     ar1_wins=int((w.ar1 < w.flat94).sum()),
                     ar1_wins_vs_cc63=int((w.ar1 < w.cc63).sum()),
                     ar1_wins_vs_blendFix=int((w.ar1 < w.blendFix).sum()),
                     full_ar1_vs_flat=100 * (1 - np.average(w.ar1, weights=n_) /
                                             np.average(w.flat94, weights=n_)),
                     noCrisis_ar1_vs_flat=100 * (1 - np.average(w.loc[nc, "ar1"],
                                                                weights=n_.loc[nc]) /
                                                 np.average(w.loc[nc, "flat94"],
                                                            weights=n_.loc[nc])),
                     full_ar1_vs_cc63=100 * (1 - np.average(w.ar1, weights=n_) /
                                             np.average(w.cc63, weights=n_)),
                     noCrisis_ar1_vs_cc63=100 * (1 - np.average(w.loc[nc, "ar1"],
                                                                weights=n_.loc[nc]) /
                                                 np.average(w.loc[nc, "cc63"],
                                                            weights=n_.loc[nc])))
            out.append(r)
        print(f"\n  {lab}:")
        print(pd.DataFrame(out).round(3).to_string(index=False))
        print(f"\n  {lab} per-year QLIKE gap (flat94 - ar1) at h=63, and share of total gap "
              f"coming from crisis years:")
        w = wide.loc[63]
        n_ = nn.loc[63]
        gap = (w.flat94 - w.ar1) * n_
        print("   " + "  ".join(f"{yy}:{(w.flat94[yy]-w.ar1[yy]):+.3f}" for yy in w.index))
        print(f"   crisis-year share of the n-weighted total gap: "
              f"{gap[[yy for yy in w.index if yy in CRISIS]].sum()/gap.sum():.3f} "
              f"({len([yy for yy in w.index if yy in CRISIS])}/{len(w)} years)")

    # ================================================================= S5 leak
    print("\n" + "=" * 120)
    print("S5. TRAIN/TEST LEAK.  Unpurged train includes rows whose h-day target lands inside the")
    print("    test year. Purged = train truncated h trading days before the test year starts.")
    print("=" * 120)
    for lab in ("INDICES", "SINGLES"):
        g = A[A.grp == lab]
        t = pd.DataFrame({m: g[g.model == m].groupby("h").apply(lambda x: wmean(x, "QL"),
                                                               include_groups=False)
                          for m in ("flat94", "ar1", "flat94_purge", "ar1_purge")})
        t["unpurged_%"] = 100 * (1 - t.ar1 / t.flat94)
        t["purged_%"] = 100 * (1 - t.ar1_purge / t.flat94_purge)
        print(f"\n  {lab}:")
        print(t.round(4).to_string())

    # ================================================================= S6 artifact
    print("\n" + "=" * 120)
    print("S6. THE STATE-DEPENDENCE ARTIFACT.  R = sigma_fwd_h / sigma_ewma(t) is sorted into")
    print("    quintiles of state = sigma_ewma/sigma_252 -- i.e. sorted on its own NOISY")
    print("    DENOMINATOR. Control: i.i.d.-shuffled returns per symbol (destroys all vol")
    print("    clustering, preserves the unconditional return distribution exactly). Under that")
    print("    null the honest correction is a CONSTANT, so any Q1/Q5 spread is pure noise.")
    print("    Both panels IN-SAMPLE with full-sample quintile edges (descriptive, matched).")
    print("=" * 120)
    lrs = lr.copy()
    for c in lrs.columns:
        v = lrs[c].values
        m = np.isfinite(v)
        x = v[m].copy()
        RNG.shuffle(x)
        v[m] = x
    close_sh = np.exp(lrs.cumsum())
    for tag, cl in (("REAL", close), ("SHUFFLED", close_sh)):
        se = L.vol_ewma(cl, 0.94)
        sl = L.vol_cc(cl, 252)
        st = se / sl
        print(f"\n  --- {tag} ---")
        for lab, cols in (("INDICES", IDX), ("SINGLES", singles)):
            recs = []
            for h in HS:
                fw = L.realized_vol_forward(cl, h)
                a = se[cols].values.ravel()
                b = fw[cols].values.ravel()
                s = st[cols].values.ravel()
                m = (np.isfinite(a) & (a >= FLOOR) & np.isfinite(b) & (b >= FLOOR)
                     & np.isfinite(s) & (s > 0))
                a, b, s = a[m], b[m], s[m]
                R = b / a
                ed = np.percentile(s, [20, 40, 60, 80])
                q = np.digitize(s, ed)
                rms = np.array([np.sqrt(np.mean(R[q == k] ** 2)) for k in range(5)])
                lr_ = np.log(R)
                ls = np.log(s)
                sl_ = float(np.cov(ls, lr_)[0, 1] / np.var(ls))
                recs.append(dict(h=h, n=len(R), Q1=rms[0], Q5=rms[4],
                                 Q5_over_Q1=rms[4] / rms[0],
                                 drop_pct=100 * (rms[4] / rms[0] - 1),
                                 slope_logR_on_logState=sl_, var_log_state=float(np.var(ls)),
                                 dlog_state_Q5_Q1=float(np.mean(ls[q == 4]) -
                                                        np.mean(ls[q == 0]))))
            print(f"   {lab}:")
            print(pd.DataFrame(recs).round(4).to_string(index=False))
            if tag == "SHUFFLED":
                globals().setdefault("_SH", {})[lab] = pd.DataFrame(recs)
            else:
                globals().setdefault("_RE", {})[lab] = pd.DataFrame(recs)
    print("\n  NULL-ADJUSTED state dependence: under zero true state-dependence the measured")
    print("  slope is -var(noise)/var(log state). var(noise) is estimated from the shuffled")
    print("  panel (where log-state variance IS the noise variance). Applied to the REAL")
    print("  log-state spread this gives the artifact-only Q5/Q1 ratio.")
    for lab in ("INDICES", "SINGLES"):
        re_, sh_ = globals()["_RE"][lab], globals()["_SH"][lab]
        t = pd.DataFrame({"h": re_.h,
                          "real_Q5/Q1": re_.Q5_over_Q1,
                          "shuf_Q5/Q1": sh_.Q5_over_Q1,
                          "real_slope": re_.slope_logR_on_logState,
                          "shuf_slope": sh_.slope_logR_on_logState,
                          "var_ls_real": re_.var_log_state,
                          "var_ls_shuf": sh_.var_log_state})
        t["noise_share_of_state_var"] = t.var_ls_shuf / t.var_ls_real
        t["null_slope_in_real"] = -t.noise_share_of_state_var
        t["artifact_Q5/Q1"] = np.exp(t.null_slope_in_real * re_.dlog_state_Q5_Q1)
        t["artifact_share_of_logspread"] = (np.log(t["artifact_Q5/Q1"]) /
                                            np.log(t["real_Q5/Q1"]))
        print(f"\n  {lab}:")
        print(t.round(4).to_string(index=False))

    # ================================================================= S7 calibration subgroups
    print("\n" + "=" * 120)
    print("S7. CALIBRATION SUBGROUPS.  claim: mean|sd(z)-1| over all (h,q) singles")
    print("    0.1446->0.0601 (-58%), indices 0.1033->0.0815; mean|P(|z|>1.5) spread Q5-Q1|")
    print("    indices 0.0518->0.0102 (-80%), singles 0.0951->0.0235 (-75%).")
    print("    POOLED = z-moments pooled over all test years (what the claim reports).")
    print("    WITHIN-YEAR = the same statistic computed inside each test year then averaged;")
    print("    a model can look calibrated pooled while being wrong in every year (Simpson).")
    print("=" * 120)
    for lab in ("INDICES", "SINGLES"):
        g = S[S.grp == lab]
        out = []
        for mn in show:
            gg = g[g.model == mn]
            pv = gg.groupby(["h", "q"]).apply(lambda x: np.sqrt(np.average(x.sq, weights=x.n)),
                                              include_groups=False).unstack()
            hv = gg.groupby(["h", "q"]).apply(lambda x: np.average(x.hit, weights=x.n),
                                              include_groups=False).unstack()
            # within-year
            gg2 = gg.assign(sdz=np.sqrt(gg.sq))
            wy = float((gg2.sdz - 1).abs().mul(gg2.n).sum() / gg2.n.sum())
            wy_un = gg.groupby(["h", "y"]).apply(
                lambda x: abs(np.sqrt(np.average(x.sq, weights=x.n)) - 1),
                include_groups=False).mean()
            hy = gg.groupby(["h", "y"]).apply(
                lambda x: (x.set_index("q").hit.reindex([1, 2, 3, 4, 5])),
                include_groups=False)
            hs_yr = float(np.nanmean(np.abs(hy[5].values - hy[1].values)))
            out.append(dict(model=mn,
                            pooled_mean_abs_sdz=float((pv - 1).abs().values.mean()),
                            pooled_mean_abs_hitspread=float(
                                np.mean(np.abs(hv[5].values - hv[1].values))),
                            withinYear_q_mean_abs_sdz=wy,
                            withinYear_pooledq_mean_abs_sdz=float(wy_un),
                            withinYear_mean_abs_hitspread=hs_yr,
                            pooled_sdz_all=float(np.sqrt(np.average(gg.sq, weights=gg.n)))))
        o = pd.DataFrame(out)
        print(f"\n  {lab}:")
        print(o.round(4).to_string(index=False))
    print("\n  full sd(z) by (h,quintile) for the key models:")
    for lab in ("INDICES", "SINGLES"):
        g = S[S.grp == lab]
        for mn in ("flat94", "ar1q", "blendFix", "blendFixq", "cc63"):
            gg = g[g.model == mn]
            pv = gg.groupby(["h", "q"]).apply(lambda x: np.sqrt(np.average(x.sq, weights=x.n)),
                                              include_groups=False).unstack()
            pv.columns = [f"Q{c}" for c in pv.columns]
            pv["Q5/Q1"] = pv.Q5 / pv.Q1
            pv["mean|sd-1|"] = (pv[[f"Q{k}" for k in range(1, 6)]] - 1).abs().mean(axis=1)
            print(f"\n   {lab} / {mn}:")
            print(pv.round(4).to_string())
    print("\n  by-YEAR pooled sd(z) (all quintiles), h=21 -- is the level right every year?")
    for lab in ("INDICES", "SINGLES"):
        g = S[(S.grp == lab) & (S.h == 21)]
        pv = g.groupby(["y", "model"]).apply(lambda x: np.sqrt(np.average(x.sq, weights=x.n)),
                                             include_groups=False).unstack()
        cc = [c for c in ("flat94", "ar1", "ar1q", "blendFix", "blendFixq", "cc63") if c in pv]
        print(f"\n   {lab} h=21 sd(z) by test year:")
        print(pv[cc].round(3).T.to_string())
    print("\n  tail-spread P(|z|>1.5) Q5-Q1, date-clustered 95% CI (252-date blocks):")
    for lab in ("INDICES", "SINGLES"):
        rr = []
        for h in (5, 21, 63):
            for mn in ("flat94", "ar1q", "blendFixq"):
                a1, a5 = dhit[(lab, h, mn, 0)], dhit[(lab, h, mn, 4)]
                v = (a1[:, 1] > 0) & (a5[:, 1] > 0)
                d = a5[v, 0] / a5[v, 1] - a1[v, 0] / a1[v, 1]
                wt = np.minimum(a1[v, 1], a5[v, 1])
                lo, hi = block_boot_ci(d, wt, 252, 2000, np.random.default_rng(9))
                rr.append(dict(h=h, model=mn, spread=float(np.average(d, weights=wt)),
                               lo=lo, hi=hi, n_dates=int(v.sum()),
                               zero_in_CI="yes" if lo <= 0 <= hi else "no"))
        print(f"\n   {lab}:")
        print(pd.DataFrame(rr).round(4).to_string(index=False))

    # ================================================================= S8 does it matter
    print("\n" + "=" * 120)
    print("S8. DOES ANY OF IT MOVE THE ACTUAL ANSWER?  4-bucket (+/-2%) probabilities built as")
    print("    scale (model) x shape (empirical TRAIN z distribution), scored OOS against")
    print("    CLIMATOLOGY from the same train years. skill = 1 - LL/LL_clim.")
    print("=" * 120)
    for lab in ("INDICES", "SINGLES"):
        g = BK[BK.grp == lab]
        piv = g.groupby(["h", "model"]).apply(lambda x: wmean(x, "LL"),
                                             include_groups=False).unstack()
        cl = g[g.model == "flat94"].groupby("h").apply(lambda x: wmean(x, "LLclim"),
                                                      include_groups=False)
        piv = piv[[m for m in BUCKET_MODELS if m in piv]]
        sk = 1 - piv.div(cl, axis=0)
        sk["LL_clim"] = cl
        sk["n"] = g[g.model == "flat94"].groupby("h").n.sum()
        print(f"\n  {lab} / log-loss SKILL vs climatology (positive = beats climatology):")
        print(sk.round(4).to_string())
        print(f"  {lab} / raw OOS log loss:")
        print(piv.round(4).to_string())
    print(f"\ndone {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
