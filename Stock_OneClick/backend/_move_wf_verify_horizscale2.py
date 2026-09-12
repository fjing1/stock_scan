"""
_move_wf_verify_horizscale2.py — follow-ups that decide the remaining questions.

A. IS THE QLIKE GAIN JUST A FLOOR?  w=0.945 means sigma_h^2 = 0.055*LR + 0.945*sigma_ewma^2,
   i.e. a soft floor at 5.5% of the long-run variance. QLIKE explodes when the forecast is too
   LOW, and EWMA(0.94) occasionally prints a near-zero vol. Controls: a hard floor max(se^2,k*LR)
   and a two-sided winsorisation, both with k fit on train only. Also logMSE (symmetric, no tail
   explosion) and a TRIMMED QLIKE that drops the worst 0.1%/1% of flat94 observations (same rows
   dropped for every model).

B. THE DECISIVE ARTIFACT TEST (split-sample state).  R = sigma_fwd/sigma_now is sorted on
   sigma_now/long-run, so the sort variable and the denominator share the SAME estimation noise.
   Build two variance estimates from DISJOINT past days -- odd lags 1,3..29 and even lags 2,4..30
   -- and a long-run window lagged 31 days so it shares nothing with either. Then
       contaminated:  sort on se_ODD /sl_lag,  R = f/se_ODD    (noise shared -> artifact present)
       clean:         sort on se_EVEN/sl_lag,  R = f/se_ODD    (noise independent -> artifact gone)
   Same estimator precision in both, so the difference IS the errors-in-variables artifact.

C. 2023 FORENSICS: how concentrated is the singles h=63 gain in single observations.

D. Indices tail-spread P(|z|>1.5) Q5-Q1: block bootstrap over DATE BLOCKS (the per-date version
   fails because only 5 index series exist per date).

Run: ../../vcp_env/bin/python _move_wf_verify_horizscale2.py
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
WGRID = np.round(np.concatenate([np.arange(0.02, 1.0, 0.025), [1.0]]), 4)
KGRID = np.array([0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50])
AGRID = np.round(np.arange(0.0, 1.351, 0.05), 4)
FITCAP = 800_000
RNG = np.random.default_rng(4242)
MODELS = ["flat94", "flat97", "ar1", "shrink1w", "floorK", "winsK", "blendFix", "ar1q"]

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 90)
pd.set_option("display.max_rows", 400)


def qlike_el(fc2, f2):
    x = f2 / fc2
    return x - np.log(x) - 1.0


def blocks_of(n, b):
    return [np.arange(s, min(s + b, n)) for s in range(0, n, b)]


def main():
    t0 = time.time()
    p = D.load()
    close, high, low, opn = p["Close"], p["High"], p["Low"], p["Open"]
    keep = [s for s in close.columns if close[s].notna().sum() >= MIN_OBS and s != "^VIX"]
    keep = [s for s in IDX if s in keep] + [s for s in keep if s not in IDX]
    close = close[keep]
    singles = [s for s in keep if s not in IDX]
    lr = np.log(close).diff()
    years = close.index.year.values
    yrs = sorted(set(years))
    nrow = len(close)

    s_e = L.vol_ewma(close, 0.94)
    s_e97 = L.vol_ewma(close, 0.97)
    s_l = L.vol_cc(close, 252)
    state = s_e / s_l
    fwd = {h: L.realized_vol_forward(close, h) for h in HS}
    frl = {h: np.log(close.shift(-h) / close) for h in HS}

    rows = []
    dh = {}
    for lab, cols in [("INDICES", IDX), ("SINGLES", singles)]:
        ci = [close.columns.get_loc(c) for c in cols]
        ncol = len(ci)
        raw = {"ew94": np.ascontiguousarray(s_e.iloc[:, ci].values),
               "ew97": np.ascontiguousarray(s_e97.iloc[:, ci].values),
               "cc252": np.ascontiguousarray(s_l.iloc[:, ci].values),
               "st": np.ascontiguousarray(state.iloc[:, ci].values)}
        lrv = np.ascontiguousarray(lr.iloc[:, ci].values)
        ok = np.ones((nrow, ncol), bool)
        for k in ("ew94", "ew97", "cc252"):
            ok &= np.isfinite(raw[k]) & (raw[k] >= FLOOR)
        ok &= np.isfinite(raw["st"])
        yrw = np.repeat(years[:, None], ncol, 1)
        rrw = np.repeat(np.arange(nrow, dtype=np.int32)[:, None], ncol, 1)
        per = {}
        for h in HS:
            F = fwd[h].iloc[:, ci].values
            RL = frl[h].iloc[:, ci].values
            m = ok & np.isfinite(F) & (F >= FLOOR) & np.isfinite(RL)
            idx = np.flatnonzero(m)
            per[h] = dict(idx=idx, yy=yrw.ravel()[idx].astype(np.int16), rr=rrw.ravel()[idx],
                          f2=F.ravel()[idx] ** 2, lf=np.log(F.ravel()[idx]),
                          rl=RL.ravel()[idx])
            for mn in MODELS:
                for q in range(5):
                    dh[(lab, h, mn, q)] = np.zeros((nrow, 2))
            del F, RL, m

        def gv(h, name, lo, hi):
            return raw[name].ravel()[per[h]["idx"][lo:hi]].astype(np.float64)

        for y in yrs[5:]:
            LRp = float(np.nanmean(lrv[years < y] ** 2))
            bd = {}
            for h in HS:
                i0, i1 = np.searchsorted(per[h]["yy"], [y, y + 1])
                if i0 >= 1500 and (i1 - i0) >= 100:
                    bd[h] = (int(i0), int(i1))
            if len(bd) < len(HS):
                continue
            scw_all, sca_all, nn = [], [], []
            for h in HS:
                i0, i1 = bd[h]
                n = i0
                sel = slice(None)
                if n > FITCAP:
                    sel = np.sort(RNG.choice(n, FITCAP, replace=False))
                f2s = per[h]["f2"][:i0][sel]
                se2 = gv(h, "ew94", 0, i0)[sel] ** 2
                sl_ = gv(h, "cc252", 0, i0)[sel]
                scw = np.array([np.log(np.mean(f2s / np.maximum(LRp + (se2 - LRp) * w,
                                                                FLOOR ** 2)))
                                + np.mean(np.log(np.maximum(LRp + (se2 - LRp) * w, FLOOR ** 2)))
                                for w in WGRID])
                le, ll = 0.5 * np.log(se2), np.log(sl_)
                sca = np.array([np.log(np.mean(f2s * np.exp(-2 * (a * le + (1 - a) * ll))))
                                + float(np.mean(2 * (a * le + (1 - a) * ll))) for a in AGRID])
                scw_all.append(scw)
                sca_all.append(sca)
                nn.append(n)
            nn = np.array(nn, float)
            w_sh = float(WGRID[int(np.argmin((nn[:, None] * np.array(scw_all)).sum(0)))])
            a_sh = float(AGRID[int(np.argmin((nn[:, None] * np.array(sca_all)).sum(0)))])

            for hi_, h in enumerate(HS):
                i0, i1 = bd[h]
                f2t, f2e = per[h]["f2"][:i0], per[h]["f2"][i0:i1]
                lft, lfe = per[h]["lf"][:i0], per[h]["lf"][i0:i1]
                rlt, rle = per[h]["rl"][:i0], per[h]["rl"][i0:i1]
                rre = per[h]["rr"][i0:i1]
                se_t, se_e = gv(h, "ew94", 0, i0), gv(h, "ew94", i0, i1)
                e7t, e7e = gv(h, "ew97", 0, i0), gv(h, "ew97", i0, i1)
                slt, sle = gv(h, "cc252", 0, i0), gv(h, "cc252", i0, i1)
                stt, ste = gv(h, "st", 0, i0), gv(h, "st", i0, i1)
                edges = np.percentile(stt, [20, 40, 60, 80])
                qt, qe = np.digitize(stt, edges), np.digitize(ste, edges)
                w_h = float(WGRID[int(np.argmin(scw_all[hi_]))])
                # hard floor k, fit on train QLIKE
                s2t = se_t ** 2
                sck = [np.log(np.mean(f2t / np.maximum(s2t, k * LRp)))
                       + np.mean(np.log(np.maximum(s2t, k * LRp))) for k in KGRID]
                kf = float(KGRID[int(np.argmin(sck))])
                # two-sided winsorisation of se^2 in units of LR, fit on train
                best, kk = None, (0.0, np.inf)
                for klo in KGRID:
                    for khi in (0.5, 1.0, 2.0, 4.0, 8.0, np.inf):
                        v = np.clip(s2t, klo * LRp, khi * LRp)
                        sc = np.log(np.mean(f2t / v)) + np.mean(np.log(v))
                        if best is None or sc < best:
                            best, kk = sc, (klo, khi)
                bt = {"flat94": se_t, "flat97": e7t,
                      "ar1": np.sqrt(np.maximum(LRp + (s2t - LRp) * w_h, FLOOR ** 2)),
                      "shrink1w": np.sqrt(np.maximum(LRp + (s2t - LRp) * w_sh, FLOOR ** 2)),
                      "floorK": np.sqrt(np.maximum(s2t, kf * LRp)),
                      "winsK": np.sqrt(np.clip(s2t, kk[0] * LRp, kk[1] * LRp)),
                      "blendFix": se_t ** a_sh * slt ** (1 - a_sh)}
                s2e = se_e ** 2
                be = {"flat94": se_e, "flat97": e7e,
                      "ar1": np.sqrt(np.maximum(LRp + (s2e - LRp) * w_h, FLOOR ** 2)),
                      "shrink1w": np.sqrt(np.maximum(LRp + (s2e - LRp) * w_sh, FLOOR ** 2)),
                      "floorK": np.sqrt(np.maximum(s2e, kf * LRp)),
                      "winsK": np.sqrt(np.clip(s2e, kk[0] * LRp, kk[1] * LRp)),
                      "blendFix": se_e ** a_sh * sle ** (1 - a_sh)}
                R = f2t / np.maximum(bt["ar1"], FLOOR) ** 2
                cq = np.array([np.sqrt(np.mean(R[qt == k])) if (qt == k).sum() > 30 else 1.0
                               for k in range(5)])
                cq = cq / np.sqrt(np.mean(R))
                bt["ar1q"], be["ar1q"] = bt["ar1"] * cq[qt], be["ar1"] * cq[qe]

                ql = {}
                for mn in MODELS:
                    b2t = np.maximum(bt[mn], FLOOR) ** 2
                    b2e = np.maximum(be[mn], FLOOR) ** 2
                    c2 = float(np.mean(f2t / b2t))
                    ql[mn] = qlike_el(c2 * b2e, f2e)
                    clg = float(np.mean(lft - 0.5 * np.log(b2t)))
                    lm = float(np.mean((lfe - 0.5 * np.log(b2e) - clg) ** 2))
                    cz = float(np.mean(rlt ** 2 / (h * b2t)))
                    z = rle / np.sqrt(cz * h * b2e)
                    az = np.abs(z)
                    for k in range(5):
                        s = qe == k
                        if s.sum() >= 20:
                            dh[(lab, h, mn, k)][:, 0] += np.bincount(
                                rre[s], weights=(az[s] > 1.5).astype(float), minlength=nrow)
                            dh[(lab, h, mn, k)][:, 1] += np.bincount(rre[s], minlength=nrow)
                    rows.append(dict(grp=lab, h=h, y=y, model=mn, n=int(i1 - i0),
                                     QL=float(ql[mn].mean()), LM=lm,
                                     medQL=float(np.median(ql[mn])),
                                     kf=kf, klo=kk[0], khi=kk[1], w_h=w_h, w_sh=w_sh,
                                     a_sh=a_sh))
                # trimmed QLIKE: drop rows in the worst 0.1% / 1% of flat94's per-obs loss
                order = np.argsort(ql["flat94"])
                for tag, frac in (("t01", 0.001), ("t1", 0.01)):
                    kp = order[:int(len(order) * (1 - frac))]
                    for mn in MODELS:
                        rows.append(dict(grp=lab, h=h, y=y, model=mn + "|" + tag,
                                         n=len(kp), QL=float(ql[mn][kp].mean()),
                                         LM=np.nan, medQL=np.nan, kf=kf, klo=kk[0], khi=kk[1],
                                         w_h=w_h, w_sh=w_sh, a_sh=a_sh))
        del raw, lrv, ok, yrw, rrw, per
        print(f"  {lab} done [{time.time()-t0:.0f}s]")

    A = pd.DataFrame(rows)

    def wm(g, c="QL"):
        return float(np.average(g[c], weights=g.n))

    print("\n" + "=" * 122)
    print("A1. IS THE QLIKE GAIN JUST A FLOOR ON THE VOL ESTIMATE?  OOS QLIKE, train-only params.")
    print("    floorK = sqrt(max(sigma_ewma^2, k*LR)) -- ONE extra parameter, no horizon logic,")
    print("    no mean reversion.  winsK = two-sided clip.  shrink1w = ONE shrink weight for")
    print("    all h.  ar1 = the claim's model (shrink weight free per h).")
    print("=" * 122)
    for lab in ("INDICES", "SINGLES"):
        g = A[(A.grp == lab) & (~A.model.str.contains(r"\|"))]
        piv = g.groupby(["h", "model"]).apply(wm, include_groups=False).unstack()[MODELS]
        piv["floor_captures_%_of_ar1_gain"] = 100 * (piv.flat94 - piv.floorK) / (piv.flat94 -
                                                                                piv.ar1)
        piv["wins_captures_%"] = 100 * (piv.flat94 - piv.winsK) / (piv.flat94 - piv.ar1)
        piv["1w_captures_%"] = 100 * (piv.flat94 - piv.shrink1w) / (piv.flat94 - piv.ar1)
        print(f"\n  {lab} / OOS QLIKE:")
        print(piv.round(4).to_string())
        print(f"  fitted floor k (median over walk-forward fits) by h: " +
              ", ".join(f"h{h}:{g[g.h==h].kf.median():.3f}" for h in HS))
        print(f"  fitted winsor (klo,khi) medians by h: " +
              ", ".join(f"h{h}:({g[g.h==h].klo.median():.3f},{g[g.h==h].khi.median():.2f})"
                        for h in HS))

    print("\n" + "-" * 122)
    print("A2. THE SAME RACE UNDER logMSE (symmetric in logs, no tail explosion) and under a")
    print("    TRIMMED QLIKE that removes the worst 0.1% / 1% of flat94 observations (identical")
    print("    rows removed for every model).  MEDIAN per-observation QLIKE also shown.")
    print("-" * 122)
    for lab in ("INDICES", "SINGLES"):
        g = A[(A.grp == lab) & (~A.model.str.contains(r"\|"))]
        lm = g.groupby(["h", "model"]).apply(lambda x: wm(x, "LM"),
                                            include_groups=False).unstack()[MODELS]
        lm["ar1_vs_flat%"] = 100 * (1 - lm.ar1 / lm.flat94)
        lm["ar1_vs_flat97%"] = 100 * (1 - lm.ar1 / lm.flat97)
        lm["BEST"] = lm[MODELS].idxmin(axis=1)
        print(f"\n  {lab} / OOS logMSE:")
        print(lm.round(4).to_string())
        md = g.groupby(["h", "model"]).apply(lambda x: wm(x, "medQL"),
                                            include_groups=False).unstack()[MODELS]
        md["ar1_vs_flat%"] = 100 * (1 - md.ar1 / md.flat94)
        print(f"\n  {lab} / OOS MEDIAN per-observation QLIKE:")
        print(md.round(4).to_string())
        for tag, lbl in (("t01", "top 0.1% dropped"), ("t1", "top 1% dropped")):
            gg = A[(A.grp == lab) & (A.model.str.endswith("|" + tag))]
            piv = gg.groupby(["h", "model"]).apply(wm, include_groups=False).unstack()
            piv.columns = [c.split("|")[0] for c in piv.columns]
            piv = piv[MODELS]
            piv["ar1_vs_flat%"] = 100 * (1 - piv.ar1 / piv.flat94)
            piv["floorK_vs_flat%"] = 100 * (1 - piv.floorK / piv.flat94)
            print(f"\n  {lab} / TRIMMED OOS QLIKE ({lbl}):")
            print(piv.round(4).to_string())

    print("\n" + "=" * 122)
    print("D. INDICES/SINGLES tail-spread P(|z|>1.5) Q5-Q1 with a 252-DATE-BLOCK bootstrap")
    print("   (ratios recomputed inside each resample, so it works with only 5 index series).")
    print("=" * 122)
    for lab in ("INDICES", "SINGLES"):
        out = []
        for h in (5, 21, 63):
            for mn in ("flat94", "flat97", "ar1", "ar1q", "floorK"):
                a1, a5 = dh[(lab, h, mn, 0)], dh[(lab, h, mn, 4)]
                v = (a1[:, 1] + a5[:, 1]) > 0
                A1, A5 = a1[v], a5[v]
                sp = A5[:, 0].sum() / A5[:, 1].sum() - A1[:, 0].sum() / A1[:, 1].sum()
                bl = blocks_of(len(A1), 252)
                rg = np.random.default_rng(3)
                bs = []
                for _ in range(2000):
                    ii = np.concatenate([bl[i] for i in rg.integers(0, len(bl), len(bl))])
                    n1, n5 = A1[ii, 1].sum(), A5[ii, 1].sum()
                    if n1 > 0 and n5 > 0:
                        bs.append(A5[ii, 0].sum() / n5 - A1[ii, 0].sum() / n1)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                out.append(dict(h=h, model=mn, spread=sp, lo=lo, hi=hi,
                                zero_in_CI="YES" if lo <= 0 <= hi else "no"))
        print(f"\n  {lab}:")
        print(pd.DataFrame(out).round(4).to_string(index=False))

    # ================================================================ B. split-sample state
    print("\n" + "=" * 122)
    print("B. DECISIVE ARTIFACT TEST.  Two variance estimates from DISJOINT past days:")
    print("   se_ODD  = sqrt(mean r^2 at lags 1,3,...,29)   (15 terms)")
    print("   se_EVEN = sqrt(mean r^2 at lags 2,4,...,30)   (15 terms, independent noise)")
    print("   sl_lag  = vol_cc(252) lagged 31 days (shares no day with either).")
    print("   R = sigma_fwd_h / se_ODD in BOTH columns; only the SORT variable changes:")
    print("     CONTAMINATED sort = se_ODD /sl_lag  (sort shares noise with R's denominator)")
    print("     CLEAN        sort = se_EVEN/sl_lag  (independent noise -> no artifact)")
    print("   Equal estimator precision in both, so CLEAN is the honest state dependence.")
    print("=" * 122)
    r2 = lr.pow(2)
    v_odd = sum(r2.shift(k) for k in range(1, 31, 2)) / 15.0
    v_evn = sum(r2.shift(k) for k in range(2, 31, 2)) / 15.0
    se_o, se_v = np.sqrt(v_odd), np.sqrt(v_evn)
    sl_lag = L.vol_cc(close, 252).shift(31)
    r2s = lr.copy()
    for c in r2s.columns:
        vv = r2s[c].values
        m = np.isfinite(vv)
        x = vv[m].copy()
        RNG.shuffle(x)
        vv[m] = x
    close_sh = np.exp(r2s.cumsum())
    r2sh = np.log(close_sh).diff().pow(2)
    vo_s = sum(r2sh.shift(k) for k in range(1, 31, 2)) / 15.0
    ve_s = sum(r2sh.shift(k) for k in range(2, 31, 2)) / 15.0
    sl_s = L.vol_cc(close_sh, 252).shift(31)
    for tag, (cl, so, sv, sl_) in (("REAL", (close, se_o, se_v, sl_lag)),
                                   ("SHUFFLED", (close_sh, np.sqrt(vo_s), np.sqrt(ve_s), sl_s))):
        print(f"\n  --- {tag} ---")
        for lab, cols in (("INDICES", IDX), ("SINGLES", singles)):
            recs = []
            for h in HS:
                fw = L.realized_vol_forward(cl, h)
                a = so[cols].values.ravel()
                b = sv[cols].values.ravel()
                s = sl_[cols].values.ravel()
                f = fw[cols].values.ravel()
                m = (np.isfinite(a) & (a >= FLOOR) & np.isfinite(b) & (b >= FLOOR)
                     & np.isfinite(s) & (s >= FLOOR) & np.isfinite(f) & (f >= FLOOR))
                a, b, s, f = a[m], b[m], s[m], f[m]
                R = f / a
                row = dict(h=h, n=len(R))
                for nm, sortv in (("contam", a / s), ("clean", b / s)):
                    ed = np.percentile(sortv, [20, 40, 60, 80])
                    q = np.digitize(sortv, ed)
                    rms = np.array([np.sqrt(np.mean(R[q == k] ** 2)) for k in range(5)])
                    ls = np.log(sortv)
                    row[f"{nm}_Q1"] = rms[0]
                    row[f"{nm}_Q5"] = rms[4]
                    row[f"{nm}_Q5/Q1"] = rms[4] / rms[0]
                    row[f"{nm}_drop%"] = 100 * (rms[4] / rms[0] - 1)
                    row[f"{nm}_slope"] = float(np.cov(ls, np.log(R))[0, 1] / np.var(ls))
                recs.append(row)
            t = pd.DataFrame(recs)
            t["artifact_share_of_drop"] = 1 - (t["clean_drop%"] / t["contam_drop%"])
            print(f"   {lab}:")
            print(t.round(4).to_string(index=False))

    # ================================================================ C. 2023 forensics
    print("\n" + "=" * 122)
    print("C. CONCENTRATION FORENSICS: singles, h=63. Where does the QLIKE gain actually come")
    print("   from? Per-year gap, and the share of each year's gap held by its top observations.")
    print("=" * 122)
    ci = [close.columns.get_loc(c) for c in singles]
    h = 63
    SE = np.ascontiguousarray(s_e.iloc[:, ci].values)
    F = np.ascontiguousarray(fwd[h].iloc[:, ci].values)
    LRV = np.ascontiguousarray(lr.iloc[:, ci].values)
    ok = np.isfinite(SE) & (SE >= FLOOR) & np.isfinite(F) & (F >= FLOOR)
    recs = []
    for y in yrs[5:]:
        LRp = float(np.nanmean(LRV[years < y] ** 2))
        tr = ok & (years[:, None] < y)
        te = ok & (years[:, None] == y)
        if tr.sum() < 1500 or te.sum() < 100:
            continue
        f2t, s2t = F[tr] ** 2, SE[tr] ** 2
        sc = [np.log(np.mean(f2t / np.maximum(LRp + (s2t - LRp) * w, FLOOR ** 2)))
              + np.mean(np.log(np.maximum(LRp + (s2t - LRp) * w, FLOOR ** 2))) for w in WGRID]
        w = float(WGRID[int(np.argmin(sc))])
        b_t = np.sqrt(np.maximum(LRp + (s2t - LRp) * w, FLOOR ** 2))
        cf = float(np.mean(f2t / s2t))
        ca = float(np.mean(f2t / b_t ** 2))
        f2e, s2e = F[te] ** 2, SE[te] ** 2
        b_e2 = np.maximum(LRp + (s2e - LRp) * w, FLOOR ** 2)
        d = qlike_el(cf * s2e, f2e) - qlike_el(ca * b_e2, f2e)
        dd = np.sort(d)[::-1]
        n = len(d)
        rr, cc2 = np.nonzero(te)
        top = np.argsort(d)[::-1][:3]
        recs.append(dict(y=y, n=n, mean_gap=d.mean(), median_gap=float(np.median(d)),
                         share_top1obs=dd[0] / d.sum() if d.sum() != 0 else np.nan,
                         share_top10=dd[:10].sum() / d.sum() if d.sum() != 0 else np.nan,
                         share_top_0p1pct=dd[:max(1, n // 1000)].sum() / d.sum()
                         if d.sum() != 0 else np.nan,
                         worst_sym=singles[cc2[top[0]]],
                         worst_date=str(close.index[rr[top[0]]].date()),
                         worst_gap=d[top[0]]))
    T = pd.DataFrame(recs)
    print(T.round(4).to_string(index=False))
    tot = float((T.mean_gap * T.n).sum() / T.n.sum())
    print(f"\n  n-weighted overall mean gap (flat94-ar1) at h=63 singles = {tot:.4f}")
    print(f"  median-of-observations gap is ~{T.median_gap.median():.4f}: the AVERAGE gap is "
          f"{tot/max(T.median_gap.median(),1e-9):.0f}x the median, i.e. the improvement is a "
          f"pure tail effect.")
    sh = (T.mean_gap * T.n)
    print(f"  share of the total gap from the single worst year: "
          f"{sh.max()/sh.sum():.3f} (year {int(T.y[sh.idxmax()])})")
    print(f"  share from the two worst years: "
          f"{sh.nlargest(2).sum()/sh.sum():.3f}")
    print(f"\ndone {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
