"""
_move_wf_volscale3.py — lock down the exact recommended formula, OOS.

From volscale/volscale2:
  - unconditional exponent beta ~ 0.45-0.47, so kappa_h = sigma_h/(sigma_1*sqrt(h)) < 1
  - the AR(1) variance form's "phi" is loss-dependent by 5-10x (long memory => misspecified),
    while the log-space shrink weight a_h is stable and near-linear in log(h)
This script validates the closed-form spec  a_h = A - B*ln(h)  against free-per-h and against
ar1+quintile, all fit on strictly prior years, and checks kappa_h stability across sub-periods.

Run: ../../vcp_env/bin/python _move_wf_volscale3.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import _move_data as D
import _move_lib as L

HS_ALL = [1, 2, 3, 5, 10, 21, 42, 63]
HS = [2, 3, 5, 10, 21, 42, 63]
IDX = ["SPY", "QQQ", "^GSPC", "IWM", "DIA"]
MIN_OBS, FLOOR = 500, 1e-4
LAM, LONG_N, MID_N = 0.94, 252, 22
RNG = np.random.default_rng(23)
PHI = np.round(np.concatenate([np.arange(0.70, 0.99, 0.005),
                               np.arange(0.990, 0.9996, 0.0005)]), 4)
pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 500)


def pf(phi, h):
    return 1.0 if phi >= 1 else float(phi * (1 - phi ** h) / (h * (1 - phi)))


def qlike(a, f):
    x = np.maximum(f, FLOOR) ** 2 / np.maximum(a, FLOOR) ** 2
    return float(np.mean(x - np.log(x) - 1))


def logmse(a, f):
    return float(np.mean((np.log(np.maximum(a, FLOOR)) - np.log(np.maximum(f, FLOOR))) ** 2))


def hstd(x, h):
    x = pd.Series(x).dropna()
    if len(x) < 5 * h + 30:
        return np.nan
    cs = x.cumsum()
    rh = (cs - cs.shift(h)).dropna().values - h * x.mean()
    return float(np.sqrt((rh ** 2).sum() / (len(rh) - 1)))


def main():
    t0 = time.time()
    p = D.load()
    close = p["Close"]
    keep = [s for s in close.columns if close[s].notna().sum() >= MIN_OBS and s != "^VIX"]
    close = close[keep]
    singles = [s for s in keep if s not in IDX]
    lr = np.log(close).diff()
    years = close.index.year.values
    yrs = sorted(set(years))

    s_e, s_m, s_l = L.vol_ewma(close, LAM), L.vol_cc(close, MID_N), L.vol_cc(close, LONG_N)
    state = s_e / s_l
    fwd = {h: L.realized_vol_forward(close, h) for h in HS}
    fret = {h: np.log(close.shift(-h) / close) for h in HS}
    print(f"panel {len(keep)} syms ({len(singles)} singles), {len(close)} rows")

    # ================================================== 1. kappa_h stability + CIs
    print("\n" + "=" * 110)
    print("1. kappa_h = sigma_h / (sigma_1*sqrt(h)) -- SUB-PERIOD STABILITY")
    print("   (sigma_h = sd of overlapping h-day log returns; kappa<1 => sqrt(h) OVERSTATES)")
    print("=" * 110)
    periods = [("full 2001-2026", np.ones(len(close), bool)),
               ("2001-2013", years < 2014), ("2014-2026", years >= 2014)]
    rows = []
    for pname, msk in periods:
        for s in IDX:
            x = lr[s][msk]
            sds = [hstd(x, h) for h in HS_ALL]
            rows.append(dict(period=pname, unit=s, n=int(x.notna().sum()),
                             **{f"k{h}": sds[i] / (sds[0] * np.sqrt(h))
                                for i, h in enumerate(HS_ALL)}))
        pv = []
        for h in HS_ALL:
            v = [hstd(lr[s][msk], h) ** 2 for s in singles if np.isfinite(hstd(lr[s][msk], h))]
            pv.append(np.sqrt(np.mean(v)))
        rows.append(dict(period=pname, unit="POOLED_SINGLES", n=int(msk.sum()),
                         **{f"k{h}": pv[i] / (pv[0] * np.sqrt(h)) for i, h in enumerate(HS_ALL)}))
    kt = pd.DataFrame(rows)
    print(kt.round(4).to_string(index=False))
    # symbol-clustered bootstrap CI for pooled singles kappa (full period)
    perk = {}
    for s in singles:
        sds = [hstd(lr[s], h) for h in HS_ALL]
        if all(np.isfinite(sds)):
            perk[s] = [sds[i] / (sds[0] * np.sqrt(h)) for i, h in enumerate(HS_ALL)]
    K = np.array(list(perk.values()))
    print(f"\nper-symbol kappa_h, singles (n={len(K)} symbols). "
          f"95% CI on the mean = bootstrap clustered by SYMBOL:")
    for i, h in enumerate(HS_ALL):
        bs = [K[RNG.integers(0, len(K), len(K)), i].mean() for _ in range(4000)]
        print(f"  h={h:>2}: mean kappa {K[:, i].mean():.4f}  median {np.median(K[:, i]):.4f}  "
              f"95% CI [{np.percentile(bs,2.5):.4f}, {np.percentile(bs,97.5):.4f}]  "
              f"share<1 {(K[:, i] < 1).mean():.3f}")
    # index kappa CI, block bootstrap on 252-day blocks, per h, SPY + GSPC
    print("\nindex kappa_h 95% CI (252-row block bootstrap, 800 reps):")
    for s in ["SPY", "^GSPC"]:
        x = lr[s].dropna().values
        bl = [np.arange(i, min(i + 252, len(x))) for i in range(0, len(x), 252)]
        samp = []
        for _ in range(800):
            pk = RNG.integers(0, len(bl), len(bl))
            xb = np.concatenate([x[bl[i]] for i in pk])
            sds = [hstd(xb, h) for h in HS_ALL]
            samp.append([sds[i] / (sds[0] * np.sqrt(h)) for i, h in enumerate(HS_ALL)])
        samp = np.array(samp)
        print(f"  {s}: " + "  ".join(
            f"h{h}:{np.percentile(samp[:,i],2.5):.3f}-{np.percentile(samp[:,i],97.5):.3f}"
            for i, h in enumerate(HS_ALL)))

    # ================================================== 2. final horse race
    print("\n" + "=" * 110)
    print("2. FINAL OOS HORSE RACE (expanding walk-forward by year, min 5y train)")
    print("   per-day sigma forecast for horizon h; sigma_h_hat = c_h * base_h * sqrt(h)")
    print("   flat      : s_e")
    print("   ar1       : sqrt(LR + (s_e^2-LR)*pf(phi,h)),  phi grid-searched on train per h")
    print("   ar1+q     : ar1 x rms quintile correction (train)")
    print("   blendFree : s_e^a * s_252^(1-a),  a free per h (train OLS, CONSTRAINED sum=1)")
    print("   blendAB   : s_e^a * s_252^(1-a),  a = A - B*ln(h)  (A,B fit on train, 2 params)")
    print("   blendAB3  : s_e^a * s_22^b * s_252^(1-a-b), a,b = closed form in ln(h) (4 params)")
    print("   blendABq  : blendAB x rms quintile correction (train)")
    print("=" * 110)
    MODELS = ["flat", "ar1", "ar1+q", "blendFree", "blendAB", "blendAB3", "blendABq"]
    rows, sdz, prm_rows = [], [], []
    for lab, cols in [("INDICES", IDX), ("SINGLES", singles)]:
        ci = [close.columns.get_loc(c) for c in cols]
        SE = s_e.iloc[:, ci].values
        SM = s_m.iloc[:, ci].values
        SL = s_l.iloc[:, ci].values
        ST = state.iloc[:, ci].values
        F = {h: fwd[h].iloc[:, ci].values for h in HS}
        RR = {h: fret[h].iloc[:, ci].values for h in HS}
        good = {h: (np.isfinite(F[h]) & (F[h] >= FLOOR) & np.isfinite(RR[h]) & np.isfinite(SE)
                    & (SE >= FLOOR) & np.isfinite(SM) & (SM >= FLOOR) & np.isfinite(SL)
                    & (SL >= FLOOR)) for h in HS}
        yrow = np.repeat(years[:, None], len(ci), 1)
        for y in yrs[5:]:
            LR = float(np.nanmean(lr.iloc[:, ci].values[years < y] ** 2))
            trm = {h: good[h] & (yrow < y) for h in HS}
            tem = {h: good[h] & (yrow == y) for h in HS}
            if min(trm[h].sum() for h in HS) < 1500 or min(tem[h].sum() for h in HS) < 100:
                continue
            # ---- fits on TRAIN
            a_free, phi_h, cq_h, cqAB_h, edges_h = {}, {}, {}, {}, {}
            for h in HS:
                m = trm[h]
                le, ll, lf = np.log(SE[m]), np.log(SL[m]), np.log(F[h][m])
                # constrained blend: log f - log sl = a*(log se - log sl) + const
                Xc = np.column_stack([le - ll, np.ones(m.sum())])
                bc, *_ = np.linalg.lstsq(Xc, lf - ll, rcond=None)
                a_free[h] = float(bc[0])
                q_ = np.empty(len(PHI))
                for i, phi in enumerate(PHI):
                    sh = np.sqrt(np.maximum(LR + (SE[m] ** 2 - LR) * pf(phi, h), FLOOR ** 2))
                    q_[i] = qlike(sh * np.sqrt(np.mean(F[h][m] ** 2 / sh ** 2)), F[h][m])
                phi_h[h] = PHI[int(np.nanargmin(q_))]
                edges_h[h] = np.percentile(ST[m], [20, 40, 60, 80])
            # A,B from OLS of a_free(h) on ln(h)  (train-derived, no test data)
            Xh = np.column_stack([np.log(HS), np.ones(len(HS))])
            ba, *_ = np.linalg.lstsq(Xh, np.array([a_free[h] for h in HS]), rcond=None)
            B, A = -float(ba[0]), float(ba[1])
            # 3-window closed form: fit a(h), b(h) each linear in ln h from free per-h 3-var OLS
            af, bf = {}, {}
            for h in HS:
                m = trm[h]
                le, lm_, ll, lf = np.log(SE[m]), np.log(SM[m]), np.log(SL[m]), np.log(F[h][m])
                X3 = np.column_stack([le - ll, lm_ - ll, np.ones(m.sum())])
                b3, *_ = np.linalg.lstsq(X3, lf - ll, rcond=None)
                af[h], bf[h] = float(b3[0]), float(b3[1])
            ba3, *_ = np.linalg.lstsq(Xh, np.array([af[h] for h in HS]), rcond=None)
            bb3, *_ = np.linalg.lstsq(Xh, np.array([bf[h] for h in HS]), rcond=None)

            def bases(h, m):
                se, sm_, sl, st = SE[m], SM[m], SL[m], ST[m]
                q = np.digitize(st, edges_h[h])
                o = {}
                o["flat"] = se
                o["ar1"] = np.sqrt(np.maximum(LR + (se ** 2 - LR) * pf(phi_h[h], h), FLOOR ** 2))
                o["blendFree"] = np.exp(a_free[h] * np.log(se) + (1 - a_free[h]) * np.log(sl))
                aAB = A - B * np.log(h)
                o["blendAB"] = np.exp(aAB * np.log(se) + (1 - aAB) * np.log(sl))
                a3 = ba3[0] * np.log(h) + ba3[1]
                b3v = bb3[0] * np.log(h) + bb3[1]
                o["blendAB3"] = np.exp(a3 * np.log(se) + b3v * np.log(sm_)
                                       + (1 - a3 - b3v) * np.log(sl))
                return o, q

            for h in HS:
                bt, qt = bases(h, trm[h])
                be, qe = bases(h, tem[h])
                ft, fe = F[h][trm[h]], F[h][tem[h]]
                rt, re = RR[h][trm[h]], RR[h][tem[h]]
                # quintile corrections on top of ar1 and blendAB (train, rms, shape-only)
                for src, dst in [("ar1", "ar1+q"), ("blendAB", "blendABq")]:
                    R = ft / np.maximum(bt[src], FLOOR)
                    cq = np.array([np.sqrt(np.mean(R[qt == k] ** 2)) if (qt == k).sum() > 30
                                   else 1.0 for k in range(5)])
                    cq = cq / np.sqrt(np.mean(R ** 2))
                    bt[dst] = bt[src] * cq[qt]
                    be[dst] = be[src] * cq[qe]
                    if dst == "ar1+q":
                        cq_h[h] = cq
                    else:
                        cqAB_h[h] = cq
                for mn in MODELS:
                    bbt, bbe = np.maximum(bt[mn], FLOOR), np.maximum(be[mn], FLOOR)
                    c_ql = np.sqrt(np.mean(ft ** 2 / bbt ** 2))
                    c_lg = float(np.exp(np.mean(np.log(ft) - np.log(bbt))))
                    c_z = np.sqrt(np.mean(rt ** 2 / (h * bbt ** 2)))
                    rows.append(dict(grp=lab, h=h, y=y, model=mn, n=int(len(fe)),
                                     QL=qlike(bbe * c_ql, fe), LM=logmse(bbe * c_lg, fe)))
                    z = re / (bbe * c_z * np.sqrt(h))
                    for k in range(5):
                        s = qe == k
                        if s.sum() < 20:
                            continue
                        sdz.append(dict(grp=lab, h=h, y=y, model=mn, q=k + 1, n=int(s.sum()),
                                        sq=float(np.mean(z[s] ** 2)),
                                        hit=float(np.mean(np.abs(z[s]) > 1.5)),
                                        hit2=float(np.mean(np.abs(z[s]) > 2.0))))
            prm_rows.append(dict(grp=lab, y=y, A=A, B=B, a3_slope=ba3[0], a3_int=ba3[1],
                                 b3_slope=bb3[0], b3_int=bb3[1], LRsig=np.sqrt(LR),
                                 **{f"a_free_{h}": a_free[h] for h in HS},
                                 **{f"phi_{h}": phi_h[h] for h in HS}))
        print(f"  {lab} done [{time.time()-t0:.0f}s]")

    A_ = pd.DataFrame(rows)
    S_ = pd.DataFrame(sdz)
    P_ = pd.DataFrame(prm_rows)

    for lab in ("INDICES", "SINGLES"):
        g = A_[A_.grp == lab]
        for met, nm in (("QL", "QLIKE"), ("LM", "logMSE")):
            piv = g.groupby(["h", "model"]).apply(
                lambda x: np.average(x[met], weights=x.n), include_groups=False).unstack()[MODELS]
            piv["best"] = piv.idxmin(axis=1)
            piv["vs_flat%"] = 100 * (1 - piv[MODELS].min(axis=1) / piv["flat"])
            piv["n_obs"] = g.groupby("h").n.sum() // len(MODELS)
            print(f"\n  {lab} / {nm} OOS (lower better):")
            print(piv.round(4).to_string())
        gs = S_[S_.grp == lab]
        summ = []
        for mn in MODELS:
            gg = gs[gs.model == mn]
            piv = gg.groupby(["h", "q"]).apply(
                lambda x: np.sqrt(np.average(x.sq, weights=x.n)), include_groups=False).unstack()
            hv = gg.groupby(["h", "q"]).apply(
                lambda x: np.average(x.hit, weights=x.n), include_groups=False).unstack()
            summ.append(dict(model=mn,
                             **{f"sdz_Q5/Q1_h{h}": piv.loc[h].iloc[4] / piv.loc[h].iloc[0]
                                for h in (2, 21, 63)},
                             mean_abs_sdz_dev=float((piv - 1).abs().values.mean()),
                             **{f"hitsprd_h{h}": hv.loc[h].iloc[4] - hv.loc[h].iloc[0]
                                for h in (2, 21, 63)},
                             mean_abs_hitsprd=float(np.mean(
                                 [abs(hv.loc[h].iloc[4] - hv.loc[h].iloc[0]) for h in HS]))))
        print(f"\n  {lab} / z-calibration OOS: sd(z) should be 1 in every quintile, "
              f"P(|z|>1.5) spread Q5-Q1 should be 0:")
        print(pd.DataFrame(summ).round(4).to_string(index=False))
        # full sd(z) table for the two front-runners
        for mn in ("flat", "blendAB", "blendABq", "ar1+q"):
            gg = gs[gs.model == mn]
            piv = gg.groupby(["h", "q"]).apply(
                lambda x: np.sqrt(np.average(x.sq, weights=x.n)), include_groups=False).unstack()
            piv.columns = [f"Q{c}" for c in piv.columns]
            piv["Q5/Q1"] = piv.Q5 / piv.Q1
            print(f"\n    {lab} / {mn}: sd(z) by quintile")
            print(piv.round(4).to_string())

    print("\n" + "=" * 110)
    print("3. RECOMMENDED PARAMETERS (median over the 21 walk-forward fits; each fit used only")
    print("   data strictly prior to its test year, so these are the values a live system sees)")
    print("=" * 110)
    for lab in ("INDICES", "SINGLES"):
        g = P_[P_.grp == lab]
        print(f"\n  {lab}: a_h = A - B*ln(h)   A={g.A.median():.4f} [{g.A.min():.4f},"
              f"{g.A.max():.4f}]   B={g.B.median():.4f} [{g.B.min():.4f},{g.B.max():.4f}]"
              f"   LRsigma={g.LRsig.median():.5f}/day")
        t = pd.DataFrame({
            "a_free_median": [g[f"a_free_{h}"].median() for h in HS],
            "a_free_min": [g[f"a_free_{h}"].min() for h in HS],
            "a_free_max": [g[f"a_free_{h}"].max() for h in HS],
            "a_AB_formula": [g.A.median() - g.B.median() * np.log(h) for h in HS],
            "phi_qlike_median": [g[f"phi_{h}"].median() for h in HS],
            "HL_qlike_days": [np.log(.5) / np.log(g[f"phi_{h}"].median()) for h in HS],
        }, index=HS)
        t.index.name = "h"
        print(t.round(4).to_string())
        print(f"  3-window closed form: a(h) = {g.a3_int.median():.4f} "
              f"{g.a3_slope.median():+.4f}*ln(h);  b(h) = {g.b3_int.median():.4f} "
              f"{g.b3_slope.median():+.4f}*ln(h);  weight on s_252 = 1-a-b")
    print(f"\ndone {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
