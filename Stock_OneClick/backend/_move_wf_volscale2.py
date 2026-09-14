"""
_move_wf_volscale2.py — decisive horse race for the horizon-scaling formula.

Follow-up to _move_wf_volscale.py. Three things that first pass showed need resolving:
  A) mean(R) and rms(R) corrections disagree a lot (fat right tail of forward vol from calm
     states), and it is the RMS/variance-matched one that governs P(|move|>2%) calibration.
  B) the required correction for POOLED SINGLES is nearly h-INDEPENDENT (~1.21/0.94 at h=2 AND
     h=63) => most of it is EWMA over-reacting to single-stock jumps, not AR(1) path decay.
     For INDICES the correction GROWS with h => genuine mean reversion. Must separate the two.
  C) a single AR(1) cannot match the observed shrink-weight curve w_h at both short and long h
     (long memory). Test a log-blend (shrink-in-log-space) form against it.

Every parameter fit on strictly prior years. Metrics: QLIKE, logMSE, and the thing that actually
matters for this project -- sd(z) and tail-hit rate of z = r_h / sigma_h_hat, BY STATE QUINTILE.

Run: ../../vcp_env/bin/python _move_wf_volscale2.py
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
LAM, LONG_N, MID_N = 0.94, 252, 22
RNG = np.random.default_rng(11)
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


def blocks_of(n, b=252):
    return [np.arange(s, min(s + b, n)) for s in range(0, n, b)]


def bb_ci(v, b=252, n_boot=1500, stat=np.mean):
    v = np.asarray(v, float)
    bl = blocks_of(len(v), b)
    if len(bl) < 3:
        return (np.nan, np.nan)
    o = []
    for _ in range(n_boot):
        pk = RNG.integers(0, len(bl), len(bl))
        x = v[np.concatenate([bl[i] for i in pk])]
        x = x[np.isfinite(x)]
        if len(x):
            o.append(stat(x))
    return tuple(np.percentile(o, [2.5, 97.5]))


def main():
    t0 = time.time()
    p = D.load()
    close, high, low, opn = p["Close"], p["High"], p["Low"], p["Open"]
    keep = [s for s in close.columns if close[s].notna().sum() >= MIN_OBS and s != "^VIX"]
    close = close[keep]
    singles = [s for s in keep if s not in IDX]
    lr = np.log(close).diff()

    s_e = L.vol_ewma(close, LAM)
    s_m = L.vol_cc(close, MID_N)
    s_l = L.vol_cc(close, LONG_N)
    s_yz = L.vol_yang_zhang(opn[keep], high[keep], low[keep], close, 10)
    state = s_e / s_l
    fwd = {h: L.realized_vol_forward(close, h) for h in HS}
    fret = {h: np.log(close.shift(-h) / close) for h in HS}

    years = close.index.year.values
    nrow, ncol = close.shape
    print(f"panel: {len(keep)} symbols ({len(IDX)} indices + {len(singles)} singles), "
          f"{nrow} rows {close.index[0].date()}->{close.index[-1].date()}   [{time.time()-t0:.0f}s]")

    # ---- flatten once, per group -------------------------------------------------------------
    def flatten(cols):
        ci = [close.columns.get_loc(c) for c in cols]
        Y = np.repeat(years[:, None], len(ci), 1).ravel()
        Dt = np.repeat(np.arange(nrow)[:, None], len(ci), 1).ravel()
        base = dict(year=Y, dt=Dt)
        for nm, df in [("se", s_e), ("sm", s_m), ("sl", s_l), ("st", state), ("yz", s_yz)]:
            base[nm] = df.iloc[:, ci].values.ravel().astype(np.float64)
        per_h = {}
        for h in HS:
            f = fwd[h].iloc[:, ci].values.ravel().astype(np.float64)
            r = fret[h].iloc[:, ci].values.ravel().astype(np.float64)
            m = (np.isfinite(f) & (f >= FLOOR) & np.isfinite(r) & np.isfinite(base["se"])
                 & (base["se"] >= FLOOR) & np.isfinite(base["sm"]) & (base["sm"] >= FLOOR)
                 & np.isfinite(base["sl"]) & (base["sl"] >= FLOOR) & np.isfinite(base["yz"])
                 & (base["yz"] >= FLOOR))
            per_h[h] = dict(f=f[m], r=r[m], idx=np.flatnonzero(m))
        return base, per_h

    G = {}
    for lab, cols in [("INDICES", IDX), ("SINGLES", singles)]:
        G[lab] = flatten(cols)
        print(f"  {lab}: n rows per h = " +
              ", ".join(f"h{h}:{len(G[lab][1][h]['f']):,}" for h in HS))

    # ============================================================ A. four aggregations
    print("\n" + "=" * 112)
    print("A. CONDITIONAL SCALING -- FOUR AGGREGATIONS of R = realized_fwd_perday / vol_ewma_now")
    print("   (R == realized h-day vol / (current daily vol * sqrt(h)); the sqrt(h) cancels)")
    print("   mean(R)   : average correction")
    print("   median(R) : robust / log-space correction  <- what a log-MSE vol forecast targets")
    print("   rms(R)    : sqrt(E[R^2])  <- the variance-matching correction that makes sd(z)=1")
    print("   vmatch    : sqrt( E[sigma_fwd^2] / E[sigma_now^2] ) within quintile")
    print("   Quintile edges from strictly prior years (expanding, min 5y train).")
    print("=" * 112)
    for lab in ("INDICES", "SINGLES"):
        base, per_h = G[lab]
        recs = []
        for h in HS:
            d = per_h[h]
            yy, st = base["year"][d["idx"]], base["st"][d["idx"]]
            se = base["se"][d["idx"]]
            R = d["f"] / se
            for y in sorted(set(yy)):
                if y < sorted(set(years))[5]:
                    continue
                tr, te = yy < y, yy == y
                if tr.sum() < 1000 or te.sum() < 50:
                    continue
                ed = np.percentile(st[tr], [20, 40, 60, 80])
                q = np.digitize(st[te], ed)
                Rt, ft, st_ = R[te], d["f"][te], se[te]
                for k in range(5):
                    s = q == k
                    if s.sum() < 20:
                        continue
                    recs.append(dict(h=h, q=k + 1, y=y, n=int(s.sum()),
                                     mean=Rt[s].mean(), med=np.median(Rt[s]),
                                     rms=np.sqrt((Rt[s] ** 2).mean()),
                                     vm=np.sqrt((ft[s] ** 2).mean() / (st_[s] ** 2).mean())))
        t = pd.DataFrame(recs)
        w = t.groupby(["h", "q"]).apply(
            lambda g: pd.Series({m: np.average(g[m], weights=g.n) for m in
                                 ("mean", "med", "rms", "vm")} | {"n": g.n.sum()}),
            include_groups=False).reset_index()
        print(f"\n--- {lab} --- (n-weighted across walk-forward test years)")
        for m in ("mean", "med", "rms", "vm"):
            piv = w.pivot(index="h", columns="q", values=m)
            piv.columns = [f"Q{c}" for c in piv.columns]
            piv["Q5/Q1"] = piv.Q5 / piv.Q1
            # normalise to the row's own all-quintile n-weighted level
            lvl = w.groupby("h").apply(lambda g: np.average(g[m], weights=g.n),
                                       include_groups=False)
            rel = piv[[f"Q{k}" for k in range(1, 6)]].div(lvl, axis=0)
            rel.columns = [f"rel_Q{k}" for k in range(1, 6)]
            print(f"\n  {m}(R):")
            print(pd.concat([piv, rel], axis=1).round(4).to_string())
        print("\n  n per (h,q):")
        print(w.pivot(index="h", columns="q", values="n").astype(int).to_string())

    # ============================================================ B. horse race
    print("\n" + "=" * 112)
    print("B. HORSE RACE of horizon-scaling rules. All params fit on strictly prior years")
    print("   (expanding walk-forward, min 5y train). sigma_h_hat = c * base_h(t) * sqrt(h).")
    print("   M1 flat    : base = s_ewma                       (textbook sqrt(h))")
    print("   M2 ar1     : base = sqrt(LR + (s_e^2-LR)*pf(phi,h)),  phi grid-searched on train")
    print("   M3 blend2  : base = s_e^a * s_252^b              (OLS in logs on train)")
    print("   M4 blend3  : base = s_e^a * s_22^b * s_252^d     (OLS in logs on train)")
    print("   M5 quint   : base = s_e * corr(q,h)              (rms lookup, train quintiles)")
    print("   M6 ar1+q   : M2 then x corr_resid(q,h)           (AR1 plus residual quintile fix)")
    print("   M7 yzflat  : base = s_yang_zhang(10)             (estimator control, no h-scaling)")
    print("   Scale c refit per metric on train: QLIKE->rms of f/base, logMSE->geo mean,")
    print("   z-metrics->rms of r_h/(sqrt(h)*base).  So all rules compete on SHAPE only.")
    print("=" * 112)

    MODELS = ["flat", "ar1", "blend2", "blend3", "quint", "ar1+q", "yzflat"]

    # long-run variance per (group, test year), from TRAIN log returns of that group only
    LRcache = {}
    for lab, cols in [("INDICES", IDX), ("SINGLES", singles)]:
        sub = lr[cols].values
        for y in sorted(set(years))[5:]:
            LRcache[(lab, y)] = float(np.nanmean(sub[years < y] ** 2))

    def build_bases(b, d, h, prm):
        se, sm, sl, yz = (b["se"][d["idx"]], b["sm"][d["idx"]],
                          b["sl"][d["idx"]], b["yz"][d["idx"]])
        st = b["st"][d["idx"]]
        q = np.digitize(st, prm["edges"])
        out = {}
        out["flat"] = se
        out["ar1"] = np.sqrt(np.maximum(prm["LR"] + (se ** 2 - prm["LR"]) * pf(prm["phi"], h),
                                        FLOOR ** 2))
        out["blend2"] = np.exp(prm["b2"][0] * np.log(se) + prm["b2"][1] * np.log(sl))
        out["blend3"] = np.exp(prm["b3"][0] * np.log(se) + prm["b3"][1] * np.log(sm)
                               + prm["b3"][2] * np.log(sl))
        out["quint"] = se * prm["cq"][q]
        out["ar1+q"] = out["ar1"] * prm["cqr"][q]
        out["yzflat"] = yz
        return out, q

    all_rows, sdz_rows, hit_rows, param_rows = [], [], [], []
    yrs = sorted(set(years))
    for lab in ("INDICES", "SINGLES"):
        b, per_h = G[lab]
        for h in HS:
            d = per_h[h]
            yy = b["year"][d["idx"]]
            f, r = d["f"], d["r"]
            se = b["se"][d["idx"]]
            st = b["st"][d["idx"]]
            for y in yrs[5:]:
                tr, te = yy < y, yy == y
                if tr.sum() < 1500 or te.sum() < 100:
                    continue
                # ---------------- fit everything on TRAIN only
                # long-run daily variance from TRAIN log returns of this group only
                LR = LRcache[(lab, y)]
                edges = np.percentile(st[tr], [20, 40, 60, 80])
                # phi by QLIKE grid search
                q_ = np.empty(len(PHI))
                for i, phi in enumerate(PHI):
                    sh = np.sqrt(np.maximum(LR + (se[tr] ** 2 - LR) * pf(phi, h), FLOOR ** 2))
                    c = np.sqrt(np.mean(f[tr] ** 2 / sh ** 2))
                    q_[i] = qlike(sh * c, f[tr])
                phi = PHI[int(np.nanargmin(q_))]
                # log-blends by OLS on train
                X2 = np.column_stack([np.log(se[tr]), np.log(b["sl"][d["idx"]][tr]),
                                      np.ones(tr.sum())])
                c2, *_ = np.linalg.lstsq(X2, np.log(f[tr]), rcond=None)
                X3 = np.column_stack([np.log(se[tr]), np.log(b["sm"][d["idx"]][tr]),
                                      np.log(b["sl"][d["idx"]][tr]), np.ones(tr.sum())])
                c3, *_ = np.linalg.lstsq(X3, np.log(f[tr]), rcond=None)
                qtr = np.digitize(st[tr], edges)
                Rtr = f[tr] / se[tr]
                cq = np.array([np.sqrt(np.mean(Rtr[qtr == k] ** 2)) if (qtr == k).sum() > 30
                               else 1.0 for k in range(5)])
                prm = dict(LR=LR, phi=phi, edges=edges, b2=c2[:2], b3=c3[:3], cq=cq,
                           cqr=np.ones(5))
                bt, _ = build_bases(b, dict(idx=d["idx"][tr]), h, prm)
                # residual quintile correction on top of AR(1), from train
                Rar = f[tr] / bt["ar1"]
                cqr = np.array([np.sqrt(np.mean(Rar[qtr == k] ** 2)) if (qtr == k).sum() > 30
                                else 1.0 for k in range(5)])
                cqr = cqr / np.sqrt(np.mean(Rar ** 2))       # keep level in c, shape here
                prm["cqr"] = cqr
                bt, _ = build_bases(b, dict(idx=d["idx"][tr]), h, prm)
                C = {}
                for mn in MODELS:
                    bb = np.maximum(bt[mn], FLOOR)
                    C[mn] = dict(
                        ql=np.sqrt(np.mean(f[tr] ** 2 / bb ** 2)),
                        lg=float(np.exp(np.mean(np.log(f[tr]) - np.log(bb)))),
                        z=np.sqrt(np.mean(r[tr] ** 2 / (h * bb ** 2))))
                # train tail threshold k such that P(|z|>k)=0.20 in train (flat-scaled z)
                # ---------------- score on TEST
                bte, qte = build_bases(b, dict(idx=d["idx"][te]), h, prm)
                for mn in MODELS:
                    bb = np.maximum(bte[mn], FLOOR)
                    all_rows.append(dict(grp=lab, h=h, y=y, model=mn, n=int(te.sum()),
                                         QL=qlike(bb * C[mn]["ql"], f[te]),
                                         LM=logmse(bb * C[mn]["lg"], f[te])))
                    z = r[te] / (bb * C[mn]["z"] * np.sqrt(h))
                    for k in range(5):
                        s = qte == k
                        if s.sum() < 20:
                            continue
                        sdz_rows.append(dict(grp=lab, h=h, y=y, model=mn, q=k + 1,
                                             n=int(s.sum()), sq=float(np.mean(z[s] ** 2)),
                                             absz=float(np.mean(np.abs(z[s]))),
                                             hit=float(np.mean(np.abs(z[s]) > 1.5))))
                param_rows.append(dict(grp=lab, h=h, y=y, phi=phi, hl=np.log(.5) / np.log(phi),
                                       a_e2=c2[0], a_l2=c2[1], sum2=c2[0] + c2[1],
                                       a_e3=c3[0], a_m3=c3[1], a_l3=c3[2],
                                       sum3=c3[0] + c3[1] + c3[2],
                                       cq=cq, LRsig=np.sqrt(LR)))
        print(f"  {lab} done [{time.time()-t0:.0f}s]")

    A = pd.DataFrame(all_rows)
    S = pd.DataFrame(sdz_rows)
    P = pd.DataFrame(param_rows)

    print("\n--- B1. OOS loss by model (n-weighted over walk-forward years) ---")
    for lab in ("INDICES", "SINGLES"):
        for met in ("QL", "LM"):
            g = A[A.grp == lab]
            piv = g.groupby(["h", "model"]).apply(
                lambda x: np.average(x[met], weights=x.n), include_groups=False).unstack()
            piv = piv[MODELS]
            piv["best"] = piv.idxmin(axis=1)
            piv["flat->best %"] = 100 * (1 - piv[MODELS].min(axis=1) / piv["flat"])
            nn = g.groupby("h").n.sum() / len(MODELS)
            piv["n_obs"] = (nn * 1).astype(int)
            print(f"\n  {lab} / {'QLIKE' if met=='QL' else 'logMSE'} (lower better):")
            print(piv.round(4).to_string())

    print("\n--- B2. sd(z) by state quintile, OOS (rms of z; 1.0 = perfectly scaled) ---")
    for lab in ("INDICES", "SINGLES"):
        g = S[S.grp == lab]
        for mn in MODELS:
            gg = g[g.model == mn]
            piv = gg.groupby(["h", "q"]).apply(
                lambda x: np.sqrt(np.average(x.sq, weights=x.n)),
                include_groups=False).unstack()
            piv.columns = [f"Q{c}" for c in piv.columns]
            piv["Q5/Q1"] = piv.Q5 / piv.Q1
            piv["mean|sd-1|"] = (piv[[f"Q{k}" for k in range(1, 6)]] - 1).abs().mean(axis=1)
            print(f"\n  {lab} / {mn}:")
            print(piv.round(4).to_string())
        # summary
        summ = []
        for mn in MODELS:
            gg = g[g.model == mn]
            piv = gg.groupby(["h", "q"]).apply(
                lambda x: np.sqrt(np.average(x.sq, weights=x.n)),
                include_groups=False).unstack()
            summ.append(dict(model=mn,
                             **{f"disp_h{h}": piv.loc[h].iloc[4] / piv.loc[h].iloc[0]
                                for h in HS},
                             mean_abs_dev=float((piv - 1).abs().values.mean())))
        print(f"\n  {lab} SUMMARY -- sd(z) Q5/Q1 dispersion by h (1.0 = state-neutral) "
              f"and overall |sd(z)-1|:")
        print(pd.DataFrame(summ).round(4).to_string(index=False))

    print("\n--- B3. tail-hit rate P(|z|>1.5) by state quintile, OOS "
          "(should be state-independent) ---")
    for lab in ("INDICES", "SINGLES"):
        g = S[S.grp == lab]
        rows = []
        for mn in MODELS:
            gg = g[g.model == mn]
            piv = gg.groupby(["h", "q"]).apply(
                lambda x: np.average(x.hit, weights=x.n), include_groups=False).unstack()
            rows.append(dict(model=mn, **{f"h{h}_Q1": piv.loc[h].iloc[0] for h in (5, 21, 63)},
                             **{f"h{h}_Q5": piv.loc[h].iloc[4] for h in (5, 21, 63)},
                             **{f"h{h}_sprd": piv.loc[h].iloc[4] - piv.loc[h].iloc[0]
                                for h in (5, 21, 63)}))
        print(f"\n  {lab}:")
        print(pd.DataFrame(rows).round(4).to_string(index=False))

    # ============================================================ C. parameters / half-life
    print("\n" + "=" * 112)
    print("C. FITTED PARAMETERS AND HALF-LIFE, by loss function")
    print("=" * 112)
    for lab in ("INDICES", "SINGLES"):
        g = P[P.grp == lab]
        t = g.groupby("h").agg(phi_med=("phi", "median"), phi_lo=("phi", "min"),
                               phi_hi=("phi", "max"), a_e2=("a_e2", "median"),
                               a_l2=("a_l2", "median"), sum2=("sum2", "median"),
                               a_e3=("a_e3", "median"), a_m3=("a_m3", "median"),
                               a_l3=("a_l3", "median"), sum3=("sum3", "median"),
                               n_years=("y", "count"))
        t["HL_qlike_d"] = np.log(.5) / np.log(t.phi_med)
        # phi implied by the log-blend weight a_e2 (matched to pf(phi,h))
        t["phi_from_a_e2"] = [PHI[int(np.argmin([abs(pf(pp, h) - a) for pp in PHI]))]
                              for h, a in zip(t.index, t.a_e2)]
        t["HL_logblend_d"] = np.log(.5) / np.log(t.phi_from_a_e2)
        print(f"\n  {lab} (median across walk-forward years):")
        print(t.round(4).to_string())
        # single phi that best fits the whole a_e2(h) curve
        aa = t.a_e2.values
        err = [np.sum([(pf(pp, h) - a) ** 2 for h, a in zip(t.index, aa)]) for pp in PHI]
        best = PHI[int(np.argmin(err))]
        print(f"  single AR(1) phi that best fits the whole shrink-weight curve a_e2(h): "
              f"{best:.4f} (HL {np.log(.5)/np.log(best):.1f} d);  "
              f"pf(phi,h) = " + " ".join(f"h{h}:{pf(best,h):.3f}" for h in HS))
        print(f"  observed a_e2(h)               = " +
              " ".join(f"h{h}:{a:.3f}" for h, a in zip(t.index, aa)))
        print("  -> gap at the two ends is the LONG-MEMORY signature: one AR(1) is too slow "
              "early / too fast late.")
        # final recommended per-quintile rms correction (last walk-forward year's fit, n-weighted)
        cqm = np.vstack([np.vstack(gg.cq.values).mean(axis=0) for _, gg in g.groupby("h")])
        cqm = pd.DataFrame(cqm, index=t.index, columns=[f"Q{k+1}" for k in range(5)])
        print(f"  train-fit rms quintile correction c(q,h), mean over walk-forward years:")
        print(cqm.round(4).to_string())
        print(f"  LRsigma (train) median = {g.LRsig.median():.5f}/day")

    # ============================================================ D. robust ACF
    print("\n" + "=" * 112)
    print("D. AUTOCORRELATION out to lag 100 -- robust versions (squared returns are")
    print("   outlier-dominated; |r| and log(r^2) are the standard long-memory displays)")
    print("=" * 112)
    def acf(x, K=100):
        x = x - x.mean()
        den = (x ** 2).sum()
        return np.array([float((x[:-k] * x[k:]).sum() / den) for k in range(1, K + 1)])

    tabs = {}
    for s in ["SPY", "^GSPC", "QQQ", "IWM"]:
        r = lr[s].dropna().values
        tabs[(s, "r2")] = acf(r ** 2)
        tabs[(s, "r2_wins99")] = acf(np.clip(r ** 2, 0, np.percentile(r ** 2, 99)))
        tabs[(s, "abs_r")] = acf(np.abs(r))
        tabs[(s, "log_r2")] = acf(np.log(np.maximum(r ** 2, 1e-12)))
    ps = {k: [] for k in ("r2", "abs_r", "log_r2")}
    for s in singles:
        r = lr[s].dropna().values
        if len(r) < 800:
            continue
        ps["r2"].append(acf(r ** 2))
        ps["abs_r"].append(acf(np.abs(r)))
        ps["log_r2"].append(acf(np.log(np.maximum(r ** 2, 1e-12))))
    for k, v in ps.items():
        tabs[("SINGLES_MEAN", k)] = np.mean(v, axis=0)
    a = pd.DataFrame({f"{s}|{k}": v for (s, k), v in tabs.items()}, index=range(1, 101))
    show = [1, 2, 3, 5, 10, 15, 21, 30, 42, 63, 90, 100]
    for k in ("r2", "r2_wins99", "abs_r", "log_r2"):
        cols = [c for c in a.columns if c.endswith("|" + k)]
        if not cols:
            continue
        print(f"\n  ACF of {k}:")
        print(a[cols].loc[show].round(4).to_string())
    print(f"\n  2/sqrt(n) band, SPY n={int(lr['SPY'].notna().sum())}: "
          f"{2/np.sqrt(lr['SPY'].notna().sum()):.4f}")
    print("\n  decay fits on lags 1..100 (exp vs power law), plus half-life:")
    for c in a.columns:
        v = a[c].values
        lg = np.arange(1., 101.)
        m = v > 0
        if m.sum() < 20:
            continue
        A1 = np.column_stack([lg[m], np.ones(m.sum())])
        b1, *_ = np.linalg.lstsq(A1, np.log(v[m]), rcond=None)
        r21 = 1 - ((np.log(v[m]) - A1 @ b1) ** 2).sum() / ((np.log(v[m]) -
                                                            np.log(v[m]).mean()) ** 2).sum()
        A2 = np.column_stack([np.log(lg[m]), np.ones(m.sum())])
        b2, *_ = np.linalg.lstsq(A2, np.log(v[m]), rcond=None)
        r22 = 1 - ((np.log(v[m]) - A2 @ b2) ** 2).sum() / ((np.log(v[m]) -
                                                            np.log(v[m]).mean()) ** 2).sum()
        print(f"   {c:>24}: rho1={v[0]:+.4f} EXP phi={np.exp(b1[0]):.4f} "
              f"HL={np.log(.5)/b1[0]:6.1f}d R2={r21:.3f} | POWER d={b2[0]:+.4f} R2={r22:.3f}"
              f" | lag where rho<rho1/2: "
              f"{(np.flatnonzero(v < v[0]/2)[0]+1) if (v < v[0]/2).any() else '>100'}")
    print(f"\ndone {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
