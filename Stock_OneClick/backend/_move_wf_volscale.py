"""
_move_wf_volscale.py — how does volatility scale with horizon, really?

  1. unconditional scaling exponent  d log sigma_h / d log h   (0.5 iff i.i.d.)
  2. CONDITIONAL scaling: realized fwd per-day vol / current vol, by vol-state quintile x horizon
  3. AR(1) mean-reverting variance forecast vs plain sqrt(h) scaling, walk-forward OOS
  4. ACF of squared returns to lag 100 (long-memory structure)
  5. sd(z) by state quintile under each scaling rule -- the calibration payoff

Method rules honoured:
  - every fitted parameter (phi, long-run var, scale c, quintile edges, corrections) comes from
    strictly prior years only (L.walk_forward_years) or from a pre-2014 train block
  - overlapping windows -> uncertainty by BLOCK BOOTSTRAP in time (252-row blocks); for pooled
    single names the series is first collapsed to a DATE-level cross-sectional mean (clusters date)
  - indices (SPY/QQQ/^GSPC/IWM/DIA) reported separately as the survivorship-free benchmark
  - degenerate rows dropped: EWMA vol or realized fwd vol < 1e-4/day (frozen/halted prices)

Run: ../../vcp_env/bin/python _move_wf_volscale.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import _move_data as D
import _move_lib as L

HS_ALL = [1, 2, 3, 5, 10, 21, 42, 63]      # for the unconditional exponent (h=1 is meaningful)
HS = [2, 3, 5, 10, 21, 42, 63]             # for forward-realized-vol scoring
#   h=1 is EXCLUDED from 2/3/5: realized_vol_forward(c,1) = rolling(1).std(ddof=1) = NaN.
#   A 1-day realized sigma does not exist from daily bars; handled separately via |r| proxy.
IDX = ["SPY", "QQQ", "^GSPC", "IWM", "DIA"]
MIN_OBS = 500
FLOOR = 1e-4                               # 0.01%/day: below this the price is frozen, not calm
RNG = np.random.default_rng(7)
LAM = 0.94
LONG_N = 252
TRAIN_END = 2014

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 400)


# ------------------------------------------------------------------ small utils
def make_blocks(n, block=252):
    return [np.arange(s, min(s + block, n)) for s in range(0, n, block)]


def block_boot_ci(vals, block=252, stat=np.mean, n_boot=1500, q=(2.5, 97.5)):
    vals = np.asarray(vals, float)
    blocks = make_blocks(len(vals), block)
    if len(blocks) < 3:
        return (np.nan, np.nan)
    out = []
    for _ in range(n_boot):
        pick = RNG.integers(0, len(blocks), len(blocks))
        v = vals[np.concatenate([blocks[i] for i in pick])]
        v = v[np.isfinite(v)]
        if len(v):
            out.append(stat(v))
    return tuple(np.percentile(out, q))


def hday_logret_std(lr, h):
    """sd of overlapping h-day LOG returns, demeaned by h*mu_1 (Lo-MacKinlay)."""
    x = lr.dropna() if isinstance(lr, pd.Series) else pd.Series(lr).dropna()
    if len(x) < 5 * h + 30:
        return np.nan, 0, 0
    csum = x.cumsum()
    rh = (csum - csum.shift(h)).dropna().values
    dev = rh - h * x.mean()
    return float(np.sqrt((dev ** 2).sum() / (len(dev) - 1))), len(dev), len(x) // h


def fit_exponent(hs, sds):
    hs, sds = np.asarray(hs, float), np.asarray(sds, float)
    m = np.isfinite(sds) & (sds > 0)
    if m.sum() < 4:
        return np.nan
    A = np.column_stack([np.log(hs[m]), np.ones(m.sum())])
    b, *_ = np.linalg.lstsq(A, np.log(sds[m]), rcond=None)
    return float(b[0])


def qlike(sig_hat, sig_real):
    x = np.maximum(sig_real, FLOOR) ** 2 / np.maximum(sig_hat, FLOOR) ** 2
    return float(np.mean(x - np.log(x) - 1))


def logmse(sig_hat, sig_real):
    return float(np.mean((np.log(np.maximum(sig_hat, FLOOR))
                          - np.log(np.maximum(sig_real, FLOOR))) ** 2))


def path_factor(phi, h):
    """(1/h) * sum_{i=1..h} phi^i  -- the weight on the CURRENT variance shock over an h-day path."""
    if phi >= 1.0:
        return 1.0
    return float(phi * (1 - phi ** h) / (h * (1 - phi)))


def ar1_sigma(var_now, LRvar, phi, h):
    return np.sqrt(np.maximum(LRvar + (var_now - LRvar) * path_factor(phi, h), FLOOR ** 2))


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    p = D.load()
    close = p["Close"]
    keep = [s for s in close.columns if close[s].notna().sum() >= MIN_OBS and s != "^VIX"]
    close = close[keep]
    singles = [s for s in keep if s not in IDX]
    lr = np.log(close).diff()
    print(f"panel: {len(keep)} usable symbols (>= {MIN_OBS} closes) = {len(IDX)} indices + "
          f"{len(singles)} single names; {len(close)} rows "
          f"{close.index[0].date()} -> {close.index[-1].date()}")

    sig_now = L.vol_ewma(close, LAM)
    sig_long = L.vol_cc(close, LONG_N)
    state = sig_now / sig_long
    var_now = sig_now ** 2
    fwd = {h: L.realized_vol_forward(close, h) for h in HS}
    # validity mask: both current and realized vol non-degenerate
    ok = {h: (sig_now >= FLOOR) & (fwd[h] >= FLOOR) & fwd[h].notna() & sig_now.notna()
          for h in HS}

    # ============================================================ 1. UNCONDITIONAL EXPONENT
    print("\n" + "=" * 108)
    print("1. UNCONDITIONAL HORIZON-SCALING EXPONENT   sigma_h = sigma_1 * h^beta   "
          "(beta = 0.5 iff i.i.d.)")
    print("   sigma_h = sd of OVERLAPPING h-day log returns; beta = OLS slope of log sd on log h "
          "over h in " + str(HS_ALL))
    print("=" * 108)
    rows = []
    for s in IDX:
        sds = [hday_logret_std(lr[s], h)[0] for h in HS_ALL]
        beta = fit_exponent(HS_ALL, sds)
        x = lr[s].dropna()
        blocks = make_blocks(len(x), 252)
        betas = []
        for _ in range(500):
            pick = RNG.integers(0, len(blocks), len(blocks))
            xb = pd.Series(np.concatenate([x.values[blocks[i]] for i in pick]))
            b = fit_exponent(HS_ALL, [hday_logret_std(xb, h)[0] for h in HS_ALL])
            if np.isfinite(b):
                betas.append(b)
        lo, hi = np.percentile(betas, [2.5, 97.5])
        rows.append(dict(sym=s, n_days=len(x), beta=beta, ci_lo=lo, ci_hi=hi,
                         sigma_1d=sds[0], sigma_21d=sds[HS_ALL.index(21)], sigma_63d=sds[-1],
                         VR_5=(sds[HS_ALL.index(5)] ** 2) / (5 * sds[0] ** 2),
                         VR_21=(sds[HS_ALL.index(21)] ** 2) / (21 * sds[0] ** 2),
                         VR_63=(sds[-1] ** 2) / (63 * sds[0] ** 2),
                         n_nonovlp_63=hday_logret_std(lr[s], 63)[2]))
    print("\nINDICES (survivorship-free benchmark). 95% CI = 252-row block bootstrap, 500 reps.")
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    srows = []
    for s in singles:
        sds = [hday_logret_std(lr[s], h)[0] for h in HS_ALL]
        b = fit_exponent(HS_ALL, sds)
        if np.isfinite(b):
            srows.append(dict(sym=s, n=int(lr[s].notna().sum()), beta=b,
                              VR_21=(sds[HS_ALL.index(21)] ** 2) / (21 * sds[0] ** 2),
                              VR_63=(sds[-1] ** 2) / (63 * sds[0] ** 2)))
    sdf = pd.DataFrame(srows)
    bs = [sdf.beta.values[RNG.integers(0, len(sdf), len(sdf))].mean() for _ in range(5000)]
    print(f"\nSINGLE NAMES, per-symbol betas (n={len(sdf)} symbols; survivorship-biased):")
    print(f"  beta  mean {sdf.beta.mean():.4f}  median {sdf.beta.median():.4f}  "
          f"sd {sdf.beta.std():.4f}  p5 {sdf.beta.quantile(.05):.4f}  "
          f"p95 {sdf.beta.quantile(.95):.4f}")
    print(f"  95% CI on the mean beta, bootstrap clustered by SYMBOL: "
          f"[{np.percentile(bs, 2.5):.4f}, {np.percentile(bs, 97.5):.4f}]")
    print(f"  median VR_21 {sdf.VR_21.median():.4f}   median VR_63 {sdf.VR_63.median():.4f}   "
          f"share(beta<0.5) = {(sdf.beta < 0.5).mean():.3f}")

    pool = []
    for h in HS_ALL:
        v = [hday_logret_std(lr[s], h)[0] ** 2 for s in singles
             if np.isfinite(hday_logret_std(lr[s], h)[0])]
        pool.append(np.sqrt(np.mean(v)))
    print("\nAGGREGATION FACTOR kappa_h = sigma_h / (sigma_1 * sqrt(h))  "
          "[pure return-autocorrelation term, NOT vol clustering]")
    kap = {}
    for s in IDX:
        sds = [hday_logret_std(lr[s], h)[0] for h in HS_ALL]
        kap[s] = [sds[i] / (sds[0] * np.sqrt(h)) for i, h in enumerate(HS_ALL)]
    kap["POOLED_SINGLES"] = [pool[i] / (pool[0] * np.sqrt(h)) for i, h in enumerate(HS_ALL)]
    ktab = pd.DataFrame(kap, index=[f"h={h}" for h in HS_ALL]).T
    ktab.loc["INDEX_MEAN"] = ktab.loc[IDX].mean()
    print(ktab.round(4).to_string())
    print(f"  pooled-singles beta = {fit_exponent(HS_ALL, pool):.4f}")
    # index kappa CI at h=21 and h=63 (block bootstrap on SPY)
    for h in (21, 63):
        x = lr["SPY"].dropna()
        blocks = make_blocks(len(x), 252)
        ks = []
        for _ in range(600):
            pick = RNG.integers(0, len(blocks), len(blocks))
            xb = pd.Series(np.concatenate([x.values[blocks[i]] for i in pick]))
            s1 = hday_logret_std(xb, 1)[0]
            sh = hday_logret_std(xb, h)[0]
            ks.append(sh / (s1 * np.sqrt(h)))
        print(f"  SPY kappa_{h} = {kap['SPY'][HS_ALL.index(h)]:.4f}  95% CI "
              f"[{np.percentile(ks, 2.5):.4f}, {np.percentile(ks, 97.5):.4f}] (block bootstrap)")

    # ============================================================ 2. CONDITIONAL SCALING
    print("\n" + "=" * 108)
    print("2. CONDITIONAL SCALING by vol state.  state = vol_ewma(0.94) / vol_cc(252)")
    print("   metric = realized h-day vol / (current daily vol * sqrt(h))")
    print("          = realized_vol_forward(c,h) / vol_ewma(c)      [the sqrt(h) cancels]")
    print("   Quintile edges from STRICTLY PRIOR years (expanding walk-forward, min 5y train).")
    print("=" * 108)

    def cond_table(cols, label, robust=False):
        # per (h, quintile): list of DATE-level cross-sectional means, in date order
        acc = {h: {k: {} for k in range(5)} for h in HS}
        allacc = {h: {} for h in HS}
        for y, tr, te in L.walk_forward_years(close.index, min_train_years=5):
            st_tr = state.loc[tr, cols].values.ravel()
            st_tr = st_tr[np.isfinite(st_tr)]
            if len(st_tr) < 1000:
                continue
            edges = np.percentile(st_tr, [20, 40, 60, 80])
            st_te = state.loc[te, cols]
            qi = np.digitize(st_te.values, edges).astype(float)
            qi[~np.isfinite(st_te.values)] = np.nan
            qi = pd.DataFrame(qi, index=st_te.index, columns=st_te.columns)
            for h in HS:
                r = (fwd[h].loc[te, cols] / sig_now.loc[te, cols]).where(ok[h].loc[te, cols])
                dm_all = r.mean(axis=1)
                for d, v in dm_all.items():
                    if np.isfinite(v):
                        allacc[h][d] = v
                for k in range(5):
                    rk = r.where(qi == k)
                    dm, cnt = rk.mean(axis=1), rk.notna().sum(axis=1)
                    for d, v, n in zip(dm.index, dm.values, cnt.values):
                        if n > 0 and np.isfinite(v):
                            acc[h][k][d] = (v, n)
        recs = []
        for h in HS:
            dts_all = sorted(allacc[h])
            lvl = float(np.mean([allacc[h][d] for d in dts_all]))
            for k in range(5):
                dd = acc[h][k]
                if not dd:
                    continue
                dts = sorted(dd)
                vals = np.array([dd[d][0] for d in dts])
                lo, hi = block_boot_ci(vals, 252, n_boot=1200)
                recs.append(dict(h=h, q=k + 1, ratio=float(vals.mean()), lo=lo, hi=hi,
                                 rel=float(vals.mean()) / lvl, all_days_level=lvl,
                                 n_days=len(vals), n_obs=int(sum(dd[d][1] for d in dts))))
        t = pd.DataFrame(recs)
        print(f"\n--- {label} ---")
        piv = t.pivot(index="h", columns="q", values="ratio")
        piv.columns = [f"Q{c}" for c in piv.columns]
        piv["all"] = t.groupby("h").all_days_level.first()
        piv["Q5/Q1"] = piv.Q5 / piv.Q1
        print("(a) RAW mean( realized_fwd_perday / vol_ewma_now ).  Q1 = calmest state, "
              "Q5 = most excited")
        print(piv.round(4).to_string())
        rel = t.pivot(index="h", columns="q", values="rel")
        rel.columns = [f"Q{c}" for c in rel.columns]
        print("\n(b) RELATIVE correction factor = quintile ratio / all-days ratio at the same h")
        print("    (this is the pure mean-reversion term; the all-days level is the separate c_h)")
        print(rel.round(4).to_string())
        print("\n(c) 95% CI (block bootstrap, 252-day blocks, on DATE-level cross-sec means) and n")
        tt = t.copy()
        tt["ci95"] = tt.apply(lambda r: f"[{r.lo:.3f}, {r.hi:.3f}]", axis=1)
        print(tt[["h", "q", "ratio", "ci95", "rel", "n_days", "n_obs"]].to_string(index=False))
        return piv, rel, t

    idx_raw, idx_rel, idx_t = cond_table(IDX, "INDICES (survivorship-free)")
    sng_raw, sng_rel, sng_t = cond_table(singles, "POOLED SINGLE NAMES (survivorship-biased)")

    print("\nState-quintile edges (vol_ewma/vol_cc252), fit on 2001-2013 for reference:")
    for lab, cols in [("INDICES", IDX), ("SINGLES", singles)]:
        v = state.loc[close.index.year < TRAIN_END, cols].values.ravel()
        v = v[np.isfinite(v)]
        print(f"  {lab} (n={len(v):,}): p20={np.percentile(v,20):.3f} p40={np.percentile(v,40):.3f} "
              f"p60={np.percentile(v,60):.3f} p80={np.percentile(v,80):.3f}   "
              f"median={np.median(v):.3f}")

    # ============================================================ 3. AR(1) vs sqrt(h)
    print("\n" + "=" * 108)
    print("3. AR(1) MEAN-REVERTING VARIANCE FORECAST vs PLAIN sqrt(h) (flat-sigma) SCALING")
    print("   sigma_h_hat/sqrt(h) = c_h * sqrt( LRvar + (sigma_t^2 - LRvar) * "
          "phi(1-phi^h)/(h(1-phi)) )")
    print("   flat baseline:  sigma_h_hat/sqrt(h) = c_h * sigma_t")
    print("   Both get a train-fit scale c_h, so the comparison is about SHAPE not level.")
    print("   Loss: QLIKE (variance-proportional, standard for vol) and logMSE. Lower = better.")
    print("=" * 108)

    PHI = np.round(np.concatenate([np.arange(0.70, 0.99, 0.01), np.arange(0.990, 0.9995, 0.001)]), 4)

    def fit_phi(vn, fr, LRvar, hs, per_h=True):
        """grid-search phi minimising QLIKE with the QLIKE-optimal scale folded in."""
        best = {}
        tot = np.full(len(PHI), 0.0)
        for hi_, h in enumerate(hs):
            v, f = vn[h], fr[h]
            q = np.empty(len(PHI))
            for i, phi in enumerate(PHI):
                sh = ar1_sigma(v, LRvar, phi, h)
                c = np.sqrt(np.mean(f ** 2 / sh ** 2))
                q[i] = qlike(sh * c, f)
            tot += q
            best[h] = PHI[int(np.nanargmin(q))]
        return best, PHI[int(np.nanargmin(tot))]

    def prep(cols, mask, hs):
        vn, fr = {}, {}
        for h in hs:
            m = mask & ok[h]
            a = var_now.loc[:, cols].where(m.loc[:, cols]).values.ravel()
            b = fwd[h].loc[:, cols].where(m.loc[:, cols]).values.ravel()
            g = np.isfinite(a) & np.isfinite(b)
            vn[h], fr[h] = a[g], b[g]
        return vn, fr

    # ---- 3a: expanding walk-forward by year, phi refit each year on prior years
    def wf_ar1(cols, label):
        out = []
        for y, tr, te in L.walk_forward_years(close.index, min_train_years=5):
            trm = pd.DataFrame(np.repeat(tr[:, None], len(close.columns), 1),
                               index=close.index, columns=close.columns)
            tem = pd.DataFrame(np.repeat(te[:, None], len(close.columns), 1),
                               index=close.index, columns=close.columns)
            LRvar = float(np.nanmean(lr.loc[tr, cols].values ** 2))
            vn_t, fr_t = prep(cols, trm, HS)
            vn_e, fr_e = prep(cols, tem, HS)
            if min(len(v) for v in vn_t.values()) < 1000:
                continue
            per_h, glob = fit_phi(vn_t, fr_t, LRvar, HS)
            for h in HS:
                if len(vn_e[h]) < 100:
                    continue
                phi = per_h[h]
                sh_t = ar1_sigma(vn_t[h], LRvar, phi, h)
                c_ar = np.sqrt(np.mean(fr_t[h] ** 2 / sh_t ** 2))
                c_fl = np.sqrt(np.mean(fr_t[h] ** 2 / vn_t[h]))
                a_ar = ar1_sigma(vn_e[h], LRvar, phi, h) * c_ar
                a_fl = np.sqrt(vn_e[h]) * c_fl
                # logMSE-matched scales (geometric), fit on train
                g_ar = np.exp(np.mean(np.log(fr_t[h]) - np.log(sh_t)))
                g_fl = np.exp(np.mean(np.log(fr_t[h]) - 0.5 * np.log(vn_t[h])))
                out.append(dict(year=y, h=h, n=len(vn_e[h]), phi=phi,
                                hl=np.log(.5) / np.log(phi), c_ar=c_ar, c_fl=c_fl,
                                q_ar=qlike(a_ar, fr_e[h]), q_fl=qlike(a_fl, fr_e[h]),
                                l_ar=logmse(ar1_sigma(vn_e[h], LRvar, phi, h) * g_ar, fr_e[h]),
                                l_fl=logmse(np.sqrt(vn_e[h]) * g_fl, fr_e[h]),
                                glob_phi=glob))
        r = pd.DataFrame(out)
        agg = []
        for h, g in r.groupby("h"):
            w = g.n / g.n.sum()
            agg.append(dict(h=h, n_obs=int(g.n.sum()), n_years=len(g),
                            phi_med=g.phi.median(), phi_min=g.phi.min(), phi_max=g.phi.max(),
                            halflife_d=np.log(.5) / np.log(g.phi.median()),
                            QLIKE_ar1=(g.q_ar * w).sum(), QLIKE_flat=(g.q_fl * w).sum(),
                            logMSE_ar1=(g.l_ar * w).sum(), logMSE_flat=(g.l_fl * w).sum(),
                            yrs_ar1_wins=f"{int((g.q_ar < g.q_fl).sum())}/{len(g)}"))
        a = pd.DataFrame(agg)
        a["QLIKE_impr%"] = 100 * (1 - a.QLIKE_ar1 / a.QLIKE_flat)
        a["logMSE_impr%"] = 100 * (1 - a.logMSE_ar1 / a.logMSE_flat)
        print(f"\n--- {label} --- OOS, expanding walk-forward by year; phi refit per year "
              f"on prior years only")
        print(a.round(4).to_string(index=False))
        print(f"    global phi (all horizons jointly) selected per year: "
              f"median {r.glob_phi.median():.4f}  range [{r.glob_phi.min():.4f}, "
              f"{r.glob_phi.max():.4f}]  -> halflife "
              f"{np.log(.5)/np.log(r.glob_phi.median()):.1f} d")
        return a, r

    idx_wf, idx_wfr = wf_ar1(IDX, "INDICES")
    print(f"[{time.time()-t0:.0f}s]")
    sng_wf, sng_wfr = wf_ar1(singles, "POOLED SINGLE NAMES")
    print(f"[{time.time()-t0:.0f}s]")

    # ---- 3b: ONE global phi fit on pre-2014, scored 2014+, head-to-head vs quintile lookup
    print("\n" + "-" * 108)
    print("3b. HEAD-TO-HEAD OOS: flat sqrt(h)  vs  AR(1) one-global-phi  vs  per-quintile lookup")
    print("    All parameters (phi, LRvar, c_h, quintile edges, per-quintile corrections) from")
    print(f"    {close.index[0].year}-{TRAIN_END-1} only; scored on {TRAIN_END}-2026.")
    print("-" * 108)
    trb = close.index.year < TRAIN_END
    teb = close.index.year >= TRAIN_END
    spec = {}
    for label, cols in [("INDICES", IDX), ("SINGLES", singles)]:
        trm = pd.DataFrame(np.repeat(trb[:, None], len(close.columns), 1),
                           index=close.index, columns=close.columns)
        tem = pd.DataFrame(np.repeat(teb[:, None], len(close.columns), 1),
                           index=close.index, columns=close.columns)
        LRvar = float(np.nanmean(lr.loc[trb, cols].values ** 2))
        vn_t, fr_t = prep(cols, trm, HS)
        vn_e, fr_e = prep(cols, tem, HS)
        per_h, phi_g = fit_phi(vn_t, fr_t, LRvar, HS)
        st_tr = state.loc[trb, cols].values.ravel()
        edges = np.percentile(st_tr[np.isfinite(st_tr)], [20, 40, 60, 80])
        rows, corr_tab = [], {}
        for h in HS:
            m_t = trm & ok[h]
            m_e = tem & ok[h]
            def flat(mask):
                a = var_now.loc[:, cols].where(mask.loc[:, cols]).values.ravel()
                b = fwd[h].loc[:, cols].where(mask.loc[:, cols]).values.ravel()
                s = state.loc[:, cols].where(mask.loc[:, cols]).values.ravel()
                g = np.isfinite(a) & np.isfinite(b) & np.isfinite(s)
                return a[g], b[g], s[g]
            va, fa, sa = flat(m_t)
            ve, fe, se = flat(m_e)
            qa = np.digitize(sa, edges)
            qe = np.digitize(se, edges)
            R = fa / np.sqrt(va)
            # per-quintile correction: QLIKE-optimal scale within the quintile (= rms of R)
            corr = np.array([np.sqrt(np.mean(R[qa == k] ** 2)) if (qa == k).sum() > 50 else np.nan
                             for k in range(5)])
            corr_tab[h] = corr
            sh_t = ar1_sigma(va, LRvar, phi_g, h)
            c_ar = np.sqrt(np.mean(fa ** 2 / sh_t ** 2))
            c_fl = np.sqrt(np.mean(R ** 2))
            a_fl = np.sqrt(ve) * c_fl
            a_ar = ar1_sigma(ve, LRvar, phi_g, h) * c_ar
            a_q = np.sqrt(ve) * corr[qe]
            g = np.isfinite(a_q)
            rows.append(dict(h=h, n=int(g.sum()), c_fl=c_fl, c_ar=c_ar,
                             QL_flat=qlike(a_fl[g], fe[g]), QL_ar1=qlike(a_ar[g], fe[g]),
                             QL_quint=qlike(a_q[g], fe[g]),
                             LM_flat=logmse(a_fl[g], fe[g]), LM_ar1=logmse(a_ar[g], fe[g]),
                             LM_quint=logmse(a_q[g], fe[g]),
                             bias_flat=float(np.mean(fe[g] / a_fl[g])),
                             bias_ar1=float(np.mean(fe[g] / a_ar[g])),
                             bias_quint=float(np.mean(fe[g] / a_q[g]))))
        r = pd.DataFrame(rows)
        r["ar1_vs_flat%"] = 100 * (1 - r.QL_ar1 / r.QL_flat)
        r["quint_vs_flat%"] = 100 * (1 - r.QL_quint / r.QL_flat)
        r["ar1_vs_quint%"] = 100 * (1 - r.QL_ar1 / r.QL_quint)
        print(f"\n--- {label} ---  phi_global = {phi_g:.4f}  -> halflife "
              f"{np.log(.5)/np.log(phi_g):.1f} trading days;  LRsigma = "
              f"{np.sqrt(LRvar):.5f}/day (train {close.index[0].year}-{TRAIN_END-1})")
        print(r.round(4).to_string(index=False))
        print(f"  train-fit per-quintile correction c(q,h) used by the lookup model "
              f"(rows=h, cols=Q1..Q5), edges={np.round(edges,3)}:")
        print(pd.DataFrame(corr_tab, index=[f"Q{k+1}" for k in range(5)]).T.round(4).to_string())
        print(f"  path_factor(phi_g,h) = " +
              "  ".join(f"h{h}:{path_factor(phi_g, h):.3f}" for h in HS))
        spec[label] = dict(phi=phi_g, LRvar=LRvar, edges=edges, corr=corr_tab,
                           c_ar=r.set_index("h").c_ar, c_fl=r.set_index("h").c_fl)

    # ---- 3c: date-clustered significance of the AR(1) - flat QLIKE gap (SPY-only, honest)
    print("\n3c. Significance of the QLIKE gap, INDICES, pre-2014 fit / 2014+ score.")
    print("    Per-DATE mean QLIKE difference (flat - ar1), block bootstrap 252-day blocks:")
    LRvar = float(np.nanmean(lr.loc[trb, IDX].values ** 2))
    trm = pd.DataFrame(np.repeat(trb[:, None], len(close.columns), 1),
                       index=close.index, columns=close.columns)
    vn_t, fr_t = prep(IDX, trm, HS)
    _ph, phi_g = fit_phi(vn_t, fr_t, LRvar, HS)
    for h in HS:
        m_t = trm & ok[h]
        va = var_now.loc[:, IDX].where(m_t.loc[:, IDX]).values.ravel()
        fa = fwd[h].loc[:, IDX].where(m_t.loc[:, IDX]).values.ravel()
        g = np.isfinite(va) & np.isfinite(fa)
        c_ar = np.sqrt(np.mean(fa[g] ** 2 / ar1_sigma(va[g], LRvar, phi_g, h) ** 2))
        c_fl = np.sqrt(np.mean(fa[g] ** 2 / va[g]))
        V = var_now.loc[teb, IDX].where(ok[h].loc[teb, IDX])
        F = fwd[h].loc[teb, IDX].where(ok[h].loc[teb, IDX])
        A_ar = ar1_sigma(V.values, LRvar, phi_g, h) * c_ar
        A_fl = np.sqrt(V.values) * c_fl
        def ql_el(a, f):
            x = np.maximum(f, FLOOR) ** 2 / np.maximum(a, FLOOR) ** 2
            return x - np.log(x) - 1
        d = pd.DataFrame(ql_el(A_fl, F.values) - ql_el(A_ar, F.values),
                         index=V.index, columns=V.columns).mean(axis=1).dropna()
        lo, hi = block_boot_ci(d.values, 252, n_boot=2000)
        print(f"  h={h:>2}: mean(flat-ar1) QLIKE = {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
              f"  n_dates={len(d)}   {'SIGNIF' if lo > 0 else 'not signif'}")

    # ---- 3d: independent phi estimate from a log-vol regression (no grid search)
    print("\n3d. Independent phi estimate: OLS log(realized fwd_h vol) = a + b*log(sigma_ewma_t)")
    print("    + (implicitly) (1-b)*log(long-run). b should equal path_factor(phi,h) if the AR(1)")
    print("    variance form holds (in log-vol space, approximately). IN-SAMPLE (descriptive).")
    for label, cols in [("INDICES", IDX), ("SINGLES", singles)]:
        bs_ = []
        for h in HS:
            m = ok[h] & (sig_long >= FLOOR)
            x1 = np.log(sig_now.loc[:, cols].where(m.loc[:, cols]).values.ravel())
            x2 = np.log(sig_long.loc[:, cols].where(m.loc[:, cols]).values.ravel())
            y = np.log(fwd[h].loc[:, cols].where(m.loc[:, cols]).values.ravel())
            gg = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(y)
            A = np.column_stack([x1[gg], x2[gg], np.ones(gg.sum())])
            b, *_ = np.linalg.lstsq(A, y[gg], rcond=None)
            # phi that reproduces this weight on the current shock
            cand = PHI[np.argmin([abs(path_factor(pp, h) - b[0]) for pp in PHI])]
            bs_.append(dict(h=h, n=int(gg.sum()), b_short=b[0], b_long=b[1], sum=b[0] + b[1],
                            implied_phi=cand, implied_HL=np.log(.5) / np.log(cand)))
        print(f"  {label}:")
        print(pd.DataFrame(bs_).round(4).to_string(index=False))

    # ============================================================ 4. ACF of squared returns
    print("\n" + "=" * 108)
    print("4. AUTOCORRELATION OF SQUARED LOG RETURNS, lags 1..100")
    print("=" * 108)
    acf = {}
    for s in ["SPY", "^GSPC", "QQQ", "IWM"]:
        x = lr[s].dropna().values ** 2
        x = x - x.mean()
        den = (x ** 2).sum()
        acf[s] = [float((x[:-k] * x[k:]).sum() / den) for k in range(1, 101)]
    # pooled singles: average ACF across symbols (each symbol's own ACF, then mean)
    ps = []
    for s in singles:
        x = lr[s].dropna().values ** 2
        if len(x) < 800:
            continue
        x = x - x.mean()
        den = (x ** 2).sum()
        ps.append([float((x[:-k] * x[k:]).sum() / den) for k in range(1, 101)])
    acf["SINGLES_MEAN"] = np.mean(ps, axis=0)
    a = pd.DataFrame(acf, index=range(1, 101))
    show = [1, 2, 3, 5, 10, 15, 21, 30, 42, 50, 63, 75, 90, 100]
    print(f"\nACF at selected lags. n(SPY)={int(lr['SPY'].notna().sum())}, "
          f"n_symbols(SINGLES_MEAN)={len(ps)}. "
          f"2/sqrt(n) white-noise band = {2/np.sqrt(lr['SPY'].notna().sum()):.4f}")
    print(a.loc[show].round(4).to_string())
    print("\nsum ACF(1..100) = integrated persistence:")
    print(a.sum().round(3).to_string())
    print("\ndecay-form fit on lags 1..60 (positive lags only): exponential vs power law")
    for s in a.columns:
        v = a[s].values[:60]
        lg = np.arange(1., 61.)
        m = v > 0
        A1 = np.column_stack([lg[m], np.ones(m.sum())])
        b1, *_ = np.linalg.lstsq(A1, np.log(v[m]), rcond=None)
        r2_1 = 1 - ((np.log(v[m]) - A1 @ b1) ** 2).sum() / \
            ((np.log(v[m]) - np.log(v[m]).mean()) ** 2).sum()
        A2 = np.column_stack([np.log(lg[m]), np.ones(m.sum())])
        b2, *_ = np.linalg.lstsq(A2, np.log(v[m]), rcond=None)
        r2_2 = 1 - ((np.log(v[m]) - A2 @ b2) ** 2).sum() / \
            ((np.log(v[m]) - np.log(v[m]).mean()) ** 2).sum()
        print(f"  {s:>14}: rho(1)={v[0]:.4f} | EXP  phi={np.exp(b1[0]):.4f} "
              f"HL={np.log(.5)/b1[0]:6.1f}d R2={r2_1:.3f} | POWER exp={b2[0]:+.4f} R2={r2_2:.3f}"
              f" | lags>0 of 60: {int(m.sum())}")

    # ============================================================ 5. sd(z) payoff
    print("\n" + "=" * 108)
    print("5. THE CALIBRATION PAYOFF: sd of z = r_h / sigma_h_hat, by state quintile.")
    print("   A correct scaling rule gives sd(z) = 1 in EVERY quintile. Deviations are exactly")
    print("   the mis-calibration the bucket probabilities will inherit.")
    print(f"   All parameters from {close.index[0].year}-{TRAIN_END-1}; z measured on {TRAIN_END}+.")
    print("=" * 108)
    for label, cols in [("INDICES", IDX), ("SINGLES", singles)]:
        sp = spec[label]
        phi_g, LRvar, edges = sp["phi"], sp["LRvar"], sp["edges"]
        out_f, out_a, out_k = {}, {}, {}
        for h in HS:
            fret = np.log(close[cols].shift(-h) / close[cols])       # log fwd return, h days
            m = (sig_now[cols] >= FLOOR) & fret.notna() & state[cols].notna()
            qe = np.digitize(state[cols].where(m).values, edges).astype(float)
            qe[~np.isfinite(state[cols].where(m).values)] = np.nan
            v = var_now[cols].where(m).values
            kap_h = ktab.loc["INDEX_MEAN" if label == "INDICES" else "POOLED_SINGLES",
                             f"h={h}"]
            s_fl = np.sqrt(v) * sp["c_fl"][h] * np.sqrt(h)
            s_ar = ar1_sigma(v, LRvar, phi_g, h) * sp["c_ar"][h] * np.sqrt(h)
            s_ak = s_ar * kap_h
            te_ = np.repeat((close.index.year >= TRAIN_END)[:, None], len(cols), 1)
            rows_f, rows_a, rows_k = {}, {}, {}
            for k in range(5):
                sel = (qe == k) & te_ & np.isfinite(fret.values)
                if sel.sum() < 100:
                    continue
                rows_f[k + 1] = float(np.nanstd(fret.values[sel] / s_fl[sel]))
                rows_a[k + 1] = float(np.nanstd(fret.values[sel] / s_ar[sel]))
                rows_k[k + 1] = float(np.nanstd(fret.values[sel] / s_ak[sel]))
            out_f[h], out_a[h], out_k[h] = rows_f, rows_a, rows_k
        for nm, dd in [("FLAT sqrt(h)", out_f), ("AR(1) path", out_a),
                       ("AR(1) path x kappa_h", out_k)]:
            t = pd.DataFrame(dd).T
            t.columns = [f"Q{c}" for c in t.columns]
            t["Q5/Q1"] = t.iloc[:, 4] / t.iloc[:, 0]
            t["mean|sd(z)-1|"] = (t.iloc[:, :5] - 1).abs().mean(axis=1)
            print(f"\n  {label} / {nm}:  sd(z) by state quintile")
            print(t.round(4).to_string())

    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
