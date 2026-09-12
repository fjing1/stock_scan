"""
_move_wf_verify_hscale2.py — part 2 of the adversarial check on the horizon-scaling claim.

Two things part 1 left open:

  A. The walk-forward "realised kappa_63 = 0.7275" in a 252-day test year is itself measured on a
     SHORT sample, where the overlapping-sum sd estimator is badly downward biased (Lo-MacKinlay:
     variance biased by ~(1-h/T) = 0.75 when h=63, T=252).  Bias-correct it with the SAME sign-
     randomisation null (which forces true kappa=1 while keeping the segment's vol clustering),
     and see how much of the OOS "confirmation" is bias.

  B. The claim's payload is a CALIBRATION statement.  Score it where it matters: 4-bucket
     (+/-2%) probability log loss vs the train-only CLIMATOLOGY baseline, walk-forward by year,
     with three competing scale rules and ONE shared standardised shape so the only difference is
     the horizon-scaling factor:
         raw    : sigma_h = sigma_ewma * sqrt(h)
         kappa  : sigma_h = sigma_ewma * sqrt(h) * kappa_hat_h      (the claim's prescription)
         cfit   : sigma_h = sigma_ewma * sqrt(h) * c_h              (plain train-fitted scale)
     Uncertainty on the log-loss differences: block bootstrap on per-DATE mean loss (clusters date).

Run: ../../vcp_env/bin/python _move_wf_verify_hscale2.py
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

import _move_data as D
import _move_lib as L

warnings.filterwarnings("ignore")

IDX = ["SPY", "QQQ", "^GSPC", "IWM", "DIA"]
MIN_OBS = 500
THR = 0.02
RNG = np.random.default_rng(31)
pd.set_option("display.width", 235)
pd.set_option("display.max_columns", 90)
pd.set_option("display.max_rows", 400)


def hstd_fast(x, h):
    T = len(x)
    if T < 3 * h:
        return np.nan
    cs = np.concatenate(([0.0], np.cumsum(x)))
    rh = cs[h:] - cs[:-h]
    dev = rh - h * x.mean()
    return float(np.sqrt((dev ** 2).sum() / (len(dev) - 1)))


def seg_kappa(seg, ext, h):
    """kappa_h measured on a short segment: sigma_1 from `seg`, h-day windows STARTING in `seg`
    (using `ext` = seg + up to h-1 later bars)."""
    n = len(seg)
    if len(ext) < n + h - 1 or n < 3 * h:
        return np.nan
    cs = np.concatenate(([0.0], np.cumsum(ext)))
    st = np.arange(n)
    dev = (cs[st + h] - cs[st]) - h * ext.mean()
    sh = np.sqrt((dev ** 2).sum() / (len(dev) - 1))
    return float(sh / (hstd_fast(seg, 1) * np.sqrt(h)))


def make_blocks(n, block):
    return [np.arange(s, min(s + block, n)) for s in range(0, n, block)]


def block_boot_ci(v, block=252, n_boot=2000, q=(2.5, 97.5)):
    v = np.asarray(v, float)
    bl = make_blocks(len(v), block)
    if len(bl) < 3:
        return (np.nan, np.nan)
    out = []
    for _ in range(n_boot):
        pick = RNG.integers(0, len(bl), len(bl))
        z = v[np.concatenate([bl[i] for i in pick])]
        z = z[np.isfinite(z)]
        if len(z):
            out.append(z.mean())
    return tuple(np.percentile(out, q))


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
    print(f"panel: {len(keep)} symbols ({len(singles)} singles), {len(close)} rows")

    # ================================================================ A. bias-corrected OOS kappa
    print("\n" + "=" * 120)
    print("A. IS THE OOS 'realised kappa' CONFIRMATION REAL, OR SHORT-SAMPLE BIAS?")
    print("   For every test segment, the same statistic is recomputed on 300 SIGN-RANDOMISED")
    print("   copies of that segment (|r| untouched => true kappa = 1).  nullA_mean is therefore")
    print("   the value a NO-AUTOCORRELATION world would produce with the same segment length and")
    print("   the same volatility path.  'obs/nullA' is the bias-corrected kappa.")
    print("=" * 120)

    def seg_pair(sym, mask, h):
        v = lr[sym].values
        fin = np.isfinite(v)
        rows = np.where(mask & fin)[0]
        if len(rows) == 0:
            return np.array([]), np.array([])
        hi = min(rows[-1] + h - 1, len(v) - 1)
        ext = np.where(fin & (np.arange(len(v)) >= rows[0]) & (np.arange(len(v)) <= hi))[0]
        return v[rows], v[ext]

    for h in (5, 21, 63):
        rows = []
        for y in yrs[5:]:
            seg, ext = seg_pair("SPY", years == y, h)
            if len(seg) < 3 * h:
                continue
            obs = seg_kappa(seg, ext, h)
            nul = []
            for _ in range(300):
                sg = RNG.integers(0, 2, len(ext)) * 2 - 1
                nul.append(seg_kappa((ext * sg)[:len(seg)], ext * sg, h))
            nul = np.array(nul, float)
            tr = lr["SPY"][years < y].dropna().values
            k_hat = hstd_fast(tr, h) / (hstd_fast(tr, 1) * np.sqrt(h))
            rows.append(dict(y=y, n_seg=len(seg), obs=obs, nullA_mean=np.nanmean(nul),
                             nullA_p2_5=np.nanpercentile(nul, 2.5),
                             corrected=obs / np.nanmean(nul), k_hat_train=k_hat,
                             p_null=float(np.nanmean(nul <= obs))))
        r = pd.DataFrame(rows)
        n_ = len(r)
        print(f"\n  SPY h={h}, n_test_years={n_}")
        print(f"    raw mean realised kappa      {r.obs.mean():.4f}")
        print(f"    NULL-A expected (true k=1)   {r.nullA_mean.mean():.4f}   "
              f"<- pure short-sample bias of the statistic")
        print(f"    BIAS-CORRECTED realised      {r.corrected.mean():.4f}  "
              f"sd across years {r.corrected.std():.4f}  "
              f"t vs 1 = {(r.corrected.mean()-1)/(r.corrected.std()/np.sqrt(n_)):.2f}  "
              f"years<1: {int((r.corrected<1).sum())}/{n_}")
        print(f"    full-sample in-sample kappa (train, last year) = {r.k_hat_train.iloc[-1]:.4f}")
        if h == 63:
            print(r.round(4).to_string(index=False))

    print("\n  Same, on NON-OVERLAPPING 3-year test blocks, all 5 indices, h=63:")
    rows = []
    for a in range(2007, 2026, 3):
        m = (years >= a) & (years <= a + 2)
        if m.sum() < 500:
            continue
        for s in IDX:
            seg, ext = seg_pair(s, m, 63)
            if len(seg) < 400:
                continue
            obs = seg_kappa(seg, ext, 63)
            nul = []
            for _ in range(200):
                sg = RNG.integers(0, 2, len(ext)) * 2 - 1
                nul.append(seg_kappa((ext * sg)[:len(seg)], ext * sg, 63))
            rows.append(dict(block=f"{a}-{a+2}", sym=s, n=len(seg), obs=obs,
                             nullA=np.nanmean(nul), corrected=obs / np.nanmean(nul)))
    r3 = pd.DataFrame(rows)
    print(r3.pivot(index="block", columns="sym", values="corrected").round(4).to_string())
    bm = r3.groupby("block").corrected.mean()
    print(f"    NULL-A bias level (mean) = {r3.nullA.mean():.4f}")
    print(f"    bias-corrected block means: {bm.round(4).to_dict()}")
    print(f"    mean {bm.mean():.4f}  sd {bm.std():.4f}  "
          f"t vs 1 = {(bm.mean()-1)/(bm.std()/np.sqrt(len(bm))):.2f}  (n={len(bm)} blocks)")

    # ================================================================ B. bucket calibration
    print("\n" + "=" * 120)
    print("B. DOES THE CLAIM PAY OFF WHERE IT MATTERS?  4-bucket (+/-2%) probability log loss,")
    print("   walk-forward by year, vs the train-only CLIMATOLOGY baseline.  One shared")
    print("   standardised shape; the ONLY difference between rules is the horizon-scaling factor.")
    print("=" * 120)
    s_e = L.vol_ewma(close, 0.94)
    BOUND = np.array([np.log(1 - THR), 0.0, np.log(1 + THR)])
    results, per_date = [], {}
    for h in (5, 21, 63):
        flog = np.log(close.shift(-h) / close)
        fsim = close.shift(-h) / close - 1.0
        for lab, cols in [("INDICES", IDX), ("SINGLES", singles)]:
            ci = [close.columns.get_loc(c) for c in cols]
            U = (flog.iloc[:, ci].values / (s_e.iloc[:, ci].values * np.sqrt(h)))
            SH = s_e.iloc[:, ci].values * np.sqrt(h)
            FS = fsim.iloc[:, ci].values
            bidx = np.where(FS <= -THR, 0, np.where(FS <= 0, 1, np.where(FS < THR, 2, 3)))
            fin = np.isfinite(U) & np.isfinite(FS) & (SH > 0)
            yrow = np.repeat(years[:, None], len(ci), 1)
            drow = np.repeat(np.arange(len(close))[:, None], len(ci), 1)
            acc = {k: [] for k in ("clim", "raw", "kappa", "cfit")}
            dts = []
            kap_used, c_used = [], []
            for y in yrs[5:]:
                tr = fin & (yrow < y)
                te = fin & (yrow == y)
                if tr.sum() < 2000 or te.sum() < 50:
                    continue
                u_tr = U[tr]
                sd_u = float(np.sqrt(np.mean(u_tr ** 2)))
                shape = np.sort(u_tr / sd_u)                # unit-2nd-moment standardised shape
                clim = np.bincount(bidx[tr], minlength=4).astype(float)
                clim /= clim.sum()
                # unconditional kappa_h from train only
                kk = []
                for s in cols:
                    xx = lr[s].values[(years < y) & np.isfinite(lr[s].values)]
                    if len(xx) > 300:
                        v = hstd_fast(xx, h) / (hstd_fast(xx, 1) * np.sqrt(h))
                        if np.isfinite(v):
                            kk.append(v)
                kap = float(np.mean(kk)) if kk else 1.0
                cfit = sd_u
                kap_used.append(kap); c_used.append(cfit)
                sh_te = SH[te]
                b_te = bidx[te]
                for nm, sc in (("raw", 1.0), ("kappa", kap), ("cfit", cfit)):
                    a = BOUND[None, :] / (sh_te * sc)[:, None]
                    cdf = np.searchsorted(shape, a) / len(shape)
                    pr = np.diff(np.column_stack([np.zeros(len(cdf)), cdf,
                                                  np.ones(len(cdf))]), axis=1)
                    pr = np.clip(pr, 1e-6, 1); pr /= pr.sum(1, keepdims=True)
                    acc[nm].append(-np.log(pr[np.arange(len(b_te)), b_te]))
                acc["clim"].append(-np.log(np.clip(clim[b_te], 1e-6, 1)))
                dts.append(drow[te])
            ll = {k: float(np.mean(np.concatenate(v))) for k, v in acc.items()}
            dd = np.concatenate(dts)
            row = dict(grp=lab, h=h, n=len(dd), n_dates=len(np.unique(dd)),
                       kappa_hat=np.mean(kap_used), c_fit=np.mean(c_used),
                       LL_clim=ll["clim"], LL_raw=ll["raw"], LL_kappa=ll["kappa"],
                       LL_cfit=ll["cfit"],
                       skill_raw=L.skill_score(ll["raw"], ll["clim"]),
                       skill_kappa=L.skill_score(ll["kappa"], ll["clim"]),
                       skill_cfit=L.skill_score(ll["cfit"], ll["clim"]))
            results.append(row)
            # per-date mean loss differences for date-clustered CIs
            for pair in (("kappa", "raw"), ("kappa", "cfit"), ("raw", "clim"), ("cfit", "clim")):
                d = np.concatenate(acc[pair[0]]) - np.concatenate(acc[pair[1]])
                s = pd.Series(d).groupby(dd).mean().sort_index()
                per_date[(lab, h, pair)] = s.values
    R = pd.DataFrame(results)
    print("\n  OOS log loss (lower better).  skill>0 = beats train-only climatology.")
    print(R.round(4).to_string(index=False))
    print("\n  Date-clustered 95% CI on log-loss DIFFERENCES (252-day block bootstrap on")
    print("  per-date mean loss).  positive => the FIRST rule is WORSE.")
    rows = []
    for (lab, h, pair), v in per_date.items():
        lo, hi = block_boot_ci(v, 252, 1500)
        rows.append(dict(grp=lab, h=h, comparison=f"{pair[0]} - {pair[1]}", n_dates=len(v),
                         mean_diff=float(np.mean(v)), lo=lo, hi=hi,
                         verdict=("first WORSE" if lo > 0 else
                                  "first BETTER" if hi < 0 else "ns")))
    print(pd.DataFrame(rows).round(5).to_string(index=False))

    # ---- calibration inside vol quintiles and per calendar year (Simpson check on the payoff)
    print("\n" + "=" * 120)
    print("C. SIMPSON CHECK on the calibration payoff: sd(z) OOS inside EWMA-vol quintiles and")
    print("   per calendar year, for each scaling rule (indices only = survivorship-free).")
    print("=" * 120)
    for h in (21, 63):
        flog = np.log(close.shift(-h) / close)
        ci = [close.columns.get_loc(c) for c in IDX]
        U = flog.iloc[:, ci].values / (s_e.iloc[:, ci].values * np.sqrt(h))
        ST = s_e.iloc[:, ci].values
        fin = np.isfinite(U) & np.isfinite(ST)
        yrow = np.repeat(years[:, None], len(ci), 1)
        recs, yrecs = [], []
        for y in yrs[5:]:
            tr, te = fin & (yrow < y), fin & (yrow == y)
            if tr.sum() < 2000 or te.sum() < 50:
                continue
            cfit = float(np.sqrt(np.mean(U[tr] ** 2)))
            kk = []
            for s in IDX:
                xx = lr[s].values[(years < y) & np.isfinite(lr[s].values)]
                v = hstd_fast(xx, h) / (hstd_fast(xx, 1) * np.sqrt(h))
                if np.isfinite(v):
                    kk.append(v)
            kap = float(np.mean(kk))
            edges = np.percentile(ST[tr], [20, 40, 60, 80])
            q = np.digitize(ST[te], edges)
            u = U[te]
            for nm, sc in (("raw", 1.0), ("kappa", kap), ("cfit", cfit)):
                yrecs.append(dict(h=h, y=y, rule=nm, n=int(te.sum()),
                                  sq=float(np.mean((u / sc) ** 2))))
                for k in range(5):
                    m = q == k
                    if m.sum() < 20:
                        continue
                    recs.append(dict(h=h, rule=nm, q=k + 1, n=int(m.sum()),
                                     sq=float(np.mean((u[m] / sc) ** 2))))
        rc = pd.DataFrame(recs)
        piv = rc.groupby(["rule", "q"]).apply(
            lambda x: np.sqrt(np.average(x.sq, weights=x.n)), include_groups=False).unstack()
        piv.columns = [f"Q{c}" for c in piv.columns]
        piv["mean|sd(z)-1|"] = (piv.iloc[:, :5] - 1).abs().mean(axis=1)
        print(f"\n  h={h}, INDICES: sd(z) by EWMA-vol quintile (Q1 calm .. Q5 excited); "
              f"perfect = 1.0 everywhere")
        print(piv.round(4).to_string())
        yc = pd.DataFrame(yrecs)
        yp = yc.pivot(index="y", columns="rule", values="sq").apply(np.sqrt)
        yp["yrs<1 raw"] = ""
        print(f"\n  h={h}, INDICES: sd(z) by test YEAR")
        print(yp[["raw", "kappa", "cfit"]].round(3).to_string())
        for nm in ("raw", "kappa", "cfit"):
            v = yp[nm].values
            print(f"    {nm:>6}: mean {v.mean():.4f}  median {np.median(v):.4f}  "
                  f"years with sd(z)<1: {int((v<1).sum())}/{len(v)}  "
                  f"mean|sd-1| {np.abs(v-1).mean():.4f}")

    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
