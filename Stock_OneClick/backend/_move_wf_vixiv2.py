"""
_move_wf_vixiv2.py — addendum: LEVEL-SPACE bias correction that is actually usable, plus the
pooled-coefficient single-name spec.

Why this file exists: the log-log fit in _move_wf_vixiv.py gives a slope/intercept, but turning a
log-space prediction back into a sigma you can push through a CDF needs a Jensen correction, and
exp(resid_sd^2/2) is WRONG here — the residual is not lognormal, it contains the vol ESTIMATOR's own
chi/|z| noise (at h=1 that noise alone has sd 1.11, so exp(s^2/2)=2.0 and the forecast comes out
~2x too big; the h=1 row of Part 3 confirms it: mean(actual/pred)=0.52). The fix is to calibrate ONE
multiplicative constant in level space on TRAINING data only, walk-forward. That constant is the
deliverable.

Two calibration targets, both reported:
  K_sig : mean(sigma_realized_train) / mean(sigma_pred_train)   -> unbiased sigma LEVEL
  K_z   : sqrt( mean( (r_h/(sigma_pred*sqrt h))^2 ) ) on train  -> makes std(z)=1, which is what
          P(|move| > 2%) actually depends on
Run: ../../vcp_env/bin/python _move_wf_vixiv2.py
"""
from __future__ import annotations

import warnings
from math import erf, sqrt

import numpy as np
import pandas as pd

import _move_data as D
import _move_lib as L
from _move_wf_vixiv import FLOOR, HORIZONS, MIN_TRAIN_YEARS, c4, clog, fwd_log_sigma, fwd_sigma, lg

warnings.filterwarnings("ignore")
rng = np.random.default_rng(11)
LO, HI = np.log(0.98), np.log(1.02)     # simple -2% / +2% as LOG thresholds (repo buckets simple)


def _Phi(x):
    return 0.5 * (1.0 + np.array([erf(v / sqrt(2)) for v in np.asarray(x, float)]))


def p_big_normal(sigma_h):
    """Model-implied P(simple r_h <= -2%) + P(simple r_h >= +2%) with mu=0, log-return normal."""
    s = np.asarray(sigma_h, float)
    return _Phi(LO / s) + (1.0 - _Phi(HI / s))


def boot_dr2_by_group(gkey, ra, rb, ta, tb, reps=2000, seed=3):
    """Block bootstrap of dR2 = R2_b - R2_a resampling GROUPS (dates or years).
    ra/rb = squared residuals, ta/tb = squared deviations from the no-lookahead benchmark.
    Precomputes per-group sums so a rep is a single bincount-style gather (fast)."""
    g = pd.DataFrame({"g": gkey, "ra": ra, "rb": rb, "ta": ta, "tb": tb}).groupby("g").sum()
    A = g.values
    r = np.random.default_rng(seed)
    n = len(A)
    out = np.empty(reps)
    for i in range(reps):
        s = A[r.integers(0, n, n)].sum(axis=0)
        out[i] = (1 - s[1] / s[3]) - (1 - s[0] / s[2])
    pt = (1 - A[:, 1].sum() / A[:, 3].sum()) - (1 - A[:, 0].sum() / A[:, 2].sum())
    return pt, out, n


def main():
    p = D.load()
    O, H, Lo, C = p["Open"], p["High"], p["Low"], p["Close"]
    keep = C.notna().sum()
    C = C.loc[:, keep >= 500]
    O, H, Lo = O[C.columns], H[C.columns], Lo[C.columns]
    iv = C["^VIX"] / 100.0 / np.sqrt(252.0)
    log_iv = np.log(iv)

    # ==================================================== PART 6: level-space WF calibration
    print("=" * 112)
    print("PART 6  LEVEL-SPACE, WALK-FORWARD BIAS CORRECTION (SPY — the survivorship-free benchmark)")
    print("  sigma_pred = K * exp(a + b*log sigma_iv [+ HAR terms]);  a, b and K all fit on PRIOR "
          "years only.")
    print("  'raw VIX' = sigma_iv itself, no fit, no K — i.e. what happens if you feed VIX in "
          "naked.")
    print("=" * 112)
    sym = "SPY"
    har = L.har_features(C[sym], H[sym], Lo[sym])
    harX = pd.DataFrame({"d": lg(har.rv_d), "w": lg(har.rv_w), "m": lg(har.rv_m)})
    gk = L.vol_garman_klass(O[sym], H[sym], Lo[sym], C[sym], 21)

    out = []
    for h in HORIZONS:
        ylog, ylev = fwd_log_sigma(C[sym], h), fwd_sigma(C[sym], h)
        rh = np.log(C[sym].shift(-h) / C[sym])
        specs = {
            "raw VIX (no fit)": None,
            "VIX":              pd.DataFrame({"liv": log_iv}),
            "GK21":             pd.DataFrame({"gk": lg(gk)}),
            "HAR":              harX,
            "VIX+GK21":         pd.DataFrame({"liv": log_iv, "gk": lg(gk)}),
            "VIX+HAR":          harX.assign(liv=log_iv),
        }
        for nm, X in specs.items():
            Xu = pd.DataFrame({"liv": log_iv}) if X is None else X
            df = pd.concat([ylog.rename("log"), ylev.rename("lev"), rh.rename("rh"), Xu],
                           axis=1).dropna()
            Xv = np.column_stack([np.ones(len(df)), df[list(Xu.columns)].values])
            rows = []
            for yr, tr, te in L.walk_forward_years(df.index, MIN_TRAIN_YEARS):
                if tr.sum() < 250 or te.sum() < 20:
                    continue
                if X is None:
                    ptr = np.exp(df["liv"].values[tr])          # = sigma_iv on train
                    pte = np.exp(df["liv"].values[te])
                    ks, kz = 1.0, 1.0
                else:
                    b, *_ = np.linalg.lstsq(Xv[tr], df["log"].values[tr], rcond=None)
                    ptr, pte = np.exp(Xv[tr] @ b), np.exp(Xv[te] @ b)
                    ks = df["lev"].values[tr].mean() / ptr.mean()
                    kz = np.sqrt(np.mean((df["rh"].values[tr] / (ptr * np.sqrt(h))) ** 2))
                rows.append(pd.DataFrame({"year": yr, "lev": df["lev"].values[te],
                                          "rh": df["rh"].values[te], "pred": pte,
                                          "K_sig": ks, "K_z": kz}, index=df.index[te]))
            res = pd.concat(rows)
            for kn in ("K_sig", "K_z"):
                pr = res.pred * res[kn]
                z = res.rh / (pr * np.sqrt(h))
                nn = slice(None, None, h)                       # non-overlapping rows
                act = (res.rh.iloc[nn] <= LO) | (res.rh.iloc[nn] >= HI)
                out.append({"h": h, "model": nm, "calib": kn,
                            "K_mean": res[kn].mean(), "K_2026": res[kn].iloc[-1],
                            "mean_act/pred": (res.lev / pr).mean(),
                            "meanAct/meanPred": res.lev.mean() / pr.mean(),
                            "sd_z": z.std(ddof=1), "sd_z_nonov": z.iloc[nn].std(ddof=1),
                            "P>2%_act": act.mean(),
                            "P>2%_model": p_big_normal(pr.iloc[nn] * np.sqrt(h)).mean(),
                            "n": len(res), "n_nonov": int(act.size)})
    o = pd.DataFrame(out)
    for kn in ("K_sig", "K_z"):
        print(f"\n--- calibration target: {kn} "
              f"({'unbiased sigma level' if kn == 'K_sig' else 'std(z)=1'}) ---")
        print(o[o.calib == kn].drop(columns=["calib"]).to_string(
            index=False, float_format=lambda x: f"{x:8.4f}"))
    print("\nNOTE 'raw VIX' rows have K=1 by construction — the gap of mean_act/pred from 1.0 IS "
          "the raw variance-risk-premium bias, and P>2%_model vs P>2%_act is what it costs you.")

    # ==================================================== PART 7: pooled single-name spec
    print("\n" + "=" * 112)
    print("PART 7  SINGLE NAMES, POOLED coefficients (what to use for a ticker with little history)")
    print("  log sigma_fwd(i,t) = alpha_i + b*log(GK21_i,t) + c*log(sigma_iv_t)")
    print("  alpha_i = per-stock TRAIN mean; b,c pooled across names, refit each year on prior "
          "years only. R^2 vs alpha_i (the per-stock expanding train mean) — a strong baseline.")
    print("=" * 112)
    cov = C.notna().sum()
    stocks = [s for s in C.columns if s not in ("SPY", "QQQ", "IWM", "DIA", "^GSPC", "^VIX")
              and cov[s] >= 1500]
    lg_gk = np.log(L.vol_garman_klass(O[stocks], H[stocks], Lo[stocks], C[stocks], 21)
                   .clip(lower=FLOOR))

    for h in HORIZONS:
        nu = 1 if h == 1 else h - 1
        proxy = (np.log(C[stocks]).diff().abs().shift(-1) if h == 1
                 else L.realized_vol_forward(C[stocks], h))
        ylog = np.log(proxy.clip(lower=FLOOR)) - clog(nu)
        ylev = proxy / c4(nu)
        d = pd.DataFrame({"y": ylog.stack(future_stack=True),
                          "lev": ylev.stack(future_stack=True),
                          "own": lg_gk.stack(future_stack=True)}).dropna()
        dates = d.index.get_level_values(0)
        d["liv"] = log_iv.reindex(dates).values
        d["sym"] = d.index.get_level_values(1).values
        d["date"] = dates

        parts = []
        cA, cB = [], []
        for yr, tr, te in L.walk_forward_years(pd.DatetimeIndex(d.date), MIN_TRAIN_YEARS):
            if tr.sum() < 5000 or te.sum() < 500:
                continue
            trd, ted = d[tr], d[te]
            ybar = trd.groupby("sym").y.mean()
            xbar = trd.groupby("sym").own.mean()
            lbar = trd.liv.mean()
            ok = ted.sym.isin(ybar.index).values
            ted = ted[ok]
            ytr = trd.y.values - ybar.reindex(trd.sym).values
            xtr = trd.own.values - xbar.reindex(trd.sym).values
            ltr = trd.liv.values - lbar
            xte = ted.own.values - xbar.reindex(ted.sym).values
            lte = ted.liv.values - lbar
            base_te = ybar.reindex(ted.sym).values
            bA, *_ = np.linalg.lstsq(xtr[:, None], ytr, rcond=None)
            bB, *_ = np.linalg.lstsq(np.column_stack([xtr, ltr]), ytr, rcond=None)
            pA = base_te + xte * bA[0]
            pB = base_te + xte * bB[0] + lte * bB[1]
            # level-space K from TRAIN only, for the pooled+VIX model
            pB_tr = ybar.reindex(trd.sym).values + xtr * bB[0] + ltr * bB[1]
            k_sig = trd.lev.values.mean() / np.exp(pB_tr).mean()
            cA.append(bA[0]); cB.append(bB)
            parts.append(pd.DataFrame({"date": ted.date.values, "y": ted.y.values,
                                       "lev": ted.lev.values, "ybar": base_te,
                                       "pA": pA, "pB": pB, "k_sig": k_sig, "year": yr}))
        a = pd.concat(parts, ignore_index=True)
        cB = np.array(cB)
        r2A = 1 - ((a.y - a.pA) ** 2).sum() / ((a.y - a.ybar) ** 2).sum()
        r2B = 1 - ((a.y - a.pB) ** 2).sum() / ((a.y - a.ybar) ** 2).sum()
        print(f"\n--- h={h} ---  n_obs={len(a):,}  n_dates={a.date.nunique():,}  "
              f"n_names={len(stocks)}  WF years={len(cB)}")
        print(f"  pooled own only : OOS R2 = {r2A:.4f}   b_own={np.mean(cA):.3f} "
              f"(sd across years {np.std(cA):.3f})")
        print(f"  pooled own+VIX  : OOS R2 = {r2B:.4f}   b_own={cB[:,0].mean():.3f} "
              f"(sd {cB[:,0].std():.3f})  c_VIX={cB[:,1].mean():.3f} (sd {cB[:,1].std():.3f})")
        # date-clustered bootstrap on non-overlapping dates
        nod = np.sort(a.date.unique())[::h]
        s = a[a.date.isin(nod)]
        pt, bs, ng = boot_dr2_by_group(s.date.values, (s.y - s.pA) ** 2, (s.y - s.pB) ** 2,
                                       (s.y - s.ybar) ** 2, (s.y - s.ybar) ** 2)
        print(f"  dR2 from VIX = {pt:+.4f}  95%CI[{np.percentile(bs,2.5):+.4f},"
              f"{np.percentile(bs,97.5):+.4f}]  P>0={(bs>0).mean():.3f}  "
              f"(date-clustered, non-overlapping: n_dates={ng}, n_obs={len(s):,})")
        pr = np.exp(a.pB) * a.k_sig
        print(f"  level: K_sig mean={a.k_sig.mean():.4f} (train-fit)  ->  OOS "
              f"meanAct/meanPred={a.lev.mean()/pr.mean():.4f}  mean(act/pred)="
              f"{(a.lev/pr).mean():.4f}   [uncorrected would be "
              f"{a.lev.mean()/np.exp(a.pB).mean():.4f}]")

    print("\nDONE2")


if __name__ == "__main__":
    main()
