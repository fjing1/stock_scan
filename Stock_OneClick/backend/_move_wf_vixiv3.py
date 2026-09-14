"""
_move_wf_vixiv3.py — final piece: (8) deployable coefficient tables, (9) does SPX-derived VIX also
help QQQ/IWM/DIA (the halfway case between SPX and a single name), (10) the no-fixed-effect pooled
single-name formula you can actually apply to a ticker on day one.

Run: ../../vcp_env/bin/python _move_wf_vixiv3.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import _move_data as D
import _move_lib as L
from _move_wf_vixiv import (FLOOR, HORIZONS, MIN_TRAIN_YEARS, c4, clog, fwd_log_sigma, fwd_sigma,
                            lg, oos_r2, wf_ols)
from _move_wf_vixiv2 import boot_dr2_by_group

warnings.filterwarnings("ignore")


def main():
    p = D.load()
    O, H, Lo, C = p["Open"], p["High"], p["Low"], p["Close"]
    keep = C.notna().sum()
    C = C.loc[:, keep >= 500]
    O, H, Lo = O[C.columns], H[C.columns], Lo[C.columns]
    iv = C["^VIX"] / 100.0 / np.sqrt(252.0)
    log_iv = np.log(iv)

    # ==================================================== PART 9: other indices
    print("=" * 108)
    print("PART 9  DOES SPX-DERIVED VIX HELP THE *OTHER* INDICES? (walk-forward OOS R^2 on "
          "log fwd sigma)")
    print("=" * 108)
    rows = []
    for sym in ["SPY", "QQQ", "IWM", "DIA"]:
        har = L.har_features(C[sym], H[sym], Lo[sym])
        harX = pd.DataFrame({"d": lg(har.rv_d), "w": lg(har.rv_w), "m": lg(har.rv_m)})
        gk = lg(L.vol_garman_klass(O[sym], H[sym], Lo[sym], C[sym], 21))
        for h in HORIZONS:
            y = fwd_log_sigma(C[sym], h)
            r = {"sym": sym, "h": h}
            store = {}
            for nm, X in (("VIX only", pd.DataFrame({"liv": log_iv})),
                          ("GK21", pd.DataFrame({"gk": gk})),
                          ("HAR", harX),
                          ("VIX+GK21", pd.DataFrame({"liv": log_iv, "gk": gk})),
                          ("VIX+HAR", harX.assign(liv=log_iv))):
                res, cf = wf_ols(X, y)
                r[nm] = oos_r2(res)
                store[nm] = res
                if nm == "VIX+HAR":
                    r["c_VIX_WFmean"] = cf["liv"].mean()
            a, b = store["HAR"], store["VIX+HAR"]
            m = a.merge(b, on=["date", "year"], suffixes=("_a", "_b")).iloc[::h]
            pt, bs, ng = boot_dr2_by_group(m.year.values, (m.y_a - m.pred_a) ** 2,
                                           (m.y_b - m.pred_b) ** 2,
                                           (m.y_a - m.ybar_tr_a) ** 2,
                                           (m.y_b - m.ybar_tr_b) ** 2, reps=2000)
            r["dR2 VIX|HAR"] = pt
            r["CI_lo"], r["CI_hi"] = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
            r["P>0"] = (bs > 0).mean()
            r["n_nonov"] = len(m)
            rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:7.4f}"))
    print("(dR2 and CI computed on NON-OVERLAPPING rows with a calendar-year block bootstrap, "
          "2000 reps.)")

    # ==================================================== PART 8: production coefficients
    print("\n" + "=" * 108)
    print("PART 8  DEPLOYABLE COEFFICIENTS — fit on the FULL sample 2001-09..2026-09 (IN-SAMPLE by")
    print("        construction; this is the 'latest fit' you would ship, and Part 2/6 already")
    print("        showed the same coefficients are stable walk-forward).")
    print("        sigma_hat_daily = K * exp(a + b*log(sigma_iv) [+ c_d,c_w,c_m * log HAR terms])")
    print("=" * 108)
    for sym in ["SPY", "^GSPC"]:
        har = L.har_features(C[sym], H[sym], Lo[sym])
        harX = pd.DataFrame({"d": lg(har.rv_d), "w": lg(har.rv_w), "m": lg(har.rv_m)})
        for h in HORIZONS:
            ylog, ylev = fwd_log_sigma(C[sym], h), fwd_sigma(C[sym], h)
            for nm, X in (("VIX", pd.DataFrame({"liv": log_iv})),
                          ("VIX+HAR", harX.assign(liv=log_iv))):
                d = pd.concat([ylog.rename("yl"), ylev.rename("lv"), X], axis=1).dropna()
                Xv = np.column_stack([np.ones(len(d)), d[list(X.columns)].values])
                b, *_ = np.linalg.lstsq(Xv, d.yl.values, rcond=None)
                pr = np.exp(Xv @ b)
                K = d.lv.values.mean() / pr.mean()
                names = ["a"] + list(X.columns)
                cs = "  ".join(f"{n}={v:+.4f}" for n, v in zip(names, b))
                print(f"  {sym:>6} h={h:>2} {nm:<8}: {cs}  K_sig={K:.4f}  n={len(d)}")

    # ==================================================== PART 10: no-FE pooled single-name
    print("\n" + "=" * 108)
    print("PART 10  SINGLE-NAME FORMULA WITH **NO PER-STOCK HISTORY REQUIRED** (no fixed effect):")
    print("         log sigma_fwd(i,t) = a + b*log(GK21_i,t) + c*log(sigma_iv_t)")
    print("         Walk-forward pooled fit; R^2 reported against BOTH the pooled train mean and")
    print("         the per-stock train mean (the harder baseline).")
    print("=" * 108)
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
        dt = d.index.get_level_values(0)
        d["liv"] = log_iv.reindex(dt).values
        d["sym"] = d.index.get_level_values(1).values
        d["date"] = dt
        parts, coefs = [], []
        for yr, tr, te in L.walk_forward_years(pd.DatetimeIndex(d.date), MIN_TRAIN_YEARS):
            if tr.sum() < 5000 or te.sum() < 500:
                continue
            trd, ted = d[tr], d[te]
            ybar_i = trd.groupby("sym").y.mean()
            for nm, cols in (("own", ["own"]), ("own+VIX", ["own", "liv"])):
                Xtr = np.column_stack([np.ones(len(trd))] + [trd[c].values for c in cols])
                b, *_ = np.linalg.lstsq(Xtr, trd.y.values, rcond=None)
                Xte = np.column_stack([np.ones(len(ted))] + [ted[c].values for c in cols])
                ptr = np.exp(Xtr @ b)
                parts.append(pd.DataFrame({
                    "date": ted.date.values, "y": ted.y.values, "lev": ted.lev.values,
                    "pred": Xte @ b, "ybar_pool": trd.y.values.mean(),
                    "ybar_i": ybar_i.reindex(ted.sym).values,
                    "K": trd.lev.values.mean() / ptr.mean(), "model": nm, "year": yr}))
                coefs.append(dict(zip(["a"] + cols, b), model=nm, year=yr))
        a = pd.concat(parts, ignore_index=True).dropna(subset=["ybar_i"])
        cf = pd.DataFrame(coefs)
        print(f"\n--- h={h} --- n_obs={len(a)//2:,} n_names={len(stocks)} WF_years="
              f"{cf.year.nunique()}")
        for nm, g in a.groupby("model"):
            r2p = 1 - ((g.y - g.pred) ** 2).sum() / ((g.y - g.ybar_pool) ** 2).sum()
            r2i = 1 - ((g.y - g.pred) ** 2).sum() / ((g.y - g.ybar_i) ** 2).sum()
            c = cf[cf.model == nm]
            cs = "  ".join(f"{k}={c[k].mean():+.4f}(sd{c[k].std():.3f})"
                           for k in ["a", "own"] + (["liv"] if nm == "own+VIX" else []))
            pr = np.exp(g.pred) * g.K
            print(f"  {nm:<8} R2_vs_pool_mean={r2p:.4f}  R2_vs_perstock_mean={r2i:.4f}  {cs}")
            print(f"           K_sig mean={g.K.mean():.4f} -> OOS meanAct/meanPred="
                  f"{g.lev.mean()/pr.mean():.4f}")
        # full-sample production fit
        Xf = np.column_stack([np.ones(len(d)), d.own.values, d.liv.values])
        bf, *_ = np.linalg.lstsq(Xf, d.y.values, rcond=None)
        Kf = d.lev.values.mean() / np.exp(Xf @ bf).mean()
        print(f"  PRODUCTION (full-sample, IN-SAMPLE): a={bf[0]:+.4f} b_own={bf[1]:+.4f} "
              f"c_VIX={bf[2]:+.4f}  K_sig={Kf:.4f}  n={len(d):,}")

    print("\nDONE3")


if __name__ == "__main__":
    main()
