"""
_move_wf_zshape4.py — final deliverable: the exact tables to ship + OOS calibration proof.

Recommended parameterisation (validated in _move_wf_zshape3.py):
    z = r_h / sigma_hat_h,   u = (z - loc_h) / scale_h,   u ~ G_asset  (one shape per asset class,
    shared across all four horizons).  P(r_h > +thr) = 1 - G((thr/sigma_hat_h - loc_h)/scale_h).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import _move_lib as L
from _move_wf_zshape2 import (HORIZONS, THR, EmpCDF, actual_idx, build, norm_cdf_vec,
                              probs_from_cdf, qq_fit_t, stack3)

pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 100)
pd.set_option("display.float_format", lambda v: f"{v:9.4f}")

GRID = [0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70,
        75, 80, 85, 90, 92.5, 95, 97, 98, 99, 99.5, 99.75, 99.9]
UGRID = [-4, -3.5, -3, -2.5, -2.25, -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0,
         0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3, 3.5, 4]


def iqs(a):
    return (np.percentile(a, 75) - np.percentile(a, 25)) / 1.34898


def main():
    close, idx, singles, Z, SH, FWD, sig, rank = build()
    groups = [("INDEX", idx), ("SINGLES", singles)]

    print("=" * 140)
    print("A. PER-HORIZON LOCATION AND SCALE of z (loc = median, scale = IQR/1.34898). IN-SAMPLE.")
    par = {}
    for label, cols in groups:
        rows = []
        for h in HORIZONS:
            a = Z[h][cols].to_numpy(dtype=float).ravel()
            a = a[np.isfinite(a)]
            loc, sc = float(np.median(a)), float(iqs(a))
            par[(label, h)] = (loc, sc)
            rows.append(dict(h=h, n=a.size, loc=loc, scale=sc, loc_over_sqrth=loc / math.sqrt(h),
                             std_raw=a.std(ddof=1)))
        print(f"\n[{label}]")
        print(pd.DataFrame(rows).set_index("h").to_string())

    print("\n" + "=" * 140)
    print("B. THE SHAPE TABLE G_asset: quantiles of u = (z - loc_h)/scale_h, POOLED over "
          "h in {1,5,10,21}. IN-SAMPLE (validated OOS in part 3).")
    shape = {}
    for label, cols in groups:
        pool = []
        for h in HORIZONS:
            a = Z[h][cols].to_numpy(dtype=float).ravel()
            a = a[np.isfinite(a)]
            loc, sc = par[(label, h)]
            pool.append((a - loc) / sc)
        u = np.concatenate(pool)
        shape[label] = u
        q = np.percentile(u, GRID)
        print(f"\n[{label}] n={u.size:,}  quantiles of u")
        print(pd.Series(q, index=[f"p{g:g}" for g in GRID]).to_string())
        cdf = np.array([(u < k).mean() for k in UGRID])
        nrm = norm_cdf_vec(UGRID)
        t = pd.DataFrame({"u": UGRID, "G(u)": cdf, "normal": nrm,
                          "G/normal_left": np.where(np.array(UGRID) < 0, cdf / nrm, np.nan),
                          "1-G vs 1-normal": np.where(np.array(UGRID) > 0,
                                                      (1 - cdf) / (1 - nrm), np.nan)})
        print(f"[{label}] CDF form (this is the table to ship)")
        print(t.to_string(index=False))
        print(f"[{label}] asymmetry of the shape table: G(-k)/(1-G(k))")
        for k in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
            lo, hi = (u < -k).mean(), (u > k).mean()
            print(f"    k={k:4.1f}:  G(-k)={lo:.5f}  1-G(k)={hi:.5f}  ratio={lo/hi:6.3f}")

    print("\n" + "=" * 140)
    print("C. OOS CALIBRATION of the recommended recipe vs the normal, pooled over walk-forward "
          "test years (shape+loc+scale fit on years < Y only; regime scale for SINGLES).")
    for label, cols in groups:
        agg = {}
        for h in HORIZONS:
            s = stack3(Z, SH, FWD, rank, h, cols)
            s["ai"] = actual_idx(s["fwd"])
            years = sorted(set(s["year"].tolist()))
            keep = {m: [] for m in ("rec", "recreg", "nrm")}
            hits, ns = [], []
            for Y in years:
                if Y - years[0] < 5:
                    continue
                tr, te = s["year"] < Y, s["year"] == Y
                if tr.sum() < 2000 or te.sum() < 100:
                    continue
                ztr, shte, aite = s["z"][tr], s["sh"][te], s["ai"][te]
                loc, sc = float(np.median(ztr)), float(iqs(ztr))
                tf = qq_fit_t(ztr)
                G = EmpCDF((ztr - loc) / sc, tail_nu=tf["nu"])
                keep["rec"].append(probs_from_cdf(lambda k: G.cdf((k - loc) / sc), shte))
                keep["nrm"].append(probs_from_cdf(norm_cdf_vec, shte))
                # regime scale-only variant
                edges = [1 / 3, 2 / 3]
                trb, teb = np.digitize(s["rk"][tr], edges), np.digitize(s["rk"][te], edges)
                P = np.zeros((int(te.sum()), 4))
                for b in range(3):
                    m = teb == b
                    aa = ztr[trb == b]
                    if not m.any():
                        continue
                    if aa.size < 500:
                        P[m] = probs_from_cdf(lambda k: G.cdf((k - loc) / sc), shte[m])
                    else:
                        mb, sb = float(np.median(aa)), float(iqs(aa))
                        P[m] = probs_from_cdf(lambda k, mb=mb, sb=sb: G.cdf((k - mb) / sb), shte[m])
                keep["recreg"].append(P)
                hits.append(aite)
                ns.append(int(te.sum()))
            ai = np.concatenate(hits)
            row = {}
            for m, plist in keep.items():
                P = np.vstack(plist)
                row[f"{m}_ll"] = L.log_loss(P, ai)
                row[f"{m}_brier"] = L.brier_multi(P, ai)
                for bi, bn in ((0, "down_big"), (3, "up_big")):
                    row[f"{m}_ece_{bn}"] = L.ece(P[:, bi], ai == bi, 10)
                    row[f"{m}_mean_p_{bn}"] = P[:, bi].mean()
            clim = np.clip(L.climatology(ai, k=4), 1e-6, None); clim /= clim.sum()
            row["clim_ll_insample"] = L.log_loss(np.tile(clim, (len(ai), 1)), ai)
            row["obs_down_big"] = float((ai == 0).mean())
            row["obs_up_big"] = float((ai == 3).mean())
            row["n"] = len(ai)
            agg[h] = row
        t = pd.DataFrame(agg).T
        t.index.name = "h"
        print(f"\n[{label}] OOS pooled over test years")
        print(t[["n", "obs_down_big", "obs_up_big", "clim_ll_insample", "nrm_ll", "rec_ll",
                 "recreg_ll"]].to_string())
        print(t[["rec_mean_p_down_big", "rec_mean_p_up_big", "nrm_mean_p_down_big",
                 "nrm_mean_p_up_big"]].to_string())
        print(t[["nrm_ece_down_big", "rec_ece_down_big", "recreg_ece_down_big",
                 "nrm_ece_up_big", "rec_ece_up_big", "recreg_ece_up_big"]].to_string())

    print("\n" + "=" * 140)
    print("D. WORKED EXAMPLE with the shipped tables (IN-SAMPLE tables, illustrative).")
    for label in ("INDEX", "SINGLES"):
        u = shape[label]
        us = np.sort(u)
        for h, sd in ((5, 0.008), (5, 0.020), (10, 0.008), (21, 0.020)):
            loc, sc = par[(label, h)]
            shh = sd * math.sqrt(h)
            k = THR / shh
            uu = np.array([(-k - loc) / sc, (0 - loc) / sc, (k - loc) / sc])
            F = np.searchsorted(us, uu) / us.size
            pdn, p0, pup = F
            print(f"  {label:8s} h={h:2d} sigma_day={sd:.3f} -> sigma_h={shh:.4f} k={k:5.2f} | "
                  f"P(r<=-2%)={pdn:.4f}  P(-2%<r<=0)={p0-pdn:.4f}  P(0<r<+2%)={pup-p0:.4f}  "
                  f"P(r>=+2%)={1-pup:.4f}  down/up={pdn/(1-pup):.3f}")


if __name__ == "__main__":
    main()
