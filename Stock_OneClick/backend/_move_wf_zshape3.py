"""
_move_wf_zshape3.py — part 3: the deliverable decisions.

1. The up/down ASYMMETRY FACTOR implied by the shipped z-table: ratio = F(-k)/(1-F(k)).
2. OOS: is the vol-regime effect SCALE or SHAPE? (regime-scale-only vs full regime table)
3. OOS: is ONE table enough? (a) one shape for all 4 horizons, (b) index table on singles and
   vice versa.
4. OOS: is the shape DRIFTING? expanding-window table vs last-10-years-only table.
All fits use years strictly before the test year.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import _move_lib as L
from _move_wf_zshape2 import (HORIZONS, THR, EmpCDF, actual_idx, build, norm_cdf_vec,
                              probs_from_cdf, qq_fit_t, stack3, t_unit_cdf_vec)

pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 80)
pd.set_option("display.float_format", lambda v: f"{v:10.4f}")

KA = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
BLOCK, NREP = 63, 2000
RNG = np.random.default_rng(7)


def iqs(a):
    return (np.percentile(a, 75) - np.percentile(a, 25)) / 1.34898


# ------------------------------------------------------------------ 1. asymmetry factor
def asym_from_table(Z, groups):
    print("=" * 130)
    print("1. ASYMMETRY FACTOR implied by the z-distribution: ratio(k) = P(z<-k) / P(z>+k).")
    print("   RAW = as shipped (drift included, what you actually predict).")
    print("   CTR = median-centred (pure shape asymmetry, 'beyond drift').")
    for label, cols in groups:
        rows = []
        for h in HORIZONS:
            a = Z[h][cols].to_numpy(dtype=float).ravel()
            a = a[np.isfinite(a)]
            med = np.median(a)
            r = dict(h=h, n=a.size, median=med)
            for k in KA:
                lo, hi = (a < -k).mean(), (a > k).mean()
                r[f"raw_{k}"] = lo / hi if hi else np.nan
                c = a - med
                lo2, hi2 = (c < -k).mean(), (c > k).mean()
                r[f"ctr_{k}"] = lo2 / hi2 if hi2 else np.nan
            rows.append(r)
        print(f"\n[{label}]")
        print(pd.DataFrame(rows).set_index("h").to_string())


def asym_ci(Z, groups):
    """Date-block bootstrap (moving blocks of BLOCK consecutive dates) on the MEDIAN-CENTRED ratio.
    Blocks cluster by date (kills cross-sectional correlation) and span >= h (kills overlap)."""
    print("\n" + "=" * 130)
    print(f"1b. Median-centred asymmetry ratio with {NREP} date-block bootstrap reps (block={BLOCK} "
          f"trading days -> clusters by DATE and absorbs the overlapping-window dependence)")
    for label, cols in groups:
        rows = []
        for h in HORIZONS:
            zz = Z[h][cols]
            med = float(np.nanmedian(zz.to_numpy(dtype=float)))
            c = zz - med
            d = {"n": c.notna().sum(axis=1)}
            for k in KA:
                d[f"lo{k}"] = (c < -k).sum(axis=1)
                d[f"hi{k}"] = (c > k).sum(axis=1)
            pdc = pd.DataFrame(d)
            pdc = pdc[pdc.n > 0]
            A = pdc.to_numpy(dtype=float)
            nd = len(pdc)
            nblk = int(math.ceil(nd / BLOCK))
            st = np.arange(0, nd - BLOCK + 1)
            cs = np.vstack([np.zeros(A.shape[1]), np.cumsum(A, axis=0)])
            blk = cs[st + BLOCK] - cs[st]
            tot = blk[RNG.integers(0, len(st), size=(NREP, nblk))].sum(axis=1)
            cl = list(pdc.columns)
            r = dict(h=h, n_obs=int(A[:, 0].sum()), n_dates=nd)
            for k in KA:
                il, ih = cl.index(f"lo{k}"), cl.index(f"hi{k}")
                pt = (A[:, il].sum() / max(A[:, ih].sum(), 1e-9))
                rb = tot[:, il] / np.maximum(tot[:, ih], 1e-12)
                l, u = np.percentile(rb, [2.5, 97.5])
                r[f"r{k}"] = pt
                r[f"ci{k}"] = f"[{l:.2f},{u:.2f}]"
                r[f"P>1_{k}"] = float((rb > 1).mean())
            rows.append(r)
        print(f"\n[{label}]")
        print(pd.DataFrame(rows).set_index("h").to_string())


# ------------------------------------------------------------------ OOS harness
def oos(Z, SH, FWD, rank, groups):
    print("\n" + "=" * 130)
    print("2-4. WALK-FORWARD OOS, 4-bucket log loss at thr=2%. Everything fit on years < Y.")
    S = {}
    for label, cols in groups:
        for h in HORIZONS:
            S[(label, h)] = stack3(Z, SH, FWD, rank, h, cols)
            S[(label, h)]["ai"] = actual_idx(S[(label, h)]["fwd"])
    rows = []
    pool_cache: dict = {}
    other_cache: dict = {}
    for label, cols in groups:
        years = sorted(set(S[(label, 1)]["year"].tolist()))
        for Y in years:
            if Y - years[0] < 5:
                continue
            for h in HORIZONS:
                s = S[(label, h)]
                tr, te = s["year"] < Y, s["year"] == Y
                if tr.sum() < 2000 or te.sum() < 100:
                    continue
                ztr, shte, aite = s["z"][tr], s["sh"][te], s["ai"][te]
                clim = np.clip(L.climatology(s["ai"][tr], k=4), 1e-6, None)
                clim /= clim.sum()
                base = L.log_loss(np.tile(clim, (te.sum(), 1)), aite)
                med, sc = np.median(ztr), iqs(ztr)
                tf = qq_fit_t(ztr)
                emp = EmpCDF(ztr, tail_nu=tf["nu"])
                row = dict(group=label, h=h, year=Y, n_test=int(te.sum()), clim_ll=base)
                row["emp_table"] = L.log_loss(probs_from_cdf(emp.cdf, shte), aite)

                # ---- rolling 10-year table (shape-drift test)
                tr10 = tr & (s["year"] >= Y - 10)
                if tr10.sum() > 2000:
                    e10 = EmpCDF(s["z"][tr10], tail_nu=tf["nu"])
                    row["emp_roll10"] = L.log_loss(probs_from_cdf(e10.cdf, shte), aite)

                # ---- regime: full table per tercile  vs  SCALE-ONLY per tercile
                edges = [1 / 3, 2 / 3]
                trb, teb = np.digitize(s["rk"][tr], edges), np.digitize(s["rk"][te], edges)
                shape = (ztr - med) / sc                          # pooled centred shape
                base_shape = EmpCDF(shape, tail_nu=tf["nu"])
                okr = all((trb == b).sum() >= 500 for b in range(3))
                if okr:
                    full = {b: EmpCDF(ztr[trb == b], tail_nu=tf["nu"]) for b in range(3)}
                    par = {b: (np.median(ztr[trb == b]), iqs(ztr[trb == b])) for b in range(3)}
                    Pf = np.zeros((te.sum(), 4)); Ps = np.zeros((te.sum(), 4))
                    for b in range(3):
                        m = teb == b
                        if not m.any():
                            continue
                        Pf[m] = probs_from_cdf(full[b].cdf, shte[m])
                        mb, sb = par[b]
                        Ps[m] = probs_from_cdf(lambda k, mb=mb, sb=sb:
                                               base_shape.cdf((k - mb) / sb), shte[m])
                    row["emp_regime_full"] = L.log_loss(Pf, aite)
                    row["emp_regime_scale"] = L.log_loss(Ps, aite)

                # ---- ONE SHAPE for all horizons: pool centred/rescaled z over all h
                key = (label, Y)
                if key not in pool_cache:
                    pool = []
                    for hh in HORIZONS:
                        s2 = S[(label, hh)]
                        z2 = s2["z"][s2["year"] < Y]
                        if z2.size > 1000:
                            pool.append((z2 - np.median(z2)) / iqs(z2))
                    pool_cache[key] = (EmpCDF(np.concatenate(pool), tail_nu=tf["nu"])
                                       if len(pool) == len(HORIZONS) else None)
                ph = pool_cache[key]
                if ph is not None:
                    row["emp_one_shape_allh"] = L.log_loss(
                        probs_from_cdf(lambda k: ph.cdf((k - med) / sc), shte), aite)

                # ---- CROSS-ASSET: use the OTHER group's centred shape, own median/scale
                other = "SINGLES" if label != "SINGLES" else "INDICES(5)"
                okey = (other, h, Y)
                if okey not in other_cache:
                    other_cache[okey] = None
                    if (other, h) in S:
                        s3 = S[(other, h)]
                        z3 = s3["z"][s3["year"] < Y]
                        if z3.size > 2000:
                            other_cache[okey] = EmpCDF((z3 - np.median(z3)) / iqs(z3),
                                                       tail_nu=tf["nu"])
                osh = other_cache[okey]
                if osh is not None:
                    row["emp_other_shape"] = L.log_loss(
                        probs_from_cdf(lambda k: osh.cdf((k - med) / sc), shte), aite)

                # ---- symmetric reference: mirror own shape about its median (kills asymmetry)
                sym = np.concatenate([shape, -shape])
                syme = EmpCDF(sym, tail_nu=tf["nu"])
                row["emp_symmetrised"] = L.log_loss(
                    probs_from_cdf(lambda k: syme.cdf((k - med) / sc), shte), aite)
                row["t_qqfit"] = L.log_loss(
                    probs_from_cdf(lambda k: t_unit_cdf_vec((k - tf["loc"]) / tf["scale"], tf["nu"]),
                                   shte), aite)
                row["normal01"] = L.log_loss(probs_from_cdf(norm_cdf_vec, shte), aite)
                row["nu"] = tf["nu"]
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_pickle("/tmp/zshape3_wf.pkl")
    mods = ["normal01", "emp_symmetrised", "t_qqfit", "emp_table", "emp_one_shape_allh",
            "emp_other_shape", "emp_roll10", "emp_regime_full", "emp_regime_scale"]
    mods = [m for m in mods if m in df.columns]
    for label in df.group.unique():
        d = df[df.group == label].dropna(subset=mods)
        print(f"\n--- [{label}] OOS skill vs train-climatology (1 - LL/clim_LL), "
              f"n_years per h = {d.groupby('h').size().to_dict()} ---")
        sk = pd.DataFrame({m: 1 - d.groupby("h")[m].mean() / d.groupby("h").clim_ll.mean()
                           for m in mods})
        print(sk.to_string())
        print(f"[{label}] paired per-test-year t-stat of (emp_table - model) log loss "
              f"(>0 => emp_table BETTER)")
        tt = {}
        for m in mods:
            if m == "emp_table":
                continue
            dd = d[m] - d.emp_table
            g = d.assign(dd=dd).groupby("h").dd
            tt[m] = g.mean() / (g.std(ddof=1) / np.sqrt(g.size()))
        print(pd.DataFrame(tt).to_string())
        print(f"[{label}] mean fitted t nu by h: {d.groupby('h').nu.mean().round(2).to_dict()}")
    return df


def main():
    close, idx, singles, Z, SH, FWD, sig, rank = build()
    groups = [("INDICES(5)", idx), ("SPY", ["SPY"]), ("SINGLES", singles)]
    asym_from_table(Z, groups)
    asym_ci(Z, [("INDICES(5)", idx), ("SPY", ["SPY"]), ("SINGLES", singles)])
    oos(Z, SH, FWD, rank, [("INDICES(5)", idx), ("SPY", ["SPY"]), ("SINGLES", singles)])


if __name__ == "__main__":
    main()
