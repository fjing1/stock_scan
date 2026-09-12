"""
_move_wf_verify_tailloc2.py — addendum to _move_wf_verify_tailloc.py.

Pins down the statistical status of the SIMPLER CONTROL that lands attack 3:
   normal_med1 = plain symmetric normal, scale left at EXACTLY 1 (i.e. sigma_hat = ewma(0.94)
   * sqrt(h) untouched), only the LOCATION shifted by the train median of z.
Questions:
  A  does dropping the 20th test year (2026) reproduce the claim's exact -0.0209 for INDEX5?
  B  is normal_med1's win over climatology significant under date-clustered block bootstrap and a
     per-test-year sign test?  is it significantly WORSE than emp_table (the claim's hero)?
  C  paired per-year comparison normal_med1 vs emp_table.
  D  how much drift does the location term actually carry, in plain units?
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import _move_lib as L
from _move_wf_verify_tailloc import (HORIZONS, IDX5, MIN_TEST_ROWS, MIN_TRAIN_ROWS,
                                     MIN_TRAIN_YEARS, THR, Emp, aidx, block_boot, build, iqs_of,
                                     ncdf, probs, sign_p, stack)
from _move_wf_zshape2 import qq_fit_t

pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", lambda v: f"{v:10.4f}")


def run(S, h, thr=THR):
    z, sh, fwd, yr, pos = S["z"], S["sh"], S["fwd"], S["year"], S["pos"]
    ai = aidx(fwd, thr)
    years = sorted(set(int(v) for v in yr))
    y0 = years[0]
    rows, obs = [], []
    for Y in years:
        if Y - y0 < MIN_TRAIN_YEARS:
            continue
        tr, te = yr < Y, yr == Y
        if tr.sum() < MIN_TRAIN_ROWS or te.sum() < MIN_TEST_ROWS:
            continue
        ztr = z[tr]
        shte, aite, n = sh[te], ai[te], int(te.sum())
        clim = np.clip(L.climatology(ai[tr], k=4), 1e-6, None)
        clim /= clim.sum()
        Pc = np.tile(clim, (n, 1))
        med = float(np.median(ztr))
        tf = qq_fit_t(ztr)
        emp = Emp(ztr, nu=tf["nu"])
        P = {"normal01": probs(ncdf, shte, thr),
             "normal_med1": probs(lambda k: ncdf(k - med), shte, thr),
             "emp_table": probs(emp.cdf, shte, thr)}
        row = dict(year=Y, n_test=n, clim_ll=L.log_loss(Pc, aite), tr_med_z=med,
                   tr_med_z_over_sqrth=med / math.sqrt(h),
                   tr_iqs_z=iqs_of(ztr), med_sh=float(np.median(shte)))
        for m, p in P.items():
            row[m] = L.log_loss(p, aite)
        rows.append(row)
        d = {"pos": pos[te], "ll_clim": -np.log(np.clip(Pc[np.arange(n), aite], 1e-12, None))}
        for m, p in P.items():
            d[f"ll_{m}"] = -np.log(np.clip(p[np.arange(n), aite], 1e-12, None))
        obs.append(pd.DataFrame(d))
    return pd.DataFrame(rows), pd.concat(obs, ignore_index=True)


def main():
    close, idx, singles, sig, Z, SH, FWD = build()
    groups = {"SPY": ["SPY"], "INDEX5": idx, "SINGLES": singles}
    print("=" * 112)
    print("(A) does excluding the 20th test year (2026) reproduce the claim's exact numbers?")
    for g in ("INDEX5", "SPY"):
        S = stack(Z, SH, FWD, 21, groups[g])
        d, o = run(S, 21)
        for lab, dd in (("all years", d), ("ex-2026", d[d.year != 2026])):
            print(f"  [{g}] {lab:10s} n_years={len(dd):2d} n_obs={int(dd.n_test.sum()):7d}  "
                  f"normal01={1 - dd.normal01.mean() / dd.clim_ll.mean():+.4f}  "
                  f"normal_med1={1 - dd.normal_med1.mean() / dd.clim_ll.mean():+.4f}  "
                  f"emp_table={1 - dd.emp_table.mean() / dd.clim_ll.mean():+.4f}")
    print("  claim: INDEX5 normal01 -0.0209 (19 test years, 24,655 obs); SPY -0.0430 (16y, 3,923)")

    print("\n" + "=" * 112)
    print("(B/C) statistical status of normal_med1 -- a SYMMETRIC normal with scale exactly 1")
    for g in ("INDEX5", "SPY", "SINGLES"):
        S = stack(Z, SH, FWD, 21, groups[g])
        d, o = run(S, 21)
        n = len(d)
        print(f"\n  [{g}] h=21, {n} test years, {int(d.n_test.sum())} overlapping obs")
        for m in ("normal01", "normal_med1", "emp_table"):
            gain = d.clim_ll - d[m]
            k = int((gain > 0).sum())
            pdte = pd.DataFrame({"pos": o.pos, "m": o[f"ll_{m}"], "c": o.ll_clim}).groupby("pos").mean()
            lo, hi, sk = block_boot(pdte.m.to_numpy(), pdte.c.to_numpy())
            print(f"    vs CLIM  {m:12s} yr_skill={1 - d[m].mean() / d.clim_ll.mean():+.4f} "
                  f"t_yr={gain.mean() / (gain.std(ddof=1) / math.sqrt(n)):+5.2f} "
                  f"yrs={k:2d}/{n} p={sign_p(k, n):.3f} "
                  f"date-boot skill={sk:+.4f} CI[{lo:+.4f},{hi:+.4f}]")
        # paired: normal_med1 vs emp_table
        gain = d.emp_table - d.normal_med1          # >0 means normal_med1 is BETTER
        k = int((gain > 0).sum())
        pdte = pd.DataFrame({"pos": o.pos, "m": o.ll_normal_med1,
                             "c": o.ll_emp_table}).groupby("pos").mean()
        lo, hi, sk = block_boot(pdte.m.to_numpy(), pdte.c.to_numpy())
        print(f"    PAIRED   normal_med1 vs emp_table: mean_ll_diff={gain.mean():+.5f} "
              f"t_yr={gain.mean() / (gain.std(ddof=1) / math.sqrt(n)):+5.2f} "
              f"normal_med1 wins {k}/{n} years p={sign_p(k, n):.3f}  "
              f"date-boot rel.skill={sk:+.4f} CI[{lo:+.4f},{hi:+.4f}]")

    print("\n" + "=" * 112)
    print("(D) what the location term actually is, in plain units (train-fitted, per test year)")
    for g in ("INDEX5", "SPY"):
        S = stack(Z, SH, FWD, 21, groups[g])
        d, _ = run(S, 21)
        print(f"\n  [{g}] median train z at h=21: {d.tr_med_z.mean():.4f} "
              f"(= {d.tr_med_z_over_sqrth.mean():.4f} per sqrt(day)); train IQR-scale of z: "
              f"{d.tr_iqs_z.mean():.4f} (normal ref 1.0)")
        print(f"        median sigma_hat_21 on test rows: {d.med_sh.mean():.4f} "
              f"-> the location shift is worth "
              f"{d.tr_med_z.mean() * d.med_sh.mean() * 100:.2f}% of 21-day return, and the")
        print(f"        scale error of ewma*sqrt(21) is {100 * (d.tr_iqs_z.mean() - 1):+.1f}% "
              f"(IQR-based).")
        print("        i.e. the whole defect is a ~1%/month drift the zero-mean normal refuses to "
              "carry.")

    print("\n" + "=" * 112)
    print("(E) all four horizons: normal_med1 (symmetric, scale=1, location fitted) vs emp_table")
    for g in ("INDEX5", "SPY", "SINGLES"):
        rows = []
        for h in HORIZONS:
            S = stack(Z, SH, FWD, h, groups[g])
            d, _ = run(S, h)
            rows.append(dict(h=h, n_years=len(d), clim_ll=d.clim_ll.mean(),
                             normal01=1 - d.normal01.mean() / d.clim_ll.mean(),
                             normal_med1=1 - d.normal_med1.mean() / d.clim_ll.mean(),
                             emp_table=1 - d.emp_table.mean() / d.clim_ll.mean()))
        print(f"\n  [{g}]")
        print(pd.DataFrame(rows).set_index("h").to_string())


if __name__ == "__main__":
    main()
