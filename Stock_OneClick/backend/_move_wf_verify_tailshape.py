"""
_move_wf_verify_tailshape.py — ADVERSARIAL verification of the "tail-shape" claim.

CLAIM UNDER TEST
  "A normal shape on top of sigma_hat = ewma(0.94)*sqrt(h) is not merely imprecise for indices at
   long horizons -- it is WORSE than the unconditional climatology baseline out of sample, while
   the empirical ASYMMETRIC z-table beats it."
  Offered: h=21 walk-forward OOS 4-bucket log loss at thr=2%, params fit on years < Y:
      INDEX(5) normal01 skill = -0.0209 ; SPY = -0.0430
      INDEX(5) emp_table skill = +0.0150 ; SPY = +0.0090
      19 / 16 test years ; 24,655 / 3,923 test obs
      INDEX h=21 observed freqs down_big 0.2233 / up_big 0.4559 vs normal's forced 0.3195/0.3195
      OOS ECE h=21: 0.0968 -> 0.0273 (down_big), 0.1363 -> 0.0406 (up_big)

ATTACKS RUN
  0  reproduce the offered numbers exactly
  1  LOOKAHEAD: expanding window already; add PURGE of train rows whose h-day forward window
     spills into the test year (real leak: at h=21 the last 21 train rows' labels live in year Y)
  2  FAKE SAMPLE SIZE: h-day windows overlap and the 5 "indices" are ~1.0 correlated.
     -> collapse to one obs per DATE, moving-block bootstrap (block >= h), and
        non-overlapping stride-h subsamples at every phase offset
  3  SIMPLER EXPLANATION: normal01 bundles SHAPE with LOCATION(=0) and SCALE(=1).
     Controls that isolate them: N(med,1), N(med,iqs), N(mean,std), and a SYMMETRISED empirical
     table with the same fitted loc/scale.  Also: how far out in the distribution is thr=2% at
     h=21 -- is this a "tail" claim at all?
  4  FRAGILITY: drop 2008 & 2020 (as test years and from train); per-test-year skill; split the
     pooled INDEX(5) into its 5 members
  5  CALIBRATION SUBGROUPS: OOS ECE for down_big/up_big overall, within vol quintiles, and within
     calendar years -- including the ECE of CLIMATOLOGY itself (the honest reference for an ECE
     comparison, which the claim omits)

Volatility, bucketing, scoring and splitting all come from _move_lib so the numbers stay
comparable with the original study.  Nothing here is downloaded; the cached panel is reused.
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

import _move_lib as L
from _move_wf_zshape2 import (HORIZONS, IDX, THR, EmpCDF, actual_idx, build, norm_cdf_vec,
                              probs_from_cdf, qq_fit_t, t_unit_cdf_vec)

pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 120)
pd.set_option("display.float_format", lambda v: f"{v:10.4f}")

RNG = np.random.default_rng(11)
NREP = 4000
MODELS = ["normal01", "normal_loc", "normal_locscale", "normal_mustd", "t_qqfit",
          "emp_sym", "emp_table", "emp_table_purged"]
DN, UP = 0, 3          # bucket indices for down_big / up_big


def iqs(a):
    return (np.percentile(a, 75) - np.percentile(a, 25)) / 1.34898


# ------------------------------------------------------------------ long-form stack keeping DATE
def stack_d(Z, SH, FWD, rank, h, cols):
    z = Z[h][cols]
    m = z.notna().to_numpy()
    dates = z.index
    dord = np.repeat(np.arange(len(dates))[:, None], len(cols), axis=1)
    yr = np.repeat(dates.year.to_numpy()[:, None], len(cols), axis=1)
    sid = np.repeat(np.arange(len(cols))[None, :], len(dates), axis=0)
    out = {}
    for name, df in (("z", z), ("sh", SH[h][cols]), ("fwd", FWD[h][cols]), ("rk", rank[cols])):
        out[name] = df.to_numpy(dtype=float)[m]
    out["year"] = yr[m]
    out["dord"] = dord[m]
    out["sid"] = sid[m]
    g = (np.isfinite(out["z"]) & np.isfinite(out["sh"])
         & np.isfinite(out["fwd"]) & np.isfinite(out["rk"]))
    return {k: v[g] for k, v in out.items()}


# ------------------------------------------------------------------ the OOS engine
def run_oos(Z, SH, FWD, rank, group, cols, horizons=HORIZONS, keep_obs=True):
    """Walk-forward by calendar year, everything fit on years strictly < Y.
    Returns (per-test-year DataFrame, per-observation dict of arrays)."""
    rows, obs = [], {}
    for h in horizons:
        s = stack_d(Z, SH, FWD, rank, h, cols)
        ai = actual_idx(s["fwd"])
        years = sorted(set(s["year"].tolist()))
        acc = {k: [] for k in ("ll_clim", "ai", "dord", "year", "rk", "sh", "p_dn_clim",
                               "p_up_clim")}
        for m in MODELS:
            acc[f"ll_{m}"] = []
            acc[f"pdn_{m}"] = []
            acc[f"pup_{m}"] = []
        for Y in years:
            tr, te = s["year"] < Y, s["year"] == Y
            if tr.sum() < 2000 or te.sum() < 100 or Y - years[0] < 5:
                continue
            ztr, shte, aite = s["z"][tr], s["sh"][te], ai[te]
            nte = int(te.sum())
            clim = np.clip(L.climatology(ai[tr], k=4), 1e-6, None)
            clim /= clim.sum()
            Pc = np.tile(clim, (nte, 1))
            ll_clim = -np.log(np.clip(Pc[np.arange(nte), aite], 1e-12, 1.0))

            med, sc = float(np.median(ztr)), float(iqs(ztr))
            mu, sd = float(ztr.mean()), float(ztr.std(ddof=1))
            tf = qq_fit_t(ztr)
            emp = EmpCDF(ztr, tail_nu=tf["nu"])
            shp = (ztr - med) / sc
            symE = EmpCDF(np.concatenate([shp, -shp]), tail_nu=tf["nu"])

            # ---- PURGED train: drop rows whose h-day forward window reaches into year Y
            first_te = s["dord"][te].min()
            trp = tr & (s["dord"] <= first_te - h - 1)
            if trp.sum() >= 2000:
                ztrp = s["z"][trp]
                tfp = qq_fit_t(ztrp)
                empp = EmpCDF(ztrp, tail_nu=tfp["nu"])
                climp = np.clip(L.climatology(ai[trp], k=4), 1e-6, None)
                climp /= climp.sum()
            else:
                empp, climp = emp, clim

            cdfs = {
                "normal01":         norm_cdf_vec,
                "normal_loc":       lambda k: norm_cdf_vec(k - med),
                "normal_locscale":  lambda k: norm_cdf_vec((k - med) / sc),
                "normal_mustd":     lambda k: norm_cdf_vec((k - mu) / sd),
                "t_qqfit":          lambda k: t_unit_cdf_vec((k - tf["loc"]) / tf["scale"], tf["nu"]),
                "emp_sym":          lambda k: symE.cdf((k - med) / sc),
                "emp_table":        emp.cdf,
                "emp_table_purged": empp.cdf,
            }
            row = dict(group=group, h=h, year=Y, n_test=nte, n_train=int(tr.sum()),
                       n_train_purged=int(trp.sum()), clim_ll=float(ll_clim.mean()),
                       clim_p_dn=float(clim[DN]), clim_p_up=float(clim[UP]),
                       obs_f_dn=float((aite == DN).mean()), obs_f_up=float((aite == UP).mean()),
                       med_z=med, iqs_z=sc, mean_z=mu, std_z=sd, nu=tf["nu"],
                       med_k_up=float(np.median(THR / shte)))
            for name, fn in cdfs.items():
                P = probs_from_cdf(fn, shte)
                ll = -np.log(np.clip(P[np.arange(nte), aite], 1e-12, 1.0))
                row[name] = float(ll.mean())
                if keep_obs:
                    acc[f"ll_{name}"].append(ll.astype(np.float32))
                    acc[f"pdn_{name}"].append(P[:, DN].astype(np.float32))
                    acc[f"pup_{name}"].append(P[:, UP].astype(np.float32))
            row["clim_purged_ll"] = float(-np.log(np.clip(
                np.tile(climp, (nte, 1))[np.arange(nte), aite], 1e-12, 1.0)).mean())
            rows.append(row)
            if keep_obs:
                acc["ll_clim"].append(ll_clim.astype(np.float32))
                acc["ai"].append(aite.astype(np.int8))
                acc["dord"].append(s["dord"][te])
                acc["year"].append(s["year"][te])
                acc["rk"].append(s["rk"][te].astype(np.float32))
                acc["sh"].append(s["sh"][te].astype(np.float32))
                acc["p_dn_clim"].append(np.full(nte, clim[DN], np.float32))
                acc["p_up_clim"].append(np.full(nte, clim[UP], np.float32))
        if keep_obs and acc["ai"]:
            obs[h] = {k: np.concatenate(v) for k, v in acc.items() if v}
    return pd.DataFrame(rows), obs


def skill_table(df, label):
    out = {}
    for m in MODELS:
        out[m] = 1 - df.groupby("h")[m].mean() / df.groupby("h").clim_ll.mean()
    t = pd.DataFrame(out)
    t["n_years"] = df.groupby("h").size()
    t["n_test_obs"] = df.groupby("h").n_test.sum()
    print(f"\n[{label}] OOS skill vs train-climatology, equal weight per TEST YEAR "
          f"(this is the original study's aggregation)")
    print(t.to_string())
    return t


# ------------------------------------------------------------------ uncertainty by DATE clustering
def per_date(o, key):
    """Collapse a per-observation array to one value per DATE (mean across symbols)."""
    d = pd.DataFrame({"d": o["dord"], "v": o[key].astype(float)})
    g = d.groupby("d").v.mean()
    return g.index.to_numpy(), g.to_numpy()


def block_boot(x, block, nrep=NREP, rng=RNG):
    """Moving-block bootstrap on a date-indexed series. block >= h absorbs window overlap."""
    n = len(x)
    if n < 2 * block:
        return np.nan, np.nan, np.nan
    nb = int(math.ceil(n / block))
    st = np.arange(0, n - block + 1)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    bs = cs[st + block] - cs[st]
    tot = bs[rng.integers(0, len(st), size=(nrep, nb))].sum(axis=1) / (nb * block)
    lo, hi = np.percentile(tot, [2.5, 97.5])
    return float(lo), float(hi), float((tot > 0).mean())


def uncertainty(obs, label, h, pairs):
    print(f"\n[{label}] h={h}  ATTACK 2 -- per-DATE clustering + moving-block bootstrap "
          f"(block={max(63, 3 * h)} trading days, {NREP} reps). diff>0 means FIRST model has "
          f"HIGHER log loss (is WORSE).")
    o = obs[h]
    rows = []
    dts, _ = per_date(o, "ll_clim")
    for a, b in pairs:
        _, va = per_date(o, f"ll_{a}" if a != "clim" else "ll_clim")
        _, vb = per_date(o, f"ll_{b}" if b != "clim" else "ll_clim")
        d = va - vb
        lo, hi, pgt = block_boot(d, max(63, 3 * h))
        # non-overlapping stride-h subsamples, one per phase; report worst/best/mean t
        ts = []
        for ph in range(h):
            sub = d[ph::h]
            if len(sub) > 30:
                ts.append(sub.mean() / (sub.std(ddof=1) / math.sqrt(len(sub))))
        rows.append(dict(pair=f"{a} - {b}", n_dates=len(d), eff_n_indep=round(len(d) / h, 1),
                         mean_diff=float(d.mean()), ci_lo=lo, ci_hi=hi, P_diff_gt0=pgt,
                         t_nonoverlap_min=min(ts) if ts else np.nan,
                         t_nonoverlap_med=float(np.median(ts)) if ts else np.nan,
                         t_nonoverlap_max=max(ts) if ts else np.nan))
    print(pd.DataFrame(rows).to_string(index=False))


# ------------------------------------------------------------------ calibration, incl. subgroups
def ece_block(obs, label, h):
    o = obs[h]
    y_dn, y_up = (o["ai"] == DN).astype(float), (o["ai"] == UP).astype(float)
    mods = ["clim"] + MODELS
    print(f"\n[{label}] h={h}  ATTACK 5 -- OOS ECE (10 equal-count bins) and 1-bin |bias|. "
          f"n={len(y_dn)} obs.  NOTE: CLIMATOLOGY's own ECE is included -- the claim compares "
          f"normal01 vs emp_table but never against the baseline it says emp_table beats.")
    rows = []
    for m in mods:
        pdn = o["p_dn_clim"] if m == "clim" else o[f"pdn_{m}"]
        pup = o["p_up_clim"] if m == "clim" else o[f"pup_{m}"]
        rows.append(dict(model=m,
                         ece_dn=L.ece(pdn.astype(float), y_dn),
                         ece_up=L.ece(pup.astype(float), y_up),
                         bias_dn=float(pdn.mean() - y_dn.mean()),
                         bias_up=float(pup.mean() - y_up.mean()),
                         mean_p_dn=float(pdn.mean()), mean_p_up=float(pup.mean())))
    print(pd.DataFrame(rows).set_index("model").to_string())
    print(f"   observed freq: down_big={y_dn.mean():.4f}  up_big={y_up.mean():.4f}")

    # ---- Simpson check: vol quintiles
    qb = np.digitize(o["rk"].astype(float), [0.2, 0.4, 0.6, 0.8])
    rows = []
    for q in range(5):
        m5 = qb == q
        if m5.sum() < 200:
            continue
        r = dict(volq=q + 1, n=int(m5.sum()), obs_dn=float(y_dn[m5].mean()),
                 obs_up=float(y_up[m5].mean()))
        for m in ("clim", "normal01", "normal_locscale", "emp_table"):
            pdn = (o["p_dn_clim"] if m == "clim" else o[f"pdn_{m}"])[m5].astype(float)
            pup = (o["p_up_clim"] if m == "clim" else o[f"pup_{m}"])[m5].astype(float)
            r[f"{m}_p_dn"] = float(pdn.mean())
            r[f"{m}_p_up"] = float(pup.mean())
            r[f"{m}_ece_dn"] = L.ece(pdn, y_dn[m5])
            r[f"{m}_ece_up"] = L.ece(pup, y_up[m5])
        rows.append(r)
    print(f"\n[{label}] h={h}  calibration WITHIN VOL QUINTILES (Simpson check)")
    print(pd.DataFrame(rows).set_index("volq").to_string())

    # ---- Simpson check: calendar years
    rows = []
    for Y in sorted(set(o["year"].tolist())):
        mY = o["year"] == Y
        r = dict(year=int(Y), n=int(mY.sum()), obs_dn=float(y_dn[mY].mean()),
                 obs_up=float(y_up[mY].mean()))
        for m in ("clim", "normal01", "normal_locscale", "emp_table"):
            pdn = (o["p_dn_clim"] if m == "clim" else o[f"pdn_{m}"])[mY].astype(float)
            pup = (o["p_up_clim"] if m == "clim" else o[f"pup_{m}"])[mY].astype(float)
            r[f"{m}_gap_dn"] = float(pdn.mean() - y_dn[mY].mean())
            r[f"{m}_gap_up"] = float(pup.mean() - y_up[mY].mean())
        rows.append(r)
    t = pd.DataFrame(rows).set_index("year")
    print(f"\n[{label}] h={h}  per-CALENDAR-YEAR signed calibration gap (pred - obs). "
          f"Aggregate ECE hides these.")
    print(t.to_string())
    print(f"   mean |gap| across years:  " +
          "  ".join(f"{m}: dn={t[f'{m}_gap_dn'].abs().mean():.4f} up={t[f'{m}_gap_up'].abs().mean():.4f}"
                    for m in ("clim", "normal01", "normal_locscale", "emp_table")))


# ------------------------------------------------------------------ main
def main():
    close, idx, singles, Z, SH, FWD, sig, rank = build()
    print(f"panel: {len(close)} rows {close.index[0].date()} -> {close.index[-1].date()}; "
          f"{len(idx)} indices, {len(singles)} singles")

    print("\n" + "=" * 150)
    print("ATTACK 2 PRELIM -- how independent are the 5 'indices'? Daily log-return correlations.")
    r = np.log(close[idx]).diff()
    print(r.corr().to_string())
    print(f"   -> INDEX(5) is NOT 5 assets. SPY/^GSPC corr={r['SPY'].corr(r['^GSPC']):.4f}.")

    print("\n" + "=" * 150)
    print("ATTACK 3 PRELIM -- is thr=2% at h=21 a TAIL at all? distribution of k=thr/sigma_hat_h.")
    rows = []
    for h in HORIZONS:
        for label, cols in (("INDEX(5)", idx), ("SPY", ["SPY"]), ("SINGLES", singles)):
            k = (THR / SH[h][cols]).to_numpy(dtype=float).ravel()
            k = k[np.isfinite(k)]
            rows.append(dict(h=h, group=label, n=k.size, p10=np.percentile(k, 10),
                             median=np.median(k), p90=np.percentile(k, 90),
                             norm_P_up=1 - float(norm_cdf_vec(np.median(k)))))
    print(pd.DataFrame(rows).set_index(["h", "group"]).to_string())
    print("   |k| ~ 0.4 means +/-2% sits INSIDE half a sigma -- the CENTRE of the distribution, "
          "not the tail. A 'tail-shape' story cannot be what drives h=21.")

    groups = [("INDEX(5)", idx, True), ("SPY", ["SPY"], True)] + \
             [(s, [s], True) for s in idx if s != "SPY"] + \
             [("SINGLES", singles, False)]

    allres, allobs = {}, {}
    for label, cols, keep in groups:
        print("\n" + "=" * 150)
        print(f"### GROUP {label}  ({len(cols)} symbols)")
        df, obs = run_oos(Z, SH, FWD, rank, label, cols, keep_obs=keep)
        allres[label] = df
        if keep:
            allobs[label] = obs
        skill_table(df, label)
        print(f"\n[{label}] ATTACK 1 -- LOOKAHEAD: train labels at the year boundary reach into "
              f"the test year. emp_table vs emp_table_purged (train rows with forward window "
              f"crossing into Y removed):")
        pt = pd.DataFrame({
            "emp_table_skill": 1 - df.groupby("h").emp_table.mean() / df.groupby("h").clim_ll.mean(),
            "emp_purged_skill": 1 - df.groupby("h").emp_table_purged.mean() / df.groupby("h").clim_ll.mean(),
            "clim_purged_skill_of_normal01": 1 - df.groupby("h").normal01.mean() / df.groupby("h").clim_purged_ll.mean(),
            "n_train": df.groupby("h").n_train.mean(),
            "n_train_purged": df.groupby("h").n_train_purged.mean()})
        print(pt.to_string())

        print(f"\n[{label}] ATTACK 4 -- FRAGILITY: skill after dropping 2008 and 2020 as TEST years")
        d2 = df[~df.year.isin([2008, 2020])]
        if len(d2):
            print(pd.DataFrame({m: 1 - d2.groupby("h")[m].mean() / d2.groupby("h").clim_ll.mean()
                                for m in MODELS}).assign(n_years=d2.groupby("h").size()).to_string())
        print(f"\n[{label}] ATTACK 4 -- per-test-year skill at h=21 "
              f"(clim_ll - model)/clim_ll; and win rate")
        d21 = df[df.h == 21]
        if len(d21):
            pv = d21.set_index("year")[["n_test", "clim_ll", "obs_f_dn", "obs_f_up", "med_z",
                                        "iqs_z", "normal01", "normal_locscale", "emp_table"]].copy()
            for m in ("normal01", "normal_locscale", "emp_table"):
                pv[f"sk_{m}"] = 1 - pv[m] / pv.clim_ll
            print(pv.to_string())
            print("   win rate vs climatology at h=21: " +
                  "  ".join(f"{m}={float((d21[m] < d21.clim_ll).mean()):.3f}"
                            for m in MODELS))

    print("\n" + "=" * 150)
    print("ATTACK 2 -- does the claimed precision survive date-clustering and non-overlap?")
    pairs = [("normal01", "clim"), ("emp_table", "clim"), ("normal_locscale", "clim"),
             ("normal_loc", "clim"), ("emp_sym", "clim"),
             ("normal01", "emp_table"), ("normal_locscale", "emp_table"),
             ("emp_sym", "emp_table"), ("normal_loc", "emp_table"),
             ("normal_loc", "normal01")]
    for label in ("INDEX(5)", "SPY"):
        for h in (10, 21):
            if label in allobs and h in allobs[label]:
                uncertainty(allobs[label], label, h, pairs)

    print("\n" + "=" * 150)
    print("ATTACK 3 -- DECOMPOSITION: how much of emp_table's gain over normal01 is just "
          "LOCATION (drift) and SCALE, with the normal shape kept?")
    for label in ("INDEX(5)", "SPY", "SINGLES"):
        df = allres[label]
        g = df.groupby("h")
        base, cl = g.normal01.mean(), g.clim_ll.mean()
        emp = g.emp_table.mean()
        t = pd.DataFrame({
            "clim_ll": cl, "normal01_ll": base, "n_loc_ll": g.normal_loc.mean(),
            "n_locscale_ll": g.normal_locscale.mean(), "emp_sym_ll": g.emp_sym.mean(),
            "emp_table_ll": emp,
            "pct_of_gain_from_LOC": (base - g.normal_loc.mean()) / (base - emp),
            "pct_of_gain_from_LOC+SCALE": (base - g.normal_locscale.mean()) / (base - emp),
            "pct_from_SYM_EMP(loc+scale+fat tails,no asym)":
                (base - g.emp_sym.mean()) / (base - emp),
        })
        print(f"\n[{label}] gain decomposition (fractions of the normal01 -> emp_table gap closed)")
        print(t.to_string())

    print("\n" + "=" * 150)
    print("ATTACK 5 -- calibration, aggregate vs subgroups")
    for label in ("INDEX(5)", "SPY"):
        if label in allobs:
            for h in (21,):
                ece_block(allobs[label], label, h)

    print("\n" + "=" * 150)
    print("ATTACK 4 -- pooled INDEX(5) vs its members, h=21 skill")
    rows = []
    for label in allres:
        d = allres[label]
        d = d[d.h == 21]
        if not len(d):
            continue
        r = dict(group=label, n_years=len(d), n_obs=int(d.n_test.sum()),
                 clim_ll=d.clim_ll.mean())
        for m in MODELS:
            r[m] = 1 - d[m].mean() / d.clim_ll.mean()
        rows.append(r)
    print(pd.DataFrame(rows).set_index("group").to_string())

    pd.to_pickle({k: v for k, v in allres.items()}, "/tmp/verify_tailshape.pkl")


if __name__ == "__main__":
    main()
