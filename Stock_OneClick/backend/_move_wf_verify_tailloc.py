"""
_move_wf_verify_tailloc.py — INDEPENDENT adversarial verification of the "tail-shape" claim.

CLAIM UNDER TEST
  "A normal shape on top of sigma_hat = ewma(0.94)*sqrt(h) is not merely imprecise for indices at
   long horizons -- it is worse than the unconditional climatology baseline out of sample, while
   the empirical asymmetric z-table beats it."
  Offered: h=21 walk-forward OOS 4-bucket log loss at thr=2%, params fit on years < Y:
      INDEX(5) normal01 skill = -0.0209 ; SPY = -0.0430
      INDEX(5) emp_table skill = +0.0150 ; SPY = +0.0090   (19 / 16 test years; 24,655 / 3,923 obs)
      INDEX h=21 observed down_big 0.2233 / up_big 0.4559 vs normal's forced 0.3195 / 0.3195
      OOS ECE h=21: 0.0968 -> 0.0273 (down_big), 0.1363 -> 0.0406 (up_big)

Everything below is re-implemented from _move_lib primitives (own erfc, own empirical CDF, own
walk-forward loop) so that a bug in the original cannot be inherited.  Only the Student-t CDF /
QQ-fit are imported (pure math, cached).

ATTACKS
  0  REPLICATE the offered numbers.
  1  LOOKAHEAD: expanding already; add a PURGE of train rows whose h-day forward window spills into
     the test year, applied to BOTH the model and the climatology baseline.
  2  FAKE n: stride-h non-overlapping subsamples at every phase; collapse to one obs per DATE;
     moving-block bootstrap over dates (block=63 >> h); per-test-year sign test.
  3  SIMPLER EXPLANATION: normal01 bundles SHAPE with LOCATION(=0) and SCALE(=1).  Controls:
     N(mean,1), N(med,1), N(0,sd), N(mean,sd), N(med,iqs), symmetrised empirical table,
     zero-centred empirical table.  Plus: how many sigma out is thr=2% at h=21 -- is this a TAIL
     statement at all?  Plus a thr sweep.  Plus: FLAT-SIGMA controls that separate "shape/location
     table" from "conditional vol information".
  4  FRAGILITY: drop 2008 & 2020 as test years and from train; per-test-year skill; INDEX(5) split
     into its 5 members (they are ~0.99 correlated, so the pooled n is fake).
  5  CALIBRATION SUBGROUPS: OOS ECE for down_big / up_big overall, by vol quintile (train-fitted
     edges) and by calendar year -- including climatology's own ECE, the missing reference.
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

import _move_data as D
import _move_lib as L
from _move_wf_zshape2 import qq_fit_t, t_unit_cdf_vec

pd.set_option("display.width", 320)
pd.set_option("display.max_columns", 140)
pd.set_option("display.max_rows", 400)
pd.set_option("display.float_format", lambda v: f"{v:10.4f}")

THR = 0.02
LAM, WARMUP, MIN_COV = 0.94, 250, 500
HORIZONS = (1, 5, 10, 21)
IDX5 = ["SPY", "QQQ", "IWM", "DIA", "^GSPC"]
MIN_TRAIN_ROWS, MIN_TEST_ROWS, MIN_TRAIN_YEARS = 2000, 100, 5
DN, UP = 0, 3
RNG = np.random.default_rng(20260911)
NREP = 2000
BLOCK = 63

MODELS = ["normal01", "normal_mean1", "normal_med1", "normal_0sd", "normal_meansd",
          "normal_mediqs", "t_qqfit", "emp_sym", "emp_zeroloc", "emp_table",
          "emp_table_purged", "flat_emp_table", "flat_clim_check"]


# ----------------------------------------------------------------- own normal CDF (NR erfc_cheb)
_COF = np.array([
    -1.3026537197817094, 6.4196979235649026e-1, 1.9476473204185836e-2,
    -9.561514786808631e-3, -9.46595344482036e-4, 3.66839497852761e-4,
    4.2523324806907e-5, -2.0278578112534e-5, -1.624290004647e-6,
    1.303655835580e-6, 1.5626441722e-8, -8.5238095915e-8,
    6.529054439e-9, 5.059343495e-9, -9.91364156e-10, -2.27365122e-10,
    9.6467911e-11, 2.394038e-12, -6.886027e-12, 8.94487e-13,
    3.13092e-13, -1.12708e-13, 3.81e-16, 7.106e-15])


def _erfc(x):
    x = np.asarray(x, float)
    z = np.abs(x)
    t = 2.0 / (2.0 + z)
    ty = 4.0 * t - 2.0
    d = np.zeros_like(z)
    dd = np.zeros_like(z)
    for j in range(len(_COF) - 1, 0, -1):
        tmp = d
        d = ty * d - dd + _COF[j]
        dd = tmp
    ans = t * np.exp(-z * z + 0.5 * (_COF[0] + ty * d) - dd)
    return np.where(x >= 0.0, ans, 2.0 - ans)


def ncdf(x):
    return 0.5 * _erfc(-np.asarray(x, float) / math.sqrt(2.0))


def _selftest():
    g = np.linspace(-6, 6, 25)
    mine = ncdf(g)
    ref = np.array([0.5 * (1.0 + math.erf(v / math.sqrt(2.0))) for v in g])
    err = np.max(np.abs(mine - ref))
    assert err < 1e-12, err
    # alignment self-test required by the repo conventions
    p = D.load()
    c = p["Close"]["SPY"].dropna()
    lr = np.log(c).diff()
    rv = L.realized_vol_forward(c, 5)
    t = 4000
    assert abs(rv.iloc[t] - lr.iloc[t + 1:t + 6].std(ddof=1)) < 1e-15
    fr = L.forward_simple_return(c, 21)
    assert abs(fr.iloc[t] - (c.iloc[t + 21] / c.iloc[t] - 1)) < 1e-12
    # my fast aidx must agree with L.bucketize exactly
    fv = fr.dropna().to_numpy()[:20000]
    ref = L.bucketize(pd.Series(fv), THR).to_numpy()
    order = {v: i for i, v in enumerate(L.BUCKETS)}
    assert (aidx(fv, THR) == np.array([order[x] for x in ref])).all()
    print(f"self-test OK: max |ncdf - erf| = {err:.2e}; realized_vol_forward / "
          f"forward_simple_return alignment and aidx == L.bucketize verified")


# ----------------------------------------------------------------- empirical CDF (own)
class Emp:
    """Plain empirical CDF with a fitted-t extrapolation outside the sample support."""

    def __init__(self, a, nu=5.0):
        a = np.sort(np.asarray(a, float))
        a = a[np.isfinite(a)]
        self.a, self.n = a, a.size
        self.med = float(np.median(a))
        self.sc = float((np.percentile(a, 75) - np.percentile(a, 25)) / 1.34898)
        self.nu = float(nu)
        self.frac = 0.5 / (self.n + 1.0)

    def cdf(self, k):
        k = np.asarray(k, float)
        r = np.searchsorted(self.a, k, side="left").astype(float)
        out = np.clip((r + 0.5) / (self.n + 1.0), self.frac, 1.0 - self.frac)
        lo, hi = k < self.a[0], k > self.a[-1]
        if lo.any():
            out[lo] = np.minimum(t_unit_cdf_vec((k[lo] - self.med) / self.sc, self.nu), self.frac)
        if hi.any():
            out[hi] = np.maximum(t_unit_cdf_vec((k[hi] - self.med) / self.sc, self.nu),
                                 1.0 - self.frac)
        return out


def probs(cdf_fn, sh, thr):
    """4-bucket probs.  r<=-thr <=> z<=-thr/sh ; 0 <=> z<=0 ; r>=thr <=> z>=thr/sh."""
    sh = np.asarray(sh, float)
    F_dn, F_0, F_up = cdf_fn(-thr / sh), cdf_fn(np.zeros_like(sh)), cdf_fn(thr / sh)
    P = np.column_stack([F_dn, F_0 - F_dn, F_up - F_0, 1.0 - F_up])
    P = np.clip(P, 1e-6, None)
    return P / P.sum(axis=1, keepdims=True)


def aidx(fwd, thr):
    """Bucket index 0..3.  Cross-checked against L.bucketize on every call path (see _selftest)."""
    r = np.asarray(fwd, float)
    return np.where(r <= -thr, 0, np.where(r <= 0.0, 1, np.where(r < thr, 2, 3))).astype(np.int8)


def iqs_of(a):
    return float((np.percentile(a, 75) - np.percentile(a, 25)) / 1.34898)


# ----------------------------------------------------------------- panel
def build():
    p = D.load()
    close = p["Close"]
    cov = close.notna().sum()
    close = close[cov[cov >= MIN_COV].index]
    sig = L.vol_ewma(close, LAM)
    ret = np.log(close).diff()
    ok = (ret.notna().cumsum() >= WARMUP) & sig.notna() & (sig > 1e-6)
    idx = [s for s in IDX5 if s in close.columns]
    singles = [c for c in close.columns if c not in idx and c != "^VIX"]
    Z, SH, FWD = {}, {}, {}
    for h in HORIZONS:
        sh = sig * math.sqrt(h)
        fwd = L.forward_simple_return(close, h)
        m = ok & fwd.notna() & np.isfinite(fwd)
        SH[h] = sh.where(m)
        FWD[h] = fwd.where(m)
        Z[h] = (fwd / sh).where(m)
    return close, idx, singles, sig, Z, SH, FWD


def stack(Z, SH, FWD, h, cols):
    """Long-form arrays for one horizon / column set, with panel row position and symbol id."""
    z = Z[h][cols]
    m = z.notna().to_numpy()
    pos = np.repeat(np.arange(len(z))[:, None], len(cols), axis=1)
    symid = np.repeat(np.arange(len(cols))[None, :], len(z), axis=0)
    yr = np.repeat(z.index.year.to_numpy()[:, None], len(cols), axis=1)
    out = {"z": z.to_numpy(float)[m], "sh": SH[h][cols].to_numpy(float)[m],
           "fwd": FWD[h][cols].to_numpy(float)[m], "pos": pos[m].astype(np.int32),
           "sym": symid[m].astype(np.int16), "year": yr[m].astype(np.int16)}
    g = np.isfinite(out["z"]) & np.isfinite(out["sh"]) & np.isfinite(out["fwd"]) & (out["sh"] > 0)
    return {k: v[g] for k, v in out.items()}


# ----------------------------------------------------------------- walk-forward engine
def wf(S, h, thr=THR, keep_obs=False, drop_years=()):
    """Expanding-window walk-forward.  Returns per-test-year rows and (optionally) per-obs preds."""
    z, sh, fwd, yr, pos = S["z"], S["sh"], S["fwd"], S["year"], S["pos"]
    ai = aidx(fwd, thr)
    years = sorted(set(int(v) for v in yr))
    y0 = years[0]
    rows, obs = [], []
    for Y in years:
        if Y - y0 < MIN_TRAIN_YEARS or Y in drop_years:
            continue
        tr = (yr < Y)
        te = (yr == Y)
        if len(drop_years):
            tr &= ~np.isin(yr, list(drop_years))
        if tr.sum() < MIN_TRAIN_ROWS or te.sum() < MIN_TEST_ROWS:
            continue
        ztr, aitr = z[tr], ai[tr]
        shte, aite, n = sh[te], ai[te], int(te.sum())

        clim = np.clip(L.climatology(aitr, k=4), 1e-6, None)
        clim /= clim.sum()
        base = L.log_loss(np.tile(clim, (n, 1)), aite)

        # purged train: forward window must end strictly before the first test row
        first_te_pos = int(pos[te].min())
        pur = tr & ((pos + h) < first_te_pos)
        climp = np.clip(L.climatology(ai[pur], k=4), 1e-6, None)
        climp /= climp.sum()
        basep = L.log_loss(np.tile(climp, (n, 1)), aite)

        mu, sd = float(ztr.mean()), float(ztr.std(ddof=1))
        med, iq = float(np.median(ztr)), iqs_of(ztr)
        tf = qq_fit_t(ztr)
        emp = Emp(ztr, nu=tf["nu"])
        empp = Emp(z[pur], nu=tf["nu"])
        emp0 = Emp(ztr - med, nu=tf["nu"])                     # empirical shape, zero location
        empsym = Emp(np.concatenate([ztr - med, -(ztr - med)]), nu=tf["nu"])   # fat tails, no skew

        cdfs = {
            "normal01": ncdf,
            "normal_mean1": lambda k: ncdf(k - mu),
            "normal_med1": lambda k: ncdf(k - med),
            "normal_0sd": lambda k: ncdf(k / sd),
            "normal_meansd": lambda k: ncdf((k - mu) / sd),
            "normal_mediqs": lambda k: ncdf((k - med) / iq),
            "t_qqfit": lambda k: t_unit_cdf_vec((k - tf["loc"]) / tf["scale"], tf["nu"]),
            "emp_sym": lambda k: empsym.cdf(k - med),
            "emp_zeroloc": emp0.cdf,
            "emp_table": emp.cdf,
            "emp_table_purged": empp.cdf,
        }
        row = dict(h=h, year=Y, n_test=n, n_train=int(tr.sum()), n_purged=int(pur.sum()),
                   clim_ll=base, clim_purged_ll=basep, thr=thr,
                   obs_dn=float((aite == DN).mean()), obs_up=float((aite == UP).mean()),
                   clim_dn=float(clim[DN]), clim_up=float(clim[UP]),
                   mean_k=float(np.mean(thr / shte)), med_k=float(np.median(thr / shte)),
                   tr_med_z=med, tr_mu_z=mu, tr_sd_z=sd, tr_iqs_z=iq, tr_nu=tf["nu"])
        P = {}
        for name, fn in cdfs.items():
            P[name] = probs(fn, shte, thr)
            row[name] = L.log_loss(P[name], aite)
            row[f"{name}_dn"] = float(P[name][:, DN].mean())
            row[f"{name}_up"] = float(P[name][:, UP].mean())

        # FLAT-SIGMA controls: same table, but sigma_hat replaced by its TRAIN median.
        # Isolates "does the table itself beat climatology" from "does conditional vol help".
        sh_flat = np.full(n, float(np.median(sh[tr])))
        Pf = probs(emp.cdf, sh_flat, thr)
        row["flat_emp_table"] = L.log_loss(Pf, aite)
        Pfn = probs(ncdf, sh_flat, thr)
        row["flat_clim_check"] = L.log_loss(Pfn, aite)
        rows.append(row)

        if keep_obs:
            Pc = np.tile(clim, (n, 1))
            keep_models = ("normal01", "normal_meansd", "emp_table")
            rowsel = np.arange(n)
            d = {"year": Y, "pos": pos[te], "sym": S["sym"][te], "sh": shte, "ai": aite,
                 "clim_dn": clim[DN], "clim_up": clim[UP],
                 "ll_clim": -np.log(np.clip(Pc[rowsel, aite], 1e-12, None)),
                 # train-fitted vol-quintile label (causal: edges from train sigma_hat only)
                 "volq": np.digitize(shte, np.percentile(sh[tr], [20, 40, 60, 80]))}
            for m in keep_models:
                d[f"{m}_dn"] = P[m][:, DN]
                d[f"{m}_up"] = P[m][:, UP]
                d[f"ll_{m}"] = -np.log(np.clip(P[m][rowsel, aite], 1e-12, None))
            obs.append(pd.DataFrame(d))
    df = pd.DataFrame(rows)
    return (df, pd.concat(obs, ignore_index=True) if obs else pd.DataFrame()) if keep_obs else df


def skills(df, models=MODELS, base="clim_ll"):
    """Equal weight per test year (as the claim does) AND obs-pooled."""
    out = {}
    for m in models:
        if m not in df.columns:
            continue
        out[m] = {
            "yr_skill": 1.0 - df[m].mean() / df[base].mean(),
            "pool_skill": 1.0 - np.average(df[m], weights=df.n_test)
                          / np.average(df[base], weights=df.n_test),
            "yrs_win": float((df[m] < df[base]).mean()),
        }
    t = pd.DataFrame(out).T
    t["n_years"] = len(df)
    t["n_obs"] = int(df.n_test.sum())
    return t


def sign_p(k, n):
    """Two-sided exact binomial p under p=0.5."""
    from math import comb
    k = min(k, n - k)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


# ----------------------------------------------------------------- main
def main():
    _selftest()
    close, idx, singles, sig, Z, SH, FWD = build()
    print(f"\nuniverse after MIN_COV={MIN_COV}: {close.shape[1]} symbols, {len(close)} rows, "
          f"{close.index[0].date()} -> {close.index[-1].date()}; indices={idx}; "
          f"singles={len(singles)}")

    groups = {"SPY": ["SPY"], "INDEX5": idx, "SINGLES": singles}
    members = {s: [s] for s in idx}

    ST = {}
    for g, cols in {**groups, **members}.items():
        for h in HORIZONS:
            ST[(g, h)] = stack(Z, SH, FWD, h, cols)

    # =============================================================== 0 REPLICATION
    print("\n" + "=" * 118)
    print("ATTACK 0 -- REPLICATION of the offered walk-forward OOS numbers (thr=2%)")
    WF, OBS = {}, {}
    for g in ("INDEX5", "SPY", "SINGLES"):
        for h in HORIZONS:
            keep = (h == 21)
            r = wf(ST[(g, h)], h, THR, keep_obs=keep)
            if keep:
                WF[(g, h)], OBS[g] = r
            else:
                WF[(g, h)] = r
    for g in ("INDEX5", "SPY"):
        d = pd.concat([WF[(g, h)] for h in HORIZONS])
        print(f"\n[{g}] skill vs TRAIN-climatology by horizon (equal weight per test year)")
        tab = pd.DataFrame({m: 1 - d.groupby("h")[m].mean() / d.groupby("h").clim_ll.mean()
                            for m in MODELS if m in d.columns})
        tab["n_years"] = d.groupby("h").size()
        tab["n_obs"] = d.groupby("h").n_test.sum()
        print(tab.to_string())
    print("\nClaim: INDEX5 normal01 -0.0209 / emp_table +0.0150 (19y, 24655 obs); "
          "SPY -0.0430 / +0.0090 (16y, 3923 obs).")

    print("\n--- the offered 'structural' frequencies, h=21, OOS test rows only ---")
    for g in ("INDEX5", "SPY"):
        d = WF[(g, 21)]
        w = d.n_test
        print(f"[{g}] observed  down_big={np.average(d.obs_dn, weights=w):.4f}  "
              f"up_big={np.average(d.obs_up, weights=w):.4f}   "
              f"normal01 pred down={np.average(d.normal01_dn, weights=w):.4f} "
              f"up={np.average(d.normal01_up, weights=w):.4f}   "
              f"train-clim down={np.average(d.clim_dn, weights=w):.4f} "
              f"up={np.average(d.clim_up, weights=w):.4f}")
        print(f"       total 'big' mass: observed "
              f"{np.average(d.obs_dn + d.obs_up, weights=w):.4f} vs normal01 "
              f"{np.average(d.normal01_dn + d.normal01_up, weights=w):.4f}"
              f"  -> the SIZE of the tails is nearly right; the SIDE is not.")

    # =============================================================== 1 LOOKAHEAD
    print("\n" + "=" * 118)
    print("ATTACK 1 -- LOOKAHEAD.  normal01 has NO fitted parameter, so the only fittable objects")
    print("are (a) the climatology baseline and (b) emp_table.  Purge train rows whose h-day")
    print("forward window reaches into the test year, for BOTH.")
    for g in ("INDEX5", "SPY", "SINGLES"):
        d = pd.concat([WF[(g, h)] for h in HORIZONS])
        t = pd.DataFrame({
            "emp_table_skill": 1 - d.groupby("h").emp_table.mean() / d.groupby("h").clim_ll.mean(),
            "emp_purged_skill": 1 - d.groupby("h").emp_table_purged.mean()
                                / d.groupby("h").clim_ll.mean(),
            "emp_purged_vs_purged_clim": 1 - d.groupby("h").emp_table_purged.mean()
                                         / d.groupby("h").clim_purged_ll.mean(),
            "normal01_skill": 1 - d.groupby("h").normal01.mean() / d.groupby("h").clim_ll.mean(),
            "normal01_vs_purged_clim": 1 - d.groupby("h").normal01.mean()
                                        / d.groupby("h").clim_purged_ll.mean(),
            "train_rows_dropped_pct": 100 * (1 - d.groupby("h").n_purged.sum()
                                             / d.groupby("h").n_train.sum()),
        })
        print(f"\n[{g}] purge test")
        print(t.to_string())

    # =============================================================== 2 FAKE SAMPLE SIZE
    print("\n" + "=" * 118)
    print("ATTACK 2 -- FAKE SAMPLE SIZE.")
    r5 = np.log(close[idx]).diff()
    cm = r5.corr()
    print("\nINDEX5 daily log-return correlation matrix (pooled n=24,655 is not 5 assets):")
    print(cm.to_string())
    print(f"mean off-diagonal corr = {(cm.values.sum() - 5) / 20:.4f}")

    print("\n(2a) per-test-year sign test at h=21, gain = clim_ll - model_ll")
    for g in ("INDEX5", "SPY", "SINGLES"):
        d = WF[(g, 21)]
        n = len(d)
        for m in ("normal01", "normal_meansd", "emp_table"):
            gain = d.clim_ll - d[m]
            k = int((gain > 0).sum())
            t = gain.mean() / (gain.std(ddof=1) / math.sqrt(n))
            print(f"  [{g}] {m:16s} mean_gain={gain.mean():+.5f} sd={gain.std(ddof=1):.5f} "
                  f"t(year-clustered,n={n})={t:+.2f}  years_won={k}/{n} "
                  f"sign_p={sign_p(k, n):.3f}")

    print("\n(2b) stride-h NON-OVERLAPPING test subsamples, every phase offset, h=21")
    for g in ("INDEX5", "SPY"):
        S = ST[(g, 21)]
        ai_all = aidx(S["fwd"], THR)
        res = {m: [] for m in ("normal01", "normal_meansd", "emp_table")}
        nobs = []
        for phase in range(21):
            keep = (S["pos"] % 21) == phase
            Ssub = {k: v[keep] for k, v in S.items()}
            # NOTE: models are still FIT on the full expanding train set (that is legitimate);
            # only the SCORED rows are thinned so they do not overlap.
            d = wf_scored_subset(S, Ssub, 21, THR)
            for m in res:
                res[m].append(1 - d[m].mean() / d.clim_ll.mean())
            nobs.append(int(d.n_test.sum()))
        t = pd.DataFrame(res)
        print(f"  [{g}] n_obs per phase ~{int(np.mean(nobs))} (vs {int(WF[(g,21)].n_test.sum())} "
              f"overlapping)")
        print(f"    {'model':16s} {'mean':>9s} {'min':>9s} {'max':>9s} {'frac>0':>8s}")
        for m in res:
            v = np.array(res[m])
            print(f"    {m:16s} {v.mean():+9.4f} {v.min():+9.4f} {v.max():+9.4f} "
                  f"{(v > 0).mean():8.2f}")

    print("\n(2c) ONE OBS PER DATE (kills the 5x cross-sectional double count) + moving-block")
    print("     bootstrap over dates, block=63 trading days >> h=21, 2000 reps.")
    for g in ("INDEX5", "SPY", "SINGLES"):
        o = OBS[g]
        for m in ("normal01", "normal_meansd", "emp_table"):
            per_date = pd.DataFrame({"pos": o.pos, "m": o[f"ll_{m}"], "c": o.ll_clim}) \
                .groupby("pos").mean()
            lo, hi, sk = block_boot(per_date.m.to_numpy(), per_date.c.to_numpy())
            print(f"  [{g}] {m:16s} date-collapsed skill={sk:+.4f} "
                  f"block-bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]  n_dates={len(per_date)}")

    # =============================================================== 3 SIMPLER EXPLANATION
    print("\n" + "=" * 118)
    print("ATTACK 3 -- SIMPLER EXPLANATION: is this about SHAPE, or about LOCATION and SCALE?")
    print("\n(3a) how far out is thr=2%?  k = thr / sigma_hat_h over the OOS rows")
    rows = []
    for g in ("INDEX5", "SPY", "SINGLES"):
        for h in HORIZONS:
            d = WF[(g, h)]
            rows.append(dict(group=g, h=h, mean_k=np.average(d.mean_k, weights=d.n_test),
                             median_k=np.average(d.med_k, weights=d.n_test)))
    kt = pd.DataFrame(rows).pivot(index="h", columns="group", values="median_k")
    print(kt.to_string())
    print("  -> at h=21 the +/-2% bucket edges sit well INSIDE one sigma. These are BODY")
    print("     quartile-ish cuts, not tails, so 'tail-shape' is the wrong name for the defect.")

    print("\n(3b) fraction of the normal01 -> emp_table log-loss gap closed by each control, h=21")
    for g in ("INDEX5", "SPY", "SINGLES"):
        d = WF[(g, 21)]
        n01, et, cl = d.normal01.mean(), d.emp_table.mean(), d.clim_ll.mean()
        gap = n01 - et
        rows = []
        for m in ("normal_mean1", "normal_med1", "normal_0sd", "normal_meansd", "normal_mediqs",
                  "t_qqfit", "emp_sym", "emp_zeroloc", "emp_table"):
            rows.append(dict(model=m, ll=d[m].mean(), frac_gap_closed=(n01 - d[m].mean()) / gap,
                             skill_vs_clim=1 - d[m].mean() / cl))
        t = pd.DataFrame(rows).set_index("model")
        print(f"\n  [{g}] clim_ll={cl:.4f} normal01_ll={n01:.4f} emp_table_ll={et:.4f} "
              f"gap={gap:.4f}")
        print(t.to_string())

    print("\n(3c) FLAT-SIGMA control: same tables but sigma_hat := train-median sigma_hat.")
    print("     If flat_emp_table ~ 0 skill, emp_table's win is CONDITIONAL VOL, not the table.")
    for g in ("INDEX5", "SPY", "SINGLES"):
        d = pd.concat([WF[(g, h)] for h in HORIZONS])
        t = pd.DataFrame({
            "emp_table": 1 - d.groupby("h").emp_table.mean() / d.groupby("h").clim_ll.mean(),
            "flat_emp_table": 1 - d.groupby("h").flat_emp_table.mean()
                              / d.groupby("h").clim_ll.mean(),
            "normal01": 1 - d.groupby("h").normal01.mean() / d.groupby("h").clim_ll.mean(),
            "flat_normal01": 1 - d.groupby("h").flat_clim_check.mean()
                             / d.groupby("h").clim_ll.mean(),
        })
        print(f"\n  [{g}]")
        print(t.to_string())

    print("\n(3d) THRESHOLD SWEEP at h=21: is normal01 < climatology a general fact, or specific")
    print("     to thr=2% (where the cut lands mid-body and the drift asymmetry dominates)?")
    for g in ("INDEX5", "SPY"):
        rows = []
        for thr in (0.01, 0.02, 0.03, 0.05, 0.08):
            d = wf(ST[(g, 21)], 21, thr)
            rows.append(dict(thr=thr, median_k=np.average(d.med_k, weights=d.n_test),
                             clim_ll=d.clim_ll.mean(),
                             normal01=1 - d.normal01.mean() / d.clim_ll.mean(),
                             normal_meansd=1 - d.normal_meansd.mean() / d.clim_ll.mean(),
                             emp_table=1 - d.emp_table.mean() / d.clim_ll.mean()))
        print(f"\n  [{g}] skill vs train-climatology")
        print(pd.DataFrame(rows).set_index("thr").to_string())

    # =============================================================== 4 FRAGILITY
    print("\n" + "=" * 118)
    print("ATTACK 4 -- FRAGILITY")
    print("\n(4a) per-test-year skill at h=21 (positive = beats that year's train-climatology)")
    for g in ("INDEX5", "SPY"):
        d = WF[(g, 21)].set_index("year")
        t = pd.DataFrame({m: 1 - d[m] / d.clim_ll
                          for m in ("normal01", "normal_meansd", "emp_table")})
        t["clim_ll"] = d.clim_ll
        t["n"] = d.n_test
        print(f"\n  [{g}]")
        print(t.to_string())

    print("\n(4b) drop 2008 & 2020 as TEST years and from TRAIN, h=21")
    for g in ("INDEX5", "SPY", "SINGLES"):
        full = WF[(g, 21)]
        d2 = wf(ST[(g, 21)], 21, THR, drop_years=(2008, 2020))
        rowf, rowd = {}, {}
        for m in ("normal01", "normal_meansd", "emp_table"):
            rowf[m] = 1 - full[m].mean() / full.clim_ll.mean()
            rowd[m] = 1 - d2[m].mean() / d2.clim_ll.mean()
        print(f"  [{g}] full ({len(full)}y): " +
              "  ".join(f"{k}={v:+.4f}" for k, v in rowf.items()))
        print(f"  [{g}] ex-08/20 ({len(d2)}y): " +
              "  ".join(f"{k}={v:+.4f}" for k, v in rowd.items()))

    print("\n(4c) INDEX5 split into members, h=21 (the pooled number is 5 copies of one asset)")
    rows = []
    for s in idx:
        d = wf(ST[(s, 21)], 21, THR)
        rows.append(dict(sym=s, n_years=len(d), n_obs=int(d.n_test.sum()),
                         clim_ll=d.clim_ll.mean(),
                         normal01=1 - d.normal01.mean() / d.clim_ll.mean(),
                         normal_meansd=1 - d.normal_meansd.mean() / d.clim_ll.mean(),
                         emp_table=1 - d.emp_table.mean() / d.clim_ll.mean()))
    print(pd.DataFrame(rows).set_index("sym").to_string())

    # =============================================================== 5 CALIBRATION SUBGROUPS
    print("\n" + "=" * 118)
    print("ATTACK 5 -- CALIBRATION.  ECE for down_big / up_big, OOS h=21.  Includes CLIMATOLOGY's")
    print("own ECE, which the claim omits -- a constant forecast can have a low ECE too.")
    for g in ("INDEX5", "SPY"):
        o = OBS[g]
        hit_dn = (o.ai == DN).to_numpy(float)
        hit_up = (o.ai == UP).to_numpy(float)
        print(f"\n  [{g}] n_obs={len(o)}")
        rows = []
        for m in ("clim", "normal01", "normal_meansd", "emp_table"):
            pd_dn = (o.clim_dn if m == "clim" else o[f"{m}_dn"]).to_numpy(float)
            pd_up = (o.clim_up if m == "clim" else o[f"{m}_up"]).to_numpy(float)
            rows.append(dict(model=m,
                             ece_dn=L.ece(pd_dn, hit_dn), ece_up=L.ece(pd_up, hit_up),
                             mean_p_dn=pd_dn.mean(), obs_dn=hit_dn.mean(),
                             mean_p_up=pd_up.mean(), obs_up=hit_up.mean()))
        print(pd.DataFrame(rows).set_index("model").to_string())
        print(f"  claim: ECE down_big 0.0968 -> 0.0273 ; up_big 0.1363 -> 0.0406")

        for label, key in (("VOL QUINTILE (train-fitted edges)", "volq"), ("CALENDAR YEAR", "year")):
            print(f"\n  [{g}] ECE within {label}")
            out = {}
            for m in ("clim", "normal01", "normal_meansd", "emp_table"):
                cdn = "clim_dn" if m == "clim" else f"{m}_dn"
                cup = "clim_up" if m == "clim" else f"{m}_up"
                rows = []
                for k, sub in o.groupby(key):
                    hd = (sub.ai == DN).to_numpy(float)
                    hu = (sub.ai == UP).to_numpy(float)
                    rows.append(dict(**{key: k}, n=len(sub),
                                     gap_dn=sub[cdn].mean() - hd.mean(),
                                     gap_up=sub[cup].mean() - hu.mean()))
                out[m] = pd.DataFrame(rows).set_index(key)
            base = out["clim"][["n"]].copy()
            for m in out:
                base[f"{m}_gapdn"] = out[m].gap_dn
            for m in out:
                base[f"{m}_gapup"] = out[m].gap_up
            print(base.to_string())
            print("  (gap = mean predicted - observed frequency; a model can be flat in aggregate "
                  "and wrong in every cell)")

    # =============================================================== singles
    print("\n" + "=" * 118)
    print("SINGLES (survivorship-biased; index rows are the honest benchmark) -- skill by horizon")
    d = pd.concat([WF[("SINGLES", h)] for h in HORIZONS])
    t = pd.DataFrame({m: 1 - d.groupby("h")[m].mean() / d.groupby("h").clim_ll.mean()
                      for m in MODELS if m in d.columns})
    t["n_years"] = d.groupby("h").size()
    t["n_obs"] = d.groupby("h").n_test.sum()
    print(t.to_string())


def block_boot(m, c, nrep=NREP, block=BLOCK, rng=RNG):
    """Moving-block bootstrap over the DATE axis of two aligned per-date mean-loss series."""
    n = len(m)
    nb = int(math.ceil(n / block))
    starts_max = n - block
    sk = 1.0 - m.mean() / c.mean()
    out = np.empty(nrep)
    for i in range(nrep):
        st = rng.integers(0, starts_max + 1, nb)
        ix = (st[:, None] + np.arange(block)[None, :]).ravel()[:n]
        out[i] = 1.0 - m[ix].mean() / c[ix].mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float(sk)


def wf_scored_subset(S_full, S_sub, h, thr):
    """Fit on the full expanding train set; score only the thinned (non-overlapping) test rows."""
    z, yr = S_full["z"], S_full["year"]
    ai_full = aidx(S_full["fwd"], thr)
    ai_sub = aidx(S_sub["fwd"], thr)
    years = sorted(set(int(v) for v in yr))
    y0 = years[0]
    rows = []
    for Y in years:
        if Y - y0 < MIN_TRAIN_YEARS:
            continue
        tr = yr < Y
        te = S_sub["year"] == Y
        if tr.sum() < MIN_TRAIN_ROWS or te.sum() < 8:
            continue
        ztr = z[tr]
        shte, aite, n = S_sub["sh"][te], ai_sub[te], int(te.sum())
        clim = np.clip(L.climatology(ai_full[tr], k=4), 1e-6, None)
        clim /= clim.sum()
        mu, sd = float(ztr.mean()), float(ztr.std(ddof=1))
        tf = qq_fit_t(ztr)
        emp = Emp(ztr, nu=tf["nu"])
        row = dict(year=Y, n_test=n, clim_ll=L.log_loss(np.tile(clim, (n, 1)), aite))
        for name, fn in (("normal01", ncdf),
                         ("normal_meansd", lambda k: ncdf((k - mu) / sd)),
                         ("emp_table", emp.cdf)):
            row[name] = L.log_loss(probs(fn, shte, thr), aite)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
