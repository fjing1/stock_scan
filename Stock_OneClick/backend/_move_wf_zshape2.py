"""
_move_wf_zshape2.py — part 2 of the z-SHAPE study.

(a) ROBUST shape statistics (pooled single-name moments are destroyed by real microcap jumps).
(b) Does the shape COLLAPSE across horizons once you remove the sqrt(h) drift?
(c) Is the vol-regime dependence about SCALE or about SHAPE?
(d) WALK-FORWARD OUT-OF-SAMPLE horse race: normal vs fitted-t vs empirical z-table vs
    regime-conditional z-table, scored by 4-bucket log loss at thr=2% against CLIMATOLOGY.
    Everything is fit on years strictly before the test year.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import _move_data as D
import _move_lib as L

pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 80)
pd.set_option("display.float_format", lambda v: f"{v:10.4f}")

HORIZONS = (1, 5, 10, 21)
LAM, WARMUP, MIN_COV, THR = 0.94, 250, 500, 0.02
IDX = ["SPY", "QQQ", "IWM", "DIA", "^GSPC"]
QFIT = [0.5, 1, 2.5, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 97.5, 99, 99.5]
QSHIP = [0.5, 1, 2, 2.5, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 97.5, 98, 99, 99.5]
NU_GRID = np.arange(2.6, 30.01, 0.1)


# --------------------------------------------------------- vectorised Student-t (no scipy)
def _betacf_vec(a, b, x, itmax=300, eps=3e-16):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    tiny = 1e-300
    c = np.ones_like(x)
    d = 1.0 - qab * x / qap
    d = np.where(np.abs(d) < tiny, tiny, d)
    d = 1.0 / d
    h = d.copy()
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = np.where(np.abs(d) < tiny, tiny, d)
        c = 1.0 + aa / c
        c = np.where(np.abs(c) < tiny, tiny, c)
        d = 1.0 / d
        h = h * d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = np.where(np.abs(d) < tiny, tiny, d)
        c = 1.0 + aa / c
        c = np.where(np.abs(c) < tiny, tiny, c)
        d = 1.0 / d
        de = d * c
        h = h * de
        if np.max(np.abs(de - 1.0)) < eps:
            break
    return h


def betainc_vec(a, b, x):
    x = np.asarray(x, dtype=float)
    out = np.zeros(x.shape)
    out[x >= 1.0] = 1.0
    m = (x > 0.0) & (x < 1.0)
    if not m.any():
        return out
    xm = x[m]
    lb = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + a * np.log(xm) + b * np.log1p(-xm))
    thr = (a + 1.0) / (a + b + 2.0)
    res = np.empty_like(xm)
    lo = xm < thr
    if lo.any():
        res[lo] = np.exp(lb[lo]) * _betacf_vec(a, b, xm[lo]) / a
    hi = ~lo
    if hi.any():
        res[hi] = 1.0 - np.exp(lb[hi]) * _betacf_vec(b, a, 1.0 - xm[hi]) / b
    out[m] = res
    return out


def t_cdf_vec(t, nu):
    t = np.asarray(t, dtype=float)
    xx = nu / (nu + t * t)
    p = 0.5 * betainc_vec(nu / 2.0, 0.5, xx)
    return np.where(t < 0, p, 1.0 - p)


def t_unit_cdf_vec(x, nu):
    """CDF of a Student-t rescaled to unit variance."""
    return t_cdf_vec(np.asarray(x, float) * math.sqrt(nu / (nu - 2.0)), nu)


def t_unit_ppf_vec(q, nu, iters=70):
    q = np.asarray(q, float)
    lo = np.full(q.shape, -200.0)
    hi = np.full(q.shape, 200.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        c = t_unit_cdf_vec(mid, nu)
        up = c < q
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    return 0.5 * (lo + hi)


def norm_cdf_vec(x):
    x = np.asarray(x, float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


_PPF_CACHE: dict[float, np.ndarray] = {}


def qq_fit_t(a, levels=QFIT):
    """Fit loc/scale/nu by QQ regression: emp_q ~ loc + scale * t_unit_q(nu). Robust (quantiles
    only, so the microcap 400-sigma jumps cannot move it) and gives shippable parameters."""
    eq = np.percentile(a, levels)
    best = None
    for nu in NU_GRID:
        key = round(float(nu), 4)
        if key not in _PPF_CACHE:
            _PPF_CACHE[key] = t_unit_ppf_vec(np.array(levels) / 100.0, nu)
        tq = _PPF_CACHE[key]
        A = np.vstack([np.ones_like(tq), tq]).T
        coef, *_ = np.linalg.lstsq(A, eq, rcond=None)
        sse = float(((A @ coef - eq) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, float(nu), float(coef[0]), float(coef[1]))
    sse, nu, loc, scale = best
    return dict(nu=nu, loc=loc, scale=scale, rmse_q=math.sqrt(sse / len(levels)))


# --------------------------------------------------------- robust shape stats
def robust_shape(a):
    a = a[np.isfinite(a)]
    q = np.percentile(a, [0.5, 1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99, 99.5])
    (q005, q1, q25_, q5, q10, q25, q50, q75, q90, q95, q975, q99, q995) = q
    iqr = q75 - q25
    lo, hi = np.percentile(a, [0.1, 99.9])
    tr = a[(a >= lo) & (a <= hi)]
    m = tr.mean(); s = tr.std(ddof=1)
    g1 = (((tr - m) / s) ** 3).mean(); g2 = (((tr - m) / s) ** 4).mean() - 3
    return dict(n=a.size, median=q50, iqr_scale=iqr / 1.34898,
                mad_scale=np.median(np.abs(a - q50)) / 0.67449,
                bowley_skew=(q90 + q10 - 2 * q50) / (q90 - q10),
                skew_5_95=(q95 + q5 - 2 * q50) / (q95 - q5),
                skew_1_99=(q99 + q1 - 2 * q50) / (q99 - q1),
                crow_kurt=(q975 - q25_) / iqr,          # normal = 2.906
                kurt_1_99=(q99 - q1) / iqr,             # normal = 3.449
                trim_std=s, trim_skew=g1, trim_exkurt=g2)


# --------------------------------------------------------- data
def build():
    p = D.load()
    close = p["Close"]
    cov = close.notna().sum()
    close = close[cov[cov >= MIN_COV].index]
    ret = np.log(close).diff()
    sig = L.vol_ewma(close, LAM)
    ok = (ret.notna().cumsum() >= WARMUP) & (sig > 1e-6) & sig.notna()
    idx = [s for s in IDX if s in close.columns]
    singles = [c for c in close.columns if c not in idx and c != "^VIX"]
    rank = sig.expanding().rank(pct=True).where(ok)
    Z, SH, FWD = {}, {}, {}
    for h in HORIZONS:
        sh = sig * math.sqrt(h)
        fwd = L.forward_simple_return(close, h)
        m = ok & fwd.notna() & np.isfinite(fwd)
        SH[h] = sh.where(m); FWD[h] = fwd.where(m); Z[h] = (fwd / sh).where(m)
    return close, idx, singles, Z, SH, FWD, sig, rank


def col(df, cols):
    return df[cols]


def stack3(Z, SH, FWD, rank, h, cols):
    """Long-form arrays for one horizon/group: z, sigma_hat_h, fwd, vol-rank, year."""
    z = Z[h][cols]
    m = z.notna()
    yr = np.repeat(z.index.year.to_numpy()[:, None], len(cols), axis=1)
    out = {}
    for name, df in (("z", z), ("sh", SH[h][cols]), ("fwd", FWD[h][cols]), ("rk", rank[cols])):
        out[name] = df.to_numpy(dtype=float)[m.to_numpy()]
    out["year"] = yr[m.to_numpy()]
    g = np.isfinite(out["z"]) & np.isfinite(out["sh"]) & np.isfinite(out["fwd"]) & np.isfinite(out["rk"])
    return {k: v[g] for k, v in out.items()}


# --------------------------------------------------------- empirical CDF model
class EmpCDF:
    def __init__(self, a, tail_nu=4.0):
        a = np.sort(a[np.isfinite(a)])
        self.a = a
        self.n = a.size
        # Pareto-ish extrapolation beyond the sample: fall back on a fitted t for |k| outside range
        self.lo, self.hi = a[0], a[-1]
        self.med = np.median(a)
        self.scale = (np.percentile(a, 75) - np.percentile(a, 25)) / 1.34898
        self.nu = tail_nu

    def cdf(self, k):
        k = np.asarray(k, float)
        r = np.searchsorted(self.a, k, side="left").astype(float)
        p = (r + 0.5) / (self.n + 1.0)
        out = np.clip(p, 0.5 / (self.n + 1.0), 1.0 - 0.5 / (self.n + 1.0))
        # beyond the empirical support, blend to a t tail so probabilities stay strictly monotone
        frac = 0.5 / (self.n + 1.0)
        far_lo, far_hi = k < self.lo, k > self.hi
        if far_lo.any():
            t = t_unit_cdf_vec((k[far_lo] - self.med) / self.scale, self.nu)
            out[far_lo] = np.minimum(t, frac)
        if far_hi.any():
            t = t_unit_cdf_vec((k[far_hi] - self.med) / self.scale, self.nu)
            out[far_hi] = np.maximum(t, 1.0 - frac)
        return out


def probs_from_cdf(cdf_fn, sh, thr=THR):
    """4-bucket probabilities from any CDF of z. r<=-thr <=> z<=-thr/sh ; r>=thr <=> z>=thr/sh."""
    k_dn, k_up = -thr / sh, thr / sh
    F_dn, F_0, F_up = cdf_fn(k_dn), cdf_fn(np.zeros_like(sh)), cdf_fn(k_up)
    P = np.column_stack([F_dn, F_0 - F_dn, F_up - F_0, 1.0 - F_up])
    P = np.clip(P, 1e-6, None)
    return P / P.sum(axis=1, keepdims=True)


def actual_idx(fwd, thr=THR):
    b = L.bucketize(pd.Series(fwd), thr).to_numpy()
    order = {v: i for i, v in enumerate(L.BUCKETS)}
    return np.array([order[x] for x in b], dtype=int)


# --------------------------------------------------------- reports
def part_a(Z, idx, singles):
    print("=" * 120)
    print("(a) ROBUST shape statistics (quantile-based; immune to the real microcap 400-sigma jumps)")
    for label, cols in (("INDICES(5)", idx), ("SPY", ["SPY"]), ("SINGLES", singles)):
        rows = []
        for h in HORIZONS:
            a = Z[h][cols].to_numpy(dtype=float).ravel()
            r = robust_shape(a)
            r["h"] = h
            rows.append(r)
        t = pd.DataFrame(rows).set_index("h")
        print(f"\n[{label}]   (normal reference: bowley/skew=0, crow_kurt=2.906, kurt_1_99=3.449)")
        print(t.to_string())


def part_b(Z, idx, singles):
    print("\n" + "=" * 120)
    print("(b) Does the shape COLLAPSE across horizons? mu_z should be c*sqrt(h) (drift/vol ratio).")
    for label, cols in (("INDICES(5)", idx), ("SPY", ["SPY"]), ("SINGLES", singles)):
        rows = []
        for h in HORIZONS:
            a = Z[h][cols].to_numpy(dtype=float).ravel()
            a = a[np.isfinite(a)]
            med = np.median(a)
            rows.append(dict(h=h, n=a.size, median=med, med_over_sqrth=med / math.sqrt(h),
                             mean_trim=np.mean(a[(a > np.percentile(a, 0.1)) & (a < np.percentile(a, 99.9))]),
                             iqr_scale=(np.percentile(a, 75) - np.percentile(a, 25)) / 1.34898))
        print(f"\n[{label}] location scaling")
        print(pd.DataFrame(rows).set_index("h").to_string())
        # median-centred, IQR-rescaled quantiles: if these collapse, one table is enough
        tab = {}
        for h in HORIZONS:
            a = Z[h][cols].to_numpy(dtype=float).ravel()
            a = a[np.isfinite(a)]
            med = np.median(a)
            s = (np.percentile(a, 75) - np.percentile(a, 25)) / 1.34898
            zz = (a - med) / s
            tab[f"h={h}"] = {f"q{q:g}": v for q, v in zip(QSHIP, np.percentile(zz, QSHIP))}
        print(f"[{label}] median-centred / IQR-rescaled quantiles (collapse test)")
        print(pd.DataFrame(tab).T.to_string())


def part_c(Z, rank, idx, singles):
    print("\n" + "=" * 120)
    print("(c) VOL REGIME: is it SCALE or SHAPE? Within each vol quintile, rescale by that "
          "quintile's own IQR scale and re-measure shape.")
    for label, cols in (("SPY", ["SPY"]), ("INDICES(5)", idx), ("SINGLES", singles)):
        rows = []
        for h in HORIZONS:
            s = stack3(Z, {h: Z[h] * 0 + 1}, {h: Z[h]}, rank, h, cols)   # need z + rk only
            z, rk = s["z"], s["rk"]
            qb = np.digitize(rk, [0.2, 0.4, 0.6, 0.8])
            for q in range(5):
                a = z[qb == q]
                med = np.median(a)
                sc = (np.percentile(a, 75) - np.percentile(a, 25)) / 1.34898
                zz = (a - med) / sc
                r = robust_shape(zz)
                rows.append(dict(h=h, volq=q + 1, n=a.size, raw_iqr_scale=sc, raw_median=med,
                                 std_scaled_bowley=r["bowley_skew"], skew_5_95=r["skew_5_95"],
                                 crow_kurt=r["crow_kurt"], kurt_1_99=r["kurt_1_99"],
                                 q1=np.percentile(zz, 1), q5=np.percentile(zz, 5),
                                 q95=np.percentile(zz, 95), q99=np.percentile(zz, 99)))
        t = pd.DataFrame(rows).set_index(["h", "volq"])
        print(f"\n[{label}] (raw_iqr_scale shows the SCALE error of ewma*sqrt(h); the rest is SHAPE)")
        print(t.to_string())


def part_d(Z, SH, FWD, rank, idx, singles):
    print("\n" + "=" * 120)
    print("(d) WALK-FORWARD OUT-OF-SAMPLE: 4-bucket log loss at thr=2%, params fit on years < Y only.")
    print("    baseline = CLIMATOLOGY of the TRAIN years (honest). skill = 1 - model/clim.")
    allrows = []
    for label, cols in (("SPY", ["SPY"]), ("INDICES(5)", idx), ("SINGLES", singles)):
        for h in HORIZONS:
            s = stack3(Z, SH, FWD, rank, h, cols)
            z, sh, fwd, rk, yr = s["z"], s["sh"], s["fwd"], s["rk"], s["year"]
            ai = actual_idx(fwd)
            years = sorted(set(yr.tolist()))
            for Y in years:
                tr = yr < Y
                te = yr == Y
                if tr.sum() < 2000 or te.sum() < 100 or Y - years[0] < 5:
                    continue
                ztr, shte, aite = z[tr], sh[te], ai[te]
                clim = L.climatology(ai[tr], k=4)
                clim = np.clip(clim, 1e-6, None); clim /= clim.sum()
                Pc = np.tile(clim, (te.sum(), 1))
                base = L.log_loss(Pc, aite)
                mu, sd = ztr.mean(), ztr.std(ddof=1)
                med = np.median(ztr)
                iqs = (np.percentile(ztr, 75) - np.percentile(ztr, 25)) / 1.34898
                tf = qq_fit_t(ztr)
                emp = EmpCDF(ztr, tail_nu=tf["nu"])
                models = {
                    "normal01": lambda k: norm_cdf_vec(k),
                    "normal_fit": lambda k: norm_cdf_vec((k - mu) / sd),
                    "normal_robust": lambda k: norm_cdf_vec((k - med) / iqs),
                    "t_qqfit": lambda k: t_unit_cdf_vec((k - tf["loc"]) / tf["scale"], tf["nu"]),
                    "emp_table": emp.cdf,
                }
                # regime-conditional empirical table (tercile of vol rank, fit on train)
                edges = [1 / 3, 2 / 3]
                trb = np.digitize(rk[tr], edges)
                teb = np.digitize(rk[te], edges)
                sub = {}
                okreg = True
                for b in range(3):
                    aa = ztr[trb == b]
                    if aa.size < 500:
                        okreg = False
                        break
                    sub[b] = EmpCDF(aa, tail_nu=tf["nu"])
                row = dict(group=label, h=h, year=Y, n_test=int(te.sum()), n_train=int(tr.sum()),
                           clim_ll=base)
                for name, fn in models.items():
                    P = probs_from_cdf(fn, shte)
                    row[name] = L.log_loss(P, aite)
                if okreg:
                    P = np.zeros((te.sum(), 4))
                    for b in range(3):
                        m = teb == b
                        if m.any():
                            P[m] = probs_from_cdf(sub[b].cdf, shte[m])
                    row["emp_regime"] = L.log_loss(P, aite)
                allrows.append(row)
    df = pd.DataFrame(allrows)
    df.to_pickle("/tmp/zshape_wf.pkl")
    mods = [c for c in ("normal01", "normal_fit", "normal_robust", "t_qqfit", "emp_table",
                        "emp_regime") if c in df.columns]
    print(f"\ntest-year rows: {len(df)}; models: {mods}")
    for label in df.group.unique():
        d = df[df.group == label]
        print(f"\n--- [{label}] mean OOS log loss by horizon (equal weight per test year) ---")
        agg = d.groupby("h")[["clim_ll"] + mods].mean()
        agg["n_years"] = d.groupby("h").size()
        agg["n_test_obs"] = d.groupby("h").n_test.sum()
        print(agg.to_string())
        sk = pd.DataFrame({m: 1 - d.groupby("h")[m].mean() / d.groupby("h").clim_ll.mean()
                           for m in mods})
        print(f"[{label}] SKILL vs train-climatology (positive = beats climatology)")
        print(sk.to_string())
        wr = pd.DataFrame({m: d.assign(w=(d[m] < d.clim_ll)).groupby("h").w.mean() for m in mods})
        print(f"[{label}] fraction of test YEARS beating climatology")
        print(wr.to_string())
        # per-year dispersion for the shipped candidate
        for m in ("emp_table", "emp_regime", "t_qqfit"):
            if m not in d.columns:
                continue
            g = d.groupby("h").apply(
                lambda x: pd.Series({"mean_gain": (x.clim_ll - x[m]).mean(),
                                     "sd_gain": (x.clim_ll - x[m]).std(ddof=1),
                                     "t_stat": (x.clim_ll - x[m]).mean() /
                                               ((x.clim_ll - x[m]).std(ddof=1) / math.sqrt(len(x)))}),
                include_groups=False)
            print(f"[{label}] {m}: per-test-year log-loss gain vs climatology (n_years as above)")
            print(g.to_string())
    return df


def part_e(Z, idx, singles):
    print("\n" + "=" * 120)
    print("(e) SHIPPABLE tables (fit on ALL data -> IN-SAMPLE; validated OOS in part d).")
    for label, cols in (("INDICES(5)", idx), ("SPY", ["SPY"]), ("SINGLES", singles)):
        tab, params = {}, []
        for h in HORIZONS:
            a = Z[h][cols].to_numpy(dtype=float).ravel()
            a = a[np.isfinite(a)]
            tab[f"h={h}"] = {f"q{q:g}": v for q, v in zip(QSHIP, np.percentile(a, QSHIP))}
            tf = qq_fit_t(a)
            tf.update(h=h, n=a.size, loc_over_sqrth=tf["loc"] / math.sqrt(h))
            params.append(tf)
        print(f"\n[{label}] raw-z quantile table to ship")
        print(pd.DataFrame(tab).T.to_string())
        print(f"[{label}] Student-t QQ fit (z ~ loc + scale * t_nu/sqrt(nu/(nu-2)))")
        print(pd.DataFrame(params).set_index("h").to_string())


def main():
    close, idx, singles, Z, SH, FWD, sig, rank = build()
    print(f"universe: {len(idx)} indices, {len(singles)} singles, rows={len(close)}")
    part_a(Z, idx, singles)
    part_b(Z, idx, singles)
    part_c(Z, rank, idx, singles)
    part_e(Z, idx, singles)
    part_d(Z, SH, FWD, rank, idx, singles)


if __name__ == "__main__":
    main()
