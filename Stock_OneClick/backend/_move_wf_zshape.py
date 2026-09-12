"""
_move_wf_zshape.py — SHAPE of standardized returns z = r_h / (vol_ewma(0.94) * sqrt(h)).

Answers: (1) moments+quantiles of z, (2) normal vs t tail mispricing, (3) up/down asymmetry with
date-block-bootstrap CIs, (4) shape stability across vol regimes, (5) shape drift 2001-13 vs 2014-26.

No scipy in this env -> normal/Student-t CDFs implemented below (erf + regularized incomplete beta).
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

import _move_data as D
import _move_lib as L

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", lambda v: f"{v:11.4f}")

HORIZONS = (1, 5, 10, 21)
LAM = 0.94
WARMUP = 250          # min valid log-returns before EWMA sigma is trusted
MIN_COV = 500         # drop near-empty symbols (repo rule)
IDX = ["SPY", "QQQ", "IWM", "DIA", "^GSPC"]
QLEV = [0.1, 0.5, 1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99, 99.5, 99.9]
KGRID = [-4, -3, -2.5, -2, -1.5, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4]
KASYM = [1.0, 1.5, 2.0, 3.0]
BLOCK = 63            # trading days per bootstrap block (>= max horizon, ~1 quarter)
NREP = 2000
RNG = np.random.default_rng(20260911)


# ----------------------------------------------------------------- distributions (no scipy)
def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _betacf(a, b, x, itmax=300, eps=3e-16):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betainc(a, b, x):
    """Regularized incomplete beta I_x(a,b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1.0 - x) / b


def t_cdf(x, nu):
    """CDF of a standard Student-t with nu df (variance nu/(nu-2), NOT 1)."""
    xx = nu / (nu + x * x)
    p = 0.5 * betainc(nu / 2.0, 0.5, xx)
    return p if x < 0 else 1.0 - p


def t_unit_cdf(x, nu):
    """CDF of a Student-t rescaled to UNIT variance (nu>2)."""
    return t_cdf(x * math.sqrt(nu / (nu - 2.0)), nu)


def t_unit_ppf(q, nu, lo=-60.0, hi=60.0):
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_unit_cdf(mid, nu) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def nu_from_exkurt(g2):
    """Excess kurtosis of t_nu = 6/(nu-4) -> nu = 4 + 6/g2. Undefined (<=4) for g2 huge."""
    return 4.0 + 6.0 / g2 if g2 > 0 else float("inf")


# ----------------------------------------------------------------- moments
def moments(a):
    a = a[np.isfinite(a)]
    n = a.size
    m = a.mean()
    d = a - m
    s2 = (d ** 2).mean()
    s = math.sqrt(s2)
    g1 = (d ** 3).mean() / s ** 3
    g2 = (d ** 4).mean() / s2 ** 2 - 3.0
    return dict(n=n, mean=m, std=a.std(ddof=1), skew=g1, exkurt=g2, sd_pop=s)


# ----------------------------------------------------------------- data
def build():
    p = D.load()
    close = p["Close"]
    cov = close.notna().sum()
    keep = cov[cov >= MIN_COV].index
    close = close[keep]
    # interior-gap audit: shift(-h) would silently span holes
    gaps = 0
    for c in close.columns:
        v = close[c]
        fv, lv = v.first_valid_index(), v.last_valid_index()
        gaps += int(v.loc[fv:lv].isna().sum())
    idx = [s for s in IDX if s in close.columns]
    singles = [s for s in close.columns if s not in idx and s != "^VIX"]
    print(f"panel: {close.shape[1]} symbols kept of {cov.size} (dropped {cov.size - len(keep)} "
          f"with <{MIN_COV} closes), {len(close)} rows {close.index[0].date()}"
          f"->{close.index[-1].date()}, interior NaN holes={gaps}")
    print(f"indices={idx}  singles n={len(singles)}")
    return close, idx, singles


def z_frames(close):
    """z[h] = simple fwd ret / (ewma sigma * sqrt h); zl[h] uses LOG fwd ret. Masked for warmup."""
    ret = np.log(close).diff()
    sig = L.vol_ewma(close, LAM)
    ok = (ret.notna().cumsum() >= WARMUP) & (sig > 1e-6) & sig.notna()
    out, outlog, sigs = {}, {}, {}
    for h in HORIZONS:
        sh = sig * math.sqrt(h)
        fwd = L.forward_simple_return(close, h)
        fwl = np.log(close.shift(-h) / close)
        m = ok & fwd.notna() & np.isfinite(fwd)
        out[h] = (fwd / sh).where(m)
        outlog[h] = (fwl / sh).where(m)
        sigs[h] = sh.where(m)
    return out, outlog, sigs, sig, ok


def flat(df, cols):
    return df[cols].to_numpy(dtype=float).ravel()


# ----------------------------------------------------------------- reports
def q_table(z, cols, label):
    rows = {}
    for h in HORIZONS:
        a = flat(z[h], cols)
        a = a[np.isfinite(a)]
        mo = moments(a)
        r = {"n": mo["n"], "mean": mo["mean"], "std": mo["std"],
             "skew": mo["skew"], "exkurt": mo["exkurt"]}
        qs = np.percentile(a, QLEV)
        for lev, v in zip(QLEV, qs):
            r[f"q{lev:g}"] = v
        rows[f"h={h}"] = r
    t = pd.DataFrame(rows).T
    print(f"\n=== [{label}] raw z moments + percentiles (IN-SAMPLE descriptive) ===")
    print(t.to_string())
    return t


def per_symbol_moments(z, cols, label):
    print(f"\n=== [{label}] per-symbol moments (median across symbols) vs POOLED ===")
    rows = []
    for h in HORIZONS:
        sub = z[h][cols]
        pooled = moments(sub.to_numpy(dtype=float).ravel())
        ms, ss, gs, ks = [], [], [], []
        zw = []
        for c in cols:
            a = sub[c].to_numpy(dtype=float)
            a = a[np.isfinite(a)]
            if a.size < 300:
                continue
            mo = moments(a)
            ms.append(mo["mean"]); ss.append(mo["std"]); gs.append(mo["skew"]); ks.append(mo["exkurt"])
            zw.append((a - mo["mean"]) / mo["std"])          # within-symbol standardized
        zw = np.concatenate(zw)
        mw = moments(zw)
        rows.append(dict(h=h, n_sym=len(ss), pooled_std=pooled["std"], med_sym_std=np.median(ss),
                         disp_sym_std=np.std(ss), pooled_skew=pooled["skew"], med_sym_skew=np.median(gs),
                         pooled_exkurt=pooled["exkurt"], med_sym_exkurt=np.median(ks),
                         within_std_exkurt=mw["exkurt"], within_std_skew=mw["skew"]))
    t = pd.DataFrame(rows).set_index("h")
    print(t.to_string())
    return t


def tail_table(z, cols, label):
    print(f"\n=== [{label}] tail probabilities: empirical vs Normal vs kurtosis-matched t ===")
    print("    (A) RAW z vs N(0,1)  -- what you get if you ship sigma_hat=ewma*sqrt(h) + normal")
    print("    (B) STANDARDIZED z=(z-mu)/s vs N(0,1) and vs unit-variance t_nu  -- pure SHAPE")
    outs = []
    for h in HORIZONS:
        a = flat(z[h], cols)
        a = a[np.isfinite(a)]
        mo = moments(a)
        zs = (a - mo["mean"]) / mo["std"]
        nu = nu_from_exkurt(mo["exkurt"])
        nu_use = max(nu, 4.3) if np.isfinite(nu) else 30.0
        for k in (-3, -2, 2, 3):
            emp_raw = float((a < k).mean()) if k < 0 else float((a > k).mean())
            emp_std = float((zs < k).mean()) if k < 0 else float((zs > k).mean())
            nrm = norm_cdf(k) if k < 0 else 1 - norm_cdf(k)
            tp = t_unit_cdf(k, nu_use) if k < 0 else 1 - t_unit_cdf(k, nu_use)
            outs.append(dict(h=h, k=k, n=mo["n"], nu_kurt=nu,
                             emp_raw=emp_raw, emp_std=emp_std, normal=nrm, t_nu=tp,
                             emp_over_normal=emp_std / nrm if nrm else np.nan,
                             emp_over_t=emp_std / tp if tp else np.nan))
    t = pd.DataFrame(outs)
    print(t.to_string(index=False))
    return t


def cdf_grid(z, cols, label):
    print(f"\n=== [{label}] SHIPPABLE table: empirical P(z<k) for raw z, plus normal for reference ===")
    rows = {}
    for h in HORIZONS:
        a = flat(z[h], cols)
        a = a[np.isfinite(a)]
        r = {"n": a.size}
        for k in KGRID:
            r[f"{k:g}"] = float((a < k).mean())
        rows[f"h={h}"] = r
    t = pd.DataFrame(rows).T
    nrm = pd.Series({f"{k:g}": norm_cdf(k) for k in KGRID}, name="N(0,1)")
    print(t.to_string())
    print("\n normal ref: " + "  ".join(f"{k:g}:{norm_cdf(k):.4f}" for k in KGRID))
    return t, nrm


def fit_t_ks(z, cols, label):
    """Ship-a-t fit: loc/scale from mean/std, nu chosen to minimise max|F_t - F_emp| on KGRID."""
    print(f"\n=== [{label}] Student-t fits (loc/scale = empirical mean/std; nu by kurtosis vs by KS) ===")
    rows = []
    for h in HORIZONS:
        a = flat(z[h], cols)
        a = a[np.isfinite(a)]
        mo = moments(a)
        zs = (a - mo["mean"]) / mo["std"]
        emp = np.array([(zs < k).mean() for k in KGRID])
        best = None
        for nu in np.arange(2.6, 30.01, 0.05):
            mod = np.array([t_unit_cdf(k, nu) for k in KGRID])
            ks = np.abs(mod - emp).max()
            if best is None or ks < best[1]:
                best = (nu, ks)
        nrm = np.array([norm_cdf(k) for k in KGRID])
        nk = nu_from_exkurt(mo["exkurt"])
        nk_use = max(nk, 4.3)
        modk = np.array([t_unit_cdf(k, nk_use) for k in KGRID])
        rows.append(dict(h=h, n=mo["n"], mu=mo["mean"], scale_corr=1 / mo["std"], std=mo["std"],
                         nu_kurt=nk, ks_nu_kurt=np.abs(modk - emp).max(),
                         nu_ks=best[0], ks_best=best[1], ks_normal=np.abs(nrm - emp).max()))
    t = pd.DataFrame(rows).set_index("h")
    print(t.to_string())
    return t


def _per_date_counts(zdf, cols, ks, center):
    sub = zdf[cols]
    if center is not None:
        sub = sub - center
    n = sub.notna().sum(axis=1)
    d = {"n": n}
    for k in ks:
        d[f"lo{k}"] = (sub < -k).sum(axis=1)
        d[f"hi{k}"] = (sub > k).sum(axis=1)
    out = pd.DataFrame(d)
    return out[out.n > 0]


def asymmetry(z, cols, label, centered):
    """Date-block bootstrap (moving blocks of BLOCK consecutive DATES) -> clusters by date AND
    absorbs the horizon-h overlap. Ratio = P(z<-k)/P(z>+k)."""
    tag = "MEAN-CENTERED (drift removed)" if centered else "RAW (drift included)"
    print(f"\n=== [{label}] up/down asymmetry, {tag}; {NREP} date-block bootstrap reps, block={BLOCK}d ===")
    rows = []
    for h in HORIZONS:
        mu = float(np.nanmean(z[h][cols].to_numpy(dtype=float))) if centered else None
        pdc = _per_date_counts(z[h], cols, KASYM, mu)
        A = pdc.to_numpy(dtype=float)
        nd = len(pdc)
        nblk = int(math.ceil(nd / BLOCK))
        starts_pool = np.arange(0, nd - BLOCK + 1)
        # precompute block sums for speed
        cs = np.vstack([np.zeros(A.shape[1]), np.cumsum(A, axis=0)])
        blk = cs[starts_pool + BLOCK] - cs[starts_pool]          # (nstart, ncol)
        pick = RNG.integers(0, len(starts_pool), size=(NREP, nblk))
        tot = blk[pick].sum(axis=1)                              # (NREP, ncol)
        nrep_n = tot[:, 0]
        cols_ = list(pdc.columns)
        rec = dict(h=h, n_obs=int(A[:, 0].sum()), n_dates=nd, mu_removed=mu)
        for k in KASYM:
            i_lo, i_hi = cols_.index(f"lo{k}"), cols_.index(f"hi{k}")
            plo_pt = A[:, i_lo].sum() / A[:, 0].sum()
            phi_pt = A[:, i_hi].sum() / A[:, 0].sum()
            r_b = (tot[:, i_lo] / np.maximum(tot[:, i_hi], 1e-12))
            lo95, hi95 = np.percentile(r_b, [2.5, 97.5])
            rec[f"P(z<-{k})"] = plo_pt
            rec[f"P(z>+{k})"] = phi_pt
            rec[f"ratio_{k}"] = plo_pt / phi_pt if phi_pt else np.nan
            rec[f"ci_{k}"] = f"[{lo95:.3f},{hi95:.3f}]"
            rec[f"p_gt1_{k}"] = float((r_b > 1).mean())
        rows.append(rec)
    t = pd.DataFrame(rows).set_index("h")
    print(t.to_string())
    return t


def vol_regime(z, sig, cols, label):
    print(f"\n=== [{label}] shape by VOL REGIME (quintile of expanding pct-rank of sigma_ewma, "
          f"no lookahead) ===")
    rk = sig[cols].expanding().rank(pct=True)
    rows = []
    for h in HORIZONS:
        zz = z[h][cols]
        m = zz.notna()
        rr = rk.where(m).to_numpy(dtype=float).ravel()
        aa = zz.to_numpy(dtype=float).ravel()
        g = np.isfinite(rr) & np.isfinite(aa)
        rr, aa = rr[g], aa[g]
        qb = np.digitize(rr, [0.2, 0.4, 0.6, 0.8])
        for q in range(5):
            a = aa[qb == q]
            mo = moments(a)
            nu = nu_from_exkurt(mo["exkurt"])
            rows.append(dict(h=h, volq=q + 1, n=mo["n"], mean=mo["mean"], std=mo["std"],
                             skew=mo["skew"], exkurt=mo["exkurt"], nu_kurt=nu,
                             p_lt_m2=float((a < -2).mean()), p_gt_p2=float((a > 2).mean()),
                             q05=np.percentile(a, 5), q95=np.percentile(a, 95),
                             q01=np.percentile(a, 1), q99=np.percentile(a, 99)))
    t = pd.DataFrame(rows).set_index(["h", "volq"])
    print(t.to_string())
    return t


def era_split(z, cols, label):
    print(f"\n=== [{label}] shape drift: 2001-2013 vs 2014-2026 (split on observation date t) ===")
    rows = []
    for h in HORIZONS:
        zz = z[h][cols]
        for name, mask in (("2001-2013", zz.index.year <= 2013), ("2014-2026", zz.index.year >= 2014)):
            a = zz[mask].to_numpy(dtype=float).ravel()
            a = a[np.isfinite(a)]
            mo = moments(a)
            r = dict(h=h, era=name, n=mo["n"], mean=mo["mean"], std=mo["std"], skew=mo["skew"],
                     exkurt=mo["exkurt"])
            for lev in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                r[f"q{lev}"] = np.percentile(a, lev)
            r["P(z<-2)"] = float((a < -2).mean())
            r["P(z>2)"] = float((a > 2).mean())
            rows.append(r)
    t = pd.DataFrame(rows).set_index(["h", "era"])
    print(t.to_string())
    return t


def k_scale_context(sigs, cols, label):
    print(f"\n=== [{label}] which |k| actually matters: k = 0.02 / sigma_hat_h ===")
    rows = []
    for h in HORIZONS:
        s = sigs[h][cols].to_numpy(dtype=float).ravel()
        s = s[np.isfinite(s) & (s > 0)]
        k = 0.02 / s
        rows.append(dict(h=h, n=k.size, med_sigma_h=np.median(s), k_p10=np.percentile(k, 10),
                         k_q1=np.percentile(k, 25), k_med=np.median(k), k_q3=np.percentile(k, 75),
                         k_p90=np.percentile(k, 90), frac_k_lt_1=float((k < 1).mean()),
                         frac_k_lt_2=float((k < 2).mean())))
    t = pd.DataFrame(rows).set_index("h")
    print(t.to_string())
    return t


def outlier_audit(z, cols, label, n=12):
    print(f"\n=== [{label}] largest |z| observations (data-error sanity check, h=1) ===")
    s = z[1][cols].stack()
    s = s.reindex(s.abs().sort_values(ascending=False).index)[:n]
    print(s.to_string())


def main():
    close, idx, singles = build()
    z, zl, sigs, sig, ok = z_frames(close)

    for label, cols in (("INDICES(5)", idx), ("SPY only", ["SPY"]), ("^GSPC only", ["^GSPC"]),
                        ("SINGLES pooled", singles)):
        q_table(z, cols, label)

    k_scale_context(sigs, singles, "SINGLES pooled")
    k_scale_context(sigs, ["SPY"], "SPY")

    outlier_audit(z, singles, "SINGLES pooled")

    per_symbol_moments(z, singles, "SINGLES pooled")

    for label, cols in (("INDICES(5)", idx), ("SPY only", ["SPY"]), ("SINGLES pooled", singles)):
        tail_table(z, cols, label)
        cdf_grid(z, cols, label)
        fit_t_ks(z, cols, label)

    for label, cols in (("INDICES(5)", idx), ("SPY only", ["SPY"]), ("SINGLES pooled", singles)):
        asymmetry(z, cols, label, centered=False)
        asymmetry(z, cols, label, centered=True)

    for label, cols in (("SPY only", ["SPY"]), ("SINGLES pooled", singles)):
        vol_regime(z, sig, cols, label)
        era_split(z, cols, label)

    print("\n=== LOG-return z (simple-vs-log skew decomposition), pooled singles & SPY ===")
    for label, cols in (("SPY", ["SPY"]), ("SINGLES", singles)):
        rows = []
        for h in HORIZONS:
            a = flat(z[h], cols); a = a[np.isfinite(a)]
            b = flat(zl[h], cols); b = b[np.isfinite(b)]
            ma, mb = moments(a), moments(b)
            rows.append(dict(h=h, n=ma["n"], skew_simple=ma["skew"], skew_log=mb["skew"],
                             exkurt_simple=ma["exkurt"], exkurt_log=mb["exkurt"],
                             mean_simple=ma["mean"], mean_log=mb["mean"],
                             std_simple=ma["std"], std_log=mb["std"]))
        print(f"\n[{label}]")
        print(pd.DataFrame(rows).set_index("h").to_string())


if __name__ == "__main__":
    main()
