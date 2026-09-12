"""
_move_wf_verify_zskew.py — INDEPENDENT adversarial verification of the z-asymmetry claim.

CLAIM UNDER TEST
  "Index z is genuinely left-asymmetric beyond drift and the asymmetry grows with horizon, but
   single-name z is symmetric beyond drift -- so the two asset classes need different shape tables,
   and P(down 2%) != P(up 2%) for single names is driven purely by drift, not by skew."

Everything is rebuilt from _move_data + _move_lib here (no import of the analysis script's build)
so a bug in the original cannot propagate silently.  z = fwd_simple_return_h / (ewma(0.94)*sqrt(h))
is reproduced exactly as the claim defines it, then attacked.

SECTIONS  (python _move_wf_verify_zskew.py 0 1 2 ...)
  0  reproduce: pooled median-centred P(z<-k)/P(z>+k), 63-day date-block bootstrap; Bowley skew
  1  how many INDEPENDENT tail episodes is the index ratio built on?  raw counts, episode clustering
  2  UNITS ARTEFACT: numerator is a SIMPLE return, denominator a LOG sigma. exp(x)-1 is convex, so
     +skew ~ O(sigma_h) is injected, and singles have 2-3x the index sigma.  Re-run in log space.
  3  POOLING ARTEFACT: pooled singles z mixes 278 different drift/vol ratios (locations). A
     right-skewed spread of locations manufactures +skew. Re-centre PER SYMBOL and re-run.
  4  REGIME MIXTURE: one pooled median mixes vol regimes whose z-locations differ. Centre within
     vol quintile.
  5  ASSET CLASS or VOL LEVEL? per-symbol skew vs log10(median sigma); index residual vs the
     singles regression line; vol-matched cohorts.
  6  FRAGILITY: drop 2008-09 & 2020; per index member (SPY/^GSPC alone); non-overlapping stride-h
     dates; longer bootstrap blocks.
  7  OOS: paired log-loss cost of symmetrising, clustered by TEST YEAR and by date-block, at all
     four horizons, with train rows purged of forward-window overlap into the test year.
  8  SIMPSON: OOS calibration (ECE) of the shipped vs symmetrised vs log-symmetric singles table
     within vol quintiles and within calendar years.
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

import _move_data as D
import _move_lib as L

pd.set_option("display.width", 320)
pd.set_option("display.max_columns", 120)
pd.set_option("display.max_rows", 300)
pd.set_option("display.float_format", lambda v: f"{v:9.4f}")

HORIZONS = (1, 5, 10, 21)
LAM, WARMUP, MIN_COV, THR = 0.94, 250, 500, 0.02
IDX = ["SPY", "QQQ", "IWM", "DIA", "^GSPC"]
KA = (1.0, 1.5, 2.0, 2.5)
NREP, SEED, BLOCK = 2000, 7, 63

# histogram grid for the block bootstrap (0.01 z-units; edge bins absorb the microcap 400-sigma tails)
HLO, HHI, HNB = -20.0, 20.0, 4000
EDGES = np.linspace(HLO, HHI, HNB + 1)
CENTERS = 0.5 * (EDGES[1:] + EDGES[:-1])


# ------------------------------------------------------------------ data
_S = {}


def prep():
    if _S:
        return _S
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
    Z, ZL, SH, FWD = {}, {}, {}, {}
    for h in HORIZONS:
        sh = sig * math.sqrt(h)
        fwd = L.forward_simple_return(close, h)
        m = ok & fwd.notna() & np.isfinite(fwd)
        SH[h] = sh.where(m)
        FWD[h] = fwd.where(m)
        Z[h] = (fwd / sh).where(m)
        ZL[h] = (np.log1p(fwd) / sh).where(m)
    _S.update(close=close, idx=idx, singles=singles, Z=Z, ZL=ZL, SH=SH, FWD=FWD,
              sig=sig, rank=rank, dates=close.index)
    return _S


def long(frame, cols, extra=None):
    """Flatten one (dates x cols) frame to values + date-row-index, dropping NaN."""
    sub = frame[cols]
    v = sub.to_numpy(dtype=float)
    m = np.isfinite(v)
    di = np.repeat(np.arange(v.shape[0])[:, None], v.shape[1], axis=1)[m]
    ci = np.repeat(np.arange(v.shape[1])[None, :], v.shape[0], axis=0)[m]
    out = dict(v=v[m], d=di, c=ci)
    for k, f in (extra or {}).items():
        out[k] = f[cols].to_numpy(dtype=float)[m]
    return out


# ------------------------------------------------------------------ histogram bootstrap machinery
def hist_by_date(v, d, ndates):
    b = np.clip(np.searchsorted(EDGES, v, side="right") - 1, 0, HNB - 1)
    H = np.zeros((ndates, HNB), dtype=np.float32)
    np.add.at(H, (d, b), 1.0)
    return H


def _q_from_cum(cum, tot, q):
    """Linear-interpolated quantile from a cumulative count vector."""
    target = q * tot
    j = np.searchsorted(cum, target, side="left")
    j = min(int(j), HNB - 1)
    c0 = cum[j - 1] if j > 0 else 0.0
    n = cum[j] - c0
    frac = 0.0 if n <= 0 else (target - c0) / n
    return EDGES[j] + frac * (EDGES[j + 1] - EDGES[j])


def _cdf_at(cum, tot, x):
    """P(z < x) from cumulative counts (linear inside the bin)."""
    if x <= HLO:
        return 0.0
    if x >= HHI:
        return 1.0
    j = int(np.searchsorted(EDGES, x, side="right") - 1)
    j = min(max(j, 0), HNB - 1)
    c0 = cum[j - 1] if j > 0 else 0.0
    frac = (x - EDGES[j]) / (EDGES[j + 1] - EDGES[j])
    return float((c0 + frac * (cum[j] - c0)) / tot)


def stats_from_counts(c, ks=KA):
    cum = np.cumsum(c)
    tot = cum[-1]
    if tot < 100:
        return None
    med = _q_from_cum(cum, tot, 0.5)
    q10 = _q_from_cum(cum, tot, 0.10)
    q90 = _q_from_cum(cum, tot, 0.90)
    q25 = _q_from_cum(cum, tot, 0.25)
    q75 = _q_from_cum(cum, tot, 0.75)
    out = {"med": med, "bowley": (q90 + q10 - 2 * med) / (q90 - q10),
           "iqs": (q75 - q25) / 1.34898, "n": float(tot)}
    for k in ks:
        lo = _cdf_at(cum, tot, med - k)
        hi = 1.0 - _cdf_at(cum, tot, med + k)
        out[f"P_lo_k{k}"] = lo
        out[f"P_hi_k{k}"] = hi
        out[f"ratio_k{k}"] = lo / hi if hi > 0 else np.nan
    return out


def block_weights(ndates, nrep, block, rng):
    """Moving-block bootstrap weight matrix (nrep x ndates): how often each date is drawn."""
    nb = int(np.ceil(ndates / block))
    starts = rng.integers(0, max(ndates - block + 1, 1), size=(nrep, nb))
    offs = np.arange(block)
    picks = (starts[:, :, None] + offs[None, None, :]).reshape(nrep, -1)
    picks = np.clip(picks, 0, ndates - 1)
    W = np.zeros((nrep, ndates), dtype=np.float32)
    rows = np.repeat(np.arange(nrep), picks.shape[1])
    np.add.at(W, (rows, picks.ravel()), 1.0)
    return W


def boot_stats(H, nrep=NREP, block=BLOCK, seed=SEED, ks=KA):
    """Point estimate + block-bootstrap distribution of every statistic in stats_from_counts."""
    rng = np.random.default_rng(seed)
    ndates = H.shape[0]
    point = stats_from_counts(H.sum(axis=0), ks)
    W = block_weights(ndates, nrep, block, rng)
    C = W @ H                                     # (nrep, HNB) aggregated counts per replicate
    reps = [stats_from_counts(C[i], ks) for i in range(nrep)]
    reps = [r for r in reps if r is not None]
    out = {}
    for key in point:
        a = np.array([r[key] for r in reps], dtype=float)
        a = a[np.isfinite(a)]
        out[key] = dict(point=point[key], lo=np.percentile(a, 2.5), hi=np.percentile(a, 97.5),
                        p_gt1=float((a > 1).mean()), p_lt0=float((a < 0).mean()), nrep=a.size)
    return out


def ratio_table(label, H, ks=KA, block=BLOCK, nrep=NREP, tag=""):
    b = boot_stats(H, nrep=nrep, block=block, ks=ks)
    rows = []
    for k in ks:
        r = b[f"ratio_k{k}"]
        rows.append(dict(k=k, ratio=r["point"], lo=r["lo"], hi=r["hi"], P_gt1=r["p_gt1"],
                         P_lo=b[f"P_lo_k{k}"]["point"], P_hi=b[f"P_hi_k{k}"]["point"]))
    bw = b["bowley"]
    print(f"  [{label}{tag}] n_obs={int(b['n']['point']):,} n_dates={H.shape[0]:,} "
          f"block={block} med={b['med']['point']:+.4f}  "
          f"bowley={bw['point']:+.4f} [{bw['lo']:+.4f},{bw['hi']:+.4f}] P(bowley<0)={bw['p_lt0']:.3f}")
    print(pd.DataFrame(rows).to_string(index=False))
    return b


# ------------------------------------------------------------------ per-symbol helpers
def iqs_arr(a):
    return (np.percentile(a, 75) - np.percentile(a, 25)) / 1.34898


def bowley_arr(a):
    q10, q50, q90 = np.percentile(a, [10, 50, 90])
    return (q90 + q10 - 2 * q50) / (q90 - q10)


def per_symbol_stats(frame, cols, h, min_n=500, ks=(1.0, 1.5, 2.0)):
    S = prep()
    sig = S["sig"]
    rows = []
    for c in cols:
        a = frame[h][c].to_numpy(dtype=float)
        a = a[np.isfinite(a)]
        if a.size < min_n:
            continue
        med = np.median(a)
        d = dict(sym=c, n=a.size, med=med, iqs=iqs_arr(a), bowley=bowley_arr(a),
                 med_sig=float(np.nanmedian(sig[c].to_numpy(dtype=float))))
        for k in ks:
            lo = float((a < med - k).mean())
            hi = float((a > med + k).mean())
            d[f"lo_k{k}"] = lo
            d[f"hi_k{k}"] = hi
            d[f"ratio_k{k}"] = lo / hi if hi > 0 else np.nan
        rows.append(d)
    return pd.DataFrame(rows)


# ==================================================================== SECTION 0
def sec0():
    S = prep()
    print("=" * 130)
    print("SECTION 0 — REPRODUCE the claimed pooled numbers (z = simple fwd return / ewma(.94)*sqrt(h))")
    print("   statistic: median-centred P(z<med-k)/P(z>med+k); 63-day moving-block date bootstrap, "
          f"{NREP} reps, seed {SEED}")
    for label, cols in (("INDEX(5)", S["idx"]), ("SINGLES", S["singles"])):
        for h in HORIZONS:
            d = long(S["Z"][h], cols)
            H = hist_by_date(d["v"], d["d"], len(S["dates"]))
            H = H[H.sum(axis=1) > 0]
            print(f"\n h={h}")
            ratio_table(label, H)


# ==================================================================== SECTION 1
def sec1():
    S = prep()
    print("=" * 130)
    print("SECTION 1 — how many INDEPENDENT observations is the INDEX ratio actually built on?")
    print("   raw tail COUNTS and the number of distinct calendar EPISODES (tail dates grouped when")
    print("   within h+1 trading days of each other). 5 index series are ~1.0 correlated => count DATES.")
    for h in HORIZONS:
        for label, cols in (("INDEX(5)", S["idx"]), ("SPY", ["SPY"])):
            z = S["Z"][h][cols]
            a = z.to_numpy(dtype=float)
            med = np.nanmedian(a)
            rows = []
            for k in KA:
                loM = (z - med) < -k
                hiM = (z - med) > k
                lo_d = np.where(loM.any(axis=1).to_numpy())[0]
                hi_d = np.where(hiM.any(axis=1).to_numpy())[0]
                ep = lambda arr: int((np.diff(arr) > h + 1).sum() + 1) if arr.size else 0
                rows.append(dict(k=k, lo_obs=int(np.nansum(loM.to_numpy())),
                                 hi_obs=int(np.nansum(hiM.to_numpy())),
                                 lo_dates=lo_d.size, hi_dates=hi_d.size,
                                 lo_episodes=ep(lo_d), hi_episodes=ep(hi_d),
                                 ratio_obs=np.nansum(loM.to_numpy()) / max(np.nansum(hiM.to_numpy()), 1)))
            print(f"\n h={h} [{label}] median={med:+.4f}")
            print(pd.DataFrame(rows).to_string(index=False))
            if h == 21 and label == "SPY":
                yrs = pd.Series(S["dates"][np.where(((z - med) < -2.0).any(axis=1).to_numpy())[0]]).dt.year
                print(f"   SPY h=21 k=2 DOWN-tail dates by year: {dict(yrs.value_counts().sort_index())}")
                yrs2 = pd.Series(S["dates"][np.where(((z - med) > 2.0).any(axis=1).to_numpy())[0]]).dt.year
                print(f"   SPY h=21 k=2   UP-tail dates by year: {dict(yrs2.value_counts().sort_index())}")


# ==================================================================== SECTION 2
def sec2():
    S = prep()
    print("=" * 130)
    print("SECTION 2 — UNITS ARTEFACT.  z_simple = (exp(x)-1)/s  where x is the h-day LOG return and")
    print("   s = sigma_hat_h. Since exp(x)-1 ~ x + x^2/2, z_simple ~ z_log + (s/2) z_log^2: a")
    print("   POSITIVE skew of order s is injected, and s(singles) ~ 2x s(index).")
    print("   Control: z_log = log(1+r_h)/sigma_hat_h  (same denominator, log numerator).")

    print("\n(a) mean/median sigma_hat_h per group (the size of the injected term)")
    rows = []
    for label, cols in (("INDEX(5)", S["idx"]), ("SINGLES", S["singles"])):
        for h in HORIZONS:
            a = S["SH"][h][cols].to_numpy(dtype=float)
            a = a[np.isfinite(a)]
            rows.append(dict(group=label, h=h, med_sigma_h=np.median(a), mean_sigma_h=a.mean(),
                             predicted_bowley_shift=0.64 * np.median(a)))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n(b) simulation check of the analytic prediction: x ~ t(nu=4, unit var) * s, compare")
    print("    Bowley of x/s (log space, exactly 0 by symmetry) with Bowley of (exp(x)-1)/s")
    rng = np.random.default_rng(11)
    nu = 4.0
    tt = rng.standard_t(nu, size=2_000_000) / math.sqrt(nu / (nu - 2))
    for s in (0.011, 0.025, 0.05, 0.11, 0.23):
        x = tt * s
        zl = x / s
        zs = (np.exp(x) - 1.0) / s
        print(f"    s={s:.3f}  bowley(z_log)={bowley_arr(zl):+.4f}  bowley(z_simple)={bowley_arr(zs):+.4f}"
              f"  shift={bowley_arr(zs) - bowley_arr(zl):+.4f}  (0.64*s={0.64*s:+.4f})")

    print("\n(c) the claim's statistics recomputed in LOG space")
    for label, cols in (("INDEX(5)", S["idx"]), ("SINGLES", S["singles"])):
        for h in HORIZONS:
            d = long(S["ZL"][h], cols)
            H = hist_by_date(d["v"], d["d"], len(S["dates"]))
            H = H[H.sum(axis=1) > 0]
            print(f"\n h={h}")
            ratio_table(label, H, tag=" z_LOG")


# ==================================================================== SECTION 3
def sec3():
    S = prep()
    print("=" * 130)
    print("SECTION 3 — POOLING ARTEFACT. The pooled singles median is ONE number for 278 names with")
    print("   different drift/vol ratios. A right-skewed cross-section of locations manufactures")
    print("   positive skew. Fix: centre (and scale) EACH SYMBOL by its own median (and IQR).")
    for kind, lab in (("Z", "z_simple"), ("ZL", "z_log")):
        print(f"\n>>> {lab}")
        for h in HORIZONS:
            ps = per_symbol_stats(S[kind], S["singles"], h)
            pi = per_symbol_stats(S[kind], S["idx"], h)
            # per-symbol demeaned pool
            pooled = []
            dts = []
            for c in S["singles"]:
                a = S[kind][h][c]
                v = a.to_numpy(dtype=float)
                m = np.isfinite(v)
                if m.sum() < 500:
                    continue
                vv = v[m]
                pooled.append(vv - np.median(vv))
                dts.append(np.where(m)[0])
            pv, pd_ = np.concatenate(pooled), np.concatenate(dts)
            H = hist_by_date(pv, pd_, len(S["dates"]))
            H = H[H.sum(axis=1) > 0]
            print(f"\n h={h}  per-symbol median of per-symbol Bowley: singles={ps.bowley.median():+.4f} "
                  f"(mean {ps.bowley.mean():+.4f}, {(ps.bowley<0).mean()*100:.0f}% negative, n_sym={len(ps)}) "
                  f"| index={pi.bowley.median():+.4f} ({(pi.bowley<0).mean()*100:.0f}% neg, n={len(pi)})")
            print(f"    per-symbol median ratio k=2: singles={ps['ratio_k2.0'].median():.3f} "
                  f"({(ps['ratio_k2.0']>1).mean()*100:.0f}% >1) | index={pi['ratio_k2.0'].median():.3f}")
            ratio_table("SINGLES per-sym-centred", H, tag=f" {lab}")
            # symbol-cluster bootstrap on the per-symbol Bowley median
            rng = np.random.default_rng(SEED)
            bs = ps.bowley.to_numpy()
            reps = np.array([np.median(rng.choice(bs, bs.size, replace=True)) for _ in range(2000)])
            print(f"    symbol-cluster bootstrap of median per-symbol bowley: "
                  f"{np.median(bs):+.4f} [{np.percentile(reps,2.5):+.4f},{np.percentile(reps,97.5):+.4f}] "
                  f"P(<0)={float((reps<0).mean()):.3f}")


# ==================================================================== SECTION 4
def sec4():
    S = prep()
    print("=" * 130)
    print("SECTION 4 — REGIME MIXTURE. One pooled median mixes vol regimes; drift/sigma is larger in")
    print("   calm regimes, so the location differs by regime and the mixture itself is skewed.")
    print("   Fix: centre WITHIN vol-rank quintile (expanding-rank, no lookahead), then pool.")
    for kind, lab in (("Z", "z_simple"), ("ZL", "z_log")):
        print(f"\n>>> {lab}")
        for label, cols in (("INDEX(5)", S["idx"]), ("SINGLES", S["singles"])):
            for h in HORIZONS:
                d = long(S[kind][h], cols, extra={"rk": S["rank"]})
                g = np.isfinite(d["rk"])
                v, dd, rk = d["v"][g], d["d"][g], d["rk"][g]
                qb = np.digitize(rk, [0.2, 0.4, 0.6, 0.8])
                out = []
                cent = np.empty_like(v)
                for b in range(5):
                    m = qb == b
                    if m.sum() < 200:
                        cent[m] = np.nan
                        continue
                    med = np.median(v[m])
                    cent[m] = v[m] - med
                    a = v[m]
                    lo = float((a < med - 2).mean())
                    hi = float((a > med + 2).mean())
                    out.append(dict(volq=b + 1, n=int(m.sum()), med=med, bowley=bowley_arr(a),
                                    ratio_k2=lo / hi if hi > 0 else np.nan))
                gg = np.isfinite(cent)
                H = hist_by_date(cent[gg], dd[gg], len(S["dates"]))
                H = H[H.sum(axis=1) > 0]
                print(f"\n h={h} [{label}] within-quintile stats")
                print(pd.DataFrame(out).to_string(index=False))
                ratio_table(f"{label} quintile-centred", H, tag=f" {lab}")


# ==================================================================== SECTION 5
def sec5():
    S = prep()
    print("=" * 130)
    print("SECTION 5 — ASSET CLASS or just VOL LEVEL?  Per-symbol Bowley skew vs log10(median daily")
    print("   sigma), fit on SINGLES ONLY, then ask where the index series sit relative to that line.")
    for kind, lab in (("Z", "z_simple"), ("ZL", "z_log")):
        print(f"\n>>> {lab}")
        for h in HORIZONS:
            ps = per_symbol_stats(S[kind], S["singles"], h)
            pi = per_symbol_stats(S[kind], S["idx"], h)
            x = np.log10(ps.med_sig.to_numpy())
            y = ps.bowley.to_numpy()
            A = np.vstack([np.ones_like(x), x]).T
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            resid = y - A @ coef
            sd = resid.std(ddof=2)
            xi = np.log10(pi.med_sig.to_numpy())
            yi = pi.bowley.to_numpy()
            pred = coef[0] + coef[1] * xi
            print(f"\n h={h} singles fit: bowley = {coef[0]:+.4f} {coef[1]:+.4f}*log10(sigma), "
                  f"resid_sd={sd:.4f}, corr(x,y)={np.corrcoef(x,y)[0,1]:+.3f}, n_sym={len(ps)}")
            print(pd.DataFrame(dict(sym=pi.sym, med_sig=pi.med_sig, bowley=yi, singles_line_pred=pred,
                                    resid_in_sd=(yi - pred) / sd)).to_string(index=False))
            # vol-matched cohort: the singles whose sigma overlaps the index range
            hi_s = pi.med_sig.max()
            lo_s = pi.med_sig.min()
            m = (ps.med_sig <= hi_s * 1.15)
            print(f"    vol-matched cohort (median sigma <= {hi_s*1.15:.4f}; index range "
                  f"{lo_s:.4f}-{hi_s:.4f}): n_sym={int(m.sum())} "
                  f"median bowley={ps.bowley[m].median():+.4f} vs index {pi.bowley.median():+.4f} "
                  f"vs all singles {ps.bowley.median():+.4f}")
            if m.sum() >= 10:
                lowest = ps.nsmallest(20, "med_sig")
                print(f"    20 LOWEST-vol singles: median sigma {lowest.med_sig.median():.4f} "
                      f"median bowley {lowest.bowley.median():+.4f}")
                highest = ps.nlargest(20, "med_sig")
                print(f"    20 HIGHEST-vol singles: median sigma {highest.med_sig.median():.4f} "
                      f"median bowley {highest.bowley.median():+.4f}")


# ==================================================================== SECTION 6
def sec6():
    S = prep()
    dates = S["dates"]
    print("=" * 130)
    print("SECTION 6 — FRAGILITY: crisis years, single index members, non-overlapping windows,")
    print("   longer bootstrap blocks.")
    crisis = pd.Series(dates).dt.year.isin([2008, 2009, 2020]).to_numpy()

    for kind, lab in (("Z", "z_simple"), ("ZL", "z_log")):
        print(f"\n>>> {lab}")
        for label, cols in (("INDEX(5)", S["idx"]), ("SPY", ["SPY"]), ("^GSPC", ["^GSPC"]),
                            ("SINGLES", S["singles"])):
            for h in HORIZONS:
                d = long(S[kind][h], cols)
                H = hist_by_date(d["v"], d["d"], len(dates))
                keepall = H.sum(axis=1) > 0
                print(f"\n h={h} [{label}]")
                ratio_table(label, H[keepall], ks=(2.0,), tag=" ALL")
                nc = keepall & ~crisis
                ratio_table(label, H[nc], ks=(2.0,), tag=" no-2008/09/2020")
                for blk in (252, 504):
                    ratio_table(label, H[keepall], ks=(2.0,), block=blk, tag=f" block{blk}")
                if h > 1:
                    for ph in (0,):
                        sel = np.zeros(len(dates), dtype=bool)
                        sel[ph::h] = True
                        ratio_table(label, H[keepall & sel], ks=(2.0,), tag=f" stride{h}")


# ==================================================================== SECTION 7 (OOS)
def _norm_cdf(x):
    from math import erf
    x = np.asarray(x, float)
    return 0.5 * (1.0 + np.vectorize(erf)(x / math.sqrt(2.0)))


class Emp:
    """Empirical CDF of a standardized sample with a Gaussian-ish tail continuation."""

    def __init__(self, u):
        self.a = np.sort(u[np.isfinite(u)])
        self.n = self.a.size
        self.med = float(np.median(self.a))
        self.sc = float(iqs_arr(self.a))

    def cdf(self, k):
        k = np.asarray(k, float)
        r = np.searchsorted(self.a, k, side="left").astype(float)
        p = (r + 0.5) / (self.n + 1.0)
        frac = 0.5 / (self.n + 1.0)
        out = np.clip(p, frac, 1.0 - frac)
        lo, hi = self.a[0], self.a[-1]
        fl, fh = k < lo, k > hi
        if fl.any():
            out[fl] = np.minimum(_norm_cdf((k[fl] - self.med) / self.sc) * frac / max(
                _norm_cdf((lo - self.med) / self.sc), 1e-12), frac)
        if fh.any():
            out[fh] = np.maximum(1.0 - (1.0 - _norm_cdf((k[fh] - self.med) / self.sc)) * frac / max(
                1.0 - _norm_cdf((hi - self.med) / self.sc), 1e-12), 1.0 - frac)
        return out


def probs4(cdf_fn, sh, thr=THR, logspace=False):
    if logspace:
        k_dn, k_up = np.log1p(-thr) / sh, np.log1p(thr) / sh
    else:
        k_dn, k_up = -thr / sh, thr / sh
    F_dn, F_0, F_up = cdf_fn(k_dn), cdf_fn(np.zeros_like(sh)), cdf_fn(k_up)
    P = np.column_stack([F_dn, F_0 - F_dn, F_up - F_0, 1.0 - F_up])
    P = np.clip(P, 1e-6, None)
    return P / P.sum(axis=1, keepdims=True)


def stack(kind, h, cols):
    S = prep()
    z = S[kind][h][cols]
    m = z.notna().to_numpy()
    yr = np.repeat(z.index.year.to_numpy()[:, None], len(cols), axis=1)[m]
    di = np.repeat(np.arange(len(z))[:, None], len(cols), axis=1)[m]
    out = dict(z=z.to_numpy(dtype=float)[m], sh=S["SH"][h][cols].to_numpy(dtype=float)[m],
               fwd=S["FWD"][h][cols].to_numpy(dtype=float)[m],
               rk=S["rank"][cols].to_numpy(dtype=float)[m], year=yr, d=di)
    g = np.isfinite(out["z"]) & np.isfinite(out["sh"]) & np.isfinite(out["fwd"]) & np.isfinite(out["rk"])
    return {k: v[g] for k, v in out.items()}


def actual_idx(fwd, thr=THR):
    b = L.bucketize(pd.Series(fwd), thr).to_numpy()
    order = {v: i for i, v in enumerate(L.BUCKETS)}
    return np.array([order[x] for x in b], dtype=int)


def oos_models(label, cols, h, purge=True, kinds=("emp", "sym", "logsym", "clim")):
    """Walk-forward OOS per-observation log loss for: empirical z table, its symmetrised twin,
    a log-space-symmetric table, and climatology.  Returns a tidy DataFrame."""
    S = prep()
    s = stack("Z", h, cols)
    sl = stack("ZL", h, cols)
    assert s["z"].size == sl["z"].size
    ai = actual_idx(s["fwd"])
    years = sorted(set(s["year"].tolist()))
    recs = []
    for Y in years:
        if Y - years[0] < 5:
            continue
        tr = s["year"] < Y
        te = s["year"] == Y
        if purge:
            # rows whose forward window spills into year Y leak the test labels
            first_te = s["d"][te].min() if te.any() else 0
            tr = tr & (s["d"] <= first_te - h - 1)
        if tr.sum() < 2000 or te.sum() < 100:
            continue
        ztr, shte = s["z"][tr], s["sh"][te]
        loc, sc = float(np.median(ztr)), iqs_arr(ztr)
        u = (ztr - loc) / sc
        G = Emp(u)
        Gs = Emp(np.concatenate([u, -u]))                      # symmetrised shape, same loc/scale
        zltr = sl["z"][tr]
        locl, scl = float(np.median(zltr)), iqs_arr(zltr)
        ul = (zltr - locl) / scl
        Gl = Emp(np.concatenate([ul, -ul]))                     # log-space-symmetric shape
        P = {}
        P["emp"] = probs4(lambda k: G.cdf((k - loc) / sc), shte)
        P["sym"] = probs4(lambda k: Gs.cdf((k - loc) / sc), shte)
        P["logsym"] = probs4(lambda k: Gl.cdf((k - locl) / scl), sl["sh"][te], logspace=True)
        clim = L.climatology(ai[tr])
        P["clim"] = np.tile(clim, (int(te.sum()), 1))
        row = dict(year=Y, n=int(te.sum()))
        for m in kinds:
            p = np.clip(P[m][np.arange(int(te.sum())), ai[te]], 1e-12, 1)
            row[m] = float(-np.log(p).mean())
            recs_ll = -np.log(p)
            row["_ll_" + m] = recs_ll
        row["_d"] = s["d"][te]
        row["_rk"] = s["rk"][te]
        row["_ai"] = ai[te]
        row["_P"] = {m: P[m] for m in kinds}
        recs.append(row)
    return recs


def paired_report(label, recs, a="emp", b="sym"):
    """Paired cost of switching a->b with three different uncertainty models."""
    if not recs:
        print(f"  [{label}] no test years")
        return
    per_ob = np.concatenate([r["_ll_" + b] - r["_ll_" + a] for r in recs])
    n = per_ob.size
    t_naive = per_ob.mean() / (per_ob.std(ddof=1) / math.sqrt(n))
    # cluster by test YEAR
    ym = np.array([r["_ll_" + b].mean() - r["_ll_" + a].mean() for r in recs])
    t_year = ym.mean() / (ym.std(ddof=1) / math.sqrt(ym.size)) if ym.size > 2 else np.nan
    # date-block bootstrap over the pooled OOS rows (block = 63 dates)
    d = np.concatenate([r["_d"] for r in recs])
    order = np.argsort(d, kind="stable")
    d_s, v_s = d[order], per_ob[order]
    ud, start = np.unique(d_s, return_index=True)
    sums = np.add.reduceat(v_s, start)
    cnts = np.diff(np.append(start, len(v_s)))
    rng = np.random.default_rng(SEED)
    nd = ud.size
    nb = int(np.ceil(nd / BLOCK))
    reps = np.empty(1000)
    for i in range(1000):
        st = rng.integers(0, max(nd - BLOCK + 1, 1), size=nb)
        pick = np.clip((st[:, None] + np.arange(BLOCK)[None, :]).ravel(), 0, nd - 1)
        reps[i] = sums[pick].sum() / cnts[pick].sum()
    print(f"  [{label}] mean d(logloss) {b}-{a} = {per_ob.mean():+.5f}  n_obs={n:,} "
          f"n_years={len(recs)} n_dates={nd:,}")
    print(f"      naive t={t_naive:+.2f} | year-clustered t={t_year:+.2f} (n={ym.size}) | "
          f"date-block 95% CI [{np.percentile(reps,2.5):+.5f},{np.percentile(reps,97.5):+.5f}] "
          f"P(>0)={float((reps>0).mean()):.3f}")
    yr_tab = pd.DataFrame(dict(year=[r["year"] for r in recs], n=[r["n"] for r in recs],
                              d=ym))
    bad = int((yr_tab.d > 0).sum())
    print(f"      years where symmetrising HURTS: {bad}/{len(yr_tab)}   "
          f"worst {yr_tab.loc[yr_tab.d.idxmax(),'year']} {yr_tab.d.max():+.4f}  "
          f"best {yr_tab.loc[yr_tab.d.idxmin(),'year']} {yr_tab.d.min():+.4f}")


def sec7():
    S = prep()
    print("=" * 130)
    print("SECTION 7 — OOS. Does symmetrising the shape table really cost anything, once the paired")
    print("   test is clustered by DATE / by TEST YEAR instead of by (overlapping) observation, and")
    print("   once train rows whose h-day window spills into the test year are PURGED?")
    print("   Also: log-space-symmetric table as a third contender, and vs CLIMATOLOGY.")
    for label, cols in (("INDEX(5)", S["idx"]), ("SPY", ["SPY"]), ("SINGLES", S["singles"])):
        for h in HORIZONS:
            recs = oos_models(label, cols, h, purge=True)
            if not recs:
                continue
            mm = {m: np.average([r[m] for r in recs], weights=[r["n"] for r in recs])
                  for m in ("emp", "sym", "logsym", "clim")}
            print(f"\n h={h} [{label}] obs-weighted OOS log loss: emp={mm['emp']:.4f} "
                  f"sym={mm['sym']:.4f} logsym={mm['logsym']:.4f} clim={mm['clim']:.4f}  "
                  f"| skill vs clim: emp={1-mm['emp']/mm['clim']:+.4f} sym={1-mm['sym']/mm['clim']:+.4f} "
                  f"logsym={1-mm['logsym']/mm['clim']:+.4f}")
            paired_report(f"{label} h={h} sym-emp", recs, "emp", "sym")
            paired_report(f"{label} h={h} logsym-emp", recs, "emp", "logsym")


# ==================================================================== SECTION 8 (Simpson)
def sec8():
    S = prep()
    print("=" * 130)
    print("SECTION 8 — SIMPSON CHECK: OOS calibration of down_big / up_big for emp vs sym vs logsym,")
    print("   overall, within vol-rank quintiles and within calendar years. ECE of CLIMATOLOGY shown")
    print("   as the honest reference.")
    for label, cols in (("INDEX(5)", S["idx"]), ("SINGLES", S["singles"])):
        for h in (1, 21):
            recs = oos_models(label, cols, h, purge=True)
            if not recs:
                continue
            ai = np.concatenate([r["_ai"] for r in recs])
            rk = np.concatenate([r["_rk"] for r in recs])
            yr = np.concatenate([np.full(r["n"], r["year"]) for r in recs])
            P = {m: np.vstack([r["_P"][m] for r in recs]) for m in ("emp", "sym", "logsym", "clim")}
            print(f"\n h={h} [{label}] n_oos={ai.size:,}")
            for bi, bname in ((0, "down_big"), (3, "up_big")):
                hit = (ai == bi).astype(float)
                line = [f"  {bname}: obs_freq={hit.mean():.4f}"]
                for m in ("emp", "sym", "logsym", "clim"):
                    line.append(f"{m}: mean_p={P[m][:,bi].mean():.4f} ECE={L.ece(P[m][:,bi], hit):.4f}")
                print("   " + " | ".join(line))
                qb = np.digitize(rk, [0.2, 0.4, 0.6, 0.8])
                rows = []
                for b in range(5):
                    m_ = qb == b
                    if m_.sum() < 500:
                        continue
                    r = dict(volq=b + 1, n=int(m_.sum()), obs=hit[m_].mean())
                    for m in ("emp", "sym", "logsym", "clim"):
                        r[m + "_p"] = P[m][m_, bi].mean()
                        r[m + "_gap"] = P[m][m_, bi].mean() - hit[m_].mean()
                    rows.append(r)
                print(f"    {bname} by vol quintile")
                print(pd.DataFrame(rows).to_string(index=False))
                rows = []
                for y in sorted(set(yr.tolist())):
                    m_ = yr == y
                    r = dict(year=y, n=int(m_.sum()), obs=hit[m_].mean())
                    for m in ("emp", "sym", "logsym"):
                        r[m + "_gap"] = P[m][m_, bi].mean() - hit[m_].mean()
                    rows.append(r)
                t = pd.DataFrame(rows)
                print(f"    {bname} by year: mean|gap| emp={t.emp_gap.abs().mean():.4f} "
                      f"sym={t.sym_gap.abs().mean():.4f} logsym={t.logsym_gap.abs().mean():.4f}  "
                      f"(worst emp year {t.loc[t.emp_gap.abs().idxmax(),'year']} "
                      f"{t.emp_gap.abs().max():.4f})")


# ==================================================================== SECTION 9
def sec9():
    """Is 'P(down2%) != P(up2%) for singles is PURELY drift' true?  Decompose the up-minus-down
    gap the shipped recipe produces into drift, shape-skew and threshold-convexity pieces.
    Plus: cross-sectional vol control (point-in-time), which the time-series vol rank misses."""
    S = prep()
    print("=" * 130)
    print("SECTION 9a — DECOMPOSITION of the up_big-minus-down_big gap at thr=2% (IN-SAMPLE, whole")
    print("   panel; this is an accounting identity check, not a forecast).  For each obs, P(up)=1-G(")
    print("   thr/s), P(dn)=G(-thr/s) with G = pooled empirical z-cdf, then counterfactuals:")
    print("     emp     : G as observed                      (drift + skew + convexity all in)")
    print("     sym     : G median-centred then symmetrised  (skew removed, drift kept)")
    print("     nodrift : G shifted so median = 0            (drift removed, skew kept)")
    print("     both    : symmetrised AND median 0           (only the 2%-threshold convexity left)")
    for label, cols in (("INDEX(5)", S["idx"]), ("SINGLES", S["singles"])):
        rows = []
        for h in HORIZONS:
            z = S["Z"][h][cols].to_numpy(dtype=float)
            sh = S["SH"][h][cols].to_numpy(dtype=float)
            fwd = S["FWD"][h][cols].to_numpy(dtype=float)
            m = np.isfinite(z) & np.isfinite(sh) & np.isfinite(fwd)
            z, sh, fwd = z[m], sh[m], fwd[m]
            med = float(np.median(z))
            u = z - med
            variants = {"emp": np.sort(z), "sym": np.sort(np.concatenate([u, -u]) + med),
                        "nodrift": np.sort(u), "both": np.sort(np.concatenate([u, -u]))}
            r = dict(h=h, n=z.size, obs_dn=float((fwd <= -THR).mean()),
                     obs_up=float((fwd >= THR).mean()))
            r["obs_gap"] = r["obs_up"] - r["obs_dn"]
            for k, a in variants.items():
                n = a.size
                p_dn = np.searchsorted(a, -THR / sh, side="left") / n
                p_up = 1.0 - np.searchsorted(a, THR / sh, side="right") / n
                r[k + "_dn"] = float(p_dn.mean())
                r[k + "_up"] = float(p_up.mean())
                r[k + "_gap"] = float((p_up - p_dn).mean())
            rows.append(r)
        t = pd.DataFrame(rows)
        print(f"\n[{label}]")
        print(t[["h", "n", "obs_dn", "obs_up", "obs_gap", "emp_dn", "emp_up", "emp_gap",
                 "sym_gap", "nodrift_gap", "both_gap"]].to_string(index=False))
        print("   share of emp_gap explained by: drift = (emp_gap - nodrift_gap)/emp_gap ; "
              "skew = (emp_gap - sym_gap)/emp_gap ; residual convexity = both_gap/emp_gap")
        for _, r in t.iterrows():
            g = r.emp_gap
            print(f"    h={int(r.h):2d}  drift={100*(g-r.nodrift_gap)/g:6.1f}%  "
                  f"skew={100*(g-r.sym_gap)/g:6.1f}%  threshold-convexity={100*r.both_gap/g:6.1f}%")

    print("\n" + "=" * 130)
    print("SECTION 9b — CROSS-SECTIONAL vol control (point-in-time: on each date rank the singles by")
    print("   sigma_hat across symbols).  If 'singles are symmetric' is a cross-sectional average of")
    print("   left-skewed low-vol names and right-skewed high-vol names, one shape table is wrong for")
    print("   both. z_simple = the coordinate the model actually ships in.")
    sig = S["sig"][S["singles"]]
    xrank = sig.rank(axis=1, pct=True)
    for kind, lab in (("Z", "z_simple"), ("ZL", "z_log")):
        print(f"\n>>> {lab}")
        for h in HORIZONS:
            d = long(S[kind][h], S["singles"], extra={"xr": xrank})
            g = np.isfinite(d["xr"])
            v, dd, xr = d["v"][g], d["d"][g], d["xr"][g]
            qb = np.digitize(xr, [1 / 3, 2 / 3])
            print(f"\n h={h}")
            for b in range(3):
                m = qb == b
                cent = v[m] - np.median(v[m])
                H = hist_by_date(cent, dd[m], len(S["dates"]))
                H = H[H.sum(axis=1) > 0]
                ratio_table(f"SINGLES xsec-vol-tercile{b+1}", H, ks=(1.0, 1.5, 2.0), tag=f" {lab}")


# ==================================================================== SECTION 10
def sec10():
    """Is the INDEX z asymmetry a RETURN-shape fact, or a VOL-FORECAST-ERROR fact?  EWMA sigma_hat
    is stale, and for indices vol spikes coincide with drops (leverage), so a crash lands while the
    denominator is still small.  Standardise instead by the CONTEMPORANEOUS realized vol of the same
    window -- lookahead BY CONSTRUCTION, diagnostic only -- and see how much asymmetry survives."""
    S = prep()
    close = S["close"]
    print("=" * 130)
    print("SECTION 10 — z standardised by CONTEMPORANEOUS realized vol (r[t+1..t+h]) instead of the")
    print("   lagging EWMA forecast.  *** USES FUTURE DATA ON PURPOSE — DIAGNOSTIC ONLY ***")
    print("   If the index asymmetry collapses, the 'shape' table is compensating for a vol-forecast")
    print("   deficiency (leverage effect), not describing a return shape.")
    for h in (5, 10, 21):
        rv = L.realized_vol_forward(close, h) * math.sqrt(h)
        fwd = L.forward_simple_return(close, h)
        for label, cols in (("INDEX(5)", S["idx"]), ("SINGLES", S["singles"])):
            base = S["Z"][h][cols].notna()
            zs = (fwd[cols] / rv[cols]).where(base)
            zl = (np.log1p(fwd[cols]) / rv[cols]).where(base)
            for nm, fr in (("simple", zs), ("log", zl)):
                d = long(fr, cols)
                H = hist_by_date(d["v"], d["d"], len(S["dates"]))
                H = H[H.sum(axis=1) > 0]
                print(f"\n h={h} [{label}] z_{nm} / REALIZED vol")
                ratio_table(label, H, ks=(1.0, 1.5, 2.0), tag=f" realized-{nm}")


# ==================================================================== SECTION 11
def sec11():
    """CONSTRUCTIVE test.  If the singles' simple-space symmetry is really the convexity of
    exp(x)-1 cancelling a genuine LOG-space left skew, then the sigma-dependence of the shape is an
    artefact of the coordinate, and a single LOG-space (asymmetric) table with the asymmetric
    thresholds log(1+/-thr) should beat the shipped simple-space table -- especially in the
    cross-sectional vol tails, where the artefact is biggest.  Walk-forward, purged, year-clustered.
    """
    S = prep()
    sig = S["sig"][S["singles"]]
    xrank = sig.rank(axis=1, pct=True)
    print("=" * 130)
    print("SECTION 11 — OOS: simple-space empirical table (SHIPPED) vs LOG-space empirical table,")
    print("   overall and within point-in-time CROSS-SECTIONAL vol terciles.  Params fit on years")
    print("   strictly before Y, train rows purged of h-day overlap into Y.")
    for label, cols in (("SINGLES", S["singles"]), ("INDEX(5)", S["idx"])):
        for h in HORIZONS:
            s = stack("Z", h, cols)
            sl = stack("ZL", h, cols)
            ai = actual_idx(s["fwd"])
            if label == "SINGLES":
                xr = long(S["Z"][h], cols, extra={"xr": xrank})["xr"]
            else:
                xr = np.full(s["z"].size, 0.5)
            years = sorted(set(s["year"].tolist()))
            per_year, LL = [], {"simple": [], "log": [], "clim": []}
            keep_ai, keep_xr, keep_P = [], [], {"simple": [], "log": []}
            for Y in years:
                if Y - years[0] < 5:
                    continue
                tr, te = s["year"] < Y, s["year"] == Y
                first_te = s["d"][te].min() if te.any() else 0
                tr = tr & (s["d"] <= first_te - h - 1)
                if tr.sum() < 2000 or te.sum() < 100:
                    continue
                loc, sc = float(np.median(s["z"][tr])), iqs_arr(s["z"][tr])
                G = Emp((s["z"][tr] - loc) / sc)
                locl, scl = float(np.median(sl["z"][tr])), iqs_arr(sl["z"][tr])
                Gl = Emp((sl["z"][tr] - locl) / scl)
                P = {"simple": probs4(lambda k: G.cdf((k - loc) / sc), s["sh"][te]),
                     "log": probs4(lambda k: Gl.cdf((k - locl) / scl), sl["sh"][te], logspace=True)}
                P["clim"] = np.tile(L.climatology(ai[tr]), (int(te.sum()), 1))
                row = dict(year=Y, n=int(te.sum()))
                for m in ("simple", "log", "clim"):
                    p = np.clip(P[m][np.arange(int(te.sum())), ai[te]], 1e-12, 1)
                    LL[m].append(-np.log(p))
                    row[m] = float(-np.log(p).mean())
                per_year.append(row)
                keep_ai.append(ai[te])
                keep_xr.append(xr[te])
                for m in ("simple", "log"):
                    keep_P[m].append(P[m])
            if not per_year:
                continue
            t = pd.DataFrame(per_year)
            w = t.n / t.n.sum()
            d_year = t.log - t.simple
            tt = d_year.mean() / (d_year.std(ddof=1) / math.sqrt(len(d_year)))
            print(f"\n h={h} [{label}] OOS log loss  simple={float((t.simple*w).sum()):.5f} "
                  f"log={float((t.log*w).sum()):.5f} clim={float((t.clim*w).sum()):.5f} | "
                  f"mean d(log-simple)={d_year.mean():+.5f}  year-clustered t={tt:+.2f} (n={len(t)}) "
                  f"| log better in {int((d_year<0).sum())}/{len(t)} years")
            AI = np.concatenate(keep_ai)
            XR = np.concatenate(keep_xr)
            PS = {m: np.vstack(keep_P[m]) for m in ("simple", "log")}
            if label != "SINGLES":
                continue
            qb = np.digitize(XR, [1 / 3, 2 / 3])
            rows = []
            for b in range(3):
                m = qb == b
                r = dict(xsec_vol_tercile=b + 1, n=int(m.sum()))
                for bi, bn in ((0, "dn"), (3, "up")):
                    obs = float((AI[m] == bi).mean())
                    r[bn + "_obs"] = obs
                    for mo in ("simple", "log"):
                        r[f"{bn}_{mo}_gap"] = float(PS[mo][m, bi].mean() - obs)
                for mo in ("simple", "log"):
                    p = np.clip(PS[mo][np.arange(len(AI))[m], AI[m]], 1e-12, 1)
                    r[mo + "_ll"] = float(-np.log(p).mean())
                rows.append(r)
            print(pd.DataFrame(rows).to_string(index=False))


SECTIONS = {0: sec0, 1: sec1, 2: sec2, 3: sec3, 4: sec4, 5: sec5, 6: sec6, 7: sec7, 8: sec8,
            9: sec9, 10: sec10, 11: sec11}

if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]] or sorted(SECTIONS)
    for a in args:
        SECTIONS[a]()
