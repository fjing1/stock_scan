"""
_move_wf_verify_hscale.py — ADVERSARIAL verification of the "horizon-scaling exponent is 0.45-0.47"
claim from _move_wf_volscale.py / _move_wf_volscale3.py.

CLAIM UNDER TEST
  beta (d log sigma_h / d log h) = 0.45-0.47 not 0.50, so kappa_63 = sigma_63/(sigma_1*sqrt(63))
  = 0.8184 for SPY => sqrt(h) overstates the sd of the 63-day index return by 22%; and this comes
  from RETURN AUTOCORRELATION, not vol clustering.

ATTACK PLAN
  0. exact replication of the original estimator
  1. NULL A: SIGN RANDOMIZATION.  r_t -> s_t*r_t with s_t = +/-1 iid.  This preserves |r_t| and
     therefore the ENTIRE volatility-clustering structure EXACTLY, while forcing the true
     autocovariances to zero, i.e. true beta = 0.5 and true kappa_h = 1 for every h.  Any measured
     deviation under this null is pure estimator bias-under-heteroskedasticity.  This is the direct
     test of the "it's autocorrelation, not vol clustering" mechanism claim.
  2. NULL B: IID PERMUTATION (kills autocorrelation AND vol clustering) — separates
     "fat tails + finite sample" from "vol clustering".
  3. HONEST INFERENCE:
       (a) Lo-MacKinlay (1988) HETEROSKEDASTICITY-CONSISTENT variance-ratio test z*(q) — the
           textbook inference that is valid under vol clustering.
       (b) non-overlapping 63-day windows only (n = T/63 ~ 99), bootstrap over those windows.
       (c) the 252-row block bootstrap used originally, for comparison — and a demonstration of
           why it is biased for h comparable to the block length.
  4. FRAGILITY: drop 2008-09, drop 2020, drop both; per-sub-period; winsorize daily returns.
  5. SIMPLER EXPLANATION / microstructure control: re-anchor kappa on a 5-day base unit
     (kappa'_h = sigma_h/(sigma_5*sqrt(h/5))).  If the deficit is bid-ask bounce / 1-day reversal
     it collapses when h=1 is removed from the denominator.
  6. LAG DECOMPOSITION: exactly how much of the kappa_63 deficit is lag 1 alone.
  7. FAKE SAMPLE SIZE on the single-name number: the original CI bootstraps over SYMBOLS, which
     ignores that all 231 names share ONE realization of market history.  Redo with a panel
     block bootstrap in TIME (clusters by date).
  8. OOS / LOOKAHEAD: the reported kappa is a full-sample IN-SAMPLE statistic.  Walk-forward:
     estimate kappa_h on years < y, score against year y's realized kappa.  Does kappa_hat beat
     kappa=1 out of sample?
  9. PRACTICAL RELEVANCE: the system does not use the unconditional sigma_1; it uses a conditional
     EWMA sigma with a train-fitted scale c_h.  Measure sd( r_h / (sigma_ewma*sqrt(h)) ) OOS to see
     whether "sqrt(h) overstates by 22%" transfers at all, and whether a fitted c_h already
     absorbs kappa (which would make the correction a no-op).

Run: ../../vcp_env/bin/python _move_wf_verify_hscale.py
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

import _move_data as D
import _move_lib as L

warnings.filterwarnings("ignore")

HS_ALL = [1, 2, 3, 5, 10, 21, 42, 63]
IDX = ["SPY", "QQQ", "^GSPC", "IWM", "DIA"]
MIN_OBS = 500
RNG = np.random.default_rng(11)
N_NULL = 2000          # sign-randomisation / permutation reps
N_BOOT = 2000

pd.set_option("display.width", 235)
pd.set_option("display.max_columns", 90)
pd.set_option("display.max_rows", 500)


# ------------------------------------------------------------------ estimators
def hstd_orig(x, h):
    """EXACT replication of _move_wf_volscale.hday_logret_std (pandas version, T-h windows)."""
    x = pd.Series(x).dropna()
    if len(x) < 5 * h + 30:
        return np.nan
    cs = x.cumsum()
    rh = (cs - cs.shift(h)).dropna().values
    dev = rh - h * x.mean()
    return float(np.sqrt((dev ** 2).sum() / (len(dev) - 1)))


def hstd_fast(x, h, mu=None):
    """Same statistic, numpy, all T-h+1 overlapping windows (one more window than the original;
    difference is O(1/T) and is verified against hstd_orig below)."""
    T = len(x)
    if T < 5 * h + 30:
        return np.nan
    cs = np.concatenate(([0.0], np.cumsum(x)))
    rh = cs[h:] - cs[:-h]
    dev = rh - h * (x.mean() if mu is None else mu)
    return float(np.sqrt((dev ** 2).sum() / (len(dev) - 1)))


def hstd_lm(x, h):
    """Lo-MacKinlay UNBIASED denominator: m/q = (T-q+1)(1-q/T)."""
    T = len(x)
    if T < 5 * h + 30:
        return np.nan
    cs = np.concatenate(([0.0], np.cumsum(x)))
    rh = cs[h:] - cs[:-h]
    dev = rh - h * x.mean()
    denom = (T - h + 1) * (1.0 - h / T)
    return float(np.sqrt((dev ** 2).sum() / denom))


def beta_from_sds(hs, sds):
    hs, sds = np.asarray(hs, float), np.asarray(sds, float)
    m = np.isfinite(sds) & (sds > 0)
    if m.sum() < 4:
        return np.nan
    A = np.column_stack([np.log(hs[m]), np.ones(m.sum())])
    b, *_ = np.linalg.lstsq(A, np.log(sds[m]), rcond=None)
    return float(b[0])


def kappas(x, hs=HS_ALL, est=hstd_fast):
    sds = np.array([est(x, h) for h in hs], float)
    return sds, sds / (sds[0] * np.sqrt(np.array(hs, float))), beta_from_sds(hs, sds)


def lm_vr_test(x, q):
    """Lo-MacKinlay 1988 heteroskedasticity-consistent variance-ratio test.
    Returns VR(q), kappa=sqrt(VR), theta(q), z*(q).  Valid under conditional heteroskedasticity
    (i.e. under vol clustering) — rejects only for genuine autocorrelation."""
    x = np.asarray(x, float)
    T = len(x)
    mu = x.mean()
    d = x - mu
    sig_a2 = (d ** 2).sum() / (T - 1)
    cs = np.concatenate(([0.0], np.cumsum(x)))
    rh = cs[q:] - cs[:-q]
    dev = rh - q * mu
    m = q * (T - q + 1) * (1.0 - q / T)
    sig_c2 = (dev ** 2).sum() / m
    VR = sig_c2 / sig_a2
    den = ((d ** 2).sum()) ** 2
    theta = 0.0
    d2 = d ** 2
    for j in range(1, q):
        delta_j = (d2[j:] * d2[:-j]).sum() / den
        theta += (2.0 * (q - j) / q) ** 2 * delta_j
    z = (VR - 1.0) / np.sqrt(theta) if theta > 0 else np.nan
    return VR, float(np.sqrt(max(VR, 0))), theta, z


def nonoverlap_sd(x, h):
    """sd of NON-overlapping h-day sums, averaged over all h possible phase offsets.
    Also returns the per-offset spread and the count of windows per offset."""
    x = np.asarray(x, float)
    T = len(x)
    mu = x.mean()
    cs = np.concatenate(([0.0], np.cumsum(x)))
    out, ns = [], []
    for off in range(h):
        idx = np.arange(off, T - h + 1, h)
        if len(idx) < 8:
            continue
        rh = cs[idx + h] - cs[idx]
        dev = rh - h * mu
        out.append(np.sqrt((dev ** 2).sum() / (len(dev) - 1)))
        ns.append(len(dev))
    return float(np.mean(out)), float(np.std(out)), int(np.mean(ns)), np.array(out)


def make_blocks(n, block):
    return [np.arange(s, min(s + block, n)) for s in range(0, n, block)]


# ------------------------------------------------------------------ panel-wide vectorised version
def panel_kappa(LR, M, hs=HS_ALL):
    """LR (T,S) log returns with NaN->0 applied, M (T,S) bool validity mask.
    Returns sd (H,S), kappa (H,S), beta (S,) using all fully-valid overlapping windows."""
    T, S = LR.shape
    X = np.where(M, LR, 0.0)
    C = np.vstack([np.zeros((1, S)), np.cumsum(X, 0)])
    CM = np.vstack([np.zeros((1, S)), np.cumsum(M.astype(np.float64), 0)])
    nobs = M.sum(0)
    mu1 = np.divide(X.sum(0), nobs, out=np.zeros(S), where=nobs > 0)
    sd = np.full((len(hs), S), np.nan)
    for i, h in enumerate(hs):
        rh = C[h:] - C[:-h]
        cnt = CM[h:] - CM[:-h]
        ok = cnt == h
        dev = np.where(ok, rh - h * mu1, 0.0)
        n = ok.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.where(n >= 5 * h + 30, (dev ** 2).sum(0) / np.maximum(n - 1, 1), np.nan)
        sd[i] = np.sqrt(v)
    hsa = np.array(hs, float)
    kap = sd / (sd[0] * np.sqrt(hsa)[:, None])
    lg = np.log(hsa)
    A = np.column_stack([lg, np.ones(len(hs))])
    beta = np.full(S, np.nan)
    good = np.isfinite(sd).all(0) & (sd > 0).all(0)
    if good.any():
        b, *_ = np.linalg.lstsq(A, np.log(sd[:, good]), rcond=None)
        beta[good] = b[0]
    return sd, kap, beta


def main():
    t0 = time.time()
    p = D.load()
    close = p["Close"]
    keep = [s for s in close.columns if close[s].notna().sum() >= MIN_OBS and s != "^VIX"]
    close = close[keep]
    singles = [s for s in keep if s not in IDX]
    lr = np.log(close).diff()
    years = close.index.year.values
    print(f"panel: {len(keep)} usable symbols = {len(IDX)} indices + {len(singles)} singles; "
          f"{len(close)} rows {close.index[0].date()} -> {close.index[-1].date()}")

    # ============================================================ 0. REPLICATION
    print("\n" + "=" * 118)
    print("0. REPLICATION of the claimed numbers with the ORIGINAL estimator")
    print("=" * 118)
    rep = []
    for s in IDX:
        x = lr[s].dropna().values
        sds_o = [hstd_orig(x, h) for h in HS_ALL]
        b_o = beta_from_sds(HS_ALL, sds_o)
        sds_f, kap_f, b_f = kappas(x)
        sds_l = [hstd_lm(x, h) for h in HS_ALL]
        b_l = beta_from_sds(HS_ALL, sds_l)
        rep.append(dict(sym=s, T=len(x), beta_orig=b_o, beta_fast=b_f, beta_LMunbiased=b_l,
                        k63_orig=sds_o[-1] / (sds_o[0] * np.sqrt(63)),
                        k63_fast=kap_f[-1],
                        k63_LMunbiased=sds_l[-1] / (sds_l[0] * np.sqrt(63))))
    print(pd.DataFrame(rep).round(4).to_string(index=False))
    print("  -> replication OK if beta_orig matches the claim (SPY 0.4548, ^GSPC 0.4556,")
    print("     DIA 0.4502, QQQ 0.4697, IWM 0.4739).  beta_LMunbiased shows the size of the")
    print("     small-sample denominator bias in the original estimator.")

    spy = lr["SPY"].dropna().values
    sds_spy, kap_spy, beta_spy = kappas(spy)
    print("\nSPY kappa_h with the fast estimator:")
    print(pd.DataFrame({"h": HS_ALL, "sigma_h": sds_spy, "kappa_h": kap_spy}).round(4)
          .to_string(index=False))

    # ============================================================ 1/2. NULLS
    print("\n" + "=" * 118)
    print("1. NULL A — SIGN RANDOMISATION (r_t -> +/-r_t).  Volatility clustering PRESERVED EXACTLY")
    print("   (|r_t| sequence untouched); true autocovariances = 0, so TRUE beta = 0.5, kappa = 1.")
    print("   Any deviation of the null distribution from 0.5/1.0 is estimator bias, NOT signal.")
    print("2. NULL B — IID PERMUTATION (kills autocorrelation AND vol clustering).")
    print(f"   {N_NULL} reps each.")
    print("=" * 118)
    null_store = {}
    for s in IDX:
        x = lr[s].dropna().values
        T = len(x)
        bA = np.empty(N_NULL); kA = np.empty(N_NULL)
        bB = np.empty(N_NULL); kB = np.empty(N_NULL)
        for i in range(N_NULL):
            sg = RNG.integers(0, 2, T) * 2 - 1
            xa = x * sg
            _, ka, ba = kappas(xa)
            bA[i], kA[i] = ba, ka[-1]
            xb = x[RNG.permutation(T)]
            _, kb, bb = kappas(xb)
            bB[i], kB[i] = bb, kb[-1]
        obs_b = beta_from_sds(HS_ALL, [hstd_fast(x, h) for h in HS_ALL])
        obs_k = kappas(x)[1][-1]
        null_store[s] = dict(bA=bA, kA=kA, bB=bB, kB=kB, obs_b=obs_b, obs_k=obs_k)
        print(f"\n  {s}: OBSERVED beta={obs_b:.4f}  kappa_63={obs_k:.4f}")
        for nm, bb, kk in (("NULL A sign-rand (vol clustering kept)", bA, kA),
                           ("NULL B iid permutation             ", bB, kB)):
            pb = (bb <= obs_b).mean()
            pk = (kk <= obs_k).mean()
            print(f"    {nm}: beta mean {bb.mean():.4f} sd {bb.std():.4f} "
                  f"[{np.percentile(bb,2.5):.4f},{np.percentile(bb,97.5):.4f}]  "
                  f"kappa63 mean {kk.mean():.4f} sd {kk.std():.4f} "
                  f"[{np.percentile(kk,2.5):.4f},{np.percentile(kk,97.5):.4f}]  "
                  f"| p(null<=obs): beta {pb:.4f} kappa {pk:.4f}")
    print("\n  READ: if NULL A's kappa_63 is centred on ~1.00 then vol clustering contributes")
    print("  nothing and the observed shortfall is genuine return autocorrelation.  If NULL A is")
    print("  centred well below 1.00, part/all of the '22%' is an artefact of measuring an sd of")
    print("  heteroskedastic overlapping sums in a finite sample.")

    # null A kappa profile across h for SPY (is the bias h-dependent?)
    print("\n  NULL A kappa_h profile, SPY (500 reps) — bias as a function of h:")
    KA = np.empty((500, len(HS_ALL)))
    for i in range(500):
        sg = RNG.integers(0, 2, len(spy)) * 2 - 1
        KA[i] = kappas(spy * sg)[1]
    prof = pd.DataFrame({"h": HS_ALL, "obs_kappa": kap_spy,
                         "nullA_mean": KA.mean(0), "nullA_p2.5": np.percentile(KA, 2.5, 0),
                         "nullA_p97.5": np.percentile(KA, 97.5, 0)})
    prof["obs/nullA"] = prof.obs_kappa / prof.nullA_mean
    print(prof.round(4).to_string(index=False))

    # ============================================================ 3. HONEST INFERENCE
    print("\n" + "=" * 118)
    print("3. HONEST INFERENCE")
    print("=" * 118)
    print("\n(a) Lo-MacKinlay heteroskedasticity-CONSISTENT variance-ratio test z*(q).")
    print("    |z*| > 1.96 => the deviation from kappa=1 is NOT explainable by vol clustering.")
    rows = []
    for s in IDX:
        x = lr[s].dropna().values
        for q in (2, 5, 21, 63):
            VR, kk, th, z = lm_vr_test(x, q)
            rows.append(dict(sym=s, q=q, T=len(x), VR=VR, kappa=kk, sqrt_theta=np.sqrt(th),
                             z_star=z, signif=abs(z) > 1.96))
    lmt = pd.DataFrame(rows)
    print(lmt.round(4).to_string(index=False))
    print("\n    kappa 95% CI implied by the HC variance-ratio test (delta method on VR):")
    for s in IDX:
        g = lmt[lmt.sym == s]
        parts = []
        for _, r in g.iterrows():
            lo, hi = r.VR - 1.96 * r.sqrt_theta, r.VR + 1.96 * r.sqrt_theta
            parts.append(f"q{int(r.q)}:{np.sqrt(max(lo,1e-9)):.3f}-{np.sqrt(max(hi,1e-9)):.3f}")
        print(f"      {s:>6}: " + "  ".join(parts))

    print("\n(b) NON-OVERLAPPING windows only (averaged over all phase offsets), + bootstrap CI")
    print("    over the ~T/h independent windows.  This is the honest small-n picture.")
    rows = []
    for s in IDX:
        x = lr[s].dropna().values
        T = len(x)
        s1 = hstd_fast(x, 1)
        for q in (5, 21, 63):
            mn, sd_off, nwin, per_off = nonoverlap_sd(x, q)
            k_no = mn / (s1 * np.sqrt(q))
            # bootstrap over non-overlapping windows of one canonical offset set (offset 0)
            cs = np.concatenate(([0.0], np.cumsum(x)))
            idx = np.arange(0, T - q + 1, q)
            rh = cs[idx + q] - cs[idx] - q * x.mean()
            bs = []
            for _ in range(N_BOOT):
                v = rh[RNG.integers(0, len(rh), len(rh))]
                bs.append(np.sqrt((v ** 2).sum() / (len(v) - 1)) / (s1 * np.sqrt(q)))
            rows.append(dict(sym=s, h=q, n_windows=nwin, kappa_nonovlp=k_no,
                             kappa_ovlp=hstd_fast(x, q) / (s1 * np.sqrt(q)),
                             sd_across_offsets=sd_off / (s1 * np.sqrt(q)),
                             boot_lo=np.percentile(bs, 2.5), boot_hi=np.percentile(bs, 97.5)))
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\n(c) The ORIGINAL 252-row block bootstrap, and why it is wrong for h near the block")
    print("    length: 63-day windows straddling a block seam join unrelated periods, which")
    print("    DESTROYS the autocorrelation the statistic is measuring and pulls kappa toward 1.")
    print("    Test: apply the same block bootstrap to a series whose kappa is known.")
    for blk in (252, 504, 1008):
        ks = []
        blocks = make_blocks(len(spy), blk)
        for _ in range(600):
            pick = RNG.integers(0, len(blocks), len(blocks))
            xb = np.concatenate([spy[blocks[i]] for i in pick])
            ks.append(kappas(xb)[1][-1])
        print(f"    SPY kappa_63 block bootstrap, block={blk:>4}: mean {np.mean(ks):.4f}  "
              f"95% CI [{np.percentile(ks,2.5):.4f}, {np.percentile(ks,97.5):.4f}]   "
              f"(point estimate {kap_spy[-1]:.4f})")
    # bias demonstration: block-bootstrap a series with a KNOWN kappa=1 (sign randomised)
    sgfix = RNG.integers(0, 2, len(spy)) * 2 - 1
    xs = spy * sgfix
    for blk in (252, 1008):
        ks = []
        blocks = make_blocks(len(xs), blk)
        for _ in range(400):
            pick = RNG.integers(0, len(blocks), len(blocks))
            xb = np.concatenate([xs[blocks[i]] for i in pick])
            ks.append(kappas(xb)[1][-1])
        print(f"    control: sign-randomised SPY (TRUE kappa_63 = 1), block={blk:>4}: "
              f"mean {np.mean(ks):.4f}  95% CI [{np.percentile(ks,2.5):.4f}, "
              f"{np.percentile(ks,97.5):.4f}]  point {kappas(xs)[1][-1]:.4f}")

    # ============================================================ 4. FRAGILITY
    print("\n" + "=" * 118)
    print("4. FRAGILITY — crises, sub-periods, winsorisation")
    print("=" * 118)
    subs = [("full 2001-2026", np.ones(len(close), bool)),
            ("ex 2008-2009", ~np.isin(years, [2008, 2009])),
            ("ex 2020", years != 2020),
            ("ex 2008-09 & 2020", ~np.isin(years, [2008, 2009, 2020])),
            ("ex 2008-09,2011,2018,2020,2022", ~np.isin(years, [2008, 2009, 2011, 2018, 2020, 2022])),
            ("2001-2013", years < 2014),
            ("2014-2026", years >= 2014),
            ("2001-2008", years <= 2008),
            ("2009-2016", (years >= 2009) & (years <= 2016)),
            ("2017-2026", years >= 2017)]
    rows = []
    for nm, msk in subs:
        for s in IDX:
            x = lr[s][msk].dropna().values
            sd_, k_, b_ = kappas(x)
            VR, kk, th, z = lm_vr_test(x, 63)
            rows.append(dict(period=nm, sym=s, T=len(x), beta=b_, k5=k_[3], k21=k_[5],
                             k63=k_[-1], z_star_63=z))
    fr = pd.DataFrame(rows)
    print("\nbeta by sub-period (rows=period, cols=index):")
    print(fr.pivot(index="period", columns="sym", values="beta").reindex([n for n, _ in subs])
          [IDX].round(4).to_string())
    print("\nkappa_63 by sub-period:")
    pk = fr.pivot(index="period", columns="sym", values="k63").reindex([n for n, _ in subs])[IDX]
    pk["mean"] = pk.mean(axis=1)
    pk["implied_overstatement%"] = 100 * (1 / pk["mean"] - 1)
    print(pk.round(4).to_string())
    print("\nkappa_21 by sub-period:")
    print(fr.pivot(index="period", columns="sym", values="k21").reindex([n for n, _ in subs])
          [IDX].round(4).to_string())
    print("\nLo-MacKinlay HC z*(63) by sub-period (|z|>1.96 = significant):")
    print(fr.pivot(index="period", columns="sym", values="z_star_63")
          .reindex([n for n, _ in subs])[IDX].round(3).to_string())
    print("\nn per sub-period (SPY):")
    print(fr[fr.sym == "SPY"].set_index("period").T.loc[["T"]].to_string())

    print("\nWINSORISATION of daily log returns (SPY): does a handful of huge reversal days do it?")
    sd1_spy = spy.std()
    rows = []
    for lab, clip in [("none", None), ("+/-5%", 0.05), ("+/-3%", 0.03), ("+/-2%", 0.02),
                      ("+/-4 sd", 4 * sd1_spy), ("+/-3 sd", 3 * sd1_spy)]:
        x = spy.copy() if clip is None else np.clip(spy, -clip, clip)
        sd_, k_, b_ = kappas(x)
        VR, kk, th, z = lm_vr_test(x, 63)
        rows.append(dict(clip=lab, beta=b_, k5=k_[3], k21=k_[5], k63=k_[-1], z63=z,
                         n_clipped=int((np.abs(x - spy) > 1e-15).sum())))
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    # ============================================================ 5. BASE-UNIT CONTROL
    print("\n" + "=" * 118)
    print("5. SIMPLER EXPLANATION — is the deficit a 1-day (microstructure / short-term reversal)")
    print("   effect rather than a genuine scaling law?  Re-anchor kappa on a 5-day base unit:")
    print("   kappa'_h = sigma_h / (sigma_5 * sqrt(h/5)).  A true power law with beta=0.455 gives")
    print("   kappa'_63 = (63/5)^(beta-0.5) = 0.891; if instead kappa'_63 ~ 1 the whole thing is a")
    print("   1-day-reversal artefact sitting in the DENOMINATOR sigma_1.")
    print("=" * 118)
    rows = []
    for s in IDX:
        x = lr[s].dropna().values
        sd_ = np.array([hstd_fast(x, h) for h in HS_ALL], float)
        i5 = HS_ALL.index(5)
        r = dict(sym=s, kappa_base1_63=sd_[-1] / (sd_[0] * np.sqrt(63)),
                 kappa_base5_63=sd_[-1] / (sd_[i5] * np.sqrt(63 / 5)),
                 kappa_base21_63=sd_[-1] / (sd_[HS_ALL.index(21)] * np.sqrt(63 / 21)),
                 kappa_base1_5=sd_[i5] / (sd_[0] * np.sqrt(5)),
                 kappa_base1_2=sd_[1] / (sd_[0] * np.sqrt(2)))
        # beta fit over h>=5 only
        r["beta_h1to63"] = beta_from_sds(HS_ALL, sd_)
        sub = [i for i, h in enumerate(HS_ALL) if h >= 5]
        r["beta_h5to63"] = beta_from_sds([HS_ALL[i] for i in sub], sd_[sub])
        sub2 = [i for i, h in enumerate(HS_ALL) if h >= 10]
        r["beta_h10to63"] = beta_from_sds([HS_ALL[i] for i in sub2], sd_[sub2])
        rows.append(r)
    bu = pd.DataFrame(rows)
    print(bu.round(4).to_string(index=False))
    print(f"\n  power-law prediction if beta were really 0.455: kappa'_63(base 5) = "
          f"{(63/5)**(0.455-0.5):.4f}, kappa'_63(base 21) = {(63/21)**(0.455-0.5):.4f}, "
          f"kappa_5(base 1) = {5**(0.455-0.5):.4f}, kappa_2(base 1) = {2**(0.455-0.5):.4f}")
    print("  NULL A check on the base-5 quantity (SPY, 500 reps):")
    K5 = np.empty(500)
    for i in range(500):
        sg = RNG.integers(0, 2, len(spy)) * 2 - 1
        xa = spy * sg
        K5[i] = hstd_fast(xa, 63) / (hstd_fast(xa, 5) * np.sqrt(63 / 5))
    obs5 = sds_spy[-1] / (sds_spy[HS_ALL.index(5)] * np.sqrt(63 / 5))
    print(f"    observed {obs5:.4f}; null A mean {K5.mean():.4f} "
          f"[{np.percentile(K5,2.5):.4f},{np.percentile(K5,97.5):.4f}]  "
          f"p(null<=obs) = {(K5<=obs5).mean():.4f}")

    # ============================================================ 6. LAG DECOMPOSITION
    print("\n" + "=" * 118)
    print("6. LAG DECOMPOSITION.  kappa_h^2 = 1 + 2*sum_{k=1}^{h-1} (1-k/h) rho_k  (Bartlett)")
    print("=" * 118)
    rows = []
    for s in IDX:
        x = lr[s].dropna().values
        d = x - x.mean()
        den = (d ** 2).sum()
        rho = np.array([(d[k:] * d[:-k]).sum() / den for k in range(1, 63)])
        w = 1 - np.arange(1, 63) / 63.0
        tot = 2 * (w * rho).sum()
        lag1 = 2 * w[0] * rho[0]
        lag2_5 = 2 * (w[1:5] * rho[1:5]).sum()
        lag6_21 = 2 * (w[5:21] * rho[5:21]).sum()
        lag22_62 = 2 * (w[21:] * rho[21:]).sum()
        rows.append(dict(sym=s, rho1=rho[0], rho2=rho[1], rho3=rho[2],
                         sum_all=tot, from_lag1=lag1, from_lag2_5=lag2_5,
                         from_lag6_21=lag6_21, from_lag22_62=lag22_62,
                         kappa63_implied=np.sqrt(max(1 + tot, 1e-9)),
                         kappa63_direct=kappas(x)[1][-1],
                         kappa63_if_only_lag1=np.sqrt(max(1 + lag1, 1e-9))))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("  -> 'from_lag1' vs 'sum_all' tells you whether this is a one-lag microstructure/reversal")
    print("     effect or a genuine long-range negative dependence.")

    # ============================================================ 7. SINGLE NAMES: CLUSTERING
    print("\n" + "=" * 118)
    print("7. SINGLE NAMES — the claimed CI [0.4676, 0.4752] bootstraps over SYMBOLS.  All 231")
    print("   names share ONE realisation of market history, so symbol-clustering ignores the")
    print("   dominant dependence.  Redo with a panel block bootstrap in TIME (clusters by date).")
    print("=" * 118)
    LRs = lr[singles].values
    Ms = np.isfinite(LRs)
    LRs = np.where(Ms, LRs, 0.0)
    sd_s, kap_s, beta_s = panel_kappa(LRs, Ms)
    good = np.isfinite(beta_s)
    print(f"  per-symbol beta (n={int(good.sum())} symbols): mean {beta_s[good].mean():.4f}  "
          f"median {np.median(beta_s[good]):.4f}  sd {beta_s[good].std():.4f}  "
          f"share<0.5 {(beta_s[good] < 0.5).mean():.3f}")
    bs_sym = [beta_s[good][RNG.integers(0, good.sum(), good.sum())].mean() for _ in range(4000)]
    print(f"  95% CI clustered by SYMBOL (as originally reported): "
          f"[{np.percentile(bs_sym,2.5):.4f}, {np.percentile(bs_sym,97.5):.4f}]  "
          f"(width {np.percentile(bs_sym,97.5)-np.percentile(bs_sym,2.5):.4f})")
    for blk in (252, 504):
        blocks = make_blocks(len(LRs), blk)
        bs_time = []
        for _ in range(300):
            pick = RNG.integers(0, len(blocks), len(blocks))
            ridx = np.concatenate([blocks[i] for i in pick])
            _, _, b = panel_kappa(LRs[ridx], Ms[ridx])
            g = np.isfinite(b)
            if g.sum() > 50:
                bs_time.append(b[g].mean())
        bs_time = np.array(bs_time)
        print(f"  95% CI clustered by DATE (panel block bootstrap, block={blk}): "
              f"[{np.percentile(bs_time,2.5):.4f}, {np.percentile(bs_time,97.5):.4f}]  "
              f"(width {np.percentile(bs_time,97.5)-np.percentile(bs_time,2.5):.4f})  "
              f"mean {bs_time.mean():.4f}")
    # NULL A for the whole single-name panel: same signs applied to all names on a date
    print("\n  NULL A for singles: apply the SAME random sign to every name on a given date")
    print("  (preserves each name's vol clustering AND the cross-sectional correlation), 200 reps:")
    nb = []
    for _ in range(200):
        sg = (RNG.integers(0, 2, len(LRs)) * 2 - 1).astype(float)[:, None]
        _, _, b = panel_kappa(LRs * sg, Ms)
        g = np.isfinite(b)
        nb.append(b[g].mean())
    nb = np.array(nb)
    print(f"    observed mean beta {beta_s[good].mean():.4f}; NULL A mean {nb.mean():.4f} "
          f"[{np.percentile(nb,2.5):.4f},{np.percentile(nb,97.5):.4f}]  "
          f"p(null<=obs) {(nb <= beta_s[good].mean()).mean():.4f}")
    print("    -> observed minus null-A mean = the bias-corrected single-name deficit:")
    print(f"       {beta_s[good].mean() - nb.mean():+.4f} in beta units "
          f"(claim implies {0.4714-0.5:+.4f})")

    # ============================================================ 8. WALK-FORWARD OOS
    print("\n" + "=" * 118)
    print("8. LOOKAHEAD / OOS.  The reported kappa is a FULL-SAMPLE IN-SAMPLE statistic.")
    print("   Walk-forward: kappa_hat_h from years < y only; realised kappa_h measured IN year y.")
    print("   Does kappa_hat beat the naive kappa=1 out of sample?")
    print("=" * 118)
    yrs = sorted(set(years))

    def realised_kappa(seg1, seg_ext, h):
        """seg1 = daily returns INSIDE the test segment (for sigma_1);
        seg_ext = the same segment extended by h-1 extra later bars, so that every h-day window
        STARTING inside the segment is measurable.  Only windows starting in the segment are used."""
        s1 = hstd_fast(seg1, 1)
        n_start = len(seg1)
        if len(seg_ext) < n_start + h - 1 or n_start < 3 * h:
            return np.nan, 0
        cs = np.concatenate(([0.0], np.cumsum(seg_ext)))
        st = np.arange(0, n_start)
        rh = cs[st + h] - cs[st]
        dev = rh - h * np.mean(seg_ext)
        sh = np.sqrt((dev ** 2).sum() / (len(dev) - 1))
        return float(sh / (s1 * np.sqrt(h))), len(dev)

    def seg_returns(sym, mask, h):
        v = lr[sym].values
        fin = np.isfinite(v)
        rows = np.where(mask & fin)[0]
        if len(rows) == 0:
            return np.array([]), np.array([])
        lastrow = rows[-1]
        ext_rows = np.where(fin & (np.arange(len(v)) <= min(lastrow + h - 1, len(v) - 1))
                            & (np.arange(len(v)) >= rows[0]))[0]
        return v[rows], v[ext_rows]

    from math import comb
    for h in (5, 21, 63):
        rows = []
        for y in yrs[5:]:
            tr = lr["SPY"][years < y].dropna().values
            seg, ext = seg_returns("SPY", years == y, h)
            if len(tr) < 1000 or len(seg) < 3 * h:
                continue
            kh = hstd_fast(tr, h) / (hstd_fast(tr, 1) * np.sqrt(h))
            kr, nw = realised_kappa(seg, ext, h)
            if not (np.isfinite(kh) and np.isfinite(kr)):
                continue
            rows.append(dict(y=y, n_win=nw, k_hat=kh, k_real=kr,
                             err_hat=abs(np.log(kr) - np.log(kh)), err_one=abs(np.log(kr))))
        r = pd.DataFrame(rows)
        if r.empty:
            continue
        wins = int((r.err_hat < r.err_one).sum())
        n_ = len(r)
        pval = sum(comb(n_, k) for k in range(wins, n_ + 1)) / 2 ** n_
        tstat = (r.k_real.mean() - 1) / (r.k_real.std() / np.sqrt(n_))
        print(f"\n  SPY h={h}: n_test_years={n_}  mean kappa_hat {r.k_hat.mean():.4f}  "
              f"mean realised kappa {r.k_real.mean():.4f}  median realised {r.k_real.median():.4f}  "
              f"years with realised kappa<1: {int((r.k_real<1).sum())}/{n_}  "
              f"t(realised vs 1) = {tstat:.2f}")
        print(f"    mean |log err|: kappa_hat {r.err_hat.mean():.4f} vs kappa=1 "
              f"{r.err_one.mean():.4f}  -> kappa_hat wins {wins}/{n_} years, "
              f"one-sided sign-test p={pval:.4f}")
        if h == 63:
            print(r.round(4).to_string(index=False))

    # non-overlapping multi-year test blocks (h=63 needs room; 3-year blocks give ~4 non-ovlp obs)
    print("\n  h=63 on NON-OVERLAPPING 3-year test blocks (expanding train), all 5 indices:")
    rows = []
    blocks3 = [(y, y + 2) for y in range(2007, 2027, 3)]
    for (a, b) in blocks3:
        m = (years >= a) & (years <= b)
        if m.sum() < 400:
            continue
        for s in IDX:
            tr = lr[s][years < a].dropna().values
            seg, ext = seg_returns(s, m, 63)
            if len(tr) < 1000 or len(seg) < 200:
                continue
            kh = hstd_fast(tr, 63) / (hstd_fast(tr, 1) * np.sqrt(63))
            kr, nw = realised_kappa(seg, ext, 63)
            rows.append(dict(block=f"{a}-{b}", sym=s, n_win=nw, k_hat=kh, k_real=kr))
    r3 = pd.DataFrame(rows)
    print(r3.pivot(index="block", columns="sym", values="k_real").round(4).to_string())
    print("    kappa_hat (train-only, same for all blocks per index):")
    print(r3.pivot(index="block", columns="sym", values="k_hat").round(4).to_string())
    bm = r3.groupby("block").k_real.mean()
    print(f"    block means: {bm.round(4).to_dict()}   mean {bm.mean():.4f}  "
          f"sd {bm.std():.4f}  t vs 1 = {(bm.mean()-1)/(bm.std()/np.sqrt(len(bm))):.2f} "
          f"(n={len(bm)} non-overlapping blocks)")


    # ============================================================ 9. PRACTICAL RELEVANCE
    print("\n" + "=" * 118)
    print("9. PRACTICAL RELEVANCE.  The system does NOT use the unconditional sigma_1; it uses a")
    print("   CONDITIONAL sigma (EWMA) with a train-fitted scale c_h.  Two questions:")
    print("   (i) does 'sqrt(h) overstates by 22%' transfer to sd( r_h / (sigma_ewma*sqrt(h)) )?")
    print("   (ii) if c_h is fitted on prior years, is the kappa correction a NO-OP?")
    print("=" * 118)
    s_e = L.vol_ewma(close, 0.94)
    for h in (5, 21, 63):
        fret = np.log(close.shift(-h) / close)
        rows = []
        for lab, cols in [("INDICES", IDX), ("SINGLES", singles)]:
            z_raw, z_kap, z_cfit = [], [], []
            for y in yrs[5:]:
                trm = years < y
                tem = years == y
                zt = (fret.loc[trm, cols] / (s_e.loc[trm, cols] * np.sqrt(h))).values.ravel()
                ze = (fret.loc[tem, cols] / (s_e.loc[tem, cols] * np.sqrt(h))).values.ravel()
                zt, ze = zt[np.isfinite(zt)], ze[np.isfinite(ze)]
                if len(zt) < 500 or len(ze) < 50:
                    continue
                c = np.sqrt(np.mean(zt ** 2))            # train-fitted scale (absorbs kappa)
                # train-estimated kappa_h from the same prior years (unconditional)
                kh = []
                for s in cols:
                    xx = lr[s][trm].dropna().values
                    v = hstd_fast(xx, h) / (hstd_fast(xx, 1) * np.sqrt(h))
                    if np.isfinite(v):
                        kh.append(v)
                kh = float(np.mean(kh)) if kh else 1.0
                z_raw.append(np.mean(ze ** 2))
                z_kap.append(np.mean((ze / kh) ** 2))
                z_cfit.append(np.mean((ze / c) ** 2))
            rows.append(dict(grp=lab, h=h, n_years=len(z_raw),
                             sd_z_raw_sqrt_h=np.sqrt(np.mean(z_raw)),
                             sd_z_kappa_corrected=np.sqrt(np.mean(z_kap)),
                             sd_z_c_fitted=np.sqrt(np.mean(z_cfit))))
        t = pd.DataFrame(rows)
        t["|sd-1| raw"] = (t.sd_z_raw_sqrt_h - 1).abs()
        t["|sd-1| kappa"] = (t.sd_z_kappa_corrected - 1).abs()
        t["|sd-1| c_fit"] = (t.sd_z_c_fitted - 1).abs()
        print(t.round(4).to_string(index=False))
    print("\n  READ: sd_z_raw_sqrt_h is what an UNCORRECTED sqrt(h) rule actually delivers OOS on")
    print("  a conditional EWMA sigma.  If it is >= 1, sqrt(h) does NOT overstate in the way the")
    print("  claim's headline implies; and dividing by kappa<1 makes calibration WORSE.")

    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
