"""
_move_wf_baseline.py — walk-forward evaluation harness + honest baseline for the
scale-and-shape move-probability model.

MODEL (provisional, as specified):
    sigma_hat_h(t) = vol_ewma(close, 0.94)[t] * sqrt(h)          # per-day log sigma -> h-day
    z              = log(close[t+h]/close[t]) / sigma_hat_h(t)   # standardized LOG return
    z-distribution = empirical quantiles of z, POOLED over the group, fit on TRAIN YEARS ONLY
    P(r_h <= -thr) = F_z( log(1-thr)/sigma_hat_h )               # simple-return threshold mapped
    P(r_h >= +thr) = 1 - F_z( log(1+thr)/sigma_hat_h )           # into log space exactly

  Why z is built on LOG returns: _move_lib's sigma is a log-return sigma, so the only
  dimensionally-correct standardization is log-return/sigma. Buckets are then defined by
  log(1 +/- thr), which is an EXACT re-expression of the simple-return bucket, not an
  approximation. No separate mu is estimated: the empirical z sample carries its own drift.

VARIANTS (for the skill decomposition):
    EMP       time-varying EWMA sigma + empirical z shape            <- the provisional model
    NORMFIT   time-varying EWMA sigma + Normal(mean(z), sd(z))       <- kills the fat tails
    NORMSTD   time-varying EWMA sigma + Normal(0, 1)                 <- also trusts sigma's scale
    CONSTSYM  ONE constant sigma per symbol (train mean) + empirical z   <- no vol timing
    CONSTGLOB ONE constant sigma for the whole group + empirical z       <- the dumb floor
    CLIM      train-year unconditional bucket frequencies, POOLED       <- the stated baseline
    CLIMSYM   train-year bucket frequencies OF THAT SYMBOL               <- the HARDER baseline;
              for a single-ticker product this is what the user could get for free by quoting
              the ticker's own historical base rate, so it is the one that decides shipping

NO-LOOKAHEAD DISCIPLINE:
  - L.walk_forward_years, expanding window, >=5 train years -> test years 2006..2026.
  - PURGE: the last h trading days of each train window are dropped, because their forward
    target reaches into the test year. Without this the fit peeks h days past the boundary.
  - Climatology is fit on train-year actuals only.
  - EWMA burn-in: the first 60 valid returns of every symbol are dropped (EWMA init bias).

PROBABILITY FLOOR: every model's 4 bucket probabilities are clipped at 1e-4 and renormalized,
i.e. no model is allowed to say "less than 0.01% chance". Log loss is otherwise dominated by a
handful of empirical-CDF zero cells and becomes an artifact of the floor. Brier is reported too
because it is floor-insensitive. Both floors are applied to CLIMATOLOGY as well, so the
comparison is fair.

UNCERTAINTY: forward windows overlap h-fold and cross-sectional correlation on a down day is
near 1, so the effective sample is far below the row count. CIs come from a DATE-BLOCK
bootstrap: contiguous blocks of h trading days are resampled with replacement and ALL symbols
on a sampled date travel together (cluster by date). n is reported as rows AND as unique dates.

SURVIVORSHIP: the single-name universe is names still listed today, so single-name tail
probabilities here are biased LOW. SPY/QQQ/IWM/DIA/^GSPC are unbiased -> the honest benchmark.

Run: ../../vcp_env/bin/python _move_wf_baseline.py
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _move_data as D
import _move_lib as L

HORIZONS = (1, 5, 10, 21)
THRESHOLDS = (0.01, 0.02, 0.03, 0.05)
LAM = 0.94
BURN = 60                 # EWMA initialisation burn-in, per symbol
MIN_COVER = 500           # drop near-empty symbols
FLOOR = 1e-4
MIN_TRAIN_YEARS = 5
INDEX_SYMS = ["SPY", "QQQ", "IWM", "DIA", "^GSPC"]     # ^VIX excluded: not a price series
NBOOT = 300
MODELS = ("EMP", "NORMFIT", "NORMSTD", "CONSTSYM", "CONSTGLOB")
RNG = np.random.default_rng(20260911)

_erfc = np.frompyfunc(math.erfc, 1, 1)


def norm_cdf(x):
    """Phi(x) = 0.5*erfc(-x/sqrt(2)). math.erfc is the C library's, so this is exact to
    double precision in both tails (no rational approximation error)."""
    return np.asarray(_erfc(-np.asarray(x, dtype=float) / math.sqrt(2.0)), dtype=float) * 0.5


# --------------------------------------------------------------------------- panel prep
def load_panel():
    p = D.load()
    close = p["Close"]
    cov = close.notna().sum()
    keep = cov[cov >= MIN_COVER].index
    close = close[keep]
    idx_syms = [s for s in INDEX_SYMS if s in keep]
    singles = [s for s in keep if s not in set(INDEX_SYMS) | {"^VIX"}]
    return close, idx_syms, singles


def flatten(close_g, h):
    """Flatten a (dates x symbols) group into 1-D observation arrays for horizon h."""
    logc = np.log(close_g)
    sig1 = L.vol_ewma(close_g, LAM)
    fwd_s = L.forward_simple_return(close_g, h)
    fwd_l = logc.shift(-h) - logc

    # per-symbol burn-in: need >= BURN observed daily returns before row t
    valid_ret = logc.diff().notna()
    warm = valid_ret.cumsum() >= BURN

    sig_h = sig1 * math.sqrt(h)
    ok = (sig_h > 0) & sig_h.notna() & fwd_s.notna() & fwd_l.notna() & warm
    ok = ok.to_numpy()
    pos = np.broadcast_to(np.arange(len(close_g))[:, None], ok.shape)[ok]
    sym = np.broadcast_to(np.arange(close_g.shape[1])[None, :], ok.shape)[ok]
    return {
        "pos": pos.astype(np.int32),
        "sym": sym.astype(np.int32),
        "yr": close_g.index.year.to_numpy()[pos].astype(np.int16),
        "sig1": sig1.to_numpy()[ok],
        "sig_h": sig_h.to_numpy()[ok],
        "fwd_l": fwd_l.to_numpy()[ok],
        "fwd_s": fwd_s.to_numpy()[ok],
        "n_sym": close_g.shape[1],
        "index": close_g.index,
    }


# --------------------------------------------------------------------------- probabilities
def probs_from_cdf(F0, Fdn, Fup):
    """Assemble the 4 bucket probabilities from CDF evaluations, then floor+renormalise."""
    p = np.column_stack([Fdn, F0 - Fdn, Fup - F0, 1.0 - Fup])
    raw_min = p.min(axis=1)
    p = np.clip(p, FLOOR, None)
    p /= p.sum(axis=1, keepdims=True)
    return p, raw_min


def probs_empirical(z_train_sorted, s_test, thr):
    n = len(z_train_sorted)
    a_dn, a_up = math.log(1 - thr), math.log(1 + thr)
    Fdn = np.searchsorted(z_train_sorted, a_dn / s_test, side="right") / n
    Fup = np.searchsorted(z_train_sorted, a_up / s_test, side="right") / n
    F0 = np.searchsorted(z_train_sorted, 0.0, side="right") / n
    return probs_from_cdf(np.full_like(s_test, F0), Fdn, Fup)


def probs_normal(mu, sd, s_test, thr):
    a_dn, a_up = math.log(1 - thr), math.log(1 + thr)
    F = lambda x: norm_cdf((x - mu) / sd)
    return probs_from_cdf(np.full_like(s_test, F(0.0)), F(a_dn / s_test), F(a_up / s_test))


def probs_floor(p):
    """Floor+renormalise an already-assembled probability matrix (used by CLIMSYM)."""
    p = np.asarray(p, dtype=float)
    raw_min = p.min(axis=1)
    p = np.clip(p, FLOOR, None)
    p = p / p.sum(axis=1, keepdims=True)
    return p, raw_min


def probs_clim(freq, n_test):
    p = np.clip(np.asarray(freq, dtype=float), FLOOR, None)
    p = p / p.sum()
    return np.tile(p, (n_test, 1)), np.full(n_test, p.min())


# --------------------------------------------------------------------------- walk-forward
def run_group(name, close_g, verbose=True, min_train_rows=5000):
    """Returns dict[(h, thr, model)] -> arrays of OOS probs / actuals / dates, concatenated
    over all test years."""
    out = {}
    for h in HORIZONS:
        t0 = time.time()
        A = flatten(close_g, h)
        idx = A["index"]
        lut = {k: i for i, k in enumerate(L.BUCKETS)}
        actual = {}
        for thr in THRESHOLDS:
            b = L.bucketize(pd.Series(A["fwd_s"]), thr).to_numpy()
            actual[thr] = np.array([lut[x] for x in b], dtype=np.int8)

        store = {(thr, m): {"p": [], "y": [], "pos": [], "sym": [], "rawmin": []}
                 for thr in THRESHOLDS
                 for m in MODELS + ("CLIM", "CLIMSYM")}

        for y, tr_mask_idx, te_mask_idx in L.walk_forward_years(idx, MIN_TRAIN_YEARS):
            last_train_pos = np.flatnonzero(tr_mask_idx)[-1]
            tr = (A["yr"] < y) & (A["pos"] <= last_train_pos - h)      # PURGE h days
            te = A["yr"] == y
            if tr.sum() < min_train_rows or te.sum() == 0:
                continue

            z_tr = A["fwd_l"][tr] / A["sig_h"][tr]
            z_srt = np.sort(z_tr)
            mu_z, sd_z = float(z_tr.mean()), float(z_tr.std(ddof=1))

            # constant-sigma variants: sigma frozen at its TRAIN mean
            csym = np.full(A["n_sym"], np.nan)
            cnt = np.bincount(A["sym"][tr], minlength=A["n_sym"])
            tot = np.bincount(A["sym"][tr], weights=A["sig1"][tr], minlength=A["n_sym"])
            good = cnt > 250
            csym[good] = tot[good] / cnt[good]
            cglob = float(A["sig1"][tr].mean())
            csym = np.where(np.isnan(csym), cglob, csym)

            s_sym_tr = csym[A["sym"][tr]] * math.sqrt(h)
            z_sym_srt = np.sort(A["fwd_l"][tr] / s_sym_tr)
            z_glob_srt = np.sort(A["fwd_l"][tr] / (cglob * math.sqrt(h)))

            s_te = A["sig_h"][te]
            s_sym_te = csym[A["sym"][te]] * math.sqrt(h)
            s_glob_te = np.full(te.sum(), cglob * math.sqrt(h))

            for thr in THRESHOLDS:
                a_tr, a_te = actual[thr][tr], actual[thr][te]
                clim = L.climatology(a_tr.astype(int), k=4)
                # per-symbol climatology: that symbol's own train-year bucket frequencies,
                # shrunk toward the pooled frequencies by 40 pseudo-counts so a symbol with a
                # short train history is not scored against a 200-row base rate.
                cs = np.bincount(A["sym"][tr].astype(np.int64) * 4 + a_tr.astype(np.int64),
                                 minlength=A["n_sym"] * 4).reshape(A["n_sym"], 4).astype(float)
                cs = (cs + 40.0 * clim[None, :])
                cs /= cs.sum(axis=1, keepdims=True)
                cand = {
                    "EMP": probs_empirical(z_srt, s_te, thr),
                    "NORMFIT": probs_normal(mu_z, sd_z, s_te, thr),
                    "NORMSTD": probs_normal(0.0, 1.0, s_te, thr),
                    "CONSTSYM": probs_empirical(z_sym_srt, s_sym_te, thr),
                    "CONSTGLOB": probs_empirical(z_glob_srt, s_glob_te, thr),
                    "CLIM": probs_clim(clim, te.sum()),
                    "CLIMSYM": probs_floor(cs[A["sym"][te]]),
                }
                for m, (p, rm) in cand.items():
                    st = store[(thr, m)]
                    st["p"].append(p.astype(np.float32))
                    st["y"].append(a_te)
                    st["pos"].append(A["pos"][te])
                    st["sym"].append(A["sym"][te])
                    st["rawmin"].append(rm.astype(np.float32))

        for k, st in store.items():
            out[(h,) + k] = {kk: np.concatenate(v) for kk, v in st.items()}
        if verbose:
            n = len(out[(h, THRESHOLDS[0], "EMP")]["y"])
            print(f"  [{name}] h={h:>2}  OOS rows={n:,}  ({time.time()-t0:.1f}s)", flush=True)
    return out, close_g.index


# --------------------------------------------------------------------------- metrics
def metrics(rec, base):
    p, y = rec["p"].astype(float), rec["y"].astype(int)
    pb = base["p"].astype(float)
    ll, bs = L.log_loss(p, y), L.brier_multi(p, y)
    ll0, bs0 = L.log_loss(pb, y), L.brier_multi(pb, y)
    i_dn, i_up = L.BUCKETS.index("down_big"), L.BUCKETS.index("up_big")
    return {
        "n": len(y), "ll": ll, "ll_clim": ll0, "lls": L.skill_score(ll, ll0),
        "bs": bs, "bs_clim": bs0, "bss": L.skill_score(bs, bs0),
        "ece_up": L.ece(p[:, i_up], y == i_up), "ece_dn": L.ece(p[:, i_dn], y == i_dn),
        "mp_up": p[:, i_up].mean(), "ob_up": (y == i_up).mean(),
        "mp_dn": p[:, i_dn].mean(), "ob_dn": (y == i_dn).mean(),
        "floored": float((rec["rawmin"] < FLOOR).mean()),
    }


def per_obs_brier(rec):
    p, y = rec["p"].astype(float), rec["y"].astype(int)
    oh = np.zeros_like(p)
    oh[np.arange(len(y)), y] = 1.0
    return ((p - oh) ** 2).sum(axis=1)


def per_obs_ll(rec, eps=1e-12):
    p, y = rec["p"].astype(float), rec["y"].astype(int)
    return -np.log(np.clip(p[np.arange(len(y)), y], eps, 1.0))


def block_bootstrap_bss(rec_m, rec_b, h, nboot=NBOOT):
    """Date-block bootstrap of the Brier skill score. Blocks = h contiguous trading days;
    every symbol on a sampled date is carried along (cluster by date)."""
    pos = rec_m["pos"]
    bm, bb = per_obs_brier(rec_m), per_obs_brier(rec_b)
    upos = np.unique(pos)
    order = np.argsort(pos, kind="stable")
    pos_s, bm_s, bb_s = pos[order], bm[order], bb[order]
    starts = np.searchsorted(pos_s, upos, side="left")
    ends = np.searchsorted(pos_s, upos, side="right")
    blk = max(1, h)
    nblk = int(np.ceil(len(upos) / blk))
    slices = [(starts[i * blk], ends[min(len(upos), (i + 1) * blk) - 1]) for i in range(nblk)]
    sums_m = np.array([bm_s[a:b].sum() for a, b in slices])
    sums_b = np.array([bb_s[a:b].sum() for a, b in slices])
    cnts = np.array([b - a for a, b in slices], dtype=float)
    outs = []
    for _ in range(nboot):
        pick = RNG.integers(0, nblk, nblk)
        c = cnts[pick].sum()
        if c == 0:
            continue
        outs.append(1.0 - (sums_m[pick].sum() / c) / (sums_b[pick].sum() / c))
    o = np.sort(outs)
    return float(np.mean(o)), float(o[int(0.025 * len(o))]), float(o[int(0.975 * len(o))]), nblk


# --------------------------------------------------------------------------- reporting
def hdr(s):
    print("\n" + "=" * 100)
    print(s)
    print("=" * 100, flush=True)


def main():
    close, idx_syms, singles = load_panel()
    print(f"panel: {close.shape[1]} symbols >= {MIN_COVER} closes, "
          f"{close.index[0].date()} -> {close.index[-1].date()}, {len(close)} rows")
    print(f"indices ({len(idx_syms)}): {idx_syms}")
    print(f"singles ({len(singles)}): {singles[:8]} ...")
    print(f"walk-forward test years: {sorted(set(close.index.year))[MIN_TRAIN_YEARS:]}")

    groups = {"INDEX": close[idx_syms], "SINGLES": close[singles]}
    res = {}
    for g, cg in groups.items():
        print(f"\nrunning {g} ({cg.shape[1]} symbols) ...", flush=True)
        res[g], _ = run_group(g, cg)

    # ---------------- 1/2: headline table
    hdr("TABLE 1 — walk-forward OOS scores, provisional model EMP vs CLIMATOLOGY (train-fit)")
    print(f"{'grp':<8}{'h':>3}{'thr':>6}{'n_rows':>11}{'n_dates':>8}"
          f"{'LL':>8}{'LL_clim':>9}{'LLskill':>9}{'Brier':>8}{'Br_clim':>9}{'BSS':>8}"
          f"{'ECE_up':>8}{'ECE_dn':>8}{'P(up)':>8}{'obs_up':>8}{'P(dn)':>8}{'obs_dn':>8}{'flr%':>7}")
    rows = []
    for g in groups:
        for h in HORIZONS:
            for thr in THRESHOLDS:
                m = metrics(res[g][(h, thr, "EMP")], res[g][(h, thr, "CLIM")])
                nd = len(np.unique(res[g][(h, thr, "EMP")]["pos"]))
                rows.append(dict(grp=g, h=h, thr=thr, n_dates=nd, **m))
                print(f"{g:<8}{h:>3}{thr:>6.0%}{m['n']:>11,}{nd:>8,}"
                      f"{m['ll']:>8.4f}{m['ll_clim']:>9.4f}{m['lls']:>9.4f}"
                      f"{m['bs']:>8.4f}{m['bs_clim']:>9.4f}{m['bss']:>8.4f}"
                      f"{m['ece_up']:>8.4f}{m['ece_dn']:>8.4f}"
                      f"{m['mp_up']:>8.3f}{m['ob_up']:>8.3f}{m['mp_dn']:>8.3f}{m['ob_dn']:>8.3f}"
                      f"{100*m['floored']:>7.2f}", flush=True)
    pd.DataFrame(rows).to_csv(Path(__file__).with_name("_move_wf_baseline_table1.csv"), index=False)

    # ---------------- bootstrap CIs on BSS at 2%
    hdr("TABLE 2 — BSS with date-block bootstrap 95% CI (block = h trading days, cluster by date)")
    print(f"{'grp':<8}{'h':>3}{'thr':>6}{'BSS':>9}{'boot_mean':>11}{'lo95':>9}{'hi95':>9}{'n_blocks':>10}")
    for g in groups:
        for h in HORIZONS:
            for thr in (0.02,):
                rm, rb = res[g][(h, thr, "EMP")], res[g][(h, thr, "CLIM")]
                bss = L.skill_score(L.brier_multi(rm["p"].astype(float), rm["y"].astype(int)),
                                    L.brier_multi(rb["p"].astype(float), rb["y"].astype(int)))
                mu, lo, hi, nb = block_bootstrap_bss(rm, rb, h)
                print(f"{g:<8}{h:>3}{thr:>6.0%}{bss:>9.4f}{mu:>11.4f}{lo:>9.4f}{hi:>9.4f}{nb:>10,}",
                      flush=True)

    # ---------------- reliability tables at 2%
    for g in groups:
        for h in HORIZONS:
            rec = res[g][(h, 0.02, "EMP")]
            p, y = rec["p"].astype(float), rec["y"].astype(int)
            for lbl, i in (("up_big", L.BUCKETS.index("up_big")),
                           ("down_big", L.BUCKETS.index("down_big"))):
                r = L.reliability(p[:, i], y == i, 10)
                hdr(f"RELIABILITY {g} h={h} thr=2% P({lbl})   ECE={L.ece(p[:, i], y == i):.4f}"
                    f"   n={len(y):,}")
                print(r.to_string(float_format=lambda v: f"{v:9.4f}"), flush=True)

    # ---------------- 3: by calendar year
    hdr("TABLE 3 — OOS log loss / BSS by calendar year (thr=2%)")
    for g in groups:
        print(f"\n--- {g} ---")
        print(f"{'year':<6}" + "".join(f"{f'h{h}_LL':>9}{f'h{h}_clim':>9}{f'h{h}_BSS':>9}" for h in HORIZONS) + f"{'n(h5)':>9}")
        yrs = sorted(set(close.index.year))[MIN_TRAIN_YEARS:]
        yearpos = {y: np.flatnonzero(close.index.year == y) for y in yrs}
        for y in yrs:
            line = f"{y:<6}"
            n5 = 0
            for h in HORIZONS:
                rm, rb = res[g][(h, 0.02, "EMP")], res[g][(h, 0.02, "CLIM")]
                sel = np.isin(rm["pos"], yearpos[y])
                if sel.sum() == 0:
                    line += f"{'-':>9}{'-':>9}{'-':>9}"
                    continue
                sub_m = {k: v[sel] for k, v in rm.items()}
                sub_b = {k: v[sel] for k, v in rb.items()}
                ll = L.log_loss(sub_m["p"].astype(float), sub_m["y"].astype(int))
                ll0 = L.log_loss(sub_b["p"].astype(float), sub_b["y"].astype(int))
                bss = L.skill_score(L.brier_multi(sub_m["p"].astype(float), sub_m["y"].astype(int)),
                                    L.brier_multi(sub_b["p"].astype(float), sub_b["y"].astype(int)))
                line += f"{ll:>9.4f}{ll0:>9.4f}{bss:>9.4f}"
                if h == 5:
                    n5 = int(sel.sum())
            print(line + f"{n5:>9,}", flush=True)

    hdr("TABLE 3b — CRISIS YEARS: reliability gap sign for P(up_big)/P(down_big), thr=2%")
    print(f"{'grp':<8}{'year':>6}{'h':>3}{'n':>10}"
          f"{'P(up)':>8}{'obs_up':>8}{'gap_up':>8}{'P(dn)':>8}{'obs_dn':>8}{'gap_dn':>8}"
          f"{'LL':>8}{'LLclim':>8}{'BSS':>8}  verdict")
    for g in groups:
        for y in (2008, 2011, 2020, 2022):
            for h in HORIZONS:
                rm, rb = res[g][(h, 0.02, "EMP")], res[g][(h, 0.02, "CLIM")]
                sel = np.isin(rm["pos"], np.flatnonzero(close.index.year == y))
                if sel.sum() == 0:
                    continue
                p, yy = rm["p"][sel].astype(float), rm["y"][sel].astype(int)
                pb = rb["p"][sel].astype(float)
                iu, idn = L.BUCKETS.index("up_big"), L.BUCKETS.index("down_big")
                gu = (yy == iu).mean() - p[:, iu].mean()
                gd = (yy == idn).mean() - p[:, idn].mean()
                ll, ll0 = L.log_loss(p, yy), L.log_loss(pb, yy)
                bss = L.skill_score(L.brier_multi(p, yy), L.brier_multi(pb, yy))
                v = "UNDER-confident (real moves bigger)" if (gu + gd) > 0.01 else \
                    ("OVER-confident (real moves smaller)" if (gu + gd) < -0.01 else "ok")
                print(f"{g:<8}{y:>6}{h:>3}{sel.sum():>10,}"
                      f"{p[:, iu].mean():>8.3f}{(yy == iu).mean():>8.3f}{gu:>8.3f}"
                      f"{p[:, idn].mean():>8.3f}{(yy == idn).mean():>8.3f}{gd:>8.3f}"
                      f"{ll:>8.4f}{ll0:>8.4f}{bss:>8.4f}  {v}", flush=True)

    # ---------------- 4/5: decomposition + dumb floor
    hdr("TABLE 4 — SKILL DECOMPOSITION (all OOS, BSS vs train-fit climatology)")
    print("EMP=EWMA sigma+empirical z | NORMFIT=EWMA sigma+N(mu,sd) | NORMSTD=EWMA sigma+N(0,1)")
    print("CONSTSYM=frozen per-symbol sigma+empirical z | CONSTGLOB=one frozen sigma (DUMB FLOOR)")
    print(f"\n{'grp':<8}{'h':>3}{'thr':>6}" +
          "".join(f"{m:>11}" for m in ("EMP", "NORMFIT", "NORMSTD", "CONSTSYM", "CONSTGLOB")) +
          f"{'vol_gain':>10}{'shape_gain':>11}{'scale_gain':>11}")
    for g in groups:
        for h in HORIZONS:
            for thr in THRESHOLDS:
                rb = res[g][(h, thr, "CLIM")]
                bss = {}
                for m in ("EMP", "NORMFIT", "NORMSTD", "CONSTSYM", "CONSTGLOB"):
                    rm = res[g][(h, thr, m)]
                    bss[m] = L.skill_score(
                        L.brier_multi(rm["p"].astype(float), rm["y"].astype(int)),
                        L.brier_multi(rb["p"].astype(float), rb["y"].astype(int)))
                print(f"{g:<8}{h:>3}{thr:>6.0%}" + "".join(f"{bss[m]:>11.4f}" for m in
                      ("EMP", "NORMFIT", "NORMSTD", "CONSTSYM", "CONSTGLOB")) +
                      f"{bss['EMP']-bss['CONSTSYM']:>10.4f}"
                      f"{bss['EMP']-bss['NORMFIT']:>11.4f}"
                      f"{bss['NORMFIT']-bss['NORMSTD']:>11.4f}", flush=True)

    hdr("TABLE 4b — same decomposition on LOG-LOSS SKILL")
    print(f"{'grp':<8}{'h':>3}{'thr':>6}" +
          "".join(f"{m:>11}" for m in ("EMP", "NORMFIT", "NORMSTD", "CONSTSYM", "CONSTGLOB")))
    for g in groups:
        for h in HORIZONS:
            for thr in THRESHOLDS:
                rb = res[g][(h, thr, "CLIM")]
                ll0 = L.log_loss(rb["p"].astype(float), rb["y"].astype(int))
                vals = []
                for m in ("EMP", "NORMFIT", "NORMSTD", "CONSTSYM", "CONSTGLOB"):
                    rm = res[g][(h, thr, m)]
                    vals.append(L.skill_score(
                        L.log_loss(rm["p"].astype(float), rm["y"].astype(int)), ll0))
                print(f"{g:<8}{h:>3}{thr:>6.0%}" + "".join(f"{v:>11.4f}" for v in vals), flush=True)

    # ---------------- non-overlapping sanity check
    hdr("TABLE 5 — non-overlapping subsample (every h-th trading day) sanity check, thr=2%")
    print(f"{'grp':<8}{'h':>3}{'n_rows':>10}{'n_dates':>9}{'BSS_all':>10}{'BSS_nonov':>11}"
          f"{'LLskill_nonov':>15}{'ECE_up_nonov':>14}")
    for g in groups:
        for h in HORIZONS:
            rm, rb = res[g][(h, 0.02, "EMP")], res[g][(h, 0.02, "CLIM")]
            bss_all = L.skill_score(L.brier_multi(rm["p"].astype(float), rm["y"].astype(int)),
                                    L.brier_multi(rb["p"].astype(float), rb["y"].astype(int)))
            sel = (rm["pos"] % h) == 0
            p, yy = rm["p"][sel].astype(float), rm["y"][sel].astype(int)
            pb = rb["p"][sel].astype(float)
            iu = L.BUCKETS.index("up_big")
            print(f"{g:<8}{h:>3}{sel.sum():>10,}{len(np.unique(rm['pos'][sel])):>9,}"
                  f"{bss_all:>10.4f}"
                  f"{L.skill_score(L.brier_multi(p, yy), L.brier_multi(pb, yy)):>11.4f}"
                  f"{L.skill_score(L.log_loss(p, yy), L.log_loss(pb, yy)):>15.4f}"
                  f"{L.ece(p[:, iu], yy == iu):>14.4f}", flush=True)

    hdr("TABLE 6 — z-distribution shape, IN-SAMPLE descriptive (all rows, no split)")
    print(f"{'grp':<8}{'h':>3}{'n':>11}{'mean':>8}{'sd':>8}{'skew':>8}{'kurt':>8}"
          f"{'q01':>8}{'q05':>8}{'q50':>8}{'q95':>8}{'q99':>8}")
    for g, cg in groups.items():
        for h in HORIZONS:
            A = flatten(cg, h)
            z = A["fwd_l"] / A["sig_h"]
            s = pd.Series(z)
            print(f"{g:<8}{h:>3}{len(z):>11,}{s.mean():>8.3f}{s.std():>8.3f}"
                  f"{s.skew():>8.3f}{s.kurt():>8.2f}" +
                  "".join(f"{s.quantile(q):>8.3f}" for q in (0.01, 0.05, 0.5, 0.95, 0.99)),
                  flush=True)


if __name__ == "__main__":
    main()
