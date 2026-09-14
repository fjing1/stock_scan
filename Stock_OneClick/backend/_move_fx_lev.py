"""
_move_fx_lev.py — does the LEVERAGE EFFECT / return-sign asymmetry add anything to move_prob?

The shipped sigma_h model is sign-blind: Parkinson range and EWMA of squared returns do not know
whether the move was up or down. This script tests whether adding signed information helps the
PROBABILITIES (BSS2 / ECE), not just the log-vol R-squared.

Families tested
  SCALE side (extra columns in the log-sigma HAR):
    RVCC   close-to-close RV at 5/22/63   -- sign-BLIND control. Any gain here is "cc returns add
                                             info over Parkinson", NOT leverage. Must be netted out.
    SEMI   log RS_down + log RS_up at 5/22/63      (Barndorff-Nielsen/Patton-Sheppard semivariance)
    SHARE  log RV_cc + downside variance share     (same span, better conditioned, no zero floor)
    GJR    log RS_down at 5/22 + signed r_5, r_22
    SIGN   standardized signed r_5, r_22 only
    ALL    SEMI + standardized signed returns
  SHAPE side (conditioning the empirical z table):
    Z3     one z table per tercile of the standardized trailing 5-day return (train-fitted cuts)
    Z3R    same, terciles of the RAW trailing 5-day return
    Z2     sign split only (r_5 < 0 vs >= 0)

Measurement is a faithful re-implementation of _move_validate.py (expanding walk-forward by
calendar year, >=5 training years, production kappa from 5 inner folds, per-symbol base rate shrunk
40 pseudo-counts toward pooled), restructured so every variant sees IDENTICAL test rows and the
OLS is solved from per-year Gram matrices (any column subset is then free).

Run:  ../../vcp_env/bin/python _move_fx_lev.py repro|scale|shape|coef|all
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _move_data as D          # noqa: E402
import _move_lib as L           # noqa: E402
import move_prob as M           # noqa: E402

THRS = (0.02, 0.05)
MIN_TRAIN_YEARS = 5
INNER_FOLDS = 5
SHRINK = 40.0
VOL_CLIP = M.VOL_CLIP
BASE = ["rv_d", "rv_w", "rv_m", "rv_q", "ewma97"]

EXTRA = (["rvcc_w", "rvcc_m", "rvcc_q", "rsdn_w", "rsup_w", "rsdn_m", "rsup_m",
          "rsdn_q", "rsup_q", "dsh_w", "dsh_m", "dsh_q", "r5", "r22", "r5z", "r22z"])

SPECS = [
    ("BASE",   [], None),
    ("RVCC",   ["rvcc_w", "rvcc_m", "rvcc_q"], None),
    ("SEMI",   ["rsdn_w", "rsup_w", "rsdn_m", "rsup_m", "rsdn_q", "rsup_q"], None),
    ("SHARE",  ["rvcc_w", "rvcc_m", "rvcc_q", "dsh_w", "dsh_m", "dsh_q"], None),
    ("GJR",    ["rsdn_w", "rsdn_m", "r5", "r22"], None),
    ("SIGN",   ["r5z", "r22z"], None),
    ("ALL",    ["rsdn_w", "rsup_w", "rsdn_m", "rsup_m", "rsdn_q", "rsup_q", "r5z", "r22z"], None),
    ("BASE_Z3",  [], "z3"),
    ("BASE_Z3R", [], "z3raw"),
    ("BASE_Z2",  [], "z2"),
]


def clog(x):
    return np.log(np.clip(x, *VOL_CLIP))


# ---------------------------------------------------------------- features
def sym_features(c, hi, lo, vix):
    """Baseline block (identical to M.build_features) + the leverage candidates."""
    f = pd.DataFrame(index=c.index)
    rv1 = L.vol_parkinson(hi, lo, 1)
    f["rv_d"] = clog(rv1)
    f["rv_w"] = clog(rv1.rolling(5).mean())
    f["rv_m"] = clog(rv1.rolling(22).mean())
    f["rv_q"] = clog(rv1.rolling(63).mean())
    ew = L.vol_ewma(c, 0.97)
    f["ewma97"] = clog(ew)
    if vix is not None:
        f["logvix"] = np.log(np.clip(vix.reindex(c.index).ffill(limit=3) / 100 / np.sqrt(252),
                                     *VOL_CLIP))
    r = np.log(c).diff()
    neg, pos = r.clip(upper=0.0), r.clip(lower=0.0)
    for n, tag in ((5, "w"), (22, "m"), (63, "q")):
        # min_periods = n-1 because r[0] is NaN by construction: without it the return-based
        # 63-day window starts one bar LATER than the Parkinson rv_q and silently drops one row
        # per symbol, which is enough to move the index betas in the 3rd decimal.
        v2 = r.pow(2).rolling(n, min_periods=n - 1).mean()
        d2 = neg.pow(2).rolling(n, min_periods=n - 1).mean()
        u2 = pos.pow(2).rolling(n, min_periods=n - 1).mean()
        f["rvcc_" + tag] = clog(np.sqrt(v2))
        f["rsdn_" + tag] = clog(np.sqrt(d2))
        f["rsup_" + tag] = clog(np.sqrt(u2))
        sh = d2 / v2
        sh[v2 == 0] = 0.5                      # all-zero-return window: neutral, keep NaN elsewhere
        f["dsh_" + tag] = sh - 0.5             # centred downside share of variance
    ewc = np.clip(ew, *VOL_CLIP)
    f["r5"] = np.log(c / c.shift(5))
    f["r22"] = np.log(c / c.shift(22))
    f["r5z"] = (f["r5"] / (ewc * np.sqrt(5))).clip(-6, 6)
    f["r22z"] = (f["r22"] / (ewc * np.sqrt(22))).clip(-6, 6)
    return f

def build_cache(panel):
    """Per (group, h): X (intercept + all candidate cols), lr, tgt, yr, sid, pos(date index)."""
    C, H, Lo = panel["Close"], panel["High"], panel["Low"]
    vix = C["^VIX"] if "^VIX" in C.columns else None
    master = C.index
    mpos = pd.Series(np.arange(len(master)), index=master)
    usable = [s for s in C.columns if s != "^VIX" and C[s].notna().sum() >= 500]
    groups = {"index": [s for s in usable if s in M.INDEX_LIKE],
              "single": [s for s in usable if s not in M.INDEX_LIKE]}
    out = {}
    for g, syms in groups.items():
        cols = BASE + (["logvix"] if (g == "index" and vix is not None) else []) + EXTRA
        per_sym = {}
        for s in syms:
            c = C[s].dropna()
            if len(c) < 400:
                continue
            f = sym_features(c, H[s].reindex(c.index), Lo[s].reindex(c.index),
                             vix if g == "index" else None)
            per_sym[s] = (c, f[cols].to_numpy(dtype=float))
        keep_syms = list(per_sym)
        for h in M.HORIZONS:
            X, lr, tg, yr, sid, pos = [], [], [], [], [], []
            for i, s in enumerate(keep_syms):
                c, Fi = per_sym[s]
                Xi = np.column_stack([np.ones(len(c)), Fi])
                li = np.log(c.shift(-h) / c).values
                ti = clog(L.realized_vol_forward(c, h)).values
                ok = np.isfinite(Xi).all(axis=1) & np.isfinite(li)
                X.append(Xi[ok]); lr.append(li[ok]); tg.append(ti[ok])
                yr.append(c.index.year.values[ok]); sid.append(np.full(int(ok.sum()), i))
                pos.append(mpos.reindex(c.index).to_numpy()[ok])
            out[(g, h)] = dict(X=np.vstack(X), lr=np.concatenate(lr), tgt=np.concatenate(tg),
                               yr=np.concatenate(yr), sid=np.concatenate(sid),
                               pos=np.concatenate(pos), cols=["_int"] + cols,
                               n_sym=len(keep_syms), symbols=keep_syms)
            print(f"  cached {g:<7} h={h:<3} rows={len(out[(g, h)]['lr']):>9,} "
                  f"cols={len(cols)}", flush=True)
        del per_sym
    return out


# ---------------------------------------------------------------- fitting helpers
def year_grams(X, tgt, yr):
    """Per-year X'X and X'tgt over rows with finite tgt. Subsets are then free."""
    fin = np.isfinite(tgt)
    Xv, tv, yv = X[fin], tgt[fin], yr[fin]
    G, c, n = {}, {}, {}
    for y in np.unique(yv):
        m = yv == y
        Xm = Xv[m]
        G[y] = Xm.T @ Xm
        c[y] = Xm.T @ tv[m]
        n[y] = int(m.sum())
    return G, c, n


def solve_sub(G, c, sub):
    A, b = G[np.ix_(sub, sub)], c[sub]
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def probs_from_z(z_sorted, sig, thr, f0=None):
    n = len(z_sorted)
    a_dn, a_up = np.log(1 - thr), np.log(1 + thr)
    f_dn = np.searchsorted(z_sorted, a_dn / sig, side="right") / n
    f_up = np.searchsorted(z_sorted, a_up / sig, side="right") / n
    f_0 = (np.searchsorted(z_sorted, 0.0, side="right") / n) if f0 is None else f0
    p = np.column_stack([f_dn, f_0 - f_dn, f_up - f_0, 1 - f_up])
    p = np.clip(p, 1e-4, None)
    return p / p.sum(axis=1, keepdims=True)


def bucket_idx(lr, thr):
    a_dn, a_up = np.log(1 - thr), np.log(1 + thr)
    return np.where(lr <= a_dn, 0, np.where(lr <= 0, 1, np.where(lr < a_up, 2, 3)))


# ---------------------------------------------------------------- the walk-forward
def run_cell(d, specs, want_coef=False, want_zdiag=False):
    """One (group, h) cell. Returns per-variant pm/pdn arrays on a common test-row ordering."""
    X, lr, tgt, yr, sid, pos = d["X"], d["lr"], d["tgt"], d["yr"], d["sid"], d["pos"]
    cols = d["cols"]
    ci = {c: i for i, c in enumerate(cols)}
    base_sub = [0] + [ci[c] for c in BASE] + ([ci["logvix"]] if "logvix" in ci else [])
    h = d["h"]
    G, cvec, _ = year_grams(X, tgt, yr)
    years = sorted(set(yr.tolist()))
    test_years = years[MIN_TRAIN_YEARS:]

    scale_groups = {}
    for name, extra, shape in specs:
        scale_groups.setdefault(tuple(extra), []).append((name, shape))

    act = {thr: bucket_idx(lr, thr) for thr in THRS}
    store = {(nm, thr): {"pm": [], "pdn": []} for nm, _, _ in specs for thr in THRS}
    ref = {thr: {"pm": [], "pdn": []} for thr in THRS}
    meta = {"y": {thr: [] for thr in THRS}, "sym": [], "pos": [], "yr": []}
    coefs, zdiag = [], []

    cum = {}
    for ty in test_years:
        pri = [y for y in years if y < ty]
        cum[ty] = (sum(G[y] for y in pri), sum(cvec[y] for y in pri))

    for ty in test_years:
        tr, te = yr < ty, yr == ty
        if tr.sum() < 2000 or te.sum() < 50:
            continue
        prior = [y for y in years if y < ty]       # MUST be recomputed here: reusing the
        # pre-loop value silently pins every year's inner kappa folds to the final year (lookahead)
        Gc, cc = cum[ty]
        te_i = np.flatnonzero(te)
        tr_i = np.flatnonzero(tr)
        meta["sym"].append(sid[te_i]); meta["pos"].append(pos[te_i])
        meta["yr"].append(np.full(len(te_i), ty))
        for thr in THRS:
            meta["y"][thr].append(act[thr][te_i])

        # ---- per-symbol climatology reference (identical for every variant)
        n_sym = d["n_sym"]
        for thr in THRS:
            a_tr, s_te = act[thr][tr_i], sid[te_i]
            pc = np.bincount(a_tr, minlength=4) / len(a_tr)
            cs = np.bincount(sid[tr_i].astype(np.int64) * 4 + a_tr.astype(np.int64),
                             minlength=n_sym * 4).reshape(n_sym, 4).astype(float)
            cs = (cs + SHRINK * pc[None, :]) / (cs.sum(axis=1, keepdims=True) + SHRINK)
            ps = cs[s_te]
            ref[thr]["pm"].append((ps[:, 0] + ps[:, 3]).astype(np.float32))
            ref[thr]["pdn"].append(ps[:, 0].astype(np.float32))

        p_all = X.shape[1]
        for extra, members in scale_groups.items():
            sub = base_sub + [ci[c] for c in extra]
            beta = solve_sub(Gc, cc, sub)
            bf = np.zeros(p_all); bf[sub] = beta
            lin = X @ bf                                   # = log(sigma_daily) for every row
            sig_tr = np.exp(lin[tr_i]) * np.sqrt(h)
            sig_te = np.exp(lin[te_i]) * np.sqrt(h)

            # production kappa: median over INNER_FOLDS expanding inner folds inside train
            ks = []
            for iy in prior[-INNER_FOLDS:]:
                itr, iva = np.flatnonzero(yr < iy), np.flatnonzero(yr == iy)
                if len(itr) < 5000 or len(iva) < 50:
                    continue
                Gi = sum(G[y] for y in years if y < iy)
                cig = sum(cvec[y] for y in years if y < iy)
                bi = solve_sub(Gi, cig, sub)
                bfi = np.zeros(p_all); bfi[sub] = bi
                lini = X @ bfi
                zi = np.sort(lr[itr] / (np.exp(lini[itr]) * np.sqrt(h)))
                if len(zi) < 5000:
                    continue
                ks.append(M._calibrate_kappa(zi, np.exp(lini[iva]) * np.sqrt(h), lr[iva]))
            kappa = float(np.median(ks)) if ks else 1.0

            z_all = lr[tr_i] / sig_tr
            z_pool = np.sort(z_all) * kappa

            if want_coef:
                fin_tr = np.isfinite(tgt[tr_i])
                yy = tgt[tr_i][fin_tr]
                pred = lin[tr_i][fin_tr]
                r2 = 1 - ((yy - pred) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
                coefs.append(dict(ty=ty, var=("+".join(extra) if extra else "BASE"),
                                  kappa=kappa, r2_train=r2,
                                  **{c: float(b) for c, b in zip([cols[j] for j in sub], beta)}))

            for nm, shape in members:
                if shape is None:
                    for thr in THRS:
                        p = probs_from_z(z_pool, sig_te, thr)
                        store[(nm, thr)]["pm"].append((p[:, 0] + p[:, 3]).astype(np.float32))
                        store[(nm, thr)]["pdn"].append(p[:, 0].astype(np.float32))
                    continue
                # ---- shape conditioning on the trailing 5-day return
                key = "r5z" if shape in ("z3", "z2") else "r5"
                v_tr, v_te = X[tr_i, ci[key]], X[te_i, ci[key]]
                if shape == "z2":
                    cuts = [0.0]
                else:
                    cuts = list(np.quantile(v_tr, [1 / 3, 2 / 3]))
                b_tr, b_te = np.digitize(v_tr, cuts), np.digitize(v_te, cuts)
                P = {thr: np.empty((len(te_i), 4)) for thr in THRS}
                for b in range(len(cuts) + 1):
                    mt = b_te == b
                    if not mt.any():
                        continue
                    zz = z_all[b_tr == b]
                    zb = np.sort(zz) * kappa if len(zz) >= 5000 else z_pool
                    if want_zdiag:
                        zdiag.append(dict(ty=ty, mode=shape, bin=b, n=len(zz),
                                          p_lo=float((zb < -2).mean()), p_hi=float((zb > 2).mean()),
                                          med=float(np.median(zb)), sd=float(zb.std())))
                    for thr in THRS:
                        P[thr][mt] = probs_from_z(zb, sig_te[mt], thr)
                for thr in THRS:
                    p = P[thr]
                    store[(nm, thr)]["pm"].append((p[:, 0] + p[:, 3]).astype(np.float32))
                    store[(nm, thr)]["pdn"].append(p[:, 0].astype(np.float32))

    out = {"sym": np.concatenate(meta["sym"]), "pos": np.concatenate(meta["pos"]),
           "yr": np.concatenate(meta["yr"]),
           "y": {thr: np.concatenate(meta["y"][thr]) for thr in THRS},
           "ref": {thr: {k: np.concatenate(v) for k, v in ref[thr].items()} for thr in THRS},
           "var": {k: {kk: np.concatenate(vv) for kk, vv in v.items()} for k, v in store.items()},
           "coefs": pd.DataFrame(coefs) if coefs else pd.DataFrame(),
           "zdiag": pd.DataFrame(zdiag) if zdiag else pd.DataFrame()}
    return out


# ---------------------------------------------------------------- metrics
def metrics(R, nm, thr):
    y = R["y"][thr]
    big = np.isin(y, [0, 3]).astype(float)
    dn = (y == 0).astype(float)
    pm = R["var"][(nm, thr)]["pm"].astype(float)
    pdn = R["var"][(nm, thr)]["pdn"].astype(float)
    rm = R["ref"][thr]["pm"].astype(float)
    rdn = R["ref"][thr]["pdn"].astype(float)
    bss2 = 1 - ((pm - big) ** 2).mean() / ((rm - big) ** 2).mean()
    yrs = np.unique(R["yr"])
    per_yr = {}
    for u in yrs:
        m = R["yr"] == u
        per_yr[u] = 1 - ((pm[m] - big[m]) ** 2).mean() / ((rm[m] - big[m]) ** 2).mean()
    return dict(n=len(y), bss2=bss2, ece=L.ece(pm, big.astype(bool)),
                ece_dn=L.ece(pdn, dn.astype(bool)),
                ece_ref=L.ece(rm, big.astype(bool)), ece_dn_ref=L.ece(rdn, dn.astype(bool)),
                obs=big.mean(), pred=pm.mean(), obs_dn=dn.mean(), pred_dn=pdn.mean(),
                per_yr=per_yr)


def block_boot_delta(R, nm, thr, h, nboot=600, seed=7):
    """Date-block bootstrap of DELTA BSS2 (variant - BASE). Exact for the ratio-of-sums form:
    delta = (SSE_base - SSE_var) / SSE_ref, all three summed over the same resampled dates."""
    y = R["y"][thr]
    big = np.isin(y, [0, 3]).astype(float)
    e_v = (R["var"][(nm, thr)]["pm"].astype(float) - big) ** 2
    e_b = (R["var"][("BASE", thr)]["pm"].astype(float) - big) ** 2
    e_r = (R["ref"][thr]["pm"].astype(float) - big) ** 2
    pos = R["pos"]
    order = np.argsort(pos, kind="stable")
    ps = pos[order]
    dates, start = np.unique(ps, return_index=True)
    Sv = np.add.reduceat(e_v[order], start)
    Sb = np.add.reduceat(e_b[order], start)
    Sr = np.add.reduceat(e_r[order], start)
    D = len(dates)
    Lb = max(21, 3 * h)
    nblk = int(np.ceil(D / Lb))
    rng = np.random.default_rng(seed)
    cv, cb, cr = np.r_[0, np.cumsum(Sv)], np.r_[0, np.cumsum(Sb)], np.r_[0, np.cumsum(Sr)]
    st = rng.integers(0, max(D - Lb, 1), size=(nboot, nblk))
    en = np.minimum(st + Lb, D)
    tv = (cv[en] - cv[st]).sum(axis=1)
    tb = (cb[en] - cb[st]).sum(axis=1)
    tr = (cr[en] - cr[st]).sum(axis=1)
    dl = (tb - tv) / tr
    return float(np.percentile(dl, 2.5)), float(np.percentile(dl, 97.5))


# ---------------------------------------------------------------- drivers
def hdr(s):
    print("\n" + "=" * 118)
    print(s)
    print("=" * 118, flush=True)


REF_PUB = {("index", 1): .192, ("index", 5): .133, ("index", 10): .092, ("index", 21): .034,
           ("single", 1): .114, ("single", 5): .055, ("single", 10): .033, ("single", 21): .018}
REF_ECE = {("index", 1): .007, ("index", 5): .016, ("index", 10): .018, ("index", 21): .026,
           ("single", 1): .013, ("single", 5): .007, ("single", 10): .007, ("single", 21): .010}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    panel = D.load()
    print("building feature cache ...", flush=True)
    cache = build_cache(panel)
    for k in cache:
        cache[k]["h"] = k[1]
    print(f"cache built in {time.time()-t0:.0f}s", flush=True)

    if mode == "repro":
        specs = [("BASE", [], None)]
    elif mode == "scale":
        specs = [s for s in SPECS if s[2] is None]
    elif mode == "shape":
        specs = [s for s in SPECS if s[2] is None and s[0] == "BASE"] + \
                [s for s in SPECS if s[2] is not None]
    else:
        specs = SPECS

    res = {}
    for g in ("index", "single"):
        for h in M.HORIZONS:
            t1 = time.time()
            R = run_cell(cache[(g, h)], specs, want_coef=(mode in ("coef", "all")),
                         want_zdiag=(mode in ("shape", "all")))
            res[(g, h)] = R
            print(f"  ran {g:<7} h={h:<3} in {time.time()-t1:.0f}s", flush=True)
            cache[(g, h)] = None

    pd.to_pickle({k: {kk: vv for kk, vv in v.items() if kk in ("coefs", "zdiag")}
                  for k, v in res.items()}, "_move_fx_lev_diag.pkl")

    # ---------------- repro check
    hdr("R. BASELINE REPRODUCTION (must match published within ~0.002)")
    print(f"{'grp':<8}{'h':>3}{'thr':>6}{'n':>10}{'BSS2':>9}{'pub':>8}{'d':>8}"
          f"{'ECE':>8}{'pubECE':>8}{'d':>8}{'yrs':>5}")
    for g in ("index", "single"):
        for h in M.HORIZONS:
            for thr in THRS:
                m = metrics(res[(g, h)], "BASE", thr)
                pb = REF_PUB[(g, h)] if thr == 0.02 else np.nan
                pe = REF_ECE[(g, h)] if thr == 0.02 else np.nan
                print(f"{g:<8}{h:>3}{thr:>6.0%}{m['n']:>10,}{m['bss2']:>9.4f}{pb:>8.3f}"
                      f"{m['bss2']-pb:>+8.4f}{m['ece']:>8.4f}{pe:>8.3f}{m['ece']-pe:>+8.4f}"
                      f"{len(m['per_yr']):>5}")

    names = [s[0] for s in specs if s[0] != "BASE"]
    if names:
        hdr("D. DELTA vs BASE — BSS2 and ECE on P(|move|>=thr), plus ECE on P(down_big)")
        print(f"{'grp':<7}{'h':>3}{'thr':>5}{'variant':>10}{'n':>9}"
              f"{'BSS2':>8}{'dBSS2':>9}{'ci95':>19}{'yrs+':>6}"
              f"{'ECE':>8}{'dECE':>9}{'ECEdn':>8}{'dECEdn':>9}")
        rows = []
        for g in ("index", "single"):
            for h in M.HORIZONS:
                R = res[(g, h)]
                for thr in THRS:
                    b = metrics(R, "BASE", thr)
                    for nm in names:
                        m = metrics(R, nm, thr)
                        wins = sum(m["per_yr"][u] > b["per_yr"][u] for u in b["per_yr"])
                        lo, hi = block_boot_delta(R, nm, thr, h)
                        rows.append(dict(grp=g, h=h, thr=thr, var=nm, n=m["n"],
                                         bss2=m["bss2"], d_bss2=m["bss2"] - b["bss2"],
                                         lo=lo, hi=hi, wins=wins, nyr=len(b["per_yr"]),
                                         ece=m["ece"], d_ece=m["ece"] - b["ece"],
                                         ece_dn=m["ece_dn"], d_ece_dn=m["ece_dn"] - b["ece_dn"],
                                         pred=m["pred"], obs=m["obs"],
                                         pred_dn=m["pred_dn"], obs_dn=m["obs_dn"]))
                        print(f"{g:<7}{h:>3}{thr:>5.0%}{nm:>10}{m['n']:>9,}"
                              f"{m['bss2']:>8.4f}{rows[-1]['d_bss2']:>+9.4f}"
                              f"  [{lo:>+7.4f},{hi:>+7.4f}]{wins:>4}/{len(b['per_yr'])}"
                              f"{m['ece']:>8.4f}{rows[-1]['d_ece']:>+9.4f}"
                              f"{m['ece_dn']:>8.4f}{rows[-1]['d_ece_dn']:>+9.4f}", flush=True)
        df = pd.DataFrame(rows)
        df.to_csv("_move_fx_lev_delta.csv", index=False)
        hdr("S. SUMMARY — mean delta BSS2 / delta ECE by variant (averaged over the 8 cells)")
        for thr in THRS:
            s = df[df.thr == thr].groupby(["grp", "var"]).agg(
                dBSS2=("d_bss2", "mean"), dECE=("d_ece", "mean"), dECEdn=("d_ece_dn", "mean"),
                wins=("wins", "sum"), nyr=("nyr", "sum"))
            print(f"\n  thr={thr:.0%}")
            print(s.to_string(float_format=lambda v: f"{v:+.4f}"))

    if mode in ("shape", "all"):
        hdr("Z. z-TABLE SHAPE BY TRAILING-5d-RETURN TERCILE (train tables, last walk-forward fit)")
        dg = pd.read_pickle("_move_fx_lev_diag.pkl")
        for g in ("index", "single"):
            for h in M.HORIZONS:
                z = dg[(g, h)]["zdiag"]
                if z.empty:
                    continue
                z = z[(z["mode"] == "z3") & (z.ty == z.ty.max())]
                if z.empty:
                    continue
                r = z.set_index("bin")
                print(f"  {g:<7} h={h:<3} " + "  ".join(
                    f"b{int(b)}: n={int(r.n[b]):>7,} P(z<-2)={r.p_lo[b]:.4f} "
                    f"P(z>2)={r.p_hi[b]:.4f} ratio={r.p_lo[b]/max(r.p_hi[b],1e-9):5.2f} "
                    f"med={r.med[b]:+.3f}" for b in r.index))

    if mode in ("coef", "all"):
        hdr("C. HAR COEFFICIENTS, mean over walk-forward TRAIN fits (IN-SAMPLE-ON-TRAIN)")
        dg = pd.read_pickle("_move_fx_lev_diag.pkl")
        for g in ("index", "single"):
            for h in M.HORIZONS:
                c = dg[(g, h)]["coefs"]
                if c.empty:
                    continue
                mm = c.groupby("var").mean(numeric_only=True).drop(columns=["ty"])
                print(f"\n[{g} h={h}]")
                print(mm.to_string(float_format=lambda v: f"{v:+.4f}"))

    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
