"""
_move_spec_prob.py -- end-to-end walk-forward probability calibration for the chosen spec.

Chosen vol spec (from _move_spec_grid.py):
  index  : const + rv_d + rv_w + rv_m + yz21 + logvix          ("har3+yz21+vix")
  single : const + rv_d + rv_w + rv_m + rv_q + yz21 + e97 + logvix ("har4+yz21+e97+vix")

z layer variants raced here:
  emp        - one pooled empirical z CDF per (group, horizon), log space
  normLS     - normal with train loc/scale
  volq_ls    - pooled shape, loc+scale per sigma_hat quintile (edges from train)
  volq_full  - separate empirical z CDF per sigma_hat quintile
Baselines: pooled climatology, per-symbol climatology (40 pseudo-count shrink).
"""
from __future__ import annotations

import math
import time
import numpy as np
import pandas as pd

HORIZONS = (1, 5, 10, 21)
THRS = (0.01, 0.02, 0.03, 0.05)
MIN_TRAIN_YEARS = 5
SHRINK = 40.0
COLS = {"index": [0, 1, 2, 4, 6], "single": [0, 1, 2, 3, 4, 5, 6]}
t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)


def design(F, cols):
    return np.column_stack([np.ones(len(F))] + [F[:, c] for c in cols])


def ols(X, y):
    G = X.T @ X
    return np.linalg.solve(G + 1e-10 * np.eye(G.shape[0]), X.T @ y)


def norm_cdf(x):
    return 0.5 * np.erfc(-x / np.sqrt(2)) if hasattr(np, "erfc") else \
        0.5 * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2)))


_erf = np.vectorize(math.erf)


def ncdf(x):
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / np.sqrt(2)))


def bucket_idx(lr, thr):
    a_dn, a_up = np.log(1 - thr), np.log(1 + thr)
    return np.where(lr <= a_dn, 0, np.where(lr <= 0, 1, np.where(lr < a_up, 2, 3))).astype(int)


def norm4(p):
    p = np.clip(p, 1e-4, None)
    return p / p.sum(axis=1, keepdims=True)


def probs_emp(zs, sig, thr):
    a_dn, a_up = np.log(1 - thr), np.log(1 + thr)
    n = len(zs)
    F_dn = np.searchsorted(zs, a_dn / sig, side="right") / n
    F_0 = np.searchsorted(zs, 0.0, side="right") / n
    F_up = np.searchsorted(zs, a_up / sig, side="right") / n
    return norm4(np.column_stack([F_dn, F_0 - F_dn, F_up - F_0, 1 - F_up]))


def probs_norm(loc, scale, sig, thr):
    a_dn, a_up = np.log(1 - thr), np.log(1 + thr)
    F_dn = ncdf((a_dn / sig - loc) / scale)
    F_0 = ncdf((0.0 - loc) / scale)
    F_up = ncdf((a_up / sig - loc) / scale)
    return norm4(np.column_stack([F_dn, F_0 - F_dn, F_up - F_0, 1 - F_up]))


def probs_emp_ls(zs0, loc, scale, sig, thr):
    """pooled standardized shape zs0 (already median-0, iqr-scale-1), shifted/scaled per row."""
    a_dn, a_up = np.log(1 - thr), np.log(1 + thr)
    n = len(zs0)
    F_dn = np.searchsorted(zs0, (a_dn / sig - loc) / scale, side="right") / n
    F_0 = np.searchsorted(zs0, (0.0 - loc) / scale, side="right") / n
    F_up = np.searchsorted(zs0, (a_up / sig - loc) / scale, side="right") / n
    return norm4(np.column_stack([F_dn, F_0 - F_dn, F_up - F_0, 1 - F_up]))


def brier(p, a):
    oh = np.zeros_like(p)
    oh[np.arange(len(a)), a] = 1.0
    return float(((p - oh) ** 2).sum(axis=1).mean())


def logloss(p, a):
    return float(-np.log(np.clip(p[np.arange(len(a)), a], 1e-12, 1)).mean())


def ece_b(pb, hit, nb=10):
    df = pd.DataFrame({"p": pb, "y": hit.astype(float)})
    df["bin"] = pd.qcut(df.p.rank(method="first"), nb, labels=False, duplicates="drop")
    gg = df.groupby("bin").agg(n=("y", "size"), mp=("p", "mean"), ob=("y", "mean"))
    return float((gg.n / gg.n.sum() * (gg.ob - gg.mp).abs()).sum())


def end_year(dt, sid, h):
    """Calendar year of the bar h rows later within the same symbol block."""
    out = np.empty(len(dt), dtype=np.int64)
    yrs = pd.DatetimeIndex(dt).year.values
    start = 0
    for i in range(1, len(sid) + 1):
        if i == len(sid) or sid[i] != sid[start]:
            blk = slice(start, i)
            y = yrs[blk]
            e = np.empty(len(y), dtype=np.int64)
            if len(y) > h:
                e[:-h] = y[h:]
                e[-h:] = y[-1]
            else:
                e[:] = y[-1]
            out[blk] = e
            start = i
    return out


def run(cache, group, hz, thr_list=THRS, recent_only=False, collect=None):
    d = cache[(group, hz)]
    F, lr, tg, yr, sid, dt = d["F"], d["lr"], d["tg"], d["yr"], d["sid"], d["dt"]
    X = design(F, COLS[group])
    ey = end_year(dt, sid, hz)
    years = sorted(set(yr))
    test_years = years[MIN_TRAIN_YEARS:]
    if recent_only:
        test_years = [y for y in test_years if y >= 2021]

    res = {t: {m: dict(ll=0.0, bs=0.0, bs2=0.0, n=0) for m in
               ("clim", "climsym", "emp", "normLS", "volq_ls", "volq_full")}
           for t in thr_list}
    acc = {t: {m: [] for m in ("clim", "climsym", "emp", "normLS", "volq_ls", "volq_full")}
           for t in thr_list}
    acc_a = {t: [] for t in thr_list}
    acc_sid, acc_yr, acc_sig = [], [], []
    peryear = {t: [] for t in thr_list}

    for yt in test_years:
        tr = (yr < yt) & (ey < yt)          # PURGED: forward window must not reach test year
        te = yr == yt
        if tr.sum() < 2000 or te.sum() == 0:
            continue
        b = ols(X[tr], tg[tr])
        sig_tr = np.exp(X[tr] @ b) * np.sqrt(hz)
        sig_te = np.exp(X[te] @ b) * np.sqrt(hz)
        z_tr = lr[tr] / sig_tr
        zs = np.sort(z_tr)
        loc = float(np.median(z_tr))
        q25, q75 = np.percentile(z_tr, [25, 75])
        scale = float((q75 - q25) / 1.349)
        zs0 = np.sort((z_tr - loc) / scale)

        # sigma quintiles (edges from train sigma_hat daily, absolute)
        sd_tr = sig_tr / np.sqrt(hz)
        edges = np.percentile(sd_tr, [20, 40, 60, 80])
        qtr = np.digitize(sd_tr, edges)
        qte = np.digitize(sig_te / np.sqrt(hz), edges)
        qloc = np.array([np.median(z_tr[qtr == q]) for q in range(5)])
        qsc = np.array([(np.percentile(z_tr[qtr == q], 75) -
                         np.percentile(z_tr[qtr == q], 25)) / 1.349 for q in range(5)])
        qz = [np.sort(z_tr[qtr == q]) for q in range(5)]

        sid_te = sid[te]
        for thr in thr_list:
            a_tr = bucket_idx(lr[tr], thr)
            a_te = bucket_idx(lr[te], thr)
            pooled = np.bincount(a_tr, minlength=4) / len(a_tr)
            P = {}
            P["clim"] = np.tile(pooled, (len(a_te), 1))
            # per-symbol climatology, shrunk
            nsym = int(sid.max()) + 1
            cnt = np.zeros((nsym, 4))
            np.add.at(cnt, (sid[tr], a_tr), 1.0)
            tot = cnt.sum(axis=1, keepdims=True)
            psym = (cnt + SHRINK * pooled) / (tot + SHRINK)
            P["climsym"] = psym[sid_te]
            P["emp"] = probs_emp(zs, sig_te, thr)
            P["normLS"] = probs_norm(loc, scale, sig_te, thr)
            P["volq_ls"] = probs_emp_ls(zs0, qloc[qte], qsc[qte], sig_te, thr)
            pv = np.zeros((len(a_te), 4))
            for q in range(5):
                m = qte == q
                if m.any():
                    pv[m] = probs_emp(qz[q], sig_te[m], thr)
            P["volq_full"] = pv
            for m, p in P.items():
                res[thr][m]["ll"] += logloss(p, a_te) * len(a_te)
                res[thr][m]["bs"] += brier(p, a_te) * len(a_te)
                pm = p[:, 0] + p[:, 3]
                hit = ((a_te == 0) | (a_te == 3)).astype(float)
                res[thr][m]["bs2"] += float(((pm - hit) ** 2).mean()) * len(a_te)
                res[thr][m]["n"] += len(a_te)
                acc[thr][m].append(p)
            acc_a[thr].append(a_te)
            peryear[thr].append((yt, {m: (logloss(P[m], a_te), brier(P[m], a_te))
                                      for m in P}))
        acc_sid.append(sid_te); acc_yr.append(np.full(te.sum(), yt)); acc_sig.append(sig_te)

    out = dict(res=res, peryear=peryear)
    if collect is not None:
        out["acc"] = {t: {m: np.vstack(v) for m, v in acc[t].items()} for t in thr_list}
        out["a"] = {t: np.concatenate(acc_a[t]) for t in thr_list}
        out["sid"] = np.concatenate(acc_sid)
        out["yr"] = np.concatenate(acc_yr)
        out["sig"] = np.concatenate(acc_sig)
    return out


def main():
    cache = pd.read_pickle("_move_spec_cache.pkl")
    log("cache loaded")
    allout = {}
    for group in ("index", "single"):
        for hz in HORIZONS:
            o = run(cache, group, hz, collect=True)
            allout[(group, hz)] = o
            log(f"done {group} h={hz}")
    pd.to_pickle(allout, "_move_spec_prob.pkl")

    print("\n=== 3. END-TO-END OOS, 4-bucket, test years 2006-2026 ===")
    print("BSS4/BSS2 = Brier skill vs PER-SYMBOL climatology (the honest baseline)")
    print(f"{'grp':<7}{'h':<4}{'thr':>5}{'n':>10}" +
          "".join(f"{k:>11}" for k in ["LLsk_pool", "BSS4_pool", "BSS4_sym", "BSS2_sym",
                                       "nLS_sym", "volqLS", "volqFull"]))
    for group in ("index", "single"):
        for hz in HORIZONS:
            r = allout[(group, hz)]["res"]
            for thr in THRS:
                R = r[thr]
                n = R["clim"]["n"]
                g = lambda m, k: R[m][k] / n
                bss = lambda m, base: 1 - g(m, "bs") / g(base, "bs")
                bss2 = lambda m, base: 1 - g(m, "bs2") / g(base, "bs2")
                print(f"{group:<7}{hz:<4}{thr*100:>4.0f}%{n:>10,}"
                      f"{1-g('emp','ll')/g('clim','ll'):>11.4f}"
                      f"{bss('emp','clim'):>11.4f}"
                      f"{bss('emp','climsym'):>11.4f}"
                      f"{bss2('emp','climsym'):>11.4f}"
                      f"{bss('normLS','climsym'):>11.4f}"
                      f"{bss('volq_ls','climsym'):>11.4f}"
                      f"{bss('volq_full','climsym'):>11.4f}")
    log("done")


if __name__ == "__main__":
    main()
