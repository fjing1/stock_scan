"""
_move_spec_ci.py -- recent-window bootstrap CIs (the promise), crisis years, and the
final 32-cell grade table with measured numbers.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

HORIZONS = (1, 5, 10, 21)
MIN_TRAIN_YEARS = 5
SHRINK = 40.0
COLS = {"index": [0, 1, 2, 4, 6], "single": [0, 1, 2, 3, 4, 5]}
t0 = time.time()


def design(F, cols):
    return np.column_stack([np.ones(len(F))] + [F[:, c] for c in cols])


def ols(X, y):
    G = X.T @ X
    return np.linalg.solve(G + 1e-10 * np.eye(G.shape[0]), X.T @ y)


def bucket_idx(lr, thr):
    return np.where(lr <= np.log(1 - thr), 0,
                    np.where(lr <= 0, 1, np.where(lr < np.log(1 + thr), 2, 3))).astype(int)


def norm4(p):
    p = np.clip(p, 1e-4, None)
    return p / p.sum(axis=1, keepdims=True)


def probs_emp(zs, sig, thr):
    n = len(zs)
    F_dn = np.searchsorted(zs, np.log(1 - thr) / sig, side="right") / n
    F_0 = np.searchsorted(zs, 0.0, side="right") / n
    F_up = np.searchsorted(zs, np.log(1 + thr) / sig, side="right") / n
    return np.column_stack([F_dn, F_0 - F_dn, F_up - F_0, 1 - F_up])


def end_year(dt, sid, h):
    out = np.empty(len(dt), dtype=np.int64)
    yrs = pd.DatetimeIndex(dt).year.values
    start = 0
    for i in range(1, len(sid) + 1):
        if i == len(sid) or sid[i] != sid[start]:
            blk = slice(start, i); y = yrs[blk]
            e = np.empty(len(y), dtype=np.int64)
            if len(y) > h:
                e[:-h] = y[h:]; e[-h:] = y[-1]
            else:
                e[:] = y[-1]
            out[blk] = e; start = i
    return out


def walk(cache, g, hz, thr):
    d = cache[(g, hz)]
    F, lr, tg, yr, sid, dt = d["F"], d["lr"], d["tg"], d["yr"], d["sid"], d["dt"]
    X = design(F, COLS[g]); ey = end_year(dt, sid, hz)
    volq = (g == "single")
    A, P, PC, DT, YR = [], [], [], [], []
    for yt in sorted(set(yr))[MIN_TRAIN_YEARS:]:
        tr = (yr < yt) & (ey < yt); te = yr == yt
        if tr.sum() < 2000 or te.sum() == 0:
            continue
        b = ols(X[tr], tg[tr])
        s_tr = np.exp(X[tr] @ b) * np.sqrt(hz); s_te = np.exp(X[te] @ b) * np.sqrt(hz)
        z_tr = lr[tr] / s_tr
        a_tr = bucket_idx(lr[tr], thr); a_te = bucket_idx(lr[te], thr)
        pooled = np.bincount(a_tr, minlength=4) / len(a_tr)
        nsym = int(sid.max()) + 1
        cnt = np.zeros((nsym, 4)); np.add.at(cnt, (sid[tr], a_tr), 1.0)
        psym = (cnt + SHRINK * pooled) / (cnt.sum(axis=1, keepdims=True) + SHRINK)
        if volq:
            sd = s_tr / np.sqrt(hz)
            edges = np.percentile(sd, [20, 40, 60, 80])
            qtr = np.digitize(sd, edges); qte = np.digitize(s_te / np.sqrt(hz), edges)
            p = np.zeros((int(te.sum()), 4))
            for q in range(5):
                m = qte == q
                if m.any():
                    p[m] = probs_emp(np.sort(z_tr[qtr == q]), s_te[m], thr)
        else:
            p = probs_emp(np.sort(z_tr), s_te, thr)
        A.append(a_te); P.append(norm4(p)); PC.append(psym[sid[te]])
        DT.append(dt[te]); YR.append(np.full(int(te.sum()), yt))
    return (np.concatenate(A), np.vstack(P), np.vstack(PC),
            np.concatenate(DT), np.concatenate(YR))


def boot(a, p, pc, dt, reps=1000, blk=63, seed=3):
    rng = np.random.default_rng(seed)
    bm = ((p - np.eye(4)[a]) ** 2).sum(axis=1)
    bc = ((pc - np.eye(4)[a]) ** 2).sum(axis=1)
    pm = p[:, 0] + p[:, 3]; pcm = pc[:, 0] + pc[:, 3]
    hit = ((a == 0) | (a == 3)).astype(float)
    b2 = (pm - hit) ** 2; b2c = (pcm - hit) ** 2
    ud, inv = np.unique(dt, return_inverse=True)
    nD = len(ud)
    agg = lambda v: np.bincount(inv, weights=v, minlength=nD)
    Sm, Sc, S2, S2c = agg(bm), agg(bc), agg(b2), agg(b2c)
    nb = int(np.ceil(nD / blk))
    v4, v2 = [], []
    for _ in range(reps):
        st = rng.integers(0, nD, nb)
        idx = np.concatenate([np.arange(s, min(s + blk, nD)) for s in st])
        v4.append(1 - Sm[idx].sum() / Sc[idx].sum())
        v2.append(1 - S2[idx].sum() / S2c[idx].sum())
    return ((1 - Sm.sum() / Sc.sum(), np.percentile(v4, [2.5, 97.5])),
            (1 - S2.sum() / S2c.sum(), np.percentile(v2, [2.5, 97.5])), nD)


def main():
    cache = pd.read_pickle("_move_spec_cache.pkl")
    print("=== R. THE PROMISE: recent window 2021-2026, thr=2%, date-block CI (63d, 1000 reps) ===")
    print(f"{'grp':<7}{'h':<4}{'n':>9}{'nD':>7}{'BSS4':>9}{'BSS4 95%CI':>21}"
          f"{'BSS2':>9}{'BSS2 95%CI':>21}")
    for g in ("index", "single"):
        for hz in HORIZONS:
            a, p, pc, dt, yr = walk(cache, g, hz, 0.02)
            m = yr >= 2021
            (b4, c4), (b2, c2), nD = boot(a[m], p[m], pc[m], dt[m])
            print(f"{g:<7}{hz:<4}{int(m.sum()):>9,}{nD:>7,}{b4:>9.4f}"
                  f"{'['+f'{c4[0]:+.4f}'+','+f'{c4[1]:+.4f}'+']':>21}{b2:>9.4f}"
                  f"{'['+f'{c2[0]:+.4f}'+','+f'{c2[1]:+.4f}'+']':>21}")

    print("\n=== S. CRISIS YEARS, thr=2%: total-move scale calibration ===")
    print(f"{'grp':<7}{'h':<4}{'year':>6}{'n':>8}{'pred_mv':>9}{'obs_mv':>8}{'gap':>8}{'BSS4':>9}")
    for g in ("index", "single"):
        for hz in (1, 5, 21):
            a, p, pc, dt, yr = walk(cache, g, hz, 0.02)
            for y in (2008, 2020, 2022, 2017):
                m = yr == y
                if m.sum() == 0:
                    continue
                pm = (p[m][:, 0] + p[m][:, 3]).mean()
                om = float(((a[m] == 0) | (a[m] == 3)).mean())
                bm = ((p[m] - np.eye(4)[a[m]]) ** 2).sum(axis=1).mean()
                bc = ((pc[m] - np.eye(4)[a[m]]) ** 2).sum(axis=1).mean()
                print(f"{g:<7}{hz:<4}{y:>6}{int(m.sum()):>8,}{pm:>9.4f}{om:>8.4f}"
                      f"{om-pm:>+8.4f}{1-bm/bc:>9.4f}")
    print(f"[{time.time()-t0:.1f}s] done")


if __name__ == "__main__":
    main()
