"""
_move_spec_dir.py -- the decisive test for the up/down split, reliability tables, refusal cells.
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


def log(*a):
    print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)


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
    out = {k: [] for k in ("a", "p", "praw", "pc", "sig", "sid", "yr", "upbase", "upsym")}
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
            praw = np.zeros((int(te.sum()), 4))
            for q in range(5):
                m = qte == q
                if m.any():
                    praw[m] = probs_emp(np.sort(z_tr[qtr == q]), s_te[m], thr)
        else:
            praw = probs_emp(np.sort(z_tr), s_te, thr)
        # train up-share given big move
        big = (a_tr == 0) | (a_tr == 3)
        upb = float((a_tr[big] == 3).mean())
        ub = (cnt[:, 3] + SHRINK * pooled[3]) / (cnt[:, 0] + cnt[:, 3] +
                                                 SHRINK * (pooled[0] + pooled[3]))
        out["a"].append(a_te); out["praw"].append(praw); out["p"].append(norm4(praw))
        out["pc"].append(psym[sid[te]]); out["sig"].append(s_te); out["sid"].append(sid[te])
        out["yr"].append(np.full(int(te.sum()), yt))
        out["upbase"].append(np.full(int(te.sum()), upb)); out["upsym"].append(ub[sid[te]])
    return {k: (np.vstack(v) if v[0].ndim == 2 else np.concatenate(v)) for k, v in out.items()}


def main():
    cache = pd.read_pickle("_move_spec_cache.pkl")
    log("cache loaded")
    thr = 0.02
    W = {(g, hz): walk(cache, g, hz, thr) for g in ("index", "single") for hz in HORIZONS}
    log("walks done")

    print("=== O. THE UP/DOWN SPLIT: conditional on a big move, was it up? (thr=2%) ===")
    print("Brier on P(up | |move|>=thr). Lower is better. 'model' = p_up/(p_up+p_dn).")
    print(f"{'grp':<7}{'h':<4}{'n_big':>9}{'obs_up':>8}{'model':>9}{'const.5':>9}"
          f"{'trainbase':>11}{'persym':>9}{'skill vs base':>14}")
    for g in ("index", "single"):
        for hz in HORIZONS:
            w = W[(g, hz)]
            a, p = w["a"], w["p"]
            big = (a == 0) | (a == 3)
            y = (a[big] == 3).astype(float)
            pm = p[big][:, 0] + p[big][:, 3]
            pu = p[big][:, 3] / pm
            b_m = float(((pu - y) ** 2).mean())
            b_h = float(((0.5 - y) ** 2).mean())
            b_b = float(((w["upbase"][big] - y) ** 2).mean())
            b_s = float(((w["upsym"][big] - y) ** 2).mean())
            print(f"{g:<7}{hz:<4}{int(big.sum()):>9,}{y.mean():>8.4f}{b_m:>9.4f}{b_h:>9.4f}"
                  f"{b_b:>11.4f}{b_s:>9.4f}{1-b_m/b_b:>14.4f}")

    print("\n=== P. RELIABILITY of P(|move|>=2%), 10 equal-count deciles, final config ===")
    for g in ("index", "single"):
        for hz in HORIZONS:
            w = W[(g, hz)]
            a, p = w["a"], w["p"]
            pm = p[:, 0] + p[:, 3]
            hit = ((a == 0) | (a == 3)).astype(float)
            df = pd.DataFrame({"p": pm, "y": hit})
            df["b"] = pd.qcut(df.p.rank(method="first"), 10, labels=False)
            gg = df.groupby("b").agg(n=("y", "size"), mp=("p", "mean"), ob=("y", "mean"))
            gg["gap"] = gg.ob - gg.mp
            print(f"{g:<7}h={hz:<3} decile pred->obs: " +
                  " ".join(f"{r.mp:.2f}/{r.ob:.2f}" for r in gg.itertuples()) +
                  f"   max|gap|={gg.gap.abs().max():.4f}")

    print("\n=== Q. REFUSAL CELLS: thresholds where the answer degenerates ===")
    print(f"{'grp':<7}{'h':<4}{'thr':>5}{'n':>10}{'med k':>8}{'p_up==0':>9}{'p_dn==0':>9}"
          f"{'BSS4':>9}{'BSS2':>9}{'obs_mv':>8}{'pred_mv':>9}")
    for g in ("index", "single"):
        for hz in HORIZONS:
            for t in (0.01, 0.02, 0.03, 0.05):
                w = walk(cache, g, hz, t)
                a, pr, p, pc = w["a"], w["praw"], w["p"], w["pc"]
                k = np.log(1 + t) / w["sig"]
                bm = lambda q: float(((q - np.eye(4)[a]) ** 2).sum(axis=1).mean())
                pmv = p[:, 0] + p[:, 3]; hit = ((a == 0) | (a == 3)).astype(float)
                b2 = float(((pmv - hit) ** 2).mean())
                pcm = pc[:, 0] + pc[:, 3]
                b2c = float(((pcm - hit) ** 2).mean())
                print(f"{g:<7}{hz:<4}{t*100:>4.0f}%{len(a):>10,}{np.median(k):>8.2f}"
                      f"{np.mean(pr[:,3]<=0):>9.4f}{np.mean(pr[:,0]<=0):>9.4f}"
                      f"{1-bm(p)/bm(pc):>9.4f}{1-b2/b2c:>9.4f}{hit.mean():>8.4f}{pmv.mean():>9.4f}")
    log("done")


if __name__ == "__main__":
    main()
