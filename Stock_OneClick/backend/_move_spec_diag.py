"""
_move_spec_diag.py -- diagnostics for the chosen spec: ECE, per-year, per-ticker, recent window,
support gate, own-vol gate, direction error, earnings multiplier.
"""
from __future__ import annotations

import json
import math
import time
import numpy as np
import pandas as pd

HORIZONS = (1, 5, 10, 21)
MIN_TRAIN_YEARS = 5
SHRINK = 40.0
COLS = {"index": [0, 1, 2, 4, 6], "single": [0, 1, 2, 3, 4, 5]}
t0 = time.time()
_erf = np.vectorize(math.erf)


def log(*a):
    print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)


def design(F, cols):
    return np.column_stack([np.ones(len(F))] + [F[:, c] for c in cols])


def ols(X, y):
    G = X.T @ X
    return np.linalg.solve(G + 1e-10 * np.eye(G.shape[0]), X.T @ y)


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


def brier(p, a):
    oh = np.zeros_like(p)
    oh[np.arange(len(a)), a] = 1.0
    return float(((p - oh) ** 2).sum(axis=1).mean())


def logloss(p, a):
    return float(-np.log(np.clip(p[np.arange(len(a)), a], 1e-12, 1)).mean())


def ece_b(pb, hit, nb=10):
    df = pd.DataFrame({"p": pb, "y": np.asarray(hit, dtype=float)})
    df["bin"] = pd.qcut(df.p.rank(method="first"), nb, labels=False, duplicates="drop")
    gg = df.groupby("bin").agg(n=("y", "size"), mp=("p", "mean"), ob=("y", "mean"))
    return float((gg.n / gg.n.sum() * (gg.ob - gg.mp).abs()).sum())


def end_year(dt, sid, h):
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


def walk(cache, group, hz, thr, use_volq):
    d = cache[(group, hz)]
    F, lr, tg, yr, sid, dt = d["F"], d["lr"], d["tg"], d["yr"], d["sid"], d["dt"]
    X = design(F, COLS[group])
    ey = end_year(dt, sid, hz)
    years = sorted(set(yr))
    A, P, PC, S, SI, YR, DT = [], [], [], [], [], [], []
    for yt in years[MIN_TRAIN_YEARS:]:
        tr = (yr < yt) & (ey < yt)
        te = yr == yt
        if tr.sum() < 2000 or te.sum() == 0:
            continue
        b = ols(X[tr], tg[tr])
        sig_tr = np.exp(X[tr] @ b) * np.sqrt(hz)
        sig_te = np.exp(X[te] @ b) * np.sqrt(hz)
        z_tr = lr[tr] / sig_tr
        a_tr = bucket_idx(lr[tr], thr)
        a_te = bucket_idx(lr[te], thr)
        pooled = np.bincount(a_tr, minlength=4) / len(a_tr)
        nsym = int(sid.max()) + 1
        cnt = np.zeros((nsym, 4))
        np.add.at(cnt, (sid[tr], a_tr), 1.0)
        tot = cnt.sum(axis=1, keepdims=True)
        psym = (cnt + SHRINK * pooled) / (tot + SHRINK)
        if use_volq:
            sd_tr = sig_tr / np.sqrt(hz)
            edges = np.percentile(sd_tr, [20, 40, 60, 80])
            qtr = np.digitize(sd_tr, edges)
            qte = np.digitize(sig_te / np.sqrt(hz), edges)
            p = np.zeros((len(a_te), 4))
            for q in range(5):
                m = qte == q
                if m.any():
                    p[m] = probs_emp(np.sort(z_tr[qtr == q]), sig_te[m], thr)
        else:
            p = probs_emp(np.sort(z_tr), sig_te, thr)
        A.append(a_te); P.append(p); PC.append(psym[sid[te]]); S.append(sig_te)
        SI.append(sid[te]); YR.append(np.full(int(te.sum()), yt)); DT.append(dt[te])
    return (np.concatenate(A), np.vstack(P), np.vstack(PC), np.concatenate(S),
            np.concatenate(SI), np.concatenate(YR), np.concatenate(DT))


def report(tag, a, p, pc, thr):
    n = len(a)
    bs, bsc = brier(p, a), brier(pc, a)
    pm, pmc = p[:, 0] + p[:, 3], pc[:, 0] + pc[:, 3]
    hit = ((a == 0) | (a == 3)).astype(float)
    bs2 = float(((pm - hit) ** 2).mean()); bs2c = float(((pmc - hit) ** 2).mean())
    e_up = ece_b(p[:, 3], a == 3); e_dn = ece_b(p[:, 0], a == 0); e_mv = ece_b(pm, hit)
    return dict(n=n, BSS4=1 - bs / bsc, BSS2=1 - bs2 / bs2c,
                LL=logloss(p, a), LLc=logloss(pc, a),
                ECEup=e_up, ECEdn=e_dn, ECEmv=e_mv,
                pred_mv=float(pm.mean()), obs_mv=float(hit.mean()),
                pred_up=float(p[:, 3].mean()), obs_up=float((a == 3).mean()),
                pred_dn=float(p[:, 0].mean()), obs_dn=float((a == 0).mean()))


def main():
    cache = pd.read_pickle("_move_spec_cache.pkl")
    log("cache loaded")
    thr = 0.02
    variants = {"index": False, "single": True}   # singles use vol-quintile z tables

    print("\n=== A. HEADLINE, thr=2%, full test 2006-2026, chosen z variant ===")
    print(f"{'grp':<7}{'h':<4}{'n':>10}{'BSS4':>9}{'BSS2':>9}{'LLsk':>9}"
          f"{'ECEup':>8}{'ECEdn':>8}{'ECEmv':>8}{'pred_mv':>9}{'obs_mv':>8}")
    keep = {}
    for group in ("index", "single"):
        for hz in HORIZONS:
            a, p, pc, s, si, yy, dd = walk(cache, group, hz, thr, variants[group])
            keep[(group, hz)] = (a, p, pc, s, si, yy, dd)
            r = report("", a, p, pc, thr)
            print(f"{group:<7}{hz:<4}{r['n']:>10,}{r['BSS4']:>9.4f}{r['BSS2']:>9.4f}"
                  f"{1-r['LL']/r['LLc']:>9.4f}{r['ECEup']:>8.4f}{r['ECEdn']:>8.4f}"
                  f"{r['ECEmv']:>8.4f}{r['pred_mv']:>9.4f}{r['obs_mv']:>8.4f}")

    print("\n=== B. RECENT WINDOW ONLY (test years 2021-2026) — the shipping promise ===")
    print(f"{'grp':<7}{'h':<4}{'n':>10}{'BSS4':>9}{'BSS2':>9}{'ECEup':>8}{'ECEdn':>8}{'ECEmv':>8}")
    for group in ("index", "single"):
        for hz in HORIZONS:
            a, p, pc, s, si, yy, dd = keep[(group, hz)]
            m = yy >= 2021
            r = report("", a[m], p[m], pc[m], thr)
            print(f"{group:<7}{hz:<4}{r['n']:>10,}{r['BSS4']:>9.4f}{r['BSS2']:>9.4f}"
                  f"{r['ECEup']:>8.4f}{r['ECEdn']:>8.4f}{r['ECEmv']:>8.4f}")

    print("\n=== C. PER-YEAR sign count (BSS4 vs climsym > 0), thr=2% ===")
    for group in ("index", "single"):
        for hz in HORIZONS:
            a, p, pc, s, si, yy, dd = keep[(group, hz)]
            pos, tot, vals = 0, 0, []
            for y in sorted(set(yy)):
                m = yy == y
                v = 1 - brier(p[m], a[m]) / brier(pc[m], a[m])
                vals.append((y, v)); tot += 1; pos += v > 0
            worst = sorted(vals, key=lambda x: x[1])[:3]
            print(f"{group:<7}h={hz:<4} positive {pos}/{tot}   worst: " +
                  " ".join(f"{y}:{v:+.4f}" for y, v in worst))

    print("\n=== D. PER-TICKER BSS4 distribution (singles, >=250 OOS rows), thr=2% ===")
    for hz in HORIZONS:
        a, p, pc, s, si, yy, dd = keep[("single", hz)]
        vals = []
        for u in np.unique(si):
            m = si == u
            if m.sum() < 250:
                continue
            vals.append(1 - brier(p[m], a[m]) / brier(pc[m], a[m]))
        v = np.array(vals)
        print(f"h={hz:<4} n_sym={len(v)}  p05={np.percentile(v,5):+.4f} "
              f"p25={np.percentile(v,25):+.4f} med={np.median(v):+.4f} "
              f"p75={np.percentile(v,75):+.4f} p95={np.percentile(v,95):+.4f} "
              f"frac<=0={np.mean(v<=0):.3f}")

    print("\n=== E. SUPPORT RATIO log(1+thr)/sigma_h, thr=2% ===")
    for group in ("index", "single"):
        for hz in HORIZONS:
            a, p, pc, s, si, yy, dd = keep[(group, hz)]
            rt = np.log(1.02) / s
            print(f"{group:<7}h={hz:<4} p05={np.percentile(rt,5):.3f} "
                  f"med={np.median(rt):.3f} p95={np.percentile(rt,95):.3f} "
                  f"frac>3={np.mean(rt>3):.3f} frac<0.4={np.mean(rt<0.4):.3f}")

    print("\n=== F. CALIBRATION BY SUPPORT-RATIO BAND (BSS4 vs climsym), thr=2% ===")
    bands = [(0, 0.2), (0.2, 0.4), (0.4, 0.7), (0.7, 1.2), (1.2, 2.0), (2.0, 3.0), (3.0, 99)]
    for group in ("index", "single"):
        for hz in (1, 5, 21):
            a, p, pc, s, si, yy, dd = keep[(group, hz)]
            rt = np.log(1.02) / s
            row = []
            for lo, hi in bands:
                m = (rt >= lo) & (rt < hi)
                if m.sum() < 300:
                    row.append("     .    ")
                    continue
                v = 1 - brier(p[m], a[m]) / brier(pc[m], a[m])
                row.append(f"{v:+.4f}({m.sum()//1000}k)")
            print(f"{group:<7}h={hz:<4}" + " ".join(f"{x:>13}" for x in row))
    print("bands: " + " ".join(f"{lo}-{hi}" for lo, hi in bands))
    pd.to_pickle({k: (v[0], v[1].astype(np.float32), v[2].astype(np.float32),
                      v[3].astype(np.float32), v[4], v[5], v[6]) for k, v in keep.items()},
                 "_move_spec_diag.pkl")
    log("done")


if __name__ == "__main__":
    main()
