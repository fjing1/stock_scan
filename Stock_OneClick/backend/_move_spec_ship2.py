"""
_move_spec_ship2.py -- produce the shippable artifact: full-sample coefficients, z tables,
quintile edges, worked examples, and the exact values for the validation test file.
"""
from __future__ import annotations

import json
import time
import numpy as np
import pandas as pd

import _move_data as D
import _move_lib as L

HORIZONS = (1, 5, 10, 21)
VOL_CLIP = (1e-3, 0.5)
INDEX_SYMS = {"SPY", "QQQ", "IWM", "DIA", "^GSPC"}
t0 = time.time()


def clip_log(x):
    return np.log(np.clip(x, *VOL_CLIP))


def log(*a):
    print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)


def ols(X, y):
    G = X.T @ X
    return np.linalg.solve(G + 1e-10 * np.eye(G.shape[0]), X.T @ y)


def feats_index(o, h, l, c, vix):
    rv1 = L.vol_parkinson(h, l, 1)
    return np.column_stack([
        np.ones(len(c)), clip_log(rv1), clip_log(rv1.rolling(5).mean()),
        clip_log(rv1.rolling(22).mean()), clip_log(L.vol_yang_zhang(o, h, l, c, 21)),
        clip_log(vix.reindex(c.index).ffill(limit=3) / 100.0 / np.sqrt(252.0))])


def feats_single(o, h, l, c):
    rv1 = L.vol_parkinson(h, l, 1)
    return np.column_stack([
        np.ones(len(c)), clip_log(rv1), clip_log(rv1.rolling(5).mean()),
        clip_log(rv1.rolling(22).mean()), clip_log(rv1.rolling(63).mean()),
        clip_log(L.vol_yang_zhang(o, h, l, c, 21)), clip_log(L.vol_ewma(c, 0.97))])


def feats_index_novix(o, h, l, c):
    rv1 = L.vol_parkinson(h, l, 1)
    return np.column_stack([
        np.ones(len(c)), clip_log(rv1), clip_log(rv1.rolling(5).mean()),
        clip_log(rv1.rolling(22).mean()), clip_log(L.vol_yang_zhang(o, h, l, c, 21))])


def feats_single_short(o, h, l, c):
    rv1 = L.vol_parkinson(h, l, 1)
    return np.column_stack([
        np.ones(len(c)), clip_log(rv1), clip_log(rv1.rolling(5).mean()),
        clip_log(rv1.rolling(22).mean()), clip_log(L.vol_yang_zhang(o, h, l, c, 21))])


def feats_closeonly(c):
    cd = np.log(c).diff().abs()
    return np.column_stack([
        np.ones(len(c)), clip_log(cd), clip_log(cd.rolling(5).mean()),
        clip_log(cd.rolling(22).mean()), clip_log(cd.rolling(63).mean()),
        clip_log(L.vol_ewma(c, 0.97))])


def main():
    p = D.load()
    O, H, Lo, C = p["Open"].copy(), p["High"].copy(), p["Low"].copy(), p["Close"].copy()
    for df in (O, H, Lo, C):
        df.mask(df <= 0, inplace=True)
    cov = C.notna().sum()
    usable = [s for s in C.columns if s != "^VIX" and cov[s] >= 500]
    vix = p["Close"]["^VIX"]
    G = {"index": [s for s in usable if s in INDEX_SYMS],
         "single": [s for s in usable if s not in INDEX_SYMS]}
    ecache = json.load(open("../../gold_pine_script/pead_earnings_cache.json"))
    log(f"index {G['index']}  singles {len(G['single'])}")

    builders = {
        ("index", "MAIN"): lambda s, c: feats_index(O[s].reindex(c.index), H[s].reindex(c.index),
                                                    Lo[s].reindex(c.index), c, vix),
        ("index", "NOVIX"): lambda s, c: feats_index_novix(O[s].reindex(c.index),
                                                           H[s].reindex(c.index),
                                                           Lo[s].reindex(c.index), c),
        ("index", "CLOSE"): lambda s, c: feats_closeonly(c),
        ("single", "MAIN"): lambda s, c: feats_single(O[s].reindex(c.index), H[s].reindex(c.index),
                                                      Lo[s].reindex(c.index), c),
        ("single", "SHORT"): lambda s, c: feats_single_short(O[s].reindex(c.index),
                                                             H[s].reindex(c.index),
                                                             Lo[s].reindex(c.index), c),
        ("single", "CLOSE"): lambda s, c: feats_closeonly(c),
    }
    LBL = {("index", "MAIN"): ["const", "rv_d", "rv_w", "rv_m", "yz21", "logvix"],
           ("index", "NOVIX"): ["const", "rv_d", "rv_w", "rv_m", "yz21"],
           ("index", "CLOSE"): ["const", "cd_d", "cd_w", "cd_m", "cd_q", "e97"],
           ("single", "MAIN"): ["const", "rv_d", "rv_w", "rv_m", "rv_q", "yz21", "e97"],
           ("single", "SHORT"): ["const", "rv_d", "rv_w", "rv_m", "yz21"],
           ("single", "CLOSE"): ["const", "cd_d", "cd_w", "cd_m", "cd_q", "e97"]}

    art = {}
    print("\n=== SHIP COEFFICIENTS (full sample 2001-09-17..2026-09-10) ===")
    for (g, tag), fb in builders.items():
        for hz in HORIZONS:
            Xs, ys, ss, ii, dd = [], [], [], [], []
            for i, s in enumerate(G[g]):
                c = C[s].dropna()
                if len(c) < 400:
                    continue
                X = fb(s, c)
                y = clip_log(L.realized_vol_forward(c, hz)).values
                lr = np.log(c.shift(-hz) / c).values
                ok = np.isfinite(X).all(axis=1) & np.isfinite(y) & np.isfinite(lr)
                Xs.append(X[ok]); ys.append(y[ok]); ss.append(lr[ok])
                ii.append(np.full(int(ok.sum()), i)); dd.append(c.index.values[ok])
            X = np.vstack(Xs); y = np.concatenate(ys); lr = np.concatenate(ss)
            b = ols(X, y)
            sig_d = np.exp(X @ b)
            sig_h = sig_d * np.sqrt(hz)
            z = lr / sig_h
            art[(g, tag, hz)] = dict(beta=b, cols=LBL[(g, tag)], n=len(y),
                                     z=np.sort(z).astype(np.float32),
                                     z_unsorted=z.astype(np.float32),
                                     sig_d=sig_d, sid=np.concatenate(ii))
            print(f"{g:<7}{tag:<6}h={hz:<3} n={len(y):>9,}  " +
                  "  ".join(f"{c}={v:+.4f}" for c, v in zip(LBL[(g, tag)], b)))

    print("\n=== SIGMA-QUINTILE EDGES for the SINGLE-name z tables (annualized sigma_hat) ===")
    print("(edges = 20/40/60/80th percentile of pooled train sigma_hat_daily * sqrt(252))")
    for hz in HORIZONS:
        a = art[("single", "MAIN", hz)]
        ann = a["sig_d"] * np.sqrt(252)
        e = np.percentile(ann, [20, 40, 60, 80])
        a["edges_ann"] = e
        print(f"h={hz:<3} edges = " + "  ".join(f"{x:.4f}" for x in e) +
              f"   |  p01={np.percentile(ann,1):.3f} p50={np.median(ann):.3f} "
              f"p99={np.percentile(ann,99):.3f}")

    print("\n=== SIGMA_HAT DISTRIBUTION BY HORIZON (annualized per-day sigma) ===")
    for g in ("index", "single"):
        for hz in HORIZONS:
            ann = art[(g, "MAIN", hz)]["sig_d"] * np.sqrt(252)
            print(f"{g:<7}h={hz:<3} p05={np.percentile(ann,5):.3f} p25={np.percentile(ann,25):.3f} "
                  f"med={np.median(ann):.3f} p75={np.percentile(ann,75):.3f} "
                  f"p95={np.percentile(ann,95):.3f}  sd(log)={np.std(np.log(ann)):.3f}")

    # -------- z tables
    UG = [-6, -5, -4, -3.5, -3, -2.5, -2.25, -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0,
          0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3, 3.5, 4, 5, 6]
    print("\n=== z TABLE: INDEX (pooled, one per horizon), CDF G(z) at grid points ===")
    for hz in HORIZONS:
        z = art[("index", "MAIN", hz)]["z"]
        vals = np.searchsorted(z, UG, side="right") / len(z)
        print(f"h={hz:<3} n_z={len(z):>9,} med={np.median(z):+.4f} "
              f"iqr/1.349={(np.percentile(z,75)-np.percentile(z,25))/1.349:.4f} "
              f"sd={z.std():.3f} P(z<-2)={np.mean(z<-2):.4f} P(z>2)={np.mean(z>2):.4f}")
        print("      " + " ".join(f"{u:+.2f}:{v:.4f}" for u, v in zip(UG, vals)))

    print("\n=== z TABLE: SINGLE by sigma quintile (Q1 calm .. Q5 wild) ===")
    for hz in HORIZONS:
        a = art[("single", "MAIN", hz)]
        ann = a["sig_d"] * np.sqrt(252)
        q = np.digitize(ann, a["edges_ann"])
        # NOTE: a["z"] is SORTED, so it must be re-derived unsorted for the quintile split
        a["qz"] = [np.sort(a["z_unsorted"][q == k]) for k in range(5)]
        print(f"-- h={hz}")
        for k in range(5):
            z = a["qz"][k]
            vals = np.searchsorted(z, UG, side="right") / len(z)
            print(f"  Q{k+1} n={len(z):>8,} med={np.median(z):+.4f} "
                  f"iqs={(np.percentile(z,75)-np.percentile(z,25))/1.349:.4f} "
                  f"P(z<-2)={np.mean(z<-2):.4f} P(z>2)={np.mean(z>2):.4f}")
            print("     " + " ".join(f"{u:+.2f}:{v:.4f}" for u, v in zip(UG, vals)))
        zp = np.sort(a["z"])
        vals = np.searchsorted(zp, UG, side="right") / len(zp)
        print(f"  POOLED n={len(zp):>8,} med={np.median(zp):+.4f} "
              f"iqs={(np.percentile(zp,75)-np.percentile(zp,25))/1.349:.4f}")
        print("     " + " ".join(f"{u:+.2f}:{v:.4f}" for u, v in zip(UG, vals)))

    pd.to_pickle({k: {kk: vv for kk, vv in v.items() if kk not in ("sig_d", "z_unsorted")}
                  for k, v in art.items()}, "_move_spec_ship2.pkl")
    log("saved")

    # -------- worked examples from the last panel bar
    def probs(zs, sig_h, thr):
        n = len(zs)
        F_dn = np.searchsorted(zs, np.log(1 - thr) / sig_h, side="right") / n
        F_0 = np.searchsorted(zs, 0.0, side="right") / n
        F_up = np.searchsorted(zs, np.log(1 + thr) / sig_h, side="right") / n
        p = np.array([F_dn, F_0 - F_dn, F_up - F_0, 1 - F_up])
        p = np.clip(p, 1e-4, None)
        return p / p.sum()

    print("\n=== WORKED EXAMPLES at the 2026-09-10 close, thr=2% ===")
    EM = {1: 1.82, 5: 1.48, 10: 1.33, 21: 1.21}
    lastday = C.index[-1]
    for sym in ["SPY", "QQQ", "IWM", "AAPL", "NVDA", "TSLA", "KO", "XOM", "JPM", "BLNK"]:
        if sym not in C.columns:
            print(f"{sym}: not in panel"); continue
        c = C[sym].dropna()
        g = "index" if sym in INDEX_SYMS else "single"
        X = builders[(g, "MAIN")](sym, c)
        row = X[-1]
        ev = ecache.get(sym) or []
        nxt = None
        if ev:
            ed = pd.DatetimeIndex(sorted(pd.to_datetime([e[0] for e in ev])))
            fut = ed[ed > c.index[-1]]
            nxt = fut[0] if len(fut) else None
        print(f"\n{sym} ({g}) last bar {pd.Timestamp(c.index[-1]).date()} "
              f"close {c.iloc[-1]:.2f}  earnings events={len(ev)} next={nxt}")
        print(f"   {'h':<4}{'sig_d ann':>11}{'sig_h':>9}{'Q':>3}{'k=ln1.02/sig_h':>16}"
              f"{'p_dn':>8}{'p_dns':>8}{'p_ups':>8}{'p_up':>8}{'p_move':>9}")
        for hz in HORIZONS:
            a = art[(g, "MAIN", hz)]
            sd = float(np.exp(row @ a["beta"]))
            sh = sd * np.sqrt(hz)
            if g == "single":
                k = int(np.digitize([sd * np.sqrt(252)], a["edges_ann"])[0])
                zs = a["qz"][k]
                qlab = f"Q{k+1}"
            else:
                zs = a["z"]; qlab = "-"
            pr = probs(zs, sh, 0.02)
            print(f"   {hz:<4}{sd*np.sqrt(252):>11.4f}{sh:>9.4f}{qlab:>3}"
                  f"{np.log(1.02)/sh:>16.3f}"
                  f"{pr[0]:>8.3f}{pr[1]:>8.3f}{pr[2]:>8.3f}{pr[3]:>8.3f}{pr[0]+pr[3]:>9.3f}")

    print("\n=== VALIDATION ANCHORS ===")
    for g in ("index", "single"):
        for hz in HORIZONS:
            a = art[(g, "MAIN", hz)]
            z = a["z"]
            print(f"{g:<7}h={hz:<3} n_fit={a['n']:>9,} n_z={len(z):>9,} "
                  f"med(z)={np.median(z):+.4f} mean(z)={z.mean():+.4f} sd(z)={z.std():.4f} "
                  f"P(z<-2)={np.mean(z<-2):.4f} P(z>2)={np.mean(z>2):.4f} "
                  f"P(z<-3)={np.mean(z<-3):.5f} P(z>3)={np.mean(z>3):.5f}")
    log("done")


if __name__ == "__main__":
    main()
