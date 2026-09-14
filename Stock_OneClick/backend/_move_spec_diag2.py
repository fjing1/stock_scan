"""
_move_spec_diag2.py -- resolution/floor analysis, direction error, own-vol gate, z-variant
comparison, earnings multiplier against the SHIPPED sigma.
"""
from __future__ import annotations

import json
import time
import numpy as np
import pandas as pd

import _move_data as D
import _move_lib as L

HORIZONS = (1, 5, 10, 21)
MIN_TRAIN_YEARS = 5
SHRINK = 40.0
COLS = {"index": [0, 1, 2, 4, 6], "single": [0, 1, 2, 3, 4, 5]}
VOL_CLIP = (1e-3, 0.5)
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


def raw_probs_emp(zs, sig, thr):
    n = len(zs)
    F_dn = np.searchsorted(zs, np.log(1 - thr) / sig, side="right") / n
    F_0 = np.searchsorted(zs, 0.0, side="right") / n
    F_up = np.searchsorted(zs, np.log(1 + thr) / sig, side="right") / n
    return np.column_stack([F_dn, F_0 - F_dn, F_up - F_0, 1 - F_up])


def brier(p, a):
    oh = np.zeros_like(p)
    oh[np.arange(len(a)), a] = 1.0
    return float(((p - oh) ** 2).sum(axis=1).mean())


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
                e[:-h] = y[h:]; e[-h:] = y[-1]
            else:
                e[:] = y[-1]
            out[blk] = e
            start = i
    return out


def walk2(cache, group, hz, thr):
    """Return per-row: actual, p_emp(raw), p_volq(raw), climsym, sigma, sid, year, nz."""
    d = cache[(group, hz)]
    F, lr, tg, yr, sid, dt = d["F"], d["lr"], d["tg"], d["yr"], d["sid"], d["dt"]
    X = design(F, COLS[group])
    ey = end_year(dt, sid, hz)
    years = sorted(set(yr))
    out = {k: [] for k in ("a", "pe", "pq", "pc", "sig", "sid", "yr", "nz", "zmin", "zmax")}
    for yt in years[MIN_TRAIN_YEARS:]:
        tr = (yr < yt) & (ey < yt); te = yr == yt
        if tr.sum() < 2000 or te.sum() == 0:
            continue
        b = ols(X[tr], tg[tr])
        sig_tr = np.exp(X[tr] @ b) * np.sqrt(hz)
        sig_te = np.exp(X[te] @ b) * np.sqrt(hz)
        z_tr = lr[tr] / sig_tr
        zs = np.sort(z_tr)
        a_tr = bucket_idx(lr[tr], thr); a_te = bucket_idx(lr[te], thr)
        pooled = np.bincount(a_tr, minlength=4) / len(a_tr)
        nsym = int(sid.max()) + 1
        cnt = np.zeros((nsym, 4)); np.add.at(cnt, (sid[tr], a_tr), 1.0)
        psym = (cnt + SHRINK * pooled) / (cnt.sum(axis=1, keepdims=True) + SHRINK)
        sd_tr = sig_tr / np.sqrt(hz)
        edges = np.percentile(sd_tr, [20, 40, 60, 80])
        qtr = np.digitize(sd_tr, edges); qte = np.digitize(sig_te / np.sqrt(hz), edges)
        pq = np.zeros((int(te.sum()), 4))
        for q in range(5):
            m = qte == q
            if m.any():
                pq[m] = raw_probs_emp(np.sort(z_tr[qtr == q]), sig_te[m], thr)
        out["a"].append(a_te); out["pe"].append(raw_probs_emp(zs, sig_te, thr))
        out["pq"].append(pq); out["pc"].append(psym[sid[te]]); out["sig"].append(sig_te)
        out["sid"].append(sid[te]); out["yr"].append(np.full(int(te.sum()), yt))
        out["nz"].append(np.full(int(te.sum()), len(zs)))
        out["zmin"].append(np.full(int(te.sum()), zs[0]))
        out["zmax"].append(np.full(int(te.sum()), zs[-1]))
    return {k: (np.vstack(v) if v[0].ndim == 2 else np.concatenate(v)) for k, v in out.items()}


def main():
    cache = pd.read_pickle("_move_spec_cache.pkl")
    log("cache loaded")
    thr = 0.02
    W = {}
    for g in ("index", "single"):
        for hz in HORIZONS:
            W[(g, hz)] = walk2(cache, g, hz, thr)
            log(f"walk {g} h={hz}")

    print("\n=== G. RESOLUTION: how often does a raw bucket probability hit exactly 0 ===")
    print(f"{'grp':<7}{'h':<4}{'n_z(last yr)':>13}{'p_up0':>9}{'p_dn0':>9}"
          f"{'p_up<1e-3':>11}{'p_dn<1e-3':>11}{'min_res':>10}")
    for g in ("index", "single"):
        for hz in HORIZONS:
            w = W[(g, hz)]
            pe = w["pe"]
            nz = w["nz"].max()
            print(f"{g:<7}{hz:<4}{nz:>13,}{np.mean(pe[:,3]<=0):>9.4f}{np.mean(pe[:,0]<=0):>9.4f}"
                  f"{np.mean(pe[:,3]<1e-3):>11.4f}{np.mean(pe[:,0]<1e-3):>11.4f}"
                  f"{1.0/nz:>10.2e}")

    print("\n=== H. z-VARIANT: pooled emp vs sigma-quintile emp (BSS4 vs climsym), thr=2% ===")
    print(f"{'grp':<7}{'h':<4}{'emp':>9}{'volq':>9}{'gain':>9}"
          f"{'emp frac_tick<=0':>18}{'volq frac_tick<=0':>19}")
    for g in ("index", "single"):
        for hz in HORIZONS:
            w = W[(g, hz)]
            a, pc = w["a"], w["pc"]
            pe, pq = norm4(w["pe"]), norm4(w["pq"])
            be, bq, bc = brier(pe, a), brier(pq, a), brier(pc, a)
            fe = fq = float("nan")
            if g == "single":
                ve, vq = [], []
                for u in np.unique(w["sid"]):
                    m = w["sid"] == u
                    if m.sum() < 250:
                        continue
                    bcm = brier(pc[m], a[m])
                    ve.append(1 - brier(pe[m], a[m]) / bcm)
                    vq.append(1 - brier(pq[m], a[m]) / bcm)
                fe, fq = float(np.mean(np.array(ve) <= 0)), float(np.mean(np.array(vq) <= 0))
            print(f"{g:<7}{hz:<4}{1-be/bc:>9.4f}{1-bq/bc:>9.4f}{be/bc-bq/bc:>9.4f}"
                  f"{fe:>18.3f}{fq:>19.3f}")

    print("\n=== I. DIRECTION: up-share predicted vs realized, by year (thr=2%) ===")
    for g, hz in (("index", 21), ("index", 5), ("single", 21), ("single", 5)):
        w = W[(g, hz)]
        a, p = w["a"], norm4(w["pq"] if g == "single" else w["pe"])
        rows = []
        for y in sorted(set(w["yr"])):
            m = w["yr"] == y
            pm = p[m][:, 0] + p[m][:, 3]
            ups = (p[m][:, 3] / pm).mean()
            hit = (a[m] == 0) | (a[m] == 3)
            obs = (a[m][hit] == 3).mean() if hit.sum() else np.nan
            rows.append((y, ups, obs, ups - obs))
        worst = sorted(rows, key=lambda r: -abs(r[3]))[:5]
        print(f"{g} h={hz}: mean|gap|={np.mean([abs(r[3]) for r in rows]):.3f}  worst: " +
              "  ".join(f"{y} {u:.2f}vs{o:.2f}({d:+.2f})" for y, u, o, d in worst))

    print("\n=== J. OWN-VOL GATE: calibration by annualized sigma_hat band, thr=2% ===")
    bands = [(0, .25), (.25, .35), (.35, .45), (.45, .55), (.55, .70), (.70, .90),
             (.90, 1.20), (1.20, 9)]
    for hz in (1, 5, 21):
        print(f"-- single h={hz}")
        w = W[("single", hz)]
        a = w["a"]; p = norm4(w["pq"]); pc = w["pc"]
        av = w["sig"] / np.sqrt(hz) * np.sqrt(252)
        for lo, hi in bands:
            m = (av >= lo) & (av < hi)
            if m.sum() < 500:
                continue
            pm = (p[m][:, 0] + p[m][:, 3]).mean()
            om = float(((a[m] == 0) | (a[m] == 3)).mean())
            e_dn = p[m][:, 0].mean() - (a[m] == 0).mean()
            e_up = p[m][:, 3].mean() - (a[m] == 3).mean()
            bss = 1 - brier(p[m], a[m]) / brier(pc[m], a[m])
            print(f"   {lo:.2f}-{hi:.2f} n={m.sum():>8,} share={m.mean():.3f} "
                  f"pred_mv={pm:.3f} obs_mv={om:.3f} ratio={pm/om:.3f} "
                  f"err_dn={e_dn:+.3f} err_up={e_up:+.3f} BSS4={bss:+.4f}")
    pd.to_pickle(W, "_move_spec_diag2.pkl")
    log("done")


if __name__ == "__main__":
    main()
