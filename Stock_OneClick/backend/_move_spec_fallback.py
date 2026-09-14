"""
_move_spec_fallback.py -- fallback ladder costs in the currency that matters (end-to-end BSS4),
date-block bootstrap CIs, non-overlapping check, earnings multiplier vs the shipped sigma.
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
VOL_CLIP = (1e-3, 0.5)
INDEX_SYMS = {"SPY", "QQQ", "IWM", "DIA", "^GSPC"}
t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)


def clip_log(x):
    return np.log(np.clip(x, *VOL_CLIP))


FN = ["rv_d", "rv_w", "rv_m", "rv_q", "yz21", "e97", "vix", "e94", "cc21",
      "cd_d", "cd_w", "cd_m", "cd_q", "park21", "gk21"]
IX = {n: i for i, n in enumerate(FN)}

SPECS = {
    "FULL_index":   ["rv_d", "rv_w", "rv_m", "yz21", "vix"],
    "FULL_single":  ["rv_d", "rv_w", "rv_m", "rv_q", "yz21", "e97", "vix"],
    "noVIX_index":  ["rv_d", "rv_w", "rv_m", "yz21"],
    "noVIX_single": ["rv_d", "rv_w", "rv_m", "rv_q", "yz21", "e97"],
    "short22_index":  ["rv_d", "rv_w", "rv_m", "yz21", "vix"],
    "short22_single": ["rv_d", "rv_w", "rv_m", "yz21", "vix"],
    "closeonly":    ["cd_d", "cd_w", "cd_m", "cd_q", "e97"],
    "closeonly+vix": ["cd_d", "cd_w", "cd_m", "cd_q", "e97", "vix"],
    "ewma94":       ["e94"],
    "park21":       ["park21"],
}


def features(o, h, l, c, vix):
    rv1 = L.vol_parkinson(h, l, 1)
    cd = np.log(c).diff().abs()
    return np.column_stack([
        clip_log(rv1), clip_log(rv1.rolling(5).mean()), clip_log(rv1.rolling(22).mean()),
        clip_log(rv1.rolling(63).mean()),
        clip_log(L.vol_yang_zhang(o, h, l, c, 21)), clip_log(L.vol_ewma(c, 0.97)),
        clip_log(vix.reindex(c.index).ffill(limit=3) / 100.0 / np.sqrt(252.0)),
        clip_log(L.vol_ewma(c, 0.94)), clip_log(L.vol_cc(c, 21)),
        clip_log(cd), clip_log(cd.rolling(5).mean()), clip_log(cd.rolling(22).mean()),
        clip_log(cd.rolling(63).mean()),
        clip_log(L.vol_parkinson(h, l, 21)), clip_log(L.vol_garman_klass(o, h, l, c, 21)),
    ])


def setup():
    p = D.load()
    O, H, Lo, C = p["Open"].copy(), p["High"].copy(), p["Low"].copy(), p["Close"].copy()
    for df in (O, H, Lo, C):
        df.mask(df <= 0, inplace=True)
    cov = C.notna().sum()
    usable = [s for s in C.columns if s != "^VIX" and cov[s] >= 500]
    vix = p["Close"]["^VIX"]
    return O, H, Lo, C, vix, {"index": [s for s in usable if s in INDEX_SYMS],
                              "single": [s for s in usable if s not in INDEX_SYMS]}


def build(O, H, Lo, C, vix, groups):
    cache = {}
    for g, syms in groups.items():
        for hz in HORIZONS:
            F, LR, TG, YR, SID, DT = [], [], [], [], [], []
            for i, s in enumerate(syms):
                c = C[s].dropna()
                if len(c) < 400:
                    continue
                f = features(O[s].reindex(c.index), H[s].reindex(c.index),
                             Lo[s].reindex(c.index), c, vix)
                lr = np.log(c.shift(-hz) / c).values
                tg = clip_log(L.realized_vol_forward(c, hz)).values
                ok = np.isfinite(f).all(axis=1) & np.isfinite(lr) & np.isfinite(tg)
                if ok.sum() < 100:
                    continue
                F.append(f[ok]); LR.append(lr[ok]); TG.append(tg[ok])
                YR.append(c.index.year.values[ok]); SID.append(np.full(int(ok.sum()), i))
                DT.append(c.index.values[ok])
            cache[(g, hz)] = dict(F=np.vstack(F), lr=np.concatenate(LR), tg=np.concatenate(TG),
                                  yr=np.concatenate(YR), sid=np.concatenate(SID),
                                  dt=np.concatenate(DT), syms=syms)
            log(f"cache {g} h={hz} rows={len(cache[(g,hz)]['lr']):,}")
    return cache


def design(F, names):
    return np.column_stack([np.ones(len(F))] + [F[:, IX[n]] for n in names])


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
    return norm4(np.column_stack([F_dn, F_0 - F_dn, F_up - F_0, 1 - F_up]))


def brier_rows(p, a):
    oh = np.zeros_like(p)
    oh[np.arange(len(a)), a] = 1.0
    return ((p - oh) ** 2).sum(axis=1)


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


def run_spec(cache, g, hz, names, thr=0.02, volq=None, earn=None, emult=None):
    d = cache[(g, hz)]
    X = design(d["F"], names)
    lr, tg, yr, sid, dt = d["lr"], d["tg"], d["yr"], d["sid"], d["dt"]
    ey = end_year(dt, sid, hz)
    if volq is None:
        volq = (g == "single")
    A, P, PC, DTa, SI = [], [], [], [], []
    for yt in sorted(set(yr))[MIN_TRAIN_YEARS:]:
        tr = (yr < yt) & (ey < yt); te = yr == yt
        if tr.sum() < 2000 or te.sum() == 0:
            continue
        b = ols(X[tr], tg[tr])
        s_tr = np.exp(X[tr] @ b) * np.sqrt(hz)
        s_te = np.exp(X[te] @ b) * np.sqrt(hz)
        if earn is not None:
            s_tr = s_tr * np.where(earn[tr], emult[hz], 1.0)
            s_te = s_te * np.where(earn[te], emult[hz], 1.0)
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
        A.append(a_te); P.append(p); PC.append(psym[sid[te]])
        DTa.append(dt[te]); SI.append(sid[te])
    a = np.concatenate(A); p = np.vstack(P); pc = np.vstack(PC)
    return a, p, pc, np.concatenate(DTa), np.concatenate(SI)


def bss(a, p, pc):
    return float(1 - brier_rows(p, a).mean() / brier_rows(pc, a).mean())


def main():
    O, H, Lo, C, vix, groups = setup()
    cache = build(O, H, Lo, C, vix, groups)

    print("\n=== K. FALLBACK LADDER, end-to-end BSS4 vs per-symbol climatology, thr=2% ===")
    ladder = {
        "index": ["FULL_index", "noVIX_index", "short22_index", "closeonly+vix",
                  "closeonly", "ewma94", "park21"],
        "single": ["FULL_single", "noVIX_single", "short22_single", "closeonly+vix",
                   "closeonly", "ewma94", "park21"],
    }
    res = {}
    for g in ("index", "single"):
        print(f"-- {g}")
        print(f"{'spec':<16}" + "".join(f"{'h='+str(h):>11}" for h in HORIZONS))
        for nm in ladder[g]:
            row = []
            for hz in HORIZONS:
                a, p, pc, dt, si = run_spec(cache, g, hz, SPECS[nm])
                v = bss(a, p, pc)
                row.append(v)
                res[(g, hz, nm)] = v
            print(f"{nm:<16}" + "".join(f"{x:>11.4f}" for x in row))

    print("\n=== L. DATE-BLOCK BOOTSTRAP CI on BSS4 (block=63 dates, 500 reps, cluster by date) ===")
    rng = np.random.default_rng(7)
    print(f"{'grp':<7}{'h':<4}{'BSS4':>9}{'95% CI':>22}{'n_dates':>9}{'nonov BSS4':>12}")
    for g in ("index", "single"):
        for hz in HORIZONS:
            nm = "FULL_index" if g == "index" else "FULL_single"
            a, p, pc, dt, si = run_spec(cache, g, hz, SPECS[nm])
            bm = brier_rows(p, a); bc = brier_rows(pc, a)
            dts = pd.DatetimeIndex(dt)
            ud, inv = np.unique(dts.values, return_inverse=True)
            nD = len(ud)
            sm = np.bincount(inv, weights=bm, minlength=nD)
            sc = np.bincount(inv, weights=bc, minlength=nD)
            cn = np.bincount(inv, minlength=nD).astype(float)
            blk = 63
            nb = int(np.ceil(nD / blk))
            vals = []
            for _ in range(500):
                st = rng.integers(0, nD, nb)
                idx = np.concatenate([np.arange(s, min(s + blk, nD)) for s in st])
                vals.append(1 - sm[idx].sum() / sc[idx].sum())
            lo, hi = np.percentile(vals, [2.5, 97.5])
            # non-overlapping: every h-th date
            keep = np.isin(inv, np.arange(0, nD, hz))
            nov = 1 - bm[keep].mean() / bc[keep].mean()
            print(f"{g:<7}{hz:<4}{1-sm.sum()/sc.sum():>9.4f}"
                  f"{'['+f'{lo:+.4f}'+','+f'{hi:+.4f}'+']':>22}{nD:>9,}{nov:>12.4f}")

    # ---------------- earnings
    print("\n=== M. EARNINGS MULTIPLIER against the SHIPPED sigma ===")
    ecache = json.load(open("../../gold_pine_script/pead_earnings_cache.json"))
    syms = groups["single"]
    for hz in HORIZONS:
        d = cache[("single", hz)]
        dt = pd.DatetimeIndex(d["dt"]); sid = d["sid"]
        flag = np.zeros(len(dt), dtype=bool)
        covered = np.zeros(len(dt), dtype=bool)
        for i, s in enumerate(syms):
            ev = ecache.get(s) or []
            if len(ev) < 5:
                continue
            m = sid == i
            if not m.any():
                continue
            dates = pd.DatetimeIndex(sorted(pd.to_datetime([e[0] for e in ev])))
            sd = dt[m]
            lo, hi = dates.min(), dates.max()
            cov = (sd >= lo) & (sd <= hi)
            # reaction bar b or b+1 inside (t, t+h]: approximate with calendar window
            pos = np.searchsorted(dates.values, sd.values, side="right")
            nxt = np.where(pos < len(dates), dates.values[np.minimum(pos, len(dates) - 1)],
                           np.datetime64("2100-01-01"))
            gap = (nxt - sd.values) / np.timedelta64(1, "D")
            f = gap <= (hz * 7.0 / 5.0) + 1.0
            idx = np.where(m)[0]
            flag[idx] = f & cov
            covered[idx] = cov
        d["flag"] = flag; d["cov"] = covered
        log(f"h={hz} earnings flag rate on covered rows: "
            f"{flag[covered].mean():.3f}  covered rows {covered.sum():,}")

    print(f"{'h':<4}{'n_flag':>9}{'n_unflag':>10}{'sig mult(resid)':>17}"
          f"{'pred_mv flag':>13}{'obs_mv flag':>12}{'gap':>8}")
    for hz in HORIZONS:
        d = cache[("single", hz)]
        a, p, pc, dtv, si = run_spec(cache, "single", hz, SPECS["FULL_single"])
        # align: run_spec drops pre-2006 rows; recompute masks on the same rows
        yr = d["yr"]; ey = end_year(d["dt"], d["sid"], hz)
        te_mask = np.zeros(len(yr), dtype=bool)
        for yt in sorted(set(yr))[MIN_TRAIN_YEARS:]:
            tr = (yr < yt) & (ey < yt)
            if tr.sum() < 2000:
                continue
            te_mask |= (yr == yt)
        fl = d["flag"][te_mask]; cv = d["cov"][te_mask]
        lr = d["lr"][te_mask]
        rv = np.exp(d["tg"][te_mask])
        mult = np.exp(np.mean(np.log(rv[fl & cv])) - np.mean(np.log(rv[cv & ~fl])))
        pm = p[:, 0] + p[:, 3]
        hit = ((a == 0) | (a == 3)).astype(float)
        print(f"{hz:<4}{int((fl&cv).sum()):>9,}{int((cv&~fl).sum()):>10,}{mult:>17.3f}"
              f"{pm[fl & cv].mean():>13.4f}{hit[fl & cv].mean():>12.4f}"
              f"{hit[fl&cv].mean()-pm[fl&cv].mean():>8.4f}")

    print("\n=== N. EARNINGS ADJUSTMENT: end-to-end effect (covered names only) ===")
    EM = {1: 1.60, 5: 1.30, 10: 1.18, 21: 1.10}
    print(f"{'h':<4}{'BSS4 base':>11}{'BSS4 +earn':>12}{'LL flag base':>14}"
          f"{'LL flag +earn':>15}{'LL flag clim':>14}")
    for hz in HORIZONS:
        d = cache[("single", hz)]
        a0, p0, pc0, _, _ = run_spec(cache, "single", hz, SPECS["FULL_single"])
        a1, p1, pc1, _, _ = run_spec(cache, "single", hz, SPECS["FULL_single"],
                                     earn=d["flag"], emult=EM)
        yr = d["yr"]; ey = end_year(d["dt"], d["sid"], hz)
        te_mask = np.zeros(len(yr), dtype=bool)
        for yt in sorted(set(yr))[MIN_TRAIN_YEARS:]:
            if ((yr < yt) & (ey < yt)).sum() >= 2000:
                te_mask |= (yr == yt)
        fl = d["flag"][te_mask] & d["cov"][te_mask]
        ll = lambda p, a: float(-np.log(np.clip(p[np.arange(len(a)), a], 1e-12, 1)).mean())
        print(f"{hz:<4}{bss(a0,p0,pc0):>11.4f}{bss(a1,p1,pc1):>12.4f}"
              f"{ll(p0[fl],a0[fl]):>14.4f}{ll(p1[fl],a1[fl]):>15.4f}{ll(pc0[fl],a0[fl]):>14.4f}")
    log("done")


if __name__ == "__main__":
    main()
