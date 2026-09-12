"""
_move_spec_grid.py -- settle the remaining design decisions for move_prob.py and produce the
numbers that go into the final implementation spec.

Walk-forward by calendar year, expanding window, every parameter fit on strictly prior years.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

import _move_data as D
import _move_lib as L

VOL_CLIP = (1e-3, 0.5)
INDEX_SYMS = {"SPY", "QQQ", "IWM", "DIA", "^GSPC"}
HORIZONS = (1, 5, 10, 21)
MIN_TRAIN_YEARS = 5
SHRINK = 40.0

t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)


def clip_log(x):
    return np.log(np.clip(x, *VOL_CLIP))


def setup():
    p = D.load()
    O, H, Lo, C = p["Open"].copy(), p["High"].copy(), p["Low"].copy(), p["Close"].copy()
    nbad = 0
    for df in (O, H, Lo, C):
        m = df <= 0
        nbad += int(m.sum().sum())
        df.mask(m, inplace=True)
    log(f"blanked {nbad:,} non-positive price cells")
    cov = C.notna().sum()
    usable = [s for s in C.columns if s != "^VIX" and cov[s] >= 500]
    vix = p["Close"]["^VIX"]
    idx = [s for s in usable if s in INDEX_SYMS]
    sng = [s for s in usable if s not in INDEX_SYMS]
    log(f"usable {len(usable)} = {len(idx)} index + {len(sng)} single")
    return O, H, Lo, C, vix, {"index": idx, "single": sng}


# feature column order
FN = ["rv_d", "rv_w", "rv_m", "rv_q", "yz21", "e97", "vix", "e94", "cc21"]


def features(o, h, l, c, vix):
    rv1 = L.vol_parkinson(h, l, 1)
    return np.column_stack([
        clip_log(rv1),
        clip_log(rv1.rolling(5).mean()),
        clip_log(rv1.rolling(22).mean()),
        clip_log(rv1.rolling(63).mean()),
        clip_log(L.vol_yang_zhang(o, h, l, c, 21)),
        clip_log(L.vol_ewma(c, 0.97)),
        clip_log(vix.reindex(c.index).ffill(limit=3) / 100.0 / np.sqrt(252.0)),
        clip_log(L.vol_ewma(c, 0.94)),
        clip_log(L.vol_cc(c, 21)),
    ])


def build_cache(O, H, Lo, C, vix, groups):
    cache = {}
    for g, syms in groups.items():
        for hz in HORIZONS:
            F, LR, TG, YR, SID, DT = [], [], [], [], [], []
            for i, s in enumerate(syms):
                c = C[s].dropna()
                if len(c) < 400:
                    continue
                o, hh, ll = O[s].reindex(c.index), H[s].reindex(c.index), Lo[s].reindex(c.index)
                f = features(o, hh, ll, c, vix)
                lr = np.log(c.shift(-hz) / c).values
                tg = clip_log(L.realized_vol_forward(c, hz)).values
                ok = np.isfinite(f).all(axis=1) & np.isfinite(lr) & np.isfinite(tg)
                if ok.sum() < 100:
                    continue
                F.append(f[ok]); LR.append(lr[ok]); TG.append(tg[ok])
                YR.append(c.index.year.values[ok]); SID.append(np.full(int(ok.sum()), i))
                DT.append(c.index.values[ok])
            cache[(g, hz)] = dict(
                F=np.vstack(F).astype(np.float64), lr=np.concatenate(LR),
                tg=np.concatenate(TG), yr=np.concatenate(YR),
                sid=np.concatenate(SID), dt=np.concatenate(DT),
                syms=syms)
            log(f"cache {g:<6} h={hz:<3} rows={len(cache[(g,hz)]['lr']):>9,} "
                f"nsym={len(set(cache[(g,hz)]['sid']))}")
    return cache


def ols(X, y):
    G = X.T @ X
    return np.linalg.solve(G + 1e-10 * np.eye(G.shape[0]), X.T @ y)


SPECS = {
    "har3+yz21":         [0, 1, 2, 4],
    "har4+yz21":         [0, 1, 2, 3, 4],
    "har4+yz21+e97":     [0, 1, 2, 3, 4, 5],
    "har3+yz21+vix":     [0, 1, 2, 4, 6],
    "har4+yz21+e97+vix": [0, 1, 2, 3, 4, 5, 6],
    "e94_affine":        [7],
    "e97_affine":        [5],
    "cc21_affine":       [8],
    "vix_affine":        [6],
}


def design(F, cols):
    return np.column_stack([np.ones(len(F))] + [F[:, c] for c in cols])


def wf_pred(cache, g, hz, cols):
    d = cache[(g, hz)]
    X = design(d["F"], cols)
    y, yr = d["tg"], d["yr"]
    years = sorted(set(yr))
    pred = np.full(len(y), np.nan)
    base = np.full(len(y), np.nan)
    for yt in years[MIN_TRAIN_YEARS:]:
        tr, te = yr < yt, yr == yt
        if tr.sum() < 500 or te.sum() == 0:
            continue
        b = ols(X[tr], y[tr])
        pred[te] = X[te] @ b
        base[te] = y[tr].mean()
    return pred, base, np.isfinite(pred)


def r2(y, p, b):
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - b) ** 2))


def main():
    O, H, Lo, C, vix, groups = setup()
    cache = build_cache(O, H, Lo, C, vix, groups)
    pd.to_pickle({k: v for k, v in cache.items()}, "_move_spec_cache.pkl")

    names = ["har3+yz21", "har4+yz21", "har4+yz21+e97", "har3+yz21+vix",
             "har4+yz21+e97+vix", "e94_affine", "e97_affine", "cc21_affine", "vix_affine"]
    print("\n=== 2. VOL SPEC HORSE RACE: OOS R2 on log forward per-day sigma ===")
    print("(walk-forward by year, test 2006-2026, baseline = train-period mean of log target)")
    hdr = f"{'group':<7}{'h':<4}{'n':>10}" + "".join(f"{k:>19}" for k in names)
    print(hdr)
    store = {}
    for g in ("index", "single"):
        for hz in HORIZONS:
            d = cache[(g, hz)]
            row, ms = [], None
            preds = {}
            for nm in names:
                p, b, m = wf_pred(cache, g, hz, SPECS[nm])
                preds[nm] = (p, b, m)
                ms = m if ms is None else (ms & m)
            for nm in names:
                p, b, m = preds[nm]
                row.append(r2(d["tg"][ms], p[ms], b[ms]))
            store[(g, hz)] = preds
            print(f"{g:<7}{hz:<4}{int(ms.sum()):>10,}" + "".join(f"{x:>19.4f}" for x in row))
    pd.to_pickle(store, "_move_spec_volpred.pkl")
    log("done")


if __name__ == "__main__":
    main()
