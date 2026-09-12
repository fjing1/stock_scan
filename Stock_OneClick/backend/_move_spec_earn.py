"""
_move_spec_earn.py -- earnings adjustment measured against the SHIPPED sigma, with exact
trading-bar flag construction and exact row alignment.
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


def main():
    p = D.load()
    O, H, Lo, C = p["Open"].copy(), p["High"].copy(), p["Low"].copy(), p["Close"].copy()
    for df in (O, H, Lo, C):
        df.mask(df <= 0, inplace=True)
    cov = C.notna().sum()
    syms = [s for s in C.columns if s != "^VIX" and cov[s] >= 500 and s not in INDEX_SYMS]
    ecache = json.load(open("../../gold_pine_script/pead_earnings_cache.json"))
    covered_syms = [s for s in syms if len(ecache.get(s) or []) >= 5]
    log(f"{len(syms)} singles, {len(covered_syms)} with >=5 earnings events")

    for hz in HORIZONS:
        F, LR, TG, YR, SID, DT, FLG, CVD = [], [], [], [], [], [], [], []
        for i, s in enumerate(syms):
            c = C[s].dropna()
            if len(c) < 400:
                continue
            o, hh, ll = O[s].reindex(c.index), H[s].reindex(c.index), Lo[s].reindex(c.index)
            rv1 = L.vol_parkinson(hh, ll, 1)
            f = np.column_stack([
                clip_log(rv1), clip_log(rv1.rolling(5).mean()), clip_log(rv1.rolling(22).mean()),
                clip_log(rv1.rolling(63).mean()),
                clip_log(L.vol_yang_zhang(o, hh, ll, c, 21)), clip_log(L.vol_ewma(c, 0.97))])
            lr = np.log(c.shift(-hz) / c).values
            tg = clip_log(L.realized_vol_forward(c, hz)).values
            # exact trading-bar earnings flag: announcement bar b with b in [t, t+h]
            ev = ecache.get(s) or []
            flag = np.zeros(len(c), dtype=bool)
            cvd = np.zeros(len(c), dtype=bool)
            if len(ev) >= 5:
                ed = pd.DatetimeIndex(sorted(pd.to_datetime([e[0] for e in ev])))
                bpos = np.unique(np.clip(c.index.searchsorted(ed, side="left"), 0, len(c) - 1))
                isann = np.zeros(len(c), dtype=bool)
                isann[bpos] = True
                # flag[t] = any announcement bar in [t, t+h]
                cs = np.concatenate([[0], np.cumsum(isann)])
                tt = np.arange(len(c))
                hi = np.minimum(tt + hz, len(c) - 1)
                flag = (cs[hi + 1] - cs[tt]) > 0
                cvd = (c.index >= ed.min()) & (c.index <= ed.max())
            ok = np.isfinite(f).all(axis=1) & np.isfinite(lr) & np.isfinite(tg)
            if ok.sum() < 100:
                continue
            F.append(f[ok]); LR.append(lr[ok]); TG.append(tg[ok])
            YR.append(c.index.year.values[ok]); SID.append(np.full(int(ok.sum()), i))
            DT.append(c.index.values[ok]); FLG.append(flag[ok]); CVD.append(cvd[ok])
        F = np.vstack(F); lr = np.concatenate(LR); tg = np.concatenate(TG)
        yr = np.concatenate(YR); sid = np.concatenate(SID); dt = np.concatenate(DT)
        flag = np.concatenate(FLG); cvd = np.concatenate(CVD)
        run(hz, F, lr, tg, yr, sid, dt, flag, cvd)


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


def ols(X, y):
    G = X.T @ X
    return np.linalg.solve(G + 1e-10 * np.eye(G.shape[0]), X.T @ y)


def brier_rows(p, a):
    oh = np.zeros_like(p); oh[np.arange(len(a)), a] = 1.0
    return ((p - oh) ** 2).sum(axis=1)


def ll_rows(p, a):
    return -np.log(np.clip(p[np.arange(len(a)), a], 1e-12, 1))


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


def run(hz, F, lr, tg, yr, sid, dt, flag, cvd, thr=0.02):
    X = np.column_stack([np.ones(len(F)), F])
    ey = end_year(dt, sid, hz)
    rowsA, rowsP, rowsPE, rowsPC, rowsIdx = [], [], [], [], []
    EM_grid = {}
    for yt in sorted(set(yr))[MIN_TRAIN_YEARS:]:
        tr = (yr < yt) & (ey < yt); te = yr == yt
        if tr.sum() < 2000 or te.sum() == 0:
            continue
        b = ols(X[tr], tg[tr])
        s_tr = np.exp(X[tr] @ b) * np.sqrt(hz)
        s_te = np.exp(X[te] @ b) * np.sqrt(hz)
        # earnings multiplier fit on TRAIN only: geometric-mean ratio of realized/forecast
        ftr, ctr = flag[tr], cvd[tr]
        rvtr = np.exp(tg[tr])
        sel = ctr
        num = np.mean(np.log(rvtr[sel & ftr]) - np.log(s_tr[sel & ftr] / np.sqrt(hz)))
        den = np.mean(np.log(rvtr[sel & ~ftr]) - np.log(s_tr[sel & ~ftr] / np.sqrt(hz)))
        em = float(np.exp(num - den))
        EM_grid[yt] = em
        s_te_e = s_te * np.where(flag[te] & cvd[te], em, 1.0)
        s_tr_e = s_tr * np.where(ftr & ctr, em, 1.0)
        a_tr = bucket_idx(lr[tr], thr); a_te = bucket_idx(lr[te], thr)
        pooled = np.bincount(a_tr, minlength=4) / len(a_tr)
        nsym = int(sid.max()) + 1
        cnt = np.zeros((nsym, 4)); np.add.at(cnt, (sid[tr], a_tr), 1.0)
        psym = (cnt + SHRINK * pooled) / (cnt.sum(axis=1, keepdims=True) + SHRINK)

        def build(s_tr_x, s_te_x):
            z = lr[tr] / s_tr_x
            sd = s_tr_x / np.sqrt(hz)
            edges = np.percentile(sd, [20, 40, 60, 80])
            qtr = np.digitize(sd, edges); qte = np.digitize(s_te_x / np.sqrt(hz), edges)
            out = np.zeros((int(te.sum()), 4))
            for q in range(5):
                m = qte == q
                if m.any():
                    out[m] = probs_emp(np.sort(z[qtr == q]), s_te_x[m], thr)
            return out

        rowsA.append(a_te); rowsP.append(build(s_tr, s_te)); rowsPE.append(build(s_tr_e, s_te_e))
        rowsPC.append(psym[sid[te]]); rowsIdx.append(np.where(te)[0])
    a = np.concatenate(rowsA); p = np.vstack(rowsP); pe = np.vstack(rowsPE)
    pc = np.vstack(rowsPC); ii = np.concatenate(rowsIdx)
    fl = flag[ii] & cvd[ii]; cv = cvd[ii]
    rv = np.exp(tg[ii])
    raw_mult = float(np.exp(np.mean(np.log(rv[cv & fl])) - np.mean(np.log(rv[cv & ~fl]))))
    hit = ((a == 0) | (a == 3)).astype(float)
    pm, pme = p[:, 0] + p[:, 3], pe[:, 0] + pe[:, 3]
    bss = lambda q, m: float(1 - brier_rows(q[m], a[m]).mean() / brier_rows(pc[m], a[m]).mean())
    print(f"h={hz:<3} EM(train,mean)={np.mean(list(EM_grid.values())):.3f} "
          f"[{min(EM_grid.values()):.3f},{max(EM_grid.values()):.3f}]  raw_rv_mult={raw_mult:.3f}")
    print(f"      flagged n={int(fl.sum()):,}  pred_mv base={pm[fl].mean():.4f} "
          f"+earn={pme[fl].mean():.4f}  obs={hit[fl].mean():.4f}  "
          f"gap base={hit[fl].mean()-pm[fl].mean():+.4f} +earn={hit[fl].mean()-pme[fl].mean():+.4f}")
    nf = cv & ~fl
    print(f"      unflagged n={int(nf.sum()):,} pred_mv base={pm[nf].mean():.4f} "
          f"+earn={pme[nf].mean():.4f} obs={hit[nf].mean():.4f} "
          f"gap base={hit[nf].mean()-pm[nf].mean():+.4f} +earn={hit[nf].mean()-pme[nf].mean():+.4f}")
    print(f"      LL flagged base={ll_rows(p[fl],a[fl]).mean():.4f} "
          f"+earn={ll_rows(pe[fl],a[fl]).mean():.4f} clim={ll_rows(pc[fl],a[fl]).mean():.4f}   "
          f"BSS4 all base={bss(p,cv):.4f} +earn={bss(pe,cv):.4f}")


if __name__ == "__main__":
    main()
