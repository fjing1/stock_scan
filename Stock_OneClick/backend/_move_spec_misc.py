"""
_move_spec_misc.py -- resolve two spec ambiguities: (a) quintile assignment before or after the
earnings multiplier; (b) exact minimum-bar requirements for each ladder rung.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

import _move_data as D
import _move_lib as L

VOL_CLIP = (1e-3, 0.5)
MIN_TRAIN_YEARS = 5
SHRINK = 40.0
HORIZONS = (1, 5, 10, 21)
INDEX_SYMS = {"SPY", "QQQ", "IWM", "DIA", "^GSPC"}


def clip_log(x):
    return np.log(np.clip(x, *VOL_CLIP))


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
    return np.column_stack([
        np.searchsorted(zs, np.log(1 - thr) / sig, side="right") / n,
        np.searchsorted(zs, 0.0, side="right") / n -
        np.searchsorted(zs, np.log(1 - thr) / sig, side="right") / n,
        np.searchsorted(zs, np.log(1 + thr) / sig, side="right") / n -
        np.searchsorted(zs, 0.0, side="right") / n,
        1 - np.searchsorted(zs, np.log(1 + thr) / sig, side="right") / n])


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


def main():
    p = D.load()
    O, H, Lo, C = p["Open"].copy(), p["High"].copy(), p["Low"].copy(), p["Close"].copy()
    for df in (O, H, Lo, C):
        df.mask(df <= 0, inplace=True)
    cov = C.notna().sum()
    syms = [s for s in C.columns if s != "^VIX" and cov[s] >= 500 and s not in INDEX_SYMS]
    ecache = json.load(open("../../gold_pine_script/pead_earnings_cache.json"))

    print("=== T. MIN BARS: first index at which each feature is finite (SPY) ===")
    c = C["SPY"].dropna()
    o, hh, ll = O["SPY"].reindex(c.index), H["SPY"].reindex(c.index), Lo["SPY"].reindex(c.index)
    rv1 = L.vol_parkinson(hh, ll, 1)
    cd = np.log(c).diff().abs()
    feats = {"rv_d": rv1, "rv_w": rv1.rolling(5).mean(), "rv_m": rv1.rolling(22).mean(),
             "rv_q": rv1.rolling(63).mean(), "yz21": L.vol_yang_zhang(o, hh, ll, c, 21),
             "e97": L.vol_ewma(c, 0.97), "e94": L.vol_ewma(c, 0.94),
             "cd_d": cd, "cd_w": cd.rolling(5).mean(), "cd_m": cd.rolling(22).mean(),
             "cd_q": cd.rolling(63).mean(), "park21": L.vol_parkinson(hh, ll, 21)}
    for k, v in feats.items():
        v = pd.Series(np.asarray(v, dtype=float), index=c.index)
        first = int(np.argmax(np.isfinite(v.values)))
        print(f"   {k:<8} first finite at bar #{first+1}")

    print("\n=== U. QUINTILE ASSIGNMENT: before vs after the earnings multiplier ===")
    EMs = {}
    for hz in HORIZONS:
        F, LR, TG, YR, SID, DT, FLG, CVD = [], [], [], [], [], [], [], []
        for i, s in enumerate(syms):
            cc = C[s].dropna()
            if len(cc) < 400:
                continue
            oo, h2, l2 = O[s].reindex(cc.index), H[s].reindex(cc.index), Lo[s].reindex(cc.index)
            r1 = L.vol_parkinson(h2, l2, 1)
            f = np.column_stack([clip_log(r1), clip_log(r1.rolling(5).mean()),
                                 clip_log(r1.rolling(22).mean()), clip_log(r1.rolling(63).mean()),
                                 clip_log(L.vol_yang_zhang(oo, h2, l2, cc, 21)),
                                 clip_log(L.vol_ewma(cc, 0.97))])
            lr = np.log(cc.shift(-hz) / cc).values
            tg = clip_log(L.realized_vol_forward(cc, hz)).values
            ev = ecache.get(s) or []
            flag = np.zeros(len(cc), dtype=bool); cvd = np.zeros(len(cc), dtype=bool)
            if len(ev) >= 5:
                ed = pd.DatetimeIndex(sorted(pd.to_datetime([e[0] for e in ev])))
                bp = np.unique(np.clip(cc.index.searchsorted(ed, side="left"), 0, len(cc) - 1))
                isann = np.zeros(len(cc), dtype=bool); isann[bp] = True
                cs = np.concatenate([[0], np.cumsum(isann)])
                tt = np.arange(len(cc)); hi = np.minimum(tt + hz, len(cc) - 1)
                flag = (cs[hi + 1] - cs[tt]) > 0
                cvd = (cc.index >= ed.min()) & (cc.index <= ed.max())
            ok = np.isfinite(f).all(axis=1) & np.isfinite(lr) & np.isfinite(tg)
            if ok.sum() < 100:
                continue
            F.append(f[ok]); LR.append(lr[ok]); TG.append(tg[ok])
            YR.append(cc.index.year.values[ok]); SID.append(np.full(int(ok.sum()), i))
            DT.append(cc.index.values[ok]); FLG.append(flag[ok]); CVD.append(cvd[ok])
        F = np.vstack(F); lr = np.concatenate(LR); tg = np.concatenate(TG)
        yr = np.concatenate(YR); sid = np.concatenate(SID); dt = np.concatenate(DT)
        flag = np.concatenate(FLG); cvd = np.concatenate(CVD)
        X = np.column_stack([np.ones(len(F)), F]); ey = end_year(dt, sid, hz)
        thr = 0.02
        out = {}
        for mode in ("q_before", "q_after"):
            A, P, PC, FL = [], [], [], []
            ems = []
            for yt in sorted(set(yr))[MIN_TRAIN_YEARS:]:
                tr = (yr < yt) & (ey < yt); te = yr == yt
                if tr.sum() < 2000 or te.sum() == 0:
                    continue
                b = ols(X[tr], tg[tr])
                s_tr = np.exp(X[tr] @ b) * np.sqrt(hz); s_te = np.exp(X[te] @ b) * np.sqrt(hz)
                ftr, ctr = flag[tr], cvd[tr]
                rvt = np.exp(tg[tr])
                em = float(np.exp(np.mean(np.log(rvt[ctr & ftr]) -
                                          np.log(s_tr[ctr & ftr] / np.sqrt(hz))) -
                                  np.mean(np.log(rvt[ctr & ~ftr]) -
                                          np.log(s_tr[ctr & ~ftr] / np.sqrt(hz)))))
                ems.append(em)
                mtr = np.where(ftr & ctr, em, 1.0); mte = np.where(flag[te] & cvd[te], em, 1.0)
                se_tr, se_te = s_tr * mtr, s_te * mte
                base_tr = (se_tr if mode == "q_after" else s_tr) / np.sqrt(hz)
                base_te = (se_te if mode == "q_after" else s_te) / np.sqrt(hz)
                edges = np.percentile(base_tr, [20, 40, 60, 80])
                qtr = np.digitize(base_tr, edges); qte = np.digitize(base_te, edges)
                z = lr[tr] / se_tr
                a_tr = bucket_idx(lr[tr], thr); a_te = bucket_idx(lr[te], thr)
                pooled = np.bincount(a_tr, minlength=4) / len(a_tr)
                nsym = int(sid.max()) + 1
                cnt = np.zeros((nsym, 4)); np.add.at(cnt, (sid[tr], a_tr), 1.0)
                psym = (cnt + SHRINK * pooled) / (cnt.sum(axis=1, keepdims=True) + SHRINK)
                pp = np.zeros((int(te.sum()), 4))
                for q in range(5):
                    m = qte == q
                    if m.any():
                        pp[m] = probs_emp(np.sort(z[qtr == q]), se_te[m], thr)
                A.append(a_te); P.append(norm4(pp)); PC.append(psym[sid[te]])
                FL.append(flag[te] & cvd[te])
            a = np.concatenate(A); pr = np.vstack(P); pc = np.vstack(PC); fl = np.concatenate(FL)
            bm = ((pr - np.eye(4)[a]) ** 2).sum(axis=1).mean()
            bc = ((pc - np.eye(4)[a]) ** 2).sum(axis=1).mean()
            pm = pr[:, 0] + pr[:, 3]; hit = ((a == 0) | (a == 3)).astype(float)
            out[mode] = (1 - bm / bc, hit[fl].mean() - pm[fl].mean(), np.mean(ems))
        print(f"h={hz:<3} q_before: BSS4={out['q_before'][0]:.4f} flaggap={out['q_before'][1]:+.4f}"
              f"   q_after: BSS4={out['q_after'][0]:.4f} flaggap={out['q_after'][1]:+.4f}"
              f"   EM_mean={out['q_before'][2]:.3f}")
        EMs[hz] = out["q_before"][2]
    print("\nwalk-forward mean earnings multipliers:", {k: round(v, 3) for k, v in EMs.items()})


if __name__ == "__main__":
    main()
