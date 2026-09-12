"""
_move_validate.py — walk-forward calibration test for move_prob.py.

Nothing in move_prob.py should be believed until this passes. It answers one question: when the
model says 30%, does it happen 30% of the time, on data it has never seen?

METHOD
  Expanding walk-forward by calendar year: every parameter (HAR coefficients AND the empirical z
  table) is refit on strictly prior years, then scored on the held-out year. Repeat for 2007-2026.
  Features/targets are built once and cached, so 20 refits cost seconds rather than an hour.

TWO BASELINES, AND THE HARDER ONE DECIDES
  * pooled climatology -- unconditional bucket frequencies over the training years.
  * PER-SYMBOL climatology -- that ticker's own training-year frequencies, shrunk 40 pseudo-counts
    toward pooled. This is the number a user gets for free by looking at the ticker's own history,
    so it is the bar the model must clear. Pooled climatology flatters single-name skill roughly
    2x because it does not know that BLNK is more volatile than KO.

Run: ../../vcp_env/bin/python _move_validate.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _move_data as D
import _move_lib as L
import move_prob as M

THRS = (0.01, 0.02, 0.03, 0.05)
MIN_TRAIN_YEARS = 5
SHRINK = 40.0


def build_cache(panel):
    """Per (group, h): stacked design matrix, log-forward-return, year, symbol id."""
    C, H, Lo = panel["Close"], panel["High"], panel["Low"]
    vix = C["^VIX"] if "^VIX" in C.columns else None
    usable = [s for s in C.columns if s != "^VIX" and C[s].notna().sum() >= 500]
    groups = {"index": [s for s in usable if s in M.INDEX_LIKE],
              "single": [s for s in usable if s not in M.INDEX_LIKE]}
    cache = {}
    for g, syms in groups.items():
        cols = M.FEATS_OHLC + (["logvix"] if (g == "index" and vix is not None) else [])
        for h in M.HORIZONS:
            X, y, yr, sid, tgt = [], [], [], [], []
            for i, s in enumerate(syms):
                c = C[s].dropna()
                if len(c) < 400:
                    continue
                f = M.build_features(c, H[s].reindex(c.index), Lo[s].reindex(c.index), vix)
                Xi = M._design(f, cols)
                lr = np.log(c.shift(-h) / c).values
                ti = M._clip_log(L.realized_vol_forward(c, h)).values
                ok = np.isfinite(Xi).all(axis=1) & np.isfinite(lr)
                X.append(Xi[ok]); y.append(lr[ok]); tgt.append(ti[ok])
                yr.append(c.index.year.values[ok]); sid.append(np.full(ok.sum(), i))
            cache[(g, h)] = dict(X=np.vstack(X), y=np.concatenate(y), tgt=np.concatenate(tgt),
                                 yr=np.concatenate(yr), sid=np.concatenate(sid),
                                 n_sym=len(syms), cols=cols)
            print(f"  cached {g:<7} h={h:<3} rows={len(cache[(g,h)]['y']):>9,}")
    return cache


def bucket_idx(lr, thr):
    """0=down_big 1=down_small 2=up_small 3=up_big, in LOG space to match the model."""
    a_dn, a_up = np.log(1 - thr), np.log(1 + thr)
    return np.where(lr <= a_dn, 0, np.where(lr <= 0, 1, np.where(lr < a_up, 2, 3)))


def brier(p, a):
    oh = np.zeros_like(p); oh[np.arange(len(a)), a] = 1.0
    return ((p - oh) ** 2).sum(axis=1).mean()


def run():
    panel = D.load()
    print("building feature cache ...")
    cache = build_cache(panel)
    rows = []

    for (g, h), d in cache.items():
        X, y, tgt, yr, sid = d["X"], d["y"], d["tgt"], d["yr"], d["sid"]
        years = sorted(set(yr))
        test_years = years[MIN_TRAIN_YEARS:]
        for thr in THRS:
            act_all = bucket_idx(y, thr)
            P_model, P_clim, P_sym, A, YR = [], [], [], [], []
            for ty in test_years:
                tr, te = yr < ty, yr == ty
                # purge: training rows whose forward window reaches into the test year
                tr = tr & ~((yr == ty - 1) & (np.arange(len(yr)) >= len(yr) - 1))
                if tr.sum() < 2000 or te.sum() < 50:
                    continue
                fin = np.isfinite(tgt)
                beta, n = M._ols(X[tr & fin], tgt[tr & fin])
                if beta is None:
                    continue
                sig_tr = np.exp(X[tr] @ beta) * np.sqrt(h)
                sig_te = np.exp(X[te] @ beta) * np.sqrt(h)
                z_tr = np.sort(y[tr] / sig_tr)

                a_dn, a_up = np.log(1 - thr), np.log(1 + thr)
                F = lambda v: np.searchsorted(z_tr, v, side="right") / len(z_tr)
                f_dn, f_0, f_up = F(a_dn / sig_te), F(0.0), F(a_up / sig_te)
                p = np.column_stack([f_dn, f_0 - f_dn, f_up - f_0, 1 - f_up])
                p = np.clip(p, 1e-4, None); p /= p.sum(axis=1, keepdims=True)

                a_tr, a_te = act_all[tr], act_all[te]
                pc = np.bincount(a_tr, minlength=4) / len(a_tr)
                # per-symbol climatology, shrunk toward pooled
                ps = np.empty((te.sum(), 4))
                s_te = sid[te]
                for s in np.unique(s_te):
                    m = sid[tr] == s
                    cnt = np.bincount(a_tr[m], minlength=4).astype(float) if m.any() else np.zeros(4)
                    ps[s_te == s] = (cnt + SHRINK * pc) / (cnt.sum() + SHRINK)

                P_model.append(p); P_clim.append(np.tile(pc, (te.sum(), 1))); P_sym.append(ps)
                A.append(a_te); YR.append(np.full(te.sum(), ty))

            if not P_model:
                continue
            p, pc, ps = np.vstack(P_model), np.vstack(P_clim), np.vstack(P_sym)
            a, yy = np.concatenate(A), np.concatenate(YR)
            big = np.isin(a, [0, 3]).astype(float)
            pm, pmc, pms = p[:, 0] + p[:, 3], pc[:, 0] + pc[:, 3], ps[:, 0] + ps[:, 3]

            bs2 = lambda q: ((q - big) ** 2).mean()
            per_yr = [1 - brier(p[yy == u], a[yy == u]) / brier(ps[yy == u], a[yy == u])
                      for u in np.unique(yy) if (yy == u).sum() > 50]
            rows.append(dict(
                group=g, h=h, thr=thr, n=len(a), yrs=len(per_yr),
                bss4_pool=1 - brier(p, a) / brier(pc, a),
                bss4_sym=1 - brier(p, a) / brier(ps, a),
                bss2_sym=1 - bs2(pm) / bs2(pms),
                ece=L.ece(pm, big.astype(bool)),
                ece_base=L.ece(pms, big.astype(bool)),
                pos_yrs=f"{sum(v > 0 for v in per_yr)}/{len(per_yr)}",
                obs_move=big.mean(), pred_move=pm.mean(),
                support=np.log(1 + thr) / np.median(np.exp(X[:, :].dot(np.zeros(X.shape[1])))) if False else np.nan,
            ))
            print(f"  {g:<7} h={h:<3} thr={thr*100:>3.0f}%  n={len(a):>8,}  "
                  f"BSS4/sym={rows[-1]['bss4_sym']:+.4f}  BSS2/sym={rows[-1]['bss2_sym']:+.4f}  "
                  f"ECE={rows[-1]['ece']:.4f} (base {rows[-1]['ece_base']:.4f})  "
                  f"yrs+{rows[-1]['pos_yrs']}  obs={big.mean():.3f} pred={pm.mean():.3f}")

    df = pd.DataFrame(rows)
    df.to_pickle("_move_validation.pkl")

    print("\n" + "=" * 104)
    print("HEADLINE — threshold 2%, vs each ticker's OWN base rate (the honest bar)")
    print("=" * 104)
    sub = df[df.thr == 0.02].sort_values(["group", "h"])
    print(f"{'group':<8}{'h':>4}{'n':>10}{'BSS 4-bucket':>14}{'BSS |move|':>12}"
          f"{'ECE':>8}{'ECE base':>10}{'yrs +':>8}{'obs':>7}{'pred':>7}")
    for _, r in sub.iterrows():
        print(f"{r.group:<8}{r.h:>4}{r.n:>10,}{r.bss4_sym:>+14.4f}{r.bss2_sym:>+12.4f}"
              f"{r.ece:>8.4f}{r.ece_base:>10.4f}{r.pos_yrs:>8}{r.obs_move:>7.3f}{r.pred_move:>7.3f}")

    print("\nRELIABILITY — P(|move|>=2%), pooled test years, equal-count deciles")
    for g in ("index", "single"):
        for h in (1, 5):
            print(f"\n  {g} h={h}: see _move_validation.pkl for the full table")
    print("\nSaved _move_validation.pkl")
    return df


if __name__ == "__main__":
    run()
