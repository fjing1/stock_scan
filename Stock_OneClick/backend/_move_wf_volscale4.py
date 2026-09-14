"""
_move_wf_volscale4.py — extract the exact implementable constants for the recommended formula.

  sigma_h_hat(t) = C_h * g(q_t, h) * sqrt( LR + (sigma_t^2 - LR) * w_h ) * sqrt(h)

where sigma_t = vol_ewma(close, 0.94), q_t = quintile of vol_ewma/vol_cc(252),
w_h = phi_h*(1-phi_h^h)/(h*(1-phi_h)), and C_h absorbs estimator bias + the sub-sqrt(h)
aggregation factor kappa_h. Every constant is the MEDIAN of 21 walk-forward fits, each using only
data strictly prior to its test year. Final OOS check of the composed formula included.

Run: ../../vcp_env/bin/python _move_wf_volscale4.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _move_data as D
import _move_lib as L

HS = [1, 2, 3, 5, 10, 21, 42, 63]
HSF = [2, 3, 5, 10, 21, 42, 63]         # horizons where realized fwd vol exists
IDX = ["SPY", "QQQ", "^GSPC", "IWM", "DIA"]
MIN_OBS, FLOOR = 500, 1e-4
PHI = np.round(np.concatenate([np.arange(0.70, 0.99, 0.005),
                               np.arange(0.990, 0.9996, 0.0005)]), 4)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)


def pf(phi, h):
    return 1.0 if phi >= 1 else float(phi * (1 - phi ** h) / (h * (1 - phi)))


def qlike(a, f):
    x = np.maximum(f, FLOOR) ** 2 / np.maximum(a, FLOOR) ** 2
    return float(np.mean(x - np.log(x) - 1))


def main():
    p = D.load()
    close = p["Close"]
    keep = [s for s in close.columns if close[s].notna().sum() >= MIN_OBS and s != "^VIX"]
    close = close[keep]
    singles = [s for s in keep if s not in IDX]
    lr = np.log(close).diff()
    years = close.index.year.values
    yrs = sorted(set(years))
    s_e, s_l = L.vol_ewma(close, 0.94), L.vol_cc(close, 252)
    state = s_e / s_l
    fwd = {h: L.realized_vol_forward(close, h) for h in HSF}
    fret = {h: np.log(close.shift(-h) / close) for h in HS}

    for lab, cols in [("INDICES", IDX), ("SINGLES", singles)]:
        ci = [close.columns.get_loc(c) for c in cols]
        SE, SL, ST = (s_e.iloc[:, ci].values, s_l.iloc[:, ci].values, state.iloc[:, ci].values)
        yrow = np.repeat(years[:, None], len(ci), 1)
        acc = {h: [] for h in HS}
        oos = []
        for y in yrs[5:]:
            LR = float(np.nanmean(lr.iloc[:, ci].values[years < y] ** 2))
            base_ok = np.isfinite(SE) & (SE >= FLOOR) & np.isfinite(SL) & (SL >= FLOOR)
            for h in HS:
                RH = fret[h].iloc[:, ci].values
                if h in fwd:
                    FH = fwd[h].iloc[:, ci].values
                    gk = base_ok & np.isfinite(FH) & (FH >= FLOOR) & np.isfinite(RH)
                else:                                    # h=1: no realized fwd vol; use |r| proxy
                    FH = np.abs(RH) * np.sqrt(np.pi / 2)
                    gk = base_ok & np.isfinite(FH) & (FH >= FLOOR) & np.isfinite(RH)
                tr, te = gk & (yrow < y), gk & (yrow == y)
                if tr.sum() < 1500 or te.sum() < 100:
                    continue
                edges = np.percentile(ST[tr], [20, 40, 60, 80])
                q_ = np.empty(len(PHI))
                for i, phi in enumerate(PHI):
                    sh = np.sqrt(np.maximum(LR + (SE[tr] ** 2 - LR) * pf(phi, h), FLOOR ** 2))
                    q_[i] = qlike(sh * np.sqrt(np.mean(FH[tr] ** 2 / sh ** 2)), FH[tr])
                phi = PHI[int(np.nanargmin(q_))]
                w = pf(phi, h)
                bt = np.sqrt(np.maximum(LR + (SE[tr] ** 2 - LR) * w, FLOOR ** 2))
                be = np.sqrt(np.maximum(LR + (SE[te] ** 2 - LR) * w, FLOOR ** 2))
                qt, qe = np.digitize(ST[tr], edges), np.digitize(ST[te], edges)
                R = FH[tr] / bt
                g = np.array([np.sqrt(np.mean(R[qt == k] ** 2)) if (qt == k).sum() > 30 else 1.0
                              for k in range(5)])
                g = g / np.sqrt(np.mean(R ** 2))          # shape only; level goes into C_h
                # C_h: rms-match z on TRAIN so that sd(z)=1  (absorbs estimator bias + kappa_h)
                C = np.sqrt(np.mean(fret[h].iloc[:, ci].values[tr] ** 2
                                    / (h * (bt * g[qt]) ** 2)))
                acc[h].append(dict(y=y, phi=phi, w=w, C=C, LRsig=np.sqrt(LR),
                                   **{f"g{k+1}": g[k] for k in range(5)},
                                   **{f"e{k+1}": edges[k] for k in range(4)}))
                # OOS check of the composed formula
                z = fret[h].iloc[:, ci].values[te] / (C * be * g[qe] * np.sqrt(h))
                for k in range(5):
                    s = qe == k
                    if s.sum() < 20:
                        continue
                    oos.append(dict(h=h, y=y, q=k + 1, n=int(s.sum()),
                                    sq=float(np.mean(z[s] ** 2)),
                                    hit=float(np.mean(np.abs(z[s]) > 1.5))))
        print("\n" + "=" * 106)
        print(f"{lab}: RECOMMENDED CONSTANTS (median of {len(acc[21])} walk-forward fits)")
        print("=" * 106)
        T = pd.DataFrame({h: pd.DataFrame(acc[h]).median(numeric_only=True) for h in HS}).T
        T.index.name = "h"
        T["HL_days"] = np.log(.5) / np.log(T.phi)
        print(T[["phi", "HL_days", "w", "C", "g1", "g2", "g3", "g4", "g5",
                 "e1", "e2", "e3", "e4", "LRsig"]].round(4).to_string())
        print("  phi range across walk-forward fits: " + "  ".join(
            f"h{h}:[{pd.DataFrame(acc[h]).phi.min():.4f},{pd.DataFrame(acc[h]).phi.max():.4f}]"
            for h in HS))
        O = pd.DataFrame(oos)
        piv = O.groupby(["h", "q"]).apply(lambda x: np.sqrt(np.average(x.sq, weights=x.n)),
                                          include_groups=False).unstack()
        piv.columns = [f"Q{c}" for c in piv.columns]
        piv["Q5/Q1"] = piv.Q5 / piv.Q1
        piv["mean|sd-1|"] = (piv[[f"Q{k}" for k in range(1, 6)]] - 1).abs().mean(axis=1)
        piv["n"] = O.groupby("h").n.sum()
        print(f"\n  OOS check of the composed formula -- sd(z) by state quintile "
              f"(target 1.000 everywhere):")
        print(piv.round(4).to_string())
        hv = O.groupby(["h", "q"]).apply(lambda x: np.average(x.hit, weights=x.n),
                                         include_groups=False).unstack()
        hv.columns = [f"Q{c}" for c in hv.columns]
        hv["Q5-Q1"] = hv.Q5 - hv.Q1
        print(f"\n  OOS P(|z|>1.5) by state quintile (target: flat across quintiles):")
        print(hv.round(4).to_string())


if __name__ == "__main__":
    main()
