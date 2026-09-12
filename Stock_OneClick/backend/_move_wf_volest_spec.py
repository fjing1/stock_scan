"""
_move_wf_volest_spec.py — final deliverable for the volatility-forecast piece.

Prints, for the recommended specification and the fallback ladder:
  * the walk-forward-fitted coefficients trained on ALL data strictly before 2026 (the set the
    live system would use today), separately for the INDEX/ETF group and the single-name group
  * the out-of-sample residual sd in log space (the "vol-of-vol" the z-distribution step inherits)
  * the out-of-sample mean log residual (should be ~0: the forecast is unbiased in log space,
    i.e. it is a MEDIAN forecast; exp(sd^2/2) converts it to a mean if ever needed)
  * today's live sigma_hat for SPY at each horizon, per-day and annualized
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _move_data as D
import _move_lib as L
from _move_wf_volest import (FLOOR, CEIL, HORIZONS, IDX, NOT_SINGLE,
                             load_panel, build_estimators, ols)

SPEC = {
    "RECOMMENDED har4+yz21+ewma97": ["har_d", "har_w", "har_m", "har_q", "yz21", "ewma97"],
    "TRIM        har4+yz21":        ["har_d", "har_w", "har_m", "har_q", "yz21"],
    "NO-OHLC     HARc4":           ["harc_d", "harc_w", "harc_m", "harc_q"],
    "SHORT-HIST  ewma94":          ["ewma94"],
    "SHORT-HIST  ewma97":          ["ewma97"],
}


def main():
    o, hi, lo, c = load_panel()
    idx_cols = [s for s in IDX if s in c.columns]
    sng_cols = [s for s in c.columns if s not in NOT_SINGLE]
    E = build_estimators(o, hi, lo, c)
    years = c.index.year.values

    for hh in HORIZONS:
        tgt = (np.log(c).diff().abs().shift(-1) if hh == 1
               else L.realized_vol_forward(c, hh))
        print("=" * 112)
        print(f"HORIZON h={hh}"
              + ("   [target = |log r[t+1]| proxy; realized_vol_forward is NaN at h=1]"
                 if hh == 1 else ""))
        for uni, cols in (("INDEX/ETF", idx_cols), ("SINGLE-NAME", sng_cols)):
            T, S = len(c.index), len(cols)
            y = np.log(np.clip(tgt[cols].values.astype(float), FLOOR, CEIL)).ravel()
            yrv = np.repeat(years, S)
            for name, fl in SPEC.items():
                X = np.column_stack([np.log(np.clip(E[f][cols].values.astype(float),
                                                    FLOOR, CEIL)).ravel() for f in fl])
                ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
                yo, Xo, yro = y[ok], X[ok], yrv[ok]
                tr = yro < 2026
                beta = ols(Xo[tr], yo[tr])
                # OOS residual stats: pooled over walk-forward test years 2006..2026
                res = []
                for yy in range(2006, 2027):
                    a, b = yro < yy, yro == yy
                    if b.sum() < 50 or a.sum() < 500:
                        continue
                    bb = ols(Xo[a], yo[a])
                    res.append(yo[b] - (bb[0] + Xo[b] @ bb[1:]))
                res = np.concatenate(res)
                terms = " ".join(f"{bi:+.4f}*ln({f})" for bi, f in zip(beta[1:], fl))
                print(f"  {uni:11s} {name:28s} n_fit={int(tr.sum()):>9,}")
                print(f"      ln(sigma_hat_{hh}) = {beta[0]:+.4f} {terms}")
                print(f"      OOS log-residual: sd={res.std(ddof=1):.4f} "
                      f"mean={res.mean():+.4f}  median->mean factor exp(sd^2/2)="
                      f"{np.exp(res.var(ddof=1)/2):.4f}  n_oos={len(res):,}")
        print()

    # live SPY reading with the recommended index-fit spec
    print("=" * 112)
    print("LIVE SPY sigma_hat as of the last closed bar, RECOMMENDED spec, INDEX/ETF fit (<2026)")
    fl = SPEC["RECOMMENDED har4+yz21+ewma97"]
    xnow = np.array([np.log(np.clip(float(E[f]["SPY"].dropna().iloc[-1]), FLOOR, CEIL))
                     for f in fl])
    print(f"  last bar {c.index[-1].date()};  features: "
          + "  ".join(f"{f}={np.exp(v)*np.sqrt(252)*100:.2f}%ann" for f, v in zip(fl, xnow)))
    for hh in HORIZONS:
        tgt = (np.log(c).diff().abs().shift(-1) if hh == 1
               else L.realized_vol_forward(c, hh))
        y = np.log(np.clip(tgt[idx_cols].values.astype(float), FLOOR, CEIL)).ravel()
        X = np.column_stack([np.log(np.clip(E[f][idx_cols].values.astype(float),
                                            FLOOR, CEIL)).ravel() for f in fl])
        yrv = np.repeat(years, len(idx_cols))
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1) & (yrv < 2026)
        beta = ols(X[ok], y[ok])
        s = float(np.exp(beta[0] + xnow @ beta[1:]))
        print(f"  h={hh:2d}: sigma_hat = {s*100:.3f}%/day = {s*np.sqrt(252)*100:.2f}% annualized"
              f"   -> h-day sigma = {s*np.sqrt(hh)*100:.2f}%")


if __name__ == "__main__":
    main()
