"""
_vix_wf_bandvslevel4.py — last competing explanations for the below-band effect:
band width (the sd in the band's own formula), SPX realized vol, and SPX trailing momentum.

Run: ../../vcp_env/bin/python _vix_wf_bandvslevel4.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _vix_data

RNG = np.random.default_rng(20260913)
N_ROT = 5000


def stars(p):
    return "***" if p < 0.01 else ("** " if p < 0.05 else ("*  " if p < 0.10 else "   "))


def fwl(y_res, dvec, Q):
    dr = dvec - Q @ (Q.T @ dvec)
    den = float(dr @ dr)
    return float(dr @ y_res) / den if den > 1e-12 else np.nan


def fe_matrix(n, fe_list, extra=None):
    cols = [np.ones(n)]
    for codes in fe_list:
        codes = np.asarray(codes)
        for k in np.unique(codes)[1:]:
            cols.append((codes == k).astype(float))
    if extra is not None:
        cols.extend([np.asarray(c, float) for c in extra])
    return np.column_stack(cols)


def test(y, X, mask, label):
    Q, _ = np.linalg.qr(X)
    y_res = y - Q @ (Q.T @ y)
    obs = fwl(y_res, mask.astype(float), Q)
    null = np.empty(N_ROT)
    for i, off in enumerate(RNG.integers(1, len(mask), size=N_ROT)):
        null[i] = fwl(y_res, np.roll(mask.astype(float), off), Q)
    null = null[~np.isnan(null)]
    p = float((np.abs(null - null.mean()) >= abs(obs - null.mean())).mean())
    print(f"  {label:<64}{obs*100:>+9.3f}%   p={p:.3f} {stars(p)}")
    return obs, p


def main():
    d = _vix_data.add_features(_vix_data.load())
    d["rv20"] = d.spx.pct_change().rolling(20).std() * np.sqrt(252)
    d["spx_r20"] = d.spx / d.spx.shift(20) - 1.0
    d["spx_r5"] = d.spx / d.spx.shift(5) - 1.0
    d["vix_sd10"] = d.vix.rolling(10).std(ddof=0)
    d["vrp"] = d.vix / 100.0 - d.rv20                     # implied minus realized

    keep = ["g5", "vix_pct1y", "bb10_width", "rv20", "spx_r20", "spx_r5", "vrp",
            "bb10_2.0_below", "bb10_2.0_above"]
    a = d[keep].dropna().copy()
    y = a.g5.values
    base = y.mean()
    below = a["bb10_2.0_below"].astype(bool).values
    above = a["bb10_2.0_above"].astype(bool).values
    print(f"frame {len(a):,} rows {a.index[0].date()} → {a.index[-1].date()}  base {base*100:+.3f}%  "
          f"below n={below.sum()}  above n={above.sum()}")

    Q = lambda s, k=10: pd.qcut(a[s], k, labels=False, duplicates="drop").values
    lvl, wid, rv, r20, r5, vrp = (Q("vix_pct1y", 20), Q("bb10_width"), Q("rv20"),
                                  Q("spx_r20"), Q("spx_r5"), Q("vrp"))

    for signm, sig in (("BELOW", below), ("ABOVE", above)):
        print("\n" + "=" * 96)
        print(f"{signm}-band dummy, D5 (g5), under competing regime controls")
        print("=" * 96)
        n = len(y)
        test(y, fe_matrix(n, []), sig, "no control")
        test(y, fe_matrix(n, [lvl]), sig, "vix_pct1y vigintile FE (level)")
        test(y, fe_matrix(n, [wid]), sig, "bb10_width decile FE (band width / VIX dispersion)")
        test(y, fe_matrix(n, [rv]), sig, "SPX realized-vol(20d) decile FE")
        test(y, fe_matrix(n, [r20]), sig, "SPX trailing 20d return decile FE (index momentum)")
        test(y, fe_matrix(n, [r5]), sig, "SPX trailing 5d return decile FE")
        test(y, fe_matrix(n, [vrp]), sig, "VIX-minus-realized-vol (VRP) decile FE")
        test(y, fe_matrix(n, [lvl, wid, rv]), sig, "level + width + realized-vol FE")
        test(y, fe_matrix(n, [lvl, wid, rv, r20, r5]), sig, "EVERYTHING (level+width+rv+2 momentum) FE")


if __name__ == "__main__":
    main()
