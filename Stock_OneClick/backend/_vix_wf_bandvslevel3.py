"""
_vix_wf_bandvslevel3.py — final robustness: is the level-neutral below-band effect just one or
two clustered episodes? Plus the continuous %B version of the same horse race.

  H. episode structure: runs of consecutive below-band days, per-episode mean g5
  I. leave-one-episode-out jackknife of the vigintile-level-FE below coefficient
  J. continuous bb10_pctb: coefficient with nonparametric level control + rotation p
  K. era table for the ABOVE band under level control (round 2 said above is weak)

Run: ../../vcp_env/bin/python _vix_wf_bandvslevel3.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _vix_data

RNG = np.random.default_rng(20260912)
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


def coef_and_p(y, X, mask_or_vec, rotate_mask=None):
    Q, _ = np.linalg.qr(X)
    y_res = y - Q @ (Q.T @ y)
    v = np.asarray(mask_or_vec, float)
    obs = fwl(y_res, v, Q)
    rot = v if rotate_mask is None else np.asarray(rotate_mask, float)
    null = np.empty(N_ROT)
    for i, off in enumerate(RNG.integers(1, len(v), size=N_ROT)):
        null[i] = fwl(y_res, np.roll(rot, off), Q)
    null = null[~np.isnan(null)]
    p = float((np.abs(null - null.mean()) >= abs(obs - null.mean())).mean())
    return obs, p


def main():
    d = _vix_data.add_features(_vix_data.load())
    keep = ["g5", "vix_pct1y", "stretch", "bb10_pctb", "bb10_2.0_below", "bb10_2.0_above"]
    a = d[keep].dropna().copy()
    a["below"] = a["bb10_2.0_below"].astype(bool)
    a["above"] = a["bb10_2.0_above"].astype(bool)
    y = a.g5.values
    base = y.mean()
    below = a.below.values
    above = a.above.values
    vig1 = pd.qcut(a.vix_pct1y, 20, labels=False, duplicates="drop").values
    dec1 = pd.qcut(a.vix_pct1y, 10, labels=False, duplicates="drop").values
    print(f"frame {len(a):,} rows {a.index[0].date()} → {a.index[-1].date()}  base g5 {base*100:+.3f}%"
          f"  below n={below.sum()}  above n={above.sum()}")

    # ---- H. episode structure
    print("\n" + "=" * 96)
    print("H. EPISODE STRUCTURE of below-band days (a run = consecutive below-band sessions)")
    print("=" * 96)
    idx = np.flatnonzero(below)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    groups = np.split(idx, breaks + 1)
    print(f"  {len(groups)} distinct episodes covering {below.sum()} days "
          f"(median length {int(np.median([len(g) for g in groups]))}, max {max(len(g) for g in groups)})")
    rows = [(a.index[g[0]].date(), len(g), y[g].mean() * 100) for g in groups]
    rows_s = sorted(rows, key=lambda r: r[2])
    print("  5 worst episodes (mean D5): " + ", ".join(f"{r[0]} n={r[1]} {r[2]:+.2f}%" for r in rows_s[:5]))
    print("  5 best  episodes (mean D5): " + ", ".join(f"{r[0]} n={r[1]} {r[2]:+.2f}%" for r in rows_s[-5:]))
    ep_means = np.array([r[2] for r in rows])
    print(f"  episode-level mean of means {ep_means.mean():+.3f}%  median {np.median(ep_means):+.3f}%  "
          f"share of episodes negative {np.mean(ep_means < 0)*100:.0f}%  "
          f"(all-day baseline {base*100:+.3f}%, baseline share of days positive {np.mean(y>0)*100:.0f}%)")
    yrs = np.array([r[0].year for r in rows])
    print(f"  episodes per decade: " + " ".join(
        f"{dd}s:{int(((yrs>=dd)&(yrs<dd+10)).sum())}" for dd in (1990, 2000, 2010, 2020)))

    # ---- I. leave-one-episode-out jackknife
    print("\n" + "=" * 96)
    print("I. LEAVE-ONE-EPISODE-OUT JACKKNIFE — below coefficient under vix_pct1y vigintile FE")
    print("=" * 96)
    X = fe_matrix(len(y), [vig1])
    Q, _ = np.linalg.qr(X)
    y_res = y - Q @ (Q.T @ y)
    full = fwl(y_res, below.astype(float), Q)
    jk = []
    for g in groups:
        m = below.copy()
        m[g] = False
        jk.append(fwl(y_res, m.astype(float), Q))
    jk = np.array(jk)
    worst = np.argmax(jk)      # dropping this episode weakens (raises toward 0) the most
    print(f"  full-sample coef {full*100:+.3f}%")
    print(f"  jackknife range  {jk.min()*100:+.3f}% .. {jk.max()*100:+.3f}%   "
          f"median {np.median(jk)*100:+.3f}%")
    print(f"  most influential episode: {rows[worst][0]} (n={rows[worst][1]} days) — dropping it moves "
          f"the coef to {jk[worst]*100:+.3f}%")
    print(f"  episodes whose removal flips the sign: {int((jk > 0).sum())} of {len(jk)}")
    # drop the 3 most influential at once
    order = np.argsort(-jk)[:3]
    m3 = below.copy()
    for i in order:
        m3[groups[i]] = False
    print(f"  dropping the 3 most influential episodes ({sum(len(groups[i]) for i in order)} days): "
          f"coef {fwl(y_res, m3.astype(float), Q)*100:+.3f}% on n={int(m3.sum())}")

    # ---- J. continuous %B
    print("\n" + "=" * 96)
    print("J. CONTINUOUS %B (bb10_pctb) vs the dummies, under nonparametric level control")
    print("=" * 96)
    pctb = a.bb10_pctb.values
    for lbl, fes, extra in [("no control", [], None),
                            ("vix_pct1y decile FE", [dec1], None),
                            ("vix_pct1y vigintile FE", [vig1], None),
                            ("vix_pct1y vigintile FE + linear stretch", [vig1], [a.stretch.values])]:
        Xc = fe_matrix(len(y), fes, extra)
        c, p = coef_and_p(y, Xc, pctb)
        print(f"  {'%B: ' + lbl:<58}{c*100:>+9.3f}% per 1.0 of %B   p={p:.3f} {stars(p)}")
    print(f"  (%B range: p1={np.percentile(pctb,1):.2f} p99={np.percentile(pctb,99):.2f}; "
          f"a move from %B=0 to %B=1 is lower-band → upper-band)")
    print(f"  corr(%B, vix_pct1y) = {np.corrcoef(pctb, a.vix_pct1y.values)[0,1]:+.3f}   "
          f"corr(%B, stretch) = {np.corrcoef(pctb, a.stretch.values)[0,1]:+.3f}")

    # ---- K. era table for ABOVE under level control
    print("\n" + "=" * 96)
    print("K. ERA TABLE under level control (vix_pct1y vigintile FE, within-era refit)")
    print("=" * 96)
    years = a.index.year
    print(f"  {'era':<10}{'sig':<10}{'n':>6}{'raw excess':>13}{'level-ctrl coef':>18}{'p':>8}")
    for nm, y0, y1 in [("1990s", 1990, 1999), ("2000s", 2000, 2009),
                       ("2010s", 2010, 2019), ("2020s", 2020, 2099)]:
        sel = (years >= y0) & (years <= y1)
        ys = y[sel]
        v = pd.qcut(pd.Series(a.vix_pct1y.values[sel]), 10, labels=False, duplicates="drop").values
        Xe = fe_matrix(len(ys), [v])
        for signm, sig in (("below", below), ("above", above)):
            s = sig[sel]
            if s.sum() < 25:
                print(f"  {nm:<10}{signm:<10}{int(s.sum()):>6}   n<25 INCONCLUSIVE")
                continue
            c, p = coef_and_p(ys, Xe, s)
            print(f"  {nm:<10}{signm:<10}{int(s.sum()):>6}{(ys[s].mean()-ys.mean())*100:>12.3f}%"
                  f"{c*100:>17.3f}%{p:>8.3f} {stars(p)}")


if __name__ == "__main__":
    main()
