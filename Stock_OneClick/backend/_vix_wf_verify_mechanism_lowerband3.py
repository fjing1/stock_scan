"""
_vix_wf_verify_mechanism_lowerband3.py — round 3: calibration + era.

Rounds 1-2 established:
  * the claim's numbers replicate exactly, and the below-band dummy SURVIVES every confound
    I could build, including saturated 2-way and 3-way cell FE (level x SPX-overbought x
    VIX-collapse, 896 cells -> -0.481%, p=.025).  So "redundant with VIX level" is wrong.
  * but the effect is entirely in the slab z10 in [-2.25,-2.00) (n=102, -0.769%, p=.003);
    the 57 DEEPEST below-band days give +0.031% (p=.922).
  * and it is a pre-2010 phenomenon: 1991-2009 -0.689% (p=.012, n=100),
    2010-2026 -0.182% (p=.535, n=59).

Round 3:
  Q. calibrate the rotation p against Newey-West and against a YEAR-BLOCK bootstrap
  R. era profile of the whole below-band family, incl. the better-powered bb10_1.5_below
  S. expanding/rolling coefficient — when did it stop working?
  T. clustering: how many distinct months/quarters/years do the 102 core days cover?
  U. did the effect ever exist out-of-sample?  strict split: fit nothing, just report
     1991-2003 (VXO-era backfilled VIX) / 2004-2012 / 2013-2026

Run: ../../vcp_env/bin/python _vix_wf_verify_mechanism_lowerband3.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _vix_data

RNG = np.random.default_rng(20260916)
PCT = 100.0


def stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "   "
    return "***" if p < 0.01 else ("** " if p < 0.05 else ("*  " if p < 0.10 else "   "))


def basis(X, tol=1e-9):
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, s > tol * s.max()]


def fe_matrix(n, fe_list=(), extra=None):
    cols = [np.ones(n)]
    for codes in fe_list:
        codes = np.asarray(codes)
        for k in np.unique(codes)[1:]:
            cols.append((codes == k).astype(float))
    if extra is not None:
        cols.extend([np.asarray(c, float) for c in extra])
    return np.column_stack(cols)


def rotation_all(y_res, dv, Q):
    n = len(dv)
    fd = np.conj(np.fft.rfft(dv))
    num = np.fft.irfft(fd * np.fft.rfft(y_res), n)
    sq = np.zeros(n)
    for j in range(Q.shape[1]):
        a = np.fft.irfft(fd * np.fft.rfft(Q[:, j]), n)
        sq += a * a
    den = float(dv @ dv) - sq
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-8, num / den, np.nan)


def rot_p(y, X, mask):
    Q = basis(X)
    yr = y - Q @ (Q.T @ y)
    c = rotation_all(yr, np.asarray(mask, float), Q)
    obs, nu = c[0], c[1:][~np.isnan(c[1:])]
    b = nu.mean()
    return obs, float((np.abs(nu - b) >= abs(obs - b)).mean())


def nw(y, X, j, lags):
    XtX = np.linalg.pinv(X.T @ X)
    b = XtX @ (X.T @ y)
    e = y - X @ b
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        A = Xe[L:].T @ Xe[:-L]
        S += w * (A + A.T)
    V = XtX @ S @ XtX
    return float(b[j]), float(b[j] / np.sqrt(max(V[j, j], 1e-30)))


def main():
    d = _vix_data.add_features(_vix_data.load())
    sx = d.spx.dropna()
    rr = sx.pct_change(fill_method=None)
    px = lambda s: s.reindex(d.index)
    d["rv20"] = px(rr.rolling(20).std() * np.sqrt(252))
    d["spx_r5"] = px(sx / sx.shift(5) - 1.0)
    d["spx_r20"] = px(sx / sx.shift(20) - 1.0)
    d["spx_z10"] = px((sx - sx.rolling(10).mean()) / sx.rolling(10).std(ddof=0))
    d["vix_chg5"] = d.vix / d.vix.shift(5) - 1.0
    d["z10"] = (d.vix - d.vix.rolling(10).mean()) / d.vix.rolling(10).std(ddof=0)
    keep = ["g5", "vix_pct1y", "bb10_width", "rv20", "spx_r5", "spx_r20", "spx_z10",
            "vix_chg5", "z10", "bb10_2.0_below", "bb10_1.5_below"]
    a = d[keep].dropna().copy()
    y, n = a.g5.values, len(a)
    below = a["bb10_2.0_below"].astype(bool).values
    below15 = a["bb10_1.5_below"].astype(bool).values
    z = a.z10.values
    core = below & (z >= -2.25)
    QC = lambda s, k: pd.qcut(a[s], k, labels=False, duplicates="drop").values
    lvl20 = QC("vix_pct1y", 20)
    Xlvl = fe_matrix(n, [lvl20])
    Xlad = fe_matrix(n, [lvl20, QC("bb10_width", 10), QC("rv20", 10), QC("spx_r20", 10),
                         QC("spx_r5", 10), QC("spx_z10", 10), QC("vix_chg5", 10)])
    print(f"FRAME {n:,} {a.index[0].date()} -> {a.index[-1].date()}  below n={below.sum()}")

    # ---------------------------------------------------------- Q. calibration
    print("\n" + "=" * 100)
    print("Q. CALIBRATION — rotation p vs Newey-West vs YEAR-BLOCK bootstrap (same coefficient)")
    print("=" * 100)
    for lab, X, m in (("level vigintile FE            ", Xlvl, below),
                      ("full ladder + mechanism FE    ", Xlad, below),
                      ("level FE, bb10_1.5_below      ", Xlvl, below15)):
        Q = basis(X)
        yr = y - Q @ (Q.T @ y)
        dv = np.asarray(m, float)
        dr = dv - Q @ (Q.T @ dv)
        Xs = np.column_stack([np.ones(n), dr])
        for lags in (5, 10, 21):
            b, t = nw(yr, Xs, 1, lags)
            if lags == 5:
                o, p = rot_p(y, X, m)
                line = f"  {lab} coef {o*PCT:+.3f}%  rot p={p:.3f}{stars(p)}  |  NW t: "
            line += f"L{lags}={t:+.2f} "
        # year-block bootstrap of the FWL coefficient
        yrs = a.index.year.values
        uy = np.unique(yrs)
        idx_by_year = {u: np.flatnonzero(yrs == u) for u in uy}
        boots = []
        for _ in range(3000):
            pick = RNG.choice(uy, size=len(uy), replace=True)
            ii = np.concatenate([idx_by_year[u] for u in pick])
            num = float(dr[ii] @ yr[ii])
            den = float(dr[ii] @ dr[ii])
            boots.append(num / den if den > 1e-12 else np.nan)
        boots = np.array(boots)
        boots = boots[~np.isnan(boots)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        pb = 2 * min((boots >= 0).mean(), (boots <= 0).mean())
        print(line + f" | yr-block boot 95% CI [{lo*PCT:+.3f}%, {hi*PCT:+.3f}%] p={pb:.3f}"
                     f"{stars(pb)}")

    # ---------------------------------------------------------- R. era profile
    print("\n" + "=" * 100)
    print("R. ERA PROFILE — level-FE coefficient inside each era, for the whole family")
    print("=" * 100)
    eras = [("1991-1999", 1991, 1999), ("2000-2009", 2000, 2009),
            ("2010-2019", 2010, 2019), ("2020-2026", 2020, 2026),
            ("1991-2009", 1991, 2009), ("2010-2026", 2010, 2026)]
    sigs = {"bb10_2.0_below": below, "bb10_1.5_below": below15,
            "core slab z in [-2.25,-2.0)": core}
    print(f"  {'era':<12}" + "".join(f"{k:>34}" for k in sigs))
    yrs = a.index.year.values
    for lab, y0, y1 in eras:
        sel = (yrs >= y0) & (yrs <= y1)
        cells = []
        for k, m in sigs.items():
            ms, ys = m[sel], y[sel]
            nb = int(ms.sum())
            if nb < 25:
                cells.append(f"n={nb} INCONCLUSIVE".rjust(34))
                continue
            lv = pd.qcut(a.vix_pct1y.values[sel], 20, labels=False, duplicates="drop")
            o, p = rot_p(ys, fe_matrix(int(sel.sum()), [lv]), ms)
            cells.append(f"n={nb:<4} {o*PCT:+7.3f}% p={p:5.3f}{stars(p)}".rjust(34))
        print(f"  {lab:<12}" + "".join(cells))

    # ---------------------------------------------------------- S. cumulative
    print("\n" + "=" * 100)
    print("S. WHEN DID IT STOP WORKING? cumulative sum of (below-day g5 minus level-cell mean)")
    print("=" * 100)
    Q = basis(Xlvl)
    yr = y - Q @ (Q.T @ y)
    contrib = np.where(below, yr, 0.0)
    cum = np.cumsum(contrib)
    marks = [1995, 2000, 2003, 2005, 2008, 2010, 2013, 2016, 2019, 2021, 2023, 2026]
    print("  cumulative level-adjusted P&L of shorting SPX for 5d on every below-band day (%):")
    for mk in marks:
        j = np.searchsorted(a.index.values, np.datetime64(f"{mk}-01-01"))
        j = min(j, n - 1)
        print(f"    through {mk}-01-01: {-cum[j]*PCT:+8.2f}%   "
              f"(below days so far {int(below[:j].sum())})")
    print(f"    end of sample     : {-cum[-1]*PCT:+8.2f}%   (below days {int(below.sum())})")

    # ---------------------------------------------------------- T. clustering
    print("\n" + "=" * 100)
    print("T. HOW MANY INDEPENDENT OBSERVATIONS IS 159 REALLY?")
    print("=" * 100)
    for lab, m in (("all below-band (159)", below), ("core slab [-2.25,-2.0) (102)", core)):
        ix = a.index[m]
        runs = 1 + int((np.diff(np.flatnonzero(m)) > 1).sum())
        # non-overlapping 5d windows
        pos = np.flatnonzero(m)
        keepn, last = 0, -99
        for p_ in pos:
            if p_ - last >= 6:
                keepn += 1
                last = p_
        print(f"  {lab:<32} runs={runs:<4} distinct months={ix.to_period('M').nunique():<4} "
              f"quarters={ix.to_period('Q').nunique():<4} years={ix.year.nunique():<4} "
              f"non-overlapping-5d obs={keepn}")
        # coefficient using only non-overlapping days
        mm = np.zeros(n, bool)
        last = -99
        for p_ in pos:
            if p_ - last >= 6:
                mm[p_] = True
                last = p_
        o, p = rot_p(y, Xlvl, mm)
        print(f"     -> level-FE coef on the {int(mm.sum())} NON-OVERLAPPING days: "
              f"{o*PCT:+.3f}%  p={p:.3f} {stars(p)}")

    # ---------------------------------------------------------- U. strict thirds
    print("\n" + "=" * 100)
    print("U. STRICT CHRONOLOGICAL THIRDS of the sample (no parameter chosen per third)")
    print("=" * 100)
    cuts = np.array_split(np.arange(n), 3)
    for i, ii in enumerate(cuts):
        sel = np.zeros(n, bool)
        sel[ii] = True
        for k, m in (("bb10_2.0_below", below), ("bb10_1.5_below", below15)):
            ms, ys = m[sel], y[sel]
            nb = int(ms.sum())
            lv = pd.qcut(a.vix_pct1y.values[sel], 20, labels=False, duplicates="drop")
            if nb < 25:
                print(f"  third {i+1} {a.index[ii[0]].date()}..{a.index[ii[-1]].date()}  "
                      f"{k:<16} n={nb} INCONCLUSIVE")
                continue
            o, p = rot_p(ys, fe_matrix(int(sel.sum()), [lv]), ms)
            print(f"  third {i+1} {a.index[ii[0]].date()}..{a.index[ii[-1]].date()}  "
                  f"{k:<16} n={nb:<4} coef {o*PCT:+7.3f}%  p={p:5.3f} {stars(p)}")


if __name__ == "__main__":
    main()
