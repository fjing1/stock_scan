"""
_move_wf_volest_x.py — addendum to _move_wf_volest.py.

Answers three things the main ranking run left open:

  A. CONTEMPORANEOUS level bias of the range estimators. The main run's `mult` mixes two effects:
     the estimator's level bias AND vol mean-reversion over the forecast horizon. Here we compare
     each range estimator to close-to-close over the SAME window, which isolates the
     overnight-gap blindness the user observed on SPY (5.7% vs 8.4% annualized).

  B. COEFFICIENT TRANSFER. The recommended spec is a pooled walk-forward OLS. Does a model fit on
     single names work on SPY/QQQ/IWM, and vice versa? If yes we ship one global coefficient set.

  C. UNIFORM SPEC. Is there one specification within a hair of the per-cell winner in all 12
     (horizon x universe) cells, so the system does not need 12 different models?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _move_data as D
import _move_lib as L
from _move_wf_volest import (FLOOR, CEIL, MIN_COV, HORIZONS, IDX, NOT_SINGLE,
                             load_panel, build_estimators, ols, BLOCK)


def geo_ratio(a, b, cols):
    """Geometric-mean ratio a/b over all finite pairs, and its 10/50/90 percentiles."""
    r = np.log(a[cols].values.astype(float)) - np.log(b[cols].values.astype(float))
    r = r[np.isfinite(r)]
    return np.exp(r.mean()), np.exp(np.percentile(r, [10, 50, 90])), len(r)


def part_a(o, hi, lo, c, idx_cols, sng_cols):
    print("#" * 110)
    print("A. CONTEMPORANEOUS level bias: geometric-mean ratio of close-to-close vol to each")
    print("   range estimator over the SAME n-day window. >1 means the estimator READS LOW.")
    rows = []
    for n in (21, 63):
        cc = L.vol_cc(c, n)
        est = {"parkinson": L.vol_parkinson(hi, lo, n),
               "garman_klass": L.vol_garman_klass(o, hi, lo, c, n),
               "rogers_satchell": L.vol_rogers_satchell(o, hi, lo, c, n),
               "yang_zhang": L.vol_yang_zhang(o, hi, lo, c, n)}
        for uni, cols in (("INDEX", idx_cols), ("SPY", ["SPY"]), ("SINGLE", sng_cols)):
            for k, v in est.items():
                g, q, nn = geo_ratio(cc, v, cols)
                rows.append({"n": n, "uni": uni, "est": k, "cc/est_geomean": g,
                             "p10": q[0], "p50": q[1], "p90": q[2], "n_obs": nn})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    print("\n   overnight variance share  mean(log(O_t/C_{t-1})^2) / mean(log(C_t/C_{t-1})^2):")
    for uni, cols in (("INDEX", idx_cols), ("SPY", ["SPY"]), ("SINGLE", sng_cols)):
        oc = np.log(o[cols] / c[cols].shift(1)).values ** 2
        cc2 = np.log(c[cols]).diff().values ** 2
        s = np.nanmean(oc) / np.nanmean(cc2)
        print(f"     {uni:7s} share={s:.3f}   implied sigma_cc/sigma_intraday = "
              f"{np.sqrt(1/(1-s)):.3f}  n={int(np.isfinite(cc2).sum()):,}")

    print("\n   TODAY vs history, SPY 21-day: cc/est ratio now and its historical percentile")
    cc21 = L.vol_cc(c, 21)["SPY"]
    for k, v in (("parkinson", L.vol_parkinson(hi, lo, 21)["SPY"]),
                 ("garman_klass", L.vol_garman_klass(o, hi, lo, c, 21)["SPY"]),
                 ("yang_zhang", L.vol_yang_zhang(o, hi, lo, c, 21)["SPY"])):
        r = (cc21 / v).dropna()
        now = float(r.iloc[-1])
        print(f"     cc21/{k:14s} today={now:5.3f}  hist median={r.median():5.3f}  "
              f"pctile of today={100*(r < now).mean():5.1f}%  n={len(r)}")


# ------------------------------------------------------------------ B & C
SPECS = {
    "har4+yz21+ewma97": ["har_d", "har_w", "har_m", "har_q", "yz21", "ewma97"],
    "har4+yz21":        ["har_d", "har_w", "har_m", "har_q", "yz21"],
    "har3+yz21":        ["har_d", "har_w", "har_m", "yz21"],
    "har3":             ["har_d", "har_w", "har_m"],
    "HARc4":            ["harc_d", "harc_w", "harc_m", "harc_q"],
    "HARc3":            ["harc_d", "harc_w", "harc_m"],
    "ewma97":           ["ewma97"],
    "ewma94":           ["ewma94"],
    "park21":           ["park21"],
    "cc21":             ["cc21"],
}
ALLF = sorted({f for v in SPECS.values() for f in v})


def build_arrays(E, tgt, cols, years):
    T, S = len(tgt.index), len(cols)
    tv = np.clip(tgt[cols].values.astype(float), FLOOR, CEIL)
    y = np.log(tv).ravel()
    X = np.empty((T * S, len(ALLF)))
    for j, f in enumerate(ALLF):
        X[:, j] = np.log(np.clip(E[f][cols].values.astype(float), FLOOR, CEIL)).ravel()
    di = np.repeat(np.arange(T), S)
    yr = np.repeat(years, S)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    return y[ok], X[ok], di[ok], yr[ok]


def score(fitdat, scoredat, spec, min_train_years=5):
    """Fit on `fitdat` train years, score on `scoredat` test years. Returns pooled OOS R2 on
    log vol, with the constant baseline = train-period mean of the SCORED universe's log vol."""
    yf, Xf, _, yrf = fitdat
    ys, Xs, dis, yrs = scoredat
    cidx = [ALLF.index(f) for f in SPECS[spec]]
    ssr = sst = 0.0
    n = 0
    err_all, dev_all, di_all = [], [], []
    ylist = sorted(set(yrs))
    for yy in ylist[min_train_years:]:
        trf, tes = yrf < yy, yrs == yy
        trs = yrs < yy
        if tes.sum() < 50 or trf.sum() < 500 or trs.sum() < 500:
            continue
        beta = ols(Xf[trf][:, cidx], yf[trf])
        pred = beta[0] + Xs[tes][:, cidx] @ beta[1:]
        mu = ys[trs].mean()
        e = (ys[tes] - pred) ** 2
        d = ys[tes] - mu
        ssr += float(e.sum()); sst += float((d ** 2).sum()); n += int(tes.sum())
        err_all.append(e); dev_all.append(d); di_all.append(dis[tes])
    return (1 - ssr / sst, n, np.concatenate(err_all), np.concatenate(dev_all),
            np.concatenate(di_all))


def boot_r2(err, dev, di, n_boot=1000, seed=11):
    rng = np.random.default_rng(seed)
    blk = di // BLOCK
    ub, inv = np.unique(blk, return_inverse=True)
    order = np.argsort(inv, kind="stable"); inv_s = inv[order]
    st = np.searchsorted(inv_s, np.arange(len(ub))); en = np.append(st[1:], len(inv_s))
    grp = [order[st[i]:en[i]] for i in range(len(ub))]
    out = np.empty(n_boot)
    for b in range(n_boot):
        sel = np.concatenate([grp[i] for i in rng.integers(0, len(ub), len(ub))])
        out[b] = 1 - err[sel].sum() / (dev[sel] ** 2).sum()
    return out


def main():
    o, hi, lo, c = load_panel()
    idx_cols = [s for s in IDX if s in c.columns]
    sng_cols = [s for s in c.columns if s not in NOT_SINGLE]
    all_cols = idx_cols + sng_cols
    part_a(o, hi, lo, c, idx_cols, sng_cols)

    E = build_estimators(o, hi, lo, c)
    years = c.index.year.values

    print("\n" + "#" * 110)
    print("B. COEFFICIENT TRANSFER — OOS R2 on log vol, fit universe -> score universe")
    print("   (a model fit on 231 single names, applied to SPY/QQQ/IWM/^GSPC, and the reverse)")
    for hh in HORIZONS:
        tgt = (np.log(c).diff().abs().shift(-1) if hh == 1
               else L.realized_vol_forward(c, hh))
        dat = {u: build_arrays(E, tgt, cols, years)
               for u, cols in (("INDEX", idx_cols), ("SINGLE", sng_cols),
                               ("ALL", all_cols), ("SPY", ["SPY"]))}
        print(f"\n  h={hh}")
        hdr = f"    {'spec':20s}" + "".join(f"{a}->{b:<12s}" for a, b in
                                            [("IDX", "IDX"), ("SNG", "IDX"), ("ALL", "IDX"),
                                             ("SNG", "SNG"), ("IDX", "SNG"), ("ALL", "SNG"),
                                             ("SNG", "SPY"), ("IDX", "SPY")])
        print(hdr)
        for sp in SPECS:
            vals = []
            for a, b in [("INDEX", "INDEX"), ("SINGLE", "INDEX"), ("ALL", "INDEX"),
                         ("SINGLE", "SINGLE"), ("INDEX", "SINGLE"), ("ALL", "SINGLE"),
                         ("SINGLE", "SPY"), ("INDEX", "SPY")]:
                r2, n, *_ = score(dat[a], dat[b], sp)
                vals.append(r2)
            print(f"    {sp:20s}" + "".join(f"{v:16.4f}" for v in vals))
        print(f"    n: IDX={dat['INDEX'][0].shape[0]:,} SNG={dat['SINGLE'][0].shape[0]:,} "
              f"SPY={dat['SPY'][0].shape[0]:,}")

    print("\n" + "#" * 110)
    print("C. UNIFORM SPEC — shortfall (R2_adj minus the best spec in that cell), own-universe fits")
    tab = {}
    for hh in HORIZONS:
        tgt = (np.log(c).diff().abs().shift(-1) if hh == 1
               else L.realized_vol_forward(c, hh))
        for u, cols in (("INDEX", idx_cols), ("SINGLE", sng_cols), ("SPY", ["SPY"])):
            d = build_arrays(E, tgt, cols, years)
            r2s = {sp: score(d, d, sp)[0] for sp in SPECS}
            best = max(r2s.values())
            tab[(u, hh)] = {"BEST": best, **{sp: r2s[sp] - best for sp in SPECS}}
    T = pd.DataFrame(tab)
    print(T.to_string(float_format=lambda v: f"{v:8.4f}"))
    print("\n  worst-case shortfall across all 12 cells:")
    for sp in SPECS:
        print(f"    {sp:20s} {T.loc[sp].min():+.4f}   (mean {T.loc[sp].mean():+.4f})")

    print("\n" + "#" * 110)
    print("D. MINIMUM HISTORY (closed bars) each spec needs before it produces a value")
    need = {"har4+yz21+ewma97": 63, "har4+yz21": 63, "har3+yz21": 22, "har3": 22,
            "HARc4": 63, "HARc3": 22, "ewma97": 2, "ewma94": 2, "park21": 21, "cc21": 22}
    for sp in SPECS:
        ohlc = "no " if sp.startswith("HARc") or sp.startswith("ewma") or sp.startswith("cc") \
               else "YES"
        print(f"    {sp:20s} needs {need[sp]:3d} bars   OHLC required: {ohlc}")


if __name__ == "__main__":
    main()
