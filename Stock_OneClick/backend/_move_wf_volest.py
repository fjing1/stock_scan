"""
_move_wf_volest.py — which volatility estimator best FORECASTS forward realized vol?

Scores every candidate on predicting L.realized_vol_forward(close, h) for h in {1,5,10,21},
separately for the index series (SPY QQQ IWM ^GSPC) and the pooled single names.

Design decisions (stated so they are auditable):
  * Target is per-DAY log-return sigma over r[t+1..t+h], from L.realized_vol_forward. At h=1 that
    function returns NaN everywhere (rolling(1).std(ddof=1) is undefined), so h=1 uses the only
    available one-bar proxy: |log r[t+1]|. Under normality E|r| = sigma*sqrt(2/pi), so the h=1
    bias multiplier absorbs sqrt(pi/2)=1.2533 and E[log|r|] = log(sigma) - 0.6352; the h=1 R^2 has
    a hard noise ceiling (log|z| has variance pi^2/8 = 1.2337) and is NOT comparable to h>=5.
  * Scored on LOG vol. R^2 denominator uses the TRAIN-period mean of log target (the vol analogue
    of L.climatology): a truly out-of-sample constant baseline, no peeking at test-set mean.
  * Two numbers per candidate:
      raw   = use log(estimator) directly as the log-vol forecast (bias hurts you)
      adj   = a + b*log(estimator), (a,b) OLS-fit on strictly prior years (bias is rescaled away)
    The gap between them is exactly "how fixable is this estimator's bias".
  * Walk-forward: L.walk_forward_years(min_train_years=5) -> test years 2006..2026.
  * All candidates are scored on the SAME rows (rows where target and every candidate are finite),
    so ranks are not driven by differing sample coverage.
  * Uncertainty: overlapping h-day windows + cross-sectional correlation. CIs come from a
    DATE-BLOCK bootstrap (blocks of 21 consecutive trading dates, all symbols in a date move
    together) so both the overlap and the cross-sectional clustering are respected.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

import _move_data as D
import _move_lib as L

FLOOR = 1e-3          # per-day sigma clip band before logs: 0.1%/day (=1.6% ann.) to 50%/day.
CEIL = 0.5            # 0.5% of 1-bar Parkinson bars are EXACTLY zero (high==low, halted/illiquid)
                      # and 1.8% of |log r| are ~zero; without a physical clip those become
                      # log(1e-5) = -11.5 outliers that wreck any regression on them. Clipped
                      # fractions are reported per cell.
MIN_COV = 500
HORIZONS = (1, 5, 10, 21)
IDX = ["SPY", "QQQ", "IWM", "^GSPC"]
NOT_SINGLE = {"SPY", "QQQ", "IWM", "DIA", "^GSPC", "^VIX"}
BLOCK = 21


# --------------------------------------------------------------------------- panel
def load_panel():
    p = D.load()
    o, h, l, c = p["Open"], p["High"], p["Low"], p["Close"]
    cov = c.notna().sum()
    keep = [s for s in c.columns if cov[s] >= MIN_COV]
    o, h, l, c = o[keep].copy(), h[keep].copy(), l[keep].copy(), c[keep].copy()
    # OHLC sanity: high must dominate, low must be dominated, all strictly positive
    bad = (h < l) | (h <= 0) | (l <= 0) | (o <= 0) | (c <= 0)
    for df in (o, h, l, c):
        df[bad] = np.nan
    print(f"panel: {len(keep)} symbols x {len(c)} rows, {c.index[0].date()} -> {c.index[-1].date()}"
          f"  (dropped {c.shape[1] and len(cov) - len(keep)} symbols with <{MIN_COV} closes, "
          f"{int(bad.values.sum())} OHLC-inconsistent bars nulled)")
    return o, h, l, c


# --------------------------------------------------------------------------- candidates
def build_estimators(o, h, l, c):
    """name -> DataFrame of per-day sigma, all computable from data up to and including row t."""
    E = {}
    for n in (5, 10, 21, 63, 126):
        E[f"cc{n}"] = L.vol_cc(c, n)
    for lam in (0.90, 0.94, 0.97):
        E[f"ewma{int(lam*100)}"] = L.vol_ewma(c, lam)
    for n in (5, 10, 21):
        E[f"park{n}"] = L.vol_parkinson(h, l, n)
        E[f"gk{n}"] = L.vol_garman_klass(o, h, l, c, n)
        E[f"rs{n}"] = L.vol_rogers_satchell(o, h, l, c, n)
        E[f"yz{n}"] = L.vol_yang_zhang(o, h, l, c, n)
    # HAR inputs (Corsi): library builds them per Series, so loop symbols and reassemble.
    # Two flavours: OHLC-based (rv_d = 1-bar Parkinson) and CLOSE-ONLY (rv_d = |log r|).
    # The close-only flavour is the fallback for symbols with no/bad OHLC.
    for tag, args in (("", True), ("c", False)):
        rvd, rvw, rvm = {}, {}, {}
        for s in c.columns:
            f = L.har_features(c[s], h[s], l[s]) if args else L.har_features(c[s])
            rvd[s], rvw[s], rvm[s] = f.rv_d, f.rv_w, f.rv_m
        E[f"har{tag}_d"] = pd.DataFrame(rvd)[c.columns]
        E[f"har{tag}_w"] = pd.DataFrame(rvw)[c.columns]
        E[f"har{tag}_m"] = pd.DataFrame(rvm)[c.columns]
    # longer HAR memory terms (quarterly / annual) off the same one-bar bases
    base_ohlc = L.vol_parkinson(h, l, 1)
    base_cc = np.log(c).diff().abs()
    E["har_q"] = base_ohlc.rolling(63).mean()
    E["har_y"] = base_ohlc.rolling(252).mean()
    E["harc_q"] = base_cc.rolling(63).mean()
    # causal long-run mean: expanding GEOMETRIC mean of cc21 (>=252 obs), i.e. mean of log
    lr = np.log(E["cc21"].clip(lower=FLOOR)).expanding(252).mean()
    E["lr"] = np.exp(lr)
    return E


NON_CAND = {"har_d", "har_w", "har_m", "har_q", "har_y",
            "harc_d", "harc_w", "harc_m", "harc_q", "lr"}


# candidate specs: name -> (kind, feature list)
#   'single' : one estimator; scored raw AND affine-rescaled
#   'multi'  : fitted linear model on log features (raw not meaningful)
def candidate_specs(E):
    singles = [k for k in E if k not in NON_CAND]
    specs = {k: ("single", [k]) for k in singles}
    specs["har3"] = ("multi", ["har_d", "har_w", "har_m"])
    specs["har4"] = ("multi", ["har_d", "har_w", "har_m", "har_q"])
    specs["har5"] = ("multi", ["har_d", "har_w", "har_m", "har_q", "har_y"])
    specs["har3+yz21"] = ("multi", ["har_d", "har_w", "har_m", "yz21"])
    specs["har3+cc21"] = ("multi", ["har_d", "har_w", "har_m", "cc21"])
    specs["har4+yz21"] = ("multi", ["har_d", "har_w", "har_m", "har_q", "yz21"])
    specs["har4+yz21+ewma97"] = ("multi", ["har_d", "har_w", "har_m", "har_q", "yz21", "ewma97"])
    specs["HARc3"] = ("multi", ["harc_d", "harc_w", "harc_m"])              # no OHLC needed
    specs["HARc4"] = ("multi", ["harc_d", "harc_w", "harc_m", "harc_q"])    # no OHLC needed
    specs["blend_cc5_cc126"] = ("multi", ["cc5", "cc126"])
    specs["blend_ewma94_lr"] = ("multi", ["ewma94", "lr"])
    specs["blend_yz5_yz21_cc63"] = ("multi", ["yz5", "yz21", "cc63"])
    specs["shrink50_cc10_lr"] = ("fixed_blend", ["cc10", "lr"])   # no fitting at all
    return specs


# --------------------------------------------------------------------------- long-format matrix
def stack(E, feats, tgt, cols):
    """Date-major ravel of every feature + target for the given symbols; keeps only rows where
    the target and EVERY feature are finite, so all candidates share one sample."""
    T = len(tgt.index)
    S = len(cols)
    tv = tgt[cols].values.astype(np.float64)
    y = np.log(np.clip(tv, FLOOR, CEIL)).ravel()
    yraw = np.clip(tv, FLOOR, CEIL).ravel()
    nclip = int(np.nansum((tv < FLOOR) | (tv > CEIL)))
    X = np.empty((T * S, len(feats)), dtype=np.float64)
    for j, f in enumerate(feats):
        X[:, j] = np.log(np.clip(E[f][cols].values.astype(np.float64), FLOOR, CEIL)).ravel()
    di = np.repeat(np.arange(T), S)
    ok = np.isfinite(y) & np.isfinite(yraw) & np.isfinite(X).all(axis=1)
    return y[ok], yraw[ok], X[ok], di[ok], nclip


# --------------------------------------------------------------------------- evaluation
def ols(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def evaluate(E, specs, tgt, cols, years, label, h):
    feats = sorted({f for _, fl in specs.values() for f in fl})
    y, yraw, X, di, nclip = stack(E, feats, tgt, cols)
    fi = {f: j for j, f in enumerate(feats)}
    yr = years[di]
    nfloor = nclip

    # accumulators over walk-forward test years
    acc = {k: dict(ssr_raw=0.0, ssr_adj=0.0, se_raw=0.0, se_adj=0.0, n=0, ql=0.0,
                   a=[], b=[], mult=[], beta=[]) for k in specs}
    sst = 0.0
    n_tot = 0
    test_rows = []          # (row indices, per-candidate squared errors) for the bootstrap
    per_year = []
    ycat = []               # for pooled log-space correlation

    # the walk-forward year schedule comes from the library splitter on the calendar index;
    # masks are then re-expressed on the stacked (date x symbol) rows via the row's year
    for yy, _trm, _tem in L.walk_forward_years(tgt.index):
        tr = yr < yy
        te = yr == yy
        if te.sum() < 50 or tr.sum() < 500:
            continue
        mu = y[tr].mean()
        sst_y = float(((y[te] - mu) ** 2).sum())
        sst += sst_y
        n_tot += int(te.sum())
        row = {"year": yy, "n": int(te.sum())}
        errs_y = {}
        for name, (kind, fl) in specs.items():
            cidx = [fi[f] for f in fl]
            if kind == "single":
                j = cidx[0]
                pr_raw = X[te, j]
                beta = ols(X[tr][:, [j]], y[tr])
                pr_adj = beta[0] + beta[1] * X[te, j]
                acc[name]["a"].append(beta[0])
                acc[name]["b"].append(beta[1])
                acc[name]["beta"].append(beta)
                acc[name]["mult"].append(float(np.exp((y[tr] - X[tr, j]).mean())))
            elif kind == "fixed_blend":
                pr_raw = 0.5 * X[te, cidx[0]] + 0.5 * X[te, cidx[1]]
                pr_adj = pr_raw
            else:
                beta = ols(X[tr][:, cidx], y[tr])
                pr_raw = beta[0] + X[te][:, cidx] @ beta[1:]
                pr_adj = pr_raw
                acc[name]["a"].append(beta[0])
                acc[name]["b"].append(float(beta[1:].sum()))
                acc[name]["beta"].append(beta)
                acc[name]["mult"].append(np.nan)
            e_raw = (y[te] - pr_raw) ** 2
            e_adj = (y[te] - pr_adj) ** 2
            acc[name]["ssr_raw"] += float(e_raw.sum())
            acc[name]["ssr_adj"] += float(e_adj.sum())
            f_raw, f_adj = np.exp(pr_raw), np.exp(pr_adj)
            acc[name]["se_raw"] += float(((f_raw - yraw[te]) ** 2).sum())
            acc[name]["se_adj"] += float(((f_adj - yraw[te]) ** 2).sum())
            # QLIKE on variance (standard vol-forecast loss); forecast clipped to the same band
            v_f = np.clip(f_adj, FLOOR, CEIL) ** 2
            v_a = np.clip(yraw[te], FLOOR, CEIL) ** 2
            acc[name]["ql"] += float((v_a / v_f - np.log(v_a / v_f) - 1.0).sum())
            acc[name]["n"] += int(te.sum())
            errs_y[name] = e_adj
            row[name] = 1.0 - float(e_adj.sum()) / sst_y if sst_y else np.nan
        per_year.append(row)
        test_rows.append((di[te], errs_y, y[te] - mu))
        ycat.append((y[te], {n_: X[te, fi[fl[0]]] for n_, (k_, fl) in specs.items()
                             if k_ == "single"}))

    corr = {}
    if ycat:
        yall = np.concatenate([t[0] for t in ycat])
        for n_ in ycat[0][1]:
            xall = np.concatenate([t[1][n_] for t in ycat])
            corr[n_] = float(np.corrcoef(yall, xall)[0, 1])

    out = []
    for name, a in acc.items():
        out.append({
            "cand": name,
            "r2_raw": 1.0 - a["ssr_raw"] / sst,
            "r2_adj": 1.0 - a["ssr_adj"] / sst,
            "corr_log": corr.get(name, np.nan),
            "rmse_log": np.sqrt(a["ssr_adj"] / a["n"]),
            "mse_raw_est": a["se_raw"] / a["n"] * 1e6,     # x1e6 so it is readable
            "mse_raw_adj": a["se_adj"] / a["n"] * 1e6,
            "qlike": a["ql"] / a["n"],
            "slope": np.nanmean(a["b"]) if a["b"] else np.nan,
            "icept": np.nanmean(a["a"]) if a["a"] else np.nan,
            "mult": np.nanmean(a["mult"]) if a["mult"] else np.nan,
            "n": a["n"],
        })
    res = pd.DataFrame(out).set_index("cand").sort_values("r2_adj", ascending=False)
    res.attrs["label"] = f"{label} h={h}"
    res.attrs["n_floor"] = nfloor
    res.attrs["n_dates"] = len(set(np.concatenate([t[0] for t in test_rows]))) if test_rows else 0
    res.attrs["betas"] = {k: (np.array(acc[k]["beta"][-1]) if acc[k]["beta"] else None)
                          for k in specs}
    res.attrs["feats"] = {k: specs[k][1] for k in specs}
    return res, pd.DataFrame(per_year), test_rows


def block_bootstrap(test_rows, names, n_boot=1000, seed=7):
    """Date-block bootstrap of R^2 (adj) differences. Blocks of BLOCK consecutive trading dates
    keep overlapping h-day windows and same-date cross-sectional correlation inside one draw."""
    rng = np.random.default_rng(seed)
    di = np.concatenate([t[0] for t in test_rows])
    dev = np.concatenate([t[2] for t in test_rows])
    err = {k: np.concatenate([t[1][k] for t in test_rows]) for k in names}
    blk = di // BLOCK
    ub, inv = np.unique(blk, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    inv_s = inv[order]
    starts = np.searchsorted(inv_s, np.arange(len(ub)))
    ends = np.append(starts[1:], len(inv_s))
    idx_by_blk = [order[starts[i]:ends[i]] for i in range(len(ub))]
    nb = len(ub)
    draws = {k: np.empty(n_boot) for k in names}
    for b in range(n_boot):
        pick = rng.integers(0, nb, nb)
        sel = np.concatenate([idx_by_blk[i] for i in pick])
        sst = float((dev[sel] ** 2).sum())
        for k in names:
            draws[k][b] = 1.0 - float(err[k][sel].sum()) / sst
    return draws


# --------------------------------------------------------------------------- overnight-gap stat
def overnight_share(o, c, cols):
    oc = np.log(o[cols] / c[cols].shift(1))
    cc = np.log(c[cols]).diff()
    num = np.nanmean(oc.values ** 2)
    den = np.nanmean(cc.values ** 2)
    return num / den


def main():
    o, h, l, c = load_panel()
    idx_cols = [s for s in IDX if s in c.columns]
    sng_cols = [s for s in c.columns if s not in NOT_SINGLE]
    print(f"indices: {idx_cols}   single names: {len(sng_cols)}")
    print(f"overnight variance share (mean oc^2 / mean cc^2): "
          f"indices {overnight_share(o,c,idx_cols):.3f}, singles {overnight_share(o,c,sng_cols):.3f}")

    E = build_estimators(o, h, l, c)
    specs = candidate_specs(E)
    print(f"{len(specs)} candidates: {sorted(specs)}\n")
    years = c.index.year.values

    store = {}
    for hh in HORIZONS:
        if hh == 1:
            tgt = np.log(c).diff().abs().shift(-1)
            tname = "|log r[t+1]| PROXY (realized_vol_forward is NaN at h=1)"
        else:
            tgt = L.realized_vol_forward(c, hh)
            tname = f"realized_vol_forward(close,{hh})"
        for label, cols in (("INDEX", idx_cols), ("SINGLE", sng_cols), ("SPY", ["SPY"])):
            res, py, tr = evaluate(E, specs, tgt, cols, years, label, hh)
            store[(hh, label)] = (res, py, tr)
            print("=" * 118)
            print(f"h={hh}  {label}  target={tname}   n={res['n'].iloc[0]:,}  "
                  f"test years {py.year.min()}-{py.year.max()}  "
                  f"test dates={res.attrs['n_dates']}  clipped target obs={res.attrs['n_floor']}")
            print(res[["r2_raw", "r2_adj", "corr_log", "rmse_log", "mse_raw_est", "mse_raw_adj",
                       "qlike", "slope", "icept", "mult"]].to_string(
                float_format=lambda v: f"{v:9.4f}" if abs(v) < 1e3 else f"{v:9.3g}"))
            print("  (mse_raw_* are x1e6 on per-day sigma; qlike is on variance, lower better)")
            print()

    # ---- bootstrap the top few + the EWMA fallback, per horizon/universe
    print("#" * 118)
    print("DATE-BLOCK BOOTSTRAP (1000 draws, blocks of 21 trading dates) — R2_adj and gap vs ewma94")
    for (hh, label), (res, py, tr) in store.items():
        top = list(res.index[:4])
        names = list(dict.fromkeys(top + ["ewma94", "ewma97", "cc21", "yz21", "park21",
                                          "har3", "har4", "HARc3", "HARc4"]))
        dr = block_bootstrap(tr, names)
        base = dr["ewma94"]
        print(f"\n-- h={hh} {label}")
        for k in names:
            d = dr[k] - base
            print(f"   {k:22s} r2={res.loc[k,'r2_adj']:7.4f}  "
                  f"boot 95% [{np.percentile(dr[k],2.5):7.4f},{np.percentile(dr[k],97.5):7.4f}]  "
                  f"vs ewma94 {d.mean():+7.4f} [{np.percentile(d,2.5):+7.4f},{np.percentile(d,97.5):+7.4f}]"
                  f"  P(>ewma94)={np.mean(d>0):.3f}")

    # ---- per-year stability of the winner vs fallback
    print("\n" + "#" * 118)
    print("PER-YEAR R2_adj (out-of-sample)")
    for key in [(1, "SINGLE"), (5, "INDEX"), (5, "SINGLE"), (10, "SINGLE"),
                (21, "INDEX"), (21, "SINGLE")]:
        res, py, _ = store[key]
        keep = ["year", "n"] + list(dict.fromkeys(list(res.index[:3]) +
                                                 ["har4", "HARc4", "ewma94", "ewma97", "park21"]))
        print(f"\n-- h={key[0]} {key[1]}")
        print(py[keep].to_string(index=False, float_format=lambda v: f"{v:7.3f}"))

    # ---- explicit bias table for the range estimators
    print("\n" + "#" * 118)
    print("BIAS: geometric-mean ratio realized/estimator, walk-forward TRAIN means (the multiplier)")
    rows = []
    for (hh, label), (res, _, _) in store.items():
        for k in ["cc5", "cc21", "cc63", "ewma94", "ewma97",
                  "park21", "gk21", "rs21", "yz21"]:
            rows.append({"h": hh, "uni": label, "est": k,
                         "mult": res.loc[k, "mult"], "slope": res.loc[k, "slope"],
                         "icept": res.loc[k, "icept"], "r2_raw": res.loc[k, "r2_raw"],
                         "r2_adj": res.loc[k, "r2_adj"], "corr": res.loc[k, "corr_log"]})
    bt = pd.DataFrame(rows)
    for v in ("mult", "slope", "corr"):
        print(f"\n{v}:")
        print(bt.pivot_table(index=["uni", "est"], columns="h", values=v).to_string(
            float_format=lambda x: f"{x:6.3f}"))
    print("\nR2_raw (no rescale) vs R2_adj (rescaled): cost of leaving the bias in")
    print(bt.pivot_table(index=["uni", "est"], columns="h", values=["r2_raw", "r2_adj"]).to_string(
        float_format=lambda v: f"{v:7.4f}"))

    # ---- recommended spec coefficients: last walk-forward fit (2026 model), for implementation
    print("\n" + "#" * 118)
    print("FITTED COEFFICIENTS of the last walk-forward model (trained on <2026), per horizon")
    for hh in HORIZONS:
        for label in ("INDEX", "SINGLE"):
            res = store[(hh, label)][0]
            for k in ("har4", "har4+yz21", "HARc4", "ewma94", "ewma97", "park21", "yz21"):
                b = res.attrs["betas"][k]
                if b is None:
                    continue
                fl = res.attrs["feats"][k]
                terms = "  ".join(f"{bi:+.4f}*log({f})" for bi, f in zip(b[1:], fl))
                print(f"  h={hh:2d} {label:6s} {k:12s}: log(sigma_hat) = {b[0]:+.4f}  {terms}")
        print()

    # ---- shortfall of the practical fallback ladder
    print("#" * 118)
    print("FALLBACK LADDER: R2_adj loss vs the per-cell winner")
    lad = ["har4+yz21", "har4", "har3", "HARc4", "HARc3", "ewma97", "ewma94", "cc21", "park21"]
    tab = {}
    for (hh, label), (res, _, _) in store.items():
        best = res["r2_adj"].max()
        tab[(label, hh)] = {"WINNER=" + res["r2_adj"].idxmax(): best,
                            **{k: res.loc[k, "r2_adj"] - best for k in lad}}
    print(pd.DataFrame(tab).to_string(float_format=lambda v: f"{v:8.4f}"))

    # ---- today's annualized readings, to connect to the user's SPY observation
    print("\n" + "#" * 118)
    print("TODAY's SPY readings, annualized (x sqrt(252)) — reproduces the reported gap")
    last = {}
    for k in ["cc21", "cc63", "ewma94", "ewma97", "park21", "gk21", "rs21", "yz21",
              "har_m", "harc_m", "lr"]:
        last[k] = float(E[k]["SPY"].dropna().iloc[-1]) * np.sqrt(252) * 100
    print("  " + "  ".join(f"{k}={v:.2f}%" for k, v in last.items()))

    pd.to_pickle({k: (v[0], v[1]) for k, v in store.items()},
                 "/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/_move_wf_volest.pkl")


if __name__ == "__main__":
    main()
