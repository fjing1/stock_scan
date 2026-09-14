"""Probe: does the FREE FINRA off-exchange short-volume feed carry information the
daily-OHLCV move_prob model does not already have?

This is a FEASIBILITY diagnostic, not a model. It answers: is it worth wiring in?

Features (all known by 17:18 ET on day t, so lagged 1 day -> no lookahead):
  sr     = ShortVolume / TotalVolume              (off-exchange short-volume ratio)
  sr_z   = 20d z-score of sr
  d_sr   = 1-day change in sr
  offx   = FINRA TotalVolume / consolidated Volume (off-exchange share of the tape)
  offx_z = 20d z-score of offx
  exr    = ShortExemptVolume / TotalVolume
  ovol   = log(FINRA TotalVolume) minus its 20d mean (off-exchange volume surprise)

Targets:
  vol: log realized vol over next 1 / 5 / 10 days
  dir: sign of next-day return

Baseline: HAR-style log-vol lags (1 / 5 / 21) + log Parkinson range -- i.e. exactly the
kind of daily-OHLC information move_prob already exploits.

Evaluation: expanding walk-forward OOS R-squared, refit yearly, pooled across symbols
with per-symbol demeaning. numpy lstsq only (no sklearn/statsmodels in this venv).
"""
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)


def sec(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


fin = pd.read_pickle("_data_probe_finra_hist.pkl")
panel = pickle.load(open("_move_panel.pkl", "rb"))
C, H, L, V = panel["Close"], panel["High"], panel["Low"], panel["Volume"]

# ---- guard against partial/in-progress bars (recurring bug in this repo)
last_common = min(C.index.max(), fin.date.max())
C, H, L, V = [x[x.index <= last_common] for x in (C, H, L, V)]
fin = fin[fin.date <= last_common]
print(f"aligned through {last_common.date()} (in-progress bar excluded)")

sec("1. OVERLAP WITH THE EXISTING PANEL")
fin_dates = pd.DatetimeIndex(sorted(fin.date.unique()))
ov = C.index.intersection(fin_dates)
print(f"panel days   : {len(C.index):,}  ({C.index.min().date()} .. {C.index.max().date()})")
print(f"FINRA days   : {len(fin_dates):,}  ({fin_dates.min().date()} .. {fin_dates.max().date()})")
print(f"OVERLAP days : {len(ov):,}  ({ov.min().date()} .. {ov.max().date()})")
print(f"-> flow features are usable on {len(ov)/len(C.index)*100:.1f}% of the panel's history")
print(f"   panel history NOT covered by FINRA: "
      f"{len(C.index) - len(ov):,} days before {fin_dates.min().date()}")

# ---------------------------------------------------------------- build the frame
sr_p = fin.pivot(index="date", columns="symbol", values="short")
to_p = fin.pivot(index="date", columns="symbol", values="total")
ex_p = fin.pivot(index="date", columns="symbol", values="exempt")

syms = [s for s in C.columns if s in to_p.columns]
idx = ov
C2, H2, L2, V2 = [x.reindex(index=idx, columns=syms) for x in (C, H, L, V)]
sr_p, to_p, ex_p = [x.reindex(index=idx, columns=syms) for x in (sr_p, to_p, ex_p)]

ret = np.log(C2).diff()
absr = ret.abs()
park = (np.log(H2 / L2) ** 2 / (4 * np.log(2))) ** 0.5

sr = sr_p / to_p
offx = to_p / V2
exr = ex_p / to_p
lto = np.log(to_p.replace(0, np.nan))


def z20(x):
    return (x - x.rolling(20, min_periods=15).mean()) / x.rolling(20, min_periods=15).std()


feat = {
    "sr": sr,
    "sr_z": z20(sr),
    "d_sr": sr.diff(),
    "offx": offx,
    "offx_z": z20(offx),
    "exr": exr,
    "ovol": lto - lto.rolling(20, min_periods=15).mean(),
}
base = {
    "lrv1": np.log(absr.clip(lower=1e-5)),
    "lrv5": np.log(absr.rolling(5, min_periods=4).mean().clip(lower=1e-5)),
    "lrv21": np.log(absr.rolling(21, min_periods=15).mean().clip(lower=1e-5)),
    "lpark": np.log(park.clip(lower=1e-5)),
}

sec("2. FEATURE SANITY (cross-sectional levels, most recent 5 days)")
for k in ("sr", "offx", "exr"):
    v = feat[k].tail(5)
    print(f"{k:6s}: mean={v.stack().mean():.4f} std={v.stack().std():.4f} "
          f"p05={v.stack().quantile(.05):.4f} p95={v.stack().quantile(.95):.4f} "
          f"n={v.stack().notna().sum():,}")
print("\noff-exchange share of consolidated volume (offx) by year -- is it drifting?")
print(offx.stack().groupby(offx.stack().index.get_level_values(0).year)
      .agg(["mean", "std", "count"]).to_string(float_format=lambda x: f"{x:,.4f}"))

sec("3. RAW PREDICTIVE CORRELATIONS (feature at t -> outcome at t+1..t+h)")
tgts = {}
for h in (1, 5, 10):
    rv = absr.rolling(h, min_periods=max(1, h - 1)).mean().shift(-h)
    tgts[f"lrv_fwd{h}"] = np.log(rv.clip(lower=1e-5))
tgts["ret_fwd1"] = ret.shift(-1)

rows = []
for fk, fv in feat.items():
    r = {"feature": fk}
    for tk, tv in tgts.items():
        a, b = fv.align(tv, join="inner")
        m = a.notna() & b.notna()
        x, y = a.values[m.values], b.values[m.values]
        if len(x) < 5000:
            r[tk] = np.nan
            continue
        # Spearman via rank-Pearson (no scipy in this venv)
        rx = pd.Series(x).rank().values
        ry = pd.Series(y).rank().values
        r[tk] = np.corrcoef(rx, ry)[0, 1]
    r["n"] = int(m.values.sum())
    rows.append(r)
print("Spearman rank correlation, pooled over all symbol-days:")
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

sec("4. INCREMENTAL OOS R-SQUARED OVER A DAILY-OHLC (HAR-LIKE) BASELINE")
print("expanding walk-forward, refit each Jan 1, pooled, per-symbol demeaned targets\n")


def build(cols, target):
    """Return long df: date, sym, y, x1..xn -- all rows complete."""
    parts = {"y": target}
    parts.update(cols)
    long = {}
    for k, v in parts.items():
        long[k] = v.stack(dropna=False)
    d = pd.DataFrame(long).dropna()
    d = d[np.isfinite(d.values).all(axis=1)]
    return d


def wf_r2(d, xcols):
    """Expanding walk-forward OOS R^2. Refit each calendar year."""
    dates = d.index.get_level_values(0)
    years = sorted(set(dates.year))
    preds, actual = [], []
    for yr in years:
        tr = d[dates < pd.Timestamp(f"{yr}-01-01")]
        te = d[(dates >= pd.Timestamp(f"{yr}-01-01")) & (dates < pd.Timestamp(f"{yr+1}-01-01"))]
        if len(tr) < 20000 or len(te) == 0:
            continue
        Xtr = np.column_stack([np.ones(len(tr))] + [tr[c].values for c in xcols])
        Xte = np.column_stack([np.ones(len(te))] + [te[c].values for c in xcols])
        beta, *_ = np.linalg.lstsq(Xtr, tr.y.values, rcond=None)
        preds.append(Xte @ beta)
        actual.append(te.y.values)
    if not preds:
        return np.nan, 0
    p = np.concatenate(preds)
    a = np.concatenate(actual)
    ss_res = ((a - p) ** 2).sum()
    ss_tot = ((a - a.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot, len(a)


bcols = list(base.keys())
fcols = list(feat.keys())
res = []
for h in (1, 5, 10):
    tgt = tgts[f"lrv_fwd{h}"]
    d = build({**base, **feat}, tgt)
    r2b, n = wf_r2(d, bcols)
    r2f, _ = wf_r2(d, bcols + fcols)
    # which single flow feature adds most
    best, bestv = None, -9
    for fk in fcols:
        r2s, _ = wf_r2(d, bcols + [fk])
        if r2s > bestv:
            best, bestv = fk, r2s
    res.append({"horizon": h, "n_oos": n, "R2_baseline": r2b,
                "R2_+all_flow": r2f, "delta": r2f - r2b,
                "best_single": best, "R2_+best_single": bestv,
                "delta_single": bestv - r2b})
print("TARGET = log realized vol over the next h days")
print(pd.DataFrame(res).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

sec("5. DIRECTIONAL TEST: does short-volume ratio predict next-day RETURN SIGN?")
d = build({"sr_z": feat["sr_z"], "d_sr": feat["d_sr"], "offx_z": feat["offx_z"]},
          tgts["ret_fwd1"])
print(f"n = {len(d):,} symbol-days")
for c in ("sr_z", "d_sr", "offx_z"):
    q = pd.qcut(d[c], 5, labels=False, duplicates="drop")
    g = d.groupby(q).y.agg(["mean", "std", "count"])
    g["mean_bps"] = g["mean"] * 1e4
    g["t_stat"] = g["mean"] / (g["std"] / np.sqrt(g["count"]))
    print(f"\n-- next-day return by quintile of {c} (Q0=low, Q4=high)")
    print(g[["mean_bps", "t_stat", "count"]].to_string(float_format=lambda x: f"{x:,.3f}"))
    spread = g["mean"].iloc[-1] - g["mean"].iloc[0]
    print(f"   Q4-Q0 spread = {spread*1e4:+.2f} bps/day")

sec("6. VERDICT ARITHMETIC")
d1 = res[0]
print(f"h=1  baseline OOS R2 = {d1['R2_baseline']:.4f}   "
      f"+ all 7 flow features = {d1['R2_+all_flow']:.4f}   delta = {d1['delta']:+.4f}")
print("Compare to the measured ceilings already established on this project:")
print("  leverage/semivariance + volume : +0.004 BSS2 (indices), +0.001 (single names)")
print("  macro calendar                 : +0.0011 BSS2")
print("R2 and BSS2 are different metrics, but the ORDER OF MAGNITUDE of the delta is")
print("what decides whether this is worth wiring in.")
