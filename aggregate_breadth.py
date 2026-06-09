#!/usr/bin/env python3
"""AGGREGATE BREADTH GAUGE from FINRA off-exchange short-volume panel.

Build a daily MARKET-WIDE dark-pool breadth signal = cross-sectional aggregation of
the per-stock dark-pool feature across the 32 names, and test whether it predicts
forward returns of (a) the equal-weight universe and (b) SPX, at 5/10/20d.

Three breadth variants (all look-ahead-safe):
  B1 = cross-sectional MEAN of the per-stock rolling z-score feature (xs.build_matrices)
  B2 = cross-sectional MEAN of the raw short_ratio LEVEL
  B3 = % of names whose feature is in their OWN expanding-window top quintile (>=0.8 pctile)

For each breadth, smoothed/level forms tested:
  - raw level
  - expanding percentile (da.expanding_pctile, look-ahead-free)

Tests:
  - IC: Spearman( breadth[t] , fwd_h return of target ) over time, with overlap-adjusted t
  - Timing backtest: exposure 1 when breadth_pctile >= 0.5 else 0 (act next day), vs buy&hold
  - OOS split: train <= 2025-08-31, test after.
"""
from __future__ import annotations
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/feijing/github.com/dix")
import numpy as np
import pandas as pd
import dix_xsection as xs
import dix_analysis as da

PANEL = "/Users/feijing/github.com/dix/data/finra/universe_panel.csv"
SPX = "/Users/feijing/github.com/dix/data/DIX_max_till_2026Jun6.csv"
SPLIT = pd.Timestamp("2025-08-31")
HORIZONS = [5, 10, 20]
WINDOW = 63

# ---------------- load ----------------
panel = pd.read_csv(PANEL)
price, feat, sr = xs.build_matrices(panel, WINDOW)  # date x symbol
spx = pd.read_csv(SPX, parse_dates=["date"]).set_index("date").sort_index()
spx_px = spx["price"]

# Equal-weight universe daily return -> equity curve
ew_ret = price.pct_change().mean(axis=1)            # mean of per-name daily returns
ew_px = (1 + ew_ret.fillna(0)).cumprod()

# ---------------- breadth signals ----------------
# B1: cross-sectional mean of per-stock z-score feature
B1 = feat.mean(axis=1)
# B2: cross-sectional mean of raw short_ratio level
B2 = sr.mean(axis=1)
# B3: fraction of names with feature in OWN expanding top quintile
#   per-stock expanding percentile of the z-score feature, then fraction >= 0.8
feat_pct = pd.DataFrame({s: da.expanding_pctile(feat[s], min_periods=120) for s in feat.columns})
B3 = (feat_pct >= 0.8).sum(axis=1) / feat_pct.notna().sum(axis=1)

breadths = {"B1_xs_mean_z": B1, "B2_mean_short_ratio": B2, "B3_pct_top_quintile": B3}

def overlap_t(ic_series: pd.Series, h: int):
    """t-stat of mean daily IC with Newey-West-ish overlap correction.
    Effective n ~ n/h for h-day overlapping forward returns."""
    s = ic_series.dropna()
    n = len(s)
    if n < 5 or s.std() == 0:
        return np.nan, np.nan, 0
    naive_t = s.mean() / (s.std() / np.sqrt(n))
    eff_n = max(1, n / h)
    eff_t = s.mean() / (s.std() / np.sqrt(eff_n))
    return float(naive_t), float(eff_t), n

def ts_ic(sig: pd.Series, target_px: pd.Series, h: int):
    """Time-series IC: align signal[t] with forward h-day return of target.
    We compute a rolling correlation-style: simply Spearman over the full series
    of (signal, fwd_h). But to get an overlap-aware t, we form the daily product
    of standardized ranks is not per-day here (single target). Instead we report
    the Spearman corr and bootstrap-free overlap t via block effective-n on the
    contribution series sign(dev)."""
    fwd = target_px.shift(-h) / target_px - 1.0
    df = pd.concat([sig, fwd], axis=1, join="inner").dropna()
    df.columns = ["s", "r"]
    if len(df) < 30 or df["s"].std() == 0 or df["r"].std() == 0:
        return None
    rho = df["s"].rank().corr(df["r"].rank())
    # overlap-aware t for a single correlation: SE ~ sqrt((1-rho^2)/(eff_n-2))
    eff_n = max(3, len(df) / h)
    t = rho * np.sqrt((eff_n - 2) / max(1e-9, (1 - rho**2)))
    return {"rho": round(float(rho), 4), "t_overlap": round(float(t), 2),
            "n_obs": int(len(df)), "eff_n": int(eff_n)}

def timing_backtest(sig: pd.Series, target_px: pd.Series, thresh_pctile=0.5,
                    min_periods=120, train_only_mask=None):
    """Exposure 1 if expanding-percentile(sig) >= thresh else 0; act NEXT day.
    Returns dict with perf of strategy vs buy&hold over the given index."""
    ret = target_px.pct_change()
    sp = da.expanding_pctile(sig, min_periods=min_periods)
    pos = (sp >= thresh_pctile).astype(float)
    pos = pos.shift(1)  # act next day -> look-ahead-safe
    al = pd.concat([pos, ret], axis=1, join="inner").dropna()
    al.columns = ["pos", "ret"]
    if len(al) < 40:
        return None
    strat = al["pos"] * al["ret"]
    bh = al["ret"]
    return {"strategy": da.perf(strat, al["pos"]), "buyhold": da.perf(bh),
            "index": al.index}

def run_block(name_tag, sig_full, idx_filter=None):
    out = {}
    for tgt_name, tgt_px in [("EW_universe", ew_px), ("SPX", spx_px)]:
        sig = sig_full.copy()
        if idx_filter is not None:
            sig = sig[sig.index <= idx_filter] if idx_filter[0] == "le" else sig
        out[tgt_name] = {}
        for h in HORIZONS:
            res = ts_ic(sig, tgt_px, h)
            out[tgt_name][f"ic_fwd{h}"] = res
    return out

# ---------------- IC: full + OOS split ----------------
print("="*92)
print("AGGREGATE DARK-POOL BREADTH — time-series IC (Spearman breadth[t] vs fwd return), overlap-aware t")
print("="*92)
for bname, b in breadths.items():
    # two forms: raw level and expanding pctile
    forms = {"raw": b, "pctile": da.expanding_pctile(b, min_periods=120)}
    for fname, sig in forms.items():
        for tgt_name, tgt_px in [("EW_univ", ew_px), ("SPX", spx_px)]:
            print(f"\n[{bname} | {fname} | target={tgt_name}]")
            for span_lbl, mask in [("FULL", None), ("TRAIN<=25-08", "tr"), ("TEST>25-08", "te")]:
                s = sig
                if mask == "tr":
                    s = sig[sig.index <= SPLIT]
                elif mask == "te":
                    s = sig[sig.index > SPLIT]
                # for fwd returns we need target over same window
                fwd_idx = s.index
                row = []
                for h in HORIZONS:
                    fwd = tgt_px.shift(-h) / tgt_px - 1.0
                    df = pd.concat([s, fwd], axis=1, join="inner").dropna()
                    df.columns = ["s", "r"]
                    if len(df) < 25 or df["s"].std() == 0 or df["r"].std() == 0:
                        row.append(f"fwd{h}: n/a")
                        continue
                    rho = df["s"].rank().corr(df["r"].rank())
                    eff_n = max(3, len(df) / h)
                    t = rho * np.sqrt((eff_n - 2) / max(1e-9, (1 - rho**2)))
                    row.append(f"fwd{h}: rho={rho:+.3f} t={t:+.2f} (n={len(df)},eff={eff_n:.0f})")
                print(f"   {span_lbl:<14} " + " | ".join(row))

# ---------------- Timing backtests (FULL, TRAIN, TEST) ----------------
print("\n" + "="*92)
print("TIMING BACKTEST: exposure=1 when expanding-pctile(breadth) >= 0.5 (act next day) vs BUY&HOLD")
print("="*92)
def bt_split(sig, tgt_px, tgt_name, bname, fname):
    ret = tgt_px.pct_change()
    sp = da.expanding_pctile(sig, min_periods=120)
    pos = (sp >= 0.5).astype(float).shift(1)
    al = pd.concat([pos, ret], axis=1, join="inner").dropna()
    al.columns = ["pos", "ret"]
    for span_lbl, sub in [("FULL", al), ("TRAIN", al[al.index <= SPLIT]), ("TEST", al[al.index > SPLIT])]:
        if len(sub) < 40:
            print(f"   {bname}|{fname}|{tgt_name}|{span_lbl}: too few obs"); continue
        strat = da.perf(sub["pos"] * sub["ret"], sub["pos"])
        bh = da.perf(sub["ret"])
        print(f"   {bname:<20}{fname:<7}{tgt_name:<9}{span_lbl:<6} "
              f"STRAT Sharpe {strat.get('sharpe',float('nan')):+.2f} CAGR {strat.get('cagr',0):+.2%} "
              f"inMkt {strat.get('time_in_mkt',0):.0%} | "
              f"B&H Sharpe {bh.get('sharpe',float('nan')):+.2f} CAGR {bh.get('cagr',0):+.2%}"
              f"  {'BEATS' if strat.get('sharpe',-9) > bh.get('sharpe',9) else 'loses'}")

for bname, b in breadths.items():
    sig = b  # use raw level into expanding pctile inside bt_split
    for tgt_name, tgt_px in [("EW_univ", ew_px), ("SPX", spx_px)]:
        bt_split(sig, tgt_px, tgt_name, bname, "raw")
    print()

# ---------------- Also: does breadth beat just "always long"? Inverse check ----------------
print("="*92)
print("SANITY: correlation sign and direction. Reporting full-sample fwd20 rho for each breadth vs EW & SPX")
print("="*92)
for bname, b in breadths.items():
    for tgt_name, tgt_px in [("EW_univ", ew_px), ("SPX", spx_px)]:
        fwd = tgt_px.shift(-20) / tgt_px - 1.0
        df = pd.concat([b, fwd], axis=1, join="inner").dropna()
        df.columns = ["s", "r"]
        rho = df["s"].rank().corr(df["r"].rank())
        print(f"   {bname:<22}{tgt_name:<9} fwd20 rho={rho:+.3f}  (n={len(df)})")
