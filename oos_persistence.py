#!/usr/bin/env python3
"""OOS persistence test for per-stock dark-pool fwd20 IC.

Split panel by date: H1 = first half, H2 = second half (by trading days).
For each of 32 names compute fwd20 IC (time-series Spearman of its dark-pool
feature vs its own forward 20d return) in each half independently.

Decisive question: does the per-stock IC ranking persist OOS?
  - corr(IC_h1, IC_h2) across the 32 names (Pearson + Spearman)
  - do top-5 names by H1 IC stay positive in H2?
  - account for ~20d overlap: effective n per half-stock ~ n_obs/20
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "/Users/feijing/github.com/dix")
import dix_xsection as xs  # noqa: E402

PANEL = "/Users/feijing/github.com/dix/data/finra/universe_panel.csv"
H = 20
WINDOW = 63

panel = pd.read_csv(PANEL)
price, feat, sr = xs.build_matrices(panel, WINDOW)
fwd = xs.fwd_returns(price, H)

dates = price.index
n = len(dates)
split_idx = n // 2
split_date = dates[split_idx]
print(f"total trading days={n}  split at idx {split_idx} = {split_date.date()}")
print(f"H1: {dates[0].date()}..{dates[split_idx-1].date()}   "
      f"H2: {dates[split_idx].date()}..{dates[-1].date()}")

# To avoid leakage across the split via the 20d forward window, the forward
# return at H1's last dates peeks into H2. We compute IC within each half using
# only rows whose feature date is in that half AND whose forward window stays
# resolvable. The shift(-20) NaNs the last 20 rows globally; for H1 we just use
# H1-dated features (their fwd returns land slightly into H2, a minor overlap,
# but each IC is computed on its own half's feature dates -> still a clean
# train/test SPLIT of the predictor timeline, which is the persistence question).

def half_ic(feat, fwd, idx_slice):
    f_h = feat.loc[idx_slice]
    r_h = fwd.loc[idx_slice]
    rows = []
    for s in feat.columns:
        f, r = f_h[s], r_h[s]
        m = f.notna() & r.notna()
        if m.sum() >= 20 and f[m].std() > 0 and r[m].std() > 0:
            ic = float(f[m].rank().corr(r[m].rank()))
            rows.append({"symbol": s, "ic": round(ic, 4), "n": int(m.sum())})
    return pd.DataFrame(rows).set_index("symbol")

h1_slice = dates[:split_idx]
h2_slice = dates[split_idx:]

ic1 = half_ic(feat, fwd, h1_slice).rename(columns={"ic": "ic_h1", "n": "n_h1"})
ic2 = half_ic(feat, fwd, h2_slice).rename(columns={"ic": "ic_h2", "n": "n_h2"})
merged = ic1.join(ic2, how="inner").dropna()

print(f"\nnames with valid IC in BOTH halves: {len(merged)}")
print(f"median n_obs per name per half: H1={merged.n_h1.median():.0f}  H2={merged.n_h2.median():.0f}")
print(f"effective n per name per half (~n/20): H1~{merged.n_h1.median()/20:.0f}  H2~{merged.n_h2.median()/20:.0f}")

pear = stats.pearsonr(merged.ic_h1, merged.ic_h2)
spear = stats.spearmanr(merged.ic_h1, merged.ic_h2)
print(f"\ncorr(IC_h1, IC_h2) across {len(merged)} names:")
print(f"  Pearson  r = {pear.statistic:+.3f}  p = {pear.pvalue:.3f}")
print(f"  Spearman r = {spear.statistic:+.3f}  p = {spear.pvalue:.3f}")

# Top-5 by H1 IC: do they stay positive in H2?
top5 = merged.sort_values("ic_h1", ascending=False).head(5)
print("\nTop-5 names by H1 IC -> their H2 IC:")
for s, row in top5.iterrows():
    print(f"  {s:<6} IC_h1 {row.ic_h1:+.3f}  ->  IC_h2 {row.ic_h2:+.3f}  "
          f"{'STILL +' if row.ic_h2 > 0 else 'FLIPPED -'}")
n_stay_pos = int((top5.ic_h2 > 0).sum())
print(f"  -> {n_stay_pos}/5 stay positive in H2  (chance ~2.5/5)")
mean_h2_of_top5 = float(top5.ic_h2.mean())
print(f"  -> mean H2 IC of H1-top5 = {mean_h2_of_top5:+.3f}  (vs overall H2 mean {merged.ic_h2.mean():+.3f})")

# Bottom-5 symmetry check
bot5 = merged.sort_values("ic_h1", ascending=True).head(5)
n_bot_stay_neg = int((bot5.ic_h2 < 0).sum())
print(f"\nBottom-5 by H1 IC: {n_bot_stay_neg}/5 stay negative in H2; mean H2 IC = {bot5.ic_h2.mean():+.3f}")

# Long/short spread test using H1 ranking applied to H2 (the tradable version)
# Long the H1-top5, short the H1-bot5, hold equal-weight buy&hold over H2 window
print("\n--- Tradable check: pick names on H1 IC, hold over H2 ---")
h2_prices = price.loc[h2_slice]
h2_ret = h2_prices.iloc[-1] / h2_prices.iloc[0] - 1.0
ew_ret = float(h2_ret.mean())
long_ret = float(h2_ret.reindex(top5.index).mean())
short_ret = float(h2_ret.reindex(bot5.index).mean())
print(f"  H2 buy&hold: H1-top5 long {long_ret:+.2%}  H1-bot5 {short_ret:+.2%}  "
      f"equal-weight-universe {ew_ret:+.2%}")
print(f"  long-top5 minus EW benchmark: {long_ret - ew_ret:+.2%}  "
      f"({'beats' if long_ret > ew_ret else 'LOSES to'} benchmark)")

print("\nFULL TABLE (sorted by H1 IC):")
print(merged.sort_values("ic_h1", ascending=False).to_string())
