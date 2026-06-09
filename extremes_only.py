#!/usr/bin/env python3
"""extremes_only hypothesis: per-stock tail effect of dark-pool short_ratio.

For each stock, compute a look-ahead-safe ROLLING percentile of its short_ratio
(rank of today's value within a trailing window). Then bucket days into
TOP decile (pct>=0.9), BOTTOM decile (pct<=0.1), and MIDDLE. Compare forward-20d
returns across buckets. Test:
  1. Is the top-vs-bottom spread consistently same-signed ACROSS names?
  2. Does it persist in a held-out date split (train<=2025-08, test after)?
  3. Account for ~20d overlap (effective n) in significance.
  4. Compare to the naive equal-weight buy&hold benchmark.
"""
from __future__ import annotations
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/feijing/github.com/dix")
import dix_xsection as xs

PANEL = "/Users/feijing/github.com/dix/data/finra/universe_panel.csv"
H = 20            # forward horizon (days)
ROLL = 252        # rolling window for percentile (look-ahead-safe)
MINP = 120        # min periods before we emit a percentile
SPLIT = "2025-08-31"

panel = pd.read_csv(PANEL)
price, _feat, sr = xs.build_matrices(panel, window=63)  # sr = short_ratio date x symbol
fwd = xs.fwd_returns(price, H)                            # forward 20d return, strictly future
ret1 = price.pct_change()                                 # daily returns for benchmark

dates = price.index
split_dt = pd.Timestamp(SPLIT)

def roll_pctile(s: pd.Series, window=ROLL, minp=MINP) -> pd.Series:
    """Rolling percentile rank of s[t] within trailing `window` obs (incl t).
    Look-ahead-safe: only uses past+current."""
    def f(w):
        return (w <= w[-1]).mean()
    return s.rolling(window, min_periods=minp).apply(f, raw=True)

# Build per-stock rolling percentile of short_ratio
pct = pd.DataFrame({s: roll_pctile(sr[s]) for s in sr.columns}, index=sr.index)

symbols = list(sr.columns)

def bucket_stats(mask_dates):
    """For a given date mask (boolean Series over `dates`), compute per-stock
    mean fwd20 in top/bottom/mid buckets and the top-minus-bottom spread."""
    rows = []
    for s in symbols:
        p = pct[s]
        r = fwd[s]
        m = mask_dates & p.notna() & r.notna()
        if m.sum() < 60:
            continue
        ps, rs = p[m], r[m]
        top = rs[ps >= 0.9]
        bot = rs[ps <= 0.1]
        mid = rs[(ps > 0.1) & (ps < 0.9)]
        if len(top) < 5 or len(bot) < 5:
            continue
        rows.append({
            "symbol": s,
            "n": int(m.sum()),
            "n_top": len(top), "n_bot": len(bot),
            "mean_top": float(top.mean()),
            "mean_bot": float(bot.mean()),
            "mean_mid": float(mid.mean()) if len(mid) else np.nan,
            "spread": float(top.mean() - bot.mean()),  # top minus bottom
        })
    return pd.DataFrame(rows)

# Effective-n adjustment for ~20d overlap: divide independent-sample count by H.
def overlap_t(spread_vals):
    """One-sample t on the cross-section of per-stock spreads. These are ~indep
    across names (different stocks), so no overlap deflation needed across names.
    But each stock's bucket mean is built from overlapping 20d windows; we report
    cross-stock t (n=#stocks) which is the across-name consistency test."""
    v = np.asarray(spread_vals, float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n < 3 or v.std(ddof=1) == 0:
        return n, np.nan, np.nan
    t = v.mean() / (v.std(ddof=1) / np.sqrt(n))
    return n, float(v.mean()), float(t)

print("="*78)
print(f"EXTREMES-ONLY: per-stock fwd{H} in TOP-decile vs BOTTOM-decile of "
      f"rolling-{ROLL}d short_ratio percentile")
print(f"symbols={len(symbols)}  days={len(dates)}  "
      f"range={dates.min().date()}..{dates.max().date()}  split={SPLIT}")
print("="*78)

# ---- FULL SAMPLE ----
all_mask = pd.Series(True, index=dates)
full = bucket_stats(all_mask)
n, mu, t = overlap_t(full["spread"])
print(f"\n[FULL SAMPLE] {len(full)} stocks with enough tail obs")
print(f"  mean top-minus-bottom spread (across names) = {mu:+.4f}  "
      f"t(n={n}) = {t:+.2f}")
print(f"  fraction of names with positive spread = {(full['spread']>0).mean():.2%}")
print(f"  median spread = {full['spread'].median():+.4f}")
print(f"  mean top fwd20 = {full['mean_top'].mean():+.4f}   "
      f"mean bot fwd20 = {full['mean_bot'].mean():+.4f}   "
      f"mean mid fwd20 = {full['mean_mid'].mean():+.4f}")

# ---- TRAIN / TEST SPLIT ----
train_mask = pd.Series(dates <= split_dt, index=dates)
test_mask  = pd.Series(dates >  split_dt, index=dates)
tr = bucket_stats(train_mask).set_index("symbol")
te = bucket_stats(test_mask).set_index("symbol")
common = tr.index.intersection(te.index)
ntr, mutr, ttr = overlap_t(tr["spread"])
nte, mute, tte = overlap_t(te["spread"])
print(f"\n[TRAIN <= {SPLIT}]  {len(tr)} stocks")
print(f"  mean spread = {mutr:+.4f}  t(n={ntr}) = {ttr:+.2f}  "
      f"frac+ = {(tr['spread']>0).mean():.2%}")
print(f"[TEST  >  {SPLIT}]  {len(te)} stocks  (held-out)")
print(f"  mean spread = {mute:+.4f}  t(n={nte}) = {tte:+.2f}  "
      f"frac+ = {(te['spread']>0).mean():.2%}")

# Sign agreement train vs test (per stock)
if len(common) >= 5:
    agree = (np.sign(tr.loc[common, "spread"]) == np.sign(te.loc[common, "spread"]))
    a = tr.loc[common, "spread"].rank()
    b = te.loc[common, "spread"].rank()
    rho = a.corr(b)  # Pearson on ranks == Spearman (no scipy needed)
    print(f"\n[OOS STABILITY] {len(common)} stocks in both periods")
    print(f"  per-stock spread sign agreement train->test = {agree.mean():.2%} "
          f"(50% = coin flip)")
    print(f"  Spearman(train spread, test spread) = {rho:+.3f}")

# ---- TRADING TEST: trade ONLY the tails, OOS ----
# Rule A: go long a name on days its short_ratio pct>=0.9 (hold 1 day, next-day ret).
# Rule B: go long on bottom decile. Compare both to equal-weight buy&hold of the
# 32-name universe over the SAME (test) period.
def tail_portfolio(mask_dates, lo, hi):
    """Equal-weight, daily-rebalanced long of names whose pct in [lo,hi] that day;
    realized as next-day return. Returns daily series."""
    days = dates[mask_dates.values]
    rets = []
    idx = []
    nxt = ret1.shift(-1)  # next-day return realized from a signal at t
    for dt in days:
        if dt not in pct.index:
            continue
        prow = pct.loc[dt]
        sel = prow[(prow >= lo) & (prow <= hi)].index
        if len(sel) == 0:
            continue
        rr = nxt.loc[dt, sel].mean()
        if pd.notna(rr):
            rets.append(float(rr)); idx.append(dt)
    return pd.Series(rets, index=idx)

import dix_analysis as da
for label, mask in [("FULL", all_mask), ("TRAIN", train_mask), ("TEST(OOS)", test_mask)]:
    top_p = tail_portfolio(mask, 0.9, 1.01)
    bot_p = tail_portfolio(mask, -0.01, 0.1)
    # equal-weight benchmark over same period (all names each day)
    bm_days = dates[mask.values]
    nxt = ret1.shift(-1)
    bm = nxt.loc[nxt.index.intersection(bm_days)].mean(axis=1).dropna()
    pt, pb, pbm = da.perf(top_p), da.perf(bot_p), da.perf(bm)
    print(f"\n[{label}] tail-only daily long portfolios (next-day, no costs):")
    print(f"  long TOP-decile-pct names : Sharpe {pt.get('sharpe','-'):>6}  "
          f"CAGR {pt.get('cagr','-')}  n={pt.get('n','-')}")
    print(f"  long BOT-decile-pct names : Sharpe {pb.get('sharpe','-'):>6}  "
          f"CAGR {pb.get('cagr','-')}  n={pb.get('n','-')}")
    print(f"  equal-weight universe (BM): Sharpe {pbm.get('sharpe','-'):>6}  "
          f"CAGR {pbm.get('cagr','-')}  n={pbm.get('n','-')}")
