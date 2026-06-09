#!/usr/bin/env python3
"""Deep-dive on the only candidate that showed anything: B2 = cross-sectional mean
short_ratio LEVEL. Questions:
  1. Is the timing 'BEATS' in TEST real, or an artifact of sitting out 67% of a bull?
  2. Is B2 just a contrarian/mean-reversion proxy (short_ratio spikes in selloffs)?
  3. What does an HONEST cost-adjusted, OOS comparison say vs buy&hold AND vs a
     50% constant-exposure benchmark (matching its ~33% time-in-market is unfair;
     compare risk-adjusted properly)?
  4. Bootstrap the IC t-stat with block resampling (block=20) to respect overlap.
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
WINDOW = 63
rng = np.random.default_rng(42)

panel = pd.read_csv(PANEL)
price, feat, sr = xs.build_matrices(panel, WINDOW)
spx = pd.read_csv(SPX, parse_dates=["date"]).set_index("date").sort_index()
spx_px = spx["price"]
ew_ret = price.pct_change().mean(axis=1)
ew_px = (1 + ew_ret.fillna(0)).cumprod()

B2 = sr.mean(axis=1)

# ---- Q2: is B2 contrarian? correlate B2 LEVEL with TRAILING 20d return ----
for tgt_name, tgt_px in [("EW", ew_px), ("SPX", spx_px)]:
    trail20 = tgt_px / tgt_px.shift(20) - 1.0
    df = pd.concat([B2, trail20], axis=1, join="inner").dropna()
    df.columns = ["b2", "tr"]
    print(f"B2 level vs TRAILING-20d {tgt_name} return: Spearman = {df['b2'].rank().corr(df['tr'].rank()):+.3f}  "
          f"(negative => short_ratio HIGH after selloffs => B2 is contrarian)")

# ---- Q4: block-bootstrap t for B2 raw fwd20 IC, EW & SPX, FULL + TEST ----
def block_boot_ic(sig, tgt_px, h=20, block=20, nboot=2000):
    fwd = tgt_px.shift(-h) / tgt_px - 1.0
    df = pd.concat([sig, fwd], axis=1, join="inner").dropna()
    df.columns = ["s", "r"]
    n = len(df)
    if n < 40:
        return None
    base = df["s"].rank().corr(df["r"].rank())
    s = df["s"].values; r = df["r"].values
    nb = int(np.ceil(n / block))
    boots = []
    for _ in range(nboot):
        starts = rng.integers(0, n - block + 1, size=nb)
        idx = np.concatenate([np.arange(st, st + block) for st in starts])[:n]
        bs, br = s[idx], r[idx]
        if np.std(bs) == 0 or np.std(br) == 0:
            continue
        boots.append(pd.Series(bs).rank().corr(pd.Series(br).rank()))
    boots = np.array(boots)
    # 2-sided bootstrap p for rho>0
    p_gt0 = (boots <= 0).mean()
    return {"rho": round(float(base), 3), "boot_mean": round(float(boots.mean()), 3),
            "boot_ci95": [round(float(np.percentile(boots, 2.5)), 3),
                          round(float(np.percentile(boots, 97.5)), 3)],
            "p(rho<=0)": round(float(p_gt0), 3), "n": n}

print("\nBlock-bootstrap (block=20d to respect overlap) fwd20 IC of B2 level:")
for span, mask in [("FULL", None), ("TRAIN", "tr"), ("TEST", "te")]:
    b = B2 if mask is None else (B2[B2.index <= SPLIT] if mask == "tr" else B2[B2.index > SPLIT])
    for tgt_name, tgt_px in [("EW", ew_px), ("SPX", spx_px)]:
        res = block_boot_ic(b, tgt_px)
        print(f"   {span:<6}{tgt_name:<5} {res}")

# ---- Q1/Q3: timing backtest with COSTS + fair benchmark ----
# Strategy: exposure 1 if expanding-pctile(B2) >= 0.5 (act next day). Cost = 5bps per side on turnover.
def bt(sig, tgt_px, thr=0.5, cost_bps=5.0):
    ret = tgt_px.pct_change()
    sp = da.expanding_pctile(sig, min_periods=120)
    pos = (sp >= thr).astype(float).shift(1)
    al = pd.concat([pos, ret], axis=1, join="inner").dropna()
    al.columns = ["pos", "ret"]
    turn = al["pos"].diff().abs().fillna(0)
    cost = turn * (cost_bps / 1e4)
    al["net"] = al["pos"] * al["ret"] - cost
    return al

print("\nTiming with 5bps/side costs vs B&H and vs constant-exposure-matched benchmark:")
for tgt_name, tgt_px in [("EW", ew_px), ("SPX", spx_px)]:
    al = bt(B2, tgt_px)
    for span, sub in [("FULL", al), ("TRAIN", al[al.index <= SPLIT]), ("TEST", al[al.index > SPLIT])]:
        net = da.perf(sub["net"], sub["pos"])
        gross = da.perf(sub["pos"] * sub["ret"])
        bh = da.perf(sub["ret"])
        expo = sub["pos"].mean()
        # fair benchmark: hold B&H scaled to same avg exposure (same beta) -> same Sharpe as B&H, lower vol
        scaled = da.perf(expo * sub["ret"])
        print(f"   {tgt_name} {span:<6} NET Sharpe {net.get('sharpe',float('nan')):+.2f} CAGR {net.get('cagr',0):+.2%} "
              f"turn/yr {net.get('turnover_ann',0):.0f} inMkt {net.get('time_in_mkt',0):.0%} | "
              f"B&H Sharpe {bh.get('sharpe',float('nan')):+.2f} | "
              f"const-{expo:.0%}-expo Sharpe {scaled.get('sharpe',float('nan')):+.2f} CAGR {scaled.get('cagr',0):+.2%}")

# ---- Q1 follow-up: how many independent 'regimes' does B2 timing actually bet on? ----
sp = da.expanding_pctile(B2, min_periods=120)
pos = (sp >= 0.5).astype(float).shift(1).dropna()
flips = (pos.diff().abs() > 0).sum()
print(f"\nB2 timing position flips over full sample: {int(flips)} (=> effective independent bets ~ {int(flips)})")
print(f"B2 expanding-pctile autocorr(1)={sp.dropna().autocorr(1):.3f}  autocorr(20)={sp.dropna().autocorr(20):.3f}")
