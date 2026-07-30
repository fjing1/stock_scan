#!/usr/bin/env python3
"""Does adding a long-MA trend filter improve the break SMA 8/22 entry?

Prior research (RESEARCH.md #22, ma_filter_sweep.py): the per-trade edge of a trend
filter peaks at the ~85-day MA (Close>SMA85 = hi-conv tier); the robust regime gate
is SMA50>SMA200 (golden cross). Here we stack each filter onto break SMA 8/22 and
measure forward-edge / win-rate (same methodology as break3avg_tune.py), to see if
gating breakouts to a long-term uptrend lifts win rate & mean return.

Variants (all = break SMA 8/22 AND <filter>):
  base            : no extra filter
  +C>SMA85        : close above 85-day SMA           (hi-conv tier from #22)
  +C>SMA200       : close above 200-day SMA
  +SMA50>SMA200   : golden-cross regime              (deployed dip_scan gate)
  +C>SMA85 & GC   : both stacked

    ../../vcp_env/bin/python break3avg_trendfilter.py

Caveat: close-to-close forward returns, no costs; per-event edge over baseline.
Filters cut fire count — watch that a higher mean isn't just a tiny-n artifact.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
import scan_stocks as scan  # noqa: E402
from rsi_ma_sweep import BASKET, INDEX, ma  # noqa: E402
from trend_alert_backtest import crossover  # noqa: E402

HORIZONS = [1, 5, 10, 20]


def prep(df: pd.DataFrame) -> dict:
    o, h, l, c = (df[k].astype(float) for k in ("Open", "High", "Low", "Close"))
    ohlc4 = (o + h + l + c) / 4
    sma8 = ma(ohlc4, 8, "SMA")
    sma22 = ma(ohlc4, 22, "SMA")
    sma3 = ma(ohlc4, 3, "SMA")
    base = (crossover(c, sma8) & (sma8 > sma8.shift(1))
            & crossover(c, sma22) & (sma3 > sma22)).fillna(False)
    return {
        "close": c, "base": base,
        "sma85": c.rolling(85).mean(),
        "sma200": c.rolling(200).mean(),
        "sma50": c.rolling(50).mean(),
        "sma200_": c.rolling(200).mean(),
        "fwd": {hh: c.shift(-hh) / c - 1 for hh in HORIZONS},
    }


def variants(p: dict) -> dict:
    b, c = p["base"], p["close"]
    gc = p["sma50"] > p["sma200"]
    return {
        "base (8/22)":       b,
        "+C>SMA85":          b & (c > p["sma85"]),
        "+C>SMA200":         b & (c > p["sma200"]),
        "+SMA50>SMA200(GC)": b & gc,
        "+C>SMA85 & GC":     b & (c > p["sma85"]) & gc,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", choices=["index", "basket", "both"], default="both")
    ap.add_argument("--rank-h", type=int, default=10, choices=HORIZONS)
    ap.add_argument("--period", default="max")
    args = ap.parse_args()

    names = {"index": INDEX, "basket": BASKET, "both": INDEX + BASKET}[args.universe]
    print(f"Downloading {len(names)} names ({args.universe})...")
    preps, base_fwd = [], {hh: [] for hh in HORIZONS}
    for sym in names:
        try:
            df = scan.download_daily(sym, period=args.period)
        except Exception:
            df = None
        if df is None or len(df) < 300:
            continue
        p = prep(df)
        preps.append(p)
        for hh in HORIZONS:
            base_fwd[hh].append(p["fwd"][hh].dropna())
    print(f"got data for {len(preps)}/{len(names)} names\n")
    base_mean = {hh: pd.concat(base_fwd[hh]).mean() for hh in HORIZONS}
    H = args.rank_h
    print(f"all-day baseline: D{H} mean {base_mean[H]:+.4f}\n")

    vnames = list(variants(preps[0]).keys())
    rows = []
    for name in vnames:
        n, fr = 0, {hh: [] for hh in HORIZONS}
        for p in preps:
            e = variants(p)[name].fillna(False)
            n += int(e.sum())
            for hh in HORIZONS:
                fr[hh].append(p["fwd"][hh][e].dropna())
        agg = {hh: pd.concat(fr[hh]) if fr[hh] else pd.Series(dtype=float) for hh in HORIZONS}
        rk = agg[H]
        sd = rk.std(ddof=1)
        t = (rk.mean() / (sd / np.sqrt(len(rk)))) if len(rk) > 1 and sd else np.nan
        rows.append({"variant": name, "n": n, "win5": (agg[5] > 0).mean(),
                     "win10": (agg[10] > 0).mean(), "win20": (agg[20] > 0).mean(),
                     f"mean{H}": rk.mean(), "edge": rk.mean() - base_mean[H], "t": t})
    tbl = pd.DataFrame(rows)

    print(f"=== break SMA 8/22 + long-MA trend filter (D{H} forward) ===")
    print(f"  {'variant':<20}{'n':>7}{'win5':>7}{'win10':>7}{'win20':>7}{f'mean{H}':>9}{'edge':>9}{'t':>7}")
    for _, r in tbl.iterrows():
        print(f"  {r['variant']:<20}{int(r['n']):>7}{r['win5']:>7.1%}{r['win10']:>7.1%}"
              f"{r['win20']:>7.1%}{r[f'mean{H}']:>+9.4f}{r['edge']:>+9.4f}{r['t']:>7.2f}")

    b = tbl[tbl["variant"] == "base (8/22)"].iloc[0]
    print(f"\n  base 8/22: win10 {b['win10']:.1%}, mean{H} {b[f'mean{H}']:+.4f}, n {int(b['n'])}")
    for _, r in tbl.iterrows():
        if r["variant"] == "base (8/22)":
            continue
        dwin = r["win10"] - b["win10"]
        dmean = r[f"mean{H}"] - b[f"mean{H}"]
        keep = (r["n"] / b["n"]) if b["n"] else np.nan
        print(f"  {r['variant']:<20} Δwin10 {dwin:>+6.1%}  Δmean{H} {dmean:>+.4f}  "
              f"keeps {keep:>4.0%} of fires")
    print("\nCaveat: no costs; a filter that lifts mean but keeps <30% of fires trades")
    print("frequency for quality — good for a hi-conv tier, fewer daily alerts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
