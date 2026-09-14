#!/usr/bin/env python3
"""Test COMBINED (blended) moving averages from the trend-alert script as signals.

    combined  = (ema8 + sma13 + sma21) / 3
    combined1 = (ema8 + sma13 + sma21 + ma50) / 4
    combined2 = (ema8 + ma5) / 2          # fast blend
All on ohlc4. The script's powerup = crossover(combined2, combined).

We compare, on the same index+basket universe and same forward-edge methodology as
break3avg_tune.py, a set of candidate signals:
  - power        : combined2 X combined            (script's own powerup)
  - power1       : combined2 X combined1
  - breakC       : close X combined  (blend rising)      [breakout through the blend]
  - breakC1      : close X combined1 (blend rising)
  - breakC_conf  : breakC  and ema3 > combined           [+ short-EMA confirm]
  - breakC1_conf : breakC1 and ema3 > combined1
  - REF break3avg 8/23 (EMA fast / SMA slow)             [current scanner default]
  - REF break SMA/SMA 8/22                               [prior best-per-signal]

    ../../vcp_env/bin/python break3avg_combined.py

Caveat: close-to-close forward returns, no costs; per-event edge over baseline.
Prefer signals with BOTH high edge AND enough fires (n) + t>~3.
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
    ema8 = ma(ohlc4, 8, "EMA")
    ema3 = ma(ohlc4, 3, "EMA")
    ma5 = ma(ohlc4, 5, "SMA")
    sma13 = ma(ohlc4, 13, "SMA")
    sma21 = ma(ohlc4, 21, "SMA")
    ma50 = ma(ohlc4, 50, "SMA")
    combined = (ema8 + sma13 + sma21) / 3
    combined1 = (ema8 + sma13 + sma21 + ma50) / 4
    combined2 = (ema8 + ma5) / 2
    return {
        "close": c, "ema3": ema3, "ema8": ema8,
        "sma3": ma(ohlc4, 3, "SMA"),
        "sma8": ma(ohlc4, 8, "SMA"), "sma22": ma(ohlc4, 22, "SMA"), "sma23": ma(ohlc4, 23, "SMA"),
        "combined": combined, "combined1": combined1, "combined2": combined2,
        "fwd": {hh: c.shift(-hh) / c - 1 for hh in HORIZONS},
    }


def signals(p: dict) -> dict:
    c, ema3, ema8 = p["close"], p["ema3"], p["ema8"]
    cb, cb1, cb2 = p["combined"], p["combined1"], p["combined2"]
    rising = lambda s: s > s.shift(1)
    return {
        "power (cb2 X cb)":     crossover(cb2, cb),
        "power1 (cb2 X cb1)":   crossover(cb2, cb1),
        "breakC (px X cb)":     crossover(c, cb) & rising(cb),
        "breakC1 (px X cb1)":   crossover(c, cb1) & rising(cb1),
        "breakC_conf":          crossover(c, cb) & rising(cb) & (ema3 > cb),
        "breakC1_conf":         crossover(c, cb1) & rising(cb1) & (ema3 > cb1),
        "REF break3avg 8/23":   crossover(c, ema8) & rising(ema8) & crossover(c, p["sma23"]) & (ema3 > p["sma23"]),
        "REF break SMA 8/22":   crossover(c, p["sma8"]) & rising(p["sma8"]) & crossover(c, p["sma22"]) & (p["sma3"] > p["sma22"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", choices=["index", "basket", "both"], default="both")
    ap.add_argument("--rank-h", type=int, default=10, choices=HORIZONS)
    ap.add_argument("--period", default="max")
    args = ap.parse_args()

    names = {"index": INDEX, "basket": BASKET, "both": INDEX + BASKET}[args.universe]
    print(f"Downloading {len(names)} names ({args.universe})...")
    preps, base = [], {hh: [] for hh in HORIZONS}
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
            base[hh].append(p["fwd"][hh].dropna())
    print(f"got data for {len(preps)}/{len(names)} names\n")
    base_mean = {hh: pd.concat(base[hh]).mean() for hh in HORIZONS}
    H = args.rank_h
    print(f"all-day baseline: D{H} mean {base_mean[H]:+.4f}\n")

    sig_names = list(signals(preps[0]).keys())
    rows = []
    for name in sig_names:
        n, fr = 0, {hh: [] for hh in HORIZONS}
        for p in preps:
            e = signals(p)[name].fillna(False)
            n += int(e.sum())
            for hh in HORIZONS:
                fr[hh].append(p["fwd"][hh][e].dropna())
        agg = {hh: pd.concat(fr[hh]) if fr[hh] else pd.Series(dtype=float) for hh in HORIZONS}
        rk = agg[H]
        sd = rk.std(ddof=1)
        t = (rk.mean() / (sd / np.sqrt(len(rk)))) if len(rk) > 1 and sd else np.nan
        rows.append({"signal": name, "n": n, "win5": (agg[5] > 0).mean(),
                     "win10": (agg[10] > 0).mean(), f"mean{H}": rk.mean(),
                     "edge": rk.mean() - base_mean[H], "t": t})
    tbl = pd.DataFrame(rows).sort_values(f"mean{H}", ascending=False).reset_index(drop=True)

    print(f"=== combined-MA signals vs references (ranked by D{H} mean forward return) ===")
    print(f"  {'signal':<22}{'n':>7}{'win5':>7}{'win10':>7}{f'mean{H}':>9}{'edge':>9}{'t':>7}")
    for _, r in tbl.iterrows():
        print(f"  {r['signal']:<22}{int(r['n']):>7}{r['win5']:>7.1%}{r['win10']:>7.1%}"
              f"{r[f'mean{H}']:>+9.4f}{r['edge']:>+9.4f}{r['t']:>7.2f}")
    top = tbl.iloc[0]
    print(f"\n  -> best: {top['signal']}  (D{H} mean {top[f'mean{H}']:+.4f}, edge {top['edge']:+.4f}, "
          f"t={top['t']:.2f}, n={int(top['n'])})")
    print("\nCaveat: no costs; per-event edge over baseline. A blended-MA crossover that fires")
    print("often but with low edge is just tracking the trend, not adding timing information.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
