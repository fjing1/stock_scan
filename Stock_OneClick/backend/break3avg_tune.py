#!/usr/bin/env python3
"""Fine-tune the break3avg (upbreak3avg) signal: sweep fast EMA x slow SMA.

Baseline signal is fast=8, slow=21:
  entry = crossover(close, emaFast) and emaFast>emaFast[1]
          and crossover(close, smaSlow) and ema3 > smaSlow
We sweep fast in {2..8}, slow in {18..24} and rank each combo by its forward-return
edge (win rate + mean vs the all-day baseline), which is what a scanner ENTRY cares
about, independent of any exit rule. Pooled across the index+basket universe for
sample size. Each name is downloaded once, then all combos evaluated in memory.

    ../../vcp_env/bin/python break3avg_tune.py                 # rank by D10 edge
    ../../vcp_env/bin/python break3avg_tune.py --rank-h 5
    ../../vcp_env/bin/python break3avg_tune.py --universe basket

Caveat: close-to-close forward returns, no costs. Edge is per-event over baseline;
a combo that fires very rarely (low n) is not trustworthy even with a high mean.
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
from rsi_ma_sweep import BASKET, INDEX  # noqa: E402
from trend_alert_backtest import crossover  # noqa: E402

FAST = [2, 3, 4, 5, 6, 7, 8]
SLOW = [18, 19, 20, 21, 22, 23, 24]
HORIZONS = [1, 5, 10, 20]


def prep(df: pd.DataFrame) -> dict:
    """Precompute the pieces reused across all combos for one name."""
    o, h, l, c = (df[k].astype(float) for k in ("Open", "High", "Low", "Close"))
    ohlc4 = (o + h + l + c) / 4
    return {
        "ohlc4": ohlc4,
        "close": c,
        "ema3": ohlc4.ewm(span=3, adjust=False).mean(),
        "emaF": {f: ohlc4.ewm(span=f, adjust=False).mean() for f in FAST},
        "smaS": {s: ohlc4.rolling(s).mean() for s in SLOW},
        "fwd": {hh: c.shift(-hh) / c - 1 for hh in HORIZONS},
    }


def fires(p: dict, fast: int, slow: int) -> pd.Series:
    c, emaF, smaS, ema3 = p["close"], p["emaF"][fast], p["smaS"][slow], p["ema3"]
    return (crossover(c, emaF) & (emaF > emaF.shift(1)) & crossover(c, smaS) & (ema3 > smaS)).fillna(False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", choices=["index", "basket", "both"], default="both")
    ap.add_argument("--rank-h", type=int, default=10, choices=HORIZONS, help="horizon to rank combos by")
    ap.add_argument("--period", default="max")
    args = ap.parse_args()

    names = {"index": INDEX, "basket": BASKET, "both": INDEX + BASKET}[args.universe]
    print(f"Downloading {len(names)} names ({args.universe})...")
    preps = []
    base = {hh: [] for hh in HORIZONS}   # all-day baseline forward returns (pooled)
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
    base_win = {hh: (pd.concat(base[hh]) > 0).mean() for hh in HORIZONS}
    print("all-day baseline:  " + "   ".join(
        f"D{hh} win {base_win[hh]:.1%} mean {base_mean[hh]:+.4f}" for hh in HORIZONS))

    rows = []
    for f in FAST:
        for s in SLOW:
            n = 0
            fr = {hh: [] for hh in HORIZONS}
            for p in preps:
                e = fires(p, f, s)
                n += int(e.sum())
                for hh in HORIZONS:
                    fr[hh].append(p["fwd"][hh][e].dropna())
            agg = {hh: pd.concat(fr[hh]) if fr[hh] else pd.Series(dtype=float) for hh in HORIZONS}
            rk = agg[args.rank_h]
            sd = rk.std(ddof=1)
            t = (rk.mean() / (sd / np.sqrt(len(rk)))) if len(rk) > 1 and sd else np.nan
            rows.append({
                "fast": f, "slow": s, "n": n,
                **{f"win{hh}": (agg[hh] > 0).mean() for hh in HORIZONS},
                **{f"mean{hh}": agg[hh].mean() for hh in HORIZONS},
                "edge": rk.mean() - base_mean[args.rank_h],
                "t": t,
            })
    tbl = pd.DataFrame(rows).sort_values(f"mean{args.rank_h}", ascending=False).reset_index(drop=True)

    H = args.rank_h
    print(f"\n=== break3avg combo sweep, ranked by D{H} mean forward return ===")
    print(f"  {'fast':>4}{'slow':>5}{'n':>7}{'win5':>7}{'win10':>7}"
          f"{f'mean{H}':>9}{f'edge{H}':>9}{'t':>7}   baseline vs")
    for _, r in tbl.iterrows():
        star = " *" if (r["fast"] == 8 and r["slow"] == 21) else ""
        print(f"  {int(r['fast']):>4}{int(r['slow']):>5}{int(r['n']):>7}{r['win5']:>7.1%}"
              f"{r['win10']:>7.1%}{r[f'mean{H}']:>+9.4f}{r['edge']:>+9.4f}{r['t']:>7.2f}"
              f"   {base_mean[H]:+.4f}{star}")

    best = tbl.iloc[0]
    print(f"\n  -> best combo by D{H} mean: fast={int(best['fast'])} slow={int(best['slow'])}  "
          f"(n={int(best['n'])}, D{H} mean {best[f'mean{H}']:+.4f}, edge {best['edge']:+.4f}, "
          f"t={best['t']:.2f}, win5 {best['win5']:.1%})")
    baseline = tbl[(tbl["fast"] == 8) & (tbl["slow"] == 21)]
    if not baseline.empty:
        b = baseline.iloc[0]
        print(f"  current (8/21):     n={int(b['n'])}, D{H} mean {b[f'mean{H}']:+.4f}, "
              f"edge {b['edge']:+.4f}, t={b['t']:.2f}  (* row above)")
    print("\nCaveat: no costs; per-event edge over baseline. Prefer combos with BOTH a high")
    print("edge AND enough fires (n) + t>2; a rare-firing combo with a big mean is likely noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
