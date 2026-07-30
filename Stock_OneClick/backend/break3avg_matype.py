#!/usr/bin/env python3
"""break3avg: is EMA or SMA better for the fast/slow lines? Test all 4 type combos.

The signal compares price to a fast line and a slow line. The original uses fast=EMA,
slow=SMA. Here we sweep the TYPE of each line over {EMA, SMA} (4 combos) across a
length grid, and rank each by forward-return edge on the index+basket universe (same
methodology as break3avg_tune.py). The short confirm line (ema3) takes the fast type.

  entry = crossover(close, fastLine) and fastLine rising
          and crossover(close, slowLine) and short3 > slowLine

    ../../vcp_env/bin/python break3avg_matype.py            # rank by D10 edge
    ../../vcp_env/bin/python break3avg_matype.py --rank-h 5

Caveat: close-to-close forward returns, no costs; per-event edge over baseline.
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

FAST = [4, 5, 6, 7, 8]
SLOW = [20, 21, 22, 23, 24]
HORIZONS = [1, 5, 10, 20]
TYPES = [("EMA", "EMA"), ("EMA", "SMA"), ("SMA", "EMA"), ("SMA", "SMA")]  # (fast, slow)


def prep(df: pd.DataFrame) -> dict:
    o, h, l, c = (df[k].astype(float) for k in ("Open", "High", "Low", "Close"))
    ohlc4 = (o + h + l + c) / 4
    lengths = sorted(set(FAST + SLOW))
    return {
        "close": c,
        "line": {("EMA", n): ma(ohlc4, n, "EMA") for n in lengths}
                | {("SMA", n): ma(ohlc4, n, "SMA") for n in lengths},
        "short3": {"EMA": ma(ohlc4, 3, "EMA"), "SMA": ma(ohlc4, 3, "SMA")},
        "fwd": {hh: c.shift(-hh) / c - 1 for hh in HORIZONS},
    }


def fires(p: dict, fast: int, slow: int, ftype: str, stype: str) -> pd.Series:
    c = p["close"]
    fl = p["line"][(ftype, fast)]
    sl = p["line"][(stype, slow)]
    sh = p["short3"][ftype]
    return (crossover(c, fl) & (fl > fl.shift(1)) & crossover(c, sl) & (sh > sl)).fillna(False)


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
    print(f"all-day baseline D{H} mean {base_mean[H]:+.4f}\n")

    def eval_combo(fast, slow, ft, st):
        n, fr = 0, {hh: [] for hh in HORIZONS}
        for p in preps:
            e = fires(p, fast, slow, ft, st)
            n += int(e.sum())
            for hh in HORIZONS:
                fr[hh].append(p["fwd"][hh][e].dropna())
        agg = {hh: pd.concat(fr[hh]) if fr[hh] else pd.Series(dtype=float) for hh in HORIZONS}
        rk = agg[H]
        sd = rk.std(ddof=1)
        t = (rk.mean() / (sd / np.sqrt(len(rk)))) if len(rk) > 1 and sd else np.nan
        return {"fast": fast, "slow": slow, "ftype": ft, "stype": st, "n": n,
                "win5": (agg[5] > 0).mean(), "win10": (agg[10] > 0).mean(),
                f"mean{H}": rk.mean(), "edge": rk.mean() - base_mean[H], "t": t}

    all_rows = [eval_combo(f, s, ft, st) for (ft, st) in TYPES for f in FAST for s in SLOW]
    tbl = pd.DataFrame(all_rows)

    # best length pair within each type combo
    print(f"=== best length pair per line-type combo (ranked by D{H} mean) ===")
    print(f"  {'fast/slow type':<16}{'best f/s':>10}{'n':>7}{'win5':>7}{'win10':>7}"
          f"{f'mean{H}':>9}{'edge':>9}{'t':>7}")
    summary = []
    for (ft, st) in TYPES:
        sub = tbl[(tbl["ftype"] == ft) & (tbl["stype"] == st)].sort_values(f"mean{H}", ascending=False)
        b = sub.iloc[0]
        summary.append(b)
        label = f"{ft}/{st}"
        cur = " (current)" if (ft, st) == ("EMA", "SMA") else ""
        print(f"  {label:<16}{f'{int(b.fast)}/{int(b.slow)}':>10}{int(b.n):>7}{b.win5:>7.1%}"
              f"{b.win10:>7.1%}{b[f'mean{H}']:>+9.4f}{b.edge:>+9.4f}{b.t:>7.2f}{cur}")

    # overall winner across all combos
    top = tbl.sort_values(f"mean{H}", ascending=False).iloc[0]
    print(f"\n  -> overall best: {top.ftype}/{top.stype}  {int(top.fast)}/{int(top.slow)}  "
          f"(D{H} mean {top[f'mean{H}']:+.4f}, edge {top.edge:+.4f}, t={top.t:.2f}, n={int(top.n)})")

    # average edge per type combo (robustness: not just the single best cell)
    print(f"\n=== average D{H} edge across the whole length grid, per type combo ===")
    for (ft, st) in TYPES:
        sub = tbl[(tbl["ftype"] == ft) & (tbl["stype"] == st)]
        cur = "  <- current" if (ft, st) == ("EMA", "SMA") else ""
        print(f"  {ft}/{st:<6} mean edge {sub['edge'].mean():+.4f}   "
              f"mean n {sub['n'].mean():.0f}   mean t {sub['t'].mean():.2f}{cur}")
    best_type = max(TYPES, key=lambda ts: tbl[(tbl.ftype == ts[0]) & (tbl.stype == ts[1])]["edge"].mean())
    print(f"\n  -> best line-type combo by AVERAGE edge (more robust than one cell): "
          f"{best_type[0]}/{best_type[1]}")
    print("\nCaveat: no costs; differences between type combos are small — check both the")
    print("best-cell and the grid-average before concluding EMA vs SMA truly matters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
