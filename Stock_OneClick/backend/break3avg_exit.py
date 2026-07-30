#!/usr/bin/env python3
"""Exit-rule + profit-target study for the break SMA 8/22 momentum entry.

The scanner emits entries only. This simulates exits on those trades to answer:
what's a realistic profit-target range, and which exit rule gives the best
per-trade expectancy / profit factor? Prior research (RESEARCH.md #5/#10/#11,
exit-strategy-edge-is-survivorship): momentum entries want a TREND-following /
time-based exit; tight stops & trailing hurt; sell-into-strength is for MR not
momentum.

Entry: at the close of the break SMA 8/22 bar (day 0). Exit rules compared:
  time D5/D10/D20/D40   : fixed hold
  close<SMA10 / SMA20   : trend-follow exit (leave when trend breaks)
  downwarn              : Pine exit (close under sma8 & sma13 & sma22)
  chandelier 3xATR      : trailing stop from highest high since entry
  target/stop grids     : first of +T% / -S% to trade (else close at D20)
Also prints the MFE/MAE distribution (max favorable/adverse excursion over 20
bars) to ground the profit-target range.

    ../../vcp_env/bin/python break3avg_exit.py

Metrics: win%, avg win, avg loss, expectancy (mean per-trade ret), profit factor,
avg hold. Caveat: close-to-close, no costs; 42-name survivor universe flatters
absolutes — trust the RULE-vs-RULE comparison. 15bps/side is noted where relevant.
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

MAXHOLD = 60


def collect(df: pd.DataFrame):
    o, h, l, c = (df[k].astype(float) for k in ("Open", "High", "Low", "Close"))
    ohlc4 = (o + h + l + c) / 4
    sma8 = ma(ohlc4, 8, "SMA")
    sma22 = ma(ohlc4, 22, "SMA")
    sma3 = ma(ohlc4, 3, "SMA")
    sma13 = ma(ohlc4, 13, "SMA")
    sma10 = c.rolling(10).mean()
    sma20 = c.rolling(20).mean()
    entry = (crossover(c, sma8) & (sma8 > sma8.shift(1))
             & crossover(c, sma22) & (sma3 > sma22)).fillna(False)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    return {k: v.values for k, v in dict(
        c=c, h=h, l=l, sma8=sma8, sma13=sma13, sma22=sma22, sma10=sma10, sma20=sma20, atr=atr
    ).items()} | {"entry_idx": np.where(entry.values)[0]}


def sim_time(d, i, N):
    j = min(i + N, len(d["c"]) - 1)
    return d["c"][j] / d["c"][i] - 1, j - i


def sim_trend(d, i, ma_key):
    c, m = d["c"], d[ma_key]
    for j in range(i + 1, min(i + MAXHOLD, len(c))):
        if np.isfinite(m[j]) and c[j] < m[j]:
            return c[j] / c[i] - 1, j - i
    j = min(i + MAXHOLD, len(c) - 1)
    return c[j] / c[i] - 1, j - i


def sim_downwarn(d, i):
    c = d["c"]
    for j in range(i + 1, min(i + MAXHOLD, len(c))):
        if c[j] < d["sma8"][j] and c[j] < d["sma13"][j] and c[j] < d["sma22"][j]:
            return c[j] / c[i] - 1, j - i
    j = min(i + MAXHOLD, len(c) - 1)
    return c[j] / c[i] - 1, j - i


def sim_chandelier(d, i, mult=3.0):
    c, h, atr = d["c"], d["h"], d["atr"]
    hh = h[i]
    for j in range(i + 1, min(i + MAXHOLD, len(c))):
        hh = max(hh, h[j])
        if np.isfinite(atr[j]) and c[j] < hh - mult * atr[j]:
            return c[j] / c[i] - 1, j - i
    j = min(i + MAXHOLD, len(c) - 1)
    return c[j] / c[i] - 1, j - i


def sim_target_stop(d, i, tgt, stop, maxhold=20):
    c, h, l = d["c"], d["h"], d["l"]
    e = c[i]
    for j in range(i + 1, min(i + maxhold, len(c))):
        if l[j] <= e * (1 - stop):      # stop checked first (conservative)
            return -stop, j - i
        if h[j] >= e * (1 + tgt):
            return tgt, j - i
    j = min(i + maxhold, len(c) - 1)
    return c[j] / c[i] - 1, j - i


def metrics(rets, holds):
    r = np.array(rets)
    hd = np.array(holds)
    if len(r) == 0:
        return {}
    wins, losses = r[r > 0], r[r <= 0]
    pf = wins.sum() / -losses.sum() if losses.sum() < 0 else np.inf
    return {"n": len(r), "win": (r > 0).mean(), "avg_win": wins.mean() if len(wins) else 0,
            "avg_loss": losses.mean() if len(losses) else 0, "exp": r.mean(),
            "pf": pf, "hold": hd.mean()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", choices=["index", "basket", "both"], default="both")
    ap.add_argument("--period", default="max")
    args = ap.parse_args()

    names = {"index": INDEX, "basket": BASKET, "both": INDEX + BASKET}[args.universe]
    print(f"Downloading {len(names)} names ({args.universe})...")
    data = []
    for sym in names:
        try:
            df = scan.download_daily(sym, period=args.period)
        except Exception:
            df = None
        if df is None or len(df) < 300:
            continue
        data.append(collect(df))
    n_entries = sum(len(d["entry_idx"]) for d in data)
    print(f"got {len(data)} names, {n_entries} break-8/22 entries\n")

    # MFE / MAE over 20 bars -> ground the profit-target range
    mfe, mae, term20 = [], [], []
    for d in data:
        c, h, l = d["c"], d["h"], d["l"]
        for i in d["entry_idx"]:
            end = min(i + 20, len(c))
            if end - i < 2:
                continue
            mfe.append(h[i + 1:end].max() / c[i] - 1)
            mae.append(l[i + 1:end].min() / c[i] - 1)
            term20.append(c[end - 1] / c[i] - 1)
    mfe, mae = np.array(mfe), np.array(mae)
    print("=== 20-bar excursion distribution (grounds the target range) ===")
    print(f"  MFE (max up) pct: 25th {np.percentile(mfe,25):+.1%}  median {np.percentile(mfe,50):+.1%}"
          f"  75th {np.percentile(mfe,75):+.1%}  90th {np.percentile(mfe,90):+.1%}")
    print(f"  MAE (max down)  : 25th {np.percentile(mae,25):+.1%}  median {np.percentile(mae,50):+.1%}"
          f"  75th {np.percentile(mae,75):+.1%}  90th {np.percentile(mae,90):+.1%}")
    print(f"  -> most trades see a favorable pop near +{np.percentile(mfe,50):.0%} and dip to "
          f"{np.percentile(mae,50):.0%} within 20 days.\n")

    rules = {
        "time D5": lambda d, i: sim_time(d, i, 5),
        "time D10": lambda d, i: sim_time(d, i, 10),
        "time D20": lambda d, i: sim_time(d, i, 20),
        "time D40": lambda d, i: sim_time(d, i, 40),
        "close<SMA10": lambda d, i: sim_trend(d, i, "sma10"),
        "close<SMA20": lambda d, i: sim_trend(d, i, "sma20"),
        "downwarn(Pine)": lambda d, i: sim_downwarn(d, i),
        "chandelier 3ATR": lambda d, i: sim_chandelier(d, i, 3.0),
        "tgt+6/stop-3": lambda d, i: sim_target_stop(d, i, 0.06, 0.03),
        "tgt+8/stop-4": lambda d, i: sim_target_stop(d, i, 0.08, 0.04),
        "tgt+10/stop-5": lambda d, i: sim_target_stop(d, i, 0.10, 0.05),
        "tgt+5/stop-5": lambda d, i: sim_target_stop(d, i, 0.05, 0.05),
    }

    print("=== exit rules on break-8/22 trades (per-trade, no costs) ===")
    print(f"  {'exit rule':<17}{'n':>6}{'win%':>7}{'avgWin':>8}{'avgLoss':>8}{'expect':>8}{'PF':>6}{'hold':>6}")
    results = []
    for name, fn in rules.items():
        rets, holds = [], []
        for d in data:
            for i in d["entry_idx"]:
                if i + 1 >= len(d["c"]):
                    continue
                r, hd = fn(d, i)
                rets.append(r)
                holds.append(hd)
        m = metrics(rets, holds)
        results.append((name, m))
        print(f"  {name:<17}{m['n']:>6}{m['win']:>7.1%}{m['avg_win']:>+8.2%}{m['avg_loss']:>+8.2%}"
              f"{m['exp']:>+8.2%}{m['pf']:>6.2f}{m['hold']:>6.1f}")

    best_exp = max(results, key=lambda x: x[1]["exp"])
    best_pf = max(results, key=lambda x: x[1]["pf"])
    print(f"\n  -> best expectancy: {best_exp[0]} ({best_exp[1]['exp']:+.2%}/trade, "
          f"PF {best_exp[1]['pf']:.2f}, hold {best_exp[1]['hold']:.0f}d)")
    print(f"  -> best profit factor: {best_pf[0]} (PF {best_pf[1]['pf']:.2f}, "
          f"win {best_pf[1]['win']:.0%}, expect {best_pf[1]['exp']:+.2%})")
    print("\nNote: ~15bps/side round-trip (~0.3%) eats tight target/stop rules more than long")
    print("holds. Survivor universe flatters let-winners-run; a stop looks worse here than")
    print("it would on the broad universe (per exit-strategy-edge-is-survivorship).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
