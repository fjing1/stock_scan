#!/usr/bin/env python3
"""Backtest the 'trend alert main' Pine indicator's signals as long/cash strategies.

Same methodology as rsi_ma_sweep.py: each signal becomes a long-when-in / cash-
when-out rule, position applied NEXT day (no look-ahead), 15 bps/turn, pooled
equal-weight across the universe, ranked by Sharpe vs buy-and-hold with a per-year
walk-forward stability check.

The Pine script has many discrete signals; we test the tradeable entry/exit pairs:
  power      : combined2=(ema8+ma5)/2  crosses  combined=(ema8+sma13+sma21)/3
  ema8xma21  : buy30 (ema8 X up ma21, rising, cycle<60)   exit crossunder(ema8,ma21)
  ma13xma21  : buy40 (ma13 X up ma21, rising, cycle<60)   exit crossunder(ma13,ma21)
  trend      : trend_up regime on / trend_down regime off
  break3avg  : upbreak3avg (close over ema8&ma21, ema3>ma21) / downwarn (close under all 3)
  buy20      : stoch_up+price_up+close X ema8+ma13 rising+cycle<50  / short20

    ../../vcp_env/bin/python trend_alert_backtest.py --universe index
    ../../vcp_env/bin/python trend_alert_backtest.py --universe basket

Caveats: long-only cash, close-to-close, 15bps/turn. Basket survivorship-biased —
trust the strategy-vs-B&H comparison, not the absolute level.
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
from rsi_ma_sweep import COST, TD, BASKET, INDEX, metrics, pooled_daily, _apply  # noqa: E402


def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def stoch(src: pd.Series, n: int) -> pd.Series:
    lo = src.rolling(n).min()
    hi = src.rolling(n).max()
    return 100 * (src - lo) / (hi - lo).replace(0, np.nan)


def build_signals(df: pd.DataFrame) -> dict:
    """Recreate the Pine indicators/signals faithfully for one name."""
    o, h, l, c = (df[k].astype(float) for k in ("Open", "High", "Low", "Close"))
    v = df["Volume"].astype(float)
    ohlc4 = (o + h + l + c) / 4
    hl2 = (h + l) / 2

    ema3 = ohlc4.ewm(span=3, adjust=False).mean()
    ema8 = ohlc4.ewm(span=8, adjust=False).mean()          # tline
    ma5 = ohlc4.rolling(5).mean()
    ma10 = ohlc4.rolling(10).mean()
    ma13 = ohlc4.rolling(13).mean()
    ma20 = ohlc4.rolling(20).mean()
    ma21 = ohlc4.rolling(21).mean()
    ma50 = ohlc4.rolling(50).mean()
    combined = (ema8 + ma13 + ma21) / 3
    combined2 = (ema8 + ma5) / 2

    # fisher transform + cycle
    high_ = hl2.rolling(9).max()
    low_ = hl2.rolling(9).min()
    raw = 0.66 * ((hl2 - low_) / (high_ - low_).replace(0, np.nan) - 0.5)
    value = np.zeros(len(df))
    prev = 0.0
    rv = raw.values
    for i in range(len(df)):
        x = rv[i] if np.isfinite(rv[i]) else 0.0
        x = x + 0.67 * prev
        x = 0.999 if x > 0.99 else (-0.999 if x < -0.99 else x)
        value[i] = x
        prev = x
    fish = np.zeros(len(df))
    pf = 0.0
    for i in range(len(df)):
        pf = 0.5 * np.log((1 + value[i]) / (1 - value[i])) + 0.5 * pf
        fish[i] = pf
    fish1 = pd.Series(fish, index=df.index)
    k3 = stoch(ohlc4, 14).rolling(3).mean() * 0.5 + fish1 * 8 + 10
    d3 = k3.rolling(3).mean()
    kd_diff = k3 - d3
    cycle = stoch(k3, 10)

    stoch_up = (kd_diff > kd_diff.shift(1)) & (k3 > d3) & (d3 > d3.shift(1))
    price_up = c > c.shift(2)
    price_down = c < c.shift(2)
    vol_up = (v > v.shift(1)) & (v.shift(1) > v.shift(2))

    # entries
    buy20 = stoch_up & price_up & crossover(ohlc4, ema8) & (ma13 > ma13.shift(1)) & (cycle < 50)
    buy30 = crossover(ema8, ma21) & (ma21 > ma21.shift(1)) & (cycle < 60)
    buy40 = crossover(ma13, ma21) & (ma21 > ma21.shift(1)) & (cycle < 60)
    upbreak3avg = crossover(c, ema8) & (ema8 > ema8.shift(1)) & crossover(c, ma21) & (ema3 > ma21)
    powerup = crossover(combined2, combined)

    # exits
    short20 = crossunder(c, ma21) & (ma21 < ma21.shift(1)) & (cycle > 40)
    downtrend1 = crossunder(ema8, ma21)
    downtrend2 = crossunder(ma13, ma21)
    downwarn = crossunder(c, ema8) & crossunder(c, ma13) & crossunder(c, ma21)
    powerdown = crossunder(combined2, combined)

    # regime (stateful on its own)
    trend_up = ((ohlc4 > ma10) | (c > ma10)) & (ma20.shift(1) > ma20.shift(2)) & (ma20 > ma50)
    trend_down = ((ohlc4 < ma10) | (c < ma20)) & (ma20.shift(1) < ma20.shift(2)) & (ma20 < ma50)

    return {
        "close": c,
        "power": (powerup, powerdown),
        "ema8xma21": (buy30, downtrend1),
        "ma13xma21": (buy40, downtrend2),
        "trend": (trend_up, trend_down),
        "break3avg": (upbreak3avg, downwarn),
        "buy20": (buy20, short20),
    }


def build_state(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """Stateful long/flat: 1 from the bar entry fires until exit fires (exit wins ties)."""
    e = entry.fillna(False).values
    x = exit_.fillna(False).values
    st = np.zeros(len(e))
    cur = 0.0
    for i in range(len(e)):
        if cur == 1.0 and x[i]:
            cur = 0.0
        elif cur == 0.0 and e[i]:
            cur = 1.0
        st[i] = cur
    return pd.Series(st, index=entry.index)


def strat_daily(df: pd.DataFrame, name: str) -> pd.Series:
    sig = build_signals(df)
    entry, exit_ = sig[name]
    state = build_state(entry, exit_)
    return _apply(sig["close"], state)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", choices=["index", "basket", "both"], default="index")
    ap.add_argument("--period", default="max")
    args = ap.parse_args()

    names = {"index": INDEX, "basket": BASKET, "both": INDEX + BASKET}[args.universe]
    print(f"Downloading {len(names)} names ({args.universe})...")
    frames: dict[str, pd.DataFrame] = {}
    for sym in names:
        try:
            df = scan.download_daily(sym, period=args.period)
        except Exception:
            df = None
        if df is None or len(df) < 300:
            continue
        frames[sym] = df
    print(f"got data for {len(frames)}/{len(names)} names")
    if not frames:
        print("no data")
        return 1
    rng = pd.DatetimeIndex(sorted(set().union(*[f.index for f in frames.values()])))
    print(f"date range {rng.min().date()} .. {rng.max().date()}\n")

    closes = {s: f["Close"].astype(float) for s, f in frames.items()}
    bh = pooled_daily(closes, lambda c: c.pct_change().fillna(0.0))
    bhm = metrics(bh)
    print(f"{'BUY&HOLD (pooled)':<14} total {bhm['total']:>+9.1%}  CAGR {bhm['cagr']:>+7.2%}  "
          f"Sharpe {bhm['sharpe']:>5.2f}  maxDD {bhm['maxdd']:>+7.1%}")

    strat_names = ["power", "ema8xma21", "ma13xma21", "trend", "break3avg", "buy20"]
    print("\n=== trend-alert signals as long/cash strategies ===")
    print(f"  {'signal':<12}{'total':>10}{'CAGR':>9}{'Sharpe':>8}{'maxDD':>9}{'%inMkt':>8}{'vs B&H Sh':>11}")
    results = []
    for nm in strat_names:
        pooled = pooled_daily(frames, lambda f, n=nm: strat_daily(f, n))
        m = metrics(pooled)
        # % time in market (avg position across pooled names)
        pos = pooled_daily(frames, lambda f, n=nm: build_state(*build_signals(f)[n]).shift(1).fillna(0.0))
        inmkt = pos.mean()
        results.append((nm, m, pooled, inmkt))
        flag = "  <-- beats B&H" if m["sharpe"] > bhm["sharpe"] else ""
        print(f"  {nm:<12}{m['total']:>+10.1%}{m['cagr']:>+9.2%}{m['sharpe']:>8.2f}"
              f"{m['maxdd']:>+9.1%}{inmkt:>8.0%}{m['sharpe']-bhm['sharpe']:>+11.2f}{flag}")

    best = max(results, key=lambda r: r[1]["sharpe"])
    print(f"\n  -> best signal by Sharpe: {best[0]}  (Sharpe {best[1]['sharpe']:.2f}, "
          f"CAGR {best[1]['cagr']:+.2%}, maxDD {best[1]['maxdd']:+.1%}, {best[3]:.0%} in market)")

    # walk-forward per-year for the best signal
    best_daily = best[2]
    print(f"\n=== walk-forward: '{best[0]}' vs B&H, by calendar year (Sharpe) ===")
    print(f"  {'year':>6}{'strat Sh':>10}{'B&H Sh':>9}{'strat ret':>11}{'B&H ret':>10}{'win?':>6}")
    wins = n_yr = 0
    for y in sorted(set(best_daily.index.year)):
        sd = best_daily[best_daily.index.year == y].dropna()
        bd = bh[bh.index.year == y].dropna()
        if len(sd) < 30:
            continue
        n_yr += 1
        ss = sd.mean() / sd.std(ddof=1) * np.sqrt(TD) if sd.std(ddof=1) else np.nan
        bs = bd.mean() / bd.std(ddof=1) * np.sqrt(TD) if bd.std(ddof=1) else np.nan
        win = ss > bs
        wins += int(win)
        print(f"  {y:>6}{ss:>+10.2f}{bs:>+9.2f}{(1+sd).prod()-1:>+11.1%}{(1+bd).prod()-1:>+10.1%}"
              f"{('Y' if win else 'n'):>6}")
    print(f"  -> strat Sharpe beat B&H in {wins}/{n_yr} years")

    print("\nCaveats: long-only cash, close-to-close, 15bps/turn. Basket survivorship-biased —")
    print("trust the strategy-vs-B&H comparison, not the level. A signal in the market <60%")
    print("of the time that still trails B&H has no timing edge, just less exposure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
