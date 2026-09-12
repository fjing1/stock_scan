#!/usr/bin/env python3
"""RSI(8,22) MA-crossover: PRICE-sourced RSI vs OBV-sourced RSI, head to head.

xunlong_panel.pine's RSI now runs on OBV instead of price (see obvSource in that file), with a
buy alert on ma8 crossing over ma22. This tests whether that switch, and the (8,22) pair itself,
actually has a forward-return edge -- using the EXACT same harness as rsi_ma_sweep.py (same
universe, cost model, next-day fill, walk-forward-by-year convention) so PRICE and OBV are
directly comparable to each other and to [[rsi-ma-crossover-no-edge]] (which only ever tested
price).

OBV here matches the Pine script exactly: cumulative sum of volume signed by the direction of
the close-to-close change (`ta.cum(math.sign(ta.change(close)) * volume)`), then Wilder-RSI'd
the same way as price.

    ../../vcp_env/bin/python rsi_obv_ma_compare.py --universe both
    ../../vcp_env/bin/python rsi_obv_ma_compare.py --universe both --fast 8 --slow 22 --sweep
"""
from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import rsi_ma_sweep as R
import scan_stocks as scan

TD = R.TD


def obv_series(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Standard OBV, matching xunlong_panel.pine's `ta.cum(math.sign(ta.change(close)) * volume)`."""
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume).cumsum()


def rsi_wilder_on(series: pd.Series, n: int) -> pd.Series:
    """Wilder's RSI on an arbitrary series (price or OBV) -- same formula as R.rsi_wilder,
    generalized to a source other than price."""
    d = series.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    rs = up.ewm(alpha=1 / n, adjust=False).mean() / dn.ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + rs)


def strat_returns_dual_src(close: pd.Series, volume: pd.Series, rsi_len: int, fast: int,
                            slow: int, ma_kind: str, source: str) -> pd.Series:
    close = close.astype(float)
    src = close if source == "price" else obv_series(close, volume.astype(float))
    r = rsi_wilder_on(src, rsi_len)
    f = r if fast <= 1 else R.ma(r, fast, ma_kind)
    s = R.ma(r, slow, ma_kind)
    signal = (f >= s).astype(float)
    return R._apply(close, signal)


def pooled_daily_df(dfs: dict, fn) -> pd.Series:
    per = {sym: fn(df) for sym, df in dfs.items()}
    idx = pd.DatetimeIndex(sorted(set().union(*[s.index for s in per.values()])))
    mat = pd.DataFrame({sym: s.reindex(idx) for sym, s in per.items()})
    return mat.mean(axis=1)


def walk_forward(daily: pd.Series, bh: pd.Series, label: str):
    print(f"\n  walk-forward ({label}) vs B&H, by calendar year (Sharpe):")
    print(f"    {'year':>6}{'strat Sh':>10}{'B&H Sh':>9}{'strat ret':>11}{'B&H ret':>10}{'win?':>6}")
    wins, n_yr = 0, 0
    for y in sorted(set(daily.index.year)):
        sd = daily[daily.index.year == y].dropna()
        bd = bh[bh.index.year == y].dropna()
        if len(sd) < 30:
            continue
        n_yr += 1
        ss = sd.mean() / sd.std(ddof=1) * np.sqrt(TD) if sd.std(ddof=1) else np.nan
        bs = bd.mean() / bd.std(ddof=1) * np.sqrt(TD) if bd.std(ddof=1) else np.nan
        win = ss > bs
        wins += int(win)
        print(f"    {y:>6}{ss:>+10.2f}{bs:>+9.2f}{(1+sd).prod()-1:>+11.1%}"
              f"{(1+bd).prod()-1:>+10.1%}{('Y' if win else 'n'):>6}")
    print(f"    -> {label} beat B&H Sharpe in {wins}/{n_yr} years")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", choices=["index", "basket", "both"], default="both")
    ap.add_argument("--rsi-len", type=int, default=14)
    ap.add_argument("--ma-type", default="SMA", choices=["SMA", "EMA", "RMA", "WMA"])
    ap.add_argument("--fast", type=int, default=8)
    ap.add_argument("--slow", type=int, default=22)
    ap.add_argument("--period", default="max")
    ap.add_argument("--sweep", action="store_true",
                     help="also sweep fast/slow pairs for OBV (does PRICE beat OBV only at 8/22, or in general?)")
    args = ap.parse_args()

    names = {"index": R.INDEX, "basket": R.BASKET, "both": R.INDEX + R.BASKET}[args.universe]
    print(f"Downloading {len(names)} names ({args.universe})...")
    dfs: dict[str, pd.DataFrame] = {}
    for sym in names:
        try:
            df = scan.download_daily(sym, period=args.period)
        except Exception:
            df = None
        if df is None or len(df) < 300 or "Volume" not in df.columns:
            continue
        dfs[sym] = df[["Close", "Volume"]].astype(float)
    print(f"got data for {len(dfs)}/{len(names)} names")
    if not dfs:
        print("no data")
        return 1
    rng = pd.DatetimeIndex(sorted(set().union(*[d.index for d in dfs.values()])))
    print(f"date range {rng.min().date()} .. {rng.max().date()}   "
          f"RSI={args.rsi_len} MA={args.ma_type} fast/slow={args.fast}/{args.slow}\n")

    bh = pooled_daily_df(dfs, lambda d: d["Close"].pct_change().fillna(0.0))
    bhm = R.metrics(bh)
    print(f"{'BUY&HOLD (pooled)':<22} total {bhm['total']:>+8.1%}  CAGR {bhm['cagr']:>+7.2%}  "
          f"Sharpe {bhm['sharpe']:>5.2f}  maxDD {bhm['maxdd']:>+7.1%}")

    print(f"\n=== RSI({args.rsi_len}) MA({args.fast},{args.slow}) crossover: PRICE vs OBV as the RSI source ===")
    print(f"  {'source':>8}{'total':>10}{'CAGR':>9}{'Sharpe':>8}{'maxDD':>9}{'vs B&H Sh':>11}")
    results = {}
    for source in ("price", "obv"):
        pooled = pooled_daily_df(dfs, lambda d, s=source: strat_returns_dual_src(
            d["Close"], d["Volume"], args.rsi_len, args.fast, args.slow, args.ma_type, s))
        m = R.metrics(pooled)
        results[source] = (m, pooled)
        flag = "  <-- beats B&H" if m["sharpe"] > bhm["sharpe"] else ""
        print(f"  {source:>8}{m['total']:>+10.1%}{m['cagr']:>+9.2%}{m['sharpe']:>8.2f}"
              f"{m['maxdd']:>+9.1%}{m['sharpe']-bhm['sharpe']:>+11.2f}{flag}")

    d_sharpe = results["obv"][0]["sharpe"] - results["price"][0]["sharpe"]
    print(f"\n  OBV vs PRICE Sharpe delta at ({args.fast},{args.slow}): {d_sharpe:+.2f}  "
          f"({'OBV better' if d_sharpe > 0 else 'PRICE better' if d_sharpe < 0 else 'tie'})")

    for source in ("price", "obv"):
        walk_forward(results[source][1], bh, f"{source.upper()} ({args.fast}/{args.slow})")

    if args.sweep:
        pairs = [(f, s) for f in [1, 2, 3, 5, 7, 8] for s in [9, 14, 21, 22, 30, 50] if f < s]
        print(f"\n=== OBV sweep: is (8,22) special, or does OBV win/lose everywhere? ===")
        print(f"  {'fast':>5}{'slow':>6}{'total':>10}{'CAGR':>9}{'Sharpe':>8}{'maxDD':>9}"
              f"{'vs B&H Sh':>11}{'vs PRICE Sh':>13}")
        for f, s in pairs:
            obv_pooled = pooled_daily_df(dfs, lambda d, ff=f, ss=s: strat_returns_dual_src(
                d["Close"], d["Volume"], args.rsi_len, ff, ss, args.ma_type, "obv"))
            price_pooled = pooled_daily_df(dfs, lambda d, ff=f, ss=s: strat_returns_dual_src(
                d["Close"], d["Volume"], args.rsi_len, ff, ss, args.ma_type, "price"))
            om, pm = R.metrics(obv_pooled), R.metrics(price_pooled)
            flag = "  <-- OBV beats B&H" if om["sharpe"] > bhm["sharpe"] else ""
            print(f"  {f:>5}{s:>6}{om['total']:>+10.1%}{om['cagr']:>+9.2%}{om['sharpe']:>8.2f}"
                  f"{om['maxdd']:>+9.1%}{om['sharpe']-bhm['sharpe']:>+11.2f}"
                  f"{om['sharpe']-pm['sharpe']:>+13.2f}{flag}")

    print("\nCaveats: long-only cash, close-to-close, 15bps/turn. Basket is survivorship-biased")
    print("(absolutes flattered); trust the OBV-vs-PRICE and strategy-vs-B&H COMPARISONS, not")
    print("the absolute level. A source/pair that wins full-sample but loses most years is")
    print("over-fit -- prefer consistency (see the walk-forward tables above).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
