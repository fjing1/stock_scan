#!/usr/bin/env python3
"""Scanner for the 'break3avg' (upbreak3avg) signal from the trend-alert Pine script.

break3avg fires on a bar when price breaks up through both the fast EMA(ohlc4,8)
and the SMA(ohlc4,21) at once, with the 8-EMA already rising and the 3-EMA above
the 21-SMA — i.e. a clean momentum breakout with the short-term structure already
turning up. In trend_alert_backtest.py it was the best of the bundle: highest Sharpe
(index 0.49 / basket 0.83) and by far the smallest drawdown (~-19% vs B&H -52/-69%),
in market only ~25% of the time, with a small but consistent forward edge
(D5 win 54.8% vs 52.4% baseline; D20 58.7% vs 56.2%).

This scans a universe and lists names whose break3avg fired on the latest bar (or
within --within trading days, for catch-up), and writes a TradingView paste list.

    ../../vcp_env/bin/python break3avg_scan.py                 # scan A-pool
    ../../vcp_env/bin/python break3avg_scan.py --within 3      # fired in last 3 bars
    ../../vcp_env/bin/python break3avg_scan.py --symbols AAPL MSFT NVDA
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
import scan_stocks as scan  # noqa: E402
from rsi_ma_sweep import ma  # noqa: E402
from trend_alert_backtest import crossover  # noqa: E402


def break3avg_signal(df: pd.DataFrame, fast: int, slow: int,
                     fast_type: str = "SMA", slow_type: str = "SMA") -> pd.Series:
    """Parametrized break3avg (upbreak3avg): close breaks up through the fast line
    (rising) AND the slow line at once, with the short(3) line already above the slow
    line. Tuned default = SMA/SMA 8/22 (best per-signal forward edge; see
    break3avg_matype.py / break3avg_combined.py)."""
    o, h, l, c = (df[k].astype(float) for k in ("Open", "High", "Low", "Close"))
    ohlc4 = (o + h + l + c) / 4
    line_f = ma(ohlc4, fast, fast_type)
    line_s = ma(ohlc4, slow, slow_type)
    short3 = ma(ohlc4, 3, fast_type)
    return (crossover(c, line_f) & (line_f > line_f.shift(1))
            & crossover(c, line_s) & (short3 > line_s)).fillna(False)


def load_all_market() -> list[str]:
    """Full ~1243-name market universe from repo-root stock_symbols_1243.py."""
    root = BACKEND_DIR.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import stock_symbols_1243 as u  # noqa: E402
    return list(dict.fromkeys(list(u.STOCK_SYMBOLS) + list(u.ETF_SYMBOLS)))


def _rs_tier(rs_excess: float) -> str:
    """12m relative-strength conviction tier (break3avg_rs.py: strong-RS wins more
    often; weak-RS breakouts are <50% D5 = low-conviction dead-cat bounces).
    Thresholds approximate the broad-universe RS terciles."""
    if rs_excess >= 0.15:
        return "strong"
    if rs_excess <= -0.10:
        return "weak"
    return "mid"


def _hit_from_df(sym: str, df: pd.DataFrame, within: int, fast: int, slow: int,
                 fast_type: str, slow_type: str, spy_12m: float) -> dict | None:
    if df is None or len(df) < 60:
        return None
    e = break3avg_signal(df, fast, slow, fast_type, slow_type)
    if not e.iloc[-within:].any():
        return None
    last_fire_pos = len(e) - 1 - e.values[::-1].argmax()   # 0 = today's close
    c = df["Close"].astype(float)
    lb = 252
    stock_12m = (c.iloc[-1] / c.iloc[-1 - lb] - 1) if len(c) > lb else (c.iloc[-1] / c.iloc[0] - 1)
    rs = stock_12m - spy_12m if np.isfinite(spy_12m) else np.nan
    return {
        "symbol": sym,
        "bars_ago": len(e) - 1 - last_fire_pos,
        "fire_date": df.index[last_fire_pos].date().isoformat(),
        "close": round(float(c.iloc[-1]), 2),
        "rs_12m": round(float(rs), 4) if np.isfinite(rs) else np.nan,
        "tier": _rs_tier(rs) if np.isfinite(rs) else "n/a",
    }


def scan_universe(symbols: list[str], within: int, period: str, fast: int, slow: int,
                  fast_type: str, slow_type: str, spy_12m: float) -> pd.DataFrame:
    """Single-ticker path (fine for small lists like the A-pool)."""
    hits = []
    for sym in symbols:
        try:
            df = scan.download_daily(scan.to_yfinance_symbol(sym), period=period)
        except Exception:
            df = None
        h = _hit_from_df(sym, df, within, fast, slow, fast_type, slow_type, spy_12m)
        if h:
            hits.append(h)
    return _order(hits)


def scan_universe_batched(symbols: list[str], within: int, period: str, fast: int, slow: int,
                          fast_type: str, slow_type: str, spy_12m: float, chunk: int = 100) -> pd.DataFrame:
    """Batched path for large universes: yf.download many tickers per request to
    avoid the rate-limiting that hits ~1000 rapid single-ticker calls."""
    import yfinance as yf
    pairs = [(s, scan.to_yfinance_symbol(s)) for s in symbols]
    hits, got, failed = [], 0, 0
    for i in range(0, len(pairs), chunk):
        batch = pairs[i:i + chunk]
        try:
            data = yf.download([y for _, y in batch], period=period, group_by="ticker",
                               auto_adjust=True, threads=True, progress=False)
        except Exception:
            failed += len(batch)
            continue
        for orig, y in batch:
            try:
                d = data[y].dropna()
            except Exception:
                failed += 1
                continue
            if d.empty:
                failed += 1
                continue
            got += 1
            h = _hit_from_df(orig, d, within, fast, slow, fast_type, slow_type, spy_12m)
            if h:
                hits.append(h)
        print(f"  ...{min(i + chunk, len(pairs))}/{len(pairs)} scanned, {got} with data, {len(hits)} hits")
    print(f"got data for {got}/{len(pairs)} names ({failed} missing/delisted)")
    return _order(hits)


def _order(hits: list) -> pd.DataFrame:
    if not hits:
        return pd.DataFrame()
    # strongest RS first (highest-conviction breakouts on top), then most recent
    tier_rank = {"strong": 0, "mid": 1, "weak": 2, "n/a": 3}
    df = pd.DataFrame(hits)
    df["_t"] = df["tier"].map(tier_rank)
    return df.sort_values(["_t", "rs_12m"], ascending=[True, False]).drop(columns="_t")


def attach_earnings(hits: pd.DataFrame, near_days: int = 5) -> pd.DataFrame:
    """Add an earnings-event column for each hit (fetched only for hit symbols, so cheap).
    Flags whether the breakout coincides with a recent earnings report (=> the move is
    event-driven / PEAD, a different mechanism than the technical break3avg edge) or if
    earnings are imminent (pre-earnings risk). Shows the last surprise %."""
    import yfinance as yf
    tags, drivers = [], []
    for _, r in hits.iterrows():
        fire = pd.Timestamp(r["fire_date"]).date()
        tag, driver = "—", ""
        try:
            ed = yf.Ticker(scan.to_yfinance_symbol(r["symbol"])).get_earnings_dates(limit=16)
            if ed is not None and not ed.empty:
                dates = pd.Series(ed.index).dt.date
                surp = ed["Surprise(%)"].values
                past = [(d, s) for d, s in zip(dates, surp) if d <= fire]
                fut = [d for d in dates if d > fire]
                if past:
                    ld, ls = max(past, key=lambda x: x[0])
                    days_ago = (fire - ld).days
                    if days_ago <= near_days:
                        sp = f"{ls:+.0f}%" if pd.notna(ls) else "n/a"
                        tag = f"{days_ago}d ago {sp}"
                        driver = "EARN"          # breakout is earnings-driven
                if not driver and fut:
                    dn = (min(fut) - fire).days
                    if dn <= near_days:
                        tag = f"in {dn}d"
                        driver = "pre-ER"        # earnings imminent = risk
        except Exception:
            pass
        tags.append(tag)
        drivers.append(driver)
    hits = hits.copy()
    hits["earnings"] = tags
    hits["driver"] = drivers
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="*", help="explicit symbols (default: scan A-pool)")
    ap.add_argument("--all", action="store_true", help="scan the full ~1243-name market universe")
    ap.add_argument("--within", type=int, default=1, help="fired within N latest bars (1=today only)")
    ap.add_argument("--period", default="2y", help="history to download (>=2y so 12m RS is computable)")
    ap.add_argument("--fast", type=int, default=8, help="fast line length (tuned default 8)")
    ap.add_argument("--slow", type=int, default=22, help="slow line length (tuned default 22)")
    ap.add_argument("--fast-type", default="SMA", choices=["SMA", "EMA"], help="fast line MA type (default SMA)")
    ap.add_argument("--slow-type", default="SMA", choices=["SMA", "EMA"], help="slow line MA type (default SMA)")
    ap.add_argument("--strong-only", action="store_true", help="TV list keeps only strong-RS (hi-conv) hits")
    ap.add_argument("--no-earnings", action="store_true", help="skip the earnings-event column (faster)")
    ap.add_argument("--no-write", action="store_true", help="don't write the TV list file")
    args = ap.parse_args()

    # SPY 12m return (trailing 252d as of latest close) — baseline for relative strength
    try:
        spy = scan.download_daily("SPY", period=args.period)["Close"].astype(float)
        spy_12m = float(spy.iloc[-1] / spy.iloc[-1 - 252] - 1) if len(spy) > 252 else float(spy.iloc[-1] / spy.iloc[0] - 1)
    except Exception:
        spy_12m = np.nan
    print(f"SPY trailing 12m return (RS baseline): {spy_12m:+.1%}")

    if args.symbols:
        symbols = args.symbols
    elif args.all:
        symbols = load_all_market()
    else:
        symbols = list(scan.A_POOL_SYMBOLS)
    print(f"Scanning {len(symbols)} names for break {args.fast_type}{args.fast}/{args.slow_type}{args.slow} "
          f"(within {args.within} bar[s])...")
    if len(symbols) > 60:
        hits = scan_universe_batched(symbols, args.within, args.period, args.fast, args.slow,
                                     args.fast_type, args.slow_type, spy_12m)
    else:
        hits = scan_universe(symbols, args.within, args.period, args.fast, args.slow,
                             args.fast_type, args.slow_type, spy_12m)

    if hits.empty:
        print("No break3avg signals.")
        return 0

    if not args.no_earnings:
        hits = attach_earnings(hits)

    n_tier = hits["tier"].value_counts().to_dict()
    print(f"\n{len(hits)} hit(s)  [strong {n_tier.get('strong',0)} / mid {n_tier.get('mid',0)} / weak {n_tier.get('weak',0)}]:")
    has_earn = "earnings" in hits.columns
    hdr = f"  {'symbol':<10}{'tier':>8}{'RS 12m':>9}{'driver':>8}{'earnings':>14}{'bars':>6}{'close':>10}"
    print(hdr if has_earn else f"  {'symbol':<10}{'tier':>8}{'RS 12m':>9}{'bars':>6}{'close':>10}")
    for _, r in hits.iterrows():
        rs = f"{r['rs_12m']:+.0%}" if pd.notna(r['rs_12m']) else "n/a"
        if has_earn:
            print(f"  {r['symbol']:<10}{r['tier']:>8}{rs:>9}{r['driver']:>8}{r['earnings']:>14}{r['bars_ago']:>6}{r['close']:>10.2f}")
        else:
            print(f"  {r['symbol']:<10}{r['tier']:>8}{rs:>9}{r['bars_ago']:>6}{r['close']:>10.2f}")

    if not args.no_write:
        tv_hits = hits[hits["tier"] == "strong"] if args.strong_only else hits
        try:
            ex_map = {s: "" for s in tv_hits["symbol"]}
            lines = [scan.build_tv_symbol(str(s).upper(), ex_map.get(s, "")) for s in tv_hits["symbol"]]
        except Exception:
            lines = [str(s).upper() for s in tv_hits["symbol"]]
        date_str = datetime.now().strftime("%Y-%m-%d")
        out_dir = BACKEND_DIR.parent / "tv_break3avg"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"tv_break3avg_{date_str}.txt"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        (out_dir / "tv_break3avg_latest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nTradingView list ({'strong-RS only' if args.strong_only else 'all tiers'}) -> {out}")

    print("\nRS tiers (break3avg_rs.py, broad universe): strong-RS breakouts win ~2-4pp more often;")
    print("weak-RS (below-market, e.g. counter-trend bounces) win <50% at D5 = low conviction.")
    print("break3avg doesn't beat buy-hold alone — a managed entry; prefer strong/mid tier.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
