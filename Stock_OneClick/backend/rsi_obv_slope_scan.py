#!/usr/bin/env python3
"""Scanner for OBV-RSI / price SLOPE DIVERGENCE on the daily chart.

The setup, as specified: the RSI-MA8 is trending HIGHER while the price MA8 is trending
LOWER. Because the RSI here is sourced from OBV (not price -- see xunlong_panel.pine /
rsi_obv_panel.pine), a rising RSI-MA8 means cumulative signed volume flow is improving.
So the pattern is "money is flowing in while price is still sliding" -- an accumulation /
bullish-divergence read.

This is the SLOPE form of divergence, which is a different (looser, always-measurable)
construction than the PIVOT form in rsi_obv_divergence.py: that one needs two confirmed
pivots 5-60 bars apart and only fires a handful of times a year; this one fires whenever
the two smoothed lines simply point opposite ways over a lookback window.

  RSI source : OBV = cumsum(sign(diff(close)) * volume)      [matches the Pine exactly]
  RSI        : Wilder RSI(--rsi-len, default 14) on that OBV series
  fast line  : SMA(rsi, --rsi-ma, default 8)      -> must be RISING
  price line : SMA(close, --price-ma, default 8)  -> must be FALLING
  "trending" : net change over --lookback bars (default 5); --strict requires the line to
               move the same way on EVERY bar in the window.

Partial bars: the in-progress daily bar is DROPPED by default (an unclosed bar has a
truncated volume, which biases OBV and therefore the whole signal). --include-partial
opts back in.

    ../../vcp_env/bin/python rsi_obv_slope_scan.py                   # A-pool, today
    ../../vcp_env/bin/python rsi_obv_slope_scan.py --all             # full ~1243 universe
    ../../vcp_env/bin/python rsi_obv_slope_scan.py --all --strict --min-rsi-slope 3 --min-price-drop 0.02
    ../../vcp_env/bin/python rsi_obv_slope_scan.py --symbols AAPL MSFT NVDA --within 5

This detects the divergence only. rsi_obv_div_watch.py turns it into a two-stage system:
divergence = watchlist, then a price breakout through the MA8 = the actual alert.

EDGE WARNING: the pivot form of this divergence was tested in rsi_obv_divergence.py and
had NO edge (bullish divergence coin-flip year to year; bearish predicted the wrong way),
and every OBV/price RSI-MA crossover lost to buy-and-hold (rsi_obv_ma_compare.py). This
slope variant is UNBACKTESTED. Treat the output as a watchlist, not as entries.
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
from rsi_obv_ma_compare import obv_series, rsi_wilder_on  # noqa: E402

MIN_BARS = 60
OUT_DIR = BACKEND_DIR.parent / "tv_rsi_obv_div"

# Counts of names dropped by the quality filters, reported at the end so nothing is silent.
FILTER_STATS = {"stale": 0, "illiquid": 0}


# ----------------------------------------------------------------------------------
# shared plumbing (also used by rsi_obv_div_watch.py)
# ----------------------------------------------------------------------------------
def resolve_partial_cutoff(include_partial: bool, quiet: bool = False):
    """Returns the session date whose bar is still OPEN, or None if nothing to drop.

    An unclosed daily bar has truncated volume, which feeds straight into OBV and biases
    every OBV-derived value -- so by default we compute on the last CLOSED bar. Resolved
    from real market data (scan.resolve_session_state), not the wall clock."""
    if include_partial:
        return None
    st = scan.resolve_session_state(datetime.now())
    if not (st["is_partial"] and st["session_date"] is not None):
        return None
    if not quiet:
        print(f"⚠️ 盘中运行：{st['session_date']} 这根日线尚未收盘，成交量不完整会污染 OBV — 已丢弃该bar，"
              f"信号基于上一根已收盘日线。（--include-partial 可保留）")
    return st["session_date"]


def trim_partial(df: pd.DataFrame, cutoff) -> pd.DataFrame:
    """Drop the trailing unclosed bar, if this frame has one."""
    if cutoff is not None and len(df) and pd.Timestamp(df.index[-1]).date() >= cutoff:
        return df.iloc[:-1]
    return df


def reference_last_bar(cutoff, period: str = "6mo"):
    """Date of the market's last CLOSED bar, taken from SPY.

    Used to reject STALE symbols: a delisted or halted ticker keeps returning its final bars
    forever, and without this check its last bar looks like "today" and the name shows up as
    a fresh signal (e.g. SBNY at $0.20, last traded 2023). Returns None if SPY is unavailable,
    in which case staleness is simply not enforced."""
    try:
        spy = scan.download_daily("SPY", period=period)
        spy = trim_partial(spy, cutoff)
        return pd.Timestamp(spy.index[-1]).date() if len(spy) else None
    except Exception:
        return None


def is_stale(df: pd.DataFrame, ref_date, max_stale_days: int) -> bool:
    """True if this symbol's last bar lags the market's last closed bar by too much."""
    if ref_date is None or not len(df):
        return False
    stale = (ref_date - pd.Timestamp(df.index[-1]).date()).days > max_stale_days
    if stale:
        FILTER_STATS["stale"] += 1
    return stale


def is_illiquid(df: pd.DataFrame, args) -> bool:
    """True if the name is too cheap or too thin to be worth listing.

    Staleness alone doesn't catch everything: a delisted-but-still-quoted stub (e.g. SBNY at
    $0.20) keeps printing current bars, and its OBV-RSI swings wildly on trivial volume. A
    price floor plus an optional dollar-volume floor removes that class of false signal."""
    if not len(df):
        return True
    close = float(df["Close"].iloc[-1])
    if close < args.min_price:
        FILTER_STATS["illiquid"] += 1
        return True
    if args.min_dollar_vol > 0:
        dv = (df["Close"].astype(float) * df["Volume"].astype(float)).tail(20).median()
        if not np.isfinite(dv) or dv < args.min_dollar_vol:
            FILTER_STATS["illiquid"] += 1
            return True
    return False


def print_filter_stats() -> None:
    if FILTER_STATS["stale"] or FILTER_STATS["illiquid"]:
        print(f"  (excluded {FILTER_STATS['stale']} stale / {FILTER_STATS['illiquid']} "
              f"sub-price-or-volume names)")


def iter_symbol_frames(symbols: list[str], period: str, chunk: int = 100, verbose: bool = True):
    """Yield (symbol, DataFrame|None) for each symbol.

    Small lists go through scan.download_daily (which hits the in-run bar cache); large
    universes are batched into one yf.download per chunk, because ~1000 rapid single-ticker
    calls get rate-limited by Yahoo."""
    if len(symbols) <= 60:
        for sym in symbols:
            try:
                df = scan.download_daily(scan.to_yfinance_symbol(sym), period=period)
            except Exception:
                df = None
            yield sym, df
        return

    import yfinance as yf
    pairs = [(s, scan.to_yfinance_symbol(s)) for s in symbols]
    got = failed = 0
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
                d = None
            if d is None or d.empty:
                failed += 1
                yield orig, None
                continue
            got += 1
            yield orig, d
        if verbose:
            print(f"  ...{min(i + chunk, len(pairs))}/{len(pairs)} fetched, {got} with data")
    if verbose:
        print(f"got data for {got}/{len(pairs)} names ({failed} missing/delisted)")


def load_all_market() -> list[str]:
    """Full ~1243-name market universe from repo-root stock_symbols_1243.py."""
    root = BACKEND_DIR.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import stock_symbols_1243 as u  # noqa: E402
    return list(dict.fromkeys(list(u.STOCK_SYMBOLS) + list(u.ETF_SYMBOLS)))


def resolve_universe(args) -> list[str]:
    if getattr(args, "symbols", None):
        return list(args.symbols)
    if getattr(args, "all", False):
        return load_all_market()
    return list(scan.A_POOL_SYMBOLS)


# ----------------------------------------------------------------------------------
# the signal
# ----------------------------------------------------------------------------------
def _trending_up(s: pd.Series, lookback: int, strict: bool) -> pd.Series:
    """True where s is trending up over the lookback window. Default = net change over the
    window; --strict additionally demands every single bar in the window moved up."""
    net_up = s > s.shift(lookback)
    if not strict:
        return net_up.fillna(False)
    every_bar_up = (s.diff() > 0).rolling(lookback).sum() == lookback
    return (net_up & every_bar_up).fillna(False)


def slope_divergence(df: pd.DataFrame, rsi_len: int, rsi_ma: int, price_ma: int,
                     lookback: int, strict: bool) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (signal, rsi_ma_line, price_ma_line).

    signal = RSI(OBV)-MA rising AND price-MA falling, both over `lookback` bars.
    price_ma=1 degenerates to raw close, if you want price itself rather than its MA."""
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)
    rsi = rsi_wilder_on(obv_series(close, volume), rsi_len)
    rsi_line = rsi.rolling(rsi_ma).mean() if rsi_ma > 1 else rsi
    px_line = close.rolling(price_ma).mean() if price_ma > 1 else close
    sig = _trending_up(rsi_line, lookback, strict) & _trending_up(-px_line, lookback, strict)
    return sig.fillna(False), rsi_line, px_line


def _streak(sig: pd.Series, end_pos: int) -> int:
    """How many consecutive bars the condition has held, ending at end_pos."""
    n = 0
    v = sig.values
    while end_pos - n >= 0 and v[end_pos - n]:
        n += 1
    return n


def _hit_from_df(sym: str, df: pd.DataFrame, args, cutoff, ref_date=None) -> dict | None:
    if df is None or len(df) < MIN_BARS:
        return None
    df = trim_partial(df.dropna(subset=["Close", "Volume"]), cutoff)
    if len(df) < MIN_BARS or is_stale(df, ref_date, args.max_stale) or is_illiquid(df, args):
        return None

    sig, rsi_line, px_line = slope_divergence(df, args.rsi_len, args.rsi_ma, args.price_ma,
                                              args.lookback, args.strict)
    if not sig.iloc[-args.within:].any():
        return None
    pos = len(sig) - 1 - sig.values[::-1].argmax()          # latest firing bar

    L = args.lookback
    rsi_chg = float(rsi_line.iloc[pos] - rsi_line.iloc[pos - L])            # RSI points
    px_chg = float(px_line.iloc[pos] / px_line.iloc[pos - L] - 1)           # fraction, negative
    if rsi_chg < args.min_rsi_slope or -px_chg < args.min_price_drop:
        return None

    c = df["Close"].astype(float)
    ma200 = c.rolling(200).mean()
    above200 = bool(c.iloc[pos] > ma200.iloc[pos]) if np.isfinite(ma200.iloc[pos]) else None
    return {
        "symbol": sym,
        "bars_ago": len(sig) - 1 - pos,
        "fire_date": pd.Timestamp(df.index[pos]).date().isoformat(),
        "close": round(float(c.iloc[pos]), 2),
        "rsi_ma": round(float(rsi_line.iloc[pos]), 1),
        "rsi_chg": round(rsi_chg, 1),
        "px_chg": round(px_chg, 4),
        "streak": _streak(sig, pos),
        "trend": "—" if above200 is None else ("above200" if above200 else "below200"),
    }


def _order(hits: list) -> pd.DataFrame:
    """Rank by how emphatic the divergence is: rank-sum of RSI rise and price drop, so the
    two incommensurable units (RSI points vs %) contribute equally."""
    if not hits:
        return pd.DataFrame()
    df = pd.DataFrame(hits)
    score = df["rsi_chg"].rank(pct=True) + (-df["px_chg"]).rank(pct=True)
    df["score"] = (score / 2 * 100).round(0).astype(int)
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def scan_universe(symbols: list[str], args, cutoff, ref_date=None) -> pd.DataFrame:
    hits = []
    for sym, df in iter_symbol_frames(symbols, args.period, verbose=len(symbols) > 60):
        h = _hit_from_df(sym, df, args, cutoff, ref_date)
        if h:
            hits.append(h)
    return _order(hits)


def write_tv_list(symbols, stem: str, out_dir: Path = OUT_DIR) -> Path:
    """Write a TradingView paste list as <stem>_<date>.txt plus a _latest.txt copy."""
    try:
        lines = [scan.build_tv_symbol(str(s).upper(), "") for s in symbols]
    except Exception:
        lines = [str(s).upper() for s in symbols]
    out_dir.mkdir(parents=True, exist_ok=True)
    body = ("\n".join(lines) + "\n") if lines else ""
    out = out_dir / f"{stem}_{datetime.now().strftime('%Y-%m-%d')}.txt"
    out.write_text(body, encoding="utf-8")
    (out_dir / f"{stem}_latest.txt").write_text(body, encoding="utf-8")
    return out


def add_signal_args(ap: argparse.ArgumentParser) -> None:
    """Signal-shape flags shared with rsi_obv_div_watch.py."""
    ap.add_argument("--rsi-len", type=int, default=14, help="Wilder RSI length on OBV (Pine default 14)")
    ap.add_argument("--rsi-ma", type=int, default=8, help="SMA length applied to the RSI (Pine ma8)")
    ap.add_argument("--price-ma", type=int, default=8, help="SMA length applied to close; 1 = raw price")
    ap.add_argument("--lookback", type=int, default=5, help="bars over which 'trending' is measured")
    ap.add_argument("--strict", action="store_true", help="require every bar in the window to move the same way")
    ap.add_argument("--min-rsi-slope", type=float, default=0.0, help="min RSI-MA rise over the window, in RSI points")
    ap.add_argument("--min-price-drop", type=float, default=0.0, help="min price-MA fall over the window, as a fraction (0.02 = 2%%)")
    ap.add_argument("--include-partial", action="store_true", help="keep the unclosed intraday bar (biases OBV)")
    ap.add_argument("--max-stale", type=int, default=5,
                    help="reject symbols whose last bar lags SPY's by more than N days (delisted/halted)")
    ap.add_argument("--min-price", type=float, default=1.0,
                    help="reject names closing under this price (default 1.0; delisted stubs)")
    ap.add_argument("--min-dollar-vol", type=float, default=0.0,
                    help="reject names whose 20d median dollar volume is under this (0 = off)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="*", help="explicit symbols (default: scan A-pool)")
    ap.add_argument("--all", action="store_true", help="scan the full ~1243-name market universe")
    ap.add_argument("--within", type=int, default=1, help="fired within N latest bars (1=latest only)")
    ap.add_argument("--period", default="2y", help="history to download")
    add_signal_args(ap)
    ap.add_argument("--below-200ma-only", action="store_true", help="keep only names under their 200MA (deeper washouts)")
    ap.add_argument("--no-write", action="store_true", help="don't write the TradingView list")
    args = ap.parse_args()

    cutoff = resolve_partial_cutoff(args.include_partial)
    ref_date = reference_last_bar(cutoff)
    symbols = resolve_universe(args)

    mode = "strict (every bar)" if args.strict else f"net over {args.lookback} bars"
    print(f"Scanning {len(symbols)} names: RSI({args.rsi_len},OBV)-MA{args.rsi_ma} UP "
          f"while price-MA{args.price_ma} DOWN [{mode}], within {args.within} bar(s)...")

    hits = scan_universe(symbols, args, cutoff, ref_date)
    print_filter_stats()
    if not hits.empty and args.below_200ma_only:
        hits = hits[hits["trend"] == "below200"].reset_index(drop=True)
    if hits.empty:
        print("No slope-divergence signals.")
        return 0

    print(f"\n{len(hits)} hit(s)  (RSI flow rising while price falling):")
    print(f"  {'symbol':<10}{'score':>7}{'RSI-MA':>9}{'ΔRSI':>8}{'Δprice':>9}{'streak':>8}{'trend':>11}{'bars':>6}{'close':>10}")
    for _, r in hits.iterrows():
        print(f"  {r['symbol']:<10}{r['score']:>7}{r['rsi_ma']:>9.1f}{r['rsi_chg']:>+8.1f}"
              f"{r['px_chg']:>+9.1%}{r['streak']:>8}{r['trend']:>11}{r['bars_ago']:>6}{r['close']:>10.2f}")

    if not args.no_write:
        out = write_tv_list(hits["symbol"], "tv_rsi_obv_div")
        print(f"\nTradingView list -> {out}")

    print("\nUNBACKTESTED signal. The pivot form of OBV-RSI divergence showed no forward edge")
    print("(rsi_obv_divergence.py) — use this as a watchlist / context, not as an entry trigger.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
