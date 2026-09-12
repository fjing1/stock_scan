"""
_btc_data_4h.py — BTC 4-hour bar loader/cache for the daily+4H MTF trend research
(task: "deep research on trending and trading based on daily and 4hr chart").

Source: Binance.US public klines API (free, keyless, no auth). Chosen over yfinance because
yfinance's intraday history is capped at 730 days (confirmed empirically 2026-08-25) — barely
covers the 2024-25 run and misses the 2021 boom / 2022 bust entirely. Binance.US has BTCUSD 4h
bars back to 2019-09-17 (~6.4yr), long enough to span the 2020 COVID crash, the 2021-22 cycle, and
2024-25 — still far short of the 11.9yr daily series in _btc_data.py, so treat sample-size/
independence caveats seriously in anything built on this (see _btc_mtf_research.py).

Binance.com (global) is geo-blocked from this environment (HTTP 451); Binance.US serves USD pairs
and is not geo-restricted here. Early bars (Sept 2019, right after launch) show flat/thin OHLC —
low liquidity, not a data bug — worth excluding from anything sensitive to microstructure.

Run: ../../vcp_env/bin/python _btc_data_4h.py --refresh
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import pandas as pd

CACHE = Path(__file__).resolve().parent / "_btc_panel_4h.pkl"
KLINES_URL = "https://api.binance.us/api/v3/klines"
SYMBOL = "BTCUSD"
INTERVAL = "4h"
INTERVAL_MS = 4 * 60 * 60 * 1000


def _fetch_klines(start_ms: int, end_ms: int) -> list:
    import json
    import urllib.parse
    import urllib.request

    params = {"symbol": SYMBOL, "interval": INTERVAL, "startTime": start_ms,
              "endTime": end_ms, "limit": 1000}
    url = f"{KLINES_URL}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=20) as resp:
        return json.load(resp)


def refresh_cache() -> pd.DataFrame:
    start_ms = 0
    now_ms = int(time.time() * 1000)
    rows = []
    cursor = start_ms
    while cursor < now_ms:
        batch = _fetch_klines(cursor, now_ms)
        if not batch:
            break
        rows.extend(batch)
        last_open = batch[-1][0]
        if last_open <= cursor:
            break
        cursor = last_open + INTERVAL_MS
        if len(batch) < 1000:
            break
        time.sleep(0.15)   # be polite to the free public endpoint

    df = pd.DataFrame(rows, columns=[
        "open_time", "Open", "High", "Low", "Close", "Volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.drop_duplicates("open_time").set_index("open_time").sort_index()
    for c in ("Open", "High", "Low", "Close", "Volume"):
        df[c] = df[c].astype(float)
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    with open(CACHE, "wb") as f:
        pickle.dump(df, f)
    print(f"  Binance.US BTCUSD 4h  {df.index[0]} -> {df.index[-1]}  {len(df)} bars")
    print(f"cached -> {CACHE}")
    return df


def load_4h() -> pd.DataFrame:
    if not CACHE.exists():
        return refresh_cache()
    with open(CACHE, "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    df = refresh_cache() if args.refresh else load_4h()
    print(f"4h bars: {df.index[0]} -> {df.index[-1]}  ({len(df)} bars, "
          f"{len(df) * 4 / 24 / 365.25:.1f} yrs)")


if __name__ == "__main__":
    main()
