"""Probe: which intraday-bar endpoints respond WITHOUT credentials?

Records status code, byte count, and a content snippet for each. No assertions --
just what the server actually said.
"""
import json
import time

import requests

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}

TESTS = [
    # --- Alpaca: is ANY part reachable with no credentials? ---
    ("alpaca_bars_nokey",
     "https://data.alpaca.markets/v2/stocks/bars"
     "?symbols=SPY&timeframe=5Min&start=2016-01-04T00:00:00Z&end=2016-01-08T00:00:00Z&limit=10"),
    ("alpaca_bars_single_nokey",
     "https://data.alpaca.markets/v2/stocks/SPY/bars?timeframe=5Min&limit=10"),

    # --- Polygon ---
    ("polygon_agg_nokey",
     "https://api.polygon.io/v2/aggs/ticker/SPY/range/5/minute/2016-01-04/2016-01-08"),
    ("polygon_agg_blankkey",
     "https://api.polygon.io/v2/aggs/ticker/SPY/range/5/minute/2016-01-04/2016-01-08?apiKey="),

    # --- Twelve Data: documented 'demo' apikey ---
    ("twelvedata_demo_5min",
     "https://api.twelvedata.com/time_series?symbol=AAPL&interval=5min&apikey=demo&outputsize=5000"),
    ("twelvedata_nokey",
     "https://api.twelvedata.com/time_series?symbol=AAPL&interval=5min&outputsize=10"),

    # --- Finnhub ---
    ("finnhub_candle_nokey",
     "https://finnhub.io/api/v1/stock/candle?symbol=SPY&resolution=5&from=1452000000&to=1452600000"),

    # --- Tiingo ---
    ("tiingo_iex_nokey",
     "https://api.tiingo.com/iex/SPY/prices?resampleFreq=5min&startDate=2020-01-02"),

    # --- Databento ---
    ("databento_nokey",
     "https://hist.databento.com/v0/metadata.list_datasets"),

    # --- Stooq: does any no-key mirror serve intraday? ---
    ("stooq_daily_csv", "https://stooq.com/q/d/l/?s=spy.us&i=d"),
    ("stooq_5min_i5",   "https://stooq.com/q/d/l/?s=spy.us&i=5"),
    ("stooq_60min_i60", "https://stooq.com/q/d/l/?s=spy.us&i=60"),
    ("stooq_intraday_a2", "https://stooq.com/q/a2/d/?s=spy.us&i=5"),

    # --- Financial Modeling Prep: documented demo key ---
    ("fmp_demo_5min",
     "https://financialmodelingprep.com/api/v3/historical-chart/5min/AAPL?apikey=demo"),

    # --- Nasdaq Data Link ---
    ("nasdaqdatalink_nokey",
     "https://data.nasdaq.com/api/v3/datasets/EOD/SPY.json?rows=3"),

    # --- Dukascopy: free no-key tick archive (binary .bi5) ---
    ("dukascopy_spx_tick",
     "https://datafeed.dukascopy.com/datafeed/USA500.IDXUSD/2018/00/04/14h_ticks.bi5"),
    ("dukascopy_aapl_tick",
     "https://datafeed.dukascopy.com/datafeed/AAPLUSUSD/2018/00/04/14h_ticks.bi5"),

    # --- Binance (crypto sanity check: known-good free intraday) ---
    ("binance_klines_5m",
     "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=5m&limit=5"),
]


def probe(name, url):
    t0 = time.time()
    try:
        r = requests.get(url, headers=UA, timeout=30)
    except Exception as exc:  # noqa: BLE001
        print(f"\n### {name}\n  EXC {type(exc).__name__}: {str(exc)[:200]}")
        return
    el = time.time() - t0
    body = r.content
    print(f"\n### {name}")
    print(f"  GET {url[:110]}")
    print(f"  status={r.status_code}  bytes={len(body)}  {el:.2f}s  ct={r.headers.get('content-type','?')[:40]}")
    snip = body[:400]
    try:
        snip = snip.decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        snip = repr(snip)
    print(f"  body[:400]={snip!r}")


for name, url in TESTS:
    probe(name, url)
    time.sleep(0.6)
