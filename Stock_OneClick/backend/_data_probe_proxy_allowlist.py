"""Probe: map which finance-data hosts the corporate egress proxy actually permits.

Signature:
  ProxyError / TunnelError  -> host BLOCKED at the proxy (cannot verify from here)
  any HTTP status code      -> host ALLOWED through (verifiable)

This distinction matters: a blocked host tells us nothing about the vendor's free
tier, only that this machine cannot reach it.
"""
import time

import requests

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}

HOSTS = [
    # intraday-bar vendors (category A targets)
    ("alpaca-data",     "https://data.alpaca.markets/v2/stocks/bars?symbols=SPY&timeframe=5Min&limit=1"),
    ("alpaca-api",      "https://api.alpaca.markets/v2/clock"),
    ("alpaca-paper",    "https://paper-api.alpaca.markets/v2/clock"),
    ("polygon",         "https://api.polygon.io/v2/aggs/ticker/SPY/prev"),
    ("twelvedata",      "https://api.twelvedata.com/time_series?symbol=AAPL&interval=5min&apikey=demo"),
    ("finnhub",         "https://finnhub.io/api/v1/stock/candle?symbol=SPY&resolution=5&from=1452000000&to=1452600000"),
    ("tiingo",          "https://api.tiingo.com/iex/SPY/prices?resampleFreq=5min"),
    ("databento-hist",  "https://hist.databento.com/v0/metadata.list_datasets"),
    ("databento-www",   "https://databento.com/"),
    ("fmp",             "https://financialmodelingprep.com/api/v3/historical-chart/5min/AAPL?apikey=demo"),
    ("marketstack",     "https://api.marketstack.com/v1/intraday?symbols=AAPL"),
    ("eodhd",           "https://eodhd.com/api/intraday/AAPL.US?api_token=demo&interval=5m&fmt=json"),
    ("nasdaqdatalink",  "https://data.nasdaq.com/api/v3/datasets/EOD/SPY.json?rows=1"),
    ("dukascopy",       "https://datafeed.dukascopy.com/datafeed/USA500.IDXUSD/2018/00/04/14h_ticks.bi5"),
    ("iexcloud",        "https://cloud.iexapis.com/stable/stock/spy/quote"),
    ("alpha-vantage",   "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=SPY&apikey=demo"),
    ("stooq",           "https://stooq.com/q/d/l/?s=spy.us&i=d"),
    ("stooq-pl",        "https://stooq.pl/q/d/l/?s=spy.us&i=d"),
    ("yahoo-q1",        "https://query1.finance.yahoo.com/v8/finance/chart/SPY?range=5d&interval=5m"),
    ("yahoo-q2",        "https://query2.finance.yahoo.com/v8/finance/chart/SPY?range=5d&interval=5m"),
    # crypto / other free intraday, as controls
    ("binance",         "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=5m&limit=1"),
    ("binance-vision",  "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/5m/BTCUSDT-5m-2020-01.zip"),
    ("kraken",          "https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=5"),
    ("coinbase",        "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=300"),
    ("bitfinex",        "https://api-pub.bitfinex.com/v2/candles/trade:5m:tBTCUSD/hist?limit=1"),
    # public-sector / academic
    ("sec-gov",         "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=AAPL&type=10-K"),
    ("cboe",            "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"),
    ("nyse",            "https://www.nyse.com/api/quotes/filter"),
    ("firstratedata",   "https://firstratedata.com/"),
    ("kibot",           "http://api.kibot.com/?action=login&user=guest&password=guest"),
    ("histdata",        "https://www.histdata.com/download-free-forex-data/"),
]


def classify(name, url):
    t0 = time.time()
    try:
        r = requests.get(url, headers=UA, timeout=25)
    except requests.exceptions.ProxyError as exc:
        return name, "BLOCKED-PROXY", "", 0, str(exc)[:80], time.time() - t0
    except Exception as exc:  # noqa: BLE001
        return name, f"ERR-{type(exc).__name__}", "", 0, str(exc)[:80], time.time() - t0
    body = r.content
    snip = body[:160].decode("utf-8", "replace").replace("\n", " ")
    return name, "ALLOWED", r.status_code, len(body), snip, time.time() - t0


print(f"{'host':18s} {'verdict':16s} {'code':>5s} {'bytes':>8s}  snippet")
print("-" * 120)
results = []
for name, url in HOSTS:
    res = classify(name, url)
    results.append(res)
    n, verdict, code, nb, snip, el = res
    print(f"{n:18s} {verdict:16s} {str(code):>5s} {nb:>8d}  {snip[:70]}")
    time.sleep(0.35)

allowed = [r for r in results if r[1] == "ALLOWED"]
print(f"\n=== {len(allowed)}/{len(HOSTS)} hosts reachable through the proxy ===")
for r in allowed:
    print(f"  ALLOWED  {r[0]:18s} HTTP {r[2]}")
print("\n=== BLOCKED at proxy (cannot be verified from this machine) ===")
for r in results:
    if r[1] != "ALLOWED":
        print(f"  {r[1]:18s} {r[0]}")
