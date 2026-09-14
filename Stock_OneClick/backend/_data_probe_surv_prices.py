"""PROBE: do any FREE daily-OHLCV sources return bars for KNOWN-DELISTED US tickers?

Test set = the exact names that returned zero bars in this repo's 284-symbol panel,
plus older delistings to test history depth.
"""
import io
import json
import os
import time
import zipfile

import pandas as pd
import requests

OUT = {}
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 " \
     "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# repo's own zero-bar casualties + deeper history tests
RECENT_DELISTED = ["SIVB", "ATVI", "VIAC", "SPLK", "CTXS", "ABMD", "CLVS"]
OLD_DELISTED = ["ENRNQ", "LEHMQ", "WCOM", "BSC", "GM", "DELL", "TWTR", "FRC", "SBNY"]
CONTROL = ["AAPL"]


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ---------------------------------------------------------------- 1. STOOQ
hr("1. STOOQ per-symbol CSV (free, NO key): https://stooq.com/q/d/l/?s=<tick>.us&i=d")
S = requests.Session()
S.headers.update({"User-Agent": UA})
for tick in CONTROL + RECENT_DELISTED + OLD_DELISTED:
    u = f"https://stooq.com/q/d/l/?s={tick.lower()}.us&i=d"
    try:
        t0 = time.time()
        r = S.get(u, timeout=45)
        dt = time.time() - t0
        body = r.text.strip()
        if r.status_code == 200 and body.lower().startswith("date"):
            df = pd.read_csv(io.StringIO(body))
            print(f"  {tick:6s} HTTP{r.status_code} rows={len(df):<6} "
                  f"{df['Date'].iloc[0]}..{df['Date'].iloc[-1]}  "
                  f"lastClose={df['Close'].iloc[-1]}  cols={list(df.columns)}  {dt:.2f}s")
            OUT[f"stooq:{tick}"] = (len(df), df["Date"].iloc[0], df["Date"].iloc[-1])
        else:
            print(f"  {tick:6s} HTTP{r.status_code} body={body[:60]!r}  {dt:.2f}s")
            OUT[f"stooq:{tick}"] = (0, body[:40], None)
    except Exception as e:
        print(f"  {tick:6s} EXC {type(e).__name__}: {e}")
    time.sleep(0.4)

hr("1b. STOOQ BULK archive (d_us_txt.zip)")
for u in ["https://static.stooq.com/db/h/d_us_txt.zip",
          "https://stooq.com/db/h/?b=d_us_txt",
          "https://stooq.pl/db/h/d_us_txt.zip"]:
    try:
        r = S.get(u, timeout=90, stream=True)
        print(f"  {u}\n    -> HTTP {r.status_code} clen={r.headers.get('Content-Length')} "
              f"ctype={r.headers.get('Content-Type')}")
        if r.status_code == 200 and "zip" in (r.headers.get("Content-Type") or ""):
            c = r.content
            print(f"    bytes={len(c):,}")
            z = zipfile.ZipFile(io.BytesIO(c))
            names = z.namelist()
            print(f"    members={len(names):,} sample={names[:5]}")
        else:
            txt = r.text[:300] if r.status_code != 200 else ""
            print(f"    body={txt!r}")
        r.close()
    except Exception as e:
        print(f"    EXC {type(e).__name__}: {e}")
    time.sleep(0.5)

# ------------------------------------------------------- 2. EODHD demo token
hr("2. EODHD free/demo token (api_token=demo)")
for tick in ["AAPL", "MSFT", "TSLA", "SIVB", "ATVI", "VIAC"]:
    u = f"https://eodhd.com/api/eod/{tick}.US?api_token=demo&fmt=json"
    try:
        r = S.get(u, timeout=90)
        if r.status_code == 200:
            try:
                j = r.json()
                print(f"  {tick:6s} HTTP200 rows={len(j):<6} "
                      f"{j[0]['date'] if j else '-'}..{j[-1]['date'] if j else '-'} "
                      f"lastClose={j[-1]['close'] if j else '-'} keys={list(j[0].keys()) if j else []}")
            except Exception:
                print(f"  {tick:6s} HTTP200 non-json {r.text[:120]!r}")
        else:
            print(f"  {tick:6s} HTTP{r.status_code} {r.text[:150]!r}")
    except Exception as e:
        print(f"  {tick:6s} EXC {type(e).__name__}: {e}")
    time.sleep(0.4)

hr("2b. EODHD exchange-symbol-list incl. DELISTED (demo token)")
for u in ["https://eodhd.com/api/exchange-symbol-list/US?api_token=demo&fmt=json",
          "https://eodhd.com/api/exchange-symbol-list/US?api_token=demo&fmt=json&delisted=1"]:
    try:
        r = S.get(u, timeout=120)
        print(f"  {u[:90]}\n    -> HTTP{r.status_code} {len(r.content):,}B")
        if r.status_code == 200:
            try:
                j = r.json()
                print(f"    rows={len(j):,} sample={j[:2]}")
            except Exception:
                print(f"    {r.text[:200]!r}")
        else:
            print(f"    {r.text[:200]!r}")
    except Exception as e:
        print(f"    EXC {e}")
    time.sleep(0.5)

# --------------------------------------------- 3. NASDAQ DATA LINK / SHARADAR
hr("3. Nasdaq Data Link (Quandl) -- free tables + Sharadar SEP")
for u in [
    "https://data.nasdaq.com/api/v3/datatables/SHARADAR/SEP.json?ticker=AAPL",
    "https://data.nasdaq.com/api/v3/datasets/WIKI/AAPL.json",
    "https://data.nasdaq.com/api/v3/datasets/EOD/AAPL.json",
    "https://data.nasdaq.com/api/v3/datatables/SHARADAR/TICKERS.json?ticker=SIVB",
    "https://data.nasdaq.com/api/v3/datasets/FRED/GDP.json",
]:
    try:
        r = S.get(u, timeout=60)
        print(f"  {u[38:110]}\n    -> HTTP{r.status_code} {len(r.content):,}B  {r.text[:220]!r}")
    except Exception as e:
        print(f"    EXC {e}")
    time.sleep(0.4)

# ------------------------------------------------- 4. FMP free (delisted list)
hr("4. FinancialModelingPrep free/demo -- delisted-companies endpoint")
for u in [
    "https://financialmodelingprep.com/api/v3/delisted-companies?apikey=demo",
    "https://financialmodelingprep.com/api/v3/historical-price-full/SIVB?apikey=demo",
    "https://financialmodelingprep.com/api/v3/historical-price-full/AAPL?apikey=demo",
    "https://financialmodelingprep.com/api/v3/stock/list?apikey=demo",
]:
    try:
        r = S.get(u, timeout=60)
        print(f"  {u[42:110]}\n    -> HTTP{r.status_code} {len(r.content):,}B {r.text[:250]!r}")
    except Exception as e:
        print(f"    EXC {e}")
    time.sleep(0.4)

# -------------------------------------------------------- 5. keyed free tiers
hr("5. Free tiers that need a signup key (measure the no-key response)")
for name, u in [
    ("Tiingo", "https://api.tiingo.com/tiingo/daily/sivb/prices"),
    ("Polygon", "https://api.polygon.io/v2/aggs/ticker/SIVB/range/1/day/2020-01-01/2023-12-31"),
    ("TwelveData", "https://api.twelvedata.com/time_series?symbol=SIVB&interval=1day"),
    ("Marketstack", "https://api.marketstack.com/v1/eod?symbols=SIVB"),
]:
    try:
        r = S.get(u, timeout=45)
        print(f"  {name:12s} HTTP{r.status_code} {r.text[:200]!r}")
    except Exception as e:
        print(f"  {name:12s} EXC {type(e).__name__}: {e}")
    time.sleep(0.3)

# ---------------------------------------------- 6. yfinance alternate symbols
hr("6. yfinance / Yahoo chart API: any alternate symbol form for delisted names?")
FORMS = []
for t in RECENT_DELISTED:
    FORMS += [t, t + "Q", t + "-DELISTED"]
FORMS += ["FRCB", "SBNY", "SIVBQ", "FTXMQ", "ENRNQ", "LEHMQ", "WAMUQ", "TWTR", "AAPL"]
for sym in FORMS:
    u = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
         f"?period1=0&period2=9999999999&interval=1d")
    try:
        r = S.get(u, timeout=30)
        j = r.json()
        res = (j.get("chart") or {}).get("result")
        err = (j.get("chart") or {}).get("error")
        if res and res[0].get("timestamp"):
            ts = res[0]["timestamp"]
            import datetime as dt
            a = dt.datetime.utcfromtimestamp(ts[0]).date()
            b = dt.datetime.utcfromtimestamp(ts[-1]).date()
            cl = res[0]["indicators"]["quote"][0]["close"]
            last = [c for c in cl if c is not None][-1:]
            print(f"  {sym:14s} HTTP{r.status_code} BARS={len(ts):<6} {a}..{b} lastClose={last}")
        else:
            code = (err or {}).get("code")
            print(f"  {sym:14s} HTTP{r.status_code} NO BARS err={code}")
    except Exception as e:
        print(f"  {sym:14s} EXC {type(e).__name__}: {str(e)[:80]}")
    time.sleep(0.25)

print("\nDONE")
