"""PROBE round 2: fix Stooq + Yahoo, then test the two highest-value delisted endpoints:
  - Alpha Vantage LISTING_STATUS (free key already in .env) -- active AND delisted rosters,
    with a point-in-time `date=` parameter. Also TIME_SERIES_DAILY_ADJUSTED on a delisted tick.
  - Wayback nasdaqtrader roster snapshots (fix the CDX collapse bug).
"""
import io
import json
import os
import re
import time

import pandas as pd
import requests

BROWSER = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
           "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def load_key():
    for p in ["/Users/feijing/github.com/stock_scan/.env"]:
        if os.path.exists(p):
            for ln in open(p):
                if ln.startswith("ALPHAVANTAGE_API_KEY"):
                    return ln.split("=", 1)[1].strip()
    return None


# ------------------------------------------------------------ 1. STOOQ, honestly
hr("1. STOOQ: what is actually in the HTML response?")
S = requests.Session()
S.headers.update({"User-Agent": BROWSER, "Accept": "text/csv,*/*",
                  "Referer": "https://stooq.com/q/d/?s=aapl.us"})
r = S.get("https://stooq.com/q/d/l/?s=aapl.us&i=d", timeout=45)
txt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", r.text))
print(f"  HTTP{r.status_code} {len(r.content)}B")
print(f"  stripped text: {txt[:400]}")

print("\n  -- alternate stooq hosts / paths --")
for u in ["https://stooq.pl/q/d/l/?s=aapl.us&i=d",
          "https://stooq.com/q/l/?s=aapl.us&f=sd2t2ohlcv&h&e=csv",
          "https://stooq.com/q/d/l/?s=^spx&i=d"]:
    try:
        rr = S.get(u, timeout=45)
        head = rr.text[:90].replace("\n", "|")
        print(f"  {u[:60]:60s} HTTP{rr.status_code} {len(rr.content):>7}B {head!r}")
    except Exception as e:
        print(f"  {u[:60]:60s} EXC {e}")
    time.sleep(0.6)

# ------------------------------------------------- 2. Yahoo chart API, debugged
hr("2. Yahoo chart API on delisted symbols (with real status/body reporting)")
Y = requests.Session()
Y.headers.update({"User-Agent": BROWSER, "Accept": "application/json"})
SYMS = ["AAPL", "SIVB", "SIVBQ", "ATVI", "VIAC", "SPLK", "CTXS", "ABMD", "CLVS",
        "TWTR", "FRC", "FRCB", "SBNY", "SBNYQ", "LEHMQ", "ENRNQ", "WCOM", "BSC",
        "SIVB.US", "SIVBQ.PK"]
import datetime as dtm
for sym in SYMS:
    u = (f"https://query2.finance.yahoo.com/v8/finance/chart/{sym}"
         f"?period1=0&period2=9999999999&interval=1d")
    try:
        r = Y.get(u, timeout=30)
        if r.status_code != 200:
            print(f"  {sym:10s} HTTP{r.status_code} body={r.text[:130]!r}")
        else:
            j = r.json()
            res = (j.get("chart") or {}).get("result")
            if res and res[0].get("timestamp"):
                ts = res[0]["timestamp"]
                a = dtm.datetime.utcfromtimestamp(ts[0]).date()
                b = dtm.datetime.utcfromtimestamp(ts[-1]).date()
                cl = [c for c in res[0]["indicators"]["quote"][0]["close"] if c is not None]
                print(f"  {sym:10s} HTTP200 BARS={len(ts):<6} {a}..{b} "
                      f"lastClose={cl[-1] if cl else None}")
            else:
                err = (j.get("chart") or {}).get("error") or {}
                print(f"  {sym:10s} HTTP200 NO BARS err={err.get('code')}/{err.get('description')}")
    except Exception as e:
        print(f"  {sym:10s} EXC {type(e).__name__}: {str(e)[:100]}")
    time.sleep(0.4)

# ----------------------------------------- 3. ALPHA VANTAGE LISTING_STATUS
hr("3. ALPHA VANTAGE LISTING_STATUS -- delisted roster + point-in-time date=")
key = load_key()
print(f"  key loaded: {'yes len=%d' % len(key) if key else 'NO'}")
A = requests.Session()
A.headers.update({"User-Agent": BROWSER})
calls = [
    ("active_now", {"function": "LISTING_STATUS", "state": "active"}),
    ("delisted_all", {"function": "LISTING_STATUS", "state": "delisted"}),
    ("active_2015-01-02", {"function": "LISTING_STATUS", "state": "active",
                           "date": "2015-01-02"}),
]
for label, params in calls:
    params["apikey"] = key
    try:
        t0 = time.time()
        r = A.get("https://www.alphavantage.co/query", params=params, timeout=180)
        dt = time.time() - t0
        print(f"\n  [{label}] HTTP{r.status_code} {len(r.content):,}B {dt:.1f}s")
        body = r.text
        if body.lstrip().startswith("{"):
            print(f"    JSON (=error/limit): {body[:400]}")
            continue
        df = pd.read_csv(io.StringIO(body))
        print(f"    rows={len(df):,} cols={list(df.columns)}")
        print(f"    head:\n{df.head(3).to_string()}")
        if "delistingDate" in df.columns:
            dd = df[df["delistingDate"] != "null"]
            print(f"    rows with a real delistingDate: {len(dd):,}")
            print(f"    delistingDate range: {dd['delistingDate'].min()} .. "
                  f"{dd['delistingDate'].max()}")
            print(f"    ipoDate range: {df['ipoDate'].min()} .. {df['ipoDate'].max()}")
        for t in ["SIVB", "ATVI", "VIAC", "SPLK", "CTXS", "ABMD", "CLVS", "TWTR",
                  "FRC", "SBNY", "AAPL"]:
            sub = df[df["symbol"] == t]
            if len(sub):
                print(f"    {t:5s} FOUND -> {sub.iloc[0].to_dict()}")
            else:
                print(f"    {t:5s} absent")
        fn = f"_data_probe_surv_av_{label}.csv"
        df.to_csv(fn, index=False)
        print(f"    saved {fn}")
    except Exception as e:
        print(f"  [{label}] EXC {type(e).__name__}: {str(e)[:200]}")
    time.sleep(1.5)

hr("3b. ALPHA VANTAGE daily bars for a DELISTED ticker (SIVB)")
for fn, sym in [("TIME_SERIES_DAILY_ADJUSTED", "SIVB"), ("TIME_SERIES_DAILY", "ATVI")]:
    p = {"function": fn, "symbol": sym, "outputsize": "full", "apikey": key,
         "datatype": "csv"}
    try:
        r = A.get("https://www.alphavantage.co/query", params=p, timeout=180)
        print(f"  {fn}/{sym} HTTP{r.status_code} {len(r.content):,}B")
        if r.text.lstrip().startswith("{"):
            print(f"    {r.text[:300]}")
        else:
            df = pd.read_csv(io.StringIO(r.text))
            print(f"    rows={len(df):,} cols={list(df.columns)} "
                  f"span={df.iloc[-1,0]}..{df.iloc[0,0]}")
            print(df.head(3).to_string())
    except Exception as e:
        print(f"  {fn}/{sym} EXC {e}")
    time.sleep(1.5)

# ------------------------ 4. Wayback nasdaqtrader roster snapshots (fixed CDX)
hr("4. Wayback nasdaqtrader nasdaqlisted.txt snapshots -- FIXED")
W = requests.Session()
W.headers.update({"User-Agent": BROWSER})
r = W.get("https://web.archive.org/cdx/search/cdx",
          params={"url": "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
                  "output": "json", "fl": "timestamp,statuscode,length",
                  "filter": "statuscode:200", "collapse": "timestamp:4"},
          timeout=180)
print(f"  CDX HTTP{r.status_code} {len(r.content)}B")
rows = json.loads(r.text)[1:] if r.status_code == 200 and r.text.strip() else []
print(f"  yearly 200 snapshots: {[x[0] for x in rows]}")
CHECK = ["SIVB", "ATVI", "SPLK", "CTXS", "ABMD", "CLVS", "TWTR", "AAPL"]
for ts, _sc, ln in rows:
    u = (f"https://web.archive.org/web/{ts}id_/"
         f"https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt")
    try:
        rr = W.get(u, timeout=180)
        lines = [l for l in rr.text.splitlines() if "|" in l]
        syms = {l.split("|")[0].strip() for l in lines[1:]}
        found = [t for t in CHECK if t in syms]
        print(f"  {ts[:8]} HTTP{rr.status_code} bytes={len(rr.content):>8,} "
              f"rows={len(lines):>6,} present={found}")
    except Exception as e:
        print(f"  {ts[:8]} EXC {type(e).__name__}: {str(e)[:90]}")
    time.sleep(1.2)

print("\nDONE")
