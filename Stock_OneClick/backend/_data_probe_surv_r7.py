"""PROBE round 7 -- MEASURE the stockanalysis.com free keyless API for delisted daily OHLCV.
This is the load-bearing result, so measure it hard:
  - full history depth per symbol (range=MAX?)
  - coverage across many delisted names incl. old ones
  - is the adjusted close populated (splits/divs) for delisted names?
  - MEASURED rate limit: how many requests before throttling, and latency
  - the full delisted ROSTER (paginated by year?)
  - wall-clock / volume arithmetic for a 15,683-symbol pull
"""
import json
import statistics
import time

import pandas as pd
import requests

BR = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
S = requests.Session()
S.headers.update({"User-Agent": BR, "Accept": "application/json"})
BASE = "https://stockanalysis.com/api/symbol/s/{sym}/history"


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def fetch(sym, rng="MAX", period="Daily", timeout=60):
    t0 = time.time()
    r = S.get(BASE.format(sym=sym), params={"range": rng, "period": period},
              timeout=timeout)
    dt = time.time() - t0
    return r, dt


hr("7.1 range= parameter: what gets the FULL history?")
for rng in ["MAX", "Max", "max", "50Y", "30Y", "10Y", "5Y"]:
    try:
        r, dt = fetch("SIVB", rng)
        if r.status_code == 200:
            j = r.json()
            d = j.get("data") or []
            print(f"  range={rng:5s} HTTP200 {len(r.content):>9,}B rows={len(d):<6} "
                  f"{d[-1]['t'] if d else '-'}..{d[0]['t'] if d else '-'} {dt:.2f}s")
        else:
            print(f"  range={rng:5s} HTTP{r.status_code} {r.text[:100]!r}")
    except Exception as e:
        print(f"  range={rng:5s} EXC {type(e).__name__}: {str(e)[:80]}")
    time.sleep(0.8)

hr("7.2 DELISTED COVERAGE -- the repo's casualties + older blowups")
BEST = "50Y"
TESTS = ["SIVB", "ATVI", "VIAC", "SPLK", "CTXS", "ABMD", "CLVS", "TWTR", "FRC", "SBNY",
         "FRCB", "SIVBQ", "LEHMQ", "ENRNQ", "WCOM", "BSC", "WAMUQ", "AABA", "AAWW",
         "ABAX", "ACAS", "AAPL"]
rows = []
lat = []
for sym in TESTS:
    try:
        r, dt = fetch(sym, BEST)
        lat.append(dt)
        if r.status_code == 200:
            j = r.json()
            d = j.get("data") or []
            if d:
                adj = sum(1 for x in d if x.get("a") is not None)
                anydiff = sum(1 for x in d if x.get("a") is not None
                              and abs(x["a"] - x["c"]) > 1e-9)
                vol = sum(1 for x in d if x.get("v"))
                print(f"  {sym:7s} rows={len(d):<6} {d[-1]['t']}..{d[0]['t']} "
                      f"lastC={d[0]['c']:<10} adjCol={adj}/{len(d)} adj!=close={anydiff} "
                      f"volNZ={vol} {len(r.content)/1024:.0f}KB {dt:.2f}s")
                rows.append((sym, len(d), d[-1]["t"], d[0]["t"], len(r.content)))
            else:
                print(f"  {sym:7s} HTTP200 EMPTY  status={j.get('status')} {dt:.2f}s")
        else:
            print(f"  {sym:7s} HTTP{r.status_code} {r.text[:90]!r} {dt:.2f}s")
    except Exception as e:
        print(f"  {sym:7s} EXC {type(e).__name__}: {str(e)[:80]}")
    time.sleep(0.5)

if rows:
    tot_rows = sum(x[1] for x in rows)
    tot_bytes = sum(x[4] for x in rows)
    print(f"\n  OK symbols={len(rows)}/{len(TESTS)}  total rows={tot_rows:,}  "
          f"total bytes={tot_bytes/1024/1024:.2f}MB")
    print(f"  mean rows/symbol={tot_rows/len(rows):,.0f}  "
          f"mean KB/symbol={tot_bytes/len(rows)/1024:.0f}")
    print(f"  latency: median={statistics.median(lat):.2f}s "
          f"min={min(lat):.2f}s max={max(lat):.2f}s")

hr("7.3 sample the ACTUAL VALUES for one delisted name (sanity vs known history)")
r, _ = fetch("ATVI", BEST)
d = r.json()["data"]
print(f"  ATVI rows={len(d)} span {d[-1]['t']}..{d[0]['t']}")
print("  last 5 rows (newest first):")
for x in d[:5]:
    print(f"    {x}")
print("  first 3 rows (oldest):")
for x in d[-3:]:
    print(f"    {x}")
df = pd.DataFrame(d)
df.to_csv("_data_probe_surv_sa_atvi.csv", index=False)
print(f"  saved _data_probe_surv_sa_atvi.csv  cols={list(df.columns)}")

hr("7.4 MEASURED RATE LIMIT: burst 60 requests as fast as possible")
BURST = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "JPM", "V",
         "UNH", "XOM", "JNJ", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "KO",
         "PEP", "COST", "WMT", "BAC", "TMO", "MCD", "CSCO", "ACN", "ADBE", "LIN"] * 2
codes = {}
t0 = time.time()
first_fail = None
for i, sym in enumerate(BURST):
    try:
        r = S.get(BASE.format(sym=sym), params={"range": "1Y", "period": "Daily"},
                  timeout=40)
        codes[r.status_code] = codes.get(r.status_code, 0) + 1
        if r.status_code != 200 and first_fail is None:
            first_fail = (i + 1, r.status_code, r.text[:90])
    except Exception as e:
        codes[type(e).__name__] = codes.get(type(e).__name__, 0) + 1
        if first_fail is None:
            first_fail = (i + 1, type(e).__name__, str(e)[:80])
el = time.time() - t0
print(f"  {len(BURST)} requests in {el:.1f}s = {len(BURST)/el:.1f} req/s")
print(f"  status counts={codes}")
print(f"  first failure at request #{first_fail}" if first_fail
      else "  NO FAILURES in the burst")

hr("7.5 the FULL delisted roster -- is it paginated by year?")
import io
import re
for u in ["https://stockanalysis.com/actions/delisted/2015/",
          "https://stockanalysis.com/actions/delisted/2010/",
          "https://stockanalysis.com/actions/delisted/2026/"]:
    try:
        r = S.get(u, headers={"Accept": "text/html"}, timeout=90)
        if r.status_code == 200:
            tabs = pd.read_html(io.StringIO(r.text))
            t = tabs[0]
            print(f"  {u[-6:-1]}: HTTP200 rows={len(t)} cols={list(t.columns)} "
                  f"first={t.iloc[0].to_dict()} last={t.iloc[-1].to_dict()}")
            t.to_csv(f"_data_probe_surv_sa_delisted_{u[-5:-1]}.csv", index=False)
        else:
            print(f"  {u}: HTTP{r.status_code}")
    except Exception as e:
        print(f"  {u}: EXC {type(e).__name__}: {str(e)[:90]}")
    time.sleep(1.2)

hr("7.6 ARITHMETIC for a delisted-inclusive panel from this API")
if rows:
    mean_kb = tot_bytes / len(rows) / 1024
    med_lat = statistics.median(lat)
    N = 15683
    print(f"  measured: {mean_kb:.0f} KB/symbol, median {med_lat:.2f}s/request")
    print(f"  {N:,} symbols -> {N*mean_kb/1024/1024:.2f} GB, "
          f"serial wall clock {N*med_lat/3600:.1f} h")
    for w in (4, 8, 16):
        print(f"    with {w} parallel workers (be polite): "
              f"{N*med_lat/3600/w:.1f} h")
    print(f"  the repo's realistic ask (1,283 current + ~7,486 delisted = 8,769): "
          f"{8769*mean_kb/1024:.0f} MB, {8769*med_lat/3600:.1f} h serial")

print("\nDONE")
