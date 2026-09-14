"""Probe: Twelve Data exact history floor + data quality, for the two demo-key
symbols that work (AAPL, QQQ).

Established: window requests reach 2020-03-02 (1084 bars) but 2015-01 returns
"No data is available". outputsize caps at 5000. `start_date` alone always
returns the NEWEST 5000 bars, so history must be walked with explicit
start+end windows.

CAVEAT to carry into the report: /api_usage says the demo key is
plan_category="ultra", plan_limit=2584 -- so the 40-req/6.7s burst measured
earlier is the ULTRA plan's limit, NOT the free tier's.
"""
import math
import time

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
BASE = "https://api.twelvedata.com/time_series"


def td(symbol, interval, start, end, outputsize=5000, key="demo", timeout=60):
    p = {"symbol": symbol, "interval": interval, "apikey": key,
         "outputsize": outputsize, "format": "JSON",
         "start_date": start, "end_date": end}
    t0 = time.time()
    r = requests.get(BASE, params=p, headers=UA, timeout=timeout)
    el = time.time() - t0
    try:
        js = r.json()
    except Exception:  # noqa: BLE001
        return r.status_code, None, "nonjson", el
    if isinstance(js, dict) and js.get("status") == "error":
        return r.status_code, None, str(js.get("message"))[:80], el
    return r.status_code, js.get("values"), "", el


print("=" * 104)
print("A -- exact floor: one 18-day window per year, 2012..2021, AAPL 5min")
print("=" * 104)
oldest = None
for y in range(2021, 2011, -1):
    code, vals, err, el = td("AAPL", "5min", f"{y}-01-02", f"{y}-01-20")
    n = len(vals) if vals else 0
    rng = f"{vals[-1]['datetime']} .. {vals[0]['datetime']}" if n else "-"
    print(f"  {y}  HTTP {code} n={n:5d}  {rng}  {err}")
    if n:
        oldest = y
    time.sleep(0.6)
print(f"  --> oldest YEAR with 5min data: {oldest}")

if oldest:
    print(f"\n  narrowing inside {oldest}: month by month")
    firstmonth = None
    for m in range(1, 13):
        code, vals, err, el = td("AAPL", "5min", f"{oldest}-{m:02d}-01",
                                 f"{oldest}-{m:02d}-19")
        n = len(vals) if vals else 0
        first = vals[-1]["datetime"] if n else "-"
        print(f"    {oldest}-{m:02d}  n={n:5d}  first={first}  {err}")
        if n and firstmonth is None:
            firstmonth = (m, first)
        time.sleep(0.6)
    print(f"  --> first month with data: {firstmonth}")
    # also check the year before, in case the floor straddles Jan
    py = oldest - 1
    for m in (7, 10, 12):
        code, vals, err, el = td("AAPL", "5min", f"{py}-{m:02d}-01", f"{py}-{m:02d}-19")
        n = len(vals) if vals else 0
        print(f"    {py}-{m:02d}  n={n:5d}  {err}")
        time.sleep(0.6)

print("\n" + "=" * 104)
print("B -- QQQ floor (QQQ is the useful one: a real index ETF)")
print("=" * 104)
for y in (2018, 2019, 2020, 2021, 2023):
    code, vals, err, el = td("QQQ", "5min", f"{y}-01-02", f"{y}-01-20")
    n = len(vals) if vals else 0
    rng = f"{vals[-1]['datetime']} .. {vals[0]['datetime']}" if n else "-"
    print(f"  QQQ {y}  HTTP {code} n={n:5d}  {rng}  {err}")
    time.sleep(0.6)

print("\n" + "=" * 104)
print("C -- intervals available at depth (AAPL, Jan 2021 window)")
print("=" * 104)
for iv in ("1min", "5min", "15min", "1h"):
    code, vals, err, el = td("AAPL", iv, "2021-01-04", "2021-01-22")
    n = len(vals) if vals else 0
    rng = f"{vals[-1]['datetime']} .. {vals[0]['datetime']}" if n else "-"
    print(f"  {iv:6s} HTTP {code} n={n:5d}  {rng}  {err}")
    time.sleep(0.6)

print("\n" + "=" * 104)
print("D -- the 5000-bar cap means ~63 trading days per request at 5min.")
print("    Measure how many requests are needed to walk AAPL back to the floor.")
print("=" * 104)
# 5min regular session = 78 bars/day -> 5000/78 = 64 days
code, vals, err, el = td("AAPL", "5min", "2021-01-04", "2021-04-10")
n = len(vals) if vals else 0
if n:
    d = pd.to_datetime([v["datetime"] for v in vals])
    print(f"  97-calendar-day request -> n={n} (cap={5000}), "
          f"covered {d.min()} .. {d.max()}, {d.normalize().nunique()} trading days")
    print(f"  => truncation confirmed: request span exceeded the 5000-bar cap"
          if n == 5000 else "  => no truncation")

print("\n" + "=" * 104)
print("E -- data quality: Twelve Data 5min vs yfinance 5min (AAPL, recent overlap)")
print("=" * 104)
import yfinance as yf  # noqa: E402

y = yf.download("AAPL", period="1mo", interval="5m", progress=False,
                auto_adjust=False, prepost=False, threads=False)
if isinstance(y.columns, pd.MultiIndex):
    y.columns = y.columns.get_level_values(0)
code, vals, err, el = td("AAPL", "5min", "2026-08-14", "2026-09-12")
if vals:
    t = pd.DataFrame(vals)
    # Twelve Data returns exchange-local (America/New_York) naive timestamps
    t["dtm"] = pd.to_datetime(t["datetime"]).dt.tz_localize(
        "America/New_York", nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC")
    for c in ("open", "high", "low", "close", "volume"):
        t[c] = pd.to_numeric(t[c])
    t = t.dropna(subset=["dtm"]).set_index("dtm")[["open", "high", "low", "close", "volume"]]
    yy = y.copy()
    yy.index = pd.to_datetime(yy.index, utc=True)
    yy = yy[["Open", "High", "Low", "Close", "Volume"]]
    yy.columns = ["open", "high", "low", "close", "volume"]
    j = t.join(yy, how="inner", lsuffix="_td", rsuffix="_yf")
    print(f"  td bars={len(t)}  yf bars={len(yy)}  matched={len(j)}")
    if len(j):
        for c in ("close", "high", "low"):
            dd = (j[f"{c}_td"] - j[f"{c}_yf"]).abs()
            rel = (dd / j[f"{c}_yf"]).dropna()
            print(f"  {c:6s} mean|diff|={dd.mean():.6f} max|diff|={dd.max():.4f} "
                  f"median rel={rel.median():.2e} bars rel>1e-4: {(rel>1e-4).sum()}")
        vd = (j["volume_td"] - j["volume_yf"]).abs()
        print(f"  volume: mean|diff|={vd.mean():.1f} exact matches="
              f"{(vd==0).sum()}/{len(j)}")
        print("\n  sample:")
        print(j[["close_td", "close_yf", "volume_td", "volume_yf"]].head(4).to_string())

print("\n" + "=" * 104)
print("F -- confirm the demo key's plan (the rate limit measured earlier is NOT free-tier)")
print("=" * 104)
r = requests.get("https://api.twelvedata.com/api_usage",
                 params={"apikey": "demo"}, headers=UA, timeout=30)
print(f"  /api_usage -> HTTP {r.status_code} {r.text}")
