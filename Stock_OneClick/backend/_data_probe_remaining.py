"""Final probe: remaining reachable sources.

1. STOOQ -- is the intraday endpoint real, or is it a JS anti-bot wall + daily-only?
2. Keyed vendors reachable through the proxy (polygon / tiingo / fmp / marketstack):
   capture the EXACT auth error so the signup requirement is documented from the
   server's own words, not from a pricing page.
3. ALPHA VANTAGE TIME_SERIES_INTRADAY with the `month=YYYY-MM` slice parameter.
   Prior repo testing (2026-06) found intraday premium-gated, but that predates
   the `month` slicing param and this is the single highest-value test in the
   intraday category: if AV free serves 20+ years of 5min for any symbol, it ends
   the search. Budget: 2 requests out of the 25/day cap.
"""
import json
import os
import time

import requests

UA_BROWSER = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}
UA_PLAIN = {"User-Agent": "python-requests/2.32"}

print("=" * 100)
print("1 -- STOOQ: intraday or daily-only? try both UAs, both hosts, session cookies")
print("=" * 100)
for host in ("stooq.com", "stooq.pl"):
    for uaname, ua in (("browser", UA_BROWSER), ("plain", UA_PLAIN)):
        s = requests.Session()
        for label, url in (
            ("daily  i=d", f"https://{host}/q/d/l/?s=spy.us&i=d"),
            ("5min   i=5", f"https://{host}/q/d/l/?s=spy.us&i=5"),
        ):
            try:
                r = s.get(url, headers=ua, timeout=25)
            except Exception as exc:  # noqa: BLE001
                print(f"  {host:10s} {uaname:7s} {label} EXC {type(exc).__name__}")
                continue
            body = r.text
            is_csv = body[:40].lower().startswith("date")
            is_js = "requires JavaScript" in body or "noscript" in body
            print(f"  {host:10s} {uaname:7s} {label} HTTP {r.status_code} "
                  f"bytes={len(r.content):7d} csv={is_csv} js_wall={is_js}")
            if is_csv:
                lines = body.strip().split("\n")
                print(f"      header: {lines[0]}")
                print(f"      rows={len(lines)-1} first={lines[1][:60]} last={lines[-1][:60]}")
            time.sleep(0.5)

print("\n" + "=" * 100)
print("2 -- keyed vendors reachable through the proxy: exact auth error + free-tier hint")
print("=" * 100)
KEYED = [
    ("polygon 5min aggs",
     "https://api.polygon.io/v2/aggs/ticker/SPY/range/5/minute/2016-01-04/2016-01-08",
     {"apiKey": "invalid_test_key"}),
    ("polygon 5min aggs (no key)",
     "https://api.polygon.io/v2/aggs/ticker/SPY/range/5/minute/2016-01-04/2016-01-08", {}),
    ("tiingo iex 5min",
     "https://api.tiingo.com/iex/SPY/prices",
     {"resampleFreq": "5min", "startDate": "2020-01-02", "token": "invalid_test_key"}),
    ("fmp 5min chart",
     "https://financialmodelingprep.com/api/v3/historical-chart/5min/SPY",
     {"apikey": "invalid_test_key"}),
    ("marketstack intraday",
     "https://api.marketstack.com/v1/intraday",
     {"symbols": "SPY", "interval": "5min", "access_key": "invalid_test_key"}),
    ("eodhd intraday (real-key path)",
     "https://eodhd.com/api/intraday/SPY.US",
     {"interval": "5m", "fmt": "json", "api_token": "invalid_test_key"}),
]
for name, url, params in KEYED:
    try:
        r = requests.get(url, params=params, headers=UA_BROWSER, timeout=25)
        print(f"  {name:32s} HTTP {r.status_code}  {r.text[:230]!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {name:32s} EXC {type(exc).__name__}: {str(exc)[:110]}")
    time.sleep(0.5)

print("\n" + "=" * 100)
print("3 -- ALPHA VANTAGE TIME_SERIES_INTRADAY with month= slicing (2 requests, real key)")
print("=" * 100)
key = None
envp = "/Users/feijing/github.com/stock_scan/.env"
if os.path.exists(envp):
    for line in open(envp):
        if line.startswith("ALPHAVANTAGE_API_KEY"):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
if not key:
    print("  no ALPHAVANTAGE_API_KEY found, skipping")
else:
    print(f"  key loaded, length={len(key)}")
    for label, params in (
        ("5min month=2020-03 (COVID slice)",
         {"function": "TIME_SERIES_INTRADAY", "symbol": "SPY", "interval": "5min",
          "month": "2020-03", "outputsize": "full", "apikey": key}),
        ("5min month=2010-06 (deep history)",
         {"function": "TIME_SERIES_INTRADAY", "symbol": "SPY", "interval": "5min",
          "month": "2010-06", "outputsize": "full", "apikey": key}),
    ):
        r = requests.get("https://www.alphavantage.co/query", params=params,
                         headers=UA_BROWSER, timeout=90)
        try:
            js = r.json()
        except Exception:  # noqa: BLE001
            print(f"  {label}: HTTP {r.status_code} NON-JSON {r.text[:200]!r}")
            continue
        keys = list(js.keys())
        print(f"\n  {label}: HTTP {r.status_code} bytes={len(r.content)} keys={keys}")
        series = next((v for k, v in js.items() if "Time Series" in k), None)
        if series:
            ts = sorted(series.keys())
            print(f"    *** DATA *** n={len(ts)} bars, {ts[0]} .. {ts[-1]}")
            sample = series[ts[0]]
            print(f"    first bar: {json.dumps(sample)}")
            print(f"    meta: {json.dumps(js.get('Meta Data', {}))[:300]}")
        else:
            for k in keys:
                print(f"    {k}: {str(js[k])[:260]}")
        time.sleep(2)
