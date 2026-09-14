"""Probe: Twelve Data free tier -- 5min depth, symbol coverage, measured rate limit.

The allowlist sweep got HTTP 200 from
  https://api.twelvedata.com/time_series?symbol=AAPL&interval=5min&apikey=demo
so measure how deep the history goes, whether the demo key is symbol-restricted,
and hammer it until it throttles to get the real requests/minute.
"""
import json
import time

import requests

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
BASE = "https://api.twelvedata.com/time_series"
KEY = "demo"


def td(symbol="AAPL", interval="5min", outputsize=5000, start=None, end=None,
       key=KEY, timeout=60):
    p = {"symbol": symbol, "interval": interval, "apikey": key,
         "outputsize": outputsize, "format": "JSON"}
    if start:
        p["start_date"] = start
    if end:
        p["end_date"] = end
    t0 = time.time()
    r = requests.get(BASE, params=p, headers=UA, timeout=timeout)
    return r, time.time() - t0


def show(r, el, label):
    try:
        js = r.json()
    except Exception:  # noqa: BLE001
        print(f"  {label:32s} HTTP {r.status_code} NON-JSON {r.text[:110]!r}")
        return None
    if isinstance(js, dict) and js.get("status") == "error":
        print(f"  {label:32s} HTTP {r.status_code} ERROR code={js.get('code')} "
              f"msg={str(js.get('message'))[:110]!r}")
        return None
    vals = js.get("values") if isinstance(js, dict) else None
    if not vals:
        print(f"  {label:32s} HTTP {r.status_code} no values: {str(js)[:140]}")
        return None
    print(f"  {label:32s} HTTP {r.status_code} n={len(vals):6d}  "
          f"{vals[-1]['datetime']} .. {vals[0]['datetime']}  ({el:.2f}s)")
    return js


print("=" * 108)
print("A -- demo key: max outputsize and resulting depth, 5min")
print("=" * 108)
for osz in (100, 1000, 5000, 10000):
    r, el = td(outputsize=osz)
    show(r, el, f"outputsize={osz}")
    time.sleep(1.0)

print("\n" + "=" * 108)
print("B -- can start_date reach back years? (5min)")
print("=" * 108)
for start in ("2026-08-01", "2026-01-02", "2025-01-02", "2022-01-03",
              "2018-01-02", "2010-01-04"):
    r, el = td(start=start, outputsize=5000)
    show(r, el, f"start_date={start}")
    time.sleep(1.0)

print("\n" + "=" * 108)
print("C -- explicit historical WINDOW deep in the past (5min)")
print("=" * 108)
for s, e in (("2026-06-01", "2026-06-20"), ("2024-01-02", "2024-01-20"),
             ("2020-03-02", "2020-03-20"), ("2015-01-02", "2015-01-20")):
    r, el = td(start=s, end=e, outputsize=5000)
    show(r, el, f"{s}..{e}")
    time.sleep(1.0)

print("\n" + "=" * 108)
print("D -- demo-key symbol coverage")
print("=" * 108)
for sym in ("AAPL", "MSFT", "SPY", "QQQ", "VTI", "NVDA", "TSLA", "GE", "JPM"):
    r, el = td(symbol=sym, outputsize=10)
    show(r, el, sym)
    time.sleep(1.0)

print("\n" + "=" * 108)
print("E -- rate limit: fire requests as fast as possible until throttled (max 40)")
print("=" * 108)
t0 = time.time()
codes = []
for i in range(40):
    r, el = td(outputsize=1, timeout=30)
    try:
        js = r.json()
    except Exception:  # noqa: BLE001
        js = {}
    code = js.get("code") if isinstance(js, dict) else None
    status = js.get("status") if isinstance(js, dict) else None
    codes.append((r.status_code, code))
    hdr = " ".join(f"{k}={v}" for k, v in r.headers.items()
                   if "rate" in k.lower() or "limit" in k.lower())
    print(f"  req {i+1:2d} HTTP {r.status_code} status={status} code={code} "
          f"{el:.2f}s t={time.time()-t0:5.1f}s {hdr}")
    if status == "error" and code in (429, 403):
        print(f"  --> THROTTLED at request {i+1} after {time.time()-t0:.1f}s")
        print(f"      message: {js.get('message')}")
        break
print(f"  total {len(codes)} requests in {time.time()-t0:.1f}s")

print("\n" + "=" * 108)
print("F -- API-usage endpoint (reports the plan's real quota)")
print("=" * 108)
for ep in ("https://api.twelvedata.com/api_usage",
           "https://api.twelvedata.com/quotes/latest"):
    try:
        r = requests.get(ep, params={"apikey": KEY}, headers=UA, timeout=30)
        print(f"  {ep}\n    HTTP {r.status_code} {r.text[:300]}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {ep} EXC {exc}")
    time.sleep(1.0)
