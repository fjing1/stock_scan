"""Probe: EODHD intraday endpoint depth with the public 'demo' token.

The allowlist sweep got HTTP 200 / 1,016,696 bytes of 5-minute bars from
  https://eodhd.com/api/intraday/AAPL.US?api_token=demo&interval=5m&fmt=json
so measure exactly:
  1. how far back the demo token will serve 5m bars (walk `from` back year by year)
  2. which symbols the demo token covers (is it AAPL-only?)
  3. what intervals exist (1m / 5m / 1h)
  4. the rate limit, by hammering until it throttles
  5. whether delisted tickers are present (survivorship)
"""
import json
import time
import datetime as dt

import requests

BASE = "https://eodhd.com/api/intraday/{sym}"
TOKEN = "demo"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def get(sym, interval="5m", frm=None, to=None, token=TOKEN, timeout=60):
    p = {"api_token": token, "interval": interval, "fmt": "json"}
    if frm is not None:
        p["from"] = int(frm)
    if to is not None:
        p["to"] = int(to)
    t0 = time.time()
    r = requests.get(BASE.format(sym=sym), params=p, headers=UA, timeout=timeout)
    el = time.time() - t0
    return r, el


def summarize(r, el, label):
    nb = len(r.content)
    if r.status_code != 200:
        print(f"  {label:34s} HTTP {r.status_code} bytes={nb} {r.text[:130]!r}")
        return None
    try:
        js = r.json()
    except Exception:  # noqa: BLE001
        print(f"  {label:34s} HTTP 200 bytes={nb} NON-JSON {r.text[:120]!r}")
        return None
    if not isinstance(js, list) or not js:
        print(f"  {label:34s} HTTP 200 bytes={nb} EMPTY/odd {str(js)[:130]}")
        return None
    d0, d1 = js[0].get("datetime"), js[-1].get("datetime")
    print(f"  {label:34s} HTTP 200 n={len(js):7d} bytes={nb:9d} "
          f"{d0} .. {d1}  ({el:.1f}s)")
    return js


print("=" * 118)
print("STEP 1 -- how deep does the demo token go for AAPL.US 5m? walk `from` back one year at a time")
print("=" * 118)
now = int(time.time())
YEAR = 365 * 24 * 3600
depth = {}
for yrs_back in [0.25, 0.5, 1, 2, 3, 5, 8, 12, 20]:
    frm = now - int(yrs_back * YEAR)
    r, el = get("AAPL.US", "5m", frm=frm, to=now)
    js = summarize(r, el, f"from -{yrs_back}y")
    depth[yrs_back] = (r.status_code, len(js) if js else 0,
                       js[0]["datetime"] if js else None,
                       js[-1]["datetime"] if js else None)
    time.sleep(1.0)

print("\n" + "=" * 118)
print("STEP 2 -- is the oldest reachable bar a hard floor? request a WINDOW deep in the past")
print("=" * 118)
for y in [2005, 2010, 2015, 2018, 2020, 2022, 2024, 2025]:
    frm = int(dt.datetime(y, 1, 2, tzinfo=dt.timezone.utc).timestamp())
    to = int(dt.datetime(y, 2, 2, tzinfo=dt.timezone.utc).timestamp())
    r, el = get("AAPL.US", "5m", frm=frm, to=to)
    summarize(r, el, f"window {y}-01-02..{y}-02-02")
    time.sleep(1.0)

print("\n" + "=" * 118)
print("STEP 3 -- which symbols does the demo token cover? (demo tokens are usually 1-5 symbols)")
print("=" * 118)
frm, to = now - 20 * 24 * 3600, now
for sym in ["AAPL.US", "MSFT.US", "SPY.US", "SPY", "AMZN.US", "TSLA.US",
            "VTI.US", "QQQ.US", "NVDA.US", "GE.US"]:
    r, el = get(sym, "5m", frm=frm, to=to)
    summarize(r, el, sym)
    time.sleep(1.0)

print("\n" + "=" * 118)
print("STEP 4 -- which intervals are served?")
print("=" * 118)
for iv in ["1m", "5m", "1h"]:
    r, el = get("AAPL.US", iv, frm=now - 20 * 24 * 3600, to=now)
    summarize(r, el, f"interval={iv}")
    time.sleep(1.0)

print("\n" + "=" * 118)
print("STEP 5 -- delisted / survivorship check (demo token, recent window)")
print("=" * 118)
for sym in ["FRC.US", "SIVB.US", "LEHMQ.US", "TWTR.US", "ATVI.US"]:
    r, el = get(sym, "5m", frm=now - 20 * 24 * 3600, to=now)
    summarize(r, el, f"delisted? {sym}")
    time.sleep(1.0)

print("\n" + "=" * 118)
print("STEP 6 -- rate limit: fire 25 small requests back to back, watch for 402/429")
print("=" * 118)
codes = []
t_start = time.time()
for i in range(25):
    r, el = get("AAPL.US", "5m", frm=now - 2 * 24 * 3600, to=now, timeout=30)
    codes.append(r.status_code)
    extra = ""
    for h in ("x-ratelimit-remaining", "x-ratelimit-limit", "retry-after",
              "x-api-calls-remaining"):
        if h in r.headers:
            extra += f" {h}={r.headers[h]}"
    print(f"  req {i+1:2d}/25 HTTP {r.status_code} bytes={len(r.content):7d} "
          f"{el:.2f}s{extra}" + (f"  body={r.text[:90]!r}" if r.status_code != 200 else ""))
    if r.status_code in (402, 403, 429):
        print("  --> throttled/blocked, stopping")
        break
print(f"  elapsed {time.time()-t_start:.1f}s for {len(codes)} requests; codes={sorted(set(codes))}")

print("\n=== response headers on a successful call (quota hints) ===")
r, el = get("AAPL.US", "5m", frm=now - 2 * 24 * 3600, to=now)
for k, v in sorted(r.headers.items()):
    print(f"  {k}: {v[:120]}")
