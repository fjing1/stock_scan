#!/usr/bin/env python3
"""Verify a real (non-demo) intraday API key in one command.

I could NOT test these free tiers myself: creating accounts is the user's job, and
alpaca / finnhub / databento are additionally blocked by this machine's egress
proxy. This script does the measurement the moment you have a key, so nothing has
to be taken on faith.

Usage -- set whichever keys you created, then run:

    export POLYGON_API_KEY=...
    export TWELVEDATA_API_KEY=...
    export TIINGO_API_KEY=...
    export EODHD_API_KEY=...
    export ALPACA_API_KEY_ID=...  ALPACA_API_SECRET_KEY=...
    /Users/feijing/github.com/stock_scan/vcp_env/bin/python _data_probe_verify_keys.py

For each key it reports: HTTP status, bar count, the OLDEST bar the free tier will
serve (the number that actually decides feasibility), and the measured
requests-per-minute before throttling.

Signup URLs (no credit card required on any of these free tiers):
  polygon    https://polygon.io/dashboard/signup
  twelvedata https://twelvedata.com/pricing        (Basic = free)
  tiingo     https://www.tiingo.com/account/api/token
  eodhd      https://eodhd.com/register
  alpaca     https://alpaca.markets/  -> Paper account -> Generate API keys
"""
import os
import time
import datetime as dt

import requests

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
YEARS_BACK = [1, 2, 3, 5, 8, 12, 20]


def probe_depth(label, make_url, parse_count, headers=None):
    """Walk a 20-day window back N years; report the oldest year that returns data."""
    print(f"\n--- {label}: history depth ---")
    oldest = None
    for yb in YEARS_BACK:
        d1 = dt.date.today() - dt.timedelta(days=int(yb * 365))
        d2 = d1 + dt.timedelta(days=20)
        url = make_url(d1.isoformat(), d2.isoformat())
        try:
            r = requests.get(url, headers=headers or UA, timeout=60)
        except Exception as exc:  # noqa: BLE001
            print(f"  -{yb:2d}y  EXC {type(exc).__name__}: {str(exc)[:90]}")
            continue
        n, note = parse_count(r)
        print(f"  -{yb:2d}y  ({d1}) HTTP {r.status_code}  bars={n:6d}  {note[:110]}")
        if n > 0:
            oldest = yb
        time.sleep(1.2)
    print(f"  ==> deepest window WITH data: ~{oldest} years back"
          if oldest else "  ==> NO historical data at any depth tested")
    return oldest


def rate_limit(label, url, headers=None, cap=70):
    """Hammer one cheap endpoint until a 429/402 appears; report req/min achieved."""
    print(f"\n--- {label}: measured rate limit ---")
    t0, n = time.time(), 0
    while n < cap:
        try:
            r = requests.get(url, headers=headers or UA, timeout=30)
        except Exception as exc:  # noqa: BLE001
            print(f"  stopped: {type(exc).__name__}")
            break
        n += 1
        if r.status_code in (429, 402, 403):
            el = time.time() - t0
            print(f"  THROTTLED after {n} requests in {el:.1f}s "
                  f"-> ~{n/el*60:.0f} req/min ceiling")
            print(f"  body: {r.text[:200]}")
            return
        if time.time() - t0 > 75:
            break
    el = time.time() - t0
    print(f"  {n} requests in {el:.1f}s with NO throttle -> >= {n/el*60:.0f} req/min")


# ---------------------------------------------------------------- POLYGON
k = os.environ.get("POLYGON_API_KEY")
print("=" * 96)
print(f"POLYGON  key={'SET' if k else 'not set -- skipping'}")
print("=" * 96)
if k:
    def pg_url(d1, d2):
        return (f"https://api.polygon.io/v2/aggs/ticker/SPY/range/5/minute/{d1}/{d2}"
                f"?adjusted=true&sort=asc&limit=50000&apiKey={k}")

    def pg_parse(r):
        try:
            j = r.json()
        except Exception:  # noqa: BLE001
            return 0, r.text[:100]
        if j.get("results"):
            res = j["results"]
            f = dt.datetime.fromtimestamp(res[0]["t"] / 1000, dt.timezone.utc)
            return len(res), f"first={f} status={j.get('status')}"
        return 0, f"status={j.get('status')} {str(j.get('error') or j.get('message'))[:80]}"

    probe_depth("polygon 5-min aggs", pg_url, pg_parse)
    # delisted-ticker / survivorship check -- the known wall for this repo
    print("\n--- polygon: does the free tier cover DELISTED tickers? ---")
    for sym, when in (("FRC", "2023-04-03"), ("SIVB", "2023-03-01"),
                      ("TWTR", "2022-06-01"), ("ATVI", "2023-09-01")):
        u = (f"https://api.polygon.io/v2/aggs/ticker/{sym}/range/5/minute/"
             f"{when}/{when}?limit=50000&apiKey={k}")
        r = requests.get(u, headers=UA, timeout=60)
        try:
            j = r.json()
            n = len(j.get("results") or [])
        except Exception:  # noqa: BLE001
            n = 0
            j = {}
        print(f"  {sym:5s} on {when}: HTTP {r.status_code} bars={n} "
              f"status={j.get('status')}")
        time.sleep(13)  # free tier is documented at 5 req/min
    rate_limit("polygon", f"https://api.polygon.io/v2/aggs/ticker/SPY/prev?apiKey={k}")

# ------------------------------------------------------------- TWELVE DATA
k = os.environ.get("TWELVEDATA_API_KEY")
print("\n" + "=" * 96)
print(f"TWELVE DATA  key={'SET' if k else 'not set -- skipping'}")
print("=" * 96)
if k:
    r = requests.get("https://api.twelvedata.com/api_usage",
                     params={"apikey": k}, headers=UA, timeout=30)
    print(f"  /api_usage -> {r.text[:300]}")

    def td_url(d1, d2):
        return (f"https://api.twelvedata.com/time_series?symbol=SPY&interval=5min"
                f"&outputsize=5000&start_date={d1}&end_date={d2}&apikey={k}")

    def td_parse(r):
        try:
            j = r.json()
        except Exception:  # noqa: BLE001
            return 0, r.text[:100]
        v = j.get("values")
        if v:
            return len(v), f"first={v[-1]['datetime']}"
        return 0, str(j.get("message") or j)[:100]

    probe_depth("twelvedata 5min", td_url, td_parse)
    rate_limit("twelvedata",
               f"https://api.twelvedata.com/time_series?symbol=SPY&interval=5min"
               f"&outputsize=1&apikey={k}")

# ------------------------------------------------------------------ TIINGO
k = os.environ.get("TIINGO_API_KEY")
print("\n" + "=" * 96)
print(f"TIINGO  key={'SET' if k else 'not set -- skipping'}")
print("=" * 96)
if k:
    def ti_url(d1, d2):
        return (f"https://api.tiingo.com/iex/SPY/prices?resampleFreq=5min"
                f"&startDate={d1}&endDate={d2}&token={k}")

    def ti_parse(r):
        try:
            j = r.json()
        except Exception:  # noqa: BLE001
            return 0, r.text[:100]
        if isinstance(j, list) and j:
            return len(j), f"first={j[0].get('date')}"
        return 0, str(j)[:100]

    probe_depth("tiingo IEX 5min", ti_url, ti_parse)
    print("  NOTE: tiingo's /iex endpoint is the IEX feed only (~2% of consolidated")
    print("        volume). Check whether prices/volumes look like the full tape.")

# ------------------------------------------------------------------- EODHD
k = os.environ.get("EODHD_API_KEY")
print("\n" + "=" * 96)
print(f"EODHD  key={'SET' if k else 'not set -- skipping'}")
print("=" * 96)
if k:
    # does a real key unlock symbols the demo token refused (SPY/QQQ/NVDA)?
    print("  symbols the DEMO token refused with 403 -- does a real key open them?")
    now = int(time.time())
    for sym in ("SPY.US", "QQQ.US", "NVDA.US", "AAPL.US"):
        r = requests.get(f"https://eodhd.com/api/intraday/{sym}",
                         params={"api_token": k, "interval": "5m", "fmt": "json",
                                 "from": now - 20 * 86400, "to": now},
                         headers=UA, timeout=90)
        try:
            n = len(r.json())
        except Exception:  # noqa: BLE001
            n = 0
        print(f"    {sym:9s} HTTP {r.status_code} bars={n} "
              f"{'' if r.status_code == 200 else r.text[:80]}")
        time.sleep(1)

    def eo_url(d1, d2):
        f = int(dt.datetime.fromisoformat(d1).replace(tzinfo=dt.timezone.utc).timestamp())
        t = int(dt.datetime.fromisoformat(d2).replace(tzinfo=dt.timezone.utc).timestamp())
        return (f"https://eodhd.com/api/intraday/SPY.US?interval=5m&fmt=json"
                f"&from={f}&to={t}&api_token={k}")

    def eo_parse(r):
        try:
            j = r.json()
        except Exception:  # noqa: BLE001
            return 0, r.text[:100]
        if isinstance(j, list) and j:
            return len(j), f"first={j[0].get('datetime')}"
        return 0, str(j)[:100]

    probe_depth("eodhd 5m (real key)", eo_url, eo_parse)

# ------------------------------------------------------------------ ALPACA
kid = os.environ.get("ALPACA_API_KEY_ID")
ksec = os.environ.get("ALPACA_API_SECRET_KEY")
print("\n" + "=" * 96)
print(f"ALPACA  keys={'SET' if (kid and ksec) else 'not set -- skipping'}")
print("=" * 96)
print("  NOTE: data.alpaca.markets is BLOCKED by this machine's egress proxy, so")
print("        this section may fail here even with valid keys. Run it elsewhere.")
if kid and ksec:
    H = {**UA, "APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": ksec}
    # THE critical test: IEX (free) vs SIP (paid) feed. IEX is ~2% of volume.
    print("\n--- alpaca: IEX vs SIP feed, same window, compare VOLUME ---")
    for feed in ("iex", "sip"):
        u = ("https://data.alpaca.markets/v2/stocks/bars?symbols=SPY&timeframe=5Min"
             f"&start=2024-06-03T13:30:00Z&end=2024-06-03T20:00:00Z&limit=100&feed={feed}")
        try:
            r = requests.get(u, headers=H, timeout=60)
            j = r.json()
            bars = (j.get("bars") or {}).get("SPY") or []
            vol = sum(b.get("v", 0) for b in bars)
            print(f"  feed={feed:3s} HTTP {r.status_code} bars={len(bars)} "
                  f"total_volume={vol:,} "
                  f"{'' if r.status_code == 200 else str(j)[:120]}")
        except Exception as exc:  # noqa: BLE001
            print(f"  feed={feed:3s} EXC {type(exc).__name__}: {str(exc)[:90]}")
        time.sleep(1)
    print("  INTERPRET: if iex total_volume is only a few percent of sip, the free")
    print("  feed is a biased price sample and realized vol from it is NOT the")
    print("  realized vol of the consolidated tape. That is the whole concern.")

    def ap_url(d1, d2):
        return ("https://data.alpaca.markets/v2/stocks/bars?symbols=SPY"
                f"&timeframe=5Min&start={d1}T00:00:00Z&end={d2}T00:00:00Z"
                "&limit=10000&feed=iex")

    def ap_parse(r):
        try:
            j = r.json()
        except Exception:  # noqa: BLE001
            return 0, r.text[:100]
        bars = (j.get("bars") or {}).get("SPY") or []
        if bars:
            return len(bars), f"first={bars[0].get('t')}"
        return 0, str(j)[:100]

    probe_depth("alpaca 5Min iex", ap_url, ap_parse, headers=H)

print("\ndone.")
