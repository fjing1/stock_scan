"""Acquisition arithmetic for 5-minute bars, using MEASURED per-request numbers.

Measured inputs (from _data_probe_eodhd*.py / _data_probe_twelvedata2.py):
  EODHD demo : 600-day hard cap/request, ~78 RTH bars/day, 174 B/bar raw JSON,
               5 symbols x 5 chunks = 25 requests in 30.8 s wall
  TwelveData : 5000-bar cap/request -> ~64 trading days/request
Also measures the true on-the-wire (gzipped) size, since requests transparently
decompresses and len(r.content) reports the INFLATED size.
"""
import time

import requests

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

print("=" * 100)
print("MEASURE the real gzip ratio on the wire (len(r.content) is decompressed)")
print("=" * 100)
import json  # noqa: E402

now = int(time.time())
P = {"api_token": "demo", "interval": "5m", "fmt": "json",
     "from": now - 590 * 86400, "to": now}
URL = "https://eodhd.com/api/intraday/VTI.US"

# pass 1: measure the COMPRESSED bytes actually transferred (raw stream, no decode)
r1 = requests.get(URL, params=P, headers=UA, timeout=180, stream=True)
compressed = sum(len(c) for c in r1.raw.stream(65536, decode_content=False))
r1.close()
time.sleep(1.0)

# pass 2: normal request for the decompressed size and the bar count
r2 = requests.get(URL, params=P, headers=UA, timeout=180)
decompressed = len(r2.content)
nbars = len(json.loads(r2.content))
print(f"  one 590-day VTI 5m request:")
print(f"    bars                 : {nbars:,}")
print(f"    on the wire (gzip)   : {compressed/1e6:.2f} MB  ({compressed/nbars:.1f} B/bar)")
print(f"    after decompression  : {decompressed/1e6:.2f} MB  ({decompressed/nbars:.1f} B/bar)")
ratio = decompressed / compressed
print(f"    gzip ratio           : {ratio:.2f}x")

BPB_WIRE = compressed / nbars
BPB_RAW = decompressed / nbars

print("\n" + "=" * 100)
print("SCENARIO 1 -- what the FREE (no-signup) EODHD demo token can actually deliver")
print("=" * 100)
sym_free = 5
yrs_free = 5.91
bars = sym_free * yrs_free * 252 * 78
req = sym_free * 5
print(f"  symbols                : {sym_free} (AAPL MSFT AMZN TSLA VTI) + EURUSD + BTC-USD")
print(f"  span                   : 2020-10-12 .. today ({yrs_free:.2f} years)")
print(f"  bars                   : {bars:,.0f}")
print(f"  requests               : {req} (measured: 25)")
print(f"  wall clock             : 30.8 s  (MEASURED end to end)")
print(f"  wire volume            : {bars*BPB_WIRE/1e6:.0f} MB gzip / "
      f"{bars*BPB_RAW/1e6:.0f} MB inflated")
print("  VERDICT: trivially feasible. Already done, data is on disk.")

print("\n" + "=" * 100)
print("SCENARIO 2 -- the target the project asked about: 280 symbols x 10 years of 5-min")
print("=" * 100)
S, Y = 280, 10
bars = S * Y * 252 * 78
print(f"  bars                   : {bars:,} ({bars/1e6:.1f} M)")
print(f"  wire volume            : {bars*BPB_WIRE/1e9:.2f} GB gzip / "
      f"{bars*BPB_RAW/1e9:.2f} GB inflated JSON")
print(f"  as float32 parquet     : ~{bars*6*4/1e9:.2f} GB (6 cols x 4 B) -- the storable size")
print()
print("  request count under each vendor's MEASURED per-request cap:")
# EODHD: 600 calendar-day cap -> 10 years needs ceil(3652/600)=7 requests/symbol
eod_req = S * 7
print(f"    EODHD  (600 cal-day cap) : {eod_req:,} requests "
      f"({7} per symbol)")
print(f"       at the measured ~0.9 s/request, serial: "
      f"{eod_req*0.9/60:.0f} min wall")
print(f"       BUT: demo token is 5 symbols only, and its history floor is "
      f"2020-10-12 (5.9 y, not 10).")
print(f"       -> 280 symbols x 10 y is NOT reachable on any free EODHD access.")
# TwelveData: 5000-bar cap -> 64 trading days -> 2520/64 = 40 requests/symbol
td_req = S * 40
print(f"    TwelveData (5000-bar cap): {td_req:,} requests ({40} per symbol)")
for name, per_day in (("free tier 800 credits/day", 800),
                      ("free tier 8 credits/min", 8 * 60 * 24)):
    print(f"       under {name}: {td_req/per_day:.1f} days of quota"
          if per_day < td_req else
          f"       under {name}: fits in 1 day")
print(f"       -> at 800 req/day the download alone takes "
      f"{td_req/800:.0f} calendar days. And the demo key exposes 2 symbols.")
# Polygon: aggregates limit=50000 -> 2 years of 5m (39,312 bars) in ONE request
poly_bars_2y = 2 * 252 * 78
print(f"    Polygon (limit=50000/req): 2 y of 5-min = {poly_bars_2y:,} bars "
      f"-> 1 request/symbol")
print(f"       280 symbols = 280 requests; at a documented 5 req/min that is "
      f"{280/5:.0f} min = {280/5/60:.1f} h")
print(f"       10 y needs 5 requests/symbol = 1,400 requests = "
      f"{1400/5/60:.1f} h of wall clock")
print("       -> the ONLY shape that makes 280 symbols plausible, IF the free tier")
print("          really serves minute aggregates. UNVERIFIED: needs a key.")

print("\n" + "=" * 100)
print("SCENARIO 3 -- forward accumulation via the repo's existing intraday_store.py")
print("=" * 100)
print("  yfinance serves 60 days of 5m for ANY symbol, unlimited symbols, no key.")
print("  intraday_store.py already appends new bars per run, so history compounds.")
print(f"  280 symbols x 60-day seed = 280 requests, ~{280*0.5/60:.0f} min")
print(f"  bars after seeding      : {280*60*5/7*78/1e6:.1f} M  (60 cal days ~= 42 sessions)")
print("  to reach 10 years of 5m : 10 years of waiting. Useless for backtesting NOW,")
print("  but it is the only free path that covers the FULL 280-name universe.")
