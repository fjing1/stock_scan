"""Probe: crypto as the free ORDER-FLOW development sandbox (control, not a model input).

Two things equities cannot give us for free are trivially free in crypto:
  1. signed trade-by-trade data with the aggressor side  -> true order-flow imbalance
  2. full limit-order-book depth snapshots                -> queue/depth features

Binance REST is geo-blocked from the US (HTTP 451) but the public historical dump host
data.binance.vision is NOT, and it ships per-day aggTrades with the maker/taker flag.
Coinbase's public REST works live with no key.
"""
import io
import time
import zipfile

import numpy as np
import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) research-probe"}


def sec(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


sec("1. BINANCE REST -- GEO BLOCK CONFIRMED")
r = requests.get("https://api.binance.com/api/v3/ping", headers=UA, timeout=30)
print(f"GET api.binance.com/api/v3/ping -> HTTP {r.status_code}")
print(f"body: {r.text[:200]}")

sec("2. data.binance.vision HISTORICAL DUMPS -- SIGNED TRADES, FREE, NO KEY")
u = ("https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/"
     "BTCUSDT-aggTrades-2026-09-11.zip")
t0 = time.time()
r = requests.get(u, headers=UA, timeout=180)
el = time.time() - t0
print(f"GET {u.split('/')[-1]} -> HTTP {r.status_code}, {len(r.content)/1e6:.2f} MB in {el:.1f}s "
      f"({len(r.content)/1e6/el:.1f} MB/s)")
z = zipfile.ZipFile(io.BytesIO(r.content))
print(f"zip members: {z.namelist()}")
cols = ["agg_id", "price", "qty", "first_id", "last_id", "ts", "is_buyer_maker",
        "best_match"]
raw = z.read(z.namelist()[0])
df = pd.read_csv(io.BytesIO(raw), header=None, names=cols)
print(f"rows={len(df):,}  uncompressed={len(raw)/1e6:.1f} MB")
print(df.head(4).to_string(index=False))
df["ts"] = pd.to_datetime(df.ts, unit="us", errors="coerce")
if df.ts.isna().all():
    df["ts"] = pd.to_datetime(pd.read_csv(io.BytesIO(raw), header=None,
                                          names=cols).ts, unit="ms")
print(f"time span: {df.ts.min()} .. {df.ts.max()}  ({df.ts.max()-df.ts.min()})")

sec("3. DERIVE REAL ORDER-FLOW IMBALANCE (the thing equities will not give us free)")
# is_buyer_maker == True  -> the AGGRESSOR was a SELLER
df["signed_qty"] = np.where(df.is_buyer_maker, -df.qty, df.qty)
df["notional"] = df.qty * df.price
df["signed_notional"] = np.where(df.is_buyer_maker, -df.notional, df.notional)
g = df.set_index("ts").resample("5min").agg(
    trades=("agg_id", "size"),
    vwap=("price", lambda s: s.mean()),
    volume=("qty", "sum"),
    signed_volume=("signed_qty", "sum"),
    notional=("notional", "sum"),
    signed_notional=("signed_notional", "sum"),
)
g["ofi"] = g.signed_volume / g.volume          # order-flow imbalance in [-1, 1]
g["taker_buy_share"] = (1 + g.ofi) / 2
print(f"5-minute bars built: {len(g):,}")
print(g.head(4).to_string(float_format=lambda x: f"{x:,.4f}"))
print(f"\nOFI stats: mean={g.ofi.mean():+.4f} std={g.ofi.std():.4f} "
      f"p05={g.ofi.quantile(.05):+.4f} p95={g.ofi.quantile(.95):+.4f}")
print(f"day totals: {len(df):,} trades, {df.qty.sum():,.2f} BTC, "
      f"${df.notional.sum()/1e9:,.3f}B notional, net taker flow "
      f"{df.signed_qty.sum():+,.2f} BTC (${df.signed_notional.sum()/1e6:+,.1f}M)")
g.to_csv("_data_probe_crypto_ofi_5min.csv")
print("saved -> _data_probe_crypto_ofi_5min.csv")

sec("4. HOW DEEP IS THE FREE HISTORY, AND WHAT DOES A FULL PULL COST?")
probes = ["2017-08-17", "2018-01-02", "2020-03-12", "2023-01-03", "2026-09-11",
          "2026-09-12", "2026-09-13"]
sizes = {}
for d in probes:
    uu = ("https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/"
          f"BTCUSDT-aggTrades-{d}.zip")
    h = requests.head(uu, headers=UA, timeout=60)
    n = int(h.headers.get("content-length", 0))
    sizes[d] = (h.status_code, n)
    print(f"  {d}  HTTP {h.status_code}  {n/1e6:>8.2f} MB")
ok = [v[1] for v in sizes.values() if v[0] == 200 and v[1] > 0]
if ok:
    avg = float(np.mean(ok))
    print(f"\navg day = {avg/1e6:.1f} MB compressed; ~3,300 days since 2017-08 ->"
          f" {avg*3300/1e9:.1f} GB for BTCUSDT alone")
print("also available per symbol/day: trades, klines (1s..1mo), bookTicker, bookDepth")
for kind in ["trades", "klines/1m", "bookTicker", "bookDepth"]:
    if kind.startswith("klines"):
        uu = ("https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1m/"
              "BTCUSDT-1m-2026-09-11.zip")
    else:
        uu = (f"https://data.binance.vision/data/spot/daily/{kind}/BTCUSDT/"
              f"BTCUSDT-{kind}-2026-09-11.zip")
    h = requests.head(uu, headers=UA, timeout=60)
    print(f"  {kind:12s} HTTP {h.status_code} "
          f"{int(h.headers.get('content-length',0))/1e6:>8.2f} MB")

sec("5. COINBASE PUBLIC REST -- LIVE, KEYLESS: BOOK DEPTH + AGGRESSOR SIDE")
r = requests.get("https://api.exchange.coinbase.com/products/BTC-USD/book?level=2",
                 headers=UA, timeout=40)
print(f"GET /book?level=2 -> HTTP {r.status_code}, {len(r.content)/1e6:.2f} MB")
b = r.json()
bids = pd.DataFrame(b["bids"], columns=["px", "sz", "n"]).astype({"px": float, "sz": float})
asks = pd.DataFrame(b["asks"], columns=["px", "sz", "n"]).astype({"px": float, "sz": float})
print(f"book levels: {len(bids):,} bids / {len(asks):,} asks   seq={b.get('sequence')}")
mid = (bids.px.iloc[0] + asks.px.iloc[0]) / 2
print(f"best bid {bids.px.iloc[0]:,.2f} / best ask {asks.px.iloc[0]:,.2f}  "
      f"mid {mid:,.2f}  spread {(asks.px.iloc[0]-bids.px.iloc[0])/mid*1e4:.2f} bps")
for bp in (5, 10, 25, 50):
    bd = bids[bids.px >= mid * (1 - bp / 1e4)].sz.sum()
    ad = asks[asks.px <= mid * (1 + bp / 1e4)].sz.sum()
    print(f"  depth within {bp:>2} bps: bid {bd:>10.4f} BTC  ask {ad:>10.4f} BTC  "
          f"imbalance {(bd-ad)/(bd+ad):+.4f}")

t0 = time.time()
r = requests.get("https://api.exchange.coinbase.com/products/BTC-USD/trades?limit=1000",
                 headers=UA, timeout=40)
el = time.time() - t0
tr = pd.DataFrame(r.json())
print(f"\nGET /trades?limit=1000 -> HTTP {r.status_code} {len(tr):,} rows in {el:.2f}s")
print(f"rate-limit headers: "
      f"{ {k: v for k, v in r.headers.items() if 'rate' in k.lower() or 'limit' in k.lower()} }")
tr["size"] = tr["size"].astype(float)
tr["time"] = pd.to_datetime(tr.time)
print(f"span {tr.time.min()} .. {tr.time.max()}  ({tr.time.max()-tr.time.min()})")
print("aggressor side counts:", tr.side.value_counts().to_dict())
sq = np.where(tr.side.eq("buy"), tr["size"], -tr["size"]).sum()
print(f"net signed size over the window: {sq:+.6f} BTC  "
      f"OFI={sq/tr['size'].sum():+.4f}")

sec("6. MEASURED COINBASE RATE LIMIT")
n_ok = 0
codes = {}
t0 = time.time()
for i in range(30):
    rr = requests.get("https://api.exchange.coinbase.com/products/BTC-USD/trades?limit=1",
                      headers=UA, timeout=20)
    codes[rr.status_code] = codes.get(rr.status_code, 0) + 1
    n_ok += rr.status_code == 200
el = time.time() - t0
print(f"30 sequential requests in {el:.2f}s = {30/el:.1f} req/s; codes={codes}")
print("Coinbase documents 10 req/s per IP on public endpoints; "
      f"{'no 429 seen' if 429 not in codes else 'HIT 429'} at this rate.")
