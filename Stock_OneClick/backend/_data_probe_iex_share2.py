"""Probe: THE decisive IEX limitation test (v2 -- fixed).

v1 problems:
  * O(n^2) buffer re-slicing made parsing 1.1 MB/s -> replaced with an offset pointer
    over a bytearray that is compacted only occasionally.
  * the single long-lived streaming GET was cut by the proxy at 288 MB
    -> replaced with sequential 32 MB HTTP Range requests with retry, fed into one
       continuous zlib decompressobj.

Measures, on one FULL IEX TOPS trading day:
  1. IEX's share of consolidated daily volume, per symbol (the "~2%" claim)
  2. 5-minute bin coverage -- how many of the 78 RTH bins have ZERO IEX trades
  3. IEX-only 5-minute realized vol vs the daily Parkinson range (bias check)
"""
import struct
import time
import zlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) research-probe"}
DATE = "20161212"
CH = 32 << 20


def sec(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


sec("0. LOCATE THE FULL-DAY FILE")
idx = requests.get("https://iextrading.com/api/1.0/hist", headers=UA, timeout=60).json()
ent = [e for e in idx[DATE] if e["feed"] == "TOPS"][0]
TOTAL = int(ent["size"])
LINK = ent["link"]
print(f"{DATE} TOPS {ent['version']}  {TOTAL/1e9:.3f} GB compressed")

sec("1. STREAM THE WHOLE DAY (ranged, resumable) AND AGGREGATE TRADES")
sym_vol = Counter()
sym_notional = Counter()
sym_trades = Counter()
bars = defaultdict(dict)          # sym -> bin -> [shares, notional, first_px, last_px]
types = Counter()
pkts = trades_n = 0
t0 = time.time()
dob = zlib.decompressobj(16 + zlib.MAX_WBITS)
buf = bytearray()
pos = 0
hdr_done = False
kind = None
sess = requests.Session()
sess.headers.update(UA)
got = 0
start = 0
unpack_rec = struct.Struct("<IIII").unpack_from
unpack_h = struct.Struct("<H").unpack_from
unpack_trade = struct.Struct("<Iqq").unpack_from
unpack_q = struct.Struct("<q").unpack_from

while start < TOTAL:
    end = min(start + CH - 1, TOTAL - 1)
    raw = None
    for attempt in range(4):
        try:
            rr = sess.get(LINK, headers={"Range": f"bytes={start}-{end}"}, timeout=300)
            if rr.status_code in (200, 206) and len(rr.content):
                raw = rr.content
                break
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    if raw is None:
        print(f"  !! gave up on range {start}-{end}")
        break
    got += len(raw)
    start = end + 1
    buf += dob.decompress(raw)

    if not hdr_done and len(buf) >= 24:
        m = struct.unpack_from("<I", buf, 0)[0]
        if m in (0xA1B2C3D4, 0xA1B23C4D):
            kind = "pcap"
            pos = 24
        elif m == 0x0A0D0D0A:
            kind = "pcapng"
            pos = 0
        else:
            raise ValueError(bytes(buf[:4]).hex())
        hdr_done = True
        print(f"  format = {kind}")

    n = len(buf)
    while True:
        if kind == "pcap":
            if pos + 16 > n:
                break
            _ts, _tf, incl, _o = unpack_rec(buf, pos)
            if pos + 16 + incl > n:
                break
            p0, p1 = pos + 16, pos + 16 + incl
            pos = p1
        else:
            if pos + 12 > n:
                break
            bt, bl = struct.unpack_from("<II", buf, pos)
            if bl < 12 or pos + bl > n:
                break
            p0 = p1 = pos
            if bt == 6 and bl >= 28:
                _i, _th, _tl, cl, _ol = struct.unpack_from("<IIIII", buf, pos + 8)
                p0, p1 = pos + 28, pos + 28 + cl
            pos += bl
            if p1 <= p0:
                continue
        pkts += 1
        if p1 - p0 < 82:
            continue
        mc = unpack_h(buf, p0 + 60)[0]
        o = p0 + 82
        for _ in range(mc):
            if o + 2 > p1:
                break
            ml = unpack_h(buf, o)[0]
            o += 2
            if o + ml > p1 or ml == 0:
                break
            mt = buf[o]
            types[chr(mt)] += 1
            if mt == 0x54 and ml >= 38:                    # 'T' trade report
                tstamp = unpack_q(buf, o + 2)[0]
                sym = bytes(buf[o + 10:o + 18]).decode("latin1").rstrip()
                sz, px, _tid = unpack_trade(buf, o + 18)
                px /= 1e4
                trades_n += 1
                sym_vol[sym] += sz
                sym_notional[sym] += sz * px
                sym_trades[sym] += 1
                b5 = tstamp // 300_000_000_000
                sb = bars[sym]
                e = sb.get(b5)
                if e is None:
                    sb[b5] = [float(sz), sz * px, px, px]
                else:
                    e[0] += sz
                    e[1] += sz * px
                    e[3] = px
            o += ml
    # compact
    if pos > (8 << 20):
        del buf[:pos]
        pos = 0
    if got % (200 << 20) < CH:
        print(f"  {got/1e9:.2f}/{TOTAL/1e9:.2f} GB  {pkts:,} pkts  {trades_n:,} trades  "
              f"{time.time()-t0:.0f}s", flush=True)

el = time.time() - t0
print(f"\nDONE: {got/1e9:.3f} of {TOTAL/1e9:.3f} GB in {el:.1f}s "
      f"({got/1e6/el:.1f} MB/s end-to-end incl. parsing)")
print(f"packets={pkts:,}  messages={sum(types.values()):,}  trades={trades_n:,}")
print(f"message census: {dict(types.most_common())}")
print(f"distinct symbols traded on IEX: {len(sym_vol):,}")
print(f"IEX total shares={sum(sym_vol.values()):,}  "
      f"notional=${sum(sym_notional.values())/1e9:,.3f} B")

sec("2. IEX SHARE OF CONSOLIDATED VOLUME, PER SYMBOL (the '~2%' claim, MEASURED)")
import yfinance as yf
probe = [s for s in ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "BAC", "GE", "F", "XOM", "JNJ",
                     "NVDA", "INTC", "CSCO", "PFE", "T", "VZ", "WFC", "KO", "GLD", "EEM"]
         if s in sym_vol]
yq = yf.download(probe, start="2016-12-09", end="2016-12-15", progress=False,
                 auto_adjust=False)
tgt = pd.Timestamp("2016-12-12")
rows = []
for s in probe:
    try:
        cons = float(yq["Volume"].loc[tgt, s])
    except Exception:
        continue
    if cons > 0:
        rows.append((s, sym_trades[s], sym_vol[s], cons, sym_vol[s] / cons))
cmp_ = pd.DataFrame(rows, columns=["symbol", "iex_trades", "iex_shares",
                                   "consolidated", "iex_share"]).sort_values(
    "iex_share", ascending=False)
print(cmp_.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
print(f"\nMEDIAN IEX share of consolidated volume = {cmp_.iex_share.median()*100:.2f}%"
      f"   (range {cmp_.iex_share.min()*100:.2f}% .. {cmp_.iex_share.max()*100:.2f}%)")

sec("3. 5-MINUTE BIN COVERAGE -- CAN HAR-RV BE BUILT FROM THIS AT ALL?")
day = pd.Timestamp("2016-12-12")
open_ns = int((day + pd.Timedelta(hours=14, minutes=30)).value)   # 09:30 ET = 14:30 UTC (EST)
bins_rth = set((open_ns + i * 300_000_000_000) // 300_000_000_000 for i in range(78))
rows = []
for s in [x for x, _ in sym_vol.most_common(60)]:
    filled = sum(1 for b in bins_rth if b in bars[s] and bars[s][b][0] > 0)
    rows.append((s, sym_vol[s], filled, 78 - filled, filled / 78))
cov = pd.DataFrame(rows, columns=["symbol", "iex_shares", "bins_filled", "bins_empty",
                                  "coverage"])
print(cov.head(25).to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
allc = []
for s in sym_vol:
    filled = sum(1 for b in bins_rth if b in bars[s] and bars[s][b][0] > 0)
    allc.append(filled / 78)
a = pd.Series(allc)
print(f"\nTOP-60 IEX names: median coverage={cov.coverage.median():.3f} "
      f"-> {(1-cov.coverage.median())*78:.0f} of 78 bins EMPTY")
print(f"ALL {len(a):,} IEX-traded symbols: mean={a.mean():.3f} median={a.median():.3f} "
      f"share with >90% of bins filled={float((a>0.9).mean()):.4f} "
      f"share with >50%={float((a>0.5).mean()):.4f}")

sec("4. IEX-ONLY 5-MIN REALIZED VOL vs THE DAILY PARKINSON RANGE (bias check)")
rows = []
for s in probe:
    px = [(k, v[3]) for k, v in sorted(bars[s].items()) if k in bins_rth and v[0] > 0]
    if len(px) < 20:
        continue
    p = np.array([v for _, v in px])
    rv = float(np.sqrt((np.diff(np.log(p)) ** 2).sum()))
    hi, lo = float(yq["High"].loc[tgt, s]), float(yq["Low"].loc[tgt, s])
    park = float(np.sqrt(np.log(hi / lo) ** 2 / (4 * np.log(2))))
    rows.append((s, len(p), rv * 100, park * 100, rv / park))
rv_df = pd.DataFrame(rows, columns=["symbol", "n_bars", "iex_rv5m_pct", "parkinson_pct",
                                    "ratio"]).sort_values("ratio")
print(rv_df.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
print(f"\nmedian IEX_RV5m / Parkinson = {rv_df.ratio.median():.3f}  "
      f"(1.0 = unbiased; >1 = noise inflation, <1 = missed variation)")
cov.to_csv("_data_probe_iex_bincoverage.csv", index=False)
cmp_.to_csv("_data_probe_iex_marketshare.csv", index=False)
rv_df.to_csv("_data_probe_iex_rvbias.csv", index=False)
print("saved -> _data_probe_iex_bincoverage.csv / _marketshare.csv / _rvbias.csv")
