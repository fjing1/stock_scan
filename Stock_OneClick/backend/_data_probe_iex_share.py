"""Probe: THE decisive IEX limitation test.

Stream one FULL IEX TOPS day (not a prefix), build 5-minute trade bars per symbol, and
measure:
  1. IEX's share of consolidated daily volume, per symbol (the "~2%" claim, measured)
  2. how many of the 78 regular-hours 5-minute bins have ZERO IEX trades
     -> this is what actually decides whether HAR-RV on IEX data is possible at all
  3. realized vol from IEX 5-min bars vs the day's Parkinson range from daily bars

Streaming, bounded memory, stdlib zlib + struct only.
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


def sec(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


sec("0. LOCATE THE FULL-DAY FILE")
idx = requests.get("https://iextrading.com/api/1.0/hist", headers=UA, timeout=60).json()
ent = [e for e in idx[DATE] if e["feed"] == "TOPS"][0]
total_nb = int(ent["size"])
print(f"{DATE} TOPS {ent['version']}  {total_nb/1e9:.3f} GB compressed")

sec("1. STREAM THE WHOLE DAY AND AGGREGATE TRADES")
trades_n = 0
sym_vol = Counter()          # shares
sym_notional = Counter()
bars = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, None, None]))  # sym->bin->[sh,notl,first,last]
types = Counter()
pkts = 0
t0 = time.time()
dob = zlib.decompressobj(16 + zlib.MAX_WBITS)
buf = b""
got = 0
MAGIC_CLASSIC = (0xA1B2C3D4, 0xA1B23C4D)
hdr_done = False
kind = None
nano = False

with requests.get(ent["link"], headers=UA, timeout=1800, stream=True) as r:
    r.raise_for_status()
    for chunk in r.iter_content(1 << 20):
        got += len(chunk)
        buf += dob.decompress(chunk)
        if not hdr_done and len(buf) >= 24:
            m = struct.unpack_from("<I", buf, 0)[0]
            if m in MAGIC_CLASSIC:
                kind, nano = "pcap", m == 0xA1B23C4D
                buf = buf[24:]
            elif m == 0x0A0D0D0A:
                kind = "pcapng"
            else:
                raise ValueError(buf[:4].hex())
            hdr_done = True
            print(f"  format = {kind}")
        # consume complete records
        while True:
            if kind == "pcap":
                if len(buf) < 16:
                    break
                ts_s, ts_f, incl, _o = struct.unpack_from("<IIII", buf, 0)
                if len(buf) < 16 + incl:
                    break
                p = buf[16:16 + incl]
                buf = buf[16 + incl:]
            else:
                if len(buf) < 12:
                    break
                bt, bl = struct.unpack_from("<II", buf, 0)
                if bl < 12 or len(buf) < bl:
                    break
                p = b""
                if bt == 6 and bl >= 28:
                    _i, _th, _tl, cl, _ol = struct.unpack_from("<IIIII", buf, 8)
                    p = buf[28:28 + cl]
                buf = buf[bl:]
                if not p:
                    continue
            pkts += 1
            if len(p) < 82:
                continue
            mc = struct.unpack_from("<H", p, 60)[0]
            o = 82
            for _ in range(mc):
                if o + 2 > len(p):
                    break
                ml = struct.unpack_from("<H", p, o)[0]
                o += 2
                body = p[o:o + ml]
                o += ml
                if not body:
                    continue
                mt = body[0]
                types[chr(mt)] += 1
                if mt == 0x54 and ml >= 38:      # 'T' trade report
                    tstamp = struct.unpack_from("<q", body, 2)[0]
                    sym = body[10:18].decode("latin1").rstrip()
                    sz, px, _tid = struct.unpack_from("<Iqq", body, 18)
                    px /= 1e4
                    trades_n += 1
                    sym_vol[sym] += sz
                    sym_notional[sym] += sz * px
                    b5 = tstamp // 300_000_000_000     # 5-minute bin index
                    e = bars[sym][b5]
                    e[0] += sz
                    e[1] += sz * px
                    if e[2] is None:
                        e[2] = px
                    e[3] = px
        if got % (100 << 20) < (1 << 20):
            print(f"  {got/1e9:.2f}/{total_nb/1e9:.2f} GB, {pkts:,} pkts, "
                  f"{trades_n:,} trades, {time.time()-t0:.0f}s", flush=True)
el = time.time() - t0
print(f"\nDONE: {got/1e9:.3f} GB downloaded+streamed in {el:.1f}s "
      f"({got/1e6/el:.1f} MB/s effective)")
print(f"packets={pkts:,}  messages={sum(types.values()):,}  trades={trades_n:,}")
print(f"message census: {dict(types.most_common())}")
print(f"distinct symbols traded on IEX: {len(sym_vol):,}")
print(f"total IEX shares={sum(sym_vol.values()):,}  "
      f"notional=${sum(sym_notional.values())/1e9:,.3f} B")

sec("2. IEX SHARE OF CONSOLIDATED VOLUME, PER SYMBOL (the '~2%' claim, measured)")
import yfinance as yf
names = [s for s, _ in sym_vol.most_common(400)]
probe = [s for s in ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "BAC", "GE", "F", "XOM", "JNJ",
                     "NVDA", "INTC", "CSCO", "PFE", "T", "VZ", "WFC", "KO"] if s in sym_vol]
yq = yf.download(probe, start="2016-12-09", end="2016-12-15", progress=False,
                 auto_adjust=False)
vol = yq["Volume"]
tgt = pd.Timestamp("2016-12-12")
rows = []
for s in probe:
    if s in vol.columns and tgt in vol.index and pd.notna(vol.loc[tgt, s]):
        cons = float(vol.loc[tgt, s])
        rows.append((s, sym_vol[s], cons, sym_vol[s] / cons if cons else np.nan))
cmp_ = pd.DataFrame(rows, columns=["symbol", "iex_shares", "consolidated", "iex_share"])
cmp_ = cmp_.sort_values("iex_share", ascending=False)
print(cmp_.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
print(f"\nMEDIAN IEX share of consolidated volume: {cmp_.iex_share.median():.4f} "
      f"({cmp_.iex_share.median()*100:.2f}%)")
print(f"min {cmp_.iex_share.min()*100:.2f}%   max {cmp_.iex_share.max()*100:.2f}%")

sec("3. 5-MINUTE BIN COVERAGE -- CAN YOU EVEN BUILD HAR-RV FROM THIS?")
# regular hours 14:30..21:00 UTC on 2016-12-12 -> 78 five-minute bins
day = pd.Timestamp("2016-12-12")
open_ns = int((day + pd.Timedelta(hours=14, minutes=30)).value)
bins_rth = [(open_ns + i * 300_000_000_000) // 300_000_000_000 for i in range(78)]
rows = []
for s in probe + [x for x, _ in sym_vol.most_common(40) if x not in probe]:
    if s not in bars:
        continue
    filled = sum(1 for b in bins_rth if b in bars[s] and bars[s][b][0] > 0)
    rows.append((s, sym_vol[s], filled, 78 - filled, filled / 78))
cov = pd.DataFrame(rows, columns=["symbol", "iex_shares", "bins_with_trade",
                                  "empty_bins", "coverage"]).sort_values("iex_shares",
                                                                        ascending=False)
print(cov.head(30).to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
print(f"\nacross these {len(cov)} names: median 5-min bin coverage = "
      f"{cov.coverage.median():.3f}, i.e. {(1-cov.coverage.median())*78:.0f} of 78 bins EMPTY")
allsyms = []
for s in list(sym_vol)[:4000]:
    filled = sum(1 for b in bins_rth if b in bars.get(s, {}) and bars[s][b][0] > 0)
    allsyms.append(filled / 78)
a = pd.Series(allsyms)
print(f"across ALL {len(a):,} IEX-traded symbols: mean coverage={a.mean():.3f} "
      f"median={a.median():.3f}  share with >90% bins filled={float((a>0.9).mean()):.3f}")

sec("4. REALIZED VOL FROM IEX 5-MIN BARS vs THE DAILY BAR")
yq2 = yf.download(probe, start="2016-12-09", end="2016-12-15", progress=False,
                 auto_adjust=False)
rows = []
for s in probe:
    b = bars.get(s, {})
    px = [(k, v[3]) for k, v in sorted(b.items()) if k in set(bins_rth) and v[0] > 0]
    if len(px) < 10:
        continue
    p = np.array([v for _, v in px])
    r5 = np.diff(np.log(p))
    rv = np.sqrt((r5 ** 2).sum())
    hi, lo = float(yq2["High"].loc[tgt, s]), float(yq2["Low"].loc[tgt, s])
    park = np.sqrt(np.log(hi / lo) ** 2 / (4 * np.log(2)))
    rows.append((s, len(p), rv * 100, park * 100, rv / park))
rv_df = pd.DataFrame(rows, columns=["symbol", "n_bars", "iex_rv5m_pct",
                                    "parkinson_pct", "ratio"])
print(rv_df.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
print(f"\nmedian IEX-RV / Parkinson ratio = {rv_df.ratio.median():.3f}")
print("A ratio far from 1.0 means the IEX-only 5-minute RV is a BIASED estimate of the")
print("true daily variation -- microstructure noise from a 2%-share venue with gappy bins.")
cov.to_csv("_data_probe_iex_bincoverage.csv", index=False)
cmp_.to_csv("_data_probe_iex_marketshare.csv", index=False)
print("\nsaved -> _data_probe_iex_bincoverage.csv, _data_probe_iex_marketshare.csv")
