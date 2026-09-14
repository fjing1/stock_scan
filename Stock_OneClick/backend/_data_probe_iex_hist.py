"""Probe: IEX Exchange free HIST files (TOPS / DEEP / DPLC pcap.gz).

Index: https://iextrading.com/api/1.0/hist  (keyless, no signup)
Files: hosted on Google Cloud Storage (www.googleapis.com), keyless.

Measures: date span, feeds per era, GB/day, total corpus size, and whether a real
byte-range download of an actual pcap.gz succeeds.
"""
import json
import time

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) research-probe"}


def sec(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


sec("1. HIST INDEX (full, keyless)")
t0 = time.time()
r = requests.get("https://iextrading.com/api/1.0/hist", headers=UA, timeout=60)
el = time.time() - t0
print(f"GET /api/1.0/hist -> HTTP {r.status_code}, {len(r.content):,} bytes, {el:.2f}s")
idx = r.json()
open("_data_probe_iex_hist_index.json", "w").write(json.dumps(idx)[:0] or "")  # placeholder
with open("_data_probe_iex_hist_index.json", "w") as f:
    json.dump(idx, f)
print(f"dates listed: {len(idx):,}")
dates = sorted(idx.keys())
print(f"span: {dates[0]} .. {dates[-1]}")

rows = []
for d, lst in idx.items():
    for e in lst:
        rows.append({"date": d, "feed": e.get("feed"), "version": e.get("version"),
                     "size_bytes": int(e.get("size") or 0), "link": e.get("link")})
df = pd.DataFrame(rows)
df["dt"] = pd.to_datetime(df.date, format="%Y%m%d")
df["year"] = df.dt.dt.year
print(f"\ntotal file entries: {len(df):,}")

sec("2. FEEDS AVAILABLE, AND GB/DAY BY FEED")
g = df.groupby("feed").agg(n_days=("date", "nunique"),
                           first=("date", "min"), last=("date", "max"),
                           mean_GB=("size_bytes", lambda s: s.mean() / 1e9),
                           max_GB=("size_bytes", lambda s: s.max() / 1e9),
                           total_GB=("size_bytes", lambda s: s.sum() / 1e9))
print(g.to_string(float_format=lambda x: f"{x:,.2f}"))
print(f"\nGRAND TOTAL corpus: {df.size_bytes.sum()/1e12:.2f} TB compressed")

sec("3. TOPS (the smallest/most usable feed) GB/DAY BY YEAR")
tops = df[df.feed == "TOPS"]
print(tops.groupby("year").agg(n_days=("date", "nunique"),
                               mean_GB=("size_bytes", lambda s: s.mean() / 1e9),
                               total_GB=("size_bytes", lambda s: s.sum() / 1e9)
                               ).to_string(float_format=lambda x: f"{x:,.3f}"))
print("\nmost recent 5 days, all feeds:")
print(df[df.date.isin(dates[-3:])][["date", "feed", "version", "size_bytes"]]
      .assign(GB=lambda x: x.size_bytes / 1e9)
      .drop(columns="size_bytes").to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

sec("4. DOES AN ACTUAL FILE DOWNLOAD WORK? (byte-range on the real GCS link)")
# pick the smallest TOPS file in the corpus so a full-file test is even conceivable
smallest = tops.nsmallest(1, "size_bytes").iloc[0]
print(f"smallest TOPS file: {smallest.date} {smallest.size_bytes/1e6:,.1f} MB")
t0 = time.time()
rr = requests.get(smallest.link, headers={**UA, "Range": "bytes=0-1048575"}, timeout=90)
el = time.time() - t0
print(f"ranged GET (first 1 MiB) -> HTTP {rr.status_code}, got {len(rr.content):,} bytes "
      f"in {el:.2f}s  content-range={rr.headers.get('content-range')}")
print(f"first 32 bytes (hex): {rr.content[:32].hex()}")
print(f"gzip magic 1f8b present: {rr.content[:2] == b'\\x1f\\x8b'}")

# measure real sustained throughput on a bigger range
t0 = time.time()
rr2 = requests.get(smallest.link, headers={**UA, "Range": "bytes=0-52428799"}, timeout=300)
el2 = time.time() - t0
mbps = len(rr2.content) / 1e6 / el2
print(f"ranged GET (first 50 MB) -> HTTP {rr2.status_code}, {len(rr2.content)/1e6:.1f} MB "
      f"in {el2:.1f}s = {mbps:.1f} MB/s")

sec("5. CAN WE PARSE IT WITH STDLIB ONLY? (gzip + pcap + IEX-TP binary)")
import gzip
import struct
raw = rr.content
try:
    dec = gzip.decompress(raw)
    print("full gzip.decompress OK")
except Exception as e:
    # streaming decompress of a truncated member
    import zlib
    dobj = zlib.decompressobj(16 + zlib.MAX_WBITS)
    dec = dobj.decompress(raw)
    print(f"streaming zlib decompress of truncated gz OK: {len(dec):,} bytes out of 1 MiB in "
          f"({type(e).__name__} on full decompress as expected for a truncated file)")
print(f"decompressed prefix {len(dec):,} bytes; first 24 hex: {dec[:24].hex()}")
# pcap global header: magic d4c3b2a1 (LE) or a1b2c3d4 (BE)
magic = dec[:4]
print(f"pcap global-header magic: {magic.hex()}  -> "
      f"{'little-endian pcap' if magic == b'\\xd4\\xc3\\xb2\\xa1' else 'other'}")
if len(dec) >= 24:
    vmaj, vmin, tz, sig, snaplen, link = struct.unpack("<HHiIII", dec[4:24])
    print(f"pcap v{vmaj}.{vmin} snaplen={snaplen} linktype={link} (1=Ethernet)")
    # first packet record header
    ts_s, ts_us, incl, orig = struct.unpack("<IIII", dec[24:40])
    print(f"first packet: ts={pd.to_datetime(ts_s, unit='s')} incl_len={incl} orig_len={orig}")
    pkt = dec[40:40 + incl]
    print(f"  eth/ip/udp header bytes: {pkt[:42].hex()}")
    # IEX-TP header sits after 14 eth + 20 ip + 8 udp = 42 bytes
    tp = pkt[42:42 + 40]
    if len(tp) >= 40:
        (ver, res, proto, ch, sess, plen, mcount, offs, seq, sendts) = struct.unpack(
            "<BBHIIHHqqq", tp[:40])
        print(f"  IEX-TP: ver={ver} msg_proto=0x{proto:04x} channel={ch} session={sess} "
              f"payload_len={plen} msg_count={mcount} send_time={pd.to_datetime(sendts, unit='ns')}")
        print("  -> IEX-TP framing decodes with struct alone; NO special library needed.")
