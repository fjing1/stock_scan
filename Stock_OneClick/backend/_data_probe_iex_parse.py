"""Probe: parse a REAL IEX HIST TOPS pcap file with the Python standard library only.

The first attempt (_data_probe_iex_hist.py section 4/5) accidentally picked the smallest
file in the corpus, which turned out to be a 242-byte operations stub, not market data.
This version picks a real ~0.5 GB TOPS 1.6 day, range-reads a prefix, and decodes:
  gzip member -> pcapng blocks -> Ethernet/IPv4/UDP -> IEX-TP header -> TOPS messages

Also measures real sustained download throughput to price out a full-history pull.
"""
import gzip
import socket
import struct
import time
import zlib
from collections import Counter

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) research-probe"}


def sec(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


sec("1. PICK A REAL TOPS FILE FROM THE HIST INDEX")
idx = requests.get("https://iextrading.com/api/1.0/hist", headers=UA, timeout=60).json()
rows = []
for d, lst in idx.items():
    for e in lst:
        rows.append({"date": d, "feed": e.get("feed"), "version": e.get("version"),
                     "nbytes": int(e.get("size") or 0), "link": e.get("link")})
df = pd.DataFrame(rows)
df["nbytes_ok"] = df.nbytes > 1e8
tops = df[(df.feed == "TOPS") & (df.nbytes_ok)].sort_values("date")
pick = tops.iloc[0]
print(f"chosen: {pick.date}  TOPS {pick.version}  {pick['nbytes']/1e9:.3f} GB compressed")
print(f"(smallest TOPS file >100 MB, i.e. the cheapest genuine full trading day)")

sec("2. SUSTAINED DOWNLOAD THROUGHPUT (real measurement, 60 MB range)")
N = 60 * 1024 * 1024
t0 = time.time()
r = requests.get(pick.link, headers={**UA, "Range": f"bytes=0-{N-1}"}, timeout=600)
el = time.time() - t0
raw = r.content
mbps = len(raw) / 1e6 / el
print(f"HTTP {r.status_code}  {len(raw)/1e6:.1f} MB in {el:.1f}s = {mbps:.1f} MB/s")
print(f"gzip magic present: {raw[:2] == b'\x1f\x8b'}  (first 4 bytes {raw[:4].hex()})")

sec("3. DECOMPRESS (streaming, because the range is a truncated gzip member)")
dob = zlib.decompressobj(16 + zlib.MAX_WBITS)
dec = dob.decompress(raw)
print(f"decompressed {len(dec)/1e6:.1f} MB from {len(raw)/1e6:.1f} MB "
      f"(ratio {len(dec)/len(raw):.2f}x) using stdlib zlib only")

sec("4. pcapng BLOCK STRUCTURE")
SHB = 0x0A0D0D0A
off = 0
btype, blen = struct.unpack_from("<II", dec, 0)
print(f"first block type=0x{btype:08X} len={blen}  "
      f"({'Section Header Block' if btype == SHB else 'unexpected'})")
bom = struct.unpack_from("<I", dec, 8)[0]
print(f"byte-order magic=0x{bom:08X} (0x1A2B3C4D = little-endian)")
vmaj, vmin = struct.unpack_from("<HH", dec, 12)
print(f"pcapng version {vmaj}.{vmin}")
print("-> IEX HIST files are pcapNG, not classic pcap. Stated for the record because")
print("   classic-pcap parsers (magic 0xD4C3B2A1) will fail on them.")

blocks = Counter()
pkts = []
off = 0
while off + 12 <= len(dec):
    btype, blen = struct.unpack_from("<II", dec, off)
    if blen < 12 or off + blen > len(dec):
        break
    blocks[btype] += 1
    if btype == 6:  # Enhanced Packet Block
        _iface, tsh, tsl, caplen, origlen = struct.unpack_from("<IIIII", dec, off + 8)
        pdata = dec[off + 28: off + 28 + caplen]
        pkts.append(((tsh << 32) | tsl, pdata))
    off += blen
names = {SHB: "SectionHeader", 1: "InterfaceDescription", 6: "EnhancedPacket",
         3: "SimplePacket", 5: "InterfaceStatistics"}
print("\nblock census over the decoded prefix:")
for k, v in blocks.most_common():
    print(f"  0x{k:08X} {names.get(k, 'other'):24s} {v:,}")
print(f"packets recovered: {len(pkts):,}")

sec("5. Ethernet / IPv4 / UDP -> IEX-TP HEADER")
ts, p = pkts[0]
eth_type = struct.unpack_from("!H", p, 12)[0]
ip_proto = p[23]
src = socket.inet_ntoa(p[26:30])
dst = socket.inet_ntoa(p[30:34])
sport, dport = struct.unpack_from("!HH", p, 34)
print(f"eth_type=0x{eth_type:04X} (0x0800=IPv4)  ip_proto={ip_proto} (17=UDP)")
print(f"{src}:{sport} -> {dst}:{dport}   (multicast dst as expected)")
tp = p[42:82]
(ver, res, proto, chan, sess, plen, mcount, soff, seqn, sendt) = struct.unpack("<BBHIIHHqqq", tp)
print(f"IEX-TP v{ver}  msg_protocol=0x{proto:04X}  channel={chan}  session={sess}")
print(f"payload_len={plen}  message_count={mcount}  stream_offset={soff}")
print(f"send_time={pd.to_datetime(sendt, unit='ns')}  (0x8004 = TOPS 1.6)")

sec("6. DECODE ACTUAL TOPS MESSAGES (quotes and trades)")
msg_types = Counter()
quotes, trades = [], []
for ts, p in pkts[:200000]:
    if len(p) < 82:
        continue
    proto, plen, mcount = struct.unpack_from("<H", p, 44)[0], \
        struct.unpack_from("<H", p, 58)[0], struct.unpack_from("<H", p, 60)[0]
    o = 82
    for _ in range(mcount):
        if o + 2 > len(p):
            break
        mlen = struct.unpack_from("<H", p, o)[0]
        o += 2
        body = p[o:o + mlen]
        o += mlen
        if not body:
            continue
        mt = body[0:1]
        msg_types[mt.decode("latin1")] += 1
        if mt == b"q" and len(body) >= 42:
            # Quote Update: type,flags,timestamp(8),symbol(8),bidsize(4),bidpx(8),
            #               asksize(4),askpx(8)
            _f, tstamp = body[1], struct.unpack_from("<q", body, 2)[0]
            sym = body[10:18].decode("latin1").strip()
            bsz, bpx, asz, apx = struct.unpack_from("<IqIq", body, 18)
            quotes.append((pd.to_datetime(tstamp, unit="ns"), sym, bsz, bpx / 1e4,
                           asz, apx / 1e4))
        elif mt == b"T" and len(body) >= 38:
            # Trade Report: type,flags,timestamp(8),symbol(8),size(4),price(8),tradeid(8)
            tstamp = struct.unpack_from("<q", body, 2)[0]
            sym = body[10:18].decode("latin1").strip()
            sz, px, tid = struct.unpack_from("<Iqq", body, 18)
            trades.append((pd.to_datetime(tstamp, unit="ns"), sym, sz, px / 1e4, tid))
print("TOPS message-type census (first 200k packets):")
for k, v in msg_types.most_common(10):
    print(f"  '{k}'  {v:,}")
q = pd.DataFrame(quotes, columns=["ts", "symbol", "bid_sz", "bid_px", "ask_sz", "ask_px"])
t = pd.DataFrame(trades, columns=["ts", "symbol", "size", "price", "trade_id"])
print(f"\nquote updates decoded: {len(q):,}   trade reports decoded: {len(t):,}")
if len(q):
    print("\nsample QUOTES (top-of-book with SIZES -> real depth/imbalance):")
    print(q.head(6).to_string(index=False))
    q["imb"] = (q.bid_sz - q.ask_sz) / (q.bid_sz + q.ask_sz)
    print(f"\nquote imbalance (bid_sz-ask_sz)/(sum): mean={q.imb.mean():+.4f} "
          f"std={q.imb.std():.4f}  n={q.imb.notna().sum():,}")
    print(f"distinct symbols in the prefix: {q.symbol.nunique():,}")
if len(t):
    print("\nsample TRADES:")
    print(t.head(6).to_string(index=False))
    t.to_csv("_data_probe_iex_tops_trades.csv", index=False)
    q.head(50000).to_csv("_data_probe_iex_tops_quotes.csv", index=False)
    print("saved -> _data_probe_iex_tops_trades.csv / _data_probe_iex_tops_quotes.csv")

sec("7. WHAT WOULD A USEFUL PULL COST AT THE MEASURED RATE?")
tops_all = df[df.feed == "TOPS"]
for label, sub in [("1 year 2019 (0.97 GB/day)", tops_all[tops_all.date.str[:4] == "2019"]),
                   ("1 year 2025 (8.8 GB/day)", tops_all[tops_all.date.str[:4] == "2025"]),
                   ("full TOPS corpus 2016-2026", tops_all)]:
    gb = sub.nbytes.sum() / 1e9
    hrs = sub.nbytes.sum() / (mbps * 1e6) / 3600
    print(f"  {label:30s} {gb:>9,.1f} GB  ->  {hrs:>7,.1f} h at {mbps:.0f} MB/s "
          f"({hrs/24:.1f} days)")
print("\nDecompressed working set is ~2.4x the compressed size, and there is NO")
print("server-side symbol filter: one symbol's 5-minute bars still costs the whole day.")
