"""Probe: parse REAL IEX HIST TOPS files with the Python standard library only.

Two corrections to earlier attempts in this session:
  (a) the "smallest TOPS file" (242 B, 2017-11-13) is an operations stub, not market data;
  (b) IEX HIST uses BOTH capture formats across eras -- the stub was pcapNG
      (SHB magic 0x0A0D0D0A) while the 2016 TOPS 1.5 day is CLASSIC little-endian pcap
      (on-disk bytes d4 c3 b2 a1). A parser must sniff the magic, not assume one.

Decodes: gzip -> pcap/pcapng -> Ethernet/IPv4/UDP -> IEX-TP -> TOPS quote/trade messages.
Everything below uses only zlib/struct/socket from the stdlib.
"""
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


def iter_packets(buf):
    """Yield (ts_ns, packet_bytes) from either classic pcap or pcapng. Stdlib only."""
    if len(buf) < 24:
        return
    m_le = struct.unpack_from("<I", buf, 0)[0]
    if m_le in (0xA1B2C3D4, 0xA1B23C4D):            # classic pcap, little-endian
        nano = m_le == 0xA1B23C4D
        fmt = "<IIII"
        off = 24
        while off + 16 <= len(buf):
            ts_s, ts_f, incl, _orig = struct.unpack_from(fmt, buf, off)
            off += 16
            if off + incl > len(buf):
                return
            yield ts_s * 10 ** 9 + (ts_f if nano else ts_f * 1000), buf[off:off + incl]
            off += incl
        return
    if struct.unpack_from("<I", buf, 0)[0] == 0x0A0D0D0A:   # pcapng
        off = 0
        while off + 12 <= len(buf):
            btype, blen = struct.unpack_from("<II", buf, off)
            if blen < 12 or off + blen > len(buf):
                return
            if btype == 6:
                _if, tsh, tsl, caplen, _ol = struct.unpack_from("<IIIII", buf, off + 8)
                yield ((tsh << 32) | tsl) * 1000, buf[off + 28: off + 28 + caplen]
            off += blen
        return
    raise ValueError(f"unrecognised capture magic {buf[:4].hex()}")


sec("1. THE HIST INDEX, AND WHICH FILE WE TEST")
idx = requests.get("https://iextrading.com/api/1.0/hist", headers=UA, timeout=60).json()
rows = [{"date": d, "feed": e.get("feed"), "version": e.get("version"),
         "nb": int(e.get("size") or 0), "link": e.get("link")}
        for d, lst in idx.items() for e in lst]
df = pd.DataFrame(rows)
tops = df[(df.feed == "TOPS") & (df.nb > 1e8)].sort_values("date")
targets = [tops.iloc[0], tops[tops.date.str.startswith("2019")].iloc[0]]
for pk in targets:
    print(f"  {pk.date}  TOPS {pk.version}  {pk.nb/1e9:.3f} GB compressed")

for pk in targets:
    sec(f"2. FETCH + DECODE  {pk.date}  TOPS {pk.version}")
    N = 40 * 1024 * 1024
    t0 = time.time()
    r = requests.get(pk.link, headers={**UA, "Range": f"bytes=0-{N-1}"}, timeout=600)
    el = time.time() - t0
    raw = r.content
    mbps = len(raw) / 1e6 / el
    print(f"HTTP {r.status_code}  {len(raw)/1e6:.1f} MB in {el:.1f}s = {mbps:.1f} MB/s"
          f"   gzip={raw[:2].hex()}")
    dec = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(raw)
    print(f"decompressed {len(dec)/1e6:.1f} MB (ratio {len(dec)/len(raw):.2f}x)")
    magic = dec[:4].hex()
    kind = ("classic pcap LE" if struct.unpack_from("<I", dec, 0)[0]
            in (0xA1B2C3D4, 0xA1B23C4D) else "pcapng")
    print(f"capture magic on disk = {magic}  -> {kind}")
    if kind.startswith("classic"):
        vmaj, vmin, _tz, _sf, snap, link = struct.unpack_from("<HHiIII", dec, 4)
        print(f"pcap v{vmaj}.{vmin}  snaplen={snap}  linktype={link} (1=Ethernet)")

    pkts = []
    for ts, p in iter_packets(dec):
        pkts.append((ts, p))
        if len(pkts) >= 400000:
            break
    print(f"packets decoded from the prefix: {len(pkts):,}")
    ts0, p0 = pkts[0]
    print(f"first packet ts = {pd.to_datetime(ts0, unit='ns')}  len={len(p0)}")
    print(f"  eth_type=0x{struct.unpack_from('!H', p0, 12)[0]:04X} "
          f"ip_proto={p0[23]}  {socket.inet_ntoa(p0[26:30])}:"
          f"{struct.unpack_from('!H', p0, 34)[0]} -> "
          f"{socket.inet_ntoa(p0[30:34])}:{struct.unpack_from('!H', p0, 36)[0]}")
    (ver, _res, proto, chan, sess, plen, mcount, soff, seqn, sendt) = \
        struct.unpack("<BBHIIHHqqq", p0[42:82])
    print(f"  IEX-TP v{ver} msg_protocol=0x{proto:04X} channel={chan} session={sess} "
          f"payload={plen} msgs={mcount}")
    print(f"  send_time={pd.to_datetime(sendt, unit='ns')}")

    types = Counter()
    quotes, trades = [], []
    for ts, p in pkts:
        if len(p) < 82:
            continue
        mcount = struct.unpack_from("<H", p, 60)[0]
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
            mt = chr(body[0])
            types[mt] += 1
            # Quote Update: type,flags,ts(8),symbol(8),bidSize(4),bidPx(8),askPx(8),askSize(4)
            if mt == "Q" and len(body) >= 42:
                tstamp = struct.unpack_from("<q", body, 2)[0]
                sym = body[10:18].decode("latin1").strip()
                bsz, bpx, apx, asz = struct.unpack_from("<IqqI", body, 18)
                quotes.append((tstamp, sym, bsz, bpx / 1e4, apx / 1e4, asz))
            # Trade Report: type,flags,ts(8),symbol(8),size(4),px(8),tradeId(8)
            elif mt == "T" and len(body) >= 38:
                tstamp = struct.unpack_from("<q", body, 2)[0]
                sym = body[10:18].decode("latin1").strip()
                sz, px, tid = struct.unpack_from("<Iqq", body, 18)
                trades.append((tstamp, sym, sz, px / 1e4, tid))
    print(f"\n  message-type census: {dict(types.most_common(8))}")
    q = pd.DataFrame(quotes, columns=["ts", "symbol", "bid_sz", "bid_px", "ask_px", "ask_sz"])
    t = pd.DataFrame(trades, columns=["ts", "symbol", "size", "price", "trade_id"])
    print(f"  quote updates: {len(q):,}   trade reports: {len(t):,}")
    if len(q):
        q["ts"] = pd.to_datetime(q.ts, unit="ns")
        print("\n  sample QUOTES (top of book WITH SIZES = real depth/imbalance):")
        print(q[q.bid_px > 0].head(6).to_string(index=False))
        v = q[(q.bid_px > 0) & (q.ask_px > 0)].copy()
        v["imb"] = (v.bid_sz - v.ask_sz) / (v.bid_sz + v.ask_sz)
        v["spr_bps"] = (v.ask_px - v.bid_px) / ((v.ask_px + v.bid_px) / 2) * 1e4
        print(f"\n  two-sided quotes: {len(v):,}  distinct symbols: {v.symbol.nunique():,}")
        print(f"  quote imbalance  mean={v.imb.mean():+.4f} std={v.imb.std():.4f}")
        print(f"  quoted spread    median={v.spr_bps.median():.2f} bps "
              f"p95={v.spr_bps.quantile(.95):.2f} bps")
        top = v.symbol.value_counts().head(6)
        print(f"  busiest symbols in prefix: {top.to_dict()}")
    if len(t):
        t["ts"] = pd.to_datetime(t.ts, unit="ns")
        print("\n  sample TRADES:")
        print(t.head(5).to_string(index=False))
        print(f"  trade notional in prefix: ${(t['size']*t.price).sum()/1e6:,.2f} M "
              f"over {t.symbol.nunique():,} symbols, "
              f"{t.ts.min().time()} .. {t.ts.max().time()}")
        t.to_csv(f"_data_probe_iex_trades_{pk.date}.csv", index=False)
        q.head(50000).to_csv(f"_data_probe_iex_quotes_{pk.date}.csv", index=False)
        print(f"  saved -> _data_probe_iex_trades_{pk.date}.csv (+ quotes)")
    last_mbps = mbps

sec("3. WHAT A USEFUL PULL COSTS AT THE MEASURED RATE")
ta = df[df.feed == "TOPS"]
print(f"measured sustained rate from the GCS host: {last_mbps:.0f} MB/s\n")
for lbl, sub in [("2017 (0.54 GB/day)", ta[ta.date.str[:4] == "2017"]),
                 ("2019 (0.97 GB/day)", ta[ta.date.str[:4] == "2019"]),
                 ("2022 (4.67 GB/day)", ta[ta.date.str[:4] == "2022"]),
                 ("2025 (8.77 GB/day)", ta[ta.date.str[:4] == "2025"]),
                 ("TOPS 2016-2026 (all)", ta)]:
    gb = sub.nb.sum() / 1e9
    hrs = sub.nb.sum() / (last_mbps * 1e6) / 3600
    print(f"  {lbl:24s} {gb:>9,.1f} GB compressed -> {hrs:>7,.1f} h "
          f"({hrs/24:>5.1f} days) of pure download, "
          f"{gb*3.9:>9,.0f} GB decompressed to stream through")
print("\nNo server-side symbol filter exists: extracting one symbol's 5-minute bars still")
print("requires downloading and streaming the ENTIRE day for every day wanted.")
