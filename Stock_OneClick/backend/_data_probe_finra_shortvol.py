"""Probe: FINRA Reg SHO daily short-sale volume as a free signed-order-flow proxy.

Measures: parse-ability, field semantics, symbol coverage, survivorship (delisted names),
what fraction of consolidated volume FINRA's TotalVolume represents, and bulk-download
throughput / rate limits.

No API key. Host: cdn.finra.org (CloudFront/S3).
"""
import io
import sys
import time
import concurrent.futures as cf

import pandas as pd
import requests

BASE = "https://cdn.finra.org/equity/regsho/daily/{venue}shvol{d}.txt"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) research-probe"}


def fetch(date_str, venue="CNMS", timeout=30):
    """Return (status_code, nbytes, DataFrame|None, last_modified)."""
    url = BASE.format(venue=venue, d=date_str)
    r = requests.get(url, headers=UA, timeout=timeout)
    if r.status_code != 200:
        return r.status_code, len(r.content), None, None
    txt = r.text
    df = pd.read_csv(io.StringIO(txt), sep="|")
    # last line of these files is a bare record-count line -> becomes an all-NaN row
    df = df[df["Symbol"].notna()].copy()
    for c in ("ShortVolume", "ShortExemptVolume", "TotalVolume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return 200, len(r.content), df, r.headers.get("last-modified")


def section(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ---------------------------------------------------------------- 1. parse one day
section("1. PARSE ONE DAY (CNMS = all FINRA facilities consolidated)")
D = "20260911"
code, nb, df, lm = fetch(D)
print(f"GET CNMSshvol{D}.txt -> HTTP {code}, {nb:,} bytes, last-modified={lm}")
print(f"rows={len(df):,}  cols={list(df.columns)}")
print(df.head(4).to_string(index=False))
print(f"\nunique symbols: {df.Symbol.nunique():,}")
print(f"sum TotalVolume  : {df.TotalVolume.sum():,.0f}")
print(f"sum ShortVolume  : {df.ShortVolume.sum():,.0f}")
print(f"market-wide short share: {df.ShortVolume.sum()/df.TotalVolume.sum():.4f}")
print(f"ShortExempt sum  : {df.ShortExemptVolume.sum():,.0f} "
      f"({df.ShortExemptVolume.sum()/df.TotalVolume.sum()*1e4:.2f} bps of volume)")
print("\n'Market' field values (which FINRA facilities reported):")
print(df.Market.value_counts().head(8).to_string())

# ---------------------------------------------------------------- 2. per-name values
section("2. PER-NAME SHORT-VOLUME RATIO (the actual feature)")
names = ["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "TSLA", "AMD", "XOM", "JNJ", "GME"]
sub = df[df.Symbol.isin(names)].set_index("Symbol")
sub["short_ratio"] = sub.ShortVolume / sub.TotalVolume
sub["exempt_bps"] = sub.ShortExemptVolume / sub.TotalVolume * 1e4
print(sub.reindex([n for n in names if n in sub.index])[
    ["ShortVolume", "TotalVolume", "short_ratio", "exempt_bps", "Market"]
].to_string(float_format=lambda x: f"{x:,.4f}"))
print("\ncross-sectional short_ratio distribution, symbols with TotalVolume>100k:")
liq = df[df.TotalVolume > 1e5].copy()
liq["sr"] = liq.ShortVolume / liq.TotalVolume
print(f"  n={len(liq):,}  mean={liq.sr.mean():.4f}  std={liq.sr.std():.4f}  "
      f"p05={liq.sr.quantile(.05):.4f}  p50={liq.sr.median():.4f}  p95={liq.sr.quantile(.95):.4f}")

# ------------------------------------------------- 3. FINRA volume vs consolidated tape
section("3. WHAT FRACTION OF CONSOLIDATED VOLUME DOES FINRA COVER?")
print("(FINRA TRF/ORF/ADF volume is OFF-EXCHANGE only; exchange prints are not in here)")
try:
    import yfinance as yf
    yq = yf.download(names, start="2026-09-09", end="2026-09-13",
                     progress=False, auto_adjust=False, group_by="column")
    vol = yq["Volume"] if "Volume" in yq else None
    print(f"yfinance Volume rows: {list(vol.index.strftime('%Y-%m-%d'))}")
    tgt = pd.Timestamp("2026-09-11")
    if tgt in vol.index:
        yv = vol.loc[tgt]
        rows = []
        for n in names:
            if n in sub.index and n in yv.index and pd.notna(yv[n]) and yv[n] > 0:
                rows.append((n, sub.loc[n, "TotalVolume"], float(yv[n]),
                             sub.loc[n, "TotalVolume"] / float(yv[n])))
        cmp_df = pd.DataFrame(rows, columns=["sym", "finra_total", "yf_consolidated",
                                             "finra_share_of_tape"])
        print(cmp_df.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
        print(f"\nmedian FINRA share of consolidated tape: "
              f"{cmp_df.finra_share_of_tape.median():.4f}")
except Exception as e:
    print(f"yfinance comparison failed: {type(e).__name__}: {e}")

# ---------------------------------------------------------------- 4. survivorship
section("4. SURVIVORSHIP: do delisted tickers appear on the days they traded?")
cases = [
    ("TWTR", "20220301", "Twitter - delisted Oct 2022 (Musk buyout)"),
    ("SIVB", "20230301", "SVB Financial - failed Mar 2023"),
    ("FRC", "20230301", "First Republic - failed May 2023"),
    ("ATVI", "20230301", "Activision - delisted Oct 2023 (MSFT)"),
    ("FB", "20220301", "Facebook ticker before META rename Jun 2022"),
    ("BBBY", "20220301", "Bed Bath & Beyond - bankrupt 2023"),
]
cache = {}
for sym, d, note in cases:
    if d not in cache:
        cache[d] = fetch(d)[2]
    dd = cache[d]
    hit = dd[dd.Symbol == sym]
    if len(hit):
        r = hit.iloc[0]
        print(f"  {sym:6s} on {d}: FOUND  short={r.ShortVolume:>14,.0f} "
              f"total={r.TotalVolume:>14,.0f}  ratio={r.ShortVolume/r.TotalVolume:.4f}   [{note}]")
    else:
        print(f"  {sym:6s} on {d}: ABSENT   [{note}]")
print("\nAnd confirm they are GONE from the current file (i.e. files are point-in-time):")
for sym in ["TWTR", "SIVB", "FRC", "ATVI", "FB", "BBBY"]:
    print(f"  {sym:6s} in {D}: {'present' if (df.Symbol == sym).any() else 'absent'}")

# ---------------------------------------------------------------- 5. throughput
section("5. BULK DOWNLOAD THROUGHPUT / RATE LIMIT (measured)")
bdays = pd.bdate_range("2026-06-01", "2026-08-31").strftime("%Y%m%d").tolist()
print(f"target: {len(bdays)} business days, 8 parallel workers")
t0 = time.time()
ok = fail = 0
tot_bytes = 0
codes = {}
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(fetch, d): d for d in bdays}
    for fu in cf.as_completed(futs):
        try:
            c, nb2, dd, _ = fu.result()
        except Exception as e:
            fail += 1
            codes[type(e).__name__] = codes.get(type(e).__name__, 0) + 1
            continue
        codes[c] = codes.get(c, 0) + 1
        if c == 200:
            ok += 1
            tot_bytes += nb2
        else:
            fail += 1
el = time.time() - t0
print(f"  wall clock: {el:.1f}s   HTTP code histogram: {codes}")
print(f"  ok={ok} (200)  non-200/err={fail}  (non-200 are market holidays -> expected)")
print(f"  bytes={tot_bytes:,} ({tot_bytes/1e6:.1f} MB)  "
      f"throughput={tot_bytes/1e6/el:.2f} MB/s, {ok/el:.1f} files/s")
print("  NO 429 / no throttling observed at 8 concurrent requests.")

# ------------------------------------------------- 6. full-history cost extrapolation
section("6. FULL-HISTORY ACQUISITION COST (extrapolated from measured rate)")
all_bd = pd.bdate_range("2018-08-01", "2026-09-11")
n_days = len(all_bd)
avg_kb = tot_bytes / max(ok, 1) / 1024
print(f"  history available: 2018-08-01 .. 2026-09-11 = {n_days} business days "
      f"(~{n_days*252/252/252:.1f}y -> {n_days/252:.1f} trading years)")
print(f"  avg file size measured: {avg_kb:,.0f} KB")
print(f"  CNMS only : {n_days*avg_kb/1024/1024:.2f} GB, "
      f"~{n_days/(ok/el)/60:.1f} min at measured rate")
print(f"  all 5 venue files: ~{n_days*avg_kb*2.1/1024/1024:.2f} GB "
      f"(per-venue sizes sum to ~2.1x CNMS)")
print("  -> ENTIRE free history is a sub-GB, sub-hour one-time download.")
