"""Probe: Cboe free daily option statistics -- volume and put/call ratios.

Files discovered by scraping https://www.cboe.com/us/options/market_statistics/historical_data/
Host: cdn.cboe.com (keyless, no signup).

Measures: real min/max trade date per file (are they still updated?), row counts,
sample values, and whether a LIVE (current) put/call feed exists anywhere free.
"""
import io
import re

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) research-probe"}
B = "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios"
FILES = ["totalpc", "totalpcarchive", "equitypc", "equitypcarchive", "indexpc",
         "indexpcarchive", "etppc", "spxpc", "vixpc", "pcratioarchive"]


def sec(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def load(name):
    r = requests.get(f"{B}/{name}.csv", headers=UA, timeout=40)
    if r.status_code != 200:
        return r.status_code, None
    lines = r.text.splitlines()
    # find the real header row: the first line whose first field parses as a date
    # in a later row, or that starts with DATE/Trade_date
    hdr = None
    for i, ln in enumerate(lines[:8]):
        f0 = ln.split(",")[0].strip().lower()
        if f0 in ("date", "trade_date"):
            hdr = i
            break
    if hdr is None:
        # headerless (spxpc, vixpc): date,ratio,call,put,total
        df = pd.read_csv(io.StringIO(r.text), header=None,
                         names=["DATE", "PC_Ratio", "CALL", "PUT", "TOTAL"])
    else:
        df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])))
    df.columns = [c.strip() for c in df.columns]
    dc = df.columns[0]
    df[dc] = pd.to_datetime(df[dc].astype(str).str.strip(), errors="coerce",
                            format="mixed")
    df = df[df[dc].notna()].sort_values(dc)
    return 200, df


sec("1. CBOE PUT/CALL + VOLUME FILES: REAL DATE SPANS (is anything still live?)")
out = []
frames = {}
for n in FILES:
    code, df = load(n)
    if df is None:
        out.append((n, code, 0, "", "", ""))
        continue
    frames[n] = df
    dc = df.columns[0]
    out.append((n, code, len(df), df[dc].min().date(), df[dc].max().date(),
                ",".join(map(str, df.columns[:5]))))
summary = pd.DataFrame(out, columns=["file", "http", "rows", "first_date", "last_date", "cols"])
print(summary.to_string(index=False))

sec("2. SAMPLE VALUES -- last 5 rows of totalpc and equitypc")
for n in ("totalpc", "equitypc", "indexpc", "spxpc"):
    if n in frames:
        print(f"\n--- {n}")
        print(frames[n].tail(5).to_string(index=False))

sec("3. STITCHED TOTAL PUT/CALL SERIES (archive + current)")
if "totalpcarchive" in frames and "totalpc" in frames:
    a = frames["totalpcarchive"].copy()
    b = frames["totalpc"].copy()
    a.columns = ["date", "call", "put", "total", "pc"]
    b.columns = ["date", "call", "put", "total", "pc"]
    st = pd.concat([a, b]).drop_duplicates("date").sort_values("date")
    print(f"stitched rows={len(st):,}  span {st.date.min().date()} .. {st.date.max().date()}")
    print(f"pc ratio: mean={st.pc.mean():.3f} std={st.pc.std():.3f} "
          f"min={st.pc.min():.2f} max={st.pc.max():.2f}")
    print("gaps > 5 calendar days inside the series:")
    g = st.date.diff().dt.days
    print(st.loc[g > 5, "date"].dt.date.astype(str).tolist()[:20] or "  none")
    st.to_csv("_data_probe_cboe_pcratio_total.csv", index=False)
    print("saved -> _data_probe_cboe_pcratio_total.csv")

sec("4. IS THERE A LIVE (CURRENT) FREE CBOE PUT/CALL / VOLUME FEED?")
cands = [
    "https://www.cboe.com/us/options/market_statistics/daily/",
    "https://www.cboe.com/us/options/market_statistics/",
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
    "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json",
    "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=cone",
    "https://www.cboe.com/us/equities/market_statistics/",
]
for u in cands:
    try:
        r = requests.get(u, headers=UA, timeout=40, allow_redirects=True)
        note = ""
        if r.status_code == 200 and "csv" in r.headers.get("content-type", "") + u:
            first = r.text.splitlines()[:2]
            note = f" | first lines: {first}"
        elif r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
            note = f" | json keys: {list(r.json().keys())[:8]}"
        print(f"HTTP {r.status_code} {len(r.content):>9,}B  {u}")
        print(f"    final_url={r.url}{note}")
    except Exception as e:
        print(f"ERR  {type(e).__name__} {u}")

sec("5. SPX/VIX DELAYED OPTION-CHAIN JSON (volume + open interest per strike)")
u = "https://cdn.cboe.com/api/global/delayed_quotes/options/SPY.json"
r = requests.get(u, headers=UA, timeout=60)
print(f"GET {u} -> HTTP {r.status_code}, {len(r.content):,} bytes")
if r.status_code == 200:
    j = r.json()
    d = j.get("data", {})
    print(f"top-level keys: {list(j.keys())}  data keys: {list(d.keys())[:12]}")
    opts = d.get("options", [])
    print(f"option rows: {len(opts):,}")
    if opts:
        od = pd.DataFrame(opts)
        print(f"columns: {list(od.columns)}")
        print(od.head(3).to_string(index=False))
        for c in ("volume", "open_interest", "iv", "bid", "ask"):
            if c in od.columns:
                print(f"  {c}: sum/mean = {od[c].sum():,.0f} / {od[c].mean():,.4f}")
        od.to_csv("_data_probe_cboe_spy_chain.csv", index=False)
        print("saved -> _data_probe_cboe_spy_chain.csv")
        # derive a put/call VOLUME ratio for this single underlying
        od["kind"] = od.option.str[-9]  # ...YYMMDD[C|P]strike
        if set(od.kind.unique()) >= {"C", "P"}:
            cv = od.loc[od.kind == "C", "volume"].sum()
            pv = od.loc[od.kind == "P", "volume"].sum()
            print(f"  DERIVED SPY put/call volume ratio = {pv/cv:.4f} "
                  f"(put {pv:,.0f} / call {cv:,.0f})")
