"""PROBE: reconstruct a POINT-IN-TIME ticker roster (incl. names later delisted) for free.

Routes tested:
  A. Wayback CDX index -> historical snapshots of SEC company_tickers.json
  B. Wayback CDX index -> historical snapshots of nasdaqtrader symbol directories
  C. Live nasdaqtrader nasdaqlisted.txt / otherlisted.txt (current, with ETF/test flags)
  D. SEC Form 25-NSE volume by year = the delisting trail; does the filing name the ticker?
  E. Public GitHub delisting / ticker-change datasets
"""
import io
import json
import re
import time
from collections import defaultdict

import requests

UA_BROWSER = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
SEC_UA = "stock_scan research AdminContact@example.com"

W = requests.Session()
W.headers.update({"User-Agent": UA_BROWSER})
SEC = requests.Session()
SEC.headers.update({"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate"})


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def cdx(url, **extra):
    p = {"url": url, "output": "json", "fl": "timestamp,statuscode,digest,length",
         "collapse": "timestamp:6", "limit": "3000"}
    p.update(extra)
    r = W.get("https://web.archive.org/cdx/search/cdx", params=p, timeout=180)
    return r


# ------------------------------------------------------ A. SEC ticker file history
hr("A. Wayback CDX: snapshots of SEC company_tickers.json (point-in-time ticker->CIK)")
for target in ["https://www.sec.gov/files/company_tickers.json",
               "https://www.sec.gov/files/company_tickers_exchange.json"]:
    r = cdx(target)
    print(f"  CDX {target}\n    -> HTTP{r.status_code} {len(r.content):,}B")
    if r.status_code == 200 and r.text.strip():
        rows = json.loads(r.text)
        hdr, data = rows[0], rows[1:]
        ok = [d for d in data if d[1] == "200"]
        years = sorted({d[0][:4] for d in ok})
        print(f"    snapshots={len(data)} status200={len(ok)} years={years}")
        if ok:
            print(f"    earliest={ok[0][0]} latest={ok[-1][0]}")
            byyear = defaultdict(int)
            for d in ok:
                byyear[d[0][:4]] += 1
            print(f"    per-year 200s={dict(sorted(byyear.items()))}")
    time.sleep(1.0)

hr("A2. Fetch 3 dated SEC ticker snapshots and check delisted names ARE present")
DEL = ["SIVB", "ATVI", "VIAC", "SPLK", "CTXS", "ABMD", "CLVS", "TWTR", "FRC", "SBNY"]
r = cdx("https://www.sec.gov/files/company_tickers.json", collapse="timestamp:4")
snaps = [d[0] for d in json.loads(r.text)[1:] if d[1] == "200"] if r.status_code == 200 else []
print(f"  yearly-collapsed 200 snapshots: {snaps}")
for ts in snaps:
    u = f"https://web.archive.org/web/{ts}id_/https://www.sec.gov/files/company_tickers.json"
    try:
        rr = W.get(u, timeout=180)
        j = json.loads(rr.text)
        ticks = {v["ticker"] for v in j.values()}
        found = [t for t in DEL if t in ticks]
        print(f"  {ts}  HTTP{rr.status_code} rows={len(j):,}  delisted-present={found}")
    except Exception as e:
        print(f"  {ts}  EXC {type(e).__name__}: {str(e)[:90]}")
    time.sleep(1.0)

# --------------------------------------- B/C. nasdaqtrader symbol directories
hr("C. LIVE nasdaqtrader symbol directories (free, no key)")
for name, u in [("nasdaqlisted", "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"),
                ("otherlisted", "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"),
                ("mfundslist", "https://www.nasdaqtrader.com/dynamic/SymDir/mfundslist.txt")]:
    try:
        r = W.get(u, timeout=90)
        lines = r.text.splitlines()
        print(f"  {name:14s} HTTP{r.status_code} {len(r.content):,}B lines={len(lines):,}")
        if r.status_code == 200 and lines:
            print(f"    header={lines[0][:150]!r}")
            print(f"    row1  ={lines[1][:150]!r}")
            print(f"    last  ={lines[-1][:80]!r}")
    except Exception as e:
        print(f"  {name} EXC {e}")
    time.sleep(0.5)

hr("B. Wayback CDX: history of nasdaqlisted.txt / otherlisted.txt "
   "(= point-in-time ticker+name+exchange rosters)")
for u in ["https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
          "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
          "ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"]:
    r = cdx(u)
    print(f"  CDX {u}\n    -> HTTP{r.status_code} {len(r.content):,}B")
    if r.status_code == 200 and r.text.strip():
        data = json.loads(r.text)[1:]
        ok = [d for d in data if d[1] == "200"]
        byyear = defaultdict(int)
        for d in ok:
            byyear[d[0][:4]] += 1
        print(f"    snapshots={len(data)} status200={len(ok)} "
              f"span={ok[0][0] if ok else '-'}..{ok[-1][0] if ok else '-'}")
        print(f"    per-year={dict(sorted(byyear.items()))}")
    time.sleep(1.0)

hr("B2. Pull dated nasdaqlisted.txt snapshots; are delisted tickers in them?")
r = cdx("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", collapse="timestamp:4")
snaps = [d[0] for d in json.loads(r.text)[1:] if d[1] == "200"] if r.status_code == 200 else []
print(f"  yearly snapshots: {snaps}")
for ts in snaps:
    u = f"https://web.archive.org/web/{ts}id_/https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    try:
        rr = W.get(u, timeout=180)
        lines = [l for l in rr.text.splitlines() if l.strip()]
        syms = {l.split("|")[0] for l in lines[1:] if "|" in l}
        found = [t for t in ["SIVB", "ATVI", "SPLK", "CTXS", "ABMD", "CLVS", "TWTR"] if t in syms]
        print(f"  {ts} HTTP{rr.status_code} rows={len(lines):,} delisted-present={found}")
    except Exception as e:
        print(f"  {ts} EXC {type(e).__name__}: {str(e)[:90]}")
    time.sleep(1.0)

# ------------------------------------------------- D. delisting trail by year
hr("D. SEC Form 25-NSE count by year (delisting trail depth)")
for y in range(2001, 2027):
    u = (f"https://efts.sec.gov/LATEST/search-index?q=&forms=25-NSE"
         f"&dateRange=custom&startdt={y}-01-01&enddt={y}-12-31")
    try:
        r = SEC.get(u, timeout=60)
        n = r.json().get("hits", {}).get("total", {}).get("value") if r.status_code == 200 else None
        print(f"  {y}: 25-NSE filings = {n}")
    except Exception as e:
        print(f"  {y}: EXC {e}")
    time.sleep(0.25)

hr("D2. Does the 25-NSE primary document name the TICKER? (SIVB 2023-05-02)")
u = ("https://efts.sec.gov/LATEST/search-index?q=%22SVB+Financial%22&forms=25-NSE")
r = SEC.get(u, timeout=60)
print(f"  search HTTP{r.status_code}")
try:
    hits = r.json()["hits"]["hits"]
    for h in hits[:2]:
        _id = h["_id"]
        adsh, doc = _id.split(":")
        cik = h["_source"]["ciks"][0]
        acc = adsh.replace("-", "")
        du = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}"
        print(f"  filing {adsh} date={h['_source']['file_date']} doc={doc}")
        rr = SEC.get(du, timeout=60)
        txt = re.sub(r"<[^>]+>", " ", rr.text)
        txt = re.sub(r"\s+", " ", txt)
        print(f"    doc HTTP{rr.status_code} {len(rr.content):,}B")
        print(f"    text[:900]={txt[:900]}")
        time.sleep(0.3)
except Exception as e:
    print(f"  EXC {e} {r.text[:200]}")

# --------------------------------------------------- E. GitHub public datasets
hr("E. Public GitHub ticker/delisting datasets")
GH = [
    ("rreichel3/US-Stock-Symbols all_tickers",
     "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"),
    ("shilewenuw/get_all_tickers",
     "https://raw.githubusercontent.com/shilewenuw/get_all_tickers/master/get_all_tickers/tickers.csv"),
    ("Wilson-Song/delisted (search)", None),
]
for name, u in GH:
    if not u:
        continue
    try:
        r = W.get(u, timeout=60)
        lines = r.text.splitlines()
        print(f"  {name}: HTTP{r.status_code} {len(r.content):,}B lines={len(lines):,} "
              f"sample={lines[:4]}")
        syms = set(x.strip().upper() for x in lines)
        print(f"    delisted present? "
              f"{[t for t in ['SIVB','ATVI','SPLK','CTXS','ABMD','CLVS'] if t in syms]}")
    except Exception as e:
        print(f"  {name}: EXC {e}")
    time.sleep(0.4)

print("\n  GitHub code search for delisting datasets:")
for q in ["delisted+tickers+csv+in:path+extension:csv",
          "topic:delisted-stocks"]:
    try:
        r = W.get(f"https://api.github.com/search/repositories?q=delisted+stocks+dataset"
                  f"&sort=stars&per_page=10", timeout=60)
        if r.status_code == 200:
            for it in r.json().get("items", [])[:10]:
                print(f"    {it['stargazers_count']:5d}* {it['full_name']:45s} "
                      f"{(it['description'] or '')[:60]}")
        else:
            print(f"    HTTP{r.status_code} {r.text[:150]}")
        break
    except Exception as e:
        print(f"    EXC {e}")

print("\nDONE")
