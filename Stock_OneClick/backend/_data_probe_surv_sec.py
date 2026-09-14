"""PROBE 1+2: SEC EDGAR -- can we build a POINT-IN-TIME listed roster incl. delisted names, free?"""
import io
import json
import time
import zipfile

import requests

# SEC requires a descriptive UA with contact info. Anything else -> 403.
UA = {"User-Agent": "stock_scan research (feijing@users.noreply.github.com)",
      "Accept-Encoding": "gzip, deflate", "Host": None}
S = requests.Session()
S.headers.update({"User-Agent": "stock_scan research AdminContact@example.com",
                  "Accept-Encoding": "gzip, deflate"})


def get(url, **kw):
    t0 = time.time()
    r = S.get(url, timeout=120, **kw)
    print(f"  GET {url[:110]}\n    -> {r.status_code} {len(r.content):,}B {time.time()-t0:.2f}s")
    return r


DELISTED = {
    "SIVB": 719739, "ATVI": 718877, "VIAC": 813828, "SPLK": 1353283,
    "CTXS": 877890, "ABMD": 815094, "CLVS": 1466301,
}

print("=" * 78, "\n1. company_tickers.json (CURRENT roster)\n", "=" * 78)
r = get("https://www.sec.gov/files/company_tickers.json")
ct = r.json()
print(f"    rows={len(ct):,}  sample={list(ct.values())[:2]}")
cur = {v["ticker"] for v in ct.values()}
for t in DELISTED:
    print(f"    {t:5s} in current file? {t in cur}")

print("\n" + "=" * 78, "\n2. company_tickers_exchange.json\n", "=" * 78)
r = get("https://www.sec.gov/files/company_tickers_exchange.json")
cte = r.json()
print(f"    fields={cte['fields']} rows={len(cte['data']):,}")
ic = cte["fields"].index("exchange")
cnt = {}
for row in cte["data"]:
    cnt[row[ic]] = cnt.get(row[ic], 0) + 1
print(f"    exchange counts={cnt}")

print("\n" + "=" * 78, "\n3. submissions API for DELISTED CIKs: ticker retained? Form 25 present?\n", "=" * 78)
for tick, cik in DELISTED.items():
    r = S.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", timeout=120)
    if r.status_code != 200:
        print(f"  {tick}: HTTP {r.status_code}")
        continue
    j = r.json()
    rec = j["filings"]["recent"]
    f25 = [(f, d) for f, d in zip(rec["form"], rec["filingDate"]) if f.startswith("25")]
    print(f"  {tick:5s} cik={cik:<8} name={j['name'][:30]:30s} tickers={j.get('tickers')} "
          f"exch={j.get('exchanges')}")
    print(f"        n_recent={len(rec['form'])} latest_filing={rec['filingDate'][0]} "
          f"FORM-25s={f25[:4]}")
    time.sleep(0.12)

print("\n" + "=" * 78, "\n4. full-index company.idx = point-in-time ACTIVE FILER roster (CIK+name only)\n", "=" * 78)
for y, q in [("2005", "QTR1"), ("2015", "QTR1"), ("2024", "QTR1")]:
    r = get(f"https://www.sec.gov/Archives/edgar/full-index/{y}/{q}/company.idx")
    if r.status_code == 200:
        lines = r.text.splitlines()
        ciks = set()
        for ln in lines[10:]:
            parts = [p for p in ln.split("  ") if p.strip()]
            if len(parts) >= 3:
                c = parts[-3].strip()
                if c.isdigit():
                    ciks.add(int(c))
        print(f"    {y}{q}: lines={len(lines):,} uniqueCIK~{len(ciks):,}")
        print(f"    sample row: {lines[11][:110]!r}")
    time.sleep(0.2)

print("\n" + "=" * 78, "\n5. EDGAR FULL-TEXT SEARCH: Form 25 / 25-NSE = the DELISTING TRAIL\n", "=" * 78)
for forms, dr in [("25-NSE", ("2023-01-01", "2023-12-31")), ("25", ("2023-01-01", "2023-12-31"))]:
    u = (f"https://efts.sec.gov/LATEST/search-index?q=&forms={forms}"
         f"&dateRange=custom&startdt={dr[0]}&enddt={dr[1]}")
    r = get(u)
    try:
        j = r.json()
        tot = j.get("hits", {}).get("total", {}).get("value")
        hits = j.get("hits", {}).get("hits", [])
        print(f"    forms={forms} {dr[0]}..{dr[1]}  TOTAL HITS={tot}  returned={len(hits)}")
        for h in hits[:4]:
            src = h["_source"]
            print(f"      {src.get('file_date')} {src.get('display_names')} "
                  f"ciks={src.get('ciks')} adsh={h['_id'][:30]}")
    except Exception as e:
        print(f"    parse fail {e}: {r.text[:300]}")
    time.sleep(0.3)

# earliest coverage of full-text search
r = get("https://efts.sec.gov/LATEST/search-index?q=&forms=25-NSE"
        "&dateRange=custom&startdt=2001-01-01&enddt=2005-12-31")
try:
    j = r.json()
    print(f"    2001-2005 25-NSE total={j['hits']['total']['value']}")
except Exception as e:
    print("   ", e, r.text[:200])

print("\n" + "=" * 78, "\n6. Financial Statement Data Sets quarterly ZIP (bulk, keyless)\n", "=" * 78)
for u in ["https://www.sec.gov/files/dera/data/financial-statement-data-sets/2015q1.zip",
          "https://www.sec.gov/files/dera/data/financial-statement-data-sets/2024q1.zip"]:
    r = S.get(u, timeout=600)
    print(f"  GET {u[-12:]} -> {r.status_code} {len(r.content):,}B")
    if r.status_code == 200:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        print(f"    members={[(i.filename, i.file_size) for i in z.infolist()]}")
        with z.open("sub.txt") as fh:
            raw = fh.read().decode("utf-8", "replace").splitlines()
        cols = raw[0].split("\t")
        print(f"    sub.txt rows={len(raw)-1:,} ncols={len(cols)}")
        print(f"    cols={cols}")
        has_tick = [c for c in cols if "tick" in c.lower() or "symb" in c.lower()]
        print(f"    TICKER-LIKE COLUMNS: {has_tick}")
        d = dict(zip(cols, raw[1].split("\t")))
        print(f"    row1 subset={{cik:{d.get('cik')}, name:{d.get('name')}, "
              f"form:{d.get('form')}, period:{d.get('period')}}}")
    time.sleep(0.3)

print("\nDONE")
