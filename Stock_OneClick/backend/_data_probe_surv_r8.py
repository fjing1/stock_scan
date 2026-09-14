"""PROBE round 8 -- characterise the stockanalysis.com API's real limits.
  8.1 Why do CTXS/ABMD/VIAC 400? Wrong symbol form, or genuinely absent?
  8.2 What is the MAX history depth? (MAX/50Y returned 1y; 10Y returned 10y -- odd)
  8.3 The FULL delisted roster (50/page) -- find the pagination / API
  8.4 Confirm the Nasdaq Data Link free-key signup page is reachable
"""
import io
import json
import re
import time

import pandas as pd
import requests

BR = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
S = requests.Session()
S.headers.update({"User-Agent": BR})
API = "https://stockanalysis.com/api/symbol/s/{s}/history"


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


hr("8.1 the 400s: symbol-form problem or genuine absence?")
MISS = ["CTXS", "ABMD", "VIAC", "CLVS", "FRC", "SBNY", "AABA", "ACAS", "WCOM", "BSC"]
for sym in MISS:
    out = []
    # a) the human page -- does stockanalysis even have this symbol?
    for path in [f"https://stockanalysis.com/stocks/{sym.lower()}/",
                 f"https://stockanalysis.com/quote/otc/{sym.upper()}/"]:
        try:
            r = S.get(path, timeout=45, allow_redirects=True)
            title = re.search(r"<title>(.*?)</title>", r.text, re.S)
            out.append(f"{path.split('/')[3]}:{r.status_code}"
                       f"({(title.group(1)[:38] if title else '')})")
        except Exception as e:
            out.append(f"{path.split('/')[3]}:EXC")
        time.sleep(0.25)
    # b) API with lowercase / other range
    for lab, kw in [("lower", {"s": sym.lower()}), ("upper", {"s": sym.upper()})]:
        try:
            r = S.get(API.format(**kw), params={"range": "10Y", "period": "Daily"},
                      timeout=45)
            n = len((r.json().get("data") or [])) if r.status_code == 200 else 0
            out.append(f"api-{lab}:{r.status_code}/rows={n}")
        except Exception:
            out.append(f"api-{lab}:EXC")
        time.sleep(0.25)
    print(f"  {sym:6s} {'  '.join(out)}")

print("\n  -- stockanalysis search API: is the symbol known under another form? --")
for q in ["CTXS", "Citrix", "Abiomed", "ViacomCBS", "First Republic"]:
    for u in [f"https://stockanalysis.com/api/search?q={q}",
              f"https://stockanalysis.com/api/search/?q={q}"]:
        try:
            r = S.get(u, timeout=45)
            print(f"  q={q:16s} {u[26:40]:16s} HTTP{r.status_code} {r.text[:200]!r}")
            if r.status_code == 200:
                break
        except Exception as e:
            print(f"  q={q} EXC {str(e)[:60]}")
        time.sleep(0.3)

hr("8.2 MAX history depth per symbol")
for sym in ["AAPL", "SIVB", "ATVI"]:
    print(f"\n  {sym}:")
    for rng in ["1Y", "5Y", "10Y", "15Y", "20Y", "25Y", "40Y", "ALL", ""]:
        try:
            p = {"period": "Daily"}
            if rng:
                p["range"] = rng
            r = S.get(API.format(s=sym), params=p, timeout=60)
            if r.status_code == 200:
                d = r.json().get("data") or []
                print(f"    range={rng or '(none)':7s} rows={len(d):<6} "
                      f"{(d[-1]['t'] if d else '-')}..{(d[0]['t'] if d else '-')} "
                      f"{len(r.content)/1024:.0f}KB")
            else:
                print(f"    range={rng or '(none)':7s} HTTP{r.status_code} {r.text[:70]!r}")
        except Exception as e:
            print(f"    range={rng:7s} EXC {str(e)[:60]}")
        time.sleep(0.3)

print("\n  -- explicit period / from-to params? --")
for p in [{"range": "10Y", "period": "Weekly"}, {"range": "10Y", "period": "Monthly"},
          {"period": "Daily", "from": "1990-01-01", "to": "2026-01-01"},
          {"range": "10Y", "period": "Daily", "page": "2"}]:
    try:
        r = S.get(API.format(s="AAPL"), params=p, timeout=60)
        d = r.json().get("data") or [] if r.status_code == 200 else []
        print(f"    {p} -> HTTP{r.status_code} rows={len(d)} "
              f"{(d[-1]['t'] if d else '-')}..{(d[0]['t'] if d else '-')}")
    except Exception as e:
        print(f"    {p} EXC {str(e)[:70]}")
    time.sleep(0.4)

hr("8.3 the FULL delisted roster")
for u in ["https://stockanalysis.com/actions/delisted/__data.json",
          "https://stockanalysis.com/api/actions/delisted",
          "https://stockanalysis.com/actions/delisted/2015/__data.json",
          "https://stockanalysis.com/api/screener/a/f?m=delisted&s=desc&c=s,n,delisted"
          "&cn=500",
          "https://stockanalysis.com/actions/delisted/2015/?p=2"]:
    try:
        r = S.get(u, timeout=90)
        print(f"  {u[26:100]}\n    HTTP{r.status_code} {len(r.content):,}B "
              f"{r.text[:180]!r}")
        if r.status_code == 200 and "__data.json" in u:
            # count symbol-looking entries
            syms = re.findall(r'"([A-Z]{1,5}[A-Z0-9.\-]{0,4})"', r.text)
            print(f"    symbol-like tokens: {len(syms)} sample={syms[:15]}")
            open("_data_probe_surv_sa_delisted_data.json", "w").write(r.text)
    except Exception as e:
        print(f"  {u[26:70]} EXC {str(e)[:80]}")
    time.sleep(1.0)

print("\n  -- how many year pages exist, 50/page? sum a few years --")
tot = 0
for y in [2010, 2015, 2020, 2023, 2026]:
    try:
        r = S.get(f"https://stockanalysis.com/actions/delisted/{y}/", timeout=90)
        t = pd.read_html(io.StringIO(r.text))[0]
        # look for pagination hints in raw html
        pg = re.findall(r"(?:of|Showing)\s*([\d,]+)\s*(?:results|stocks|rows)", r.text, re.I)
        nav = re.findall(r'href="[^"]*delisted/%s[^"]*p=(\d+)' % y, r.text)
        print(f"  {y}: table rows={len(t)}  result-count hints={pg[:3]}  "
              f"page links={sorted(set(nav))[:8]}")
        tot += len(t)
    except Exception as e:
        print(f"  {y}: EXC {str(e)[:80]}")
    time.sleep(1.0)

hr("8.4 Nasdaq Data Link free-key signup reachable?")
for u in ["https://data.nasdaq.com/sign-up", "https://data.nasdaq.com/account/profile"]:
    try:
        r = S.get(u, timeout=60, allow_redirects=True)
        print(f"  {u} -> HTTP{r.status_code} {len(r.content):,}B "
              f"final={r.url}")
    except Exception as e:
        print(f"  {u} EXC {str(e)[:70]}")
    time.sleep(0.5)

print("\nDONE")
