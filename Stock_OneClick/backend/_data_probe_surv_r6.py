"""PROBE round 6 -- extract real data from the two live leads, plus the last pricing gap.
  6.1 stockanalysis.com: delisted daily OHLC history -- find the data endpoint, count rows
  6.2 stockanalysis.com delisted-stock ROSTER -- how many names, is it downloadable
  6.3 Tiingo free-tier terms (bot-UA page text)
  6.4 Wikipedia S&P 500 constituent changes (proper UA)
  6.5 Sharadar price -- last attempts
"""
import io
import json
import re
import time

import pandas as pd
import requests

BOT = "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
BR = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def txt_of(h):
    t = re.sub(r"<script.*?</script>", " ", h, flags=re.S | re.I)
    t = re.sub(r"<style.*?</style>", " ", t, flags=re.S | re.I)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", t))


S = requests.Session()
S.headers.update({"User-Agent": BR})

hr("6.1 stockanalysis.com -- delisted OHLC history endpoints")
# the page embeds SvelteKit data; look for the __data.json route + any api path
r = S.get("https://stockanalysis.com/stocks/sivb/history/", timeout=90)
print(f"  page HTTP{r.status_code} {len(r.content):,}B")
paths = sorted(set(re.findall(r'["\'](/api/[^"\']{4,120})["\']', r.text)))
print(f"  /api/ paths found in page: {paths[:20]}")
dj = sorted(set(re.findall(r'["\']([^"\']*__data\.json[^"\']*)["\']', r.text)))
print(f"  __data.json refs: {dj[:6]}")
# quick look for an inline JSON blob of prices
m = re.findall(r'\\"(?:t|date)\\":\s?"?\d{4}-\d\d-\d\d', r.text)[:3]
print(f"  inline date-looking keys: {m}")

CANDS = [
    "https://stockanalysis.com/stocks/sivb/history/__data.json",
    "https://stockanalysis.com/api/symbol/s/SIVB/history?range=10Y&period=Daily",
    "https://stockanalysis.com/api/charts/s/SIVB/max",
    "https://stockanalysis.com/api/charts/s/sivb/max",
    "https://stockanalysis.com/api/quotes/s/sivb/history",
]
for u in CANDS:
    try:
        rr = S.get(u, timeout=90)
        head = rr.text[:250].replace("\n", " ")
        print(f"  {u[26:]:58s} HTTP{rr.status_code} {len(rr.content):>9,}B {head[:150]!r}")
        if rr.status_code == 200 and len(rr.content) > 2000:
            open("_data_probe_surv_sa_sivb.json", "w").write(rr.text)
            print("      saved _data_probe_surv_sa_sivb.json")
    except Exception as e:
        print(f"  {u[26:]:58s} EXC {str(e)[:70]}")
    time.sleep(1.0)

print("\n  -- pandas.read_html on the visible history table --")
try:
    tabs = pd.read_html(io.StringIO(r.text))
    print(f"  tables={len(tabs)}")
    for i, t in enumerate(tabs[:4]):
        print(f"   table{i}: shape={t.shape} cols={list(t.columns)[:8]}")
        print(t.head(4).to_string()[:600])
except Exception as e:
    print(f"  EXC {type(e).__name__}: {str(e)[:120]}")

hr("6.2 stockanalysis.com delisted ROSTER")
r2 = S.get("https://stockanalysis.com/actions/delisted/", timeout=90)
print(f"  HTTP{r2.status_code} {len(r2.content):,}B")
try:
    tabs = pd.read_html(io.StringIO(r2.text))
    print(f"  tables={len(tabs)}")
    for i, t in enumerate(tabs[:3]):
        print(f"   table{i}: shape={t.shape} cols={list(t.columns)}")
        print(t.head(8).to_string()[:900])
        t.to_csv(f"_data_probe_surv_sa_delisted_{i}.csv", index=False)
except Exception as e:
    print(f"  read_html EXC {type(e).__name__}: {str(e)[:120]}")
t2 = txt_of(r2.text)
for h in re.findall(r".{0,70}(?:delisted stocks|total of|showing)\s?[\d,]+.{0,60}", t2, re.I)[:8]:
    print(f"    count-hint> {h.strip()[:160]}")

hr("6.3 Tiingo free-tier terms")
B = requests.Session()
B.headers.update({"User-Agent": BOT})
for u in ["https://www.tiingo.com/pricing", "https://www.tiingo.com/documentation/end-of-day",
          "https://www.tiingo.com/documentation/general/overview"]:
    try:
        rr = B.get(u, timeout=90)
        t = txt_of(rr.text)
        print(f"\n  {u} HTTP{rr.status_code} text={len(t):,}")
        for pat in [r".{0,80}(?:per hour|per day|per month|hourly limit|daily limit)"
                    r".{0,80}",
                    r".{0,70}unique symbol.{0,80}",
                    r".{0,70}delist.{0,90}",
                    r".{0,50}\$\s?\d[\d,]*(?:\.\d\d)?\s?/?\s?month.{0,60}"]:
            for h in re.findall(pat, t, re.I)[:6]:
                print(f"     > {h.strip()[:190]}")
    except Exception as e:
        print(f"  {u} EXC {str(e)[:80]}")
    time.sleep(1.5)

hr("6.4 Wikipedia S&P 500 constituent changes (proper UA)")
try:
    rw = S.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", timeout=90)
    print(f"  HTTP{rw.status_code} {len(rw.content):,}B")
    tabs = pd.read_html(io.StringIO(rw.text))
    print(f"  tables={len(tabs)}")
    chg = None
    for t in tabs:
        cols = " ".join(str(c) for c in t.columns)
        if "Removed" in cols and "Added" in cols:
            chg = t
            break
    if chg is not None:
        chg.columns = ["_".join(str(x) for x in c) if isinstance(c, tuple) else str(c)
                       for c in chg.columns]
        print(f"  CHANGES rows={len(chg)} cols={list(chg.columns)}")
        print(chg.head(5).to_string())
        dc = [c for c in chg.columns if "Date" in c][0]
        yr = pd.to_datetime(chg[dc], errors="coerce").dt.year.dropna()
        print(f"  span {int(yr.min())}..{int(yr.max())} n_dated={len(yr)}")
        rc = [c for c in chg.columns if "Removed" in c and "Ticker" in c]
        if rc:
            rem = chg[rc[0]].dropna().astype(str)
            print(f"  distinct removed tickers={rem.nunique()}")
            chk = ["SIVB", "ATVI", "SPLK", "CTXS", "ABMD", "TWTR", "FRC", "SBNY", "VIAC",
                   "CLVS"]
            print(f"  our delistings in Removed: {[c for c in chk if c in set(rem)]}")
        chg.to_csv("_data_probe_surv_sp500_changes.csv", index=False)
        print("  saved _data_probe_surv_sp500_changes.csv")
    else:
        print(f"  no changes table; shapes={[t.shape for t in tabs[:6]]}")
except Exception as e:
    print(f"  EXC {type(e).__name__}: {str(e)[:150]}")

hr("6.5 Sharadar price -- last attempts")
for u in ["https://lite.duckduckgo.com/lite/?q=Sharadar+SEP+%22%2Fmonth%22+nasdaq+data+link",
          "https://html.duckduckgo.com/html/?q=%22Sharadar%22+SEP+subscription+price+month"]:
    try:
        rr = S.get(u, timeout=90)
        t = txt_of(rr.text)
        print(f"  {u[:70]} HTTP{rr.status_code} text={len(t):,}")
        hits = re.findall(r".{0,110}\$\s?\d[\d,]*(?:\.\d\d)?.{0,70}", t)
        for h in hits[:14]:
            print(f"     $ {h.strip()[:190]}")
        if not hits:
            print(f"     sample: {t[:400]}")
    except Exception as e:
        print(f"  EXC {str(e)[:90]}")
    time.sleep(3)

print("\nDONE")
