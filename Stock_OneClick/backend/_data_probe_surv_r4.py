"""PROBE round 4 -- close the remaining gaps.
  4.1 SHARADAR/TICKERS says premium:false -> does it work with no key? what key is needed?
  4.2 Sharadar SEP actual price (archived server-rendered Quandl pages + search)
  4.3 Tiingo free-tier terms + delisted coverage (docs)
  4.4 Alpha Vantage premium price tiers = cost of full history for delisted names
  4.5 Yahoo: one careful, slow delisted test
  4.6 THE ARITHMETIC: size of a delisted-inclusive universe and what it costs to download
"""
import io
import json
import re
import time

import pandas as pd
import requests

B = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
     "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
S = requests.Session()
S.headers.update({"User-Agent": B})


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def txt_of(html):
    t = re.sub(r"<script.*?</script>", " ", html, flags=re.S | re.I)
    t = re.sub(r"<style.*?</style>", " ", t, flags=re.S | re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t)


hr("4.1 SHARADAR/TICKERS (metadata says premium:false) -- keyless?")
for u in ["https://data.nasdaq.com/api/v3/datatables/SHARADAR/TICKERS.json?qopts.per_page=3",
          "https://data.nasdaq.com/api/v3/datatables/SHARADAR/TICKERS.csv?ticker=SIVB",
          "https://data.nasdaq.com/api/v3/datatables/SHARADAR/SEP.csv?ticker=SIVB&"
          "date.gte=2023-01-01"]:
    try:
        r = S.get(u, timeout=60)
        print(f"  {u[44:120]}\n    -> HTTP{r.status_code} {len(r.content)}B {r.text[:230]!r}")
    except Exception as e:
        print(f"    EXC {e}")
    time.sleep(0.4)

hr("4.2 Sharadar SEP price -- archived server-rendered pages + search engines")
W = requests.Session()
W.headers.update({"User-Agent": B})
cands = []
for target in ["www.quandl.com/databases/SFA", "www.quandl.com/databases/SEP",
               "data.nasdaq.com/databases/SFA/pricing",
               "data.nasdaq.com/databases/SEP/pricing"]:
    for attempt in range(3):
        try:
            r = W.get("https://web.archive.org/cdx/search/cdx",
                      params={"url": target, "output": "json",
                              "fl": "timestamp,original,length",
                              "filter": "statuscode:200", "limit": "200"}, timeout=180)
            if r.status_code == 200 and r.text.strip().startswith("["):
                rows = json.loads(r.text)[1:]
                rows.sort(key=lambda x: int(x[2] or 0), reverse=True)
                print(f"  CDX {target}: {len(rows)} snaps, largest="
                      f"{[(x[0][:8], x[2]) for x in rows[:4]]}")
                cands += [(x[0], x[1], int(x[2] or 0)) for x in rows[:3]]
                break
            print(f"  CDX {target} HTTP{r.status_code} retry{attempt}")
        except Exception as e:
            print(f"  CDX {target} EXC {str(e)[:70]}")
        time.sleep(6)
    time.sleep(2)

cands.sort(key=lambda x: -x[2])
seen = set()
for ts, orig, ln in cands[:8]:
    if (orig, ts[:6]) in seen:
        continue
    seen.add((orig, ts[:6]))
    u = f"https://web.archive.org/web/{ts}id_/{orig}"
    try:
        rr = W.get(u, timeout=180)
        t = txt_of(rr.text)
        hits = re.findall(r".{0,80}\$\s?[\d,]+(?:\.\d\d)?\s?(?:/|per\s)?\s?"
                          r"(?:mo|month|yr|year|annual)?.{0,50}", t, re.I)
        print(f"  {ts[:8]} {orig[-40:]:40s} HTTP{rr.status_code} raw={len(rr.content):,} "
              f"text={len(t):,} $hits={len(hits)}")
        for h in hits[:10]:
            print(f"      $ {h.strip()[:150]}")
    except Exception as e:
        print(f"  {ts[:8]} EXC {str(e)[:80]}")
    time.sleep(2)

print("\n  -- DuckDuckGo html search for the price --")
for q in ["Sharadar SEP Nasdaq Data Link price per month",
          "Sharadar Core US Equities Bundle price month subscription"]:
    try:
        r = S.get("https://html.duckduckgo.com/html/", params={"q": q}, timeout=60)
        t = txt_of(r.text)
        print(f"  q={q!r} HTTP{r.status_code} textlen={len(t):,}")
        for h in re.findall(r".{0,110}\$\s?\d[\d,]*(?:\.\d\d)?.{0,60}", t)[:12]:
            print(f"     $ {h.strip()[:180]}")
    except Exception as e:
        print(f"  EXC {str(e)[:90]}")
    time.sleep(3)

hr("4.3 Tiingo free tier + delisted coverage")
for u in ["https://api.tiingo.com/documentation/end-of-day",
          "https://www.tiingo.com/pricing",
          "https://api.tiingo.com/tiingo/utilities/search?query=SIVB"]:
    try:
        r = S.get(u, timeout=60)
        t = txt_of(r.text)
        print(f"  {u} HTTP{r.status_code} {len(r.content):,}B text={len(t):,}")
        print(f"    {t[:600]}")
    except Exception as e:
        print(f"  {u} EXC {str(e)[:80]}")
    time.sleep(1)

hr("4.4 Alpha Vantage premium tiers (cost of FULL history incl. delisted)")
try:
    r = S.get("https://www.alphavantage.co/premium/", timeout=60)
    t = txt_of(r.text)
    print(f"  HTTP{r.status_code} {len(r.content):,}B text={len(t):,}")
    for h in re.findall(r".{0,70}\$\s?\d[\d,]*(?:\.\d\d)?.{0,70}", t)[:20]:
        print(f"    $ {h.strip()[:170]}")
    for h in re.findall(r".{0,60}(?:requests? per minute|API requests|per day).{0,60}",
                        t, re.I)[:14]:
        print(f"    quota> {h.strip()[:150]}")
except Exception as e:
    print(f"  EXC {e}")

hr("4.5 Yahoo one careful delisted test")
Y = requests.Session()
Y.headers.update({"User-Agent": B, "Accept": "application/json"})
import datetime as dtm
for sym in ["TWTR", "ATVI", "AAPL"]:
    for attempt in range(3):
        r = Y.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
                  f"?period1=0&period2=9999999999&interval=1d", timeout=40)
        if r.status_code == 200:
            break
        print(f"  {sym} attempt{attempt+1} HTTP{r.status_code}; backing off 25s")
        time.sleep(25)
    if r.status_code != 200:
        print(f"  {sym:6s} FINAL HTTP{r.status_code} {r.text[:60]!r}")
        continue
    j = r.json()
    res = (j.get("chart") or {}).get("result")
    if res and res[0].get("timestamp"):
        ts = res[0]["timestamp"]
        print(f"  {sym:6s} BARS={len(ts)} "
              f"{dtm.datetime.utcfromtimestamp(ts[0]).date()}.."
              f"{dtm.datetime.utcfromtimestamp(ts[-1]).date()}")
    else:
        e = (j.get("chart") or {}).get("error") or {}
        print(f"  {sym:6s} NO BARS err={e.get('code')}: {str(e.get('description'))[:90]}")
    time.sleep(8)

hr("4.6 ARITHMETIC: how big is a delisted-inclusive daily universe?")
d = pd.read_csv("_data_probe_surv_av_delisted_all.csv")
a10 = pd.read_csv("_data_probe_surv_av_active_2010-01-04.csv")
a15 = pd.read_csv("_data_probe_surv_av_active_2015-01-02.csv")
anow = pd.read_csv("_data_probe_surv_av_active_now.csv")
d_st = d[d["assetType"] == "Stock"]
u = set(anow[anow.assetType == "Stock"]["symbol"]) | set(d_st["symbol"]) \
    | set(a10[a10.assetType == "Stock"]["symbol"]) | set(a15[a15.assetType == "Stock"]["symbol"])
print(f"  AV active-now stocks     : {len(set(anow[anow.assetType=='Stock']['symbol'])):,}")
print(f"  AV delisted stocks       : {len(set(d_st['symbol'])):,}")
print(f"  AV active@2010 stocks    : {len(set(a10[a10.assetType=='Stock']['symbol'])):,}")
print(f"  AV active@2015 stocks    : {len(set(a15[a15.assetType=='Stock']['symbol'])):,}")
print(f"  UNION (delisted-inclusive US common-stock universe 2010-2026) = {len(u):,}")
d2 = d_st.copy()
d2["dy"] = d2["delistingDate"].str[:4].astype(float)
post2010 = d2[d2["dy"] >= 2010]
print(f"  of the delisted stocks, {len(post2010):,} delisted in 2010 or later")
# bar arithmetic
avg_life_days = 252 * 8
print(f"\n  bar arithmetic for a 16-year (2010-2026) delisted-inclusive daily panel:")
print(f"    {len(u):,} symbols x ~{avg_life_days:,} avg trading days "
      f"= ~{len(u)*avg_life_days/1e6:.1f}M rows")
print(f"    at ~60 bytes/row CSV -> ~{len(u)*avg_life_days*60/1e9:.2f} GB raw CSV")
print(f"    Alpha Vantage FREE (25 req/day, 1 symbol/req, 100 bars only): "
      f"{len(u)/25/365:.1f} YEARS of wall clock, and only 100 bars each -> INFEASIBLE")
print(f"    a vendor at 100k req/day, 1 req/symbol full history: "
      f"{len(u)/100000:.2f} days of quota -> feasible in one afternoon")

print("\nDONE")
