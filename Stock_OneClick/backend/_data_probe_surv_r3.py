"""PROBE round 3 -- the crux questions.
  3.1 How far back does Alpha Vantage LISTING_STATUS `date=` reach?
  3.2 Does the AV FREE tier serve daily BARS for a DELISTED ticker? (roster w/o prices is useless)
  3.3 Yahoo: any route at all to delisted bars? (slow, respectful of the 429)
  3.4 Wayback nasdaqtrader roster snapshots -- extends the roster back to 2008 (retry the 504)
  3.5 GitHub "36,000+ delisted stocks" repo -- real data or landing page?
"""
import datetime as dtm
import io
import json
import os
import time

import pandas as pd
import requests

BROWSER = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
           "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
KEY = None
for ln in open("/Users/feijing/github.com/stock_scan/.env"):
    if ln.startswith("ALPHAVANTAGE_API_KEY"):
        KEY = ln.split("=", 1)[1].strip()

AV_CALLS = 0


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


A = requests.Session()
A.headers.update({"User-Agent": BROWSER})


def av(params, label):
    global AV_CALLS
    params["apikey"] = KEY
    AV_CALLS += 1
    t0 = time.time()
    r = A.get("https://www.alphavantage.co/query", params=params, timeout=180)
    print(f"  [{label}] call#{AV_CALLS} HTTP{r.status_code} {len(r.content):,}B "
          f"{time.time()-t0:.2f}s")
    return r


# ------------------------------------------------- 3.1 how deep does date= go?
hr("3.1 AV LISTING_STATUS date= depth  (roster history horizon)")
for d in ["2010-01-04", "2008-01-02", "2005-01-03"]:
    r = av({"function": "LISTING_STATUS", "state": "active", "date": d}, f"active@{d}")
    body = r.text
    if body.lstrip().startswith("{") or len(body) < 400:
        print(f"    RAW BODY ({len(body)}B): {body[:390]!r}")
        continue
    df = pd.read_csv(io.StringIO(body))
    print(f"    rows={len(df):,}  ipoDate max={df['ipoDate'].max()}  "
          f"exchanges={df['exchange'].value_counts().to_dict()}")
    print(f"    assetType={df['assetType'].value_counts().to_dict()}")
    print(f"    sample={df.head(3)[['symbol','name','exchange','ipoDate']].to_dict('records')}")
    df.to_csv(f"_data_probe_surv_av_active_{d}.csv", index=False)
    time.sleep(1.5)

# --------------------- 3.2 does AV free serve BARS for a DELISTED ticker?
hr("3.2 AV FREE daily bars for DELISTED tickers (outputsize=compact is the free tier)")
for sym in ["ATVI", "TWTR", "SIVB"]:
    r = av({"function": "TIME_SERIES_DAILY", "symbol": sym, "outputsize": "compact",
            "datatype": "csv"}, f"DAILY/{sym}")
    if r.text.lstrip().startswith("{"):
        print(f"    {r.text[:260]}")
    else:
        try:
            df = pd.read_csv(io.StringIO(r.text))
            if len(df):
                print(f"    ROWS={len(df)} cols={list(df.columns)} "
                      f"span={df.iloc[-1, 0]}..{df.iloc[0, 0]}")
                print(f"    last 3:\n{df.head(3).to_string(index=False)}")
            else:
                print("    EMPTY CSV")
        except Exception as e:
            print(f"    parse err {e}: {r.text[:200]}")
    time.sleep(1.5)
print(f"\n  TOTAL Alpha Vantage calls used by this script: {AV_CALLS}")

# ------------------------------------------ 3.3 Yahoo delisted, slowly
hr("3.3 Yahoo chart API on delisted symbols -- slow, 6s apart, after the 429 cooldown")
Y = requests.Session()
Y.headers.update({"User-Agent": BROWSER, "Accept": "application/json"})
for sym in ["AAPL", "ATVI", "TWTR", "SIVB", "SIVBQ", "CTXS", "ABMD"]:
    u = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
         f"?period1=0&period2=9999999999&interval=1d")
    try:
        r = Y.get(u, timeout=40)
        if r.status_code != 200:
            print(f"  {sym:8s} HTTP{r.status_code} {r.text[:90]!r}")
        else:
            j = r.json()
            res = (j.get("chart") or {}).get("result")
            if res and res[0].get("timestamp"):
                ts = res[0]["timestamp"]
                cl = [c for c in res[0]["indicators"]["quote"][0]["close"] if c is not None]
                print(f"  {sym:8s} HTTP200 BARS={len(ts):<6} "
                      f"{dtm.datetime.utcfromtimestamp(ts[0]).date()}.."
                      f"{dtm.datetime.utcfromtimestamp(ts[-1]).date()} "
                      f"lastClose={cl[-1] if cl else None}")
            else:
                e = (j.get("chart") or {}).get("error") or {}
                print(f"  {sym:8s} HTTP200 NO BARS err={e.get('code')}: "
                      f"{str(e.get('description'))[:80]}")
    except Exception as ex:
        print(f"  {sym:8s} EXC {type(ex).__name__}: {str(ex)[:90]}")
    time.sleep(6.0)

# --------------- 3.4 Wayback nasdaqtrader rosters (retry, one snapshot per year)
hr("3.4 Wayback nasdaqtrader nasdaqlisted.txt -- retry; roster WITH tickers back to 2008")
W = requests.Session()
W.headers.update({"User-Agent": BROWSER})
rows = []
for attempt in range(4):
    r = W.get("https://web.archive.org/cdx/search/cdx",
              params={"url": "www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
                      "output": "json", "fl": "timestamp,statuscode,length",
                      "collapse": "timestamp:4", "limit": "500"},
              timeout=240)
    print(f"  CDX attempt{attempt+1} HTTP{r.status_code} {len(r.content)}B")
    if r.status_code == 200 and r.text.strip().startswith("["):
        rows = [x for x in json.loads(r.text)[1:] if x[1] == "200"]
        break
    time.sleep(8)
print(f"  yearly 200 snapshots: {[x[0][:8] for x in rows]}")
CHK = ["SIVB", "ATVI", "SPLK", "CTXS", "ABMD", "CLVS", "TWTR", "AAPL"]
for row in rows:
    ts = row[0]
    u = (f"https://web.archive.org/web/{ts}id_/"
         f"https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt")
    try:
        rr = W.get(u, timeout=240)
        lines = [l for l in rr.text.splitlines() if "|" in l]
        syms = {l.split("|")[0].strip() for l in lines[1:]}
        print(f"  {ts[:8]} HTTP{rr.status_code} {len(rr.content):>8,}B rows={len(lines):>6,} "
              f"present={[t for t in CHK if t in syms]}")
    except Exception as e:
        print(f"  {ts[:8]} EXC {type(e).__name__}: {str(e)[:80]}")
    time.sleep(1.5)

# ---------------------------------------- 3.5 the "36,000 delisted" GitHub repo
hr("3.5 GitHub BlackFalconData-org/delisted-stocks-list -- real data?")
G = requests.Session()
G.headers.update({"User-Agent": BROWSER})
r = G.get("https://api.github.com/repos/BlackFalconData-org/delisted-stocks-list", timeout=60)
print(f"  repo meta HTTP{r.status_code}")
if r.status_code == 200:
    j = r.json()
    print(f"    size={j.get('size')}KB stars={j.get('stargazers_count')} "
          f"pushed={j.get('pushed_at')} desc={(j.get('description') or '')[:100]}")
    br = j.get("default_branch")
    rr = G.get(f"https://api.github.com/repos/BlackFalconData-org/delisted-stocks-list"
               f"/git/trees/{br}?recursive=1", timeout=60)
    if rr.status_code == 200:
        tree = rr.json().get("tree", [])
        print(f"    files={len(tree)}")
        for t in tree[:25]:
            print(f"      {t['path']:60s} {t.get('size','-')}")
    else:
        print(f"    tree HTTP{rr.status_code}")
else:
    print(f"    {r.text[:200]}")

print("\n  -- other GitHub delisting datasets by stars --")
for q in ["delisted stocks csv", "survivorship bias free dataset stocks",
          "ticker changes delisted dataset"]:
    r = G.get("https://api.github.com/search/repositories",
              params={"q": q, "sort": "stars", "per_page": 6}, timeout=60)
    print(f"  q={q!r} HTTP{r.status_code}")
    if r.status_code == 200:
        for it in r.json().get("items", []):
            print(f"    {it['stargazers_count']:6d}* {it['full_name']:48s} "
                  f"{(it['description'] or '')[:55]}")
    time.sleep(2)

print("\nDONE")
