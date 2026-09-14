"""PROBE round 5 -- final gaps.
  5.1 Sharadar/Tiingo SPA pages via a bot UA (these sites use prerender for crawlers)
  5.2 yfinance library (curl_cffi browser impersonation) on delisted symbols -- different
      TLS fingerprint than plain requests, may get past the 429
  5.3 Wikipedia S&P 500 historical constituent-change table = free point-in-time index membership
  5.4 stockanalysis.com delisted-stocks list
"""
import io
import json
import re
import time

import pandas as pd
import requests

BOT = "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
BROWSER = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
           "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def txt_of(h):
    t = re.sub(r"<script.*?</script>", " ", h, flags=re.S | re.I)
    t = re.sub(r"<style.*?</style>", " ", t, flags=re.S | re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t)


hr("5.1 SPA pages via bot UA (prerender)")
for ua_name, ua in [("googlebot", BOT), ("browser", BROWSER)]:
    S = requests.Session()
    S.headers.update({"User-Agent": ua})
    for u in ["https://data.nasdaq.com/databases/SEP/pricing",
              "https://data.nasdaq.com/databases/SFA/pricing",
              "https://www.tiingo.com/pricing",
              "https://www.tiingo.com/documentation/end-of-day"]:
        try:
            r = S.get(u, timeout=90)
            t = txt_of(r.text)
            hits = re.findall(r".{0,70}\$\s?\d[\d,]*(?:\.\d\d)?.{0,60}", t)
            print(f"  [{ua_name}] {u[24:]:44s} HTTP{r.status_code} raw={len(r.content):>7,} "
                  f"text={len(t):>6,} $hits={len(hits)}")
            for h in hits[:10]:
                print(f"        $ {h.strip()[:160]}")
            if len(t) > 300 and not hits:
                print(f"        text sample: {t[:300]}")
        except Exception as e:
            print(f"  [{ua_name}] {u} EXC {str(e)[:80]}")
        time.sleep(1.0)

hr("5.2 yfinance library on DELISTED symbols (curl_cffi impersonation)")
import yfinance as yf
print(f"  yfinance {yf.__version__}")
for sym in ["AAPL", "TWTR", "ATVI", "SIVB", "SIVBQ", "CTXS", "ABMD", "CLVS", "VIAC"]:
    try:
        t0 = time.time()
        df = yf.Ticker(sym).history(period="max", auto_adjust=True)
        dt = time.time() - t0
        if len(df):
            print(f"  {sym:8s} BARS={len(df):<6} {df.index[0].date()}..{df.index[-1].date()} "
                  f"lastClose={df['Close'].iloc[-1]:.4f}  {dt:.1f}s")
        else:
            print(f"  {sym:8s} ZERO BARS  {dt:.1f}s")
    except Exception as e:
        print(f"  {sym:8s} EXC {type(e).__name__}: {str(e)[:100]}")
    time.sleep(2.5)

hr("5.3 Wikipedia S&P 500 historical constituent CHANGES (free point-in-time membership)")
try:
    tabs = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    print(f"  tables={len(tabs)}")
    cur, chg = tabs[0], tabs[1]
    print(f"  current constituents: rows={len(cur)} cols={list(cur.columns)[:6]}")
    chg.columns = ["_".join([str(x) for x in c]) if isinstance(c, tuple) else str(c)
                   for c in chg.columns]
    print(f"  CHANGES table: rows={len(chg)} cols={list(chg.columns)}")
    print(chg.head(6).to_string())
    dcol = [c for c in chg.columns if "Date" in c][0]
    yrs = pd.to_datetime(chg[dcol], errors="coerce").dt.year.dropna()
    print(f"  changes span {int(yrs.min())}..{int(yrs.max())}, n={len(yrs)}")
    rem = [c for c in chg.columns if "Removed" in c and "Ticker" in c]
    if rem:
        removed = chg[rem[0]].dropna().astype(str)
        print(f"  distinct REMOVED tickers = {removed.nunique()}")
        chk = ["SIVB", "ATVI", "SPLK", "CTXS", "ABMD", "TWTR", "FRC", "SBNY", "VIAC"]
        print(f"  our test delistings present in Removed column: "
              f"{[c for c in chk if c in set(removed)]}")
    chg.to_csv("_data_probe_surv_sp500_changes.csv", index=False)
    print("  saved _data_probe_surv_sp500_changes.csv")
except Exception as e:
    print(f"  EXC {type(e).__name__}: {str(e)[:200]}")

hr("5.4 stockanalysis.com delisted-stocks list")
S = requests.Session()
S.headers.update({"User-Agent": BROWSER})
for u in ["https://stockanalysis.com/actions/delisted/",
          "https://stockanalysis.com/api/screener/s/f?m=marketCap&s=desc&c=s,n,marketCap"
          "&cn=20&i=delisted",
          "https://stockanalysis.com/stocks/sivb/history/"]:
    try:
        r = S.get(u, timeout=60)
        t = txt_of(r.text)
        print(f"  {u[:70]}\n    HTTP{r.status_code} raw={len(r.content):,} text={len(t):,}")
        print(f"    {t[:400]}")
    except Exception as e:
        print(f"  {u[:70]} EXC {str(e)[:90]}")
    time.sleep(1.5)

print("\nDONE")
