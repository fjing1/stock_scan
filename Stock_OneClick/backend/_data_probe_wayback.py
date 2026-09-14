"""Hunt archived CBOE single-stock volatility index files via the Wayback CDX API."""
import requests, time, json
H={"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
def cdx(url, extra=""):
    q=f"https://web.archive.org/cdx/search/cdx?url={url}&output=json&limit=40&collapse=digest{extra}"
    for attempt in range(4):
        try:
            r=requests.get(q,headers=H,timeout=60)
            if r.status_code==429:
                time.sleep(6*(attempt+1)); continue
            print(f"  HTTP {r.status_code} {len(r.content)}B  {url}")
            if r.status_code==200 and r.text.strip():
                try: return r.json()
                except Exception: print("   nonjson:",r.text[:150])
            return None
        except Exception as e:
            print("  ERR",type(e).__name__,str(e)[:80]); time.sleep(4)
    print("  gave up (429)"); return None

PATTERNS=[
 "cboe.com/publish/ScheduledTask/MktData/datahouse/vxaplcurrent.csv",
 "cboe.com/publish/scheduledtask/mktdata/datahouse/vxapldailyprices.csv",
 "cboe.com/publish/*vxapl*",
 "cboe.com/*vxapl*",
 "cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
 "cdn.cboe.com/api/global/us_indices/daily_prices/*",
]
for p in PATTERNS:
    print("\n### ", p)
    j=cdx(p)
    if j and len(j)>1:
        hdr=j[0]; print("   cols:",hdr)
        for row in j[1:12]:
            d=dict(zip(hdr,row)); print("   ",d.get("timestamp"),d.get("statuscode"),d.get("length"),d.get("original")[:110])
    time.sleep(2)
