"""Harvest every CBOE volatility-index daily-price history archived on Wayback.
Each snapshot is cumulative -> take the largest. Reports which have IV history Yahoo lacks."""
import requests, time, io, pandas as pd, json
H={"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
OUT="/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/"
BASE="cboe.com/publish/scheduledtask/mktdata/datahouse/"
FILES=["VXIBMDailyPrices.csv","VXEEMDailyPrices.csv","VXEWZDailyPrices.csv","VXFXIDailyPrices.csv",
 "VXGDXDailyPrices.csv","VXSLVDailyPrices.csv","VXXLEDailyPrices.csv","VXEFADailyData.csv",
 "RVXDailyPrices.csv","vxvdailyprices.csv","VXSTCurrent.csv","vxocurrent.csv","vxncurrent.csv",
 "tyvixdailyprices.csv","VXTYNDailyprices.csv","ivxdailyprices.csv","vix9ddailyprices.csv",
 "vix3mdailyprices.csv","vix6mdailyprices.csv","Skewdailyprices.csv","VXDOHLCPrices.csv","vxmtdailyprices.csv"]
def get(u,tries=5):
    for i in range(tries):
        try:
            r=requests.get(u,headers=H,timeout=120)
            if r.status_code==429: time.sleep(8*(i+1)); continue
            return r
        except Exception: time.sleep(5)
    return None
res={}
for fn in FILES:
    r=get(f"https://web.archive.org/cdx/search/cdx?url={BASE}{fn}&output=json&filter=statuscode:200&collapse=digest")
    snaps=[]
    if r is not None and r.status_code==200 and r.text.strip():
        try:
            j=r.json(); snaps=[dict(zip(j[0],row)) for row in j[1:]] if len(j)>1 else []
        except Exception: pass
    if not snaps:
        print(f"{fn:26} NO ARCHIVE"); res[fn]={"snapshots":0}; time.sleep(1.5); continue
    best=max(snaps,key=lambda d:int(d["length"]))
    rr=get(f"https://web.archive.org/web/{best['timestamp']}id_/{best['original']}")
    if rr is None or rr.status_code!=200:
        print(f"{fn:26} DL FAIL"); res[fn]={"snapshots":len(snaps),"dl":"fail"}; time.sleep(1.5); continue
    lines=[l for l in rr.text.splitlines() if l.strip()]
    hi=next((i for i,l in enumerate(lines[:12]) if "date" in l.lower()), None)
    if hi is None:
        print(f"{fn:26} NO HEADER  raw={rr.text[:60]!r}"); res[fn]={"snapshots":len(snaps),"parse":"nohdr"}; time.sleep(1.5); continue
    try:
        df=pd.read_csv(io.StringIO("\n".join(lines[hi:])))
    except Exception as e:
        print(f"{fn:26} PARSE ERR {e}"); time.sleep(1.5); continue
    df.columns=[str(c).strip() for c in df.columns]
    dc=df.columns[0]; df[dc]=pd.to_datetime(df[dc],errors="coerce")
    df=df.dropna(subset=[dc]).sort_values(dc)
    if df.empty: print(f"{fn:26} EMPTY"); time.sleep(1.5); continue
    vcols=[c for c in df.columns if c!=dc]
    v=pd.to_numeric(df[vcols[-1]],errors="coerce").dropna()
    print(f"{fn:26} snaps={len(snaps):>3} rows={len(df):>5}  {df[dc].iloc[0].date()} -> {df[dc].iloc[-1].date()}  mean={v.mean():.2f}  cols={vcols}")
    res[fn]={"snapshots":len(snaps),"rows":int(len(df)),
             "span":[str(df[dc].iloc[0].date()),str(df[dc].iloc[-1].date())],
             "mean":round(float(v.mean()),2),"cols":vcols}
    df.to_csv(OUT+"_data_probe_cboe_"+fn.lower().replace("dailyprices","").replace("daily","").replace("current","").replace(".csv","")+".csv",index=False)
    time.sleep(1.5)
print("\n"+json.dumps(res,indent=1))
