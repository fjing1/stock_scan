"""Download CBOE single-stock 30-day IV index history (VXAPL/VXAZN/VXGOG/VXIBM/VXGS)
from the Wayback Machine. Each archived CSV is cumulative, so the LAST snapshot = full history."""
import requests, time, io, pandas as pd, json, os
H={"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
OUT="/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/"

def get(url, tries=5, **kw):
    for i in range(tries):
        try:
            r=requests.get(url,headers=H,timeout=90,**kw)
            if r.status_code==429: time.sleep(8*(i+1)); continue
            return r
        except Exception as e:
            print("   retry",type(e).__name__); time.sleep(5)
    return None

def cdx_all(pattern):
    u=f"https://web.archive.org/cdx/search/cdx?url={pattern}&output=json&filter=statuscode:200&collapse=digest"
    r=get(u)
    if r is None or r.status_code!=200 or not r.text.strip(): return []
    try: j=r.json()
    except Exception: return []
    if len(j)<2: return []
    hdr=j[0]; return [dict(zip(hdr,row)) for row in j[1:]]

SYMS=["VXAPL","VXAZN","VXGOG","VXIBM","VXGS"]
summary={}
for s in SYMS:
    print(f"\n===== {s} =====")
    snaps=cdx_all(f"cboe.com/publish/scheduledtask/mktdata/datahouse/{s}dailyprices.csv")
    print(f"  archived snapshots (status200, dedup): {len(snaps)}")
    if not snaps:
        summary[s]={"snapshots":0}; time.sleep(2); continue
    snaps.sort(key=lambda d:d["timestamp"])
    print(f"  first {snaps[0]['timestamp']} len={snaps[0]['length']}   last {snaps[-1]['timestamp']} len={snaps[-1]['length']}")
    # pick the biggest file (most complete history)
    best=max(snaps,key=lambda d:int(d["length"]))
    print(f"  largest snapshot: {best['timestamp']} len={best['length']}")
    url=f"https://web.archive.org/web/{best['timestamp']}id_/{best['original']}"
    t0=time.time(); r=get(url); dt=time.time()-t0
    if r is None: print("  DOWNLOAD FAILED"); continue
    print(f"  GET {url[:100]}...  HTTP {r.status_code} {len(r.content)}B {dt:.2f}s")
    txt=r.text
    print("  raw head:", repr(txt[:200]))
    # parse: CBOE files have a couple of junk header lines
    lines=[l for l in txt.splitlines() if l.strip()]
    hdr_i=None
    for i,l in enumerate(lines[:8]):
        if "date" in l.lower(): hdr_i=i; break
    df=None
    if hdr_i is not None:
        df=pd.read_csv(io.StringIO("\n".join(lines[hdr_i:])))
        df.columns=[c.strip() for c in df.columns]
        dc=df.columns[0]
        df[dc]=pd.to_datetime(df[dc],errors="coerce")
        df=df.dropna(subset=[dc]).sort_values(dc)
        vc=[c for c in df.columns if c!=dc]
        print(f"  PARSED rows={len(df)} cols={list(df.columns)}")
        print(f"  span {df[dc].iloc[0].date()} -> {df[dc].iloc[-1].date()}")
        v=pd.to_numeric(df[vc[-1]],errors="coerce").dropna()
        print(f"  value col '{vc[-1]}': n={len(v)} min={v.min():.2f} mean={v.mean():.2f} max={v.max():.2f}")
        print("  head:\n", df.head(3).to_string(index=False))
        print("  tail:\n", df.tail(3).to_string(index=False))
        df.to_csv(OUT+f"_data_probe_cboe_{s.lower()}.csv",index=False)
        summary[s]={"snapshots":len(snaps),"rows":len(df),
                    "span":[str(df[dc].iloc[0].date()),str(df[dc].iloc[-1].date())],
                    "mean":round(float(v.mean()),2)}
    time.sleep(2)
print("\n\n==== SUMMARY ====")
print(json.dumps(summary,indent=1))
