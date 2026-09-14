import json, time, requests
key=[l.strip().split("=",1)[1] for l in open("/Users/feijing/github.com/stock_scan/.env") if l.startswith("ALPHAVANTAGE_API_KEY=")][0]
BASE="https://www.alphavantage.co/query"
def call(p,label,save=None):
    p=dict(p); p["apikey"]=key
    t0=time.time(); r=requests.get(BASE,params=p,timeout=90); dt=time.time()-t0
    print(f"\n=== {label} === HTTP {r.status_code} {len(r.content)}B {dt:.2f}s")
    try: j=r.json()
    except Exception: print(r.text[:400]); return None
    if isinstance(j,dict):
        for k in ("Information","Note","Error Message","message","endpoint"):
            if k in j: print(f"  {k}: {str(j[k])[:250]}")
    if save: json.dump(j,open(save,"w"))
    return j

OUT="/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/"
# A) REALTIME_OPTIONS full inspection
j=call({"function":"REALTIME_OPTIONS","symbol":"IBM"},"A. REALTIME_OPTIONS IBM",OUT+"_data_probe_av_realtime_options.json")
if j:
    d=j.get("data")
    print("  data type:",type(d).__name__,"len:",len(d) if hasattr(d,'__len__') else None)
    if isinstance(d,list) and d:
        print("  row[0]:",json.dumps(d[0],indent=1)[:1200])
        print("  row keys:",list(d[0].keys()))
    elif isinstance(d,list):
        print("  EMPTY LIST -> no rows served")
    print("  raw first 900:",json.dumps(j)[:900])
time.sleep(20)
# B) HISTORICAL_OPTIONS with date, well spaced
j=call({"function":"HISTORICAL_OPTIONS","symbol":"IBM","date":"2024-03-15"},"B. HISTORICAL_OPTIONS IBM date=2024-03-15",OUT+"_data_probe_av_histopt_dated.json")
if j and isinstance(j.get("data"),list):
    d=j["data"]; print("  rows:",len(d))
    if d: print("  row[0]:",json.dumps(d[0],indent=1)[:900])
