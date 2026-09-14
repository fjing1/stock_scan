import json, time, requests
key=[l.strip().split("=",1)[1] for l in open("/Users/feijing/github.com/stock_scan/.env") if l.startswith("ALPHAVANTAGE_API_KEY=")][0]
BASE="https://www.alphavantage.co/query"
def call(p,label):
    p=dict(p); p["apikey"]=key
    r=requests.get(BASE,params=p,timeout=60)
    print(f"\n=== {label} === HTTP {r.status_code} {len(r.content)}B")
    try: j=r.json()
    except Exception: print(r.text[:300]); return None
    if isinstance(j,dict):
        print("keys:",list(j.keys())[:8])
        for k in ("Information","Note","Error Message"):
            if k in j: print(f"  {k}: {j[k][:220]}")
    return j
# 2) realtime options (documented premium, verify)
call({"function":"REALTIME_OPTIONS","symbol":"IBM"},"2. REALTIME_OPTIONS IBM")
# 3) explicit historical date, in case gating differs by param
call({"function":"HISTORICAL_OPTIONS","symbol":"IBM","date":"2017-11-15"},"3. HISTORICAL_OPTIONS IBM date=2017-11-15")
# 4) key liveness check on a known-free endpoint (1 of 25/day)
j=call({"function":"TIME_SERIES_DAILY","symbol":"IBM","outputsize":"compact"},"4. TIME_SERIES_DAILY IBM (liveness)")
if j and "Time Series (Daily)" in j:
    ts=j["Time Series (Daily)"]; ds=sorted(ts)
    print("  FREE-TIER OK: rows",len(ts),"span",ds[0],"->",ds[-1],"last close",ts[ds[-1]]["4. close"])
