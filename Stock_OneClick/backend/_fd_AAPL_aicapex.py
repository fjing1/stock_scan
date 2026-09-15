import urllib.request, json, gzip, os, datetime as dt
UA={"User-Agent":"Research Analyst research.analyst.fj@gmail.com"}
def get(u): return urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=180).read()
CIK={"AAPL":"0000320193","MSFT":"0000789019","GOOGL":"0001652044","META":"0001326801","AMZN":"0001018724"}
def facts(sym):
    p=f"_fd_AAPL_cf_{sym}.json.gz"
    if not os.path.exists(p):
        b=get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK[sym]}.json"); open(p,"wb").write(gzip.compress(b))
    return json.loads(gzip.open(p).read())
def annual(d, tags, form="10-K"):
    for tg in tags:
        node=d["facts"].get("us-gaap",{}).get(tg)
        if not node: continue
        out={}
        for unit,arr in node["units"].items():
            for a in arr:
                if a.get("form")!=form or "start" not in a: continue
                days=(dt.date.fromisoformat(a["end"])-dt.date.fromisoformat(a["start"])).days
                if days<330 or days>400: continue
                out[a["end"]]=a["val"]
        if out: return tg,out
    return None,{}
REV=["RevenueFromContractWithCustomerExcludingAssessedTax","Revenues","RevenueFromContractWithCustomerIncludingAssessedTax"]
RD=["ResearchAndDevelopmentExpense"]
CAPEX=["PaymentsToAcquirePropertyPlantAndEquipment","PaymentsToAcquireProductiveAssets"]
print("FACT: R&D and capex intensity, from SEC XBRL companyfacts (10-K annual durations)")
print(f"{'sym':6s} {'FY end':12s} {'revenue':>12s} {'R&D':>11s} {'R&D%':>6s} {'capex':>11s} {'capex%':>7s}")
store={}
for s in CIK:
    d=facts(s)
    tr,rev=annual(d,REV); tg,rd=annual(d,RD); tc,cx=annual(d,CAPEX)
    ends=sorted(set(rev)&set(cx), reverse=True)[:4]
    store[s]={}
    for e in ends:
        r=rev[e]; q=rd.get(e); c=cx[e]
        store[s][e]=(r,q,c)
        print(f"{s:6s} {e:12s} {r/1e6:12,.0f} {(q or 0)/1e6:11,.0f} {100*(q or 0)/r:5.1f}% {c/1e6:11,.0f} {100*c/r:6.1f}%")
    print(f"      tags rev={tr} rd={tg} capex={tc}")
