"""Attack the 'revenue-line corroboration': corr(iPhone YoY, NVDA reported revenue YoY) = -0.64, n=12.
Tests: (1) reproduce, (2) autocorrelation-adjusted significance, (3) PLACEBO -- do companies with
zero plausible link to NVDA (KO, PG, JNJ, MCD) show the same negative corr?  (4) is it just
NVDA's monotone deceleration vs a mean-reverting iPhone series?"""
import json, gzip, os, numpy as np, pandas as pd, requests, time
pd.set_option("display.width",240)
B="/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/"
UA={"User-Agent":"Research Analyst research.analyst@gmail.com"}

def facts(cik, tag_sym):
    p=B+f"_fd_AAPL_verify_cf_{tag_sym}.json.gz"
    if not os.path.exists(p):
        u=f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
        rr=requests.get(u,headers=UA,timeout=60); rr.raise_for_status()
        with gzip.open(p,"wt") as f: f.write(rr.text)
        time.sleep(0.4)
    with gzip.open(p,"rt") as f: return json.load(f)

def qseries(fj, tags):
    """quarterly (3M) duration facts, tag ladder, keyed by period end"""
    out={}
    for tg in tags:
        node=fj["facts"].get("us-gaap",{}).get(tg)
        if not node: continue
        for u,arr in node["units"].items():
            if u!="USD": continue
            for it in arr:
                if "start" not in it: continue
                d=(pd.Timestamp(it["end"])-pd.Timestamp(it["start"])).days
                if not (80<=d<=100): continue
                e=pd.Timestamp(it["end"])
                key=e
                # prefer latest accession
                if key not in out or it["end"]>=out[key][1]:
                    out[key]=(it["val"], it["accn"], tg)
    s=pd.Series({k:v[0] for k,v in out.items()}).sort_index()
    return s

# ---- NVDA (CIK 1045810) total revenue, quarterly
nv=facts(1045810,"NVDA")
nv_rev=qseries(nv,["RevenueFromContractWithCustomerExcludingAssessedTax","Revenues",
                   "RevenueFromContractWithCustomerIncludingAssessedTax"])
print("NVDA quarterly revenue facts:",len(nv_rev), nv_rev.index.min().date(),"->",nv_rev.index.max().date())
print((nv_rev/1e6).tail(16).round(0).to_string())

# ---- AAPL iPhone revenue from the dimensional segment facts already fetched
aapl=facts(320193,"AAPL")
def dim_iphone():
    # pull IPhoneMember from companyfacts?  companyfacts has NO dimensions -> use local seg csv
    d=pd.read_csv(B+"_fd_AAPL_seg_ProductOrServiceAxis.csv",parse_dates=["end"])
    ip=d[d.mem.str.contains("IPhone",case=False,na=False)].sort_values("end")
    return ip.set_index("end")["val"], ip
ip,ipraw=dim_iphone()
print("\niPhone quarterly rev facts:",len(ip), ip.index.min().date(),"->",ip.index.max().date())
print((ip/1e6).round(0).to_string())

def yoy(s, tol=45):
    """date-matched YoY (never positional).  365d +- tol"""
    o={}
    for d,v in s.items():
        tgt=d-pd.Timedelta(days=365)
        cand=[(abs((x-tgt).days),x) for x in s.index if abs((x-tgt).days)<=tol]
        if not cand: continue
        _,pd_ = min(cand)
        o[d]=v/s[pd_]-1
    return pd.Series(o).sort_index()

ip_y=yoy(ip); nv_y=yoy(nv_rev)
print("\niPhone YoY:\n",(ip_y*100).round(1).to_string())

# align: NVDA fiscal quarters end ~1 month before Apple's -> match nearest within 60d
rows=[]
for d,v in ip_y.items():
    cand=[(abs((x-d).days),x) for x in nv_y.index if abs((x-d).days)<=60]
    if cand:
        _,nd=min(cand); rows.append((d,nd,v,nv_y[nd]))
al=pd.DataFrame(rows,columns=["aapl_end","nvda_end","iphone_yoy","nvda_yoy"])
print("\naligned panel:\n",al.assign(iphone_yoy=lambda x:(x.iphone_yoy*100).round(1),
                                     nvda_yoy=lambda x:(x.nvda_yoy*100).round(1)).to_string(index=False))
n=len(al); c=np.corrcoef(al.iphone_yoy,al.nvda_yoy)[0,1]
t=c*np.sqrt(n-2)/np.sqrt(1-c**2)
print(f"\nREPRODUCE: corr = {c:+.3f}  n={n}  naive t={t:+.2f}")

# autocorrelation of each series -> effective sample size
def ac1(x):
    x=np.asarray(x); x=x-x.mean()
    return (x[1:]@x[:-1])/(x@x)
a1,a2=ac1(al.iphone_yoy),ac1(al.nvda_yoy)
n_eff=n*(1-a1*a2)/(1+a1*a2)
t_adj=c*np.sqrt(max(n_eff,3)-2)/np.sqrt(1-c**2)
print(f"AR(1): iPhone {a1:+.2f}  NVDA {a2:+.2f}  ->  n_eff (Quenouille/Bartlett) = {n_eff:.1f}, "
      f"adj t = {t_adj:+.2f}  {'NOT SIGNIFICANT' if abs(t_adj)<2.0 else 'significant'}")

# detrend NVDA (it decelerates monotonically) -> is the corr just two trends?
for lab,tr in [("raw",al.nvda_yoy.values),("first-diff",None)]:
    pass
d_ip=np.diff(al.iphone_yoy.values); d_nv=np.diff(al.nvda_yoy.values)
cd=np.corrcoef(d_ip,d_nv)[0,1]
nd=len(d_ip); td=cd*np.sqrt(nd-2)/np.sqrt(1-cd**2)
print(f"FIRST DIFFERENCES (removes common trend): corr = {cd:+.3f} n={nd} t={td:+.2f}")

# PLACEBO: companies with no conceivable AI transmission channel
PLAC={"KO":21344,"PG":80424,"JNJ":200406,"MCD":63908,"PEP":77476,"WMT":104169}
print("\nPLACEBO: corr(company revenue YoY, NVDA revenue YoY) over the SAME quarters")
print(f"{'co':<6} {'corr':>7} {'n':>4} {'t':>7}")
print(f"{'AAPL(iPhone)':<6} {c:>7.3f} {n:>4} {t:>7.2f}")
for sym,cik in PLAC.items():
    try:
        fj=facts(cik,sym)
        s=qseries(fj,["RevenueFromContractWithCustomerExcludingAssessedTax","Revenues",
                      "RevenueFromContractWithCustomerIncludingAssessedTax","SalesRevenueNet",
                      "RevenueFromContractWithCustomerIncludingAssessedTax"])
        sy=yoy(s)
        rr=[]
        for d,v in sy.items():
            if d < al.aapl_end.min()-pd.Timedelta(days=50) or d > al.aapl_end.max()+pd.Timedelta(days=50): continue
            cand=[(abs((x-d).days),x) for x in nv_y.index if abs((x-d).days)<=60]
            if cand:
                _,ndte=min(cand); rr.append((v,nv_y[ndte]))
        if len(rr)<6: print(f"{sym:<6} insufficient (n={len(rr)})"); continue
        A=np.array(rr); cc=np.corrcoef(A[:,0],A[:,1])[0,1]; nn=len(A)
        tt=cc*np.sqrt(nn-2)/np.sqrt(1-cc**2)
        print(f"{sym:<6} {cc:>7.3f} {nn:>4} {tt:>7.2f}")
    except Exception as ex:
        print(f"{sym:<6} ERR {ex}")
