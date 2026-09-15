"""EVENT STUDY on the CLAIMED CATALYST (Apple's own AI announcements) versus the alternative
(iPhone hardware launches). Dates from Apple's newsroom pages (_fd_AAPL_news_events.json),
abnormal returns net of SPY and net of XLK. n is small by construction -- reported as such."""
import json, numpy as np, pandas as pd
pd.set_option("display.width",240)
ev=pd.DataFrame(json.load(open("/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/_fd_AAPL_news_events.json")))
ev["date"]=pd.to_datetime(ev.date, errors="coerce")
ev=ev.dropna(subset=["date"]).sort_values("date")
r=pd.read_pickle("/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/_fd_AAPL_rets.pkl")
idx=r.index
def win(i,lo,hi,col):
    a,z=max(0,i+lo),min(len(idx)-1,i+hi); return col.iloc[a:z+1].sum()
rows=[]
for _,e in ev.iterrows():
    if e.date > idx[-1]: 
        rows.append({"kind":e.kind,"date":e.date.date(),"slug":e.slug[:44],"note":"after last price bar"}); continue
    i=idx.searchsorted(e.date, side="left")
    i=min(i,len(idx)-1)
    ar_s=r["AAPL"]-r["SPY"]; ar_x=r["AAPL"]-r["XLK"]
    rows.append({"kind":e.kind,"date":e.date.date(),"slug":e.slug[:44],
                 "d0_raw":r["AAPL"].iloc[i], "d0_ar_spy":ar_s.iloc[i], "d0_ar_xlk":ar_x.iloc[i],
                 "m5p5_ar_spy":win(i,-5,5,ar_s), "m1p20_ar_spy":win(i,-1,20,ar_s),
                 "m1p20_ar_xlk":win(i,-1,20,ar_x), "p1p20_ar_spy":win(i,1,20,ar_s),
                 "m5p5_raw":win(i,-5,5,r["AAPL"])})
o=pd.DataFrame(rows)
num=[c for c in o.columns if c not in ("kind","date","slug","note")]
for c in num: o[c]=(o[c]*100).round(2)
print(o.to_string(index=False))
print("\n=== summary by event class (log %, mean / median / t / hit) ===")
for kind,lab in [("AI","claimed catalyst: Apple's OWN AI announcements"),("HW","alternative: iPhone hardware launches")]:
    s=o[(o.kind==kind)].dropna(subset=["d0_ar_spy"])
    print(f"\n{lab}   n={len(s)}")
    for c in ["d0_raw","d0_ar_spy","d0_ar_xlk","m5p5_ar_spy","m1p20_ar_spy","p1p20_ar_spy","m1p20_ar_xlk"]:
        v=s[c].values.astype(float)
        t=v.mean()/(v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else np.nan
        print(f"   {c:<14} mean {v.mean():>+7.2f}  median {np.median(v):>+7.2f}  t {t:>+6.2f}  hit>0 {np.mean(v>0)*100:>5.0f}%   [n={len(v)}]")
o.to_csv("/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/_fd_AAPL_event_catalyst.csv",index=False)
print("\nNOTE: n=6-7 per class. With daily sigma ~1.5-2%, the minimum detectable mean effect at")
print("t=2 is roughly 1.2-1.6% per event. Anything smaller is invisible at this sample size.")
