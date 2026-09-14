import pandas as pd, numpy as np
px=pd.read_pickle("_arm_apple_panel.pkl")
px=px.loc[px.ARM.notna()]      # ARM IPO onward only
idx=px.index
print("panel:",idx[0].date(),"->",idx[-1].date(),"n=",len(idx))

ANN = {"2023-09-12":"iPhone 15/15 Pro unveiled","2024-09-09":"iPhone 16/16 Pro unveiled",
       "2025-09-09":"iPhone 17/17 Pro/Air unveiled","2026-09-09":"iPhone 18 Pro/Duo unveiled"}
ANN_MID={"2025-02-19":"iPhone 16e announced","2026-03-02":"iPhone 17e announced"}
REL={"2023-09-22":"iPhone 15 on sale","2024-09-20":"iPhone 16 on sale","2025-09-19":"iPhone 17 on sale",
     "2025-02-28":"iPhone 16e on sale","2026-03-11":"iPhone 17e available"}
AAPL_ER=["2023-11-02","2024-02-01","2024-05-02","2024-08-01","2024-10-31","2025-01-30","2025-05-01",
         "2025-07-31","2025-10-30","2026-01-29","2026-04-30","2026-07-30"]
WINS=[(0,0,"d0"),(0,1,"d0_1"),(-1,1,"m1p1"),(-5,5,"m5p5"),(-1,20,"m1p20")]

def loc(d):
    d=pd.Timestamp(d); p=idx.searchsorted(d)
    return p if (p<len(idx) and idx[p]==d) else None
def car(i,a,b,col):
    """cumulative return from close(i+a-1) to close(i+b)"""
    lo,hi=i+a-1,i+b
    if lo<0 or hi>=len(idx): return np.nan
    v=px[col].iloc[hi]/px[col].iloc[lo]-1
    return np.nan if pd.isna(v) else v*100

def study(name,events,shift=0):
    rows=[]
    for d,lab in (events.items() if isinstance(events,dict) else [(e,"AAPL ER") for e in events]):
        i=loc(d)
        if i is None: print(f"  !! {d} not a trading day ({lab})"); continue
        i+=shift
        rec={"date":d,"anchor":str(idx[i].date()),"label":lab}
        for a,b,t in WINS:
            A=car(i,a,b,"ARM"); S=car(i,a,b,"SMH")
            rec[f"ARM_{t}"]=A; rec[f"AAPL_{t}"]=car(i,a,b,"AAPL"); rec[f"rel_{t}"]=A-S
        rows.append(rec)
    df=pd.DataFrame(rows)
    print(f"\n{'='*118}\n{name}   n={len(df)}\n{'='*118}")
    print(df[["date","anchor","label","ARM_d0","AAPL_d0","rel_d0","ARM_m5p5","rel_m5p5","ARM_m1p20","rel_m1p20"]].round(2).to_string(index=False))
    print("  MEANS:")
    for a,b,t in WINS:
        A=df[f"ARM_{t}"].dropna(); R=df[f"rel_{t}"].dropna()
        if len(A)<2: continue
        ta=A.mean()/(A.std(ddof=1)/np.sqrt(len(A))); tr=R.mean()/(R.std(ddof=1)/np.sqrt(len(R)))
        print(f"   {t:6s} n={len(A):2d} | ARM raw {A.mean():+7.2f}% med {A.median():+7.2f}% t={ta:+5.2f} up={int((A>0).sum())}/{len(A)}"
              f" || ARM-SMH {R.mean():+7.2f}% med {R.median():+7.2f}% t={tr:+5.2f} up={int((R>0).sum())}/{len(R)}")
    return df

d1=study("A. FLAGSHIP iPHONE ANNOUNCEMENT (Apple keynote, ~1pm ET, so d0 is a reaction day)",ANN)
d2=study("B. ALL iPHONE ANNOUNCEMENTS (flagship + e-series)",{**ANN,**ANN_MID})
d3=study("C. iPHONE ON-SALE DAYS",REL)
d4=study("D. AAPL EARNINGS, anchored to the t+1 reaction day (Apple reports after the close)",AAPL_ER,shift=1)

print(f"\n{'='*118}\nE. UNCONDITIONAL BASELINE (every trading day since IPO) -- what a random day looks like\n{'='*118}")
base={}
for a,b,t in WINS:
    A=np.array([car(i,a,b,"ARM") for i in range(len(idx))]); S=np.array([car(i,a,b,"SMH") for i in range(len(idx))])
    m=~(np.isnan(A)|np.isnan(S)); A,S=A[m],S[m]; R=A-S; base[t]=(A,R)
    print(f"  {t:6s} n={len(A):4d} | ARM raw {A.mean():+7.2f}% med {np.median(A):+7.2f}% sd {A.std(ddof=1):6.2f} | rel {R.mean():+7.2f}% med {np.median(R):+7.2f}% sd {R.std(ddof=1):6.2f}")

print(f"\n{'='*118}\nF. IS THE EVENT MEAN DISTINGUISHABLE FROM A RANDOM DAY? (20k bootstrap of baseline, same n)\n{'='*118}")
rng=np.random.default_rng(7)
for nm,df in [("A flagship",d1),("B all-ann",d2),("C on-sale",d3),("D AAPL-ER",d4)]:
    for a,b,t in WINS:
        obs=df[f"rel_{t}"].dropna().values
        if len(obs)<2: continue
        A,R=base[t]; n=len(obs)
        dr=np.array([rng.choice(R,n,replace=True).mean() for _ in range(20000)])
        lo,hi=np.percentile(dr,[5,95]); p=(np.abs(dr-R.mean())>=abs(obs.mean()-R.mean())).mean()
        print(f"  {nm:11s} {t:6s} n={n:2d} rel {obs.mean():+7.2f}%  baseline 90%CI[{lo:+7.2f},{hi:+7.2f}]  p={p:.3f}  {'** OUTSIDE **' if (obs.mean()<lo or obs.mean()>hi) else 'inside'}")

print(f"\n{'='*118}\nG. HEAD-TO-HEAD ON THE 4 iPHONE LAUNCH WEEKS: did ARM follow AAPL?\n{'='*118}")
for d,lab in ANN.items():
    i=loc(d)
    if i is None: continue
    lo=max(0,i-2); hi=min(len(idx)-1,i+5)
    sl=px.iloc[lo:hi+1]
    rr=(sl/sl.iloc[0]-1)*100
    print(f"\n{d}  {lab}   (t=0 is {idx[i].date()})")
    out=pd.DataFrame({"t":[j-i for j in range(lo,hi+1)],"ARM":rr.ARM.values,"AAPL":rr.AAPL.values,
                      "SMH":rr.SMH.values,"NVDA":rr.NVDA.values},index=[str(x.date()) for x in sl.index])
    print(out.round(2).to_string())
