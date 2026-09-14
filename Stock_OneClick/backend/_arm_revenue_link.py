import pandas as pd, numpy as np
# ---- ARM royalty revenue, $M, from Arm shareholder letters (6-K Ex-99.2). Current-qtr and
# ---- prior-year-comparative columns of the "Financial Overview" table. Files _arm_sl_fye*.txt
arm = {"2022-09":442,"2022-12":425,"2023-03":374,"2023-06":400,   # from prior-year columns
       "2023-09":418,"2023-12":470,"2024-03":514,"2024-06":467,
       "2024-09":514,"2024-12":580,"2025-03":607,"2025-06":585,
       "2025-09":620,"2025-12":737,"2026-03":671,"2026-06":715}
# ---- Apple iPhone net sales, $M, from Apple 10-Q "disaggregated net sales" + 10-K
# ---- Sep(FQ4) quarters derived as FY total (10-K) minus 9-month YTD (Q3 10-Q)
aapl={"2022-09":205489-162863,"2022-12":65775,"2023-03":51334,"2023-06":39669,
      "2023-09":200583-156778,"2023-12":69702,"2024-03":45963,"2024-06":39296,
      "2024-09":201183-154961,"2024-12":69138,"2025-03":46841,"2025-06":44582,
      "2025-09":209586-160561,"2025-12":85269,"2026-03":56994,"2026-06":54252}
d=pd.DataFrame({"arm_royalty":pd.Series(arm),"aapl_iphone":pd.Series(aapl)})
d.index=pd.PeriodIndex(d.index,freq="Q")
d["arm_yoy"]=d.arm_royalty.pct_change(4)*100
d["iph_yoy"]=d.aapl_iphone.pct_change(4)*100
d["arm_qoq"]=d.arm_royalty.pct_change()*100
d["iph_qoq"]=d.aapl_iphone.pct_change()*100
print(d.round(1).to_string())
print("\nARM royalty $M is only PART of ARM revenue; Apple iPhone $M is Apple's largest line.")

def cr(a,b,lag=0):
    x=d[a]; y=d[b].shift(lag)
    s=pd.concat([x,y],axis=1).dropna()
    if len(s)<4: return None
    r=np.corrcoef(s.iloc[:,0],s.iloc[:,1])[0,1]
    n=len(s); t=r*np.sqrt((n-2)/max(1e-12,1-r**2))
    return r,n,t

print("\n=== CORRELATION: ARM royalty growth vs Apple iPhone revenue growth ===")
print(f"{'pair':46s} {'r':>7s} {'n':>4s} {'t':>7s}")
for lag in [0,1,2,-1]:
    for a,b,nm in [("arm_yoy","iph_yoy","y/y growth"),("arm_qoq","iph_qoq","q/q growth")]:
        z=cr(a,b,lag)
        if z: print(f"{nm} ARM_t vs iPhone_t-{lag:<2d}{'':22s} {z[0]:+7.3f} {z[1]:4d} {z[2]:+7.2f}")
print("\n=== LEVELS (spurious-trend prone, shown for completeness) ===")
z=cr("arm_royalty","aapl_iphone",0); print(f"levels r={z[0]:+.3f} n={z[1]} t={z[2]:+.2f}")

print("\n=== THE MATURITY CHECK: Apple iPhone revenue, fiscal-year, from Apple 10-K ===")
fy={"FY2022":205489,"FY2023":200583,"FY2024":201183,"FY2025":209586}
p=pd.Series(fy); print(pd.DataFrame({"iPhone_$M":p,"yoy%":(p.pct_change()*100).round(1)}).to_string())
print("FY2026 9M (through 2026-06-27, 10-Q):",196515,"vs FY2025 9M",160561,"=",round(100*(196515/160561-1),1),"% y/y")
print("\n=== ARM total revenue growth vs royalty growth vs iPhone growth, last 8 q ===")
armtot={"2025-06":1053,"2025-09":1135,"2025-12":1242,"2026-03":1490,"2026-06":1289}
print(pd.Series(armtot).to_string())
