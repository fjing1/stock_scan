import pandas as pd, numpy as np
px=pd.read_pickle("_arm_apple_panel.pkl"); px=px.loc[px.ARM.notna()]
r=px.pct_change()*100; idx=px.index
ER=["2023-11-02","2024-02-01","2024-05-02","2024-08-01","2024-10-31","2025-01-30","2025-05-01",
    "2025-07-31","2025-10-30","2026-01-29","2026-04-30","2026-07-30"]
rows=[]
for d in ER:
    i=idx.searchsorted(pd.Timestamp(d))
    if idx[i]!=pd.Timestamp(d): continue
    j=i+1
    rows.append({"er":d,"react":str(idx[j].date()),"AAPL":r.AAPL.iloc[j],"ARM":r.ARM.iloc[j],
                 "SMH":r.SMH.iloc[j],"ARMrel":r.ARM.iloc[j]-r.SMH.iloc[j]})
d=pd.DataFrame(rows)
print("=== Apple earnings reaction days: does ARM follow AAPL? ===")
print(d.round(2).to_string(index=False))
def ols(y,x):
    X=np.column_stack([np.ones(len(x)),x]); b,_,_,_=np.linalg.lstsq(X,y,rcond=None)
    e=y-X@b; n=2; s2=e@e/(len(y)-2); se=np.sqrt(np.diag(s2*np.linalg.inv(X.T@X)))
    r2=1-(e@e)/(((y-y.mean())**2).sum()); return b,se,b/se,r2
for yn,y in [("ARM raw",d.ARM.values),("ARM - SMH",d.ARMrel.values)]:
    b,se,t,r2=ols(y,d.AAPL.values)
    print(f"\n  {yn} = a + b*AAPL_reaction   (n={len(d)})")
    print(f"    a={b[0]:+.3f} (t={t[0]:+.2f})   b={b[1]:+.3f} (se={se[1]:.3f}, t={t[1]:+.2f})   R2={r2:.4f}")
print(f"\n  sign agreement ARM vs AAPL on reaction days: {int((np.sign(d.ARM)==np.sign(d.AAPL)).sum())}/{len(d)}")
print(f"  corr(ARM, AAPL) on the 12 reaction days: {np.corrcoef(d.ARM,d.AAPL)[0,1]:+.3f}")
print(f"  corr(ARM-SMH, AAPL): {np.corrcoef(d.ARMrel,d.AAPL)[0,1]:+.3f}")

print("\n=== Apple's 15 BEST and 15 WORST single days since ARM's IPO: what did ARM do? ===")
rr=r.dropna(subset=["AAPL","ARM","SMH"])
for lbl,sel in [("AAPL best 15",rr.nlargest(15,"AAPL")),("AAPL worst 15",rr.nsmallest(15,"AAPL"))]:
    print(f"\n  {lbl}: mean AAPL {sel.AAPL.mean():+.2f}% | mean ARM {sel.ARM.mean():+.2f}% | mean ARM-SMH {(sel.ARM-sel.SMH).mean():+.2f}% | ARM up {int((sel.ARM>0).sum())}/15")
print(f"\n  Full-sample baseline: mean ARM {rr.ARM.mean():+.2f}% | mean ARM-SMH {(rr.ARM-rr.SMH).mean():+.2f}%")
