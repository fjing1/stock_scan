import pandas as pd, numpy as np
px=pd.read_pickle("_arm_apple_panel.pkl")
r=np.log(px).diff().dropna()   # log returns
r=r.loc["2023-09-14":]         # ARM IPO 2023-09-14 first trade
print("sample:",r.index[0].date(),"->",r.index[-1].date(),"n=",len(r))

def ols(y,X,names):
    X=np.column_stack([np.ones(len(X))]+[X[:,i] for i in range(X.shape[1])])
    b,_,_,_=np.linalg.lstsq(X,y,rcond=None)
    resid=y-X@b; n,k=X.shape
    s2=resid@resid/(n-k); XtXi=np.linalg.inv(X.T@X); se=np.sqrt(np.diag(s2*XtXi))
    t=b/se
    ss_tot=((y-y.mean())**2).sum(); r2=1-(resid@resid)/ss_tot
    adj=1-(1-r2)*(n-1)/(n-k)
    return b,se,t,r2,adj,n

def show(title,ycol,xcols,data):
    d=data[[ycol]+xcols].dropna()
    y=d[ycol].values; X=d[xcols].values
    b,se,t,r2,adj,n=ols(y,X,xcols)
    print(f"\n--- {title}  (n={n}) ---")
    print(f"{'term':8s} {'beta':>8s} {'se':>7s} {'t':>7s}")
    print(f"{'alpha':8s} {b[0]:8.5f} {se[0]:7.5f} {t[0]:7.2f}")
    for i,c in enumerate(xcols):
        print(f"{c:8s} {b[i+1]:8.3f} {se[i+1]:7.3f} {t[i+1]:7.2f}")
    print(f"R2={r2:.4f}  adjR2={adj:.4f}")
    return r2

print("\n########## UNIVARIATE: which single factor explains ARM variance? ##########")
res={}
for c in ["AAPL","NVDA","SMH","QQQ","SPY","QCOM","AVGO","TSM","SOXX","MU"]:
    res[c]=show(f"ARM ~ {c}","ARM",[c],r)
print("\n=== univariate R2 ranking ===")
for k,v in sorted(res.items(),key=lambda x:-x[1]): print(f"{k:6s} R2={v:.4f}  corr={np.sqrt(v):.3f}")

print("\n########## JOINT ##########")
show("ARM ~ AAPL + NVDA + SMH + QQQ","ARM",["AAPL","NVDA","SMH","QQQ"],r)
show("ARM ~ AAPL + SMH","ARM",["AAPL","SMH"],r)
show("ARM ~ AAPL + NVDA","ARM",["AAPL","NVDA"],r)
show("ARM ~ AAPL + QQQ","ARM",["AAPL","QQQ"],r)

print("\n########## INCREMENTAL R2 OF AAPL ##########")
base=show("ARM ~ NVDA+SMH+QQQ (no AAPL)","ARM",["NVDA","SMH","QQQ"],r)
full=show("ARM ~ NVDA+SMH+QQQ+AAPL","ARM",["NVDA","SMH","QQQ","AAPL"],r)
print(f"\nIncremental R2 from adding AAPL: {full-base:+.5f}")
base2=show("ARM ~ AAPL only","ARM",["AAPL"],r)
print(f"Incremental R2 from adding NVDA+SMH+QQQ to AAPL: {full-base2:+.5f}")

print("\n########## ROLLING 126d correlation ARM vs each ##########")
roll=pd.DataFrame({c:r.ARM.rolling(126).corr(r[c]) for c in ["AAPL","NVDA","SMH","QQQ"]}).dropna()
print(roll.resample("QE").last().round(3).to_string())
print("\nmean rolling corr:\n",roll.mean().round(3).to_string())
print("\nLAST 63d corr:", {c: round(r.ARM.tail(63).corr(r[c].tail(63)),3) for c in ["AAPL","NVDA","SMH","QQQ","QCOM"]})
