import pandas as pd, numpy as np
px=pd.read_pickle("_arm_apple_panel.pkl")
r=px.pct_change()
end=px.index[-1]; start=end-pd.Timedelta(days=95)
w=px.loc[px.index>=start]
rw=r.loc[r.index>=start]
print("window",w.index[0].date(),"->",w.index[-1].date(),"n=",len(w))
for c in ["ARM","SMH","NVDA","AAPL","QQQ"]:
    print(f"{c:5s} {w[c].iloc[0]:9.2f} -> {w[c].iloc[-1]:9.2f}  {100*(w[c].iloc[-1]/w[c].iloc[0]-1):+7.2f}%")
print("\n=== ARM worst 15 days in window (ARM ret, SMH ret, ARM-SMH) ===")
d=pd.DataFrame({"ARM":rw.ARM*100,"SMH":rw.SMH*100,"NVDA":rw.NVDA*100,"AAPL":rw.AAPL*100})
d["rel"]=d.ARM-d.SMH
print(d.sort_values("ARM").head(15).round(2).to_string())
print("\n=== cumulative rel (ARM - SMH) path, weekly ===")
cum=(1+rw.ARM).cumprod()/(1+rw.SMH).cumprod()
print((100*(cum.resample("W").last()-1)).round(1).to_string())
