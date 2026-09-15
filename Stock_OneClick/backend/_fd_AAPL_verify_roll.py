"""(1) replicate the rolling-126d beta figures; (2) run the IDENTICAL rolling beta for placebo
large caps -- if KO/PG/JNJ show the same 'flip to negative since 2024', the rolling-beta evidence
is a statement about AI-complex flow concentration, not about AAPL.  (3) recent-window revenue corr."""
import numpy as np, pandas as pd
pd.set_option("display.width",240)
B="/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/"
px=pd.read_pickle(B+"_fd_AAPL_verify_px.pkl")
r=np.log(px).diff().dropna()
ai=r["SMH"]-r["SPY"]; nv=r["NVDA"]-r["SPY"]

def roll_beta(y,x,w=126):
    out={}
    for i in range(w,len(y)+1):
        yy=y.iloc[i-w:i].values; xx=x.iloc[i-w:i].values
        X=np.column_stack([np.ones(w),xx]); b,*_=np.linalg.lstsq(X,yy,rcond=None)
        e=yy-X@b; s2=(e@e)/(w-2); se=np.sqrt(s2*np.linalg.inv(X.T@X)[1,1])
        out[y.index[i-1]]=(b[1],b[1]/se)
    return pd.DataFrame(out,index=["beta","t"]).T

print("="*100); print("(1) REPLICATE the claim's rolling beta for AAPL"); print("="*100)
rb=roll_beta(r["AAPL"]-r["SPY"],ai)
print(rb.groupby(rb.index.year).agg({"beta":["mean","min","max"],"t":"mean"}).round(2).to_string())
print("latest:", rb.tail(1).round(3).to_dict("records"))
print(f"share negative since 2024: {(rb.loc['2024':].beta<0).mean()*100:.0f}%   "
      f"share negative 2018-2023: {(rb.loc[:'2023'].beta<0).mean()*100:.0f}%")

print("\n"+"="*100)
print("(2) SAME rolling 126d beta on (SMH-SPY) for PLACEBO large caps. Does everyone flip?")
print("="*100)
print(f"{'ticker':<7} {'mean b 2018-23':>15} {'%neg 2018-23':>13} {'mean b since2024':>17} {'%neg since2024':>15} {'latest b':>9} {'latest t':>9} {'2026 mean b':>12}")
for tk in ["AAPL","KO","PG","JNJ","PEP","MCD","WMT","V","JPM","XLP","XLV","MSFT","GOOGL","AMZN","META"]:
    if tk not in r.columns: continue
    rbx=roll_beta(r[tk]-r["SPY"],ai)
    pre=rbx.loc[:"2023"]; post=rbx.loc["2024":]; y26=rbx.loc["2026":]
    print(f"{tk:<7} {pre.beta.mean():>15.3f} {(pre.beta<0).mean()*100:>12.0f}% "
          f"{post.beta.mean():>17.3f} {(post.beta<0).mean()*100:>14.0f}% "
          f"{rbx.beta.iloc[-1]:>9.3f} {rbx.t.iloc[-1]:>9.2f} {y26.beta.mean():>12.3f}")

print("\n"+"="*100)
print("(3) FORWARD REVERSAL: iPhone YoY vs NVDA YoY -- recent sub-window")
print("="*100)
panel=pd.DataFrame({
 "aapl_end":["2023-07-01","2023-09-30","2024-03-30","2024-06-29","2024-09-28","2025-03-29",
             "2025-06-28","2025-09-27","2026-03-28","2026-06-27"],
 "iphone":[-2.4,2.8,-10.5,-0.9,5.5,1.9,13.5,6.1,21.7,21.7],
 "nvda":[101.5,205.5,262.1,122.4,93.6,69.2,55.6,62.5,85.2,105.9]})
for k in (10,8,6,5,4):
    s=panel.tail(k); c=np.corrcoef(s.iphone,s.nvda)[0,1]
    t=c*np.sqrt(k-2)/np.sqrt(1-c**2)
    print(f"  last {k:>2} quarters: corr = {c:+.3f}  t = {t:+.2f}   ({s.aapl_end.iloc[0]} -> {s.aapl_end.iloc[-1]})")
print("\n  DIRECTION of the two most recent reported quarters:")
print("    NVDA YoY  +55.6 -> +62.5 -> +85.2 -> +105.9   (RE-ACCELERATING)")
print("    iPhone YoY +13.5 ->  +6.1 -> +21.7 ->  +21.7   (ALSO ACCELERATING)")
