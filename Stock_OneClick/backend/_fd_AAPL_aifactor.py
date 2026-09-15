"""Is AAPL's negative loading on the AI complex stable, or a recent artifact?
AI factor = SMH minus SPY (a market-neutral AI/semis basket). Rolling 126-day OLS.
Then: how much of the acceleration is already in the 38.1x multiple, arithmetically."""
import numpy as np, pandas as pd
pd.set_option("display.width",200)
r=pd.read_pickle("/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/_fd_AAPL_rets.pkl").dropna()
ai = r["SMH"]-r["SPY"]
nv = r["NVDA"]-r["SPY"]
def roll_beta(y, x, w=126):
    out={}
    for i in range(w, len(y)+1):
        yy=y.iloc[i-w:i].values; xx=x.iloc[i-w:i].values
        X=np.column_stack([np.ones(w),xx])
        b,*_=np.linalg.lstsq(X,yy,rcond=None)
        e=yy-X@b; s2=(e@e)/(w-2); se=np.sqrt(s2*np.linalg.inv(X.T@X)[1,1])
        out[y.index[i-1]]=(b[1], b[1]/se)
    return pd.DataFrame(out, index=["beta","t"]).T
# dependent variable: AAPL's market-neutral return
ex = r["AAPL"]-r["SPY"]
rb_ai = roll_beta(ex, ai); rb_nv = roll_beta(ex, nv)
print("=== rolling 126d beta of (AAPL-SPY) on (SMH-SPY) — the AI factor ===")
yr = rb_ai.groupby(rb_ai.index.year).agg({"beta":["mean","min","max"],"t":"mean"}).round(2)
print(yr.to_string())
print("\nlatest values:")
print(rb_ai.tail(3).round(3).to_string(), "\n(NVDA factor)\n", rb_nv.tail(3).round(3).to_string())
print(f"\nshare of days since 2024 with NEGATIVE AI-factor beta: {(rb_ai.loc['2024':].beta<0).mean()*100:.0f}%"
      f"   (t<-2 on {(rb_ai.loc['2024':].t<-2).mean()*100:.0f}% of days)")
print(f"share of days 2018-2023 with negative AI-factor beta: {(rb_ai.loc[:'2023'].beta<0).mean()*100:.0f}%")

print("\n" + "="*100)
print("HOW MUCH IS PRICED: arithmetic on the point-in-time multiple")
h=pd.read_csv("/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/_fd_AAPL_pe_history.csv",parse_dates=["Date"]).set_index("Date")
cur_pe=h.pe.iloc[-1]; cur_eps=h.eps_ttm_split_adj.iloc[-1]; px=h.close.iloc[-1]
for lab, target in [("its own 2019-2026 median (29.8x)",29.8),("its own 2021-2026 median (31.2x)",31.2),
                    ("its own 2024-2026 median (34.6x)",34.6),("2010-2026 median (18.9x)",18.9)]:
    need=cur_pe/target-1
    print(f"  to de-rate to {lab}: EPS must rise {need:+.0%} with the price flat -> at +20%/yr that takes "
          f"{np.log(1+need)/np.log(1.20):.1f} yrs; at +10%/yr {np.log(1+need)/np.log(1.10):.1f} yrs")
print(f"  or, holding EPS at {cur_eps:.2f}, those multiples imply prices of: " +
      ", ".join(f"{t:.1f}x=${cur_eps*t:.0f} ({cur_eps*t/px-1:+.0%})" for t in (29.8,31.2,34.6,38.1,43.8)))
