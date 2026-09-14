"""Validate the HF per-stock IV dataset: coverage, gaps, survivorship, and an INDEPENDENT
cross-check of its ATM_IV against CBOE's own VXAPL/VXAZN/VXGOG/VXGS/VXIBM IV30 indices."""
import pandas as pd, numpy as np, pickle, os
B="/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/"
USE=["symbol","date","ATM_IV","sOTM_IV","OTM_IV","DOTM_IV","ITM_IV","DITM_IV",
     "calls_open_interest","puts_open_interest","calls_contracts_traded","puts_contracts_traded",
     "contracts_number","expirations_number","hv_20","VIX"]
print("loading 523MB csv (selected cols)...")
iv=pd.read_csv(B+"_data_probe_hf_iv_sp500.csv",usecols=USE,low_memory=False)
iv["date"]=pd.to_datetime(iv["date"])
print("rows",len(iv),"symbols",iv.symbol.nunique(),"dates",iv.date.nunique())
d=sorted(iv.date.unique())
print("date span",pd.Timestamp(d[0]).date(),"->",pd.Timestamp(d[-1]).date())
# trading-day completeness
allbd=pd.bdate_range(d[0],d[-1])
print(f"business days in span: {len(allbd)}  distinct dates present: {len(d)}  ratio {len(d)/len(allbd):.3f}")
gaps=pd.Series(pd.to_datetime(d)).diff().dt.days.dropna()
print("date gap distribution (calendar days):",gaps.value_counts().head(8).to_dict())
print("gaps > 5 days:",int((gaps>5).sum()),"max gap",int(gaps.max()))
# per-symbol coverage
cov=iv.groupby("symbol").size()
print("\nper-symbol row counts: n_sym",len(cov),"median",int(cov.median()),
      " >=900d:",int((cov>=900).sum())," >=500d:",int((cov>=500).sum())," <100d:",int((cov<100).sum()))
# survivorship: names that VANISH before the end (delisted/acquired during the window)
last=iv.groupby("symbol").date.max()
end=pd.Timestamp(d[-1])
dead=last[last < end - pd.Timedelta(days=90)]
print(f"\nSURVIVORSHIP: {len(dead)} of {len(cov)} symbols stop >90d before file end -> dataset DOES carry names through delisting")
print("  examples:",[f"{s}({last[s].date()})" for s in list(dead.sort_values().index[-14:])])
known_dead=["AABA","ATVI","SIVB","FRC","TWTR","VMW","CTXS","XLNX","ZNGA","MRNA","SGEN","HZNP","PXD","ABMD"]
print("  known delisted lookup:",{s:(int(cov[s]),str(last[s].date())) for s in known_dead if s in cov.index})

# ---- independent cross-check vs CBOE single-stock IV30 indices
print("\n"+"="*100)
print("CROSS-CHECK: HF ATM_IV  vs  CBOE's own single-stock IV30 index (independent source)")
print("="*100)
MAP={"AAPL":"vxapl","AMZN":"vxazn","GOOG":"vxgog","GOOGL":"vxgog","GS":"vxgs","IBM":"vxibm"}
for sym,f in MAP.items():
    p=B+f"_data_probe_cboe_{f}.csv"
    if not os.path.exists(p): continue
    cb=pd.read_csv(p); cb["Date"]=pd.to_datetime(cb["Date"]); cb=cb.set_index("Date")["Close"]
    a=iv[iv.symbol==sym].set_index("date")["ATM_IV"].sort_index()
    j=pd.concat([a.rename("hf_atm"),cb.rename("cboe_iv30")],axis=1).dropna()
    if len(j)<20: print(f"{sym:6} overlap {len(j)} rows - too few"); continue
    r=np.corrcoef(j.hf_atm,j.cboe_iv30)[0,1]
    diff=j.hf_atm-j.cboe_iv30
    print(f"{sym:6} overlap n={len(j):>3} {j.index[0].date()}->{j.index[-1].date()}  "
          f"corr={r:.4f}  mean(HF)={j.hf_atm.mean():.2f} mean(CBOE)={j.cboe_iv30.mean():.2f} "
          f"bias={diff.mean():+.2f} sd(diff)={diff.std():.2f}")
    # lead-lag: does HF IV align at t, or is it shifted?
    for k in (-2,-1,0,1,2):
        jj=pd.concat([a.shift(k).rename("h"),cb.rename("c")],axis=1).dropna()
        print(f"        shift {k:+d}: corr={np.corrcoef(jj.h,jj.c)[0,1]:.4f} (n={len(jj)})")
iv.to_pickle(B+"_data_probe_iv_slim.pkl")
print("\nslim frame saved:",B+"_data_probe_iv_slim.pkl", os.path.getsize(B+"_data_probe_iv_slim.pkl")//1_000_000,"MB")
