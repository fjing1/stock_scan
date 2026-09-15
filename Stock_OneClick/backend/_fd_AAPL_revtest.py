"""REVENUE-LEVEL TEST. If AAPL's acceleration were an AI-cycle phenomenon, its revenue growth
should co-move with the AI complex's own REPORTED revenue. Both series come from SEC XBRL
(date-matched YoY, fiscal Q4 derived), so this is a filings-vs-filings comparison."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,"/Users/feijing/github.com/stock_scan/Stock_OneClick/backend")
import _fund_data as fd
pd.set_option("display.width",240)
panel=fd.load()
print("symbols in panel:", sorted(panel.symbol.unique()))

def full_q(sym, concept="revenue"):
    q=fd.series(panel,sym,concept,annual=False); a=fd.series(panel,sym,concept,annual=True)
    if q.empty: return pd.Series(dtype=float)
    rows=q.to_dict("records"); have=set(q.end)
    for _,yr in a.iterrows():
        ins=q[(q.end>yr.end-pd.Timedelta(days=360))&(q.end<=yr.end)]
        if len(ins)==3 and yr.end not in have:
            rows.append({"end":yr.end,"val":float(yr.val)-float(ins.val.sum())})
    d=pd.DataFrame(rows).sort_values("end").drop_duplicates("end",keep="first")
    return d.set_index("end").val

def yoy(s, tol=45):
    out={}
    for e,v in s.items():
        tgt=e-pd.Timedelta(days=365)
        cand=s[(abs(s.index-tgt)<=pd.Timedelta(days=tol))&(s.index<e-pd.Timedelta(days=180))]
        if len(cand): out[e]=v/cand.iloc[-1]-1
    return pd.Series(out)

SYMS=["AAPL","NVDA","AVGO","MSFT","GOOGL","AMD","MU","QCOM","ARM","TSM"]
g={}
for s in SYMS:
    ser=full_q(s)
    if len(ser)>8: g[s]=yoy(ser)
G=pd.DataFrame(g)
# align to calendar quarter to compare across different fiscal calendars
G.index=pd.to_datetime(G.index)
Gq=G.copy(); Gq["cq"]=Gq.index.to_period("Q")
Gq=Gq.groupby("cq").last()
print("\n=== reported revenue YoY by calendar quarter (from each company's own filings) ===")
print((Gq.tail(14)*100).round(1).to_string())

print("\n=== correlation of AAPL revenue-growth with each AI-complex name's revenue-growth ===")
for c in Gq.columns:
    if c=="AAPL": continue
    d=Gq[["AAPL",c]].dropna()
    d2=d.tail(16)
    if len(d)>=8:
        r_all=np.corrcoef(d.AAPL,d[c])[0,1]
        r_rec=np.corrcoef(d2.AAPL,d2[c])[0,1] if len(d2)>=8 else np.nan
        # growth-tracks-growth: change in growth
        dd=d.diff().dropna()
        r_delta=np.corrcoef(dd.AAPL,dd[c])[0,1] if len(dd)>=8 else np.nan
        print(f"  AAPL vs {c:<6} n={len(d):>3}  corr(levels of growth) {r_all:>+5.2f}   last16q {r_rec:>+5.2f}   corr(ACCELERATION) {r_delta:>+5.2f}")

# iPhone-specific: is iPhone growth linked to the AI complex, or to its own product cadence?
pc=pd.read_csv("/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/_fd_AAPL_seg_ProductOrServiceAxis.csv",parse_dates=["end"])
w=(pc.pivot(index="end",columns="mem",values="val")/1e6).sort_index()
ip=yoy(w["IPhoneMember"]); sv=yoy(w["ServiceMember"])
T=pd.DataFrame({"iPhone_yoy":ip,"Services_yoy":sv})
T.index=pd.to_datetime(T.index); T["cq"]=T.index.to_period("Q"); T=T.groupby("cq").last()
J=T.join(Gq[["NVDA","AVGO","MU"]], how="inner").dropna()
print(f"\n=== iPhone vs Services vs AI-complex revenue growth, by calendar quarter (n={len(J)}) ===")
print((J*100).round(1).to_string())
for c in ["NVDA","AVGO","MU"]:
    print(f"  corr(iPhone YoY, {c} YoY) = {np.corrcoef(J.iPhone_yoy,J[c])[0,1]:+.2f}   "
          f"corr(Services YoY, {c} YoY) = {np.corrcoef(J.Services_yoy,J[c])[0,1]:+.2f}    n={len(J)}")
