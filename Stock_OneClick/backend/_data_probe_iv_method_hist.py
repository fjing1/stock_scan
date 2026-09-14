"""Validate iv_snapshot.py's CONSTRUCTION on real historical chains (HF CodyJiang/nvda-option-chains,
OptionsDX format, 2020-2022). Question: does the collector's recipe (interpolated ATM, mid of C/P,
DTE>=7, 30d constant maturity in variance-time) beat the naive alternative (nearest strike, call
side only, nearest expiry incl. 0-DTE) at forecasting NVDA's own forward realized vol?"""
import pandas as pd, numpy as np, pickle, warnings, sys
warnings.filterwarnings("ignore")
sys.path.insert(0,"/Users/feijing/github.com/stock_scan/Stock_OneClick/backend")
import move_prob as MP
from _move_wf_vixiv import fwd_log_sigma
B="/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/"
COLS=["[QUOTE_DATE]","[UNDERLYING_LAST]","[DTE]","[STRIKE]","[C_IV]","[P_IV]","[C_BID]","[P_BID]","[C_VOLUME]","[P_VOLUME]"]
raw=pd.read_csv(B+"_data_probe_hf_nvda_chains.csv",low_memory=False)
raw.columns=[c.strip() for c in raw.columns]
d=raw[[c for c in COLS if c in raw.columns]].copy()
d.columns=[c.strip("[]") for c in d.columns]
for c in d.columns:
    if c!="QUOTE_DATE": d[c]=pd.to_numeric(d[c],errors="coerce")
d["QUOTE_DATE"]=pd.to_datetime(d["QUOTE_DATE"],errors="coerce")
d=d.dropna(subset=["QUOTE_DATE","STRIKE","DTE","UNDERLYING_LAST"])
print(f"chain rows={len(d)}  dates={d.QUOTE_DATE.nunique()}  {d.QUOTE_DATE.min().date()} -> {d.QUOTE_DATE.max().date()}")
print(f"DTE range {d.DTE.min():.0f}-{d.DTE.max():.0f}  expiries/date median={d.groupby('QUOTE_DATE').DTE.nunique().median():.0f}")

def atm_interp(g, spot, ivcol, bidcol):
    g=g[(g[bidcol]>0)&(g[ivcol]>0)]
    if len(g)<4: return np.nan
    lo=g[g.STRIKE<=spot].sort_values("STRIKE"); hi=g[g.STRIKE>spot].sort_values("STRIKE")
    if lo.empty or hi.empty: return np.nan
    a,b=lo.iloc[-1],hi.iloc[0]
    w=0 if b.STRIKE<=a.STRIKE else (spot-a.STRIKE)/(b.STRIKE-a.STRIKE)
    return (1-w)*a[ivcol]+w*b[ivcol]

def interp_var(pts,tgt):
    p=sorted((t,v) for t,v in pts if np.isfinite(t) and np.isfinite(v) and t>0)
    if not p: return np.nan
    if len(p)==1 or tgt<=p[0][0]: return p[0][1]
    if tgt>=p[-1][0]: return p[-1][1]
    for (t1,v1),(t2,v2) in zip(p,p[1:]):
        if t1<=tgt<=t2:
            w1,w2=v1*v1*t1,v2*v2*t2
            return float(np.sqrt(max(w1+(w2-w1)*(tgt-t1)/(t2-t1),0)/tgt))
    return np.nan

rows=[]
for dt,g in d.groupby("QUOTE_DATE"):
    spot=float(g.UNDERLYING_LAST.iloc[0])
    # --- collector recipe
    pts=[]
    for dte,e in g[g.DTE>=7].groupby("DTE"):
        c=atm_interp(e,spot,"C_IV","C_BID"); p=atm_interp(e,spot,"P_IV","P_BID")
        both=[x for x in (c,p) if np.isfinite(x)]
        if both: pts.append((float(dte),float(np.mean(both))))
    good=interp_var(pts,30)
    # --- naive: nearest expiry (any DTE incl 0), nearest strike, CALL only, no liquidity filter
    near=g[g.DTE==g.DTE.min()]
    n2=near.iloc[(near.STRIKE-spot).abs().argsort()[:1]]
    naive=float(n2.C_IV.iloc[0]) if len(n2) else np.nan
    # --- middle: nearest expiry >=7d, nearest strike, call only
    g7=g[g.DTE>=7]
    mid=np.nan
    if len(g7):
        n7=g7[g7.DTE==g7.DTE.min()]
        n3=n7.iloc[(n7.STRIKE-spot).abs().argsort()[:1]]
        mid=float(n3.C_IV.iloc[0]) if len(n3) else np.nan
    rows.append(dict(date=dt,spot=spot,iv30_collector=good,iv_naive=naive,iv_near7_call=mid,n_exp=len(pts)))
S=pd.DataFrame(rows).set_index("date").sort_index()
print(f"\ndaily IV series built: {len(S)} days, n_exp used median={S.n_exp.median():.0f}")
print(S[["iv30_collector","iv_naive","iv_near7_call"]].describe().round(4).to_string())
print("\nday-to-day NOISE (sd of daily change, vol points) -- lower is a cleaner series:")
for c in ["iv30_collector","iv_naive","iv_near7_call"]:
    ch=S[c].diff().dropna()*100
    print(f"  {c:16} sd(dIV)={ch.std():6.3f}  mean|dIV|={ch.abs().mean():6.3f}  n={len(ch)}")

# ---- does it forecast NVDA's own forward vol better?
p=pickle.load(open(B+"_move_panel.pkl","rb"))
C,H,Lo,V=p["Close"],p["High"],p["Low"],p["Volume"]; vix=C["^VIX"]
c=C["NVDA"].dropna()
f=MP.build_features(c,H["NVDA"].reindex(c.index),Lo["NVDA"].reindex(c.index),vix,V["NVDA"].reindex(c.index))
conv=lambda s: np.log(np.clip(s,*MP.VOL_CLIP))
for c_ in ["iv30_collector","iv_naive","iv_near7_call"]:
    f[c_]=conv(S[c_].reindex(c.index)/np.sqrt(252))   # already a decimal annualised vol
BASE=MP.FEATS_HAR+MP.FEATS_LEV+MP.FEATS_VOL
print("\n"+"="*104)
print("Single-name NVDA: OOS R^2 on forward log-sigma, expanding-quarter walk-forward, common sample")
print("="*104)
print(f"{'h':>3} {'base+VIX':>10} {'+collector':>11} {'+naive':>10} {'+near7call':>11}   dR2 collector   dR2 naive")
for h in (5,10,21):
    g=f.copy(); g["y"]=fwd_log_sigma(c,h); g["date"]=g.index
    g=g.dropna(subset=BASE+["logvix","iv30_collector","iv_naive","iv_near7_call","y"])
    qs=sorted(g.date.dt.to_period("Q").unique())
    def run(cols):
        out=[]
        for i in range(3,len(qs)):
            tr=g[g.date.dt.to_period("Q")<qs[i]]; te=g[g.date.dt.to_period("Q")==qs[i]]
            if len(tr)<120 or len(te)<20: continue
            Xt=np.column_stack([np.ones(len(tr))]+[tr[x].values for x in cols])
            Xe=np.column_stack([np.ones(len(te))]+[te[x].values for x in cols])
            b,*_=np.linalg.lstsq(Xt,tr.y.values,rcond=None)
            out.append(pd.DataFrame({"y":te.y.values,"pred":Xe@b,"ybar":tr.y.mean()}))
        r=pd.concat(out) if out else None
        return (1-((r.y-r.pred)**2).sum()/((r.y-r.ybar)**2).sum(),r) if r is not None else (np.nan,None)
    r0,_=run(BASE+["logvix"]); r1,_=run(BASE+["logvix","iv30_collector"])
    r2,_=run(BASE+["logvix","iv_naive"]); r3,_=run(BASE+["logvix","iv_near7_call"])
    print(f"{h:>3} {r0:>10.4f} {r1:>11.4f} {r2:>10.4f} {r3:>11.4f}   {r1-r0:>+13.4f} {r2-r0:>+11.4f}   (n={len(g)})")
