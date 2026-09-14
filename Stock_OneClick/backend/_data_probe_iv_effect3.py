"""Robustness: (1) does the ownIV edge survive using STRICTLY LAGGED IV (t-1, t-2)? -> rules out
any same-day/lookahead timestamp error in the dataset. (2) per-quarter stability. (3) is it just
COVID? (4) does it hold on the names that later delisted?"""
import numpy as np, pandas as pd, pickle, warnings, sys
warnings.filterwarnings("ignore")
sys.path.insert(0,"/Users/feijing/github.com/stock_scan/Stock_OneClick/backend")
import move_prob as MP
from _move_wf_vixiv import fwd_log_sigma
B="/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/"
HOR=(5,10,21)
def _load_iv():
    """Slim per-stock IV frame. Rebuild from the raw 523MB HF csv if the cache is absent."""
    import os
    p = B + "_data_probe_iv_slim.pkl"
    if os.path.exists(p):
        return pd.read_pickle(p)
    use = ["symbol", "date", "ATM_IV", "sOTM_IV", "OTM_IV", "DOTM_IV", "ITM_IV", "DITM_IV",
           "calls_open_interest", "puts_open_interest", "calls_contracts_traded",
           "puts_contracts_traded", "contracts_number", "expirations_number", "hv_20", "VIX"]
    d = pd.read_csv(B + "_data_probe_hf_iv_sp500.csv", usecols=use, low_memory=False)
    d["date"] = pd.to_datetime(d["date"])
    d.to_pickle(p)
    return d


iv = _load_iv()
p=pickle.load(open(B+"_move_panel.pkl","rb"))
C,H,Lo,V=p["Close"],p["High"],p["Low"],p["Volume"]; vix=C["^VIX"]
syms=[s for s in C.columns if s not in MP.INDEX_LIKE and not s.startswith("^")]
have=set(iv.symbol.unique()); use=[s for s in syms if s in have]
ivs={s:g.set_index("date") for s,g in iv[iv.symbol.isin(use)].groupby("symbol")}
conv=lambda x: np.log(np.clip(pd.to_numeric(x,errors="coerce")/100/np.sqrt(252),*MP.VOL_CLIP))
rows=[]
for s in use:
    c=C[s].dropna()
    if len(c)<800: continue
    f=MP.build_features(c,H[s].reindex(c.index),Lo[s].reindex(c.index),vix,V[s].reindex(c.index))
    atm=pd.to_numeric(ivs[s]["ATM_IV"],errors="coerce").reindex(c.index)
    f["iv_l0"]=conv(atm); f["iv_l1"]=conv(atm.shift(1)); f["iv_l2"]=conv(atm.shift(2))
    f["iv_l5"]=conv(atm.shift(5))
    for h in HOR:
        d=f.copy(); d["y"]=fwd_log_sigma(c,h); d["sym"]=s; d["h"]=h; d["date"]=d.index
        rows.append(d)
P=pd.concat(rows,ignore_index=True)
BASE=MP.FEATS_HAR+MP.FEATS_LEV+MP.FEATS_VOL
P=P.dropna(subset=BASE+["logvix","iv_l0","iv_l1","iv_l2","iv_l5","y"])
print(f"common rows/horizon={len(P)//len(HOR)}  span {P.date.min().date()}->{P.date.max().date()}")
def wf(df,cols):
    d=df.sort_values("date"); qs=sorted(d.date.dt.to_period("Q").unique()); out=[]
    for i in range(6,len(qs)):
        tr=d[d.date.dt.to_period("Q")<qs[i]]; te=d[d.date.dt.to_period("Q")==qs[i]]
        if len(tr)<1500 or len(te)<200: continue
        Xt=np.column_stack([np.ones(len(tr))]+[tr[c].values for c in cols])
        Xe=np.column_stack([np.ones(len(te))]+[te[c].values for c in cols])
        b,*_=np.linalg.lstsq(Xt,tr.y.values,rcond=None)
        out.append(pd.DataFrame({"date":te.date.values,"sym":te.sym.values,"y":te.y.values,
                                 "pred":Xe@b,"ybar_tr":tr.y.mean(),"q":str(qs[i])}))
    return pd.concat(out,ignore_index=True)
def dr2(a,b,h,mask=None):
    m=a.merge(b,on=["date","sym","q"],suffixes=("_a","_b")).sort_values("date")
    dts=np.array(sorted(m.date.unique()))[::h]; m=m[m.date.isin(dts)]
    if mask is not None: m=m[m.sym.isin(mask)]
    if len(m)<50: return np.nan,0
    ra=((m.y_a-m.pred_a)**2).sum(); rb=((m.y_b-m.pred_b)**2).sum()
    ta=((m.y_a-m.ybar_tr_a)**2).sum()
    return (rb-ra)/-ta, len(m)
# names that stopped >90d before file end == delisted during window
last=iv.groupby("symbol").date.max(); end=iv.date.max()
dead=set(last[last<end-pd.Timedelta(days=90)].index)&set(use)
print(f"delisted-during-window names in the 221: {len(dead)} -> {sorted(dead)}")
print("\n"+"="*104)
print("(1) LOOKAHEAD TEST: dR2 of own-stock ATM IV over base+VIX, at increasing LAG")
print("="*104)
print(f"{'h':>3} {'lag0':>9} {'lag1':>9} {'lag2':>9} {'lag5':>9}   (all vs same base+VIX model)")
store={}
for h in HOR:
    df=P[P.h==h]
    base=wf(df,BASE+["logvix"]); store[h]=(df,base)
    vals=[]
    for c in ["iv_l0","iv_l1","iv_l2","iv_l5"]:
        m=wf(df,BASE+["logvix",c]); v,_=dr2(base,m,h); vals.append(v)
    print(f"{h:>3} "+" ".join(f"{v:>+9.4f}" for v in vals))
print("\n"+"="*104)
print("(2) PER-QUARTER dR2 (lag-1 IV, the strictly-tradeable version) -- is one quarter driving it?")
print("="*104)
for h in HOR:
    df,base=store[h]
    m1=wf(df,BASE+["logvix","iv_l1"])
    mm=base.merge(m1,on=["date","sym","q"],suffixes=("_a","_b")).sort_values("date")
    dts=np.array(sorted(mm.date.unique()))[::h]; mm=mm[mm.date.isin(dts)]
    per=[]
    for q,g in mm.groupby("q"):
        ra=((g.y_a-g.pred_a)**2).sum(); rb=((g.y_b-g.pred_b)**2).sum()
        ta=((g.y_a-g.ybar_tr_a)**2).sum()
        per.append((q,(rb-ra)/-ta if ta>0 else np.nan,len(g)))
    pos=sum(1 for _,v,_ in per if v>0)
    print(f"h={h:>2}  {pos}/{len(per)} quarters positive   "+"  ".join(f"{q[-2:]}:{v:+.3f}" for q,v,_ in per))
print("\n"+"="*104)
print("(3) SUBSAMPLE: dR2 (lag-1) on delisted-during-window names only vs survivors only")
print("="*104)
surv=set(use)-dead
for h in HOR:
    df,base=store[h]
    m1=wf(df,BASE+["logvix","iv_l1"])
    a,na=dr2(base,m1,h,mask=dead); b,nb=dr2(base,m1,h,mask=surv)
    print(f"h={h:>2}  delisted-names dR2={a:+.4f} (n={na})   survivor-names dR2={b:+.4f} (n={nb})")
