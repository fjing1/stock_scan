"""EFFECT SIZE: does per-stock ATM IV add OOS log-vol R^2 over the shipped single-name feature
set (HAR+LEV+VOL) and over market-wide VIX?  Uses the HF per-stock IV history (2019-10..2023-07)
joined to the repo's own daily panel.  Same target/metric conventions as _move_wf_vixiv.py."""
import numpy as np, pandas as pd, pickle, warnings, sys
warnings.filterwarnings("ignore")
sys.path.insert(0,"/Users/feijing/github.com/stock_scan/Stock_OneClick/backend")
import _move_lib as L
import move_prob as MP
from _move_wf_vixiv import FLOOR, fwd_log_sigma, clog, c4

B="/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/"
HOR=(1,5,10,21)
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
C,H,Lo,V=p["Close"],p["High"],p["Low"],p["Volume"]
vix=C["^VIX"]
INDEXY=MP.INDEX_LIKE
syms=[s for s in C.columns if s not in INDEXY and not s.startswith("^")]
have=set(iv.symbol.unique())
use=[s for s in syms if s in have]
print(f"repo panel single names: {len(syms)}   with HF IV coverage: {len(use)}")
ivs={s:g.set_index("date") for s,g in iv[iv.symbol.isin(use)].groupby("symbol")}

def lg(s): return np.log(np.clip(s,FLOOR,None))
rows=[]
for s in use:
    c=C[s].dropna()
    if len(c)<800: continue
    f=MP.build_features(c,H[s].reindex(c.index),Lo[s].reindex(c.index),vix,V[s].reindex(c.index))
    g=ivs[s]
    # IV in % annualised -> per-day sigma, then log (identical transform to logvix)
    conv=lambda x: np.log(np.clip(pd.to_numeric(x,errors="coerce")/100/np.sqrt(252),*MP.VOL_CLIP))
    f["iv_atm"]=conv(g["ATM_IV"]).reindex(c.index)
    f["iv_otm"]=conv(g["OTM_IV"]).reindex(c.index)
    f["iv_itm"]=conv(g["DITM_IV"]).reindex(c.index)
    f["iv_skew"]=f["iv_otm"]-f["iv_atm"]
    f["iv_slope"]=f["iv_itm"]-f["iv_atm"]
    oi=(pd.to_numeric(g["calls_open_interest"],errors="coerce")+
        pd.to_numeric(g["puts_open_interest"],errors="coerce")).reindex(c.index)
    f["l_oi"]=np.log(oi.clip(lower=1))
    for h in HOR:
        y=fwd_log_sigma(c,h)
        d=f.copy(); d["y"]=y; d["sym"]=s; d["h"]=h; d["date"]=d.index
        rows.append(d)
P=pd.concat(rows,ignore_index=True)
print("stacked panel rows:",len(P))
BASE=MP.FEATS_HAR+MP.FEATS_LEV+MP.FEATS_VOL
SETS={
 "A base(HAR+LEV+VOL)":BASE,
 "B +mktVIX"          :BASE+["logvix"],
 "C +ownIV"           :BASE+["iv_atm"],
 "D +VIX+ownIV"       :BASE+["logvix","iv_atm"],
 "E D+skew+slope"     :BASE+["logvix","iv_atm","iv_skew","iv_slope"],
 "F ownIV only+HAR"   :MP.FEATS_HAR+["iv_atm"],
}
def wf(df,cols):
    d=df.dropna(subset=cols+["y"]).sort_values("date")
    if len(d)<2000: return None
    qs=sorted(d.date.dt.to_period("Q").unique())
    out=[]
    for i in range(6,len(qs)):          # >=6 quarters (1.5y) of training
        tr=d[d.date.dt.to_period("Q")<qs[i]]; te=d[d.date.dt.to_period("Q")==qs[i]]
        if len(tr)<1500 or len(te)<200: continue
        Xt=np.column_stack([np.ones(len(tr))]+[tr[c].values for c in cols])
        Xe=np.column_stack([np.ones(len(te))]+[te[c].values for c in cols])
        b,*_=np.linalg.lstsq(Xt,tr.y.values,rcond=None)
        out.append(pd.DataFrame({"date":te.date.values,"sym":te.sym.values,"y":te.y.values,
                                 "pred":Xe@b,"ybar_tr":tr.y.mean(),"q":str(qs[i])}))
    return pd.concat(out,ignore_index=True) if out else None
def r2(r): return 1-((r.y-r.pred)**2).sum()/((r.y-r.ybar_tr)**2).sum()
def boot(a,bb,h,reps=1500,seed=7):
    m=a.merge(bb,on=["date","sym"],suffixes=("_a","_b"))
    m=m.sort_values("date")
    # non-overlapping in time: keep every h-th distinct date
    dts=np.array(sorted(m.date.unique()))[::h]
    m=m[m.date.isin(dts)]
    mo=m.date.values.astype("datetime64[M]")
    keys=np.unique(mo); rng=np.random.default_rng(seed); ds=[]
    ra=(m.y_a-m.pred_a)**2; rb=(m.y_b-m.pred_b)**2
    ta=(m.y_a-m.ybar_tr_a)**2
    idx={k:np.where(mo==k)[0] for k in keys}
    for _ in range(reps):
        pick=np.concatenate([idx[k] for k in rng.choice(keys,len(keys),replace=True)])
        ds.append((rb.values[pick].sum()-ra.values[pick].sum())/-ta.values[pick].sum())
    ds=np.array(ds)
    pt=(rb.sum()-ra.sum())/-ta.sum()
    return pt,np.percentile(ds,2.5),np.percentile(ds,97.5),(ds>0).mean(),len(m),len(keys)

print("\n"+"="*118)
print("OOS R^2 on forward log-sigma, pooled single names, expanding-quarter walk-forward")
print("="*118)
res={}
for h in HOR:
    df=P[P.h==h]
    line=f"h={h:>2}  "
    store={}
    for nm,cols in SETS.items():
        r=wf(df,cols)
        store[nm]=r
        line+=f"{nm.split()[0]}={r2(r):.4f} " if r is not None else f"{nm.split()[0]}=NA "
    n=len(store['A base(HAR+LEV+VOL)']) if store['A base(HAR+LEV+VOL)'] is not None else 0
    print(line+f" (n_test={n})")
    res[h]=store
print("\n"+"="*118)
print("INCREMENTAL dR^2 (block bootstrap by CALENDAR MONTH, non-overlapping rows every h days)")
print("="*118)
print(f"{'h':>3} {'comparison':<34} {'dR2':>9} {'95% CI':>22} {'P(d>0)':>8} {'n':>7} {'blocks':>7}")
for h in HOR:
    st=res[h]
    for lab,a,b in [("VIX | base","A base(HAR+LEV+VOL)","B +mktVIX"),
                    ("ownIV | base","A base(HAR+LEV+VOL)","C +ownIV"),
                    ("ownIV | base+VIX","B +mktVIX","D +VIX+ownIV"),
                    ("skew+slope | base+VIX+ownIV","D +VIX+ownIV","E D+skew+slope")]:
        if st[a] is None or st[b] is None: continue
        pt,lo,hi,pg,n,nb=boot(st[a],st[b],h)
        print(f"{h:>3} {lab:<34} {pt:>+9.4f}  [{lo:>+8.4f},{hi:>+8.4f}] {pg:>8.3f} {n:>7} {nb:>7}")
pickle.dump(res,open(B+"_data_probe_iv_effect.pkl","wb"))
