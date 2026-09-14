"""Fixed version: restrict EVERY model to the same rows (where per-stock IV exists), so the
comparison is apples-to-apples. Also tests smoothed IV, since raw nearest-expiry ATM IV is noisy."""
import numpy as np, pandas as pd, pickle, warnings, sys
warnings.filterwarnings("ignore")
sys.path.insert(0,"/Users/feijing/github.com/stock_scan/Stock_OneClick/backend")
import _move_lib as L, move_prob as MP
from _move_wf_vixiv import FLOOR, fwd_log_sigma
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
    g=ivs[s]
    atm=pd.to_numeric(g["ATM_IV"],errors="coerce").reindex(c.index)
    f["iv_atm"]=conv(atm)
    f["iv_atm5"]=conv(atm.rolling(5,min_periods=3).mean())     # smoothed
    f["iv_atm21"]=conv(atm.rolling(21,min_periods=10).mean())
    f["iv_skew"]=conv(g["OTM_IV"]).reindex(c.index)-f["iv_atm"]
    f["iv_slope"]=conv(g["DITM_IV"]).reindex(c.index)-f["iv_atm"]
    f["iv_vrp"]=f["iv_atm"]-f["ewma97"]                        # variance risk premium proxy
    for h in HOR:
        d=f.copy(); d["y"]=fwd_log_sigma(c,h); d["sym"]=s; d["h"]=h; d["date"]=d.index
        rows.append(d)
P=pd.concat(rows,ignore_index=True)
BASE=MP.FEATS_HAR+MP.FEATS_LEV+MP.FEATS_VOL
NEEDED=BASE+["logvix","iv_atm","iv_atm5","iv_atm21","iv_skew","iv_slope","iv_vrp","y"]
P=P.dropna(subset=NEEDED)          # <-- COMMON SAMPLE for every model
print(f"symbols={len(use)}  common-sample rows={len(P)//len(HOR)} per horizon  "
      f"date span {P.date.min().date()} -> {P.date.max().date()}")
SETS={"A base":BASE,"B +VIX":BASE+["logvix"],"C +ownIV":BASE+["iv_atm"],
      "D +VIX+ownIV":BASE+["logvix","iv_atm"],
      "D5 +VIX+ownIV5":BASE+["logvix","iv_atm5"],
      "D21 +VIX+ownIV21":BASE+["logvix","iv_atm21"],
      "E +skew/slope":BASE+["logvix","iv_atm5","iv_skew","iv_slope"],
      "G +VRP":BASE+["logvix","iv_vrp"]}
def wf(df,cols):
    d=df.sort_values("date")
    qs=sorted(d.date.dt.to_period("Q").unique()); out=[]
    for i in range(6,len(qs)):
        tr=d[d.date.dt.to_period("Q")<qs[i]]; te=d[d.date.dt.to_period("Q")==qs[i]]
        if len(tr)<1500 or len(te)<200: continue
        Xt=np.column_stack([np.ones(len(tr))]+[tr[c].values for c in cols])
        Xe=np.column_stack([np.ones(len(te))]+[te[c].values for c in cols])
        b,*_=np.linalg.lstsq(Xt,tr.y.values,rcond=None)
        out.append(pd.DataFrame({"date":te.date.values,"sym":te.sym.values,"y":te.y.values,
                                 "pred":Xe@b,"ybar_tr":tr.y.mean(),"q":str(qs[i])}))
    return pd.concat(out,ignore_index=True) if out else None
def r2(r): return 1-((r.y-r.pred)**2).sum()/((r.y-r.ybar_tr)**2).sum()
def boot(a,bb,h,reps=2000,seed=11):
    m=a.merge(bb,on=["date","sym"],suffixes=("_a","_b")).sort_values("date")
    dts=np.array(sorted(m.date.unique()))[::h]; m=m[m.date.isin(dts)]
    mo=m.date.values.astype("datetime64[M]"); keys=np.unique(mo)
    ra=((m.y_a-m.pred_a)**2).values; rb=((m.y_b-m.pred_b)**2).values
    ta=((m.y_a-m.ybar_tr_a)**2).values
    idx={k:np.where(mo==k)[0] for k in keys}; rng=np.random.default_rng(seed); ds=[]
    for _ in range(reps):
        pk=np.concatenate([idx[k] for k in rng.choice(keys,len(keys),replace=True)])
        ds.append((rb[pk].sum()-ra[pk].sum())/-ta[pk].sum())
    ds=np.array(ds); return (rb.sum()-ra.sum())/-ta.sum(),np.percentile(ds,2.5),np.percentile(ds,97.5),(ds>0).mean(),len(m),len(keys)
print("\n"+"="*126)
print("OOS R^2, forward log-sigma, pooled single names, COMMON SAMPLE (all models identical rows)")
print("="*126)
res={}
for h in HOR:
    df=P[P.h==h]; st={}; line=f"h={h:>2} "
    for nm,cols in SETS.items():
        r=wf(df,cols); st[nm]=r
        line+=f"{nm.split()[0]}={r2(r):.4f} " if r is not None else f"{nm.split()[0]}=NA "
    print(line+f"(n_test={len(st['A base']) if st['A base'] is not None else 0})")
    res[h]=st
print("\n"+"="*126)
print("INCREMENTAL dR^2 (month-block bootstrap, non-overlapping rows every h days)")
print("="*126)
print(f"{'h':>3} {'comparison':<32} {'dR2':>9} {'95% CI':>22} {'P>0':>6} {'n':>7} {'mo':>4}")
CMP=[("mktVIX | base","A base","B +VIX"),
     ("ownIV(raw) | base+VIX","B +VIX","D +VIX+ownIV"),
     ("ownIV(5d avg) | base+VIX","B +VIX","D5 +VIX+ownIV5"),
     ("ownIV(21d avg) | base+VIX","B +VIX","D21 +VIX+ownIV21"),
     ("ownIV instead of VIX","B +VIX","C +ownIV"),
     ("skew+slope | +ownIV5","D5 +VIX+ownIV5","E +skew/slope"),
     ("VRP(IV-EWMA) | base+VIX","B +VIX","G +VRP")]
for h in HOR:
    st=res[h]
    for lab,a,b in CMP:
        if st.get(a) is None or st.get(b) is None: continue
        pt,lo,hi,pg,n,nb=boot(st[a],st[b],h)
        print(f"{h:>3} {lab:<32} {pt:>+9.4f}  [{lo:>+8.4f},{hi:>+8.4f}] {pg:>6.3f} {n:>7} {nb:>4}")
