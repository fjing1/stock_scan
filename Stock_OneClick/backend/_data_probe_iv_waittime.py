"""How LONG must we accumulate daily IV snapshots before the feature is honestly validated?
Measured directly by truncating the walk-forward: vary the TRAIN months (which is dead time -- no
OOS evidence at all) and the TEST months, and see when the month-block bootstrap CI clears zero.
Uses the strictly-lagged (t-1) IV, the version a live collector can actually supply."""
import numpy as np, pandas as pd, pickle, warnings, sys
warnings.filterwarnings("ignore")
sys.path.insert(0,"/Users/feijing/github.com/stock_scan/Stock_OneClick/backend")
import move_prob as MP
from _move_wf_vixiv import fwd_log_sigma
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
    f["iv_l1"]=conv(pd.to_numeric(ivs[s]["ATM_IV"],errors="coerce").reindex(c.index).shift(1))
    for h in HOR:
        d=f.copy(); d["y"]=fwd_log_sigma(c,h); d["sym"]=s; d["h"]=h; d["date"]=d.index
        rows.append(d)
P=pd.concat(rows,ignore_index=True)
BASE=MP.FEATS_HAR+MP.FEATS_LEV+MP.FEATS_VOL
P=P.dropna(subset=BASE+["logvix","iv_l1","y"])
print(f"symbols={len(use)}  rows/horizon={len(P)//len(HOR)}  "
      f"IV window {P.date.min().date()} -> {P.date.max().date()}  "
      f"({(P.date.max()-P.date.min()).days/30.44:.1f} calendar months total)")

def run(df,cols,train_m,test_m):
    """Expanding walk-forward refit MONTHLY. train_m = months of history before the first OOS
    month; test_m = how many OOS months we then accumulate."""
    d=df.sort_values("date"); mo=sorted(d.date.dt.to_period("M").unique())
    if len(mo)<train_m+test_m: return None
    out=[]
    for i in range(train_m,min(train_m+test_m,len(mo))):
        tr=d[d.date.dt.to_period("M")<mo[i]]; te=d[d.date.dt.to_period("M")==mo[i]]
        if len(tr)<800 or len(te)<100: continue
        Xt=np.column_stack([np.ones(len(tr))]+[tr[c].values for c in cols])
        Xe=np.column_stack([np.ones(len(te))]+[te[c].values for c in cols])
        b,*_=np.linalg.lstsq(Xt,tr.y.values,rcond=None)
        out.append(pd.DataFrame({"date":te.date.values,"sym":te.sym.values,"y":te.y.values,
                                 "pred":Xe@b,"ybar_tr":tr.y.mean()}))
    return pd.concat(out,ignore_index=True) if out else None

def dr2ci(a,b,h,reps=2000,seed=5):
    m=a.merge(b,on=["date","sym"],suffixes=("_a","_b")).sort_values("date")
    dts=np.array(sorted(m.date.unique()))[::h]; m=m[m.date.isin(dts)]
    mo=m.date.values.astype("datetime64[M]"); keys=np.unique(mo)
    if len(keys)<3: return np.nan,np.nan,np.nan,len(m),len(keys)
    ra=((m.y_a-m.pred_a)**2).values; rb=((m.y_b-m.pred_b)**2).values
    ta=((m.y_a-m.ybar_tr_a)**2).values
    idx={k:np.where(mo==k)[0] for k in keys}; rng=np.random.default_rng(seed); ds=[]
    for _ in range(reps):
        pk=np.concatenate([idx[k] for k in rng.choice(keys,len(keys),replace=True)])
        ds.append((rb[pk].sum()-ra[pk].sum())/-ta[pk].sum())
    ds=np.array(ds)
    return (rb.sum()-ra.sum())/-ta.sum(),np.percentile(ds,2.5),np.percentile(ds,97.5),len(m),len(keys)

TRAIN=12   # months of pure accumulation before the first out-of-sample month
print("\n"+"="*118)
print(f"dR2 of own-stock ATM IV (lagged 1d) over base+VIX, as OOS MONTHS accumulate "
      f"(after {TRAIN} months of train-only accumulation)")
print("="*118)
print(f"{'h':>3} {'testM':>6} {'totalM':>7} {'dR2':>9} {'95% CI':>22} {'CI>0?':>6} {'n_rows':>8} {'blocks':>7}")
verdict={}
for h in HOR:
    df=P[P.h==h]
    for tm in (3,6,9,12,15,18,21,24):
        a=run(df,BASE+["logvix"],TRAIN,tm); b=run(df,BASE+["logvix","iv_l1"],TRAIN,tm)
        if a is None or b is None: continue
        pt,lo,hi,n,nb=dr2ci(a,b,h)
        if not np.isfinite(pt): continue
        ok="YES" if lo>0 else "no"
        if ok=="YES" and h not in verdict: verdict[h]=(tm,TRAIN+tm,pt,lo,hi)
        print(f"{h:>3} {tm:>6} {TRAIN+tm:>7} {pt:>+9.4f}  [{lo:>+8.4f},{hi:>+8.4f}] {ok:>6} {n:>8} {nb:>7}")
    print("-"*118)
print("\nFIRST POINT AT WHICH THE 95% CI EXCLUDES ZERO (train=12 months + N test months):")
for h in HOR:
    if h in verdict:
        tm,tot,pt,lo,hi=verdict[h]
        print(f"  h={h:>2}: {tm} OOS months  -> {tot} calendar months total   dR2={pt:+.4f} [{lo:+.4f},{hi:+.4f}]")
    else:
        print(f"  h={h:>2}: never clears zero within 24 OOS months")
print("\n"+"="*118)
print("SENSITIVITY: how little TRAIN accumulation is enough? (fixed 12 OOS months)")
print("="*118)
print(f"{'h':>3} "+" ".join(f"tr={t:>2}m" for t in (3,6,9,12,18)))
for h in HOR:
    df=P[P.h==h]; cells=[]
    for tr in (3,6,9,12,18):
        a=run(df,BASE+["logvix"],tr,12); b=run(df,BASE+["logvix","iv_l1"],tr,12)
        if a is None or b is None: cells.append("   n/a"); continue
        pt,lo,hi,n,nb=dr2ci(a,b,h)
        cells.append(f"{pt:+.4f}{'*' if lo>0 else ' '}")
    print(f"{h:>3} "+" ".join(f"{c:>7}" for c in cells)+"    (* = 95% CI excludes zero)")
