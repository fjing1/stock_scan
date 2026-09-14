import requests, io, pandas as pd, time
H={"User-Agent":"Mozilla/5.0"}
OUT="/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/"
def dl(ds,fn,save):
    u=f"https://huggingface.co/datasets/{ds}/resolve/main/{fn}"
    t0=time.time(); r=requests.get(u,headers=H,timeout=180); dt=time.time()-t0
    print(f"\n===== {ds}/{fn}\n  GET {u}\n  HTTP {r.status_code} {len(r.content):,}B {dt:.2f}s")
    if r.status_code!=200: print("  ",r.text[:200]); return None
    open(OUT+save,"wb").write(r.content)
    return r.content

c=dl("gauss314/options-IV-SP500","data_IV_USA.csv","_data_probe_hf_iv_sp500.csv")
if c:
    df=pd.read_csv(io.BytesIO(c))
    print("  shape:",df.shape)
    print("  columns:",list(df.columns))
    print(df.head(4).to_string()[:1600])
    for cand in ["date","Date","fecha","time","timestamp"]:
        if cand in df.columns:
            d=pd.to_datetime(df[cand],errors="coerce")
            print(f"  date col '{cand}': {d.min()} -> {d.max()}  n_unique={d.nunique()}")
            break
    for cand in ["ticker","symbol","Symbol","act_symbol","underlying"]:
        if cand in df.columns:
            print(f"  symbol col '{cand}': n_unique={df[cand].nunique()} sample={sorted(df[cand].dropna().unique())[:15]}")
            break
    ivc=[x for x in df.columns if "iv" in x.lower() or "vol" in x.lower()]
    print("  IV-ish columns:",ivc)
    for x in ivc[:6]:
        s=pd.to_numeric(df[x],errors="coerce").dropna()
        if len(s): print(f"    {x}: n={len(s)} min={s.min():.4g} med={s.median():.4g} max={s.max():.4g}")

c=dl("CodyJiang/nvda-option-chains","nvda_2020_2022.csv","_data_probe_hf_nvda_chains.csv")
if c:
    df=pd.read_csv(io.BytesIO(c))
    print("  shape:",df.shape); print("  columns:",list(df.columns))
    print(df.head(3).to_string()[:1400])
