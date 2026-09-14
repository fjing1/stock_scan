"""Verify the collector: (1) its SPY/QQQ iv30 against CBOE's own ^VIX/^VXN (independent),
(2) store idempotency, (3) sanity of the constant-maturity interpolation vs raw expiries."""
import pandas as pd, numpy as np, yfinance as yf, sys, warnings, subprocess
warnings.filterwarnings("ignore")
sys.path.insert(0,"/Users/feijing/github.com/stock_scan/Stock_OneClick/backend")
import iv_snapshot as S
B="/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/"
h=pd.read_csv(B+"_iv_history.csv")
print("store rows:",len(h),"cols:",len(h.columns))
print(h[["date","symbol","spot","iv30","iv60","iv90","ts_slope","iv_near","dte_near",
         "skew25","cp_spread_pts","n_exp_used","atm_oi","quality"]].to_string(index=False))

print("\n"+"="*104)
print("(1) EXTERNAL CHECK: collector iv30 on SPY/QQQ vs CBOE's own 30-day vol index for the same underlying")
print("="*104)
for etf,idx in [("SPY","^VIX"),("QQQ","^VXN"),("IWM","^RVX")]:
    r=h[h.symbol==etf]
    if r.empty:
        row=S.iv_term_structure(etf); ours=row["iv30"]
    else: ours=float(r.iv30.iloc[-1])
    try:
        v=yf.Ticker(idx).history(period="5d")["Close"].dropna()
        cb=float(v.iloc[-1])/100
    except Exception as e: print(etf,"idx fail",e); continue
    print(f"  {etf:4} collector iv30={100*ours:6.2f}%   {idx}={100*cb:6.2f}%   ratio ATM/VIX={ours/cb:.3f}")
print("  (ATM IV should sit BELOW a VIX-style index: VIX integrates the whole skew strip, ours is ATM only.")
print("   A consistent ratio across independent ETF/index pairs is the evidence the construction is right.)")

print("\n"+"="*104)
print("(2) IDEMPOTENCY: re-run on 3 symbols, store must not duplicate (date,symbol)")
print("="*104)
n0=len(h)
S.collect(["AAPL","MSFT","NVDA"],verbose=False)
h2=pd.read_csv(B+"_iv_history.csv")
dup=h2.duplicated(subset=["date","symbol"]).sum()
print(f"  rows before={n0} after={len(h2)}  duplicate (date,symbol) rows={dup}")

print("\n"+"="*104)
print("(3) INTERPOLATION SANITY: raw per-expiry ATM mids vs the interpolated constant-maturity curve")
print("="*104)
for sym in ["AAPL","YELP"]:
    t=yf.Ticker(sym); today=pd.Timestamp.today().normalize()
    try: spot=float(t.fast_info["last_price"])
    except Exception: spot=float(t.history(period="5d")["Close"].iloc[-1])
    pts=[]
    for e in t.options[:10]:
        d=(pd.Timestamp(e)-today).days
        if not (S.MIN_DTE<=d<=S.MAX_DTE): continue
        ch=t.option_chain(e)
        c,_,_,_=S._atm_iv_one_side(ch.calls,spot); p,_,_,_=S._atm_iv_one_side(ch.puts,spot)
        both=[x for x in (c,p) if np.isfinite(x)]
        if both: pts.append((d,float(np.mean(both))))
    print(f"\n  {sym} spot={spot:.2f}  raw expiry ATM mids: "+", ".join(f"{d}d={100*v:.2f}%" for d,v in sorted(pts)))
    for tg in (30,60,90):
        print(f"    interp {tg}d = {100*S._interp_var_time(pts,tg):.2f}%   "
              f"(nearest raw expiry to {tg}d = {100*min(pts,key=lambda x:abs(x[0]-tg))[1]:.2f}% "
              f"at {min(pts,key=lambda x:abs(x[0]-tg))[0]}d)")
