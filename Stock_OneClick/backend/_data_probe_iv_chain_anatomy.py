"""Measure the decisions the collector has to make, on real yfinance chains:
   (a) nearest-strike vs interpolated ATM   (b) call vs put vs mid disagreement
   (c) near-expiry IV behaviour vs DTE      (d) how illiquid strikes corrupt IV."""
import numpy as np, pandas as pd, yfinance as yf, warnings, time, datetime as dt
warnings.filterwarnings("ignore")
pd.set_option("display.width",200)
SYMS=["AAPL","MSFT","NVDA","SPY","JNJ","KLAC","EPAM","YELP","FORM","AVAV"]
today=pd.Timestamp.today().normalize()
recs=[]
for sym in SYMS:
    t=yf.Ticker(sym)
    try: exps=t.options
    except Exception as e: print(sym,"no options",e); continue
    try: spot=float(t.fast_info["last_price"])
    except Exception:
        spot=float(t.history(period="5d")["Close"].iloc[-1])
    print(f"\n{'='*112}\n{sym}  spot={spot:.2f}  n_expiries={len(exps)}  first 8: {exps[:8]}")
    for e in exps[:8]:
        dte=(pd.Timestamp(e)-today).days
        try: ch=t.option_chain(e)
        except Exception as ex: print("   ",e,"ERR",ex); continue
        for side,df in (("C",ch.calls),("P",ch.puts)):
            d=df.copy()
            d["iv"]=pd.to_numeric(d.impliedVolatility,errors="coerce")
            d["oi"]=pd.to_numeric(d.openInterest,errors="coerce").fillna(0)
            d["vol"]=pd.to_numeric(d.volume,errors="coerce").fillna(0)
            d["bid"]=pd.to_numeric(d.bid,errors="coerce"); d["ask"]=pd.to_numeric(d.ask,errors="coerce")
            d["moneyness"]=d.strike/spot-1
            d=d.dropna(subset=["iv"])
            d=d[d.iv>0]
            if d.empty: continue
            near=d.iloc[(d.strike-spot).abs().argsort()[:1]]
            # interpolate IV at K=spot between the two straddling strikes
            below=d[d.strike<=spot].sort_values("strike"); above=d[d.strike>spot].sort_values("strike")
            iv_interp=np.nan; k_lo=k_hi=np.nan
            if len(below) and len(above):
                lo=below.iloc[-1]; hi=above.iloc[0]; k_lo,k_hi=lo.strike,hi.strike
                w=(spot-lo.strike)/(hi.strike-lo.strike) if hi.strike>lo.strike else 0
                iv_interp=(1-w)*lo.iv+w*hi.iv
            n0=int(((d.bid==0)|(d.bid.isna())).sum())
            recs.append(dict(sym=sym,exp=e,dte=dte,side=side,spot=spot,
                             n_strikes=len(d),
                             iv_near=float(near.iv.iloc[0]),k_near=float(near.strike.iloc[0]),
                             iv_interp=iv_interp,k_lo=k_lo,k_hi=k_hi,
                             oi_near=float(near.oi.iloc[0]),vol_near=float(near.vol.iloc[0]),
                             spr_near=float((near.ask.iloc[0]-near.bid.iloc[0])) if pd.notna(near.ask.iloc[0]) else np.nan,
                             mid_near=float((near.ask.iloc[0]+near.bid.iloc[0])/2) if pd.notna(near.ask.iloc[0]) else np.nan,
                             zero_bid_frac=n0/len(d),
                             oi_total=float(d.oi.sum()),vol_total=float(d.vol.sum())))
    time.sleep(0.3)
R=pd.DataFrame(recs)
R.to_csv("/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/_data_probe_iv_chain_anatomy.csv",index=False)
print("\n\n"+"#"*112)
print("(a) NEAREST-STRIKE vs INTERPOLATED ATM  -- |difference| in vol points")
print("#"*112)
R["d_interp"]=(R.iv_near-R.iv_interp).abs()*100
R["k_gap_pct"]=(R.k_hi-R.k_lo)/R.spot*100
print(R.groupby("sym")[["d_interp","k_gap_pct"]].describe().round(3)[[("d_interp","mean"),("d_interp","50%"),("d_interp","max"),("k_gap_pct","mean")]].to_string())
print(f"\nOVERALL |nearest - interp| : mean={R.d_interp.mean():.3f} median={R.d_interp.median():.3f} p90={R.d_interp.quantile(.9):.3f} max={R.d_interp.max():.3f} vol pts")
print("\n(b) CALL vs PUT ATM IV DISAGREEMENT (interpolated), vol points")
piv=R.pivot_table(index=["sym","exp","dte"],columns="side",values="iv_interp")
piv["diff_pts"]=(piv.C-piv.P)*100; piv["mid"]=(piv.C+piv.P)/2
print(f"  n pairs={piv.diff_pts.notna().sum()}  mean(C-P)={piv.diff_pts.mean():+.3f}  "
      f"median={piv.diff_pts.median():+.3f}  mean|C-P|={piv.diff_pts.abs().mean():.3f}  "
      f"p90|C-P|={piv.diff_pts.abs().quantile(.9):.3f}  max|C-P|={piv.diff_pts.abs().max():.3f}")
print("  by DTE bucket:")
pr=piv.reset_index(); pr["b"]=pd.cut(pr.dte,[-1,2,7,21,45,120,10000],labels=["0-2","3-7","8-21","22-45","46-120","120+"])
print(pr.groupby("b").agg(n=("diff_pts","size"),mean_CmP=("diff_pts","mean"),mean_abs=("diff_pts",lambda s:s.abs().mean()),mean_mid=("mid","mean")).round(4).to_string())
print("\n(c) NEAR-EXPIRY IV vs DTE (mid of C/P interp), per symbol -- the 0-DTE anomaly")
tab=pr.pivot_table(index="dte",columns="sym",values="mid").round(4)*100
print(tab.head(9).to_string())
print("\n(d) LIQUIDITY at the ATM strike, by DTE bucket")
R["b"]=pd.cut(R.dte,[-1,2,7,21,45,120,10000],labels=["0-2","3-7","8-21","22-45","46-120","120+"])
print(R.groupby("b").agg(n=("iv_near","size"),oi_near_med=("oi_near","median"),vol_near_med=("vol_near","median"),
     zero_bid_frac=("zero_bid_frac","mean"),spr_near_med=("spr_near","median"),mid_near_med=("mid_near","median"),
     n_strikes_med=("n_strikes","median")).round(4).to_string())
print("\n  relative ATM spread (ask-bid)/mid by DTE bucket:")
R["rel"]=R.spr_near/R.mid_near
print(R.groupby("b")["rel"].describe().round(3)[["count","25%","50%","75%","max"]].to_string())
print("\n  per-symbol total chain OI/volume (liquidity ranking):")
print(R.groupby("sym").agg(oi=("oi_total","sum"),vol=("vol_total","sum"),n_exp=("exp","nunique")).sort_values("oi").to_string())
