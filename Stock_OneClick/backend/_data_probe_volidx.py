"""Does Yahoo serve HISTORY for CBOE per-stock / per-ETF implied-vol indices?
These are VIX-methodology 30-day IV on a single underlying -> real per-stock IV history, free."""
import yfinance as yf, pandas as pd, warnings, time
warnings.filterwarnings("ignore")
CAND = {
 # CBOE single-stock vol indices (VIX methodology on one equity)
 "^VXAPL":"Apple","^VXAZN":"Amazon","^VXGOG":"Google","^VXIBM":"IBM","^VXGS":"Goldman",
 "VXAPL":"Apple(no caret)","^VXAPLC":"AppleC",
 # ETF / sector vol indices
 "^VXN":"Nasdaq100","^VXD":"Dow","^RVX":"Russell2000","^VVIX":"VIX-of-VIX","^VIX9D":"VIX 9d",
 "^VIX3M":"VIX 3m","^VIX6M":"VIX 6m","^VXTLT":"TLT","^OVX":"Oil/USO","^GVZ":"Gold/GLD",
 "^EVZ":"EuroFX","^VXEEM":"EEM","^VXFXI":"FXI","^VXEWZ":"EWZ","^VXSLV":"SLV","^VXGDX":"GDX",
 "^VXXLE":"XLE","^VIX":"SPX(control)","^SKEW":"SKEW","^VPD":"PutDelta","^VPN":"PutNaked",
}
rows=[]
t0=time.time()
for t,nm in CAND.items():
    try:
        h=yf.Ticker(t).history(period="max",auto_adjust=False)
        if h is None or h.empty:
            rows.append((t,nm,0,"","",None,None)); continue
        c=h["Close"].dropna()
        rows.append((t,nm,len(c),str(c.index[0].date()),str(c.index[-1].date()),round(float(c.iloc[-1]),2),round(float(c.mean()),2)))
    except Exception as e:
        rows.append((t,nm,-1,type(e).__name__,"",None,None))
print(f"elapsed {time.time()-t0:.1f}s for {len(CAND)} tickers\n")
print(f"{'ticker':9} {'name':14} {'rows':>6} {'first':>11} {'last':>11} {'lastval':>8} {'mean':>7}")
for r in rows:
    print(f"{r[0]:9} {r[1]:14} {r[2]:>6} {str(r[3]):>11} {str(r[4]):>11} {str(r[5]):>8} {str(r[6]):>7}")
ok=[r for r in rows if r[2]>100]
print(f"\nUSABLE HISTORY ({len(ok)} of {len(CAND)}):")
for r in sorted(ok,key=lambda x:-x[2]): print(f"  {r[0]:9} {r[2]:>6} rows  {r[3]} -> {r[4]}  last={r[5]}")
