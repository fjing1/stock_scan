import pandas as pd
pd.set_option('display.width',260)
P=239.01; SH=1068.0; NETCASH=3888.0
REV_TTM=5156.0; EPS_FY28_NG=3.056; EPS_FY27_NG=2.224; EPS_TTM_GAAP=0.97; EPS_TTM_NG=1.77-0.35+0.45
peers=pd.DataFrame([
 ('ARM' ,54.19, 78.22,243.89,0.22,0.08),
 ('NVDA',17.29, 13.51, 26.70,1.06,0.66),
 ('AVGO',19.79, 17.78, 43.86,0.86,0.54),
 ('QCOM', 4.45, 17.65, 20.59,-0.04,0.19),
 ('CDNS',13.84, 29.23, 55.37,0.24,0.29),
 ('SNPS', 8.86, 21.76, 66.68,0.42,0.15),
 ('TXN' ,12.98, 24.75, 40.03,0.23,0.43),
 ('MRVL',22.05, 32.37, 72.46,0.36,0.17),
],columns=['tk','EV_S','fwdPE','trailPE','rev_g','GAAP_opm']).set_index('tk')
peers['EV_S_per_growth_pt']=(peers.EV_S/(peers.rev_g*100)).round(2)
print("PEER SET (yfinance .info, 2026-09-14; ARM cross-checked to filings)")
print(peers.to_string())
print()
print(f"ARM TTM non-GAAP EPS = ${EPS_TTM_NG:.2f}  -> trailing non-GAAP P/E {P/EPS_TTM_NG:.0f}x")
print(f"ARM TTM GAAP EPS     = ${EPS_TTM_GAAP:.2f}  -> trailing GAAP P/E {P/EPS_TTM_GAAP:.0f}x")
print(f"ARM consensus non-GAAP EPS: FY27 ${EPS_FY27_NG:.2f} (P/E {P/EPS_FY27_NG:.0f}x)  FY28 ${EPS_FY28_NG:.2f} (P/E {P/EPS_FY28_NG:.0f}x)")
print()
print("ARM implied share price at PEER multiples")
rows=[]
ex=peers.drop('ARM')
for tk,r in ex.iterrows():
    px_evs=(r.EV_S*REV_TTM+NETCASH)/SH
    rows.append(dict(peer=tk, EV_S=r.EV_S, ARM_px_at_that_EV_S=round(px_evs,0),
                     fwdPE=r.fwdPE, ARM_px_on_FY28ng=round(r.fwdPE*EPS_FY28_NG,0)))
d=pd.DataFrame(rows)
print(d.to_string(index=False))
print()
print(f"median peer EV/S {ex.EV_S.median():.1f}x -> ARM ${(ex.EV_S.median()*REV_TTM+NETCASH)/SH:.0f}")
print(f"MAX peer EV/S    {ex.EV_S.max():.1f}x -> ARM ${(ex.EV_S.max()*REV_TTM+NETCASH)/SH:.0f}   (MRVL, growing 36% vs ARM 22%)")
print(f"median peer fwdPE {ex.fwdPE.median():.1f}x -> ARM on FY28 nonGAAP ${ex.fwdPE.median()*EPS_FY28_NG:.0f}")
print(f"MAX peer fwdPE    {ex.fwdPE.max():.1f}x -> ARM on FY28 nonGAAP ${ex.fwdPE.max()*EPS_FY28_NG:.0f}")
print()
print("="*80)
print("DRAWDOWN DECOMPOSITION  (ATH 439.46 on 2026-06-18 -> 239.01 on 2026-09-14)")
ath=439.46; sh_ath=1064.0; nc_ath=3601.0
ev_ath=ath*sh_ath-nc_ath; evs_ath=ev_ath/4920.0
ev_now=P*SH-NETCASH;      evs_now=ev_now/REV_TTM
print(f"  At ATH: mcap ${ath*sh_ath/1000:.1f}bn, EV ${ev_ath/1000:.1f}bn, EV/Sales(FY26 actual 4,920) = {evs_ath:.1f}x")
print(f"          P/E on FY26 non-GAAP $1.77 = {ath/1.77:.0f}x ; on FY26 GAAP $0.85 = {ath/0.85:.0f}x")
print(f"  Now   : mcap ${P*SH/1000:.1f}bn, EV ${ev_now/1000:.1f}bn, EV/Sales(TTM 5,156) = {evs_now:.1f}x")
print(f"          P/E on TTM non-GAAP $1.87 = {P/1.87:.0f}x ; on TTM GAAP $0.97 = {P/0.97:.0f}x")
print()
fund=5156.0/4920.0-1; mult=evs_now/evs_ath-1
print(f"  Fundamentals (TTM revenue) over the window : {fund*100:+.1f}%")
print(f"  TTM non-GAAP EPS 1.77 -> 1.87              : {(1.87/1.77-1)*100:+.1f}%")
print(f"  Consensus FY27 non-GAAP EPS, last 90 days  : 2.177 -> 2.224 = {(2.22407/2.17688-1)*100:+.1f}%  (27 up vs 4 down, 30d)")
print(f"  EV/Sales multiple                          : {mult*100:+.1f}%")
print(f"  check: (1{fund:+.4f}) x (1{mult:+.4f}) = {(1+fund)*(1+mult):.4f}  vs actual price ratio {P/ath:.4f}")
print()
print(f"  => {mult/(P/ath-1)*100*(P/ath-1)/(P/ath-1):.0f}% attribution: essentially ALL of the -45.6% is de-rating.")
