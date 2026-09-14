import pandas as pd
pd.set_option('display.width',260)
# All figures from filed income statements (20-F FY2026 p.152 disaggregation; 6-K quarterly ICs)
q = pd.DataFrame([
 # label, external, related_party
 ('Q1FY25 2024-06-30', 815, 124),
 ('Q2FY25 2024-09-30', 652, 192),
 ('Q3FY25 2024-12-31', 698, 285),
 ('Q4FY25 2025-03-31',1019, 222),
 ('Q1FY26 2025-06-30', 725, 328),
 ('Q2FY26 2025-09-30', 713, 422),
 ('Q3FY26 2025-12-31', 904, 338),
 ('Q4FY26 2026-03-31',1079, 411),
 ('Q1FY27 2026-06-30', 901, 388),
], columns=['q','external','related'])
q['total']=q.external+q.related
q['RP_share_%']=(q.related/q.total*100).round(1)
q['ext_yoy_%']=(q.external/q.external.shift(4)-1).mul(100).round(1)
q['RP_yoy_%'] =(q.related /q.related.shift(4)-1).mul(100).round(1)
q['tot_yoy_%']=(q.total   /q.total.shift(4)-1).mul(100).round(1)
print("QUARTERLY: EXTERNAL CUSTOMERS vs RELATED PARTIES (Arm China + SoftBank affiliates), $m")
print(q.to_string(index=False))
print()
ttm_e=q.external.rolling(4).sum(); ttm_r=q.related.rolling(4).sum()
print("TTM external ", ttm_e.iloc[-1], " vs prior TTM ", ttm_e.iloc[-2], f" = {ttm_e.iloc[-1]/ttm_e.iloc[-5]*0+0:.0f}")
print(f"TTM to Jun-26: external {ttm_e.iloc[-1]:.0f}  related {ttm_r.iloc[-1]:.0f}  RP share {ttm_r.iloc[-1]/(ttm_e.iloc[-1]+ttm_r.iloc[-1])*100:.1f}%")
print(f"TTM to Jun-25: external {ttm_e.iloc[-5]:.0f}  related {ttm_r.iloc[-5]:.0f}")
print(f"TTM external yoy: {(ttm_e.iloc[-1]/ttm_e.iloc[-5]-1)*100:+.1f}%   TTM related yoy: {(ttm_r.iloc[-1]/ttm_r.iloc[-5]-1)*100:+.1f}%")
print()
print("ANNUAL (20-F FY2026 Note 4, audited), $m")
a=pd.DataFrame({'FY2024':[1051,380,1458,344,2509,724,3233],
                'FY2025':[1421,418,1763,405,3184,823,4007],
                'FY2026':[1298,1009,2123,490,3421,1499,4920]},
  index=['License&other-EXTERNAL','License&other-RELATED','Royalty-EXTERNAL','Royalty-RELATED',
         'TOTAL EXTERNAL','TOTAL RELATED','TOTAL REVENUE'])
a['FY25 yoy%']=(a.FY2025/a.FY2024-1).mul(100).round(1)
a['FY26 yoy%']=(a.FY2026/a.FY2025-1).mul(100).round(1)
print(a.to_string())
print()
print("--- FY2026 revenue GROWTH attribution (20-F Note 20 named amounts) ---")
tot_g=4920-4007
sbg_g=704.4-145.5; china_g=790.6-670.4
print(f"Total FY26 revenue growth                    +${tot_g:,.0f}m  (+{tot_g/4007*100:.1f}%)")
print(f"  SoftBank affiliate (Consulting Agreement)  +${sbg_g:,.1f}m  = {sbg_g/tot_g*100:.1f}% of all growth")
print(f"  Arm China (IPLA)                           +${china_g:,.1f}m  = {china_g/tot_g*100:.1f}% of all growth")
print(f"  => SoftBank affil + Arm China              +${sbg_g+china_g:,.1f}m  = {(sbg_g+china_g)/tot_g*100:.1f}% of all growth")
r26=4920-704.4-790.6; r25=4007-145.5-670.4
print(f"Revenue EXCLUDING SoftBank affiliate & Arm China: FY26 ${r26:,.1f}m vs FY25 ${r25:,.1f}m = {(r26/r25-1)*100:+.1f}%")
print(f"Of the $704.4m SoftBank-affiliate revenue, ${645.8:,.1f}m ({645.8/704.4*100:.0f}%) sat in UNBILLED contract assets at 2026-03-31")
