import numpy as np, pandas as pd
pd.set_option('display.width',260)

# ---------- inputs, all from filings ----------
P=239.01; SH=1068.0                    # BS 2026-06-30 shares issued & outstanding
MCAP=P*SH; NETCASH=3888.0; EV=MCAP-NETCASH
REV0=5156.0                            # TTM revenue to 2026-06-30
print(f"Mkt cap ${MCAP/1000:.1f}bn | net cash ${NETCASH/1000:.2f}bn | EV ${EV/1000:.1f}bn | TTM rev ${REV0:,.0f}m")

# TTM economics (20-F FY26 + 6-K Q1FY27)
sbc_ttm   = 1052-241+343          # 1154
nongaap_fcf_ttm = 1397.0
cash_withhold_ttm = 529-85+278    # 722  (financing line, NOT in FCF)
gaap_op_ttm = 900-114+91          # 877
nongaap_op_ttm = 2115-412+531     # 2234
for lbl,v in [('SBC expense TTM',sbc_ttm),('cash SBC withholding TTM',cash_withhold_ttm),
              ('GAAP op income TTM',gaap_op_ttm),('non-GAAP op income TTM',nongaap_op_ttm)]:
    print(f"  {lbl:28s} ${v:,.0f}m  ({v/REV0*100:.1f}% of rev)")
print(f"  economic FCF (nonGAAP FCF - SBC)   ${nongaap_fcf_ttm-sbc_ttm:,.0f}m ({(nongaap_fcf_ttm-sbc_ttm)/REV0*100:.1f}%)")
print(f"  cash FCF (nonGAAP FCF - withhold)  ${nongaap_fcf_ttm-cash_withhold_ttm:,.0f}m ({(nongaap_fcf_ttm-cash_withhold_ttm)/REV0*100:.1f}%)")
print()

def dcf(g, term_fcf_margin, wacc, years=10, tg=0.03, start_margin=None):
    """revenue CAGR g for `years`, FCF margin ramps linearly start->terminal, then perpetuity."""
    if start_margin is None: start_margin=(nongaap_fcf_ttm-cash_withhold_ttm)/REV0
    pv=0.0; rev=REV0
    for t in range(1,years+1):
        rev*= (1+g)
        m = start_margin + (term_fcf_margin-start_margin)*t/years
        pv += rev*m/(1+wacc)**t
    tv = rev*(1+tg)*term_fcf_margin/(wacc-tg)
    pv += tv/(1+wacc)**years
    return pv

def solve_g(term_m, wacc):
    lo,hi=0.0,1.2
    for _ in range(200):
        mid=(lo+hi)/2
        if dcf(mid,term_m,wacc)<EV: lo=mid
        else: hi=mid
    return (lo+hi)/2

def solve_margin(g,wacc):
    lo,hi=0.0,3.0
    for _ in range(200):
        mid=(lo+hi)/2
        if dcf(g,mid,wacc)<EV: lo=mid
        else: hi=mid
    return (lo+hi)/2

print("="*90)
print("A) REQUIRED 10-YR REVENUE CAGR to justify EV=$251bn, given a TERMINAL FCF MARGIN")
print("   (FCF margin ramps from today's 13.1% cash-FCF margin; 3% perpetual growth after yr10)")
rows=[]
for wacc in [0.08,0.09,0.10,0.11]:
    r={'WACC':f'{wacc:.0%}'}
    for tm in [0.30,0.40,0.50,0.60]:
        g=solve_g(tm,wacc); r[f'termFCFm={tm:.0%}']=f'{g*100:.1f}%'
    rows.append(r)
print(pd.DataFrame(rows).set_index('WACC').to_string())
print()
print("B) REQUIRED TERMINAL FCF MARGIN, given a 10-yr revenue CAGR")
rows=[]
for wacc in [0.08,0.09,0.10,0.11]:
    r={'WACC':f'{wacc:.0%}'}
    for g in [0.15,0.20,0.25,0.30,0.35]:
        m=solve_margin(g,wacc); r[f'CAGR={g:.0%}']=(f'{m*100:.0f}%' if m<2.99 else 'impossible')
    rows.append(r)
print(pd.DataFrame(rows).set_index('WACC').to_string())
print()
print("C) What price do DEFENSIBLE assumptions give?  (rev CAGR, terminal FCF margin, WACC)")
scen=[('Bull  ',0.28,0.50,0.09),('Bull-  ',0.25,0.45,0.09),('Base  ',0.20,0.40,0.09),
      ('Base- ',0.18,0.35,0.10),('Bear  ',0.12,0.30,0.10),('Bear- ',0.08,0.25,0.10)]
out=[]
for n,g,m,w in scen:
    ev=dcf(g,m,w); eq=ev+NETCASH; px=eq/SH
    out.append(dict(scenario=n.strip(),rev_CAGR=f'{g:.0%}',term_FCFm=f'{m:.0%}',WACC=f'{w:.0%}',
        EV_bn=round(ev/1000,1), fair_px=round(px,0), vs_price=f'{px/P-1:+.0%}', vs_cost257=f'{px/257-1:+.0%}',
        rev_yr10_bn=round(REV0*(1+g)**10/1000,1)))
print(pd.DataFrame(out).to_string(index=False))
