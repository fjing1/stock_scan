import numpy as np, pandas as pd
pd.set_option('display.width',240)
REV26=4920.0
china=790.6; sbg=704.4; ampere=3.6; rp_total=1499.0
print("=== IDENTIFYING THE THREE >=10% CUSTOMERS (20-F FY2026) ===")
print("20-F Note 4: three customers = 42% of revenue, at 16%, 14%, 12%.")
print(f"20-F Note 20 named related-party amounts: Arm China ${china}m, SoftBank affiliate ${sbg}m, Ampere ${ampere}m")
print(f"   sum = ${china+sbg+ampere:.1f}m  vs Note 4 'TOTAL RELATED' ${rp_total}m  (diff ${rp_total-(china+sbg+ampere):.1f}m) -> accounts for essentially all of it")
print(f"   Arm China          {china/REV26*100:.1f}% of revenue  -> matches the '16%' customer")
print(f"   SoftBank affiliate {sbg/REV26*100:.1f}% of revenue  -> matches the '14%' customer")
print(f"   => only ONE arm's-length customer exceeded 10%: the '12%' one = ~${0.12*REV26:.0f}m")
print(f"   => SoftBank-orbit customers = {(china+sbg)/REV26*100:.1f}% of FY2026 revenue")
print()
print("=== TIGHTEST BOUND ON APPLE ===")
print("Apple is NOT named as a customer anywhere in the 20-F (1 mention, in 'History', as a 1990 JV founder).")
print(f"Best case: Apple = the 12% customer -> <= ${0.12*REV26:.0f}m of FY26 revenue ({0.12*100:.0f}%), ALL revenue not just royalty.")
print(f"Otherwise: Apple < 10% of revenue -> < ${0.10*REV26:.0f}m.")
print("Apple's share of ROYALTY revenue specifically: NOT DISCLOSED. Per-customer royalty rates are confidential.")
print()
print("=== BOUND ON NON-DATA-CENTRE ('ubiquity') ROYALTY GROWTH, Q1 FY27 ===")
print("Filed facts (6-K 2026-07-29): total royalty $715m vs $585m (+22%); 'data center royalties more than doubled y/y'.")
R1,R0=715.0,585.0
rows=[]
for d in [0.05,0.10,0.15,0.20,0.25,0.30]:
    dc0=d*R0; dc1=2*dc0                      # 'more than doubled' -> at least 2x
    non1=R1-dc1; non0=R0-dc0
    rows.append(dict(DC_share_of_prior_royalty=f'{d:.0%}', DC_rev_prior=round(dc0),
        DC_rev_now_min=round(dc1), nonDC_prior=round(non0), nonDC_now_max=round(non1),
        nonDC_growth_MAX=f'{(non1/non0-1)*100:+.1f}%'))
print(pd.DataFrame(rows).to_string(index=False))
print("\nDC share is not disclosed, so this is a bound, not a point estimate.")
print("Reading: for ANY plausible data-centre share >=15%, ALL non-data-centre royalty")
print("(smartphones incl. Apple, IoT, auto, consumer) grew <=8.5% y/y -- at most a third of headline royalty growth.")
print()
print("=== 'MERELY CONTINUE' vs 'STEP-CHANGE' ===")
print(f"Actual revenue CAGR FY2023->FY2026: {(4920/2679)**(1/3)-1:.1%}   (2,679 -> 4,920)")
print(f"Actual EXTERNAL rev CAGR FY24->FY26: {(3421/2509)**(1/2)-1:.1%}   (2,509 -> 3,421)")
print(f"Non-GAAP op margin: FY25 46.7% -> FY26 43.0%  (CONTRACTED)")
print(f"GAAP op margin:     FY25 20.7% -> FY26 18.3% -> TTM 17.0% -> Q1FY27 7.1%  (CONTRACTED)")
print(f"Non-GAAP EPS: FY25 $1.63 -> FY26 $1.77 = {(1.77/1.63-1)*100:+.1f}%  on revenue +22.8%  => NEGATIVE operating leverage")
print()
print("Price requires (from reverse DCF): ~27-30% revenue CAGR for 10 yrs AND FCF margin 13% -> 50%.")
print("=> BOTH axes require a step-change. Continuation of the actual trend (~20-22% rev, flat/down margin)")
print("   prices out around $116 (Base scenario), i.e. -52% from here.")
