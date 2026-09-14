import pandas as pd, numpy as np
px=pd.read_pickle("_arm_apple_panel.pkl"); px=px.loc[px.ARM.notna()]
P=239.01; SH=1064055252/1e9   # Form 144 08/27/2026 & 20-F cover: 1,064,055,252 ordinary shares o/s at 2026-03-31

print("="*100); print("1) VALUATION -- what is already priced (all inputs from Arm FY2026 20-F, FYE 2026-03-31)")
print("="*100)
REV,ROY,LIC,NI,EPS = 4920,2613,2307,904,0.85
for tag,p in [("today 239.01",239.01),("cost basis 257.00",257.00),("ATH 2026-06-18 439.46",439.46)]:
    mc=p*SH
    print(f"  {tag:24s} mktcap ${mc:6.1f}B | P/S(FY26 {REV/1000:.2f}B)={mc*1000/REV:6.1f}x | P/E(GAAP dil EPS ${EPS})={p/EPS:6.0f}x")
print(f"\n  FY26 revenue mix: royalty ${ROY}M = {100*ROY/REV:.1f}% of revenue ; license ${LIC}M = {100*LIC/REV:.1f}%")
print(f"  FY26 GAAP net income ${NI}M -> net margin {100*NI/REV:.1f}%")

print("\n"+"="*100); print("2) THE APPLE CHANNEL -- tightest bound derivable from disclosure")
print("="*100)
mob_share={"FY2024":0.35,"FY2025":0.46,"FY2026":0.43}     # 20-F: mobile applications processors % of ROYALTY revenue
roy={"FY2024":1802,"FY2025":2168,"FY2026":2613}           # sum of quarterly letters; FY25/26 tie to 20-F exactly
tot={"FY2024":3233,"FY2025":4007,"FY2026":4920}
for k in mob_share:
    m=mob_share[k]*roy[k]
    print(f"  {k}: royalty ${roy[k]}M x {mob_share[k]:.0%} mobile-AP = ${m:,.0f}M mobile-AP royalty = {100*m/tot[k]:.1f}% of TOTAL revenue")
m26=mob_share["FY2026"]*roy["FY2026"]
print(f"\n  ALL smartphones on earth (Apple + every Android OEM) = ${m26:,.0f}M = {100*m26/tot['FY2026']:.1f}% of Arm FY26 revenue.")
print(f"  Apple is a SUBSET of that. Apple's share of the number is NOT DISCLOSED anywhere in the 20-F.")
print(f"  Upper bound if Apple were somehow 100% of mobile-AP royalty: {100*m26/tot['FY2026']:.1f}% of revenue.")
for s in [0.20,0.25,0.30]:
    print(f"    if Apple = {s:.0%} of mobile-AP royalty -> ${m26*s:,.0f}M = {100*m26*s/tot['FY2026']:.1f}% of Arm revenue")
print("\n  SENSITIVITY: a +10% y/y jump in Apple's iPhone royalty contribution moves Arm total revenue by:")
for s in [0.20,0.25,0.30]:
    print(f"    {s:.0%} Apple share -> +{100*0.10*m26*s/tot['FY2026']:.2f}% on total revenue")

print("\n"+"="*100); print("3) WHERE FY2026 GROWTH ACTUALLY CAME FROM (20-F verbatim table)")
print("="*100)
print("                     FY2026   FY2025   change    % change   share of total $ increase")
for nm,a,b in [("External customers",3421,3184),("Related parties (Arm China + SoftBank)",1499,823),("TOTAL",4920,4007)]:
    print(f"  {nm:38s} {a:6d}  {b:6d}   {a-b:+6d}    {100*(a/b-1):+6.1f}%      {100*(a-b)/913:5.1f}%")
print(f"  External LICENSE revenue alone: 1,298 vs 1,421 = -9%  (third-party licensing SHRANK)")
print(f"  Related parties went from {100*823/4007:.1f}% to {100*1499/4920:.1f}% of total revenue.")

print("\n"+"="*100); print("4) WHAT THE -37% ACTUALLY IS: anatomy of the move")
print("="*100)
ath=pd.Timestamp("2026-06-18")
for lbl,a,b in [("2026 low->ATH melt-up","2026-06-10","2026-06-18"),
                ("ATH -> today","2026-06-18","2026-09-14"),
                ("3 months ago -> today","2026-06-11","2026-09-14"),
                ("start of 2026 -> today","2026-01-02","2026-09-14"),
                ("pre-melt-up 2026-05-29 -> today","2026-05-29","2026-09-14"),
                ("IPO close -> today","2023-09-14","2026-09-14")]:
    try:
        a_=px.index[px.index.searchsorted(pd.Timestamp(a))]; b_=px.index[px.index.searchsorted(pd.Timestamp(b))]
        pa,pb=px.ARM.loc[a_],px.ARM.loc[b_]; sa,sb=px.SMH.loc[a_],px.SMH.loc[b_]
        print(f"  {lbl:34s} {a_.date()} {pa:7.2f} -> {b_.date()} {pb:7.2f}  ARM {100*(pb/pa-1):+7.1f}%  SMH {100*(sb/sa-1):+7.1f}%  rel {100*(pb/pa)/(sb/sa)-100:+7.1f}%")
    except Exception as e: print(lbl,"err",e)
r=px.pct_change()
w=r.loc["2026-06-19":]
print(f"\n  Since the 2026-06-18 ATH: {len(w)} trading days. ARM down days {int((w.ARM<0).sum())}, up {int((w.ARM>0).sum())}.")
print(f"  Largest single-day ARM drop in that span: {100*w.ARM.min():.2f}% on {w.ARM.idxmin().date()}")
print(f"  Sum of the 5 worst days: {100*w.ARM.nsmallest(5).sum():.1f}%  -> the decline is a GRIND, not one event.")
print(f"  Arm SEC filings between 2026-06-18 and today: 6-K 2026-07-29 (Q1 FY27 earnings, BEAT+RAISE),")
print(f"    6-K 2026-08-10 (AGM notice), 6-K 2026-09-10 (AGM voting results). NO guidance cut, NO 8-K-style adverse event.")
print(f"  Earnings-reaction day 2026-07-30: ARM {100*r.ARM.loc['2026-07-30']:+.2f}% vs SMH {100*r.SMH.loc['2026-07-30']:+.2f}% -> rel {100*(r.ARM.loc['2026-07-30']-r.SMH.loc['2026-07-30']):+.2f}%")

print("\n"+"="*100); print("5) SMARTPHONE MATURITY, from the filings of the phone makers themselves")
print("="*100)
ip={"FY2022":205489,"FY2023":200583,"FY2024":201183,"FY2025":209586}
s=pd.Series(ip); print("  Apple iPhone net sales, $M (Apple 10-K):")
print(pd.DataFrame({"iPhone_$M":s,"yoy%":(s.pct_change()*100).round(1)}).to_string(header=True))
print(f"  FY2022->FY2025 CAGR: {100*((209586/205489)**(1/3)-1):+.2f}% per year")
print(f"  FY2026 first 9 months (10-Q 2026-06-27): $196,515M vs $160,561M = {100*(196515/160561-1):+.1f}% y/y  <-- ACCELERATION")
q={"QCOM FY2023":22570,"QCOM FY2024":24863,"QCOM FY2025":27793}
print("\n  Qualcomm QCT Handsets revenue, $M (QCOM 10-K) -- merchant Android AP dollars:")
qs=pd.Series(q); print(pd.DataFrame({"$M":qs,"yoy%":(qs.pct_change()*100).round(1)}).to_string())
print("  NOTE: these are DOLLARS (content/ASP), not units. Apple stopped disclosing iPhone units in FY2019;")
print("  IDC/Canalys/Counterpoint unit data was NOT reachable from this environment, so no unit figure is cited.")
