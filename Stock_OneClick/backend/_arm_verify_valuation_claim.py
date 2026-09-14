"""Adversarial verification of the ARM valuation claim.
All inputs re-derived from filings by me, not taken from the claim.

PRIMARY SOURCES USED (verified by grep in this session):
  20-F FY2026 (CIK 1973239, FY ended 2026-03-31), _arm_20f_fy2026.txt
    L1088 Total revenue 4,920 / 4,007 / 3,233  (FY26/FY25/FY24)
    L1097 Operating income 900 (18%) / 831 (21%) / 111 (3%)
    L2459 Payments of withholding tax on vested shares (529)/(120)/(158)
  20-F FY2024, _arm_20f_fy2024.txt
    L1089 Total revenue 3,233 / 2,679 / 2,703  (FY24/FY23/FY22)  <-- FY22>FY23
  6-K 2026-07-29 Q1FY27, _arm_q1fy27_financials.txt
    L105 Total revenue 1,289 / 1,053 ; L115 Operating income 91 / 114
    L235 1,068 shares issued & outstanding @2026-06-30 (1,064 @2026-03-31)
    L174/175 Cash 3,058 + ST investments 830 = 3,888 ; no borrowings line exists
    L374 Payments of withholding tax on vested shares (278)/(85)
    L1383 "...shift to withhold-to-cover method in satisfaction of tax obligations"
  Shareholder letter FY26Q4 (6-K 2026-05-06), _arm_sl_fye26q431-marx26.txt
    L115 "GAAP op margin decreased to 18.3% from 20.7%... Non-GAAP decreased to 43.0% from 46.7%"
  Shareholder letter FY25Q4 (6-K 2025-05-07), _arm_sl_fye25q431-marx25.txt
    L99  "GAAP op margin INCREASED to 20.7% from 3.4%... Non-GAAP INCREASED to 46.7% from 43.6%"
  Shareholder letter FY27Q1, _arm_sl_fye27q130-junx26.txt
    L471 Non-GAAP FCF TTM 1,397
"""
import numpy as np, pandas as pd
pd.set_option('display.width', 250)

P, SH, NETCASH = 239.01, 1068.0, 3888.0
EV = P*SH - NETCASH
REV0 = 4920.0 - 1053.0 + 1289.0            # TTM to 2026-06-30, from filings
assert REV0 == 5156.0

print("="*100)
print("PART 1 -- SOURCE INTEGRITY: does every load-bearing number reconcile to a filing?")
print("="*100)
checks = [
 ("shares out 1,068m",            SH, 1068.0, "6-K Q1FY27 BS L235", True),
 ("net cash $3,888m",             NETCASH, 3058.0+830.0, "6-K Q1FY27 BS L174/175", True),
 ("TTM revenue $5,156m",          REV0, 5156.0, "20-F FY26 L1088 + 6-K L105", True),
 ("EV $251.4bn",                  round(EV/1000,1), 251.4, "derived", True),
 ("non-GAAP FCF TTM $1,397m",     1397.0, 1397.0, "SL FY27Q1 L471", True),
 ("cash withhold TTM $722m",      529-85+278, 722.0, "20-F L2459 + 6-K L374", True),
 ("FY26 rev growth +22.8%",       round((4920/4007-1)*100,1), 22.8, "20-F L1088", True),
 ("FY23-26 rev CAGR 22.5%",       round(((4920/2679)**(1/3)-1)*100,1), 22.5, "20-F FY24 L1089", True),
 ("GAAP OM 20.7%->18.3%",         round(900/4920*100,1), 18.3, "20-F L1097", True),
 ("Q1FY27 GAAP OM 7.1%",          round(91/1289*100,1), 7.1, "6-K L1250", True),
 ("non-GAAP EPS +8.6%",           round((1.77/1.63-1)*100,1), 8.6, "SL FY26Q4 / FY25Q4", True),
]
for lbl, got, exp, src, ok in checks:
    flag = "OK " if abs(got-exp) < 0.06 else "MISMATCH"
    print(f"  [{flag}] {lbl:26s} computed {got:>10} vs claimed {exp:>8}   src: {src}")

print()
print("  [FAIL] 'margins CONTRACTED both years' -- the actual filed series is:")
print("         non-GAAP OM  FY24 43.6%  ->  FY25 46.7%  ->  FY26 43.0%   (UP then DOWN)")
print("         GAAP OM      FY24  3.4%  ->  FY25 20.7%  ->  FY26 18.3%   (UP HUGE then down)")
print("         src: SL FY25Q4 L99 says 'INCREASED to 46.7% from 43.6%' and 'INCREASED to 20.7% from 3.4%'")
print("         => Margins contracted in ONE fiscal year (FY26) + Q1FY27, not 'both years'.")
print("         => Over the full 3-yr record non-GAAP OM is FLAT (43.6->43.0), GAAP is UP 15pts.")

# how much of the FY24->FY25 GAAP jump is one-time IPO SBC rolling off?
ng_op_fy24 = 0.436*3233; ng_op_fy25 = 0.467*4007; ng_op_fy26 = 0.430*4920
print(f"\n  Context: implied non-GAAP op income FY24 ${ng_op_fy24:,.0f}m vs GAAP $111m -> ${ng_op_fy24-111:,.0f}m of")
print(f"  excluded charges (IPO-vintage SBC). FY26 gap is ${ng_op_fy26-900:,.0f}m. So most of the GAAP 'expansion'")
print(f"  is a one-time IPO grant cliff rolling off, NOT operating leverage. (INFERENCE)")

print()
print("="*100)
print("PART 2 -- THE STARTING FCF MARGIN: three defensible definitions, and who they favour")
print("="*100)
sbc_ttm = 1052-241+343
wh_ttm  = 529-85+278
defs = [
 ("as-REPORTED non-GAAP FCF (what sell-side uses)", 1397.0,              "no SBC charge at all"),
 ("cash-settled SBC charged (the claim's choice)",  1397.0-wh_ttm,       f"less ${wh_ttm}m withholding"),
 ("full SBC expense charged (strictest)",           1397.0-sbc_ttm,      f"less ${sbc_ttm}m SBC expense"),
]
for lbl, fcf, note in defs:
    print(f"  {lbl:48s} ${fcf:>7,.0f}m = {fcf/REV0*100:5.1f}% of revenue   ({note})")
print()
print("  FACT: the withholding IS filed in FINANCING activities (20-F L2459), so ARM's reported")
print("        FCF legitimately excludes it under ASU 2016-09. It is economically a buyback:")
print("        company pays tax cash and issues FEWER shares (6-K L1383 confirms the method shift).")
print("  INFERENCE: charging it is internally consistent ONLY with a FROZEN share count -- which is")
print("        what the DCF does (divides by a fixed 1,068m). So the 13.1% is defensible, and it sits")
print("        BETWEEN the bull's 27.1% and the strict 4.7%. The claim never shows the 27.1% a bull uses.")
print("  FACT: cash withholding history 158 (FY24) -> 120 (FY25) -> 529 (FY26) -> 278 in Q1FY27 alone.")
print("        6-K L1383 attributes the jump to share price AND the withhold-to-cover method shift,")
print("        so $722m TTM is an ELEVATED, price-dependent number, not a stable run-rate.")

def dcf(g, term_m, wacc, start_m, years=10, tg=0.03):
    pv, rev = 0.0, REV0
    for t in range(1, years+1):
        rev *= (1+g)
        pv += rev*(start_m + (term_m-start_m)*t/years)/(1+wacc)**t
    tv = rev*(1+tg)*term_m/(wacc-tg)
    return pv + tv/(1+wacc)**years, tv/(1+wacc)**years/(pv+tv/(1+wacc)**years)

def solve_g(term_m, wacc, start_m):
    lo, hi = 0.0, 1.5
    for _ in range(200):
        mid = (lo+hi)/2
        if dcf(mid, term_m, wacc, start_m)[0] < EV: lo = mid
        else: hi = mid
    return (lo+hi)/2

print()
print("="*100)
print("PART 3 -- STEELMAN: required 10yr revenue CAGR under EACH starting-margin definition")
print("="*100)
rows = []
for lbl, sm in [("as-reported 27.1%", (1397.0)/REV0), ("claim's 13.1%", (1397.0-wh_ttm)/REV0),
                ("strict 4.7%", (1397.0-sbc_ttm)/REV0)]:
    r = {"start FCF margin": lbl}
    for tm in [0.30, 0.40, 0.50]:
        r[f"term={tm:.0%}"] = f"{solve_g(tm, 0.09, sm)*100:.1f}%"
    rows.append(r)
print(pd.DataFrame(rows).set_index("start FCF margin").to_string())
print("\n  => The starting margin barely matters. Even granting the bull's as-reported 27.1% FCF margin,")
print("     $239 still needs 24-31% revenue CAGR for a DECADE vs 22.5% delivered. The claim's choice of")
print("     13.1% is NOT what drives its conclusion. This attack fails to overturn the claim.")

print()
print("="*100)
print("PART 4 -- THE '$116 CONTINUATION' FRAMING: is it really a continuation?")
print("="*100)
sm = (1397.0-wh_ttm)/REV0
scen = [
 ("claim's 'Base' = the $116 number", 0.20, 0.40),
 ("TRUE continuation: delivered 22.5% CAGR, margin FLAT at 13.1%", 0.225, sm),
 ("delivered 22.5% CAGR, margin to 40% (growth-only continuation)", 0.225, 0.40),
 ("delivered 22.5% CAGR, margin to 27.1% (=as-reported today)", 0.225, 0.271),
]
out = []
for lbl, g, tm in scen:
    ev, tvshare = dcf(g, tm, 0.09, sm)
    out.append(dict(scenario=lbl, CAGR=f"{g:.1%}", term_m=f"{tm:.1%}",
                    fair_px=round((ev+NETCASH)/SH), vs_239=f"{(ev+NETCASH)/SH/P-1:+.0%}",
                    TV_share=f"{tvshare:.0%}"))
print(pd.DataFrame(out).to_string(index=False))
print()
print("  [FLAW] The claim says $239 'requires a step-change on BOTH growth AND margin ... not a")
print("         continuation; continuation prices at roughly $116'. But the $116 scenario ITSELF")
print("         already grants the full 13.1%->40% margin step-change. Between $116 and $239 the")
print("         ONLY moving part is the growth rate. So 'BOTH ... simultaneously' misdescribes the")
print("         claim's own arithmetic: margin expansion is common to both sides of the comparison.")
print("  [DIRECTION] This error makes the claim CONSERVATIVE, not aggressive: a genuine continuation")
print("         (22.5% CAGR AND margins flat) prices far BELOW $116, not above it.")
print("  [NOTE] The $116 case also uses 20% CAGR, below the 22.5% actually delivered -- again")
print("         understating the bull case, i.e. an error against the claim's own direction.")

print()
print("="*100)
print("PART 5 -- FRAGILITY: how much of the value is terminal, and the 3%-perpetuity choice")
print("="*100)
for g, tm in [(0.269, 0.50), (0.337, 0.30), (0.20, 0.40)]:
    ev, tvs = dcf(g, tm, 0.09, sm)
    print(f"  CAGR {g:.1%}, term margin {tm:.0%}: EV ${ev/1000:.1f}bn, terminal value = {tvs:.0%} of PV")
print("  INFERENCE: 60-70% of value sits in the terminal block, so the output is highly sensitive to")
print("  the 3% perpetual growth and 9% WACC. These are ASSUMPTIONS, not filed facts. A reverse DCF")
print("  is a statement about required expectations, NOT a measurement. The claim's '$116' should be")
print("  read as one point on a wide surface, not a computed fair value.")
print(f"  Sanity: revenue in yr10 at 26.9% CAGR = ${REV0*1.269**10/1000:.1f}bn vs ${REV0/1000:.1f}bn today")
print(f"          revenue in yr10 at 22.5% CAGR = ${REV0*1.225**10/1000:.1f}bn")

print()
print("="*100)
print("PART 6 -- SAMPLE SIZE: how many independent observations back '22.5% is the trend'?")
print("="*100)
rev = {"FY22":2703,"FY23":2679,"FY24":3233,"FY25":4007,"FY26":4920}
ks = list(rev)
for a,b in zip(ks, ks[1:]):
    print(f"  {a}->{b}: {rev[a]:>5} -> {rev[b]:>5}  = {(rev[b]/rev[a]-1)*100:+6.1f}%")
print(f"\n  FACT: only 4 YoY growth observations exist, and one of them is NEGATIVE (FY22->FY23).")
print(f"  FACT: choosing FY23 (2,679) as the base starts from a TROUGH -- the most bull-friendly base.")
print(f"        CAGR from FY22 (2,703) over 4 yrs = {((4920/2703)**(1/4)-1)*100:.1f}%, vs 22.5% from FY23.")
print(f"  => n=4 annual points (12 quarters since IPO). Any 10-YEAR extrapolation from this, in either")
print(f"     direction, is speculation. The claim's '~27-30% for a decade' requirement is arithmetic and")
print(f"     survives; the implied confidence that 22.5% IS the forward trend does NOT.")
