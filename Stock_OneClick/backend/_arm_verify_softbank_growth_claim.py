"""Adversarial verification of the claim:
  "74.4% of ARM's entire FY2026 revenue growth came from SoftBank's own orbit ...
   excluding the SoftBank affiliate and Arm China, FY2026 revenue grew +7.3%."

ALL INPUTS ARE FROM PRIMARY SOURCES (SEC EDGAR, ARM HOLDINGS PLC /UK, CIK 1973239):

[A] 20-F FY2026, accession 0001973239-26-000097, arm-20260331.htm, filed 2026-05-26
    (audited; report of independent registered public accounting firm dated 2026-05-26)
    - Consolidated Income Statements (face): external / related-party / total revenue
    - Note 4 - Revenue: disaggregation by external vs related party x license vs royalty;
      footnote (1) over-time vs point-in-time; geography table; customer concentration
    - Note 20 - Related Party Transactions: Arm China IPLA; SoftBank affiliate Consulting
      Agreement; Ampere
    - Item 7B Related Party Transactions: same Consulting Agreement figures
    - Item 3D Risk Factors: top-5 customer concentration, Arm China control language
[B] 20-F FY2025, accession 0001973239-25-000016, arm-20250331.htm, filed 2025-05-28
[C] 6-K interim financials FY26 Q2 (period 2025-09-30), accession 0001973239-25-000043
[D] 6-K interim financials FY26 Q3 (period 2025-12-31), accession 0001973239-26-000006
[E] 6-K interim financials FY27 Q1 (period 2026-06-30), accession 0001973239-26-000114
"""

# ----------------------------------------------------------------------------------
# [A] Consolidated Income Statements, face, in $m (20-F FY2026)
# ----------------------------------------------------------------------------------
ext = {2026: 3421, 2025: 3184, 2024: 2509}
rel = {2026: 1499, 2025: 823, 2024: 724}
tot = {2026: 4920, 2025: 4007, 2024: 3233}

# [A] Note 4 - Revenue, disaggregation ($m)
lic_ext = {2026: 1298, 2025: 1421, 2024: 1051}
lic_rel = {2026: 1009, 2025: 418, 2024: 380}
roy_ext = {2026: 2123, 2025: 1763, 2024: 1458}
roy_rel = {2026: 490, 2025: 405, 2024: 344}

# [A] Note 4 footnote (1): timing of license & other revenue ($m)
overtime = {2026: 1080, 2025: 467, 2024: 121}
pointintime = {2026: 1227, 2025: 1372, 2024: 1310}

# [A] Note 20 / Item 7B, named related parties ($m, to $0.1m)
sb_consult = {2026: 704.4, 2025: 145.5, 2024: 0.0}   # FY24: not disclosed => nil/immaterial
armchina_ipla = {2026: 790.6, 2025: 670.4, 2024: 670.8}
ampere = {2026: 3.6, 2025: 3.5, 2024: 49.3}
other_sb = {2026: 0.0, 2025: 0.0, 2024: 4.4}

# [A] Note 4 geography ($m)
geo = {
    "United States": {2026: 1761, 2025: 1716, 2024: 1413},
    "PRC":           {2026: 874,  2025: 749,  2024: 697},
    "Japan":         {2026: 825,  2025: 296,  2024: 121},
    "Taiwan":        {2026: 695,  2025: 629,  2024: 522},
    "Rep of Korea":  {2026: 392,  2025: 324,  2024: 308},
    "Other":         {2026: 373,  2025: 293,  2024: 172},
}


def pct(a, b):
    return (a / b - 1.0) * 100.0


def line(s):
    print(s)


print("=" * 92)
print("STEP 1 - REPRODUCE THE CLAIM'S ARITHMETIC  [source A]")
print("=" * 92)
g_tot = tot[2026] - tot[2025]
line(f"Total revenue         FY26 ${tot[2026]:,}m  FY25 ${tot[2025]:,}m   "
     f"delta +${g_tot}m  = {pct(tot[2026], tot[2025]):+.2f}%   (filing text: '+$913 million, or 23%')")

d_sb = sb_consult[2026] - sb_consult[2025]
d_ac = armchina_ipla[2026] - armchina_ipla[2025]
line(f"SoftBank affiliate    FY26 ${sb_consult[2026]}m  FY25 ${sb_consult[2025]}m  "
     f"delta +${d_sb:.1f}m  = {d_sb / g_tot * 100:.1f}% of total growth")
line(f"Arm China IPLA        FY26 ${armchina_ipla[2026]}m  FY25 ${armchina_ipla[2025]}m  "
     f"delta +${d_ac:.1f}m  = {d_ac / g_tot * 100:.1f}% of total growth")
line(f"  SUM                                                        "
     f"+${d_sb + d_ac:.1f}m  = {(d_sb + d_ac) / g_tot * 100:.1f}% of total growth   "
     f"<-- claim says 74.4%")

exA26 = tot[2026] - sb_consult[2026] - armchina_ipla[2026]
exA25 = tot[2025] - sb_consult[2025] - armchina_ipla[2025]
line(f"\nEx-SoftBank-affiliate AND ex-Arm-China: ${exA26:,.1f}m vs ${exA25:,.1f}m "
     f"= {pct(exA26, exA25):+.2f}%   <-- claim says +7.3%")

print("\nVERDICT ON ARITHMETIC: every figure reproduces to the stated precision.")

print()
print("=" * 92)
print("STEP 2 - ATTACK: DOES EXCLUDING ARM CHINA SURVIVE SCRUTINY?  [source A]")
print("=" * 92)
exB26 = tot[2026] - sb_consult[2026]
exB25 = tot[2025] - sb_consult[2025]
line(f"Excluding ONLY the SoftBank affiliate consulting fee:")
line(f"  ${exB26:,.1f}m vs ${exB25:,.1f}m = {pct(exB26, exB25):+.2f}%   "
     f"(NOT +7.3%; the claim's own 61.2% share implies this)")
line(f"Arm China IPLA growth FY25->FY26: +{pct(armchina_ipla[2026], armchina_ipla[2025]):.1f}%  "
     f"(and it was FLAT FY24->FY25: ${armchina_ipla[2024]}m -> ${armchina_ipla[2025]}m, "
     f"{pct(armchina_ipla[2025], armchina_ipla[2024]):+.2f}%)")
line(f"Cross-check vs geography: PRC revenue ${geo['PRC'][2025]}m -> ${geo['PRC'][2026]}m "
     f"= +${geo['PRC'][2026] - geo['PRC'][2025]}m ({pct(geo['PRC'][2026], geo['PRC'][2025]):+.1f}%), "
     f"consistent with Arm China +${d_ac:.1f}m -> the Arm China delta is PRC end-demand, not a SoftBank cheque.")

print()
print("=" * 92)
print("STEP 3 - THE DECOMPOSITION THE CLAIM DOES NOT DO: ROYALTY vs LICENSE  [source A, Note 4]")
print("=" * 92)
line(f"{'':34s} {'FY26':>9s} {'FY25':>9s} {'delta':>9s} {'%chg':>8s}")
for nm, d in [("External ROYALTY", roy_ext), ("External LICENSE & other", lic_ext),
              ("Related-party ROYALTY (Arm China)", roy_rel), ("Related-party LICENSE & other", lic_rel)]:
    line(f"{nm:34s} {d[2026]:>9,} {d[2025]:>9,} {d[2026] - d[2025]:>+9,} {pct(d[2026], d[2025]):>+7.1f}%")
tot_roy26, tot_roy25 = roy_ext[2026] + roy_rel[2026], roy_ext[2025] + roy_rel[2025]
tot_lic26, tot_lic25 = lic_ext[2026] + lic_rel[2026], lic_ext[2025] + lic_rel[2025]
line(f"{'TOTAL ROYALTY':34s} {tot_roy26:>9,} {tot_roy25:>9,} {tot_roy26 - tot_roy25:>+9,} "
     f"{pct(tot_roy26, tot_roy25):>+7.1f}%   <-- 0% SoftBank consulting in this line")
line(f"{'TOTAL LICENSE & OTHER':34s} {tot_lic26:>9,} {tot_lic25:>9,} {tot_lic26 - tot_lic25:>+9,} "
     f"{pct(tot_lic26, tot_lic25):>+7.1f}%")
line(f"\nRoyalty share of total revenue: FY26 {tot_roy26 / tot[2026] * 100:.1f}%, "
     f"FY25 {tot_roy25 / tot[2025] * 100:.1f}%")
line(f"SoftBank consulting delta (+${d_sb:.1f}m) vs TOTAL license&other delta "
     f"(+${tot_lic26 - tot_lic25}m): the consulting fee is "
     f"{d_sb / (tot_lic26 - tot_lic25) * 100:.0f}% of the license line's growth.")
lic_ex_sb26 = tot_lic26 - sb_consult[2026]
lic_ex_sb25 = tot_lic25 - sb_consult[2025]
line(f"=> License & other EX-SoftBank-consulting: ${lic_ex_sb26:,.1f}m vs ${lic_ex_sb25:,.1f}m "
     f"= {pct(lic_ex_sb26, lic_ex_sb25):+.1f}%  (i.e. it SHRANK)")
line(f"\nSo the damage is confined to the LICENSE line. Royalty revenue -- the line that actually")
line(f"measures Arm-based chips SHIPPING in end products -- grew "
     f"{pct(tot_roy26, tot_roy25):+.1f}% with zero SoftBank content.")
line(f"Note 4 fn(1): point-in-time license revenue ${pointintime[2025]:,}m -> ${pointintime[2026]:,}m "
     f"({pct(pointintime[2026], pointintime[2025]):+.1f}%); over-time ${overtime[2025]}m -> "
     f"${overtime[2026]:,}m ({pct(overtime[2026], overtime[2025]):+.0f}%).")

print()
print("=" * 92)
print("STEP 4 - RELATED-PARTY TOTAL RECONCILIATION  [source A, Note 4 + Note 20]")
print("=" * 92)
named26 = sb_consult[2026] + armchina_ipla[2026] + ampere[2026]
line(f"Named related parties FY26: SoftBank affil ${sb_consult[2026]} + Arm China ${armchina_ipla[2026]}"
     f" + Ampere ${ampere[2026]} = ${named26:.1f}m  vs Note 4 related-party total ${rel[2026]:,}m "
     f"(diff ${rel[2026] - named26:+.1f}m)  <-- claim says $1,498.6m vs $1,499.0m: CONFIRMED")
named25 = sb_consult[2025] + armchina_ipla[2025] + ampere[2025]
line(f"Named related parties FY25: ${named25:.1f}m vs Note 4 total ${rel[2025]}m "
     f"(diff ${rel[2025] - named25:+.1f}m)")

print()
print("=" * 92)
print("STEP 5 - CONCENTRATION-NOTE ARITHMETIC  [source A, Note 4 + Item 3D]")
print("=" * 92)
line(f"Arm China / total   = {armchina_ipla[2026] / tot[2026] * 100:.2f}%  -> disclosed 'largest customer 16%'")
line(f"SB affiliate / total= {sb_consult[2026] / tot[2026] * 100:.2f}%  -> disclosed 'second largest 14%'")
line("Item 3D names Arm China explicitly as 'our largest customer individually' and says top-5")
line("'(including Arm China and SoftBank Group)' = 57%/56%/54% of revenue FY26/25/24. So the")
line("IDENTITY of #1 is DISCLOSED; the identity of the 14% customer is INFERRED (well-supported).")
line("Third-largest customer = 12% (~$590m) is NOT named in Note 4; Item 3D separately discloses")
line("Qualcomm at 9% of FY26 revenue, so the 12% customer is a DIFFERENT, unnamed customer.")

print()
print("=" * 92)
print("STEP 6 - IS THIS NEW INFORMATION? QUARTERLY DISCLOSURE PATH  [sources A,B,C,D,E]")
print("=" * 92)
# SoftBank affiliate consulting revenue, quarterly, from the 6-K interim notes
q = [
    ("FY25 Q1 (Jun-24)", 0.0,  "[B] FY25 20-F: nil until Q2"),
    ("FY25 Q2 (Sep-24)", 43.2, "[C] 6-K Q2FY26 comparative"),
    ("FY25 Q3 (Dec-24)", 51.1, "[D] 6-K Q3FY26 comparative"),
    ("FY25 Q4 (Mar-25)", 145.5 - 94.3, "[B] FY25 20-F FY total 145.5 less 9M 94.3"),
    ("FY26 Q1 (Jun-25)", 126.1, "[E] 6-K Q1FY27 comparative"),
    ("FY26 Q2 (Sep-25)", 177.9, "[C] 6-K Q2FY26"),
    ("FY26 Q3 (Dec-25)", 200.2, "[D] 6-K Q3FY26"),
    ("FY26 Q4 (Mar-26)", 704.4 - 504.2, "[A] FY26 total 704.4 less [D] 9M 504.2"),
    ("FY27 Q1 (Jun-26)", 192.9, "[E] 6-K Q1FY27"),
]
for nm, v, src in q:
    line(f"  {nm}  ${v:>6.1f}m    {src}")
line("\nEvery FY26 quarter's shareholder letter carried 'Revenue from related parties' on the FACE")
line("of the income statement; every 6-K interim note named 'an affiliate of SoftBank Group' and")
line("gave the dollar figure. => The information was public in real time, not a 20-F revelation.")
line("(Checked: the exhibit-99 shareholder LETTERS contain 0 occurrences of 'Consulting Agreement'")
line(" and 0 of 'affiliate of SoftBank' -- the item-level detail is in the 6-K notes only.)")

print()
print("=" * 92)
print("STEP 7 - FORWARD TEST: DOES THE 'CONTAMINATION' REVERSE?  [source E, 6-K FY27 Q1]")
print("=" * 92)
# 6-K FY27Q1 (three months ended 2026-06-30), income statement face
q1_tot = {2027: 1289, 2026: 1053}
q1_rel = {2027: 388, 2026: 328}   # 2027 derived: 30% of 1,289 per MD&A % table; 2026 stated
q1_ext = {y: q1_tot[y] - q1_rel[y] for y in q1_tot}
line(f"FY27 Q1 total revenue      ${q1_tot[2027]:,}m vs ${q1_tot[2026]:,}m = {pct(q1_tot[2027], q1_tot[2026]):+.1f}%")
line(f"FY27 Q1 related-party      ${q1_rel[2027]:,}m vs ${q1_rel[2026]:,}m = {pct(q1_rel[2027], q1_rel[2026]):+.1f}%  "
     f"(filing text: 'increased $60 million, or 18%')")
line(f"FY27 Q1 EXTERNAL revenue   ${q1_ext[2027]:,}m vs ${q1_ext[2026]:,}m = {pct(q1_ext[2027], q1_ext[2026]):+.1f}%")
line(f"FY27 Q1 SoftBank consulting ${192.9}m vs ${126.1}m = {pct(192.9, 126.1):+.1f}%  -> annualising ~${192.9 * 4:.0f}m")
line("\n=> In the MOST RECENT reported quarter the sign FLIPS: external revenue grows FASTER than")
line("   related-party revenue. The FY26 'external only +7%' is a fiscal-year artifact of the")
line("   lumpy license line, not a persistent condition. And the SoftBank fee is RECURRING and")
line("   GROWING, not a one-off that mechanically reverses out of the growth rate.")

print()
print("=" * 92)
print("STEP 8 - CASH QUALITY OF THE SOFTBANK REVENUE  [sources A, B]")
print("=" * 92)
ca26, ca25 = 645.8, 145.5
cum = sb_consult[2026] + sb_consult[2025]
line(f"Current contract assets from the SoftBank affiliate: ${ca25}m @2025-03-31 -> ${ca26}m @2026-03-31")
line(f"Claim's framing: ${ca26}m / FY26 revenue ${sb_consult[2026]}m = {ca26 / sb_consult[2026] * 100:.1f}% "
     f"'unbilled'  -> arithmetic CORRECT")
line(f"But the balance is CUMULATIVE, not FY26-specific: cumulative revenue recognised "
     f"${cum:.1f}m, contract asset ${ca26}m => ${cum - ca26:.1f}m ({(cum - ca26) / cum * 100:.0f}%) has "
     f"converted out of contract assets.")
line(f"[B] FY25 20-F: 'As of March 31, 2025, the Company had contract assets of $145.5 million and")
line(f"    did not have any accounts receivable' -> 100% of the FY25 fee was ALSO unbilled, yet [A]")
line(f"    reports no impairment of it. No expected-credit-loss allowance against the SoftBank")
line(f"    affiliate is disclosed in [A] (contrast: Arm China ECL $12.3m FY26 / $16.0m FY25).")
line(f"[A] Item 7B: a statement of work was restructured into 'a fixed payment amount of $300")
line(f"    million, which will be paid during the fiscal year ending March 31, 2027.'")

print()
print("=" * 92)
print("STEP 9 - THE SAME DECOMPOSITION ON THE MOST RECENT QUARTER  [source E]")
print("   6-K accession 0001973239-26-000114, three months ended 2026-06-30 (unaudited)")
print("=" * 92)
# Income statement face, 6-K FY27Q1
t27, t26 = 1289.0, 1053.0
sb27, sb26 = 192.9, 126.1          # Note: Consulting Agreement
ac27, ac26 = 189.4, 201.0          # Note: Arm China IPLA
am27, am26 = 5.4, 0.6              # Note: Ampere
g = t27 - t26
line(f"Total revenue        ${t27:,.0f}m vs ${t26:,.0f}m  = +${g:.0f}m  {pct(t27, t26):+.1f}%")
line(f"SoftBank consulting  ${sb27}m vs ${sb26}m = {sb27 - sb26:+.1f}m -> {(sb27 - sb26) / g * 100:+.1f}% of growth")
line(f"Arm China IPLA       ${ac27}m vs ${ac26}m = {ac27 - ac26:+.1f}m -> {(ac27 - ac26) / g * 100:+.1f}% of growth")
line(f"Ampere               ${am27}m vs ${am26}m = {am27 - am26:+.1f}m -> {(am27 - am26) / g * 100:+.1f}% of growth")
line(f"  SUM 'SoftBank orbit' share of growth = "
     f"{((sb27 - sb26) + (ac27 - ac26) + (am27 - am26)) / g * 100:+.1f}%   "
     f"(vs 74.4% claimed for FY26)")
e27 = t27 - sb27 - ac27 - am27
e26 = t26 - sb26 - ac26 - am26
line(f"\nEx-SoftBank-affiliate, ex-Arm-China, ex-Ampere: ${e27:.1f}m vs ${e26:.1f}m = {pct(e27, e26):+.1f}%")
line(f"  ...compared with the +{pct(t27, t26):.1f}% headline. THE SIGN OF THE ADJUSTMENT FLIPS:")
line(f"  in FY26 the 'clean' growth was {7.33:.1f}% vs {22.79:.1f}% headline (-15.5pp);")
line(f"  in Q1 FY27 the 'clean' growth is {pct(e27, e26):.1f}% vs {pct(t27, t26):.1f}% headline "
     f"({pct(e27, e26) - pct(t27, t26):+.1f}pp).")
line(f"\nEx-SoftBank only: ${t27 - sb27:.1f}m vs ${t26 - sb26:.1f}m = {pct(t27 - sb27, t26 - sb26):+.1f}%")
line(f"External license & other RECOVERED: $310m vs $255m = {pct(310, 255):+.0f}% "
     f"(vs -8.7% for full FY26)")
line(f"External royalty: $591m vs $470m = {pct(591, 470):+.0f}%")
line(f"Arm China contribution is NEGATIVE this quarter ({pct(ac27, ac26):+.1f}%).")

print()
print("=" * 92)
print("STEP 10 - CONTRACT ASSET IS CONVERTING TO CASH  [source E]")
print("=" * 92)
ca_mar26 = 645.8
ca_jun26 = 576.1 + 1.1
conv = ca_mar26 + sb27 - ca_jun26
line(f"SoftBank-affiliate contract assets: ${ca_mar26}m @2026-03-31 -> "
     f"${ca_jun26:.1f}m @2026-06-30 (current $576.1m + non-current $1.1m)")
line(f"Q1 FY27 revenue recognised ${sb27}m; balance FELL ${ca_mar26 - ca_jun26:.1f}m")
line(f"=> ${conv:.1f}m billed/settled out of contract assets in ONE quarter "
     f"({conv / ca_mar26 * 100:.0f}% of the 2026-03-31 balance).")
line("=> The 'stuck unbilled' framing does not survive the next quarter's data.")

print()
print("=" * 92)
print("STEP 11 - WHAT THE MULTIPLE ACTUALLY CAPITALISES  [source: 6-K 0001973239-26-000113 ex-99.1/99.2]")
print("=" * 92)
line("Q2 FY27 company guidance: revenue $1.38bn +/- $50m. Q2 FY26 actual: $1,135m.")
line(f"  => guided {pct(1380, 1135):+.1f}% y/y at the midpoint -- a FORWARD number, and the SoftBank")
line("     fee is a known, disclosed, recurring component of it.")
line("The claim's phrase 'the headline +22.8% growth rate that the multiple capitalises' is an")
line("ASSERTION ABOUT MARKET MECHANICS, not a filing fact: no primary source establishes that the")
line("market is capitalising the FY2026 historical growth rate.")
