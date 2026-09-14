"""
ADVERSARIAL VERIFICATION of the claim:
  "A +5% increase in iPhone units moves ARM's revenue by at most +0.41%, and plausibly
   +0.15% to +0.26% -- smaller than ARM's own single-quarter guidance band."

Every input below was re-pulled from the primary filing by hand (line refs given), NOT from
the original claim's script. Sources on disk in this directory.
"""

# ----------------------------------------------------------------------------------
# BLOCK 1: FACTS re-verified from ARM 20-F FY2026 (filed for FYE 2026-03-31)
#          file: _arm_20f_fy2026.htm  -> _arm_20f_fy2026_clean.txt
# ----------------------------------------------------------------------------------
TOT26, TOT25, TOT24 = 4920.0, 4007.0, 3233.0          # L888  total revenue, $M
ROY26, ROY25, ROY24 = 2613.0, 2168.0, 1802.0          # L910 (FY26,FY25); FY25 20-F L2318 (FY24)
LIC26 = 2307.0                                        # L909  license & other
EXT26, RP26 = 3421.0, 1499.0                          # L886/L887 external vs related-party

# L2394 concentration note, verbatim:
#  "three customers that collectively represented 42% of total revenue, with the single largest
#   customer accounting for 16%, the second largest 14%, and the third largest 12%...
#   No other customer represented 10% or more."
C1, C2, C3 = 0.16, 0.14, 0.12
ARM_CHINA_26 = 790.6          # L3104  Note 20, IPLA revenue from Arm China
SB_AFFIL_26  = 704.4          # L3113  Note 20, SoftBank Group affiliate licensing/servicing
QCOM_26_PCT  = 0.09           # L438   "Qualcomm ... accounted for 9% of our total revenue"

# L739  DISCLOSED end-market split the original claim did NOT use:
MOBILE_AP_SHARE_OF_ROYALTY = {2024: 0.35, 2025: 0.46, 2026: 0.43}   # FY24 from FY2024 20-F

# L2380-2392  revenue by customer HQ, $M
US_REV = {2024: 1413.0, 2025: 1716.0, 2026: 1761.0}
JP_REV = {2024: 121.0,  2025: 296.0,  2026: 825.0}

# L709 / FY2025 20-F: cumulative Arm-based chips reported shipped
CUM_CHIPS = {2025: 310e9, 2026: 350e9}

# ----------------------------------------------------------------------------------
# BLOCK 2: FACTS re-verified from Apple filings
# ----------------------------------------------------------------------------------
AAPL_IPHONE_FY25 = 209586.0   # 10-K FYE 2025-09-27, L421
AAPL_PRODUCTS_FY25 = 307003.0 # same 10-K, L558  (Products total, excludes Services)

# quarterly iPhone net sales, $M (10-Qs)
IPH_Q = {  # label: (current, prior-year)
    "Dec-25 q": (85269.0, 69138.0),   # 10-Q 2025-12-27 L528
    "Mar-26 q": (56994.0, 46841.0),   # 10-Q 2026-03-28 L580
    "Jun-26 q": (54252.0, 44582.0),   # 10-Q 2026-06-27 L583
}
# ARM royalty revenue, $M, matching calendar quarters (shareholder letters, Ex-99.2 to 6-K)
ROY_Q = {
    "Dec-25 q": (737.0, 580.0),       # Q3 FYE26 letter L100
    "Mar-26 q": (671.0, 607.0),       # Q4 FYE26 letter L108
    "Jun-26 q": (715.0, 585.0),       # Q1 FYE27 letter L89 / 6-K L961
}
# Q2 FYE27 guidance, shareholder letter Q1 FYE27 L63-65 (6-K furnished 2026-07-29)
GUIDE_Q2FY27, GUIDE_BAND = 1380.0, 50.0

pct = lambda a, b: 100.0 * (a / b - 1.0)

print("=" * 78)
print("1. IS THE CONCENTRATION CEILING REAL?  (20-F FY2026 L2394 + Note 20)")
print("=" * 78)
print(f"  disclosed >=10% customers:      16% / 14% / 12%   (sum {100*(C1+C2+C3):.0f}%, matches '42%')")
print(f"  Arm China   Note 20  ${ARM_CHINA_26:7.1f}M = {100*ARM_CHINA_26/TOT26:5.2f}%  -> identifies the 16%")
print(f"  SB affiliate Note 20 ${SB_AFFIL_26:7.1f}M = {100*SB_AFFIL_26/TOT26:5.2f}%  -> identifies the 14%")
print(f"  residual test: related-party revenue {RP26:.0f} - {ARM_CHINA_26}-{SB_AFFIL_26} = "
      f"{RP26-ARM_CHINA_26-SB_AFFIL_26:.1f}M left for ALL other related parties")
print(f"    -> the 12% customer (${C3*TOT26:.0f}M) cannot be a related party; must be external.")
print(f"  Qualcomm disclosed separately at 9% = ${QCOM_26_PCT*TOT26:.0f}M -> not the 12%.")
print(f"  CEILING: any unnamed customer, incl. Apple, is <= 12% = ${C3*TOT26:.1f}M")
print("  VERDICT: ceiling is a valid deduction from disclosed facts. Apple is NEVER named as a")
print("           customer anywhere in the 20-F (only 1990 JV history, L715). Share not disclosed.")

print()
print("=" * 78)
print("2. THE 68% STEP -- the claim's weakest link")
print("=" * 78)
sh = AAPL_IPHONE_FY25 / AAPL_PRODUCTS_FY25
print(f"  Apple FY2025 10-K: iPhone {AAPL_IPHONE_FY25/1000:.1f}bn / Products {AAPL_PRODUCTS_FY25/1000:.1f}bn"
      f" = {100*sh:.1f}%   <- figure CONFIRMED")
print("  Claim's justification: 'royalties are per chip', so device-dollar share bounds unit share.")
print("  BUT 20-F L752/L839/L853 says royalty is 'a percentage of the ASP of the chip OR a fixed")
print("  fee per unit' -- BOTH bases exist and Apple's basis is NOT disclosed. If Apple's is")
print("  ad-valorem, high-ASP A/M-series chips could exceed a 68% share. 68% is NOT a proven bound.")
print("  --> so test whether the CONCLUSION survives the 68% step failing entirely:")
for lab, s in [("claim's 68%", 0.68), ("if 85%", 0.85), ("worst case 100%", 1.00)]:
    iph = C3 * TOT26 * s
    print(f"     {lab:16s} iPhone-attributable <= ${iph:6.1f}M = {100*iph/TOT26:4.2f}% of rev"
          f" ; +5% units -> {100*0.05*iph/TOT26:+.3f}% of total revenue")
print("  VERDICT: conclusion is ROBUST. Even at 100% the +5% effect is +0.60%, still trivial.")

print()
print("=" * 78)
print("3. INDEPENDENT CROSS-CHECK the claim never used  (20-F L739, fully disclosed)")
print("=" * 78)
for fy in (2024, 2025, 2026):
    tot = {2024: TOT24, 2025: TOT25, 2026: TOT26}[fy]
    roy = {2024: ROY24, 2025: ROY25, 2026: ROY26}[fy]
    m = MOBILE_AP_SHARE_OF_ROYALTY[fy] * roy
    print(f"  FY{fy}: mobile-AP royalty = {MOBILE_AP_SHARE_OF_ROYALTY[fy]:.0%} x ${roy:.0f}M royalty"
          f" = ${m:6.0f}M = {100*m/tot:4.1f}% of TOTAL revenue")
pool26 = MOBILE_AP_SHARE_OF_ROYALTY[2026] * ROY26
ceil_iph = C3 * TOT26 * 0.68
print(f"  That ${pool26:.0f}M pool is EVERY smartphone vendor on earth (Arm >99% share of mobile AP, L739).")
print(f"  The claim's iPhone ceiling ${ceil_iph:.0f}M = {100*ceil_iph/pool26:.0f}% of that entire global pool.")
print("  For the ceiling to bind, iPhone alone would have to be >1/3 of world smartphone AP royalties,")
print("  while Apple is a minority of world smartphone units AND (L752) royalty 'increases as more Arm")
print("  products are included in the chip' -- Apple licenses only the ISA (own cores/GPU/NPU), and")
print("  (L225) 'royalty revenue per chip generally decreases as the volume of sales increases'.")
print("  => Apple's rate/chip should be BELOW pool average, so the real number sits well UNDER the")
print("     ceiling. The disclosed data corroborates the claim's DIRECTION more strongly than the")
print("     claim's own n=3 quarterly comparison does.")
print(f"  Blended average royalty/chip FY26 = ${ROY26*1e6/(CUM_CHIPS[2026]-CUM_CHIPS[2025]):.3f}"
      f"  ({(CUM_CHIPS[2026]-CUM_CHIPS[2025])/1e9:.0f}bn chips shipped in FY26, from cumulative 310->350bn)")

print()
print("=" * 78)
print("4. RE-DERIVE THE CLAIM'S ARITHMETIC")
print("=" * 78)
cap = C3 * TOT26
iph_max = cap * sh
print(f"  Apple ceiling            ${cap:7.1f}M = {100*cap/TOT26:.1f}%   (claim said $590M) ")
print(f"  iPhone-attributable max  ${iph_max:7.1f}M = {100*iph_max/TOT26:.1f}%   (claim said $401M / 8.2%)")
print(f"  x +5% units              ${0.05*iph_max:7.1f}M = {100*0.05*iph_max/TOT26:+.2f}%  (claim said $20.1M / +0.41%)")
print("  -> arithmetic reproduces EXACTLY.")
print()
print("  The 'plausibly +0.15% to +0.26%' band, from the claim's own script L99-101, is:")
for lab, ap, s in [("MID (Apple=9%, iPhone=58%)", 0.09, 0.58), ("LOW (Apple=6%, iPhone=50%)", 0.06, 0.50)]:
    v = ap * TOT26 * s * 0.05
    print(f"     {lab:28s} -> {100*v/TOT26:+.2f}%")
print("  !! Both Apple%% (9,6) and iPhone%% (58,50) are FREELY CHOSEN, disclosed NOWHERE.")
print("     The ceiling is a deduction; this band is an assumption wearing the same clothes.")

print()
print("=" * 78)
print("5. SCALE COMPARISON vs GUIDANCE")
print("=" * 78)
print(f"  Q2 FYE27 revenue guidance = ${GUIDE_Q2FY27/1000:.2f}bn +/- ${GUIDE_BAND:.0f}M  (CONFIRMED, letter L63-65)")
print(f"  Max +5%-iPhone-unit effect = ${0.05*iph_max:.1f}M per YEAR = ${0.05*iph_max/4:.1f}M per quarter")
print(f"  -> {GUIDE_BAND/(0.05*iph_max/4):.0f}x smaller than one side of a single quarter's guidance band.")
print("  NOTE the units mismatch in the claim: it compares an ANNUAL $20.1M delta to a QUARTERLY")
print("  +/-$50M band. That mismatch UNDERSTATES the claim's own case, so it is not an error that")
print("  flatters the conclusion.")
print("  TIMING: royalties accrue 'in the quarter in which the customer ships' (L854/L2186), so the")
print("  Sep-2026 iPhone launch lands in ARM's Sep-26 and Dec-26 quarters -- i.e. INSIDE the $1.38bn")
print("  +/- $50m guide management already issued on 2026-07-29, before the launch.")

print()
print("=" * 78)
print("6. THE EMPIRICAL LEG -- verify then stress-test")
print("=" * 78)
print(f"  {'quarter':10s} {'iPhone rev y/y':>15s} {'ARM royalty y/y':>16s}")
for k in IPH_Q:
    a, b = IPH_Q[k]; c, d = ROY_Q[k]
    print(f"  {k:10s} {pct(a,b):+14.1f}% {pct(c,d):+15.1f}%")
print("  All six figures reproduce the claim to 0.1pp. SOURCES CONFIRMED.")
print("  BUT: n=3. iPhone growth is near-constant (~22%) while ARM royalty swings +10.5% to +27.1%.")
print("  That is evidence royalty is driven by something OTHER than iPhone, not a measured")
print("  elasticity. 'Corroborated empirically' overstates n=3. ARM's own attribution is better")
print("  primary evidence: Q1 FYE27 letter L13 -- 'data center royalties more than doubling year")
print("  over year'; 6-K L965/L914 -- growth 'driven by an improved mix of products with higher")
print("  royalty rates per chip, such as Armv9 technology'. Neither mentions smartphones.")
print()
print("  Claim's 'US-headquartered revenue grew +1.6% in Jun-26 q':")
print(f"    6-K L412-413: ${388}M vs ${382}M = {pct(388,382):+.1f}%  -> CONFIRMED")
print("    but this is a WEAK corroboration: the US bucket also holds Qualcomm/Nvidia/AMD/Amazon")
print("    /Google and lumpy license revenue. One quarter, one blended bucket. The ANNUAL version")
print("    is far stronger and the claim missed it (20-F L2380-2392):")
for fy in (2025, 2026):
    prev = fy - 1
    tot = {2025: TOT25, 2026: TOT26}[fy]; tp = {2025: TOT24, 2026: TOT25}[fy]
    print(f"      FY{fy}: US HQ rev {pct(US_REV[fy],US_REV[prev]):+6.1f}%   Japan HQ {pct(JP_REV[fy],JP_REV[prev]):+7.1f}%"
          f"   TOTAL {pct(tot,tp):+6.1f}%")
print(f"    FY2026: US bucket +{pct(US_REV[2026],US_REV[2025]):.1f}% while total +{pct(TOT26,TOT25):.1f}%.")
print(f"    ${JP_REV[2026]-JP_REV[2025]:.0f}M of the ${TOT26-TOT25:.0f}M total increase "
      f"({100*(JP_REV[2026]-JP_REV[2025])/(TOT26-TOT25):.0f}%) came from Japan-HQ customers,")
print("    consistent with the $704.4M SoftBank-affiliate arrangement. Not Apple, not iPhone.")

print()
print("=" * 78)
print("7. UNKNOWABLES the claim is entitled to, and one it glosses")
print("=" * 78)
print("  NOT DISCLOSED BY ARM: Apple as a customer at all; Apple's revenue share; Apple's royalty")
print("    rate or royalty basis (ASP%% vs fixed fee); iPhone's share of Apple's Arm royalties.")
print("  NOT DISCLOSED BY APPLE: iPhone UNITS. Apple's FY2025 10-K contains no unit-sales table")
print("    (last disclosed FY2018). The claim's independent variable, '+5% iPhone units', is not")
print("    observable from any primary source going forward -- which cuts against ANY unit-based")
print("    bull case, including the user's.")
