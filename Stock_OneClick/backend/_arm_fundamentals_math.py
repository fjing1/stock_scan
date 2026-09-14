"""ARM fundamentals arithmetic. Every input labelled with its primary source."""

# ---- FACT: annual revenue split (in $M) -------------------------------------
# FY21-23: 424B4 IPO prospectus (2023-09-14), Note 3 Disaggregation of Revenue
# FY24-26: 20-F FY2026 (filed 2026-05-26), Note 4 Disaggregation of Revenue
ANNUAL = {  # fiscal year ended 31 March: (license_and_other, royalty)
 2021:(714,1313), 2022:(1141,1562), 2023:(1004,1675),
 2024:(1431,1802), 2025:(1839,2168), 2026:(2307,2613),
}
print("=== ANNUAL ($M) — 424B4 (FY21-23), 20-F FY2026 Note 4 (FY24-26) ===")
print(f"{'FY':>5} {'Lic':>7} {'Roy':>7} {'Tot':>7} {'Lic%':>6} {'Roy%':>6} {'Roy/Tot':>8}")
prev=None
for fy,(l,r) in ANNUAL.items():
    t=l+r
    lg=rg=None
    if prev: lg=100*(l/prev[0]-1); rg=100*(r/prev[1]-1)
    print(f"{fy:>5} {l:>7} {r:>7} {t:>7} {('%+.1f'%lg) if lg is not None else '   n/a':>6} {('%+.1f'%rg) if rg is not None else '   n/a':>6} {100*r/t:>7.1f}%")
    prev=(l,r)

# ---- FACT: quarterly revenue split (in $M) ---------------------------------
# Source: each quarter's 6-K EX-99.2 shareholder letter "Financial Overview"
Q = [  # (label, license, royalty)
 ("FY25Q1 Jun-24",472,467),("FY25Q2 Sep-24",330,514),("FY25Q3 Dec-24",403,580),("FY25Q4 Mar-25",634,607),
 ("FY26Q1 Jun-25",468,585),("FY26Q2 Sep-25",515,620),("FY26Q3 Dec-25",505,737),("FY26Q4 Mar-26",819,671),
 ("FY27Q1 Jun-26",574,715),
]
print("\n=== QUARTERLY ($M) — 6-K EX-99.2 shareholder letters ===")
print(f"{'Qtr':>14} {'Lic':>6} {'Roy':>6} {'Tot':>6} {'Roy YoY':>8} {'Lic YoY':>8}")
for i,(lab,l,r) in enumerate(Q):
    t=l+r
    ry=ly=''
    if i>=4:
        ry='%+.1f%%'%(100*(r/Q[i-4][2]-1)); ly='%+.1f%%'%(100*(l/Q[i-4][1]-1))
    print(f"{lab:>14} {l:>6} {r:>6} {t:>6} {ry:>8} {ly:>8}")

# ---- FACT: average royalty per Arm-based chip -------------------------------
# chips shipped (millions): 424B4 "Number of chips shipped" operating metric
CHIPS = {2021:25281, 2022:29190, 2023:30583}
print("\n=== IMPLIED AVG ROYALTY PER CHIP (only years where units are disclosed) ===")
for fy,n in CHIPS.items():
    print(f"  FY{fy}: ${ANNUAL[fy][1]}M / {n}M chips = ${1e6*ANNUAL[fy][1]/(1e6*n):.4f}/chip")
print("  NOTE: 'Number of chips shipped' was DISCONTINUED after the IPO; absent in all three 20-Fs.")

# ---- FACT: mobile applications-processor share of royalty -------------------
# 20-F FY2024 = ~35%; 20-F FY2025 = ~46%; 20-F FY2026 = ~43% (risk factor + Item 4B)
MOB = {2024:0.35, 2025:0.46, 2026:0.43}
print("\n=== MOBILE AP ROYALTY ($M) — % from each 20-F, applied to that year's royalty ===")
for fy,p in MOB.items():
    print(f"  FY{fy}: {p:.0%} x ${ANNUAL[fy][1]}M = ${p*ANNUAL[fy][1]:,.0f}M   (non-mobile ${(1-p)*ANNUAL[fy][1]:,.0f}M)")
m25,m26=MOB[2025]*ANNUAL[2025][1], MOB[2026]*ANNUAL[2026][1]
n25,n26=(1-MOB[2025])*ANNUAL[2025][1],(1-MOB[2026])*ANNUAL[2026][1]
print(f"  FY26 mobile-AP royalty growth   = {100*(m26/m25-1):+.1f}%")
print(f"  FY26 NON-mobile royalty growth  = {100*(n26/n25-1):+.1f}%")
# rounding sensitivity on the "approximately" percentages
lo=(0.425*ANNUAL[2026][1])/(0.465*ANNUAL[2025][1])-1
hi=(0.435*ANNUAL[2026][1])/(0.455*ANNUAL[2025][1])-1
print(f"  rounding band on mobile growth  = {100*lo:+.1f}% .. {100*hi:+.1f}%")

# ---- FACT: SoftBank-affiliate related-party revenue -------------------------
# 20-F FY2026 Note 20 + Item 7B: Consulting Agreement revenue
SB = {2025:145.5, 2026:704.4}
AMPERE = {2025:3.5, 2026:3.6}
print("\n=== REVENUE QUALITY: strip the SoftBank Consulting Agreement (Note 20) ===")
for fy in (2025,2026):
    l,r=ANNUAL[fy]; t=l+r
    print(f"  FY{fy}: reported total ${t}M ; SoftBank-affiliate consulting ${SB[fy]}M = {100*SB[fy]/t:.1f}% of revenue")
t25=sum(ANNUAL[2025]); t26=sum(ANNUAL[2026])
ex25=t25-SB[2025]; ex26=t26-SB[2026]
print(f"  reported total growth            = {100*(t26/t25-1):+.1f}%")
print(f"  ex-SoftBank-affiliate growth     = {100*(ex26/ex25-1):+.1f}%   (${ex25:,.1f}M -> ${ex26:,.1f}M)")
lx25=ANNUAL[2025][0]-SB[2025]; lx26=ANNUAL[2026][0]-SB[2026]
print(f"  reported license+other growth    = {100*(ANNUAL[2026][0]/ANNUAL[2025][0]-1):+.1f}%")
print(f"  ex-SoftBank license+other growth = {100*(lx26/lx25-1):+.1f}%   (${lx25:,.1f}M -> ${lx26:,.1f}M)")
print(f"  share of FY26 total revenue INCREASE from SoftBank affiliate = {100*(SB[2026]-SB[2025])/(t26-t25):.1f}%")
# Q1 FY27 6-K Note: consulting revenue 192.9 vs 126.1
q27,q26 = 1289, 1053
s27,s26 = 192.9, 126.1
print(f"  Q1FY27: reported +{100*(q27/q26-1):.1f}% ; ex-SoftBank +{100*((q27-s27)/(q26-s26)-1):.1f}%  (SB = {100*s27/q27:.1f}% of qtr revenue)")

# ---- Apple bound ------------------------------------------------------------
print("\n=== APPLE UPPER BOUND (20-F FY2026 concentration note + Note 20) ===")
TOT26=t26
armchina=790.6; sbaff=SB[2026]
print(f"  20-F: three customers >=10% at 16%, 14%, 12%; no other >=10%.")
print(f"  Arm China (Note 20)      = ${armchina}M = {100*armchina/TOT26:.1f}%  -> the '16%' customer")
print(f"  SoftBank affiliate       = ${sbaff}M = {100*sbaff/TOT26:.1f}%  -> the '14%' customer")
print(f"  => remaining '12%' customer is EXTERNAL, approx ${0.12*TOT26:,.0f}M")
print(f"  Qualcomm separately disclosed at 9% = ${0.09*TOT26:,.0f}M")
cap = 0.12*TOT26
print(f"  Therefore Apple <= ${cap:,.0f}M (12%) if Apple IS that customer; else Apple < ${0.10*TOT26:,.0f}M (10%).")

# iPhone share of Apple's Arm-chip base -- bracketed
# Apple FY2025 10-K: iPhone 209,586 ; Mac 33,708 ; iPad 28,023 ; WHA 35,686 ; Services 109,158
hw = 209586+33708+28023+35686
print(f"\n  Apple FY2025 10-K: iPhone ${209586/1000:,.1f}bn of ${hw/1000:,.1f}bn product revenue = {100*209586/hw:.1f}% of hardware $")
print("  Royalties are PER CHIP, not per dollar, so revenue share is an UPPER bound on iPhone's unit share")
print("  (AirPods/Watch/accessories are high-unit, low-revenue). Bracket iPhone at 50%-68% of Apple Arm units.")

for lab, apple_pct, iph_share in [("MAX  (Apple=12%, iPhone=68% of units)",0.12,0.68),
                                  ("MID  (Apple= 9%, iPhone=58%)",0.09,0.58),
                                  ("LOW  (Apple= 6%, iPhone=50%)",0.06,0.50)]:
    apple=apple_pct*TOT26; iph=apple*iph_share
    print(f"\n  {lab}")
    print(f"     Apple total ARM revenue      ~ ${apple:,.0f}M ({100*apple/TOT26:.1f}% of ARM FY26 revenue)")
    print(f"     iPhone-attributable          ~ ${iph:,.0f}M ({100*iph/TOT26:.2f}% of ARM FY26 revenue)")
    for g in (0.05,0.10):
        d=iph*g
        print(f"     +{g:.0%} iPhone UNITS -> +${d:,.1f}M = {100*d/TOT26:+.3f}% of ARM total revenue")

print("\n  Reference scale: Q2 FY27 revenue guidance = $1.38bn +/- $50m (6-K 2026-07-29).")
print(f"  One quarter's guidance band (+/-$50M) = ${200:,}M annualised, i.e. {200/ (0.12*TOT26*0.68*0.05):.0f}x the MAX +5%-iPhone-unit effect.")
