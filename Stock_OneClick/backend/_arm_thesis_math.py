#!/usr/bin/env python
"""
_arm_thesis_math.py

Tests the user's ARM thesis ("Apple shipped new iPhones -> ARM long-term bullish")
against Arm's own disclosed economics.

EVERY input below is traced to a primary source in the comment above it.
No estimates are presented as facts; assumption-driven lines are labelled ASSUMPTION.
"""

PRICE = 239.01            # given (2026-09-14 close, confirmed via yfinance)
COST = 257.00             # user's cost

# --- Arm Holdings plc 20-F for FY ended 2026-03-31, filed 2026-05-26 (acc 0001973239-26-000097)
FY26 = dict(rev=4920.0, ext=3421.0, rp=1499.0,
            lic_ext=1298.0, lic_rp=1009.0, roy_ext=2123.0, roy_rp=490.0,
            royalty=2613.0, license=2307.0,
            op_inc=900.0, net_inc=904.0, rnd=2776.0, sbc=1052.0, rpo=2071.4,
            purchase_oblig=1056.5)
FY25 = dict(rev=4007.0, ext=3184.0, rp=823.0,
            lic_ext=1421.0, lic_rp=418.0, roy_ext=1763.0, roy_rp=405.0,
            royalty=2168.0, license=1839.0,
            op_inc=831.0, net_inc=792.0, rnd=2071.0, sbc=820.0, rpo=2225.5,
            purchase_oblig=912.0)
FY24 = dict(rev=3233.0, ext=2509.0, rp=724.0, royalty=1802.0,
            rpo=2484.4, purchase_oblig=340.0, sbc=1037.0)

# 20-F Note 20 Related Party Transactions (FY26 20-F)
ARM_CHINA_IPLA = {2026: 790.6, 2025: 670.4, 2024: 670.8}
SOFTBANK_AFFILIATE = {2026: 704.4, 2025: 145.5}     # "Consulting Agreement" licensing/servicing
SB_CONTRACT_ASSET_FY26 = 645.8                       # unbilled, current contract asset

# 20-F Item 4B: "royalty revenue from the mobile applications processors market
# constituted approximately 43% of our total royalty revenue" (FY26)
MOBILE_AP_SHARE_OF_ROYALTY = {2026: 0.43, 2025: 0.46, 2024: 0.35}

# 20-F Item 7A: SoftBank 922,733,999 sh of 1,068,078,760 as of 2026-05-21;
# Arm's own text: "our publicly traded ADSs ... is 145,344,760"
SHARES_OUT = 1_068_078_760
SOFTBANK_SH = 922_733_999
FLOAT_SH = 145_344_760
PLEDGED_SH = 769_029_000     # SoftBank Group Facility margin loan collateral (72.0% of Arm)

# Apple Inc. 10-K FY2025, filed 2025-10-31: iPhone net sales ($M)
IPHONE = {2023: 200_583, 2024: 201_183, 2025: 209_586}

# 20-F Note 4: only 3 customers >=10% of FY26 revenue: 16%, 14%, 12%.
# "No other customer represented 10% or more".  Risk factor: Qualcomm = 9%.
TOP3_FY26 = [0.16, 0.14, 0.12]
QUALCOMM_FY26 = 0.09

sep = '=' * 78


def pct(a, b):
    return (a / b - 1) * 100


print(sep)
print('1. WHERE DID FY26 GROWTH ACTUALLY COME FROM?   [Arm 20-F FY26, Item 5A + Note 20]')
print(sep)
d = FY26['rev'] - FY25['rev']
print(f"Total revenue            {FY25['rev']:>8,.0f} -> {FY26['rev']:>8,.0f}   {pct(FY26['rev'],FY25['rev']):+6.1f}%   (+${d:,.0f}M)")
print(f"  External customers     {FY25['ext']:>8,.0f} -> {FY26['ext']:>8,.0f}   {pct(FY26['ext'],FY25['ext']):+6.1f}%   = {(FY26['ext']-FY25['ext'])/d*100:4.0f}% of the growth")
print(f"  Related parties        {FY25['rp']:>8,.0f} -> {FY26['rp']:>8,.0f}   {pct(FY26['rp'],FY25['rp']):+6.1f}%   = {(FY26['rp']-FY25['rp'])/d*100:4.0f}% of the growth")
print()
sb = SOFTBANK_AFFILIATE[2026] - SOFTBANK_AFFILIATE[2025]
ac = ARM_CHINA_IPLA[2026] - ARM_CHINA_IPLA[2025]
print(f"  of which SoftBank-affiliate 'Consulting Agreement': {SOFTBANK_AFFILIATE[2025]:>6.1f} -> {SOFTBANK_AFFILIATE[2026]:>6.1f}  (+{sb:,.1f}M) = {sb/d*100:.0f}% of ALL company growth")
print(f"  of which Arm China IPLA:                            {ARM_CHINA_IPLA[2025]:>6.1f} -> {ARM_CHINA_IPLA[2026]:>6.1f}  (+{ac:,.1f}M) = {ac/d*100:.0f}% of ALL company growth")
print()
print(f"  Related parties = {FY26['rp']/FY26['rev']*100:.1f}% of FY26 revenue (FY25 {FY25['rp']/FY25['rev']*100:.1f}%, FY24 {FY24['rp']/FY24['rev']*100:.1f}%)")
print(f"  SoftBank affiliate alone = {SOFTBANK_AFFILIATE[2026]/FY26['rev']*100:.1f}% of FY26 revenue")
print(f"  ...of which {SB_CONTRACT_ASSET_FY26/SOFTBANK_AFFILIATE[2026]*100:.0f}% (${SB_CONTRACT_ASSET_FY26:,.1f}M) sat as an UNBILLED contract asset at 2026-03-31")
print()
print(f"  EXTERNAL license & other revenue: {FY25['lic_ext']:,.0f} -> {FY26['lic_ext']:,.0f} = {pct(FY26['lic_ext'],FY25['lic_ext']):+.1f}%  <-- DECLINED")
print(f"  EXTERNAL royalty revenue:         {FY25['roy_ext']:,.0f} -> {FY26['roy_ext']:,.0f} = {pct(FY26['roy_ext'],FY25['roy_ext']):+.1f}%")

print()
print(sep)
print('2. THE APPLE / iPHONE LINK, SIZED FROM ARM\'S OWN DISCLOSURE')
print(sep)
mob = FY26['royalty'] * MOBILE_AP_SHARE_OF_ROYALTY[2026]
print(f"Mobile applications-processor royalty (43% of ${FY26['royalty']:,.0f}M royalty) = ${mob:,.0f}M")
print(f"  = {mob/FY26['rev']*100:.1f}% of Arm's TOTAL revenue")
print(f"  Arm market cap / entire global smartphone-AP royalty pool = {PRICE*SHARES_OUT/1e6/mob:,.0f}x")
print()
print("Mobile-AP share OF ROYALTY over time (20-F Item 4B, each year):")
for y in (2024, 2025, 2026):
    print(f"   FY{y}: {MOBILE_AP_SHARE_OF_ROYALTY[y]*100:.0f}%   (mobile AP royalty ~${(FY24 if y==2024 else FY25 if y==2025 else FY26)['royalty']*MOBILE_AP_SHARE_OF_ROYALTY[y]:,.0f}M)")
print()
print("Tightest bound on APPLE (20-F Note 4 + risk factor; Apple is never named as a customer):")
print(f"   3 customers >=10% of FY26 revenue: {', '.join(f'{x*100:.0f}%' for x in TOP3_FY26)}")
print(f"   16% is disclosed as Arm China; 14% reconciles to the SoftBank affiliate (${SOFTBANK_AFFILIATE[2026]:.1f}M = {SOFTBANK_AFFILIATE[2026]/FY26['rev']*100:.1f}%)")
print(f"   Qualcomm is disclosed at {QUALCOMM_FY26*100:.0f}%, i.e. NOT one of the three >=10% slots")
print(f"   => Apple is EITHER the 12% customer, OR is below 10% of Arm revenue. Not resolvable from filings.")
print(f"   => UPPER BOUND on Apple = 12% of FY26 revenue = ${0.12*FY26['rev']:,.0f}M")
print()
print("ASSUMPTION-BASED sensitivity (labelled: not disclosed):")
for apple_share in (0.05, 0.08, 0.12):
    for vol in (0.05, 0.10):
        print(f"   if Apple = {apple_share*100:>2.0f}% of revenue and Apple UNIT volume rises {vol*100:.0f}%,"
              f" Arm revenue +{apple_share*vol*100:.2f}%  (+${apple_share*vol*FY26['rev']:,.0f}M)")
print()
print("Apple's own numbers (Apple 10-K FY2025, filed 2025-10-31), iPhone net sales $M:")
for y in (2023, 2024, 2025):
    prev = IPHONE.get(y - 1)
    g = f"{pct(IPHONE[y],prev):+.1f}%" if prev else '   n/a'
    print(f"   FY{y}: {IPHONE[y]:>8,}   {g}")
cagr = (IPHONE[2025] / IPHONE[2023]) ** 0.5 - 1
print(f"   FY23->FY25 iPhone revenue CAGR = {cagr*100:.1f}%/yr")
print("   Apple's stated reason for the FY25 increase: 'higher net sales of Pro models' (mix/price, not units).")

print()
print(sep)
print('3. IS THE FORWARD IP PIPELINE GROWING?  [Arm 20-F / 6-K, Remaining Performance Obligations]')
print(sep)
for y, k in ((2024, FY24), (2025, FY25), (2026, FY26)):
    print(f"   FY{y} RPO ${k['rpo']:,.1f}M")
print(f"   FY24->FY26 RPO change: {pct(FY26['rpo'],FY24['rpo']):+.1f}%   while revenue changed {pct(FY26['rev'],FY24['rev']):+.1f}%")
print("   Q1 FY27 (2026-06-30) RPO $2,122.6M vs $2,232.4M a year earlier = -4.9% y/y, on revenue +22% y/y.")
print("   Arm's Q1 FY27 shareholder letter: 'beginning with Q1 FYE27, we are no longer reporting the")
print("   remaining performance obligations and the number of extant Arm Total Access and Arm Flexible")
print("   Access licenses in our shareholder letter.'  ACV (its replacement) grew +13% y/y to $1,732M.")

print()
print(sep)
print('4. OPERATING LEVERAGE AND SHARE-BASED COMP  [Arm XBRL companyfacts]')
print(sep)
print(f"   Revenue        FY25 {FY25['rev']:>7,.0f} -> FY26 {FY26['rev']:>7,.0f}  {pct(FY26['rev'],FY25['rev']):+6.1f}%")
print(f"   GAAP op income FY25 {FY25['op_inc']:>7,.0f} -> FY26 {FY26['op_inc']:>7,.0f}  {pct(FY26['op_inc'],FY25['op_inc']):+6.1f}%")
print(f"   R&D            FY25 {FY25['rnd']:>7,.0f} -> FY26 {FY26['rnd']:>7,.0f}  {pct(FY26['rnd'],FY25['rnd']):+6.1f}%")
print(f"   SBC            FY25 {FY25['sbc']:>7,.0f} -> FY26 {FY26['sbc']:>7,.0f}  {pct(FY26['sbc'],FY25['sbc']):+6.1f}%"
      f"   = {FY26['sbc']/FY26['rev']*100:.1f}% of revenue")
print(f"   GAAP op margin FY25 {FY25['op_inc']/FY25['rev']*100:.1f}% -> FY26 {FY26['op_inc']/FY26['rev']*100:.1f}%")
print("   Q1 FY27: revenue +22%, GAAP op income $91M vs $114M = -20%; GAAP op margin 7.1% vs 10.8%.")
print("            SBC $343M = 26.6% of revenue.  GAAP net income $270M INCLUDES a $128M")
print("            non-operating gain on investments (vs $4M a year earlier).")

print()
print(sep)
print('5. ARM IS NOW BUYING WAFERS  [Arm 20-F/6-K purchase obligations]')
print(sep)
for y, k in ((2024, FY24), (2025, FY25), (2026, FY26)):
    print(f"   FY{y} purchase obligations ${k['purchase_oblig']:,.1f}M")
print(f"   FY24->FY26 growth {pct(FY26['purchase_oblig'],FY24['purchase_oblig']):+.0f}%")
print("   April 2026: +$305.0M amendment with a cloud computing web services provider, through 2029.")
print("   Separately disclosed: ~$100M of purchase commitments over 12 months to arrange third-party")
print("   semiconductor supply to a customer, 'likely to grow materially in subsequent years'.")

print()
print(sep)
print('6. THE FLOAT / OVERHANG  [Arm 20-F Item 7A + IPO 424B4]')
print(sep)
print(f"   Shares outstanding (2026-05-21)  {SHARES_OUT:>15,}")
print(f"   SoftBank Group                   {SOFTBANK_SH:>15,}  = {SOFTBANK_SH/SHARES_OUT*100:.1f}%")
print(f"   Public float (Arm's own figure)  {FLOAT_SH:>15,}  = {FLOAT_SH/SHARES_OUT*100:.1f}%")
print(f"   Market cap  @ ${PRICE}           ${PRICE*SHARES_OUT/1e9:>14,.1f}B")
print(f"   Value of the float               ${PRICE*FLOAT_SH/1e9:>14,.1f}B")
print(f"   Shares pledged to SoftBank margin loan {PLEDGED_SH:,} = {PLEDGED_SH/SHARES_OUT*100:.1f}% of Arm")
print(f"       = {PLEDGED_SH/FLOAT_SH:.2f}x the entire public float")
print("   SoftBank share count: 1,025,233,999 pre-IPO -> 929,733,999 at IPO (90.6%) ->")
print("   922,733,999 in the FY24, FY25 AND FY26 20-Fs. Identical for three years:")
print("   SoftBank has sold nothing since the IPO greenshoe. The 90.6%->86.4% drift is pure dilution.")
print("   Facility has prepayment triggers if 'the trading price of our ADSs declines below certain")
print("   thresholds', and can be margin-called; Arm's risk factor says SoftBank 'may consider it")
print("   advisable to sell shares... which number of shares may... be significant'.")
print("   No lock-up remains; SoftBank has unlimited registration rights (max 3 demands / 12 months).")
print(f"   Selling even 3% of the pledged block = {PLEDGED_SH*0.03/FLOAT_SH*100:.1f}% of the float.")

print()
print(sep)
print('7. WHAT IS PRICED IN  [yfinance 2026-09-14; cross-checked to filings]')
print(sep)
mc = PRICE * SHARES_OUT / 1e9
ttm_rev = 5156.0   # yfinance TTM; = FY26 4,920 - Q1FY26 1,053 + Q1FY27 1,289 = 5,156  (checks)
print(f"   check: FY26 4,920 - Q1FY26 1,053 + Q1FY27 1,289 = {4920-1053+1289:,} vs yfinance TTM {ttm_rev:,.0f}")
print(f"   Market cap ${mc:,.1f}B / TTM revenue ${ttm_rev:,.0f}M = {mc*1000/ttm_rev:.1f}x sales")
print(f"   Trailing GAAP P/E 243.9  |  Forward P/E 78.2  |  EV/EBITDA 236.8  |  P/B 29.6  |  beta 3.89")
gaap_ttm_ni = 904 - 130 + 270
print(f"   TTM GAAP net income = 904 - 130 + 270 = ${gaap_ttm_ni:,}M  -> P/E {mc*1000/gaap_ttm_ni:.0f}x")
print()
print("   Reverse test: what has to happen to justify $239.01 at a 30x exit P/E?")
for yrs in (5, 10):
    for exitpe in (30, 40):
        req_ni = mc * 1000 / exitpe                       # $M of net income needed at exit
        g = (req_ni / gaap_ttm_ni) ** (1 / yrs) - 1
        print(f"     {yrs}yr horizon, exit P/E {exitpe}x, ZERO price return: GAAP net income must reach "
              f"${req_ni:,.0f}M = {req_ni/gaap_ttm_ni:.1f}x today, i.e. {g*100:.1f}%/yr for {yrs}yr")
print()
print("   Same test on the mobile-AP royalty stream the user's thesis actually points at:")
print(f"     entire global smartphone-AP royalty pool = ${mob:,.0f}M/yr.")
print(f"     Market cap is {mc*1000/mob:,.0f}x that pool. Arm already holds >99% share of it")
print("     (20-F: 'market share in the mobile applications processor market of greater than 99%")
print("     for many years'), so there is no share left to win -- only unit growth.")

print()
print(sep)
print('8. POSITION ARITHMETIC')
print(sep)
print(f"   Cost {COST:.2f}, price {PRICE:.2f} = {pct(PRICE,COST):+.1f}%")
print(f"   Breakeven requires +{(COST/PRICE-1)*100:.1f}%")
print("   Price path (yfinance): 2026-05-15 close 209.16 -> 2026-06-18 high close 439.46 (+110% in 5wks)")
print("   -> 2026-09-14 close 239.01. The entire June spike has round-tripped; 239.01 is only")
print(f"   +{(239.01/209.16-1)*100:.1f}% above the pre-spike May base.")
