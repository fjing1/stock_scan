"""ARM growth rate, adjusted for the SoftBank Group consulting agreement.
All inputs from 20-F FY2026 and the 6-K Ex-99.2 letters. No estimates.
"""
print("=" * 104)
print("ARM REVENUE GROWTH, HEADLINE vs EX-SOFTBANK-CONSULTING-AGREEMENT")
print("=" * 104)
# 20-F FY2026: total revenue; related-party section: SoftBank Consulting Agreement revenue
FY = {"FY2025": (4007.0, 145.5), "FY2026": (4920.0, 704.4)}
print(f"  {'':8} {'total rev':>11} {'SB consult':>11} {'ex-SB rev':>11} {'headline y/y':>13} {'ex-SB y/y':>11}")
prev = None
for k in ("FY2025", "FY2026"):
    tot, sb = FY[k]
    ex = tot - sb
    if prev:
        hy = 100 * (tot / prev[0] - 1)
        ey = 100 * (ex / prev[1] - 1)
        print(f"  {k:8} {tot:>11,.0f} {sb:>11,.1f} {ex:>11,.1f} {hy:>12.1f}% {ey:>10.1f}%")
    else:
        print(f"  {k:8} {tot:>11,.0f} {sb:>11,.1f} {ex:>11,.1f} {'-':>13} {'-':>11}")
    prev = (tot, ex)
tot26, sb26 = FY["FY2026"]
tot25, sb25 = FY["FY2025"]
print(f"\n  *** FY2026 headline revenue growth +{100 * (tot26 / tot25 - 1):.1f}%.")
print(f"      Excluding the consulting agreement with its own 86.4% controlling shareholder: "
      f"+{100 * ((tot26 - sb26) / (tot25 - sb25) - 1):.1f}%. ***")
print(f"      The consulting agreement supplied ${sb26 - sb25:.1f}m of the ${tot26 - tot25:.0f}m total increase "
      f"= {100 * (sb26 - sb25) / (tot26 - tot25):.0f}%.")

print("\n  Q1 FYE27 (6-K 2026-07-29, related-party note): consulting revenue $192.9m vs $126.1m")
q27, q26 = 1289.0, 1053.0
c27, c26 = 192.9, 126.1
print(f"    headline Q1 y/y +{100 * (q27 / q26 - 1):.1f}%   ex-consulting +{100 * ((q27 - c27) / (q26 - c26) - 1):.1f}%")
print(f"    consulting revenue is now {100 * c27 / q27:.1f}% of quarterly revenue, run-rating ~${4 * c27:.0f}m/yr")

print("\n" + "=" * 104)
print("REVENUE QUALITY: the consulting revenue is largely UNBILLED")
print("=" * 104)
print("  20-F FY2026 related-party section, verbatim:")
print("    'For the fiscal years ended March 31, 2026 and 2025, revenue from the licensing and")
print("     servicing arrangements was $704.4 million and $145.5 million, respectively, and as of")
print("     March 31, 2026 and 2025, the Company had current contract assets of $645.8 million and")
print("     $145.5 million from an affiliate of SoftBank Group.'")
print(f"  -> ${645.8:.1f}m of the ${704.4:.1f}m recognised ({100 * 645.8 / 704.4:.0f}%) was still an unbilled")
print("     contract asset at year-end. Contract assets are revenue booked before the right to")
print("     invoice becomes unconditional.")
print("  And: 'we also agreed to modify the structure of one of the statements of work under the")
print("       Consulting Agreement resulting in a fixed payment amount of $300 million, which will")
print("       be paid during the fiscal year ending March 31, 2027.'")
print(f"  -> ${704.4:.1f}m of revenue recognised; one SOW restructured to a ${300}m fixed cash payment.")
print("  6-K Q1 FYE27 balance sheet: total contract assets $881m current + $366m non-current = $1,247m,")
print("     of which $576.1m + $1.1m = $577.2m is from the SoftBank affiliate = "
      f"{100 * 577.2 / 1247:.0f}% of ALL contract assets.")

print("\n" + "=" * 104)
print("GAAP EARNINGS QUALITY, Q1 FYE27 (6-K 2026-07-29 income statement)")
print("=" * 104)
print("  Revenue $1,289m -> GAAP OPERATING income only $91m (7.1% margin, down from 10.8%).")
print("  GAAP NET income $270m, of which:")
print("    +$128m 'income from equity investments, net'  (non-operating)")
print("    +$31m  interest income, net                   (non-operating)")
print("    +$17m  income tax BENEFIT                     (non-operating)")
print(f"    -> non-operating items = $176m of the $253m pre-tax = {100 * 176 / 253:.0f}% of pre-tax income.")
print("  Share-based compensation in the quarter (cash-flow statement): $343m = "
      f"{100 * 343 / 1289:.1f}% of revenue,")
print("    i.e. 3.8x the entire GAAP operating income. Non-GAAP EPS of $0.45 excludes it.")
print("  Employee payroll taxes payable jumped $119m -> $375m 'primarily related to vested RSUs'")
print("    -- that is a REAL CASH cost that scales with the share price, and it is excluded from")
print("    non-GAAP. A higher share price mechanically raises ARM's cash opex.")

print("\n" + "=" * 104)
print("THE GROSS-MARGIN CLIFF THAT HAS NOT ARRIVED YET")
print("=" * 104)
print("  20-F FY2026: cost of sales $121m on revenue $4,920m -> 98% gross margin.")
print("    'Cost of sales is comprised primarily of the costs of providing technical support and")
print("     training to our customers.' There is no silicon in there.")
print("  20-F FY2026: 'In March 2026, we announced the expansion of our compute platform into")
print("    production silicon products with the Arm AGI CPU, which DID NOT HAVE A MATERIAL IMPACT")
print("    to our revenue for the fiscal year ended March 31, 2026' and production is 'expected by")
print("    the end of calendar year 2026.'")
print("  Q1 FYE27 letter: AGI CPU customer demand 'now exceeds $2 billion across fiscal 2027 and 2028'.")
print("  Risk factor: 'we recently entered into an agreement to arrange for certain semiconductor")
print("    products to be supplied by a third party to a customer... purchase commitments of")
print("    approximately $100 million to be purchased over the next 12 months. We expect that our")
print("    purchase commitments are likely to GROW MATERIALLY in subsequent years.'")
print()
print("  ARITHMETIC of the mix shift (illustrative, my calculation, not a company figure):")
ip_rev, ip_gm = 5900.0, 0.98   # ~FY27 IP revenue at ~20% growth
for si_rev in (500, 1000, 1500):
    for si_gm in (0.35, 0.50, 0.65):
        blended = (ip_rev * ip_gm + si_rev * si_gm) / (ip_rev + si_rev)
        print(f"    IP ${ip_rev:,.0f}m @98% + silicon ${si_rev:,}m @{si_gm:.0%} "
              f"-> blended gross margin {100 * blended:.1f}%  (vs 98.0% today)")
print("  Each point of blended gross margin on ~$6-7bn of revenue is ~$65m of gross profit. The")
print("  business ARM is being valued as (98%-margin IP toll road) is not the business it is")
print("  becoming (IP + merchant silicon). A P/S of 49.5x is a multiple for the former.")
print()
print("  And the strategic cost, from ARM's OWN risk factor, verbatim:")
print("    'Many of our semiconductor and systems company customers who have historically licensed")
print("     our IP may face direct competition from us in certain market segments... these customers")
print("     or partners may terminate or materially reduce their relationship with us, seek")
print("     alternative architectures or products, develop their own proprietary architectures,")
print("     withhold sensitive roadmap information... or pursue legal action.'")
print("  ARM is now a competitor to the customers who pay it royalties. That is the actual change")
print("  in the business -- and it cuts in BOTH directions, unlike 'Apple sells iPhones'.")
