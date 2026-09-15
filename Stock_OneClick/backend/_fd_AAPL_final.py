"""Final consolidated numbers, each cross-checked against a second independent primary source."""
import numpy as np
P=333.08   # 2026-09-14 close per the measured technical layer
print("="*104)
print("CROSS-CHECK 1: TTM revenue -- XBRL derivation vs the 10-Q's own nine-month table")
print("="*104)
fy25=416161; m9_fy25=313695; m9_fy26=364357
q4fy25=fy25-m9_fy25
print(f"  FY2025 (10-K accn 0000320193-25-000079)                     {fy25:>10,}")
print(f"  9M FY2025 (Q3FY26 10-Q comparative, accn ...26-000020)      {m9_fy25:>10,}")
print(f"  => derived Q4 FY2025                                        {q4fy25:>10,}   press release says '$102.5 billion' -> MATCH")
print(f"  9M FY2026 (10-Q accn ...26-000020)                          {m9_fy26:>10,}")
ttm_rev=m9_fy26+q4fy25
print(f"  TTM revenue                                                 {ttm_rev:>10,}   vs fund_metrics revenue_ttm 416,161 (= stale FY figure)")
print(f"  TTM revenue growth vs prior-year TTM (408,625)              {ttm_rev/408625-1:>+10.1%}")

print("\n"+"="*104)
print("CROSS-CHECK 2: TTM diluted EPS -- XBRL derivation vs the four press releases")
print("="*104)
pr={"Q4FY25 (8-K 2025-10-30)":1.85,"Q1FY26 (8-K 2026-01-29)":2.84,"Q2FY26 (8-K 2026-04-30)":2.01,"Q3FY26 (8-K 2026-07-30)":2.02}
for k,v in pr.items(): print(f"    {k:<28} {v:>5.2f}")
eps_pr=sum(pr.values()); eps_xbrl=1.84+2.84+2.01+2.02
print(f"  TTM EPS from press releases {eps_pr:.2f}   from XBRL (Q4 derived) {eps_xbrl:.2f}   agree within $0.01")
print(f"  P/E at {P:.2f}:  {P/eps_pr:.1f}x  (fund_metrics printed 39.5x because it summed Jun-25+Dec-25+Mar-26+Jun-26,")
print(f"                          skipping the Sep-2025 quarter -> EPS 8.44 instead of {eps_pr:.2f})")
print(f"  ex the one-time tariff refund of $0.11 in Q3FY26: TTM EPS {eps_pr-0.11:.2f} -> P/E {P/(eps_pr-0.11):.1f}x")

print("\n"+"="*104)
print("CROSS-CHECK 3: EV / Sales")
print("="*104)
cash=39544+22855+84118; debt=1997+11007+71340
print(f"  cash+marketable securities (10-Q BS 2026-06-27)  {cash:>9,}   (fund_metrics 146,517 -> MATCH)")
print(f"  commercial paper + term debt cur + noncur        {debt:>9,}   (fund_metrics 84,297; filed sum is {debt:,})")
print(f"  net cash                                        {cash-debt:>9,}")
for sh,lab in [(14656,"diluted (10-Q Note 3)"),(14609,"outstanding (10-Q cover/BS)")]:
    mcap=P*sh/1000; ev=mcap-(cash-debt)/1000
    print(f"  shares {sh:,}M [{lab:<26}] mcap ${mcap:,.0f}M  EV ${ev:,.0f}M  "
          f"EV/TTM-Sales {ev/(ttm_rev/1000):.2f}x   (vs 11.6x on the stale FY figure)")

print("\n"+"="*104)
print("BOUND: how much of iPhone's +21.7% can list-price increases explain?")
print("="*104)
print("  Apple's own newsroom, Pro tier list price (U.S., base storage):")
print("    2024-09-09 iPhone 16 Pro   $  999   / 16 Pro Max $1,199")
print("    2025-09-09 iPhone 17 Pro   $1,099   / 17 Pro Max $1,199    -> Pro +10.0%, Pro Max  0.0%")
print("    2026-09-09 iPhone 18 Pro   $1,199   / 18 Pro Max $1,299    -> Pro  +9.1%, Pro Max +8.3%")
print("    (base iPhone 17 held at $799; iPhone 17e $599; iPhone Duo, new SKU, $1,999)")
g=0.217; pmax=0.100
print(f"\n  Quarter ended 2026-06-27 sold the iPhone 17 generation. The largest list-price rise anywhere")
print(f"  in that generation vs the prior one was +{pmax:.1%} (Pro tier).")
print(f"  Even if EVERY unit sold were that tier, price explains at most +{pmax:.1%} of the +{g:.1%}.")
print(f"  => units and/or mix must supply at least {(1+g)/(1+pmax)-1:+.1%}.  Apple discloses NO unit data")
print(f"     (FY2025 10-K: 'unit sales' 0 hits, 'units sold' 0 hits), so this bound is the deliverable.")
print(f"\n  End market, from a PRIMARY filing (QCOM FY2025 10-K, accn 0000804328-25-000085):")
print(f"    'For calendar year 2025, we estimate that consumer demand for smartphones will remain")
print(f"     approximately flat relative to calendar year 2024.'")
print(f"  => a flat end market with >={(1+g)/(1+pmax)-1:+.1%} unit/mix growth at Apple means SHARE GAIN + mix,")
print(f"     not an industry replacement wave. Bounded by content/price/share, exactly as a mature market implies.")

print("\n"+"="*104)
print("ONE-OFFS AND BASE EFFECTS in the headline growth numbers")
print("="*104)
print("  Q3FY26 gross margin 50.1% includes ~2pp tariff refunds (8-K 2026-07-30). Ex-refund ~48.1% vs 46.5%.")
print("  Q1FY26 48.2% vs 46.9% and Q2FY26 49.3% vs 47.1% contain NO tariff refunds (10-Qs: 0 hits).")
print("  => underlying gross-margin expansion +1.3 to +2.2pp is real and mix-driven; the 3.6pp headline is not.")
print("  Q3FY26 EPS +29% reported; ex-$0.11 refund $1.91 vs $1.57 = +21.7%, i.e. in line with revenue+margin.")
print("  FY2024 Q4 carried a $10.2bn one-time State Aid tax charge (FY2024 10-K accn 0000320193-24-000123),")
print(f"  which depresses the year-ago TTM EPS base of $6.59 by ~$10.2bn/15.3bn sh = ~$0.67.")
b_raw, b_adj = 6.59, 6.59+0.67
print(f"    raw  : TTM EPS {b_raw:.2f} -> {eps_pr:.2f} = {eps_pr/b_raw-1:+.1%}; price +42.0% => multiple {1.420/(eps_pr/b_raw):.3f} = {1.420/(eps_pr/b_raw)-1:+.1%}")
print(f"    clean: TTM EPS {b_adj:.2f} -> {eps_pr:.2f} = {eps_pr/b_adj-1:+.1%}; price +42.0% => multiple {1.420/(eps_pr/b_adj)-1:+.1%}")
print(f"    => on a clean base, roughly {np.log(eps_pr/b_adj)/np.log(1.420)*100:.0f}% of the 1-year return is earnings,")
print(f"       {100-np.log(eps_pr/b_adj)/np.log(1.420)*100:.0f}% is multiple expansion.")
