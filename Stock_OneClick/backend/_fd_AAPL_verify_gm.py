"""Part 2: gross-margin decomposition + base-effect tests for the AAPL Q3 FY26 claim.
All inputs are FACTS lifted from 10-Q acc 0000320193-26-000020 MD&A gross-margin table
and press release acc 0000320193-26-000018 (EX-99.1), reproduced in the text dumps
_fd_AAPL_verify_10q.txt / _fd_AAPL_verify_pr_q3fy26.txt.
"""

# --- FACT: 10-Q MD&A gross margin table ($M) ---
Q3FY26 = dict(rev=109417, prod_rev=78678, svc_rev=30739,
              gm_tot=54770, gm_prod=31525, gm_svc=23245)   # prod GM = 54770-23245
Q3FY25 = dict(rev=94036, prod_rev=66613, svc_rev=27423,
              gm_tot=43718, gm_prod=22993, gm_svc=20725)
M9FY26 = dict(rev=364357, gm_tot=178782, gm_svc=69963, gm_prod=108819)
M9FY25 = dict(rev=313695, gm_tot=146860, gm_svc=60670, gm_prod=86190)

print("=== check MD&A table internal consistency (FACT) ===")
for n, d in (("Q3FY26", Q3FY26), ("Q3FY25", Q3FY25)):
    print(f" {n}: prodGM+svcGM = {d['gm_prod']+d['gm_svc']:,} vs total {d['gm_tot']:,}"
          f"   GM% {d['gm_tot']/d['rev']*100:.1f}%  prod% {d['gm_prod']/d['prod_rev']*100:.1f}%"
          f"  svc% {d['gm_svc']/d['svc_rev']*100:.1f}%")
print(f" 9M FY26 prodGM+svcGM = {108819+69963:,} vs {178782:,};  GM% {178782/364357*100:.1f}%")
print(f" 9M FY25 GM% {146860/313695*100:.1f}%")

print("\n=== Services share of gross margin (claim: 47.4% -> 42.4%) ===")
a = Q3FY26['gm_svc']/Q3FY26['gm_tot']*100
b = Q3FY25['gm_svc']/Q3FY25['gm_tot']*100
print(f"  Q3FY26 {a:.2f}%   Q3FY25 {b:.2f}%   change {a-b:+.2f}pp   -> CLAIM REPRODUCES")
print(f"  BUT Services GM DOLLARS: {Q3FY25['gm_svc']:,} -> {Q3FY26['gm_svc']:,} "
      f"= {Q3FY26['gm_svc']/Q3FY25['gm_svc']*100-100:+.1f}%  (grew)")
print(f"  Services GM% itself: {Q3FY25['gm_svc']/Q3FY25['svc_rev']*100:.1f}% -> "
      f"{Q3FY26['gm_svc']/Q3FY26['svc_rev']*100:.1f}%  (10-Q: 'flat')")
print("  => share fell only because PRODUCTS gm% jumped "
      f"{Q3FY25['gm_prod']/Q3FY25['prod_rev']*100:.1f}% -> "
      f"{Q3FY26['gm_prod']/Q3FY26['prod_rev']*100:.1f}%")

print("\n=== how much of the Services-GM-share decline is the tariff refund itself? ===")
for pp in (1.5, 2.0, 2.5):
    refund = Q3FY26['rev']*pp/100
    gm_ex = Q3FY26['gm_tot'] - refund
    share_ex = Q3FY26['gm_svc']/gm_ex*100
    print(f"  refund {pp:.1f}pp of co. revenue = ${refund:,.0f}M -> total GM {gm_ex:,.0f} "
          f"({gm_ex/Q3FY26['rev']*100:.1f}%), Services share {share_ex:.2f}% "
          f"= {share_ex-b:+.2f}pp vs prior yr (reported {a-b:+.2f}pp); "
          f"refund manufactures {(1-(share_ex-b)/(a-b))*100:.0f}% of the decline")

print("\n=== gross margin expansion: refund vs underlying (claim: 2pp of 3.6pp = 56%) ===")
exp = Q3FY26['gm_tot']/Q3FY26['rev']*100 - Q3FY25['gm_tot']/Q3FY25['rev']*100
print(f"  reported expansion {exp:+.2f}pp (50.1% vs 46.5%); refund ~2.0pp = {2.0/exp*100:.0f}% "
      f"-> CLAIM'S 56% REPRODUCES")
refund = Q3FY26['rev']*0.02
print(f"  ex-refund company GM {(Q3FY26['gm_tot']-refund)/Q3FY26['rev']*100:.2f}% "
      f"= {(Q3FY26['gm_tot']-refund)/Q3FY26['rev']*100 - Q3FY25['gm_tot']/Q3FY25['rev']*100:+.2f}pp "
      "STILL EXPANDING")
print(f"  ex-refund PRODUCTS GM {(Q3FY26['gm_prod']-refund)/Q3FY26['prod_rev']*100:.2f}% "
      f"vs {Q3FY25['gm_prod']/Q3FY25['prod_rev']*100:.2f}% = "
      f"{(Q3FY26['gm_prod']-refund)/Q3FY26['prod_rev']*100-Q3FY25['gm_prod']/Q3FY25['prod_rev']*100:+.2f}pp"
      "  (Pro-model mix, not refund)")

print("\n=== EPS ex-refund (claim: +22% not +29%) ===")
eps26, eps25_impl = 2.02, 2.02/1.29
print(f"  PR FACT: $2.02, +29% incl $0.11 refund. implied prior-yr EPS ~${eps25_impl:.3f}")
for prior in (1.56, 1.57, 1.58):
    print(f"   if prior=${prior:.2f}: reported {(eps26/prior-1)*100:+.1f}%, "
          f"ex-refund {((eps26-0.11)/prior-1)*100:+.1f}%")
print("  -> ex-refund ~+22% REPRODUCES")

print("\n=== REVENUE: can the tariff refund contribute to the +16.4%? ===")
print("  10-Q FACT (verbatim): 'has recognized any refunds received as a reduction of")
print("  products cost of sales.'  => refund is a COST item; contribution to revenue = $0.")
print(f"  +16.4% = {Q3FY26['rev']:,}/{Q3FY25['rev']:,}-1 = "
      f"{Q3FY26['rev']/Q3FY25['rev']*100-100:.2f}% -- a pure net-sales ratio.")

print("\n=== base-effect test: 2-year stacked growth (June quarters) ===")
june = {2023: 81797, 2024: 85777, 2025: 94036, 2026: 109417}
for y in (2025, 2026):
    two = june[y]/june[y-2]
    print(f"  Q3FY{y}: 1yr {june[y]/june[y-1]*100-100:+.1f}%  2yr cum {two*100-100:+.1f}% "
          f"(CAGR {(two**0.5-1)*100:+.1f}%)")
print("  -> 2-yr CAGR 7.2% -> 13.0%: acceleration survives the 2-yr stack, NOT a soft base")

print("\n=== Services share OF REVENUE (thesis said 'Services mix rising') ===")
for n, d in (("Q3FY25", Q3FY25), ("Q3FY26", Q3FY26)):
    print(f"  {n}: {d['svc_rev']/d['rev']*100:.2f}%")
print(f"  9M FY25 {80408/313695*100:.2f}% -> 9M FY26 {91728/364357*100:.2f}%")

print("\n=== Services quarterly YoY band (from XBRL series, date-matched) ===")
svc = [("2024-12-28",13.9),("2025-03-29",11.6),("2025-06-28",13.3),
       ("2025-12-27",13.9),("2026-03-28",16.3),("2026-06-27",12.1)]
vals=[v for _,v in svc]
print("  last 6 q:", ", ".join(f"{d[-5:]} {v:+.1f}%" for d,v in svc))
print(f"  range {min(vals):.1f}%-{max(vals):.1f}%, mean {sum(vals)/len(vals):.1f}%; "
      f"current 12.1% is ABOVE the 11.6% trough printed in Q2FY25")
print("  annual: FY23 +9.1%, FY24 +12.9%, FY25 +13.5%, 9M FY26 +14.1% -> ACCELERATING")
