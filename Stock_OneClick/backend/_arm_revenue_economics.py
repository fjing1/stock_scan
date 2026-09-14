"""ARM revenue economics -- ALL figures from primary filings.

Sources:
 [A] 20-F FY2026 (acc 0001973239-26-000097, period 2026-03-31), Note 4 "Disaggregation of Revenue"
     -> _arm_txt_20F_FY2026.txt @ offset ~695287
 [B] 20-F FY2026 risk factor "Our revenues predominantly come from a limited number of end markets"
     -> mobile applications processor % of royalty revenue
 [C] 20-F FY2026 risk factor "A significant portion of our total revenue comes from a limited number
     of customers" -> top-5 = 57/56/54%, largest (Arm China) = 16/17/21%
 [D] 6-K Ex-99.2 shareholder letter, Q1 FYE27 (acc 0001973239-26-000113)
"""
import pandas as pd

pd.set_option("display.width", 240)

# ---- [A] Note 4, 20-F FY2026, USD millions -----------------------------------
rev = pd.DataFrame(
    {
        "lic_ext": {"FY2024": 1051, "FY2025": 1421, "FY2026": 1298},
        "lic_rp": {"FY2024": 380, "FY2025": 418, "FY2026": 1009},
        "roy_ext": {"FY2024": 1458, "FY2025": 1763, "FY2026": 2123},
        "roy_rp": {"FY2024": 344, "FY2025": 405, "FY2026": 490},
    }
)
rev["license"] = rev.lic_ext + rev.lic_rp
rev["royalty"] = rev.roy_ext + rev.roy_rp
rev["external"] = rev.lic_ext + rev.roy_ext
rev["relparty"] = rev.lic_rp + rev.roy_rp
rev["total"] = rev.license + rev.royalty

print("=" * 110)
print("[A] ARM REVENUE, 20-F FY2026 Note 4 'Disaggregation of Revenue' ($m, FY ends 3/31)")
print("=" * 110)
print(rev[["lic_ext", "lic_rp", "license", "roy_ext", "roy_rp", "royalty", "external", "relparty", "total"]].to_string())
print("\n  Note 4 total check: FY26 $4,920m / FY25 $4,007m / FY24 $3,233m  <- matches filing")

print("\n--- growth rates (%) ---")
g = rev.pct_change() * 100
print(g[["license", "royalty", "external", "relparty", "total"]].round(1).to_string())

print("\n--- related-party share of total revenue (%) ---")
print((100 * rev.relparty / rev.total).round(1).to_string())

print("\n" + "=" * 110)
print("*** WHERE FY2026's REVENUE GROWTH ACTUALLY CAME FROM ***")
print("=" * 110)
d = rev.loc["FY2026"] - rev.loc["FY2025"]
tot = d["total"]
print(f"  Total revenue increase FY25 -> FY26:                 +${tot:,.0f}m  (+{100 * tot / rev.loc['FY2025', 'total']:.1f}%)")
print(f"    from RELATED PARTIES (SoftBank Group / Arm China):  +${d['relparty']:,.0f}m  = {100 * d['relparty'] / tot:.0f}% of the increase")
print(f"    from EXTERNAL customers:                           +${d['external']:,.0f}m  = {100 * d['external'] / tot:.0f}% of the increase")
print()
print(f"  Related-party LICENSE revenue alone:  ${rev.loc['FY2025', 'lic_rp']:,.0f}m -> ${rev.loc['FY2026', 'lic_rp']:,.0f}m  "
      f"= +${d['lic_rp']:,.0f}m (+{100 * d['lic_rp'] / rev.loc['FY2025', 'lic_rp']:.0f}%)  = {100 * d['lic_rp'] / tot:.0f}% of ALL revenue growth")
print(f"  External  LICENSE revenue:            ${rev.loc['FY2025', 'lic_ext']:,.0f}m -> ${rev.loc['FY2026', 'lic_ext']:,.0f}m  "
      f"= {d['lic_ext']:+,.0f}m ({100 * d['lic_ext'] / rev.loc['FY2025', 'lic_ext']:+.0f}%)   <-- DECLINED")
print(f"  External  ROYALTY revenue:            ${rev.loc['FY2025', 'roy_ext']:,.0f}m -> ${rev.loc['FY2026', 'roy_ext']:,.0f}m  "
      f"= +${d['roy_ext']:,.0f}m (+{100 * d['roy_ext'] / rev.loc['FY2025', 'roy_ext']:.0f}%)")
print()
print(f"  => EXTERNAL-CUSTOMER revenue grew {100 * d['external'] / rev.loc['FY2025', 'external']:.1f}% while HEADLINE revenue grew {100 * tot / rev.loc['FY2025', 'total']:.1f}%.")

# ---- geography, same Note 4 --------------------------------------------------
geo = pd.DataFrame(
    {
        "FY2024": {"United States": 1413, "PRC": 697, "Japan": 121, "Taiwan": 522, "Korea": 308, "Other": 172},
        "FY2025": {"United States": 1716, "PRC": 749, "Japan": 296, "Taiwan": 629, "Korea": 324, "Other": 293},
        "FY2026": {"United States": 1761, "PRC": 874, "Japan": 825, "Taiwan": 695, "Korea": 392, "Other": 373},
    }
)
print("\n" + "=" * 110)
print("[A] REVENUE BY CUSTOMER HQ GEOGRAPHY ($m) -- 20-F FY2026 Note 4")
print("=" * 110)
geo["FY26 yoy %"] = (geo.FY2026 / geo.FY2025 - 1) * 100
geo["FY26 % of tot"] = 100 * geo.FY2026 / geo.FY2026.sum()
geo["chg FY25->26"] = geo.FY2026 - geo.FY2025
print(geo.round(1).to_string())
print(f"\n  Japan (SoftBank Group's domicile): ${geo.loc['Japan', 'FY2024']:,.0f}m -> ${geo.loc['Japan', 'FY2025']:,.0f}m -> ${geo.loc['Japan', 'FY2026']:,.0f}m")
print(f"  Japan alone = ${geo.loc['Japan', 'chg FY25->26']:,.0f}m of the ${tot:,.0f}m total FY26 increase = "
      f"{100 * geo.loc['Japan', 'chg FY25->26'] / tot:.0f}%")
print(f"  United States revenue grew only {geo.loc['United States', 'FY26 yoy %']:.1f}% in FY2026 "
      f"(${geo.loc['United States', 'chg FY25->26']:,.0f}m) -- and Apple, NVIDIA, Qualcomm, Google, AWS, Microsoft are ALL in that bucket.")

# ---- [B] mobile AP ----------------------------------------------------------
mob_share = {"FY2024": 0.35, "FY2025": 0.46, "FY2026": 0.43}
print("\n" + "=" * 110)
print("[B] MOBILE APPLICATIONS PROCESSOR ROYALTY -- from the 20-F risk factors (all vendors combined)")
print("=" * 110)
mob = pd.Series({k: v * rev.loc[k, "royalty"] for k, v in mob_share.items()})
for k in mob.index:
    print(f"  {k}: {mob_share[k]:.0%} of ${rev.loc[k, 'royalty']:,.0f}m royalty = ${mob[k]:,.0f}m "
          f"= {100 * mob[k] / rev.loc[k, 'total']:.1f}% of TOTAL revenue")
print(f"\n  mobile AP royalty y/y: FY25 {100 * (mob.FY2025 / mob.FY2024 - 1):+.1f}%   FY26 {100 * (mob.FY2026 / mob.FY2025 - 1):+.1f}%")
print(f"  FY24->FY26 CAGR: mobile AP royalty {100 * ((mob.FY2026 / mob.FY2024) ** 0.5 - 1):+.1f}%/yr "
      f"vs TOTAL revenue {100 * ((rev.loc['FY2026', 'total'] / rev.loc['FY2024', 'total']) ** 0.5 - 1):+.1f}%/yr")
dm = mob.FY2026 - mob.FY2025
print(f"  Of the +${d['royalty']:,.0f}m royalty increase in FY26, mobile AP = +${dm:,.0f}m ({100 * dm / d['royalty']:.0f}%); "
      f"everything-not-mobile = +${d['royalty'] - dm:,.0f}m ({100 * (d['royalty'] - dm) / d['royalty']:.0f}%)")
print(f"  Mobile AP royalty is {100 * mob.FY2026 / rev.loc['FY2026', 'total']:.1f}% of total revenue and its share of royalty FELL 46% -> 43%.")
print("  20-F FY2026, verbatim: 'We have maintained market share in the mobile applications processor")
print("  market of greater than 99% for many years' AND 'our substantial existing market share may")
print("  limit opportunities for future growth.'  <-- ARM's own filing says mobile share is a CEILING.")

# ---- [C] the Apple bound ---------------------------------------------------
print("\n" + "=" * 110)
print("[C] HOW BIG CAN APPLE BE?  Tightest bounds the disclosures permit")
print("=" * 110)
T26 = rev.loc["FY2026", "total"]
print("  Note 4, 20-F FY2026 verbatim: 'the Company had three customers that collectively represented 42%")
print("  of total revenue, with the single largest customer accounting for 16%, the second largest 14%,")
print("  and the third largest 12%... No other customer represented 10% or more of total revenue.'")
print("  Risk factor: 'our top five customers (including Arm China and SoftBank Group) collectively")
print("  accounted for approximately 57%... our largest customer individually, Arm China, ... 16%.'")
print()
print(f"  Named buckets: #1 Arm China 16% (=${0.16 * T26:,.0f}m), #2 14% (=${0.14 * T26:,.0f}m), #3 12% (=${0.12 * T26:,.0f}m)")
print(f"  #4 + #5 = 57 - 42 = 15% combined, each < 10%")
print(f"  Related-party revenue = ${rev.loc['FY2026', 'relparty']:,.0f}m = {100 * rev.loc['FY2026', 'relparty'] / T26:.1f}% of total.")
print(f"    Arm China (16%) + one more related party at ~14% = ~30% ~= the {100 * rev.loc['FY2026', 'relparty'] / T26:.1f}% related-party total")
print(f"    => INFERENCE: the #2 customer (14%) is very likely the SoftBank Group related party.")
print()
print("  BOUND ON APPLE:")
print(f"    Upper bound if Apple is the #3 customer:  12% of revenue = ${0.12 * T26:,.0f}m")
print(f"    Upper bound if Apple is NOT top-3:        <10% of revenue = <${0.10 * T26:,.0f}m")
print(f"    Independent end-market ceiling: ALL mobile-AP royalty (Apple+Qualcomm+MediaTek+Samsung+")
print(f"      Google+Unisoc+...) = ${mob.FY2026:,.0f}m = {100 * mob.FY2026 / T26:.1f}% of total revenue.")
print(f"      Apple's iPhone AP royalty is a strict SUBSET of that ${mob.FY2026:,.0f}m.")
print("    NOT DISCLOSED: Apple is never named as a customer. 'Apple' appears once in the FY2026 20-F")
print("      (1990 joint-venture founder); 'iPhone' appears ZERO times in the FY2026 and FY2025 20-Fs.")

# ---- sensitivity: what would an iPhone cycle have to do? --------------------
print("\n" + "=" * 110)
print("SENSITIVITY: what could a strong iPhone cycle actually do to ARM revenue?")
print("=" * 110)
print(f"  Total mobile-AP royalty across ALL vendors: ${mob.FY2026:,.0f}m ({100 * mob.FY2026 / T26:.1f}% of revenue).")
print("  Apple's share of global smartphone UNITS is ~18-20% (industry data, not an ARM disclosure),")
print("  and Apple holds a founder-era architecture licence, historically the LOWEST royalty rate")
print("  structure in ARM's book (per-unit rate NOT disclosed - see 'unknowable').")
for ap_share in (0.15, 0.20, 0.30):
    apple_roy = mob.FY2026 * ap_share
    print(f"\n  If Apple = {ap_share:.0%} of mobile-AP royalty -> ${apple_roy:,.0f}m = {100 * apple_roy / T26:.1f}% of total revenue")
    for iph in (0.05, 0.10, 0.20):
        inc = apple_roy * iph
        print(f"     a {iph:+.0%} better iPhone unit cycle -> +${inc:,.0f}m revenue = +{100 * inc / T26:.2f}% on total revenue")

print("\n" + "=" * 110)
print("[D] Q1 FYE27 (qtr ended 6/30/26), 6-K Ex-99.2 shareholder letter")
print("=" * 110)
print("  Revenue $1,289m +22% y/y | royalty $715m +22% | license $574m +23%")
print("  non-GAAP EPS $0.45 vs $0.35 (+29%), BEAT guidance of $0.40 +/- 0.04")
print("  Q2 FYE27 guidance: revenue $1.38bn +/- $50m; non-GAAP EPS $0.47 +/- 0.04")
print("  GAAP operating margin FELL to 7.1% from 10.8%: GAAP opex +28% vs revenue +22%")
print("  ACV $1,732m, +13% y/y  <-- committed-fee base growing at HALF the reported revenue rate")
print("  Verbatim: 'beginning with Q1 FYE27, we are no longer reporting the remaining performance")
print("  obligations and the number of extant Arm Total Access and Arm Flexible Access licenses'")
print("  <-- two forward-looking disclosures REMOVED in the same quarter.")
print(f"\n  ACV $1,732m annualised committed licence fees vs FY26 actual licence revenue ${rev.loc['FY2026', 'license']:,.0f}m")
print(f"  => reported FY26 licence revenue is {rev.loc['FY2026', 'license'] / 1732:.2f}x the annualised committed base.")
