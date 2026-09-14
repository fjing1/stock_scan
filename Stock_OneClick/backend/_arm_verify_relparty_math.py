"""Adversarial verification of the claim:

  "Arm's FY2026 headline +23% revenue growth is 74% related-party, with 61% of ALL company
   growth coming from a single SoftBank Group affiliate under a Consulting Agreement
   ($145.5M -> $704.4M), 92% of which was unbilled contract asset; external license
   revenue actually declined 8.7%."

Method: every figure below was typed by hand from a FRESH download of the primary source,
not copied from any prior script in this repo.

PRIMARY SOURCES
  [A] Arm Holdings plc Form 20-F, FY ended 2026-03-31, filed 2026-05-26,
      accession 0001973239-26-000097, primary doc arm-20260331.htm.
      Re-downloaded from
      https://www.sec.gov/Archives/edgar/data/1973239/000197323926000097/arm-20260331.htm
      md5 377660b8b258e5316f92d587d912ff18 (matches the copy already on disk).
      Saved: _arm_verify_20f_fy2026_fresh.htm / .txt
      - Consolidated Income Statements, p.126 (audited, Deloitte & Touche LLP, 2026-05-26)
      - Item 5A "Comparison of Performance for the Fiscal Years Ended March 31, 2026 and 2025"
        -> external x related-party BY license/royalty matrix
      - Note 4, "Disaggregation of Revenue" -> same matrix, 3 years
      - Note 20, "Related Party Transactions"
      - Item 7B, "Transactions with SoftBank Group - Consulting Agreement"
  [B] Arm Holdings plc Form 6-K, Q1 FY2027 (3 months ended 2026-06-30), filed 2026-07-29.
      Local text: _arm_txt_Q1FY27_6K.txt

RESULT: claim NOT refuted on the facts. Scope correction required -- see FORWARD TEST.
"""

# ---------------------------------------------------------------- [A] FY2026 20-F
# Consolidated Income Statements, p.126 (GAAP, audited). $M.
ext = {2026: 3421, 2025: 3184, 2024: 2509}          # Revenue from external customers
rp = {2026: 1499, 2025: 823, 2024: 724}             # Revenue from related parties
tot = {2026: 4920, 2025: 4007, 2024: 3233}          # Total revenue

# Item 5A revenue matrix / Note 4 Disaggregation of Revenue. $M.
lic_ext = {2026: 1298, 2025: 1421, 2024: 1051}
lic_rp = {2026: 1009, 2025: 418, 2024: 380}
roy_ext = {2026: 2123, 2025: 1763, 2024: 1458}
roy_rp = {2026: 490, 2025: 405, 2024: 344}

# Note 20 Related Party Transactions. $M.
china = {2026: 790.6, 2025: 670.4, 2024: 670.8}     # revenue under the IPLA with Arm China
sbaff = {2026: 704.4, 2025: 145.5}                  # "an affiliate of SoftBank Group", Consulting Agmt
ampere = {2026: 3.6, 2025: 3.5, 2024: 49.3}
sbaff_contract_asset = {2026: 645.8, 2025: 145.5}   # CURRENT contract assets, balance-sheet date
# Note 20 verbatim: "the Company did not recognize material revenue or other income from other
# entities controlled by SoftBank Group besides an affiliate of SoftBank Group and Ampere."

# ---------------------------------------------------------------- [B] Q1 FY2027 6-K
q_ext = {2027: 901, 2026: 725}                      # 3 months ended June 30
q_rp = {2027: 388, 2026: 328}
q_tot = {2027: 1289, 2026: 1053}
q_lic_ext = {2027: 310, 2026: 255}
q_roy_ext = {2027: 591, 2026: 470}
q_sbaff = {2027: 192.9, 2026: 126.1}                # Consulting Agreement revenue in the quarter
q_sbaff_ca = 576.1 + 1.1                            # current + non-current, as of 2026-06-30


def rule(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


rule("0. INTEGRITY: does the filing cross-foot?")
for y in (2026, 2025, 2024):
    assert ext[y] + rp[y] == tot[y], y
    assert lic_ext[y] + roy_ext[y] == ext[y], y
    assert lic_rp[y] + roy_rp[y] == rp[y], y
print("  external + related-party == total, and license + royalty == each column: OK, all 3 years")
n20 = china[2026] + sbaff[2026] + ampere[2026]
print(f"  Note 20 components {china[2026]}+{sbaff[2026]}+{ampere[2026]} = {n20:.1f} vs "
      f"income-statement related-party {rp[2026]}  (delta {rp[2026]-n20:+.1f}, rounding)")

rule("1. THE CLAIM'S FOUR NUMBERS, RECOMPUTED  [source A]")
g = tot[2026] - tot[2025]
g_ext, g_rp = ext[2026] - ext[2025], rp[2026] - rp[2025]
g_sb, g_cn = sbaff[2026] - sbaff[2025], china[2026] - china[2025]
print(f"  headline total growth        {tot[2025]} -> {tot[2026]}  {tot[2026]/tot[2025]-1:+.2%}   "
      f"(filing says '23%')            CLAIM SAYS +23%   OK")
print(f"  related-party share of grow  +{g_rp}/+{g} = {g_rp/g:.1%}                      "
      f"CLAIM SAYS 74%    OK")
print(f"  SoftBank affiliate share     +{g_sb:.1f}/+{g} = {g_sb/g:.1%}                    "
      f"CLAIM SAYS 61%    OK")
print(f"  external license & other     {lic_ext[2025]} -> {lic_ext[2026]} = {lic_ext[2026]/lic_ext[2025]-1:+.2%}  "
      f"(filing says '(9)%')  CLAIM SAYS -8.7%  OK")
print(f"  contract asset / SB-aff rev  {sbaff_contract_asset[2026]}/{sbaff[2026]} = "
      f"{sbaff_contract_asset[2026]/sbaff[2026]:.1%}                  CLAIM SAYS 92%    OK")
print(f"\n  (also: Arm China IPLA +{g_cn:.1f} = {g_cn/g:.1%} of all growth; "
      f"external growth +{g_ext} = {g_ext/g:.1%})")
print(f"  external total {ext[2026]/ext[2025]-1:+.2%} (filing '7%'); "
      f"related-party {rp[2026]/rp[2025]-1:+.2%} (filing '82%')")

rule("2. WHAT THE CLAIM'S SENTENCE LEAVES OUT  [source A, same table]")
print(f"  external ROYALTY revenue     {roy_ext[2025]} -> {roy_ext[2026]} = "
      f"{roy_ext[2026]/roy_ext[2025]-1:+.2%}   <-- the durable line, GREW")
print(f"  external license & other     {lic_ext[2025]} -> {lic_ext[2026]} = "
      f"{lic_ext[2026]/lic_ext[2025]-1:+.2%}   <-- the lumpy line, the one quoted")
print(f"  external TOTAL               {ext[2025]} -> {ext[2026]} = {ext[2026]/ext[2025]-1:+.2%}   "
      f"<-- external business did NOT shrink")
print("  Arm's own stated cause of the license move (Item 5A): 'fluctuation in timing and size")
print("  of multiple high-value license agreements'.")
print("\n  growth rate on various ex- bases:")
print(f"    headline                       {tot[2026]/tot[2025]-1:+.2%}")
print(f"    ex SoftBank affiliate only     {(tot[2026]-sbaff[2026])/(tot[2025]-sbaff[2025])-1:+.2%}")
print(f"    ex all related party           {ext[2026]/ext[2025]-1:+.2%}")

rule("3. IS '92% UNBILLED' A STOCK/FLOW CONFLATION?  [source A Note 20 + source B]")
billed = sbaff_contract_asset[2025] + sbaff[2026] - sbaff_contract_asset[2026]
cum = sbaff[2025] + sbaff[2026]
print(f"  $645.8M is a BALANCE at 2026-03-31; $704.4M is a FY26 FLOW. Ratio = "
      f"{sbaff_contract_asset[2026]/sbaff[2026]:.1%} is arithmetically right but mixes stock and flow.")
print(f"  transferred out of the contract asset during FY26 = "
      f"{sbaff_contract_asset[2025]}+{sbaff[2026]}-{sbaff_contract_asset[2026]} = ${billed:.1f}M")
print(f"  cumulative FY25+FY26 Consulting Agmt revenue ${cum:.1f}M, still unbilled ${sbaff_contract_asset[2026]}M "
      f"= {sbaff_contract_asset[2026]/cum:.1%} of cumulative")
print(f"  by 2026-06-30 the SB-affiliate contract asset had DRAWN DOWN to ${q_sbaff_ca:.1f}M "
      f"({q_sbaff_ca/sbaff_contract_asset[2026]-1:+.1%})  [source B Note 19]")
print("  source A Item 7B also discloses a CONTRACTUAL $300M fixed payment 'which will be paid")
print("  during the fiscal year ending March 31, 2027' -- i.e. the asset is not open-ended.")
print("  source B Note 4: $308.7M of contract assets transferred to accounts receivable in Q1 FY27;")
print("  Q1 FY27 operating cash flow $902M vs $332M a year earlier.")

rule("4. FORWARD TEST -- DOES THE PATTERN HOLD?  [source B, 6-K filed 2026-07-29]")
qg = q_tot[2027] - q_tot[2026]
qg_ext, qg_rp = q_ext[2027] - q_ext[2026], q_rp[2027] - q_rp[2026]
qg_sb = q_sbaff[2027] - q_sbaff[2026]
print(f"  Q1 FY27 total revenue      {q_tot[2026]} -> {q_tot[2027]} = {q_tot[2027]/q_tot[2026]-1:+.1%}")
print(f"    external                 {q_ext[2026]} -> {q_ext[2027]} = {q_ext[2027]/q_ext[2026]-1:+.1%}"
      f"   = {qg_ext/qg:.1%} of growth")
print(f"    related party            {q_rp[2026]} -> {q_rp[2027]} = {q_rp[2027]/q_rp[2026]-1:+.1%}"
      f"   = {qg_rp/qg:.1%} of growth")
print(f"    Consulting Agreement     {q_sbaff[2026]} -> {q_sbaff[2027]} = "
      f"{q_sbaff[2027]/q_sbaff[2026]-1:+.1%}  = {qg_sb/qg:.1%} of growth")
print(f"    external license & other {q_lic_ext[2026]} -> {q_lic_ext[2027]} = "
      f"{q_lic_ext[2027]/q_lic_ext[2026]-1:+.1%}   <-- FY26's -8.7% REVERSED")
print(f"    external royalty         {q_roy_ext[2026]} -> {q_roy_ext[2027]} = "
      f"{q_roy_ext[2027]/q_roy_ext[2026]-1:+.1%}")
print(f"\n  FY26: related party = {g_rp/g:.0%} of growth. Q1 FY27: related party = {qg_rp/qg:.0%} of growth.")
print("  => The decomposition is a TRUE statement about FY2026 and an OBSOLETE statement about")
print("     Arm's current revenue mix trajectory. Latest reported quarter is the mirror image.")

rule("5. RELATED-PARTY REVENUE WAS NEVER SMALL -- what is actually NEW?")
for y in (2024, 2025, 2026):
    print(f"  FY{y}: related party {rp[y]/tot[y]:5.1%} of revenue | "
          f"Arm China IPLA {china[y]/tot[y]:5.1%} | "
          f"SB affiliate {(sbaff.get(y,0.0))/tot[y]:5.1%}")
print("  => Arm China IPLA has been ~16-21% of revenue every year since before the IPO: NOT news.")
print("     The genuinely new item is the SoftBank affiliate: 0.0% -> 3.6% -> 14.3% of revenue.")
print("     The claim correctly centres on that and does not overclaim on Arm China.")

rule("6. UNDISCLOSED-AS-FACT CHECK")
print("  The filing says only 'an affiliate of SoftBank Group' -- it is never named.")
print("  The claim says 'a single SoftBank Group affiliate'. That MATCHES the disclosure and does")
print("  not invent an identity. No royalty rate and no named customer is asserted. PASSES.")
print("  Note 20 explicitly rules out other material SoftBank-controlled revenue, so the")
print("  3-component decomposition is complete, not a selective subset.")
