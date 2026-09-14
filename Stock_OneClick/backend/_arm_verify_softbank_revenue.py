"""Verify the claim that FY2026 ARM revenue growth is 61% attributable to the
SoftBank-affiliate Consulting Agreement. All inputs are transcribed from the
primary filings downloaded to this directory (see SOURCES below)."""
import re, sys

# ---------------- SOURCES ----------------
# 20-F FY2026, accession 0001973239-26-000097, filed 2026-05-26 (arm-20260331.htm)
#   Note 4 - Revenue, "Disaggregation of Revenue" table
#   Note 20 - Related Party Transactions
#   Item 5 MD&A revenue table
# 20-F FY2025, accession 0001973239-25-000016, filed 2025-05-28
# 6-K filed 2026-07-29, accession 0001973239-26-000114 (arm-20260630.htm) Q1 FY2027

# Note 4 disaggregation, $M
note4 = {
    'lic_ext':  {2026: 1298, 2025: 1421, 2024: 1051},
    'lic_rp':   {2026: 1009, 2025:  418, 2024:  380},
    'roy_ext':  {2026: 2123, 2025: 1763, 2024: 1458},
    'roy_rp':   {2026:  490, 2025:  405, 2024:  344},
}
note4['lic_tot'] = {y: note4['lic_ext'][y] + note4['lic_rp'][y] for y in (2026,2025,2024)}
note4['roy_tot'] = {y: note4['roy_ext'][y] + note4['roy_rp'][y] for y in (2026,2025,2024)}
note4['total']   = {y: note4['lic_tot'][y] + note4['roy_tot'][y] for y in (2026,2025,2024)}
note4['rp_tot']  = {y: note4['lic_rp'][y]  + note4['roy_rp'][y]  for y in (2026,2025,2024)}

# Note 20 named related-party revenue, $M (as-filed decimals)
armchina = {2026: 790.6, 2025: 670.4, 2024: 670.8}
softbank = {2026: 704.4, 2025: 145.5, 2024: 0.0}   # FY24: Note 20 FY25 20-F says no
                                                    # AR/contract asset/liability existed
ampere   = {2026: 3.6, 2025: 3.5, 2024: 49.3}
other_sb = {2026: 0.0, 2025: 0.0, 2024: 4.4}

def pct(a, b): return (a/b - 1) * 100

print("="*72)
print("1. DOES THE FILING SUPPORT THE HEADLINE ARITHMETIC?")
print("="*72)
print(f"Total revenue         FY26 {note4['total'][2026]:>6}  FY25 {note4['total'][2025]:>6}  "
      f"{pct(note4['total'][2026], note4['total'][2025]):+.1f}%   (20-F says '+23%', $913M)")
print(f"  delta = {note4['total'][2026]-note4['total'][2025]}")
d_sb = softbank[2026] - softbank[2025]
d_tot = note4['total'][2026] - note4['total'][2025]
print(f"SoftBank consulting   FY26 {softbank[2026]:>6.1f}  FY25 {softbank[2025]:>6.1f}  delta {d_sb:+.1f}")
print(f"  share of total revenue increase = {d_sb/d_tot*100:.1f}%   (claim: 61%)")

ex26, ex25 = note4['total'][2026]-softbank[2026], note4['total'][2025]-softbank[2025]
print(f"Ex-SoftBank total     FY26 {ex26:>6.1f}  FY25 {ex25:>6.1f}  {pct(ex26,ex25):+.2f}%   (claim +9.2%)")
lx26, lx25 = note4['lic_tot'][2026]-softbank[2026], note4['lic_tot'][2025]-softbank[2025]
print(f"Ex-SB licence+other   FY26 {lx26:>6.1f}  FY25 {lx25:>6.1f}  {pct(lx26,lx25):+.2f}%   (claim -5.4%)")
print(f"Royalty (untouched)   FY26 {note4['roy_tot'][2026]:>6}  FY25 {note4['roy_tot'][2025]:>6}  "
      f"{pct(note4['roy_tot'][2026],note4['roy_tot'][2025]):+.1f}%")

print()
print("="*72)
print("2. IS THE $704.4M REALLY ALL INSIDE 'LICENCE AND OTHER REVENUE'?")
print("   (the -5.4% figure collapses if part of it sits in royalty)")
print("="*72)
# Note 20 gives Arm China TOTAL revenue only, not its licence/royalty split.
# Test: assume ALL related-party royalty is Arm China (20-F FY26 Item 7B says the
# SoftBank SOW royalty terms "will be negotiated at a later date" => none yet;
# Ampere total is only $3.6M).
for y in (2026, 2025, 2024):
    ac_lic_implied = armchina[y] - note4['roy_rp'][y]
    recon = ac_lic_implied + softbank[y] + ampere[y] + other_sb[y]
    print(f"FY{y}: implied Arm China licence = {armchina[y]:.1f} - {note4['roy_rp'][y]} "
          f"= {ac_lic_implied:.1f}")
    print(f"       + SoftBank {softbank[y]:.1f} + Ampere {ampere[y]:.1f} + other {other_sb[y]:.1f}"
          f" = {recon:.1f}  vs reported RP licence {note4['lic_rp'][y]}  "
          f"residual {recon-note4['lic_rp'][y]:+.1f}")

print()
print("Independent residual test on TOTAL related-party revenue:")
for y in (2026, 2025, 2024):
    named = armchina[y] + softbank[y] + ampere[y] + other_sb[y]
    print(f"  FY{y}: reported RP total {note4['rp_tot'][y]}  named sum {named:.1f}  "
          f"unexplained {note4['rp_tot'][y]-named:+.1f}")

print()
print("="*72)
print("3. CORROBORATION FROM LINES THE CLAIM DID NOT USE")
print("="*72)
# Note 4 footnote (1): over-time vs point-in-time split of licence & other
overtime = {2026: 1080, 2025: 467, 2024: 121}
pointin  = {2026: 1227, 2025: 1372, 2024: 1310}
print("Note 4 fn(1) licence&other by timing of recognition, $M:")
for y in (2026,2025,2024):
    print(f"  FY{y}: over-time {overtime[y]:>5}   point-in-time {pointin[y]:>5}   "
          f"sum {overtime[y]+pointin[y]:>5} (=reported {note4['lic_tot'][y]})")
print(f"  over-time delta FY26 vs FY25 = {overtime[2026]-overtime[2025]:+} "
      f"vs SoftBank delta {d_sb:+.1f}")
print(f"  POINT-IN-TIME (classic one-off IP licences): "
      f"{pct(pointin[2026],pointin[2025]):+.1f}% YoY, and "
      f"{pct(pointin[2026],pointin[2024]):+.1f}% vs FY2024")

geo = {'United States': (1761,1716,1413), 'PRC': (874,749,697), 'Japan': (825,296,121),
       'Taiwan': (695,629,522), 'Korea': (392,324,308), 'Other': (373,293,172)}
print("\nNote 4 revenue by customer HQ geography, $M (FY26/FY25/FY24) and YoY:")
for k,(a,b,c) in geo.items():
    print(f"  {k:<14} {a:>5} {b:>5} {c:>5}   {pct(a,b):+7.1f}%")
print(f"  Japan delta FY26 vs FY25 = {825-296:+} vs SoftBank delta {d_sb:+.1f} "
      f"(SoftBank Group Corp HQ = Tokyo)")

print("\nItem 3.D customer concentration, 20-F FY2026:")
print("  FY26 top-3 = 42%: #1 16%, #2 14%, #3 12%.  Item 3.D names Arm China as #1.")
print(f"  SoftBank affiliate as % of total revenue: "
      f"FY26 {softbank[2026]/note4['total'][2026]*100:.1f}%  "
      f"FY25 {softbank[2025]/note4['total'][2025]*100:.1f}%")
print("  => the 14% #2 customer slot is consistent with the SoftBank affiliate.")
print("  FY25 20-F risk factor: 'top five customers (including Arm China)'")
print("  FY26 20-F risk factor: 'top five customers (including Arm China AND SOFTBANK GROUP)'")

print()
print("="*72)
print("4. DOES IT PERSIST? Q1 FY2027 6-K (filed 2026-07-29)")
print("="*72)
q = {'Consulting Agreement rev': (192.9, 126.1), 'Ampere rev': (5.4, 0.6)}
for k,(a,b) in q.items():
    print(f"  {k:<26} Jun-26 Q {a:>6.1f}   Jun-25 Q {b:>6.1f}   {pct(a,b):+.1f}%")
print(f"  annualised run-rate of consulting rev = {192.9*4:.1f} vs FY2026 actual {softbank[2026]}")
print("  Related-party contract assets: $646.0M (31-Mar-26) -> $577.2M (30-Jun-26)")
print("  Q1 FY27 geography: US 388 vs 382 = %+.1f%%; Japan 276 vs 154 = %+.1f%%"
      % (pct(388,382), pct(276,154)))

print()
print("="*72)
print("5. THE GAAP / NON-GAAP SWITCH IN THE CLAIM'S LAST SENTENCE")
print("="*72)
print("  20-F FY2026 GAAP income statement:")
rd = (2776, 2071); sga = (1115, 984); opex = (3899, 3055)
print(f"    R&D            {rd[0]} vs {rd[1]}   {pct(*rd):+.1f}%   (claim asserts +43% 'non-GAAP')")
print(f"    SG&A           {sga[0]} vs {sga[1]}   {pct(*sga):+.1f}%")
print(f"    Total opex     {opex[0]} vs {opex[1]}   {pct(*opex):+.1f}%  (claim asserts +33% 'non-GAAP')")
print("  => GAAP R&D +34.0%, GAAP opex +27.6%. The claim's +43%/+33% are NOT the 20-F numbers.")

print()
print("="*72)
print("6. DOES THE FY26 PATTERN PERSIST? Q1 FY2027 ex-SoftBank")
print("   Source: 6-K acc 0001973239-26-000114 (arm-20260630.htm), filed 2026-07-29")
print("="*72)
q_tot=(1289,1053); q_lic=(574,468); q_roy=(715,585); q_sb=(192.9,126.1)
print(f"  Total revenue        {q_tot[0]} vs {q_tot[1]}   {pct(*q_tot):+.1f}%")
print(f"  Licence & other      {q_lic[0]} vs {q_lic[1]}   {pct(*q_lic):+.1f}%")
print(f"  Royalty              {q_roy[0]} vs {q_roy[1]}   {pct(*q_roy):+.1f}%")
print(f"  SoftBank consulting  {q_sb[0]} vs {q_sb[1]}  = {q_sb[0]/q_tot[0]*100:.1f}% of Q revenue")
print(f"  SB share of the {q_tot[0]-q_tot[1]}M revenue increase = "
      f"{(q_sb[0]-q_sb[1])/(q_tot[0]-q_tot[1])*100:.1f}%   (FY26 full year was 61.2%)")
exq=(q_tot[0]-q_sb[0], q_tot[1]-q_sb[1]); exl=(q_lic[0]-q_sb[0], q_lic[1]-q_sb[1])
print(f"  EX-SOFTBANK total    {exq[0]:.1f} vs {exq[1]:.1f}   {pct(*exq):+.1f}%  "
      f"(FY26 full year: +9.2%)")
print(f"  EX-SOFTBANK lic&oth  {exl[0]:.1f} vs {exl[1]:.1f}   {pct(*exl):+.1f}%  "
      f"(FY26 full year: -5.4%)")
print("  Contract asset from SB affiliate 645.8 (31Mar26) -> 577.2 (30Jun26);")
print(f"  implied billed in the quarter = 192.9 + (645.8-577.2) = {192.9+645.8-577.2:.1f}M")
print("  => it is converting to receivables/cash, and still growing.")

print()
print("="*72)
print("7. NON-GAAP FIGURES, traced to 6-K acc 0001973239-26-000062 Ex-99.2")
print("   (GAAP to Non-GAAP Reconciliation, fiscal years ended 31-Mar-26 / 31-Mar-25)")
print("="*72)
print(f"  Non-GAAP R&D    1,911 vs 1,340   {pct(1911,1340):+.1f}%  (claim said +43%)")
print(f"  Non-GAAP opex   2,717 vs 2,049   {pct(2717,2049):+.1f}%  (claim said +33%)")
