import pandas as pd
pd.set_option('display.width',250)
# 20-F FY26 L3931: SBC (equity settled) FY26 1,052 / FY25 820 / FY24 1,037
# 6-K Q1FY27 L1082: Q1FY27 343 / Q1FY26 241
# revenue: 20-F L1088 + 6-K L105
d=pd.DataFrame([
 ('FY24',3233,1037),('FY25',4007,820),('FY26',4920,1052),('TTM 6/26',5156,1052-241+343)
],columns=['period','revenue','SBC']).set_index('period')
d['SBC_pct_rev']=(d.SBC/d.revenue*100).round(1)
print("THE BULL'S BEST COUNTER: 'FCF margin will converge to the 43% non-GAAP OM as IPO-era SBC")
print("rolls off, so 13%->40% is mechanical, not a step-change.'  Test it on the filed SBC series:")
print(d.to_string())
print(f"\n  Q1FY27 SBC ${343}m vs Q1FY26 ${241}m = {(343/241-1)*100:+.1f}% YoY")
print(f"  Q1FY27 rev ${1289}m vs Q1FY26 ${1053}m = {(1289/1053-1)*100:+.1f}% YoY")
print("\n  FACT: SBC/revenue fell 32.1%->20.5% after the FY24 IPO year, then STOPPED and REVERSED:")
print("        20.5% (FY25) -> 21.4% (FY26) -> 22.4% (TTM). SBC is growing ~2x as fast as revenue.")
print("  => The 'IPO grants roll off and margins mechanically converge' steelman is REFUTED by the")
print("     filings. There is no observed deleveraging of SBC. This SUPPORTS the claim's direction.")
print("\n  20-F FY26 L688: SoftBank owns 86.4% as of 2026-05-21 -> public float ~13.6% (145.3m ADSs).")
print("  INFERENCE: the reverse DCF multiplies a price set in a 13.6% float by the FULL 1,068m share")
print("  count. 'What the market requires' is really 'what a thin float requires'. This cuts BOTH ways:")
print("  it undermines reading the DCF as a deep consensus, while also explaining how the multiple got")
print("  to 48.8x EV/S. It does NOT rescue the fundamental case for paying 257.")
