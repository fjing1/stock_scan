import pandas as pd
pd.set_option('display.width',240)
P=239.01; COST=257.0
print("=== DEFENSIBLE FAIR-VALUE RANGE, three independent methods ===")
methods=[
 ('Reverse DCF, continuation (20% CAGR, 40% term FCF margin, 9% WACC)', 116, 116),
 ('Reverse DCF, aggressive-but-arguable (25% CAGR, 45% margin)',        188, 188),
 ('Reverse DCF, heroic (28% CAGR, 50% margin) = today\'s price',        260, 260),
 ('Peer EV/Sales, median 13.8x -> max 22.1x (MRVL, faster grower)',      70, 110),
 ('Peer fwd P/E on ARM consensus FY28 non-GAAP $3.06 (21.8x-32.4x)',     66,  99),
]
for n,lo,hi in methods:
    print(f"  {n:66s} ${lo:>4}  - ${hi:>4}")
lo_all,hi_all=66,260
print()
print(f"Union of all methods:                 ${lo_all} - ${hi_all}")
print(f"Excluding the single heroic DCF case:  $66 - $188")
print(f"Current price ${P}: at/above the top of every method except the heroic DCF.")
print(f"User's cost ${COST}: ABOVE the top of every defensible method except the heroic DCF (${260}).")
print(f"  -> cost sits {COST/188-1:+.0%} above the top of the non-heroic range, and {COST/110-1:+.0%} above the max peer EV/Sales value.")
print(f"  -> to break even at ${COST}, ARM must deliver the heroic case (28% rev CAGR for a decade to $61bn revenue,")
print(f"     with FCF margin expanding 13% -> 50%) AND the market must keep paying for it 10 years from now.")
print()
print("=== CAPEX / BUSINESS-MODEL CHANGE (20-F + 6-K cash flow statements) ===")
cap=pd.DataFrame({'PP&E capex $m':[34,64,92,219,545],'Revenue $m':[2703,2679,3233,4007,4920]},
                 index=['FY2022','FY2023','FY2024','FY2025','FY2026'])
cap['capex % of rev']=(cap['PP&E capex $m']/cap['Revenue $m']*100).round(1)
print(cap.to_string())
print("  Q1 FY2027 alone: PP&E capex $197m (vs $154m Q1 FY26) -> run-rate ~$790m/yr, ~15% of revenue")
print("  Property & equipment, net: $772m (Mar-26) -> $923m (Jun-26), +20% in one quarter")
print("  20-F: 'we also expanded our offerings to include our first production silicon product, the Arm AGI CPU'")
print("  Letter: 'secured the manufacturing capacity required'; demand 'exceeds $2 billion across fiscal 2027 and 2028'")
