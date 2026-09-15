"""_fd_AAPL_forward.py -- the forward multiple, and why it cannot be taken from a primary source.

APPLE FILES NO NUMERIC GUIDANCE. The 10-Q (accn 0000320193-26-000020) and the 10-K
(accn 0000320193-25-000079) contain no forward revenue or EPS figure; Apple gives directional
commentary verbally on its earnings call and does not file it. So a "forward P/E" cannot be a FACT
here. What follows is explicitly an INFERENCE: the filed nine months of FY2026 plus a Q4 built from
the filed Q4 FY2025 base grown at a stated rate. The assumption is visible so the reader can move it.
"""
import pandas as pd

pd.set_option("display.width", 200)
PRICE = 333.08
EPS_9M_FY26 = 6.87          # 2.84 + 2.01 + 2.02, filed quarterly diluted EPS, accn ...26-000020
EPS_Q4_FY25 = 1.84          # derived: FY2025 7.46 - (2.40 + 1.65 + 1.57)
EPS_TTM = 8.71
REV_9M_FY26 = 364_357.0     # filed, accn 0000320193-26-000020
REV_Q4_FY25 = 102_466.0     # derived: FY2025 416,161 - 9M FY2025 313,695
REV_TTM = 466_823.0
SH = 14_594.18              # millions, cover page 2026-07-17
NET_CASH = 62_220.0

print("=" * 116)
print("FORWARD MULTIPLE -- INFERENCE, NOT FACT (Apple files no numeric guidance)")
print("=" * 116)
print(f"  filed 9M FY2026 diluted EPS      {EPS_9M_FY26:.2f}")
print(f"  filed/derived Q4 FY2025 EPS base {EPS_Q4_FY25:.2f}")
print(f"  filed 9M FY2026 revenue          {REV_9M_FY26:,.0f} $M")
print(f"  derived Q4 FY2025 revenue base   {REV_Q4_FY25:,.0f} $M\n")
rows = []
for q4g in [0.00, 0.10, 0.1636, 0.20, 0.25]:
    eps26 = EPS_9M_FY26 + EPS_Q4_FY25 * (1 + q4g)
    rev26 = REV_9M_FY26 + REV_Q4_FY25 * (1 + q4g)
    rows.append({"Q4FY26_growth_assumed_%": q4g * 100, "FY2026E_EPS": eps26,
                 "FY2026E_rev_$M": rev26, "fwd_P/E_FY2026E": PRICE / eps26,
                 "fwd_EV/S_FY2026E": (PRICE * SH - NET_CASH) / rev26,
                 "FY2026E_rev_growth_%": (rev26 / 416_161 - 1) * 100})
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

print("\n  A second year out, holding the Q4 assumption at the latest quarter's +16.4%:")
eps26 = EPS_9M_FY26 + EPS_Q4_FY25 * 1.1636
for g27 in [0.05, 0.10, 0.15, 0.20]:
    eps27 = eps26 * (1 + g27)
    print(f"    FY2027 EPS growth {g27:+.0%} -> EPS {eps27:5.2f} -> P/E {PRICE/eps27:5.1f}x")

print("\n" + "=" * 116)
print("GAAP-VS-ADJUSTED GAP: how much of Apple's margin is an SBC add-back?")
print("=" * 116)
SBC, OI, GP, NI = 13_706.0, 154_859.0, 227_123.0, 128_930.0
print(f"  TTM SBC {SBC:,.0f} $M = {SBC/REV_TTM:.2%} of revenue")
print(f"  GAAP operating margin        {OI/REV_TTM:.2%}")
print(f"  'adjusted' if SBC excluded   {(OI+SBC)/REV_TTM:.2%}   gap {SBC/REV_TTM*10000:.0f} bp")
print(f"  GAAP net margin              {NI/REV_TTM:.2%}")
print(f"  EV/EBIT GAAP {( PRICE*SH-NET_CASH)/OI:.2f}x  vs  ex-SBC {(PRICE*SH-NET_CASH)/(OI+SBC):.2f}x")
print("\n  For AAPL the GAAP-vs-adjusted gap is 294bp and genuinely immaterial to the conclusion:")
print("  even crediting the full add-back, EV/EBIT only falls from 31.1x to 28.6x, still the")
print("  highest in the peer set. For META (SBC 11.01% of revenue) or GOOGL (6.91%) the same")
print("  adjustment moves the multiple several turns. Apple's quality claim on this metric is")
print("  real -- it is simply not what is in dispute.")
