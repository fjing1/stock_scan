"""ARM valuation: what is priced in at 239.01.

Share counts, net income, revenue: 20-F FY2026 (acc 0001973239-26-000097).
Cash: 6-K Q1 FYE27 shareholder letter (acc 0001973239-26-000113).
Price: yfinance close 2026-09-14.
Consensus: yfinance (analyst aggregate) -- labelled as consensus, not a filing.
"""
import numpy as np
import pandas as pd
import yfinance as yf

pd.set_option("display.width", 220)

PX = 239.01           # yfinance close 2026-09-14
COST = 257.00         # user's cost

# --- from the 20-F FY2026 cover page + Item 6/7 ---
SH_OUT = 1_068_078_760      # ordinary shares outstanding as of 2026-05-21 (20-F)
SH_DILUTED_FY26 = 1_068e6   # weighted avg diluted, FY26 (20-F income statement)
SB_SHARES = 922_733_999     # SoftBank beneficial, 2026-05-21 (20-F)
FLOAT_ADS = 145_344_760     # "our publicly traded ADSs ... is 145,344,760" (20-F)

# --- 20-F FY2026 consolidated income statement ---
FY = pd.DataFrame(
    {
        "FY2024": {"revenue": 3233, "opex": 2968, "op_income": 111, "net_income": 306, "eps_dil": 0.29},
        "FY2025": {"revenue": 4007, "opex": 3055, "op_income": 831, "net_income": 792, "eps_dil": 0.75},
        "FY2026": {"revenue": 4920, "opex": 3899, "op_income": 900, "net_income": 904, "eps_dil": 0.85},
    }
)
CASH_ST = 3888  # cash + ST investments, $m, Q1 FYE27 letter

mc = PX * SH_OUT / 1e9
print("=" * 104)
print("SHARE STRUCTURE  [20-F FY2026]")
print("=" * 104)
print(f"  ordinary shares outstanding (2026-05-21):  {SH_OUT:,}")
print(f"  SoftBank Group beneficial:                 {SB_SHARES:,}  = {100 * SB_SHARES / SH_OUT:.1f}%")
print(f"  publicly traded ADSs (the actual float):   {FLOAT_ADS:,}  = {100 * FLOAT_ADS / SH_OUT:.1f}%")
print(f"  => FREE FLOAT IS {100 * FLOAT_ADS / SH_OUT:.1f}% OF THE COMPANY.")
print(f"  Market cap @ ${PX:.2f}:  ${mc:.1f}bn      Float market value: ${PX * FLOAT_ADS / 1e9:.1f}bn")
print(f"  Enterprise value (cash ${CASH_ST:,}m, no material debt): ~${mc - CASH_ST / 1e3:.1f}bn")

print("\n" + "=" * 104)
print("WHAT 239.01 PAYS FOR  [multiples on FILED FY2026 results]")
print("=" * 104)
r = FY["FY2026"]
ev = mc - CASH_ST / 1e3
print(f"  FY2026 revenue          ${r.revenue:,.0f}m   ->  P/S   = {mc * 1e3 / r.revenue:6.1f}x     EV/S = {ev * 1e3 / r.revenue:6.1f}x")
print(f"  FY2026 GAAP op income   ${r.op_income:,.0f}m   ->  EV/EBIT = {ev * 1e3 / r.op_income:6.1f}x")
print(f"  FY2026 GAAP net income  ${r.net_income:,.0f}m   ->  P/E   = {mc * 1e3 / r.net_income:6.1f}x")
print(f"  FY2026 GAAP diluted EPS ${r.eps_dil:.2f}      ->  P/E   = {PX / r.eps_dil:6.1f}x")
print(f"  FY2026 GAAP operating margin: {100 * r.op_income / r.revenue:.1f}%")
print(f"  TTM non-GAAP FCF (Q1 FYE27 letter): $1,397m -> P/FCF = {mc * 1e3 / 1397:.1f}x, FCF yield {100 * 1397 / (mc * 1e3):.2f}%")

# non-GAAP run-rate from Q1 FYE27 + Q2 guide
ng_q1, ng_q2g = 0.45, 0.47
print(f"\n  Non-GAAP: Q1 FYE27 EPS ${ng_q1:.2f} actual, Q2 guide ${ng_q2g:.2f}")
print(f"    annualising Q1 x4 = ${4 * ng_q1:.2f} -> fwd non-GAAP P/E = {PX / (4 * ng_q1):.0f}x")
print(f"    annualising Q2 guide x4 = ${4 * ng_q2g:.2f} -> {PX / (4 * ng_q2g):.0f}x")
print("  NOTE: non-GAAP excludes share-based comp. FY26 GAAP opex $3,899m vs non-GAAP;")
print("        the GAAP/non-GAAP gap in Q1 FYE27 alone was $531m - $91m = $440m of operating income.")
print(f"        Non-GAAP op margin 41.2% vs GAAP 7.1% -- a {41.2 - 7.1:.1f}pt gap, almost all SBC.")

print("\n" + "=" * 104)
print("REVERSE DCF-STYLE SANITY CHECK: what growth does $239 require?")
print("=" * 104)
print("  Take GAAP net income $904m and ask for how many years ARM must compound to justify $255bn.")
for target_pe in (25, 30, 35):
    for yrs in (5, 10):
        need_ni = mc * 1e3 / target_pe
        g = (need_ni / r.net_income) ** (1 / yrs) - 1
        print(f"    to trade at {target_pe}x GAAP earnings in {yrs}y with NO price appreciation, GAAP net income must")
        print(f"      reach ${need_ni:,.0f}m ({need_ni / r.net_income:.1f}x today) = {100 * g:.1f}%/yr compounding for {yrs} years")
print("  For scale: FY26 GAAP net income grew 14% ($792m -> $904m) on 22.8% revenue growth,")
print("  because GAAP opex grew 27.6% ($3,055m -> $3,899m).")

print("\n" + "=" * 104)
print("PRICE HISTORY: where 239 and 257 sit")
print("=" * 104)
px = pd.read_pickle("_arm_px.pkl")
c = px["Close"]
for label, v in [("current 239.01", PX), ("cost 257.00", COST)]:
    pct = 100 * (c < v).mean()
    print(f"  {label}: {pct:.1f}th percentile of all {len(c)} post-IPO closes")
print(f"  ATH close {c.max():.2f} ({c.idxmax().date()}); {PX} is {100 * (PX / c.max() - 1):+.1f}% from it")
print(f"  2026 fiscal-year-end close (2026-03-31): {c.loc['2026-03-31']:.2f}")
print(f"  => at 239.01 the stock is still {100 * (PX / c.loc['2026-03-31'] - 1):+.1f}% ABOVE its 2026-03-31 level")
print(f"  => and {100 * (PX / c.loc['2025-09-15':'2025-09-16'].iloc[0] - 1):+.1f}% vs one year ago "
      f"({c.loc['2025-09-15':'2025-09-16'].iloc[0]:.2f})")

# valuation history: P/S through time using the then-latest reported TTM revenue
print("\n" + "=" * 104)
print("P/S THROUGH TIME (price x 1,068m shares / latest reported TTM revenue from filings)")
print("=" * 104)
ttm = [  # (date the number became public = 6-K/20-F filing date, TTM revenue $m)
    ("2024-05-08", 3233), ("2024-07-31", 3372), ("2024-11-06", 3661),
    ("2025-02-05", 3861), ("2025-05-07", 4007), ("2025-07-30", 4118),
    ("2025-11-05", 4400), ("2026-02-04", 4670), ("2026-05-06", 4920),
    ("2026-07-29", 5156),
]
rows = []
for d, tr in ttm:
    seg_end = pd.Timestamp(d) + pd.Timedelta(days=95)
    seg = c.loc[d:min(seg_end, c.index[-1])]
    if not len(seg):
        continue
    rows.append({"since": d, "ttm_rev$m": tr, "px_lo": seg.min(), "px_hi": seg.max(),
                 "P/S_lo": seg.min() * 1068 / tr, "P/S_hi": seg.max() * 1068 / tr})
h = pd.DataFrame(rows)
print(h.round(1).to_string(index=False))
print(f"\n  TODAY: P/S = {PX * 1068 / 5156:.1f}x on TTM revenue $5,156m (FY26 $4,920m - Q1FY26 $1,053m + Q1FY27 $1,289m)")
print(f"  Peak on 2026-06-18 (close 439.46): P/S = {439.46 * 1068 / 5156:.1f}x")
print(f"  Trough on 2026-01-30 area: P/S = {c.loc['2026-01-01':'2026-02-03'].min() * 1068 / 4670:.1f}x")

# --- consensus (yfinance, labelled) ---
print("\n" + "=" * 104)
print("CONSENSUS + NEXT EARNINGS  (yfinance aggregate -- NOT a filing; commentary-grade)")
print("=" * 104)
t = yf.Ticker("ARM")
try:
    ed = t.earnings_dates
    print(ed.head(12).to_string())
except Exception as e:
    print("earnings_dates err", e)
try:
    info = t.info
    for k in ["trailingPE", "forwardPE", "trailingEps", "forwardEps", "priceToSalesTrailing12Months",
              "marketCap", "sharesOutstanding", "floatShares", "heldPercentInsiders",
              "targetMeanPrice", "numberOfAnalystOpinions", "revenueGrowth", "earningsGrowth"]:
        if k in info:
            print(f"  {k:34s} {info[k]}")
except Exception as e:
    print("info err", e)
