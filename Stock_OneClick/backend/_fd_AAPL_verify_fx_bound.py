"""ADVERSARIAL VERIFY #3 -- the FX hole in the claim's attribution.

The claim decomposes iPhone's +21.7% into list price (<=+10.0%) and 'units/mix/share' (>=+10.6%).
Apple's OWN Q3FY26 10-Q (accn 0000320193-26-000020) says, in the Segment Operating Performance
section, that 'the strength in foreign currencies relative to the U.S. dollar had a net favorable
year-over-year impact' on Europe (9M) and Rest of Asia Pacific (Q3) net sales. Translation FX is
NEITHER price NOR units NOR mix NOR share. Apple does not quantify it in any filing (it is call
commentary only), so the deliverable is a BOUND.

Bound = (non-US revenue share) x (revenue-weighted YoY move of the major currencies over the
fiscal quarter). Currency weights are proxied by the DISCLOSED segment revenue shares, which is the
tightest thing a filing supports; Apple does not disclose currency-of-denomination.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

# ---- geographic segment revenue, from the Q3FY26 10-Q instance (parsed in _fd_AAPL_verify_seg.py)
D = pd.read_csv("_fd_AAPL_verify_seg_raw.csv", parse_dates=["start", "end"])
Q = D[(D.days >= 80) & (D.days <= 100)]
geo = Q[Q.member.str.contains("StatementBusinessSegmentsAxis", na=False)]
geo = geo.assign(seg=geo.member.str.split("=").str[-1].str.replace("SegmentMember", "", regex=False))
gp = geo.pivot_table(index="end", columns="seg", values="val", aggfunc="max") / 1e6
gp = gp.sort_index()
print("=" * 120)
print("A. NET SALES BY REPORTABLE SEGMENT ($M) -- Q3FY26 10-Q accn 0000320193-26-000020 and priors")
print("=" * 120)
print(gp.to_string(float_format=lambda x: f"{x:,.0f}"))
tot = gp.sum(axis=1)
print("\n  segment sum vs consolidated (should tie):")
cons = Q[Q.member.isna() | (Q.member == "")].groupby("end").val.max() / 1e6
print(pd.DataFrame({"segsum": tot, "consolidated": cons}).to_string(float_format=lambda x: f"{x:,.0f}"))

# DATE-MATCHED, not positional: the Sept (Q4) quarter is absent from the quarterly panel,
# so gp.index[-5] reaches back 15 months. This is the exact trap the brief warns about.
import pandas as _pd
cur = gp.index[-1]
prior = _pd.Timestamp("2025-06-28")
assert prior in gp.index
print(f"\n  YoY by segment, {prior.date()} -> {cur.date()}:")
for s in gp.columns:
    a, b = gp.loc[cur, s], gp.loc[prior, s]
    print(f"    {s:<16} {b:>9,.0f} -> {a:>9,.0f}   {a/b-1:+7.1%}   share of total increment "
          f"{(a-b)/(tot[cur]-tot[prior]):>6.1%}")

nonus_share = 1 - gp.loc[cur, "Americas"] / tot[cur]
print(f"\n  Americas share of Q3FY26 net sales {gp.loc[cur,'Americas']/tot[cur]:.1%}"
      f"  ->  non-Americas {nonus_share:.1%}")
print("  NOTE: 'Americas' includes Canada + Latin America, so USD-denominated share is LOWER than")
print("        the Americas share. The 10-Q's US-specific disclosure is in the geographic note:")
us = Q[Q.member.str.contains("StatementGeographicalAxis=US", na=False)]
if not us.empty:
    for _, r in us.sort_values("end").tail(4).iterrows():
        print(f"        US net sales {r.end.date()}  {r.val/1e6:>9,.0f}M")

# ---- FX: revenue-weighted YoY move of the majors over the fiscal quarter
print("\n" + "=" * 120)
print("B. FX MOVE OVER THE FISCAL QUARTER (market data, yfinance) -- Apr/Jun 2026 vs Apr/Jun 2025")
print("=" * 120)
PAIRS = {"EURUSD=X": "EUR", "JPY=X": "JPY(inv)", "CNY=X": "CNY(inv)", "GBPUSD=X": "GBP",
         "AUDUSD=X": "AUD", "CAD=X": "CAD(inv)", "INR=X": "INR(inv)", "KRW=X": "KRW(inv)",
         "DX-Y.NYB": "DXY"}
cur_win = ("2026-03-29", "2026-06-27")
pri_win = ("2025-03-30", "2025-06-28")
rows = []
for t, lab in PAIRS.items():
    try:
        d = yf.download(t, start="2025-01-01", end="2026-09-15", progress=False, auto_adjust=False)
        c = d["Close"]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
        c.index = c.index.tz_localize(None)
        a = float(c[(c.index >= cur_win[0]) & (c.index <= cur_win[1])].mean())
        b = float(c[(c.index >= pri_win[0]) & (c.index <= pri_win[1])].mean())
        inv = "(inv)" in lab
        move = (b / a - 1) if inv else (a / b - 1)   # >0 = currency STRONGER vs USD
        rows.append({"pair": t, "ccy": lab, "avg_prior": b, "avg_cur": a, "ccy_vs_usd": move})
    except Exception as e:
        rows.append({"pair": t, "ccy": lab, "avg_prior": np.nan, "avg_cur": np.nan, "ccy_vs_usd": np.nan})
FX = pd.DataFrame(rows)
print(FX.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

# weights: proxy each segment by its dominant currency basket
W = {"Europe": ("EUR/GBP basket", 0.5 * FX.set_index("ccy").loc["EUR", "ccy_vs_usd"]
                + 0.2 * FX.set_index("ccy").loc["GBP", "ccy_vs_usd"]
                + 0.3 * FX.set_index("ccy").loc["INR(inv)", "ccy_vs_usd"]),
     "GreaterChina": ("CNY", FX.set_index("ccy").loc["CNY(inv)", "ccy_vs_usd"]),
     "Japan": ("JPY", FX.set_index("ccy").loc["JPY(inv)", "ccy_vs_usd"]),
     "RestOfAsiaPacific": ("AUD/KRW basket", 0.5 * FX.set_index("ccy").loc["AUD", "ccy_vs_usd"]
                           + 0.5 * FX.set_index("ccy").loc["KRW(inv)", "ccy_vs_usd"]),
     "Americas": ("mostly USD; CAD+LatAm minority", 0.15 * FX.set_index("ccy").loc["CAD(inv)", "ccy_vs_usd"])}
print("\n  translation impact bound, segment by segment (prior-year revenue x currency move):")
tot_fx = 0.0
for s, (lab, mv) in W.items():
    base = gp.loc[prior, s]
    imp = base * mv
    tot_fx += imp
    print(f"    {s:<18} prior-yr rev {base:>9,.0f}M  x  {lab:<28} {mv:+7.2%}  =  {imp:+8,.0f}M")
inc = tot[cur] - tot[prior]
print(f"\n  TOTAL estimated FX translation tailwind  {tot_fx:+,.0f}M")
print(f"  Q3FY26 total revenue increment           {inc:+,.0f}M")
print(f"  >>> FX is roughly {tot_fx/inc:.0%} of the consolidated increment, i.e. about "
      f"{tot_fx/tot[prior]:+.1f}pp of the +{tot[cur]/tot[prior]-1:.1%} headline growth")
print("  LABEL: INFERENCE (segment revenue is FACT from the 10-Q; the currency baskets and weights")
print("         are my assumption because Apple discloses no currency-of-denomination breakdown).")
print("         Apple's DIRECTIONAL statement that FX was 'net favorable' is FACT (10-Q, ...26-000020).")
