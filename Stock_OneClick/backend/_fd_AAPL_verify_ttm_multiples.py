"""Adversarial verification of the claim:
   "corrected trailing multiples are 38.2x GAAP P/E and 10.4x EV/Sales, not 39.5x / 11.6x,
    because the layer's TTM fell back to FY2025 revenue and summed 4 NON-CONSECUTIVE quarterly EPS"

Step 1: dump the layer's own quarterly series (revenue, eps_diluted, shares_diluted) so the
        windows the layer actually sums are visible, not asserted.
Step 2: recompute the layer's revenue_ttm / eps_ttm / multiples by hand from the panel.
Step 3: independent EDGAR pull -- companyfacts annual FY2025 revenue + EPS, and the Q4FY25 8-K.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import _fund_data as F

pd.set_option("display.width", 240)
panel = F.load()

print("=" * 120)
print("STEP 1  layer's own quarterly series for AAPL (point-in-time panel, all filings)")
print("=" * 120)
for c in ("revenue", "eps_diluted", "shares_diluted", "net_income"):
    s = F.series(panel, "AAPL", c, annual=False)
    s = s.tail(9)[["end", "val", "tag", "form", "filed", "accn", "days"]].copy()

    print(f"\n--- quarterly {c}  (last 9 rows)")
    print(s.to_string(index=False))

print("\n" + "=" * 120)
print("STEP 2  reproduce the layer's arithmetic exactly")
print("=" * 120)
rev_q = F.series(panel, "AAPL", "revenue", annual=False)
rev_a = F.series(panel, "AAPL", "revenue", annual=True)
span = (rev_q.end.iloc[-1] - rev_q.end.iloc[-4]).days
print(f"  last-4 quarterly revenue window: {rev_q.end.iloc[-4].date()} .. {rev_q.end.iloc[-1].date()}"
      f"   span={span} days   -> layer's guard is 250<=span<=300  => {'4-row sum' if 250<=span<=300 else 'FALLBACK to revenue_fy'}")
print(f"  naive 4-row quarterly sum        {rev_q.val.iloc[-4:].sum():>14,.0f}")
print(f"  revenue_fy (FY2025)              {rev_a.val.iloc[-1]:>14,.0f}   filed {rev_a.filed.iloc[-1].date()} end {rev_a.end.iloc[-1].date()}")
print("  ^ CHECK: is the 4-row sum a *different* wrong number than the fallback? both are wrong vs true TTM")

eps_q = F.series(panel, "AAPL", "eps_diluted", annual=False)
w = eps_q.tail(4)
_win = ", ".join(str(d.date()) for d in w.end)
print(f"\n  layer's EPS window (NO span guard in code): {_win}")
print(f"  layer eps_ttm = {w.val.sum():.4f}   (spans {(w.end.iloc[-1]-w.end.iloc[0]).days} days"
      f" = {(w.end.iloc[-1]-w.end.iloc[0]).days/365:.2f} years of *end dates*, i.e. 5 fiscal quarters)")
eps_a = F.series(panel, "AAPL", "eps_diluted", annual=True)
print(f"  annual eps FY2025 = {eps_a.val.iloc[-1]:.2f}  (end {eps_a.end.iloc[-1].date()})")

sh_q = F.series(panel, "AAPL", "shares_diluted", annual=False)
print(f"\n  layer shares (latest quarterly {sh_q.end.iloc[-1].date()}, tag {sh_q.tag.iloc[-1]}) "
      f"= {sh_q.val.iloc[-1]:,.0f}")
print(f"  ^ compare to Q3FY26 filed diluted weighted-average 14,714,676 thousand = 14,714,676,000")

PRICE = 333.08
NET_CASH = 62220.0e6 if False else 62220.0   # $M
for label, sh in (("layer shares 14,656M", sh_q.val.iloc[-1] / 1e6),
                  ("claim shares 14,714.7M", 14714.676)):
    mcap = PRICE * sh
    ev = mcap - NET_CASH
    print(f"\n  [{label}] mcap ${mcap/1e6:.4f}T  EV ${ev/1e6:.4f}T")
    print(f"      EV/FY2025rev(416,161) = {ev/416161:.3f}x    EV/trueTTM(466,823) = {ev/466823:.3f}x")

print("\n" + "=" * 120)
print("STEP 3  independent EDGAR: companyfacts FY2025 annual facts + does a Q4-FY25 3-month fact exist?")
print("=" * 120)
import json, gzip, os, requests
UA = {"User-Agent": "stock_scan research fj.research.contact@gmail.com",
      "Accept-Encoding": "gzip, deflate"}
p = "_fd_AAPL_verify_companyfacts.json"
if not os.path.exists(p):
    r = requests.get("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
                     headers=UA, timeout=90)
    print("  companyfacts HTTP", r.status_code, len(r.content), "bytes")
    r.raise_for_status()
    open(p, "wb").write(r.content)
CF = json.load(open(p))

def facts(tag, unit="USD"):
    d = CF["facts"]["us-gaap"][tag]["units"][unit]
    f = pd.DataFrame(d)
    for c in ("start", "end", "filed"):
        if c in f:
            f[c] = pd.to_datetime(f[c])
    return f

rev = facts("RevenueFromContractWithCustomerExcludingAssessedTax")
rev = rev[rev.start.notna()].copy()
rev["days"] = (rev.end - rev.start).dt.days
ann = rev[(rev.days > 330) & (rev.days < 400)].sort_values("end")
print("\n  ANNUAL revenue facts (last 4, form/accn shown):")
print(ann.drop_duplicates("end", keep="last")[["start", "end", "val", "form", "accn", "filed"]]
      .tail(4).to_string(index=False))
qtr = rev[(rev.days > 60) & (rev.days < 110)].sort_values("end").drop_duplicates("end", keep="last")
print("\n  QUARTERLY (3-month) revenue facts, last 8 -- look for a 2025-09-27 row:")
print(qtr[["start", "end", "val", "form", "accn", "filed"]].tail(8).to_string(index=False))
print(f"\n  Does companyfacts contain ANY 3-month revenue fact ending 2025-09-27? "
      f"{'YES' if (qtr.end == pd.Timestamp('2025-09-27')).any() else 'NO -> Q4 gap is real'}")

eps = facts("EarningsPerShareDiluted", unit="USD/shares")
eps["days"] = (eps.end - eps.start).dt.days
epsq = eps[(eps.days > 60) & (eps.days < 110)].sort_values("end").drop_duplicates("end", keep="last")
print("\n  QUARTERLY diluted EPS facts, last 8:")
print(epsq[["start", "end", "val", "form", "accn"]].tail(8).to_string(index=False))
epsa = eps[(eps.days > 330) & (eps.days < 400)].sort_values("end").drop_duplicates("end", keep="last")
print("\n  ANNUAL diluted EPS facts, last 3:")
print(epsa[["start", "end", "val", "form", "accn"]].tail(3).to_string(index=False))

# derive Q4FY25 from annual minus the three filed quarters -- fully independent of the 8-K parse
fy25 = float(ann[ann.end == pd.Timestamp("2025-09-27")].val.iloc[-1])
q123 = qtr[qtr.end.isin([pd.Timestamp("2024-12-28"), pd.Timestamp("2025-03-29"),
                         pd.Timestamp("2025-06-28")])]
print(f"\n  FY2025 revenue from companyfacts   {fy25:,.0f}")
print(q123[["end", "val", "accn"]].to_string(index=False))
print(f"  9M FY2025 sum                      {q123.val.sum():,.0f}")
print(f"  => DERIVED Q4FY25 revenue          {fy25-q123.val.sum():,.0f}   "
      f"(claim's 8-K parse said 102,466 $M)")

# and the true TTM straight from companyfacts + the derived Q4
q26 = qtr[qtr.end.isin([pd.Timestamp("2025-12-27"), pd.Timestamp("2026-03-28"),
                        pd.Timestamp("2026-06-27")])]
print(q26[["end", "val", "accn"]].to_string(index=False))
ttm = (fy25 - q123.val.sum()) + q26.val.sum()
print(f"  => TRUE TTM revenue (derived Q4FY25 + 3 filed FY26 quarters) {ttm:,.0f}  "
      f"vs claim 466,823,000,000 ; vs FY2025 {ttm/fy25-1:+.2%}")

# prior-year TTM, to test whether the multiple-on-TTM story is growth-consistent
fy24 = float(ann[ann.end == pd.Timestamp("2024-09-28")].val.iloc[-1])
q123_25 = qtr[qtr.end.isin([pd.Timestamp("2023-12-30"), pd.Timestamp("2024-03-30"),
                            pd.Timestamp("2024-06-29")])]
q4fy24 = fy24 - q123_25.val.sum()
prior_ttm = q4fy24 + q123.val.sum()
print(f"\n  FY2024 revenue {fy24:,.0f} ; derived Q4FY24 {q4fy24:,.0f}")
print(f"  prior TTM (Q4FY24..Q3FY25) {prior_ttm:,.0f}  ->  TTM YoY {ttm/prior_ttm-1:+.2%}")

# TTM net income / EPS independent derivation
ni = facts("NetIncomeLoss")
ni = ni[ni.start.notna()].copy(); ni["days"] = (ni.end - ni.start).dt.days
nia = ni[(ni.days > 330) & (ni.days < 400)].sort_values("end").drop_duplicates("end", keep="last")
niq = ni[(ni.days > 60) & (ni.days < 110)].sort_values("end").drop_duplicates("end", keep="last")
ni_fy25 = float(nia[nia.end == pd.Timestamp("2025-09-27")].val.iloc[-1])
ni_9m = niq[niq.end.isin([pd.Timestamp("2024-12-28"), pd.Timestamp("2025-03-29"),
                          pd.Timestamp("2025-06-28")])].val.sum()
ni_26 = niq[niq.end.isin([pd.Timestamp("2025-12-27"), pd.Timestamp("2026-03-28"),
                          pd.Timestamp("2026-06-27")])].val.sum()
print(f"\n  FY2025 net income {ni_fy25:,.0f} ; 9M {ni_9m:,.0f} ; derived Q4FY25 {ni_fy25-ni_9m:,.0f}"
      f"  (claim's parse said 27,466 $M)")
print(f"  TRUE TTM net income {(ni_fy25-ni_9m)+ni_26:,.0f}   (claim 128,930 $M)")

eps_fy25 = float(epsa[epsa.end == pd.Timestamp("2025-09-27")].val.iloc[-1])
eps_9m25 = epsq[epsq.end.isin([pd.Timestamp("2024-12-28"), pd.Timestamp("2025-03-29"),
                               pd.Timestamp("2025-06-28")])].val.sum()
eps_26 = epsq[epsq.end.isin([pd.Timestamp("2025-12-27"), pd.Timestamp("2026-03-28"),
                             pd.Timestamp("2026-06-27")])].val.sum()
print(f"\n  FY2025 diluted EPS {eps_fy25:.2f} ; sum of 3 filed FY25 quarters {eps_9m25:.2f} ;"
      f" implied Q4FY25 {eps_fy25-eps_9m25:.2f}  (claim's 8-K parse said 1.85)")
print(f"  TRUE TTM EPS (implied Q4 + 3 FY26 quarters) {(eps_fy25-eps_9m25)+eps_26:.4f}   (claim 8.72)")
print(f"  P/E on that {PRICE/((eps_fy25-eps_9m25)+eps_26):.2f}x ; P/E on claim 8.72 {PRICE/8.72:.2f}x")
