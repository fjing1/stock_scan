"""ADVERSARIAL VERIFY #1 -- rebuild TTM revenue / EPS / net cash from a FRESH SEC companyfacts pull,
independent of the repo's _fund_data cache. Every printed number carries its accession number.
"""
from __future__ import annotations
import json, itertools
import pandas as pd

CF = json.load(open("_fd_AAPL_verify_cf.json"))
US = CF["facts"]["us-gaap"]
DEI = CF["facts"].get("dei", {})


def facts(concept, unit=None):
    if concept not in US:
        return pd.DataFrame()
    rows = []
    for u, arr in US[concept]["units"].items():
        if unit and u != unit:
            continue
        for f in arr:
            rows.append({"concept": concept, "unit": u, "start": f.get("start"), "end": f["end"],
                         "val": f["val"], "fy": f.get("fy"), "fp": f.get("fp"),
                         "form": f.get("form"), "filed": f.get("filed"), "accn": f.get("accn"),
                         "frame": f.get("frame")})
    d = pd.DataFrame(rows)
    for c in ("start", "end", "filed"):
        if c in d:
            d[c] = pd.to_datetime(d[c])
    return d


def dur(concept, lo, hi, form=None):
    d = facts(concept)
    if d.empty:
        return d
    d = d[d.start.notna()].copy()
    d["days"] = (d.end - d.start).dt.days
    d = d[(d.days >= lo) & (d.days <= hi)]
    if form:
        d = d[d.form.isin(form)]
    return d.sort_values(["end", "filed"])


print("=" * 118)
print("A. REVENUE -- every duration fact, tag by tag, so tag migration is visible")
print("=" * 118)
for tag in ("RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax", "Revenues", "SalesRevenueNet"):
    d = facts(tag)
    if d.empty:
        print(f"  {tag:<60} ABSENT")
        continue
    dd = d[d.start.notna()]
    print(f"  {tag:<60} n={len(dd):<5} {dd.end.min().date()} -> {dd.end.max().date()}")

REV = "RevenueFromContractWithCustomerExcludingAssessedTax"

# quarterly (80-100d) and annual (350-380d) and nine-month (260-285d) consolidated totals
qrev = dur(REV, 80, 100)
qrev = qrev[qrev.frame.notna() | True]
arev = dur(REV, 350, 380)
n9 = dur(REV, 255, 290)

print("\n  ANNUAL (350-380 day) revenue facts, deduped on period end, earliest filing kept:")
a = arev.drop_duplicates("end", keep="first")[["end", "val", "form", "filed", "accn"]]
print(a.tail(6).to_string(index=False))

print("\n  NINE-MONTH (255-290 day) revenue facts -- these are what the Q3 10-Q reports:")
n = n9.drop_duplicates(["end", "accn"])[["start", "end", "val", "form", "filed", "accn"]]
print(n.tail(8).to_string(index=False))

print("\n  QUARTERLY (80-100 day) revenue facts, deduped on end, earliest filing:")
q = qrev.drop_duplicates("end", keep="first")[["start", "end", "val", "form", "filed", "accn"]]
print(q.tail(14).to_string(index=False))
print(f"\n  >>> gap between last quarterly row and the 4th-back one: "
      f"{(q.end.iloc[-1] - q.end.iloc[-4]).days} days  "
      f"(fund_metrics guard is 250<=span<=300 -> FAILS -> falls back to FY revenue)")

print("\n" + "=" * 118)
print("B. TTM REVENUE -- three independent constructions")
print("=" * 118)
fy25 = float(a[a.end == pd.Timestamp("2025-09-27")].val.iloc[0])
fy24 = float(a[a.end == pd.Timestamp("2024-09-28")].val.iloc[0])
# nine-month figures as filed in the FY26 Q3 10-Q (current + comparative)
q3fy26 = n9[n9.end == n9.end.max()]
print("  nine-month rows at the latest end date:")
print(q3fy26[["start", "end", "val", "form", "accn"]].to_string(index=False))
m9_26 = float(q3fy26.val.iloc[0])
prior9 = n9[n9.end == pd.Timestamp("2025-06-28")]
m9_25 = float(prior9.val.iloc[0])
print(f"\n  FY2025 full year   {fy25:>12,.0f}")
print(f"  9M FY2025          {m9_25:>12,.0f}   (comparative column, same 10-Q)")
print(f"  => derived Q4 FY25 {fy25 - m9_25:>12,.0f}")
print(f"  9M FY2026          {m9_26:>12,.0f}")
ttm_A = m9_26 + (fy25 - m9_25)
print(f"  TTM (A) = 9M26 + derived Q4FY25 = {ttm_A:>12,.0f}")

# construction B: sum the four individual quarters (Q4FY25 derived from 10-K minus 9M)
qs = q.set_index("end").val
q1_26 = float(qs[pd.Timestamp("2025-12-27")]); q2_26 = float(qs[pd.Timestamp("2026-03-28")])
q3_26 = float(qs[pd.Timestamp("2026-06-27")])
q4_25 = fy25 - m9_25
ttm_B = q4_25 + q1_26 + q2_26 + q3_26
print(f"  TTM (B) = Q4FY25 {q4_25:,.0f} + Q1 {q1_26:,.0f} + Q2 {q2_26:,.0f} + Q3 {q3_26:,.0f} = {ttm_B:,.0f}")

# prior-year TTM for the growth rate: Q4FY24 + Q1..Q3 FY25
m9_24 = float(n9[n9.end == pd.Timestamp("2024-06-29")].val.iloc[0])
q4_24 = fy24 - m9_24
ttm_prior = q4_24 + m9_25
print(f"  prior-year TTM = Q4FY24 {q4_24:,.0f} + 9M FY25 {m9_25:,.0f} = {ttm_prior:,.0f}")
print(f"  >>> TTM revenue growth = {ttm_A / ttm_prior - 1:+.2%}    (claim says +14.2%)")
print(f"  >>> FY2025 growth      = {fy25 / fy24 - 1:+.2%}    (brief says +6.4%)")

print("\n" + "=" * 118)
print("C. DILUTED EPS -- XBRL, with Q4 derived, plus the four-quarter TTM")
print("=" * 118)
eq = dur("EarningsPerShareDiluted", 80, 100).drop_duplicates("end", keep="first")
ea = dur("EarningsPerShareDiluted", 350, 380).drop_duplicates("end", keep="first")
e9 = dur("EarningsPerShareDiluted", 255, 290).drop_duplicates(["end", "accn"])
print(eq[["end", "val", "form", "filed", "accn"]].tail(10).to_string(index=False))
print("\n  annual:")
print(ea[["end", "val", "form", "filed", "accn"]].tail(4).to_string(index=False))
print("\n  nine-month:")
print(e9[["start", "end", "val", "form", "accn"]].tail(6).to_string(index=False))
fy25e = float(ea[ea.end == pd.Timestamp("2025-09-27")].val.iloc[0])
m9_25e = float(e9[e9.end == pd.Timestamp("2025-06-28")].val.iloc[0])
q4_25e = fy25e - m9_25e
es = eq.set_index("end").val
ttm_eps = q4_25e + float(es[pd.Timestamp("2025-12-27")]) + float(es[pd.Timestamp("2026-03-28")]) + float(es[pd.Timestamp("2026-06-27")])
print(f"\n  FY25 diluted EPS {fy25e:.2f} - 9M FY25 {m9_25e:.2f} = derived Q4FY25 {q4_25e:.2f}")
print(f"  TTM diluted EPS (derived Q4 + Q1..Q3 FY26) = {ttm_eps:.2f}   (claim: 8.71 XBRL / 8.72 press)")
naive = float(eq.val.iloc[-4:].sum())
print(f"  what fund_metrics sums (last 4 quarterly ROWS, skipping Sept-25) = {naive:.2f}"
      f"  -> P/E 333.08/{naive:.2f} = {333.08/naive:.1f}x")
print(f"  correct TTM -> P/E 333.08/{ttm_eps:.2f} = {333.08/ttm_eps:.1f}x   (claim: 38.2x)")
# also compute net-income-based EPS as a third check
ni_a = dur("NetIncomeLoss", 350, 380).drop_duplicates("end", keep="first").set_index("end").val
ni_q = dur("NetIncomeLoss", 80, 100).drop_duplicates("end", keep="first").set_index("end").val
ni_9 = dur("NetIncomeLoss", 255, 290).drop_duplicates("end", keep="first").set_index("end").val
ni_q4_25 = float(ni_a[pd.Timestamp("2025-09-27")]) - float(ni_9[pd.Timestamp("2025-06-28")])
ttm_ni = ni_q4_25 + float(ni_q[pd.Timestamp("2025-12-27")]) + float(ni_q[pd.Timestamp("2026-03-28")]) + float(ni_q[pd.Timestamp("2026-06-27")])
print(f"\n  cross-check via net income: TTM NI = {ttm_ni:,.0f}  (Q4FY25 derived {ni_q4_25:,.0f})")
wa = dur("WeightedAverageNumberOfDilutedSharesOutstanding", 80, 100).drop_duplicates("end", keep="first")
print(wa[["end", "val", "accn"]].tail(5).to_string(index=False))
sh = float(wa.val.iloc[-1])
print(f"  TTM NI / latest-quarter diluted shares = {ttm_ni/sh:.2f}  (approximation; EPS sum is the right figure)")

print("\n" + "=" * 118)
print("D. BALANCE SHEET at 2026-06-27 -- net cash, line by line, from XBRL instants")
print("=" * 118)
BS_DATE = pd.Timestamp("2026-06-27")
def inst(concept):
    d = facts(concept)
    if d.empty:
        return None
    d = d[d.start.isna() & (d.end == BS_DATE)]
    if d.empty:
        return None
    d = d.sort_values("filed")
    return float(d.val.iloc[-1]), d.accn.iloc[-1]

for c in ("CashAndCashEquivalentsAtCarryingValue", "MarketableSecuritiesCurrent",
          "MarketableSecuritiesNoncurrent", "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
          "OtherShortTermInvestments", "ShortTermInvestments", "LongTermInvestments",
          "CommercialPaper", "LongTermDebtCurrent", "LongTermDebtNoncurrent", "LongTermDebt",
          "OtherLiabilitiesCurrent", "DebtCurrent", "NotesPayableCurrent"):
    r = inst(c)
    print(f"  {c:<58} {'' if r is None else f'{r[0]:>13,.0f}   {r[1]}'}" if r else f"  {c:<58} ABSENT")

cash = inst("CashAndCashEquivalentsAtCarryingValue")[0]
mktc = inst("MarketableSecuritiesCurrent")[0]
mktn = inst("MarketableSecuritiesNoncurrent")[0]
cp = inst("CommercialPaper")[0]
ltdc = inst("LongTermDebtCurrent")[0]
ltdn = inst("LongTermDebtNoncurrent")[0]
tot_cash = cash + mktc + mktn
tot_debt = cp + ltdc + ltdn
print(f"\n  cash+securities = {cash:,.0f} + {mktc:,.0f} + {mktn:,.0f} = {tot_cash:,.0f}")
print(f"  debt            = CP {cp:,.0f} + LTD-cur {ltdc:,.0f} + LTD-noncur {ltdn:,.0f} = {tot_debt:,.0f}")
print(f"  NET CASH        = {tot_cash - tot_debt:,.0f}    (claim 62,173 / repo 62,220)")
ltd_total = inst("LongTermDebt")
if ltd_total:
    print(f"  NOTE LongTermDebt (fair-value note, not the BS line) = {ltd_total[0]:,.0f} -- using this "
          f"instead of the two BS lines changes net cash to {tot_cash - (cp + ltd_total[0]):,.0f}")

print("\n" + "=" * 118)
print("E. EV / SALES on the corrected TTM")
print("=" * 118)
P = 333.08
dei_sh = DEI.get("EntityCommonStockSharesOutstanding")
if dei_sh:
    rows = sorted(dei_sh["units"]["shares"], key=lambda f: f["end"])[-4:]
    for f in rows:
        print(f"  dei EntityCommonStockSharesOutstanding  as of {f['end']}  {f['val']:>16,}   {f['accn']}")
cso = facts("CommonStockSharesOutstanding")
net_cash = tot_cash - tot_debt
for shm, lab in ((sh, "weighted diluted, Q3FY26"), (14609.0, "claim's 'outstanding'"),):
    mcap = P * shm
    ev = mcap - net_cash
    print(f"  shares {shm:>10,.0f}M [{lab:<26}] mcap {mcap:>12,.0f}M  EV {ev:>12,.0f}M  "
          f"EV/TTM-sales {ev/ttm_A:>5.2f}x   EV/FY-sales {ev/fy25:>5.2f}x")
