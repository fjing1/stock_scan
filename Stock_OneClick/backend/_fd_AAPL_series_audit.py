"""_fd_AAPL_series_audit.py -- dump AAPL's point-in-time XBRL series with dates + accessions.

Purpose: fund_metrics.py reported revenue_ttm == revenue_fy (416,161) while the balance sheet is
dated 2026-06-27, i.e. three quarters later. That is the missing-Q4 guard firing. This script
establishes exactly which quarterly periods exist so a CORRECT trailing-twelve-month revenue and
EPS can be built, and so the multiples can be recomputed on a consistent as-of date.
"""
import pandas as pd
import _fund_data as F

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 300)

panel = F.load()
SYM = "AAPL"


def dump(concept, annual, label):
    s = F.series(panel, SYM, concept, annual=annual)
    if s.empty:
        print(f"\n=== {label}: EMPTY")
        return s
    s = s.copy()
    s["days"] = (s["end"] - s["start"]).dt.days if "start" in s and s["start"].notna().any() else None
    cols = [c for c in ["start", "end", "days", "val", "tag", "form", "fp", "fy", "filed", "accn"] if c in s.columns]
    print(f"\n=== {label}  (n={len(s)})")
    print(s[cols].tail(24).to_string(index=False))
    return s


rev_q = dump("revenue", False, "revenue QUARTERLY")
rev_a = dump("revenue", True, "revenue ANNUAL")
eps_q = dump("eps_diluted", False, "eps_diluted QUARTERLY")
eps_a = dump("eps_diluted", True, "eps_diluted ANNUAL")
ni_q = dump("net_income", False, "net_income QUARTERLY")
sh_q = dump("shares_diluted", False, "shares_diluted QUARTERLY")
gp_q = dump("gross_profit", False, "gross_profit QUARTERLY")
oi_q = dump("operating_income", False, "operating_income QUARTERLY")

# ---- the exact defect: what does the naive last-4 sum span?
for name, s in [("revenue", rev_q), ("eps_diluted", eps_q), ("net_income", ni_q),
                ("gross_profit", gp_q), ("operating_income", oi_q)]:
    if len(s) >= 4:
        span = (s["end"].iloc[-1] - s["end"].iloc[-4]).days
        print(f"\n[SPAN] {name:20s} last-4 end-to-end span = {span} days "
              f"({s['end'].iloc[-4].date()} -> {s['end'].iloc[-1].date()})  "
              f"naive_sum={s['val'].iloc[-4:].sum():,.2f}")
