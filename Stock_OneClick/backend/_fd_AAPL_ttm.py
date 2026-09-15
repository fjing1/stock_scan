"""Rebuild AAPL TTM revenue / EPS / margins correctly by DERIVING the missing fiscal-Q4
duration fact (annual minus the three reported quarters), then recompute EV/Sales and P/E.

Why: fund_metrics.revenue_ttm falls back to revenue_fy when the last-4-quarter span guard
(250..300d) fails, which it does for AAPL (span 364d). And pe_gaap sums the last 4 quarterly
EPS rows, which for AAPL are Jun-25, Dec-25, Mar-26, Jun-26 -- i.e. Sep-25 quarter is missing
and the Jun-25 quarter is counted in its place."""
import sys, pandas as pd, numpy as np
sys.path.insert(0, "/Users/feijing/github.com/stock_scan/Stock_OneClick/backend")
import _fund_data as fd
pd.set_option("display.width", 220)
panel = fd.load()

def q(c): return fd.series(panel, "AAPL", c, annual=False)
def a(c): return fd.series(panel, "AAPL", c, annual=True)

def build_full_q(concept):
    """Quarterly series with fiscal Q4 derived from the annual fact when it is absent."""
    qq = q(concept).copy(); aa = a(concept).copy()
    if qq.empty or aa.empty: return qq
    rows = qq.to_dict("records")
    have = set(pd.to_datetime(qq.end).dt.date)
    for _, yr in aa.iterrows():
        fy_end = yr.end
        # the three reported quarters that fall inside this fiscal year
        inside = qq[(qq.end > fy_end - pd.Timedelta(days=360)) & (qq.end <= fy_end)]
        if len(inside) == 3 and fy_end.date() not in have:
            rows.append({"end": fy_end, "val": float(yr.val) - float(inside.val.sum()),
                         "tag": "DERIVED=FY-minus-3Q", "form": yr.form, "filed": yr.filed,
                         "accn": yr.accn, "days": 91.0})
    out = pd.DataFrame(rows).sort_values("end").drop_duplicates("end", keep="first").reset_index(drop=True)
    return out

for c in ("revenue", "eps_diluted", "net_income", "gross_profit", "operating_income", "ocf", "sbc"):
    s = build_full_q(c)
    s = s.tail(13).copy()
    unit = 1e6 if c not in ("eps_diluted",) else 1
    s["v"] = s.val/unit
    print(f"\n=== {c} quarterly (Q4 derived where needed), last 13 ===")
    print(s[["end","v","tag","form","filed","accn"]].to_string(index=False))

# ---- TTM aggregates on the derived-complete series
def ttm(concept, k=4):
    s = build_full_q(concept)
    s = s.sort_values("end")
    last4 = s.tail(k)
    span = (last4.end.iloc[-1] - last4.end.iloc[0]).days
    return float(last4.val.sum()), span, list(last4.end.dt.date), last4

print("\n" + "="*90)
for c in ("revenue","net_income","eps_diluted","gross_profit","operating_income","ocf","sbc"):
    v, span, ends, _ = ttm(c)
    print(f"TTM {c:<17} = {v:>14,.2f}   span(first->last end)={span}d  quarters={ends}")

# prior-year TTM for growth
def ttm_prior(concept):
    s = build_full_q(concept).sort_values("end")
    return float(s.tail(8).head(4).val.sum()), list(s.tail(8).head(4).end.dt.date)

print()
for c in ("revenue","net_income","eps_diluted","gross_profit","operating_income","ocf"):
    cur,_,_,_ = ttm(c); prev, ends = ttm_prior(c)
    print(f"{c:<17} TTM {cur:>14,.2f} vs prior-year TTM {prev:>14,.2f}  = {cur/prev-1:+.1%}   prior quarters={ends}")
