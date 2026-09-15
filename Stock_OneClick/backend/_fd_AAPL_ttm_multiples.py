"""_fd_AAPL_ttm_multiples.py -- rebuild a CORRECT trailing-twelve-month basis for AAPL and recompute
every valuation multiple on it.

WHY THIS SCRIPT EXISTS. fund_metrics.py's quarterly series for AAPL has a structural hole: Apple's
10-K carries only the FULL YEAR as a duration fact, so the Q4 quarter (fiscal Sep quarter) never
appears as a ~90-day duration. The last four QUARTERLY ROWS are therefore:
    2025-06-28 (Q3 FY25) | 2025-12-27 (Q1 FY26) | 2026-03-28 (Q2 FY26) | 2026-06-27 (Q3 FY26)
i.e. a window with Q4 FY25 (Jun-Sep 2025) MISSING and Q3 FY25 substituted in its place.
  * revenue_ttm has a 250-300 day span guard, which correctly fires (span=364) and falls back to
    revenue_fy = FY2025 = 416,161 -- a denominator that is THREE QUARTERS STALE.
  * eps_ttm_gaap / pe_gaap have NO such guard, so P/E is computed on the gapped 8.44 sum.
Both are fixed here by DERIVING the missing Q4 as (annual - Q1 - Q2 - Q3) for the same fiscal year.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import _fund_data as F

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 400)
HERE = Path(__file__).resolve().parent
SYM = "AAPL"
PRICE = 333.08          # 2026-09-14 close, from the pre-measured technical layer
panel = F.load()


def q_series(concept):
    s = F.series(panel, SYM, concept, annual=False)
    return s.set_index(s["end"].dt.date)["val"].astype(float) if not s.empty else pd.Series(dtype=float)


def a_series(concept):
    s = F.series(panel, SYM, concept, annual=True)
    return s.set_index(s["end"].dt.date)["val"].astype(float) if not s.empty else pd.Series(dtype=float)


# Apple fiscal-year quarter end dates, newest years. Q4 end == fiscal year end.
FY = {
    2026: ["2025-12-27", "2026-03-28", "2026-06-27", None],          # Q4 not yet reported
    2025: ["2024-12-28", "2025-03-29", "2025-06-28", "2025-09-27"],
    2024: ["2023-12-30", "2024-03-30", "2024-06-29", "2024-09-28"],
    2023: ["2022-12-31", "2023-04-01", "2023-07-01", "2023-09-30"],
    2022: ["2021-12-25", "2022-03-26", "2022-06-25", "2022-09-24"],
    2021: ["2020-12-26", "2021-03-27", "2021-06-26", "2021-09-25"],
    2020: ["2019-12-28", "2020-03-28", "2020-06-27", "2020-09-26"],
}
D = lambda s: pd.Timestamp(s).date()


def build_quarters(concept):
    """Full quarterly series with Q4 DERIVED as annual - (Q1+Q2+Q3). Returns dict[date] = value."""
    q, a = q_series(concept), a_series(concept)
    out, derived = {}, []
    for d, v in q.items():
        out[d] = v
    for fy, ends in FY.items():
        q4 = ends[3]
        if q4 is None:
            continue
        if D(q4) in out:                      # Q4 already present as a real quarterly fact
            continue
        if D(q4) not in a.index:
            continue
        first3 = [out.get(D(e)) for e in ends[:3]]
        if any(x is None or not np.isfinite(x) for x in first3):
            continue
        out[D(q4)] = float(a.loc[D(q4)]) - float(sum(first3))
        derived.append((fy, q4, out[D(q4)]))
    return dict(sorted(out.items())), derived


CONCEPTS = ["revenue", "gross_profit", "operating_income", "net_income", "eps_diluted", "ocf",
            "capex", "sbc"]
Q, DERIV = {}, {}
for c in CONCEPTS:
    Q[c], DERIV[c] = build_quarters(c)

print("=" * 118)
print("DERIVED Q4 VALUES  (annual minus Q1+Q2+Q3 of the same fiscal year -- these are the periods")
print("that are absent from XBRL as ~90-day duration facts)")
print("=" * 118)
for c in CONCEPTS:
    for fy, end, val in DERIV[c]:
        unit = "" if c == "eps_diluted" else " $M"
        scale = 1.0 if c == "eps_diluted" else 1e6
        print(f"  {c:18s} FY{fy} Q4 ending {end}: {val/scale:>14,.2f}{unit}")

# ---------------------------------------------------------------- TTM windows
TTM_END = D("2026-06-27")
TTM_Q = ["2025-09-27", "2025-12-27", "2026-03-28", "2026-06-27"]      # true 4 quarters
PRIOR_TTM_Q = ["2024-09-28", "2024-12-28", "2025-03-29", "2025-06-28"]
FY25_Q = ["2024-12-28", "2025-03-29", "2025-06-28", "2025-09-27"]


def wsum(concept, ends):
    vals = [Q[concept].get(D(e)) for e in ends]
    if any(v is None or not np.isfinite(v) for v in vals):
        return np.nan
    return float(sum(vals))


print("\n" + "=" * 118)
print(f"TRUE TTM  (4 quarters ending {TTM_END}) vs the GAPPED last-4-rows sum fund_metrics uses")
print("=" * 118)
gapped_ends = ["2025-06-28", "2025-12-27", "2026-03-28", "2026-06-27"]
rows = []
for c in CONCEPTS:
    true_ttm, gapped, prior = wsum(c, TTM_Q), wsum(c, gapped_ends), wsum(c, PRIOR_TTM_Q)
    scale = 1.0 if c == "eps_diluted" else 1e6
    rows.append({
        "concept": c,
        "TTM_true": true_ttm / scale,
        "gapped_sum": gapped / scale,
        "err_pct": (gapped / true_ttm - 1) * 100 if np.isfinite(true_ttm) and true_ttm else np.nan,
        "TTM_prior_yr": prior / scale,
        "TTM_yoy_pct": (true_ttm / prior - 1) * 100 if np.isfinite(prior) and prior else np.nan,
        "FY2025": wsum(c, FY25_Q) / scale,
    })
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

# ---------------------------------------------------------------- share count & net cash
cik = F.ticker_to_cik()[SYM]
cf = F.fetch_companyfacts(cik)


def raw_units(taxo, tag):
    try:
        u = cf["facts"][taxo][tag]["units"]
        return pd.DataFrame(list(u.values())[0])
    except Exception:
        return pd.DataFrame()


dei_sh = raw_units("dei", "EntityCommonStockSharesOutstanding")
if not dei_sh.empty:
    dei_sh = dei_sh.sort_values("end")
    print("\n=== dei:EntityCommonStockSharesOutstanding (ACTUAL count on the filing cover page)")
    print(dei_sh[["end", "val", "form", "filed", "accn"]].tail(6).to_string(index=False))

dil = raw_units("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding")
if not dil.empty:
    dil = dil[dil.get("form").isin(["10-Q", "10-K"])].sort_values(["end", "filed"])
    print("\n=== us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding (does this tag exist?)")
    print(dil[["start", "end", "val", "form", "filed", "accn"]].tail(8).to_string(index=False))
else:
    print("\n=== us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding : ABSENT from companyfacts")

for tag in ["DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet",
            "DepreciationAndAmortization"]:
    d = raw_units("us-gaap", tag)
    if not d.empty:
        d = d.sort_values("end")
        print(f"\n=== us-gaap:{tag}  (for EBITDA)")
        print(d[["start", "end", "val", "form", "filed", "accn"]].tail(8).to_string(index=False))
