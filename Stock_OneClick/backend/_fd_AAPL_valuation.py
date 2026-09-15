"""_fd_AAPL_valuation.py -- valuation multiples on the CORRECTED TTM basis, plus the quarterly
growth shape needed to test whether the +16.4% latest-quarter acceleration is structural or a base
effect.

All inputs are XBRL facts or values derived from them by the Q4 = annual - (Q1+Q2+Q3) identity that
_fd_AAPL_ttm_multiples.py established. Price is the 2026-09-14 close from the technical layer.
"""
import numpy as np
import pandas as pd
import _fund_data as F

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 400)
SYM, PRICE = "AAPL", 333.08
panel = F.load()
D = lambda s: pd.Timestamp(s).date()

FY = {
    2026: ["2025-12-27", "2026-03-28", "2026-06-27", None],
    2025: ["2024-12-28", "2025-03-29", "2025-06-28", "2025-09-27"],
    2024: ["2023-12-30", "2024-03-30", "2024-06-29", "2024-09-28"],
    2023: ["2022-12-31", "2023-04-01", "2023-07-01", "2023-09-30"],
    2022: ["2021-12-25", "2022-03-26", "2022-06-25", "2022-09-24"],
    2021: ["2020-12-26", "2021-03-27", "2021-06-26", "2021-09-25"],
    2020: ["2019-12-28", "2020-03-28", "2020-06-27", "2020-09-26"],
    2019: ["2018-12-29", "2019-03-30", "2019-06-29", "2019-09-28"],
}


def build_quarters(concept):
    qs = F.series(panel, SYM, concept, annual=False)
    ann = F.series(panel, SYM, concept, annual=True)
    q = qs.set_index(qs["end"].dt.date)["val"].astype(float).to_dict() if not qs.empty else {}
    a = ann.set_index(ann["end"].dt.date)["val"].astype(float) if not ann.empty else pd.Series(dtype=float)
    for fy, ends in FY.items():
        q4 = ends[3]
        if q4 is None or D(q4) in q or D(q4) not in a.index:
            continue
        f3 = [q.get(D(e)) for e in ends[:3]]
        if any(x is None or not np.isfinite(x) for x in f3):
            continue
        q[D(q4)] = float(a.loc[D(q4)]) - float(sum(f3))
    return pd.Series(dict(sorted(q.items())))


rev, gp, oi, ni, eps = (build_quarters(c) for c in
                        ["revenue", "gross_profit", "operating_income", "net_income", "eps_diluted"])

# ---------------------------------------------------------------- quarterly growth shape
print("=" * 130)
print("QUARTERLY REVENUE, YoY, AND 2-YEAR STACKED CAGR  (2yr CAGR strips a soft/hard base effect)")
print("=" * 130)
rows = []
for d in rev.index:
    yoy = prev = two = np.nan
    cand = [x for x in rev.index if 340 <= (d - x).days <= 390]
    if cand:
        prev = rev[cand[-1]]
        yoy = rev[d] / prev - 1
    cand2 = [x for x in rev.index if 705 <= (d - x).days <= 760]
    if cand2:
        two = (rev[d] / rev[cand2[-1]]) ** 0.5 - 1
    rows.append({"q_end": d, "rev_$M": rev[d] / 1e6, "yoy_%": yoy * 100 if np.isfinite(yoy) else np.nan,
                 "yr_ago_$M": prev / 1e6 if np.isfinite(prev) else np.nan,
                 "2yr_cagr_%": two * 100 if np.isfinite(two) else np.nan,
                 "gm_%": gp.get(d, np.nan) / rev[d] * 100, "om_%": oi.get(d, np.nan) / rev[d] * 100,
                 "nm_%": ni.get(d, np.nan) / rev[d] * 100})
g = pd.DataFrame(rows)
print(g.tail(17).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

# ---------------------------------------------------------------- TTM aggregates
TTM = ["2025-09-27", "2025-12-27", "2026-03-28", "2026-06-27"]
PRIOR = ["2024-09-28", "2024-12-28", "2025-03-29", "2025-06-28"]
S = lambda s, ends: float(sum(s[D(e)] for e in ends))
rev_ttm, gp_ttm, oi_ttm, ni_ttm, eps_ttm = (S(x, TTM) for x in [rev, gp, oi, ni, eps])
da_ttm = 11_698e6 - 8_571e6 + 9_973e6          # FY25 - 9M FY25 + 9M FY26, all us-gaap:DDA facts
ebitda_ttm = oi_ttm + da_ttm

# ---------------------------------------------------------------- balance sheet & shares
def inst(concept):
    s = F.series(panel, SYM, concept, annual=False, instant=True) if False else None
    d = panel[(panel.symbol == SYM) & (panel.concept == concept)]
    d = d[d["start"].isna()] if d["start"].isna().any() else d
    if d.empty:
        return np.nan, None
    d = d.sort_values(["end", "filed"])
    return float(d["val"].iloc[-1]), d["end"].iloc[-1].date()


parts = {}
for c in ["cash", "short_term_inv", "long_term_inv", "debt_term_total", "debt_term_nc",
          "debt_term_c", "debt_short"]:
    parts[c] = inst(c)
print("\n" + "=" * 130)
print("BALANCE SHEET COMPONENTS (latest instant fact per concept)")
print("=" * 130)
for k, (v, d) in parts.items():
    print(f"  {k:16s} {v/1e6:>12,.0f} $M   as of {d}")
cash_inv = sum(parts[c][0] for c in ["cash", "short_term_inv", "long_term_inv"] if np.isfinite(parts[c][0]))
debt = parts["debt_term_total"][0] + (parts["debt_short"][0] if np.isfinite(parts["debt_short"][0]) else 0)
net_cash = cash_inv - debt
print(f"\n  cash+investments {cash_inv/1e6:>12,.0f} $M")
print(f"  total debt       {debt/1e6:>12,.0f} $M   (LongTermDebt total + CommercialPaper; NOT summing nc+c+total)")
print(f"  NET CASH         {net_cash/1e6:>12,.0f} $M")

SHARES = {"cover_page_2026-07-17 (dei, actual outstanding)": 14_594_180_000,
          "diluted_wtd_avg_Q3FY26": 14_714_676_000,
          "basic_wtd_avg_Q3FY26 (what fund_metrics used)": 14_656_110_000}

print("\n" + "=" * 130)
print(f"VALUATION MULTIPLES AT ${PRICE:.2f}  (2026-09-14 close)   ALL GAAP, TTM ending 2026-06-27")
print("=" * 130)
print(f"  TTM revenue          {rev_ttm/1e6:>12,.0f} $M   (fund_metrics used FY2025 416,161 -> "
      f"{(416_161e6/rev_ttm-1)*100:+.1f}% stale)")
print(f"  TTM gross profit     {gp_ttm/1e6:>12,.0f} $M   GM {gp_ttm/rev_ttm*100:.2f}%")
print(f"  TTM operating income {oi_ttm/1e6:>12,.0f} $M   OM {oi_ttm/rev_ttm*100:.2f}%")
print(f"  TTM D&A              {da_ttm/1e6:>12,.0f} $M")
print(f"  TTM EBITDA           {ebitda_ttm/1e6:>12,.0f} $M   margin {ebitda_ttm/rev_ttm*100:.2f}%")
print(f"  TTM net income       {ni_ttm/1e6:>12,.0f} $M   NM {ni_ttm/rev_ttm*100:.2f}%")
print(f"  TTM diluted EPS      {eps_ttm:>12,.2f}      (fund_metrics used gapped 8.44)")

out = []
for label, sh in SHARES.items():
    mcap = PRICE * sh
    ev = mcap - net_cash
    out.append({"share_basis": label, "shares_M": sh / 1e6, "mcap_$B": mcap / 1e9, "EV_$B": ev / 1e9,
                "P/E_ttm": mcap / ni_ttm, "P/S": mcap / rev_ttm, "EV/S": ev / rev_ttm,
                "EV/GP": ev / gp_ttm, "EV/EBITDA": ev / ebitda_ttm, "EV/EBIT": ev / oi_ttm})
print()
print(pd.DataFrame(out).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

# ---------------------------------------------------------------- what fund_metrics reported
print("\n" + "=" * 130)
print("RECONCILIATION TO fund_metrics.py's PRINTED FIGURES")
print("=" * 130)
mcap_fm = PRICE * 14_656_110_000
print(f"  fund_metrics EV/Sales 11.6x = EV {(mcap_fm-net_cash)/1e9:,.0f}B / FY2025 rev 416,161M "
      f"= {(mcap_fm-net_cash)/416_161e6:.2f}x")
print(f"  CORRECTED    EV/Sales       = same EV / TTM rev {rev_ttm/1e6:,.0f}M "
      f"= {(mcap_fm-net_cash)/rev_ttm:.2f}x   -> reported figure is "
      f"{((mcap_fm-net_cash)/416_161e6)/((mcap_fm-net_cash)/rev_ttm)-1:+.1%} too high")
print(f"  fund_metrics PE(GAAP) 39.5x = {PRICE:.2f} / gapped EPS 8.44 = {PRICE/8.44:.2f}x")
print(f"  CORRECTED    PE(GAAP)       = {PRICE:.2f} / true TTM EPS {eps_ttm:.2f} = {PRICE/eps_ttm:.2f}x")
print(f"  CORRECTED    PE on mcap/NI  = {mcap_fm/ni_ttm:.2f}x")
