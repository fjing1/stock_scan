"""_fd_AAPL_peers2.py -- peer comparison on the only basis that is genuinely identical across these
six filers, with the two contaminated cells excluded rather than quietly reported.

WHAT THE FIRST PASS GOT WRONG, and why it matters:
  * GOOGL TTM net income came out at 54.8% of revenue. That is REAL AS FILED -- the quarter ending
    2026-06-30 booked net income of $112,193M against operating income of $40,770M, i.e. ~$71.4B of
    NON-OPERATING income in a single quarter (accn 0001652044-26-000071). GOOGL's P/E is therefore
    not comparable to anything and is suppressed here; EV/EBIT is used instead.
  * AMZN has no GrossProfit tag after 2009, and its CostOfGoodsAndServicesSold excludes fulfilment,
    technology and marketing, so a "gross margin" of 0.7% is a tagging artefact, not economics.
    AMZN gross-margin and EV/GP cells are suppressed.
Growth is put on an identical footing by computing the LATEST QUARTER YoY for every name, deriving
any missing Q4 as (fiscal year - 9-month YTD).
"""
import numpy as np
import pandas as pd
import yfinance as yf

import _fund_data as F

pd.set_option("display.width", 260)
panel = F.load()
PEERS = ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "QCOM"]
SUPPRESS_PE = {"GOOGL"}                      # one-off non-operating gain
SUPPRESS_GM = {"AMZN"}                       # no comparable GrossProfit tagging


def dur(sym, concept):
    d = panel[(panel.symbol == sym) & (panel.concept == concept)].copy()
    d = d[d["start"].notna()]
    if d.empty:
        return d
    d["days"] = (d["end"] - d["start"]).dt.days
    return d.sort_values(["end", "days", "filed"]).drop_duplicates(["start", "end"], keep="last")


def annual(sym, concept):
    d = dur(sym, concept)
    return d[(d.days >= 340) & (d.days <= 400)].sort_values("end") if not d.empty else d


def ttm(sym, concept):
    a = annual(sym, concept)
    if a.empty:
        return np.nan
    fy_val, fy_end, fy_start = float(a.val.iloc[-1]), a.end.iloc[-1], a.start.iloc[-1]
    d = dur(sym, concept)
    ytd = d[(d.start > fy_end) & (d.days >= 60)].sort_values("days")
    if ytd.empty:
        return fy_val
    cur = ytd.iloc[-1]
    n = int(cur.days)
    pp = d[(d.start.between(fy_start - pd.Timedelta(days=7), fy_start + pd.Timedelta(days=7))) &
           (d.days.between(n - 12, n + 12))]
    return fy_val + float(cur.val) - float(pp.val.iloc[-1]) if not pp.empty else fy_val


def latest_q_yoy(sym, concept="revenue"):
    """Latest quarter and the same quarter a year earlier, deriving Q4 = FY - 9M YTD if needed."""
    d = dur(sym, concept)
    if d.empty:
        return np.nan, None, np.nan
    q = d[(d.days >= 60) & (d.days <= 120)].sort_values("end")
    a = annual(sym, concept)
    derived = {}
    for _, r in a.iterrows():
        pool = d[(d.start.between(r.start - pd.Timedelta(days=7), r.start + pd.Timedelta(days=7))) &
                 (d.days.between(250, 290))]
        if not pool.empty:
            derived[r.end] = float(r.val) - float(pool.val.iloc[-1])
    qmap = {r.end: float(r.val) for _, r in q.iterrows()}
    qmap.update({k: v for k, v in derived.items() if k not in qmap})
    ends = sorted(qmap)
    if not ends:
        return np.nan, None, np.nan
    last = ends[-1]
    prior = [e for e in ends if 340 <= (last - e).days <= 390]
    if not prior:
        return qmap[last], last.date(), np.nan
    return qmap[last], last.date(), qmap[last] / qmap[prior[-1]] - 1


def inst(sym, concept):
    d = panel[(panel.symbol == sym) & (panel.concept == concept)].copy()
    d = d[d["start"].isna()]
    return float(d.sort_values(["end", "filed"]).val.iloc[-1]) if not d.empty else np.nan


def shares(sym):
    d = dur(sym, "shares_diluted")
    d = d[(d.days >= 60) & (d.days <= 120)]
    return float(d.sort_values(["end", "filed"]).val.iloc[-1]) if not d.empty else np.nan


px = yf.download(PEERS, period="10d", progress=False, auto_adjust=False)["Close"].ffill().iloc[-1]
rows = []
for s in PEERS:
    rev, oi, ni, sbc = (ttm(s, c) for c in ["revenue", "operating_income", "net_income", "sbc"])
    gp, cor = ttm(s, "gross_profit"), ttm(s, "cost_of_revenue")
    if not np.isfinite(gp) and np.isfinite(cor):
        gp = rev - cor
    if s in SUPPRESS_GM:
        gp = np.nan
    cash = np.nansum([inst(s, c) for c in ["cash", "short_term_inv", "long_term_inv"]])
    dt = inst(s, "debt_term_total")
    if not np.isfinite(dt):
        dt = np.nansum([inst(s, "debt_term_nc"), inst(s, "debt_term_c")])
    net_cash = cash - np.nansum([dt, inst(s, "debt_short")])
    sh, p = shares(s), float(px[s])
    mcap, qv, qend, qg = p * sh, *latest_q_yoy(s)
    ev = mcap - net_cash
    a = annual(s, "revenue")
    rows.append({
        "sym": s, "mcap_$B": mcap / 1e9, "EV_$B": ev / 1e9, "netcash_$B": net_cash / 1e9,
        "TTMrev_$B": rev / 1e9, "lastQ": qend,
        "lastQ_rev_yoy_%": qg * 100 if np.isfinite(qg) else np.nan,
        "FYrev_g_%": (float(a.val.iloc[-1]) / float(a.val.iloc[-2]) - 1) * 100 if len(a) >= 2 else np.nan,
        "GM_%": gp / rev * 100 if np.isfinite(gp) else np.nan,
        "OM_%": oi / rev * 100, "SBC/rev_%": sbc / rev * 100 if np.isfinite(sbc) else np.nan,
        "P/E": np.nan if s in SUPPRESS_PE else (mcap / ni if ni > 0 else np.nan),
        "EV/S": ev / rev, "EV/EBIT": ev / oi if oi > 0 else np.nan,
        "EV/GP": ev / gp if np.isfinite(gp) else np.nan,
    })
t = pd.DataFrame(rows)
print("=" * 200)
print("PEER TABLE, IDENTICAL BASIS: GAAP, TTM = FY + latest YTD - prior YTD, latest close")
print("  GOOGL P/E suppressed (one-off non-operating gain); AMZN GM/EV-GP suppressed (tagging)")
print("=" * 200)
print(t.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

print("\n" + "=" * 200)
print("WHERE DOES AAPL RANK: multiple vs growth, and multiple per point of growth")
print("=" * 200)
u = t.dropna(subset=["EV/EBIT"]).copy()
u["EV/EBIT_per_g"] = u["EV/EBIT"] / u["lastQ_rev_yoy_%"]
u["EV/S_per_g"] = u["EV/S"] / u["lastQ_rev_yoy_%"]
print(u[["sym", "EV/S", "EV/EBIT", "P/E", "lastQ_rev_yoy_%", "OM_%", "GM_%", "EV/EBIT_per_g", "EV/S_per_g"]]
      .sort_values("EV/EBIT", ascending=False).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
aapl = t[t.sym == "AAPL"].iloc[0]
for col in ["EV/S", "EV/EBIT", "EV/GP", "P/E", "lastQ_rev_yoy_%", "OM_%", "GM_%"]:
    v = t[col].dropna()
    if aapl[col] is np.nan or not np.isfinite(aapl[col]):
        continue
    rank = (v > aapl[col]).sum() + 1
    print(f"  AAPL {col:16s} = {aapl[col]:7.2f}   rank {rank} of {len(v)} (1 = highest)")
