"""_fd_AAPL_peers.py -- peer multiples on an IDENTICAL basis, using a TTM construction that is
immune to the missing-Q4 problem.

THE GENERIC FIX. Rather than summing four quarterly duration facts (which fails for any filer whose
10-K omits a Q4 duration -- Apple, MSFT, QCOM all do this), build TTM from CUMULATIVE year-to-date
facts, which every filer tags:

        TTM = last completed fiscal year  +  latest YTD  -  prior-year same-length YTD

For Apple this reproduces 416,161 + 364,357 - 313,695 = 466,823, matching the value derived
independently via the Q4 = annual - (Q1+Q2+Q3) identity in _fd_AAPL_ttm_multiples.py. Two
independent routes agreeing is the check that the number is right.

Peers chosen and why:
  MSFT, GOOGL, META  mega-cap platform businesses of comparable scale, gross margin and cash
                     generation -- the set a manager actually chooses between when allocating to
                     "quality mega-cap"
  QCOM               the smartphone-hardware-cycle comparable: same end market and same
                     replacement-cycle exposure as the iPhone line that is driving Apple's
                     acceleration, so it prices what the market pays for that specific risk
  AMZN               consumer-platform scale comparable, included to widen the multiple/growth cloud
"""
import numpy as np
import pandas as pd
import yfinance as yf

import _fund_data as F

pd.set_option("display.width", 260)
panel = F.load()

PEERS = ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "QCOM"]


def dur(sym, concept):
    """All duration facts for a concept, newest revision per (start,end) kept."""
    d = panel[(panel.symbol == sym) & (panel.concept == concept)].copy()
    d = d[d["start"].notna()]
    if d.empty:
        return pd.DataFrame()
    d["days"] = (d["end"] - d["start"]).dt.days
    d = d.sort_values(["end", "days", "filed"]).drop_duplicates(["start", "end"], keep="last")
    return d


def ttm(sym, concept):
    """TTM via annual + latest YTD - prior-year same YTD. Returns (value, as_of, method)."""
    d = dur(sym, concept)
    if d.empty:
        return np.nan, None, "no data"
    ann = d[(d.days >= 340) & (d.days <= 400)].sort_values("end")
    if ann.empty:
        return np.nan, None, "no annual"
    fy_val, fy_end = float(ann.val.iloc[-1]), ann.end.iloc[-1]
    fy_start = ann.start.iloc[-1]
    # YTD facts starting the day after the last fiscal year end
    ytd = d[(d.start > fy_end) & (d.days >= 60)].sort_values("days")
    if ytd.empty:
        return fy_val, fy_end.date(), "FY only (no newer YTD)"
    cur = ytd.iloc[-1]
    n = int(cur.days)
    # prior-year YTD of the same length, starting the day after the PRIOR fiscal year end
    prior_pool = d[(d.start <= fy_start + pd.Timedelta(days=7)) &
                   (d.start >= fy_start - pd.Timedelta(days=7)) &
                   (d.days.between(n - 12, n + 12))].sort_values("filed")
    if prior_pool.empty:
        return fy_val, fy_end.date(), "FY only (no prior YTD match)"
    prior = prior_pool.iloc[-1]
    return fy_val + float(cur.val) - float(prior.val), cur.end.date(), \
        f"FY{fy_end.date()} + {n}d YTD - prior {int(prior.days)}d YTD"


def inst(sym, concept):
    d = panel[(panel.symbol == sym) & (panel.concept == concept)].copy()
    d = d[d["start"].isna()]
    if d.empty:
        return np.nan
    d = d.sort_values(["end", "filed"])
    return float(d.val.iloc[-1])


def shares_outstanding(sym):
    d = panel[(panel.symbol == sym) & (panel.concept == "shares_diluted")].copy()
    d = d[d["start"].notna()]
    if d.empty:
        return np.nan
    d["days"] = (d["end"] - d["start"]).dt.days
    d = d[(d.days >= 60) & (d.days <= 120)].sort_values(["end", "filed"])
    return float(d.val.iloc[-1]) if not d.empty else np.nan


px = yf.download(PEERS, period="10d", progress=False, auto_adjust=False)["Close"].ffill().iloc[-1]

rows = []
for s in PEERS:
    rev, asof, meth = ttm(s, "revenue")
    gp, _, _ = ttm(s, "gross_profit")
    cor, _, _ = ttm(s, "cost_of_revenue")
    if not np.isfinite(gp) and np.isfinite(cor):
        gp = rev - cor
    oi, _, _ = ttm(s, "operating_income")
    ni, _, _ = ttm(s, "net_income")
    sbc, _, _ = ttm(s, "sbc")
    rnd, _, _ = ttm(s, "rnd")
    # prior-year TTM = the completed FY before last, plus nothing: use FY-over-FY as the growth
    # anchor, and separately the YTD-over-YTD growth which is the current run rate.
    d = dur(s, "revenue")
    ann = d[(d.days >= 340) & (d.days <= 400)].sort_values("end")
    fy_g = float(ann.val.iloc[-1]) / float(ann.val.iloc[-2]) - 1 if len(ann) >= 2 else np.nan
    fy_end, fy_start = ann.end.iloc[-1], ann.start.iloc[-1]
    ytd = d[(d.start > fy_end) & (d.days >= 60)].sort_values("days")
    ytd_g = np.nan
    if not ytd.empty:
        cur = ytd.iloc[-1]
        n = int(cur.days)
        pp = d[(d.start.between(fy_start - pd.Timedelta(days=7), fy_start + pd.Timedelta(days=7))) &
               (d.days.between(n - 12, n + 12))].sort_values("filed")
        if not pp.empty:
            ytd_g = float(cur.val) / float(pp.val.iloc[-1]) - 1

    cash = np.nansum([inst(s, c) for c in ["cash", "short_term_inv", "long_term_inv"]])
    dt = inst(s, "debt_term_total")
    if not np.isfinite(dt):
        dt = np.nansum([inst(s, "debt_term_nc"), inst(s, "debt_term_c")])
    debt = np.nansum([dt, inst(s, "debt_short")])
    net_cash = cash - debt
    sh = shares_outstanding(s)
    p = float(px[s])
    mcap = p * sh
    ev = mcap - net_cash
    rows.append({
        "sym": s, "px": p, "sh_M": sh / 1e6, "mcap_$B": mcap / 1e9, "netcash_$B": net_cash / 1e9,
        "TTM_rev_$B": rev / 1e9, "asof": asof,
        "FY_rev_g_%": fy_g * 100, "YTD_rev_g_%": ytd_g * 100,
        "GM_%": gp / rev * 100 if np.isfinite(gp) else np.nan,
        "OM_%": oi / rev * 100 if np.isfinite(oi) else np.nan,
        "NM_%": ni / rev * 100 if np.isfinite(ni) else np.nan,
        "SBC/rev_%": sbc / rev * 100 if np.isfinite(sbc) else np.nan,
        "RnD/rev_%": rnd / rev * 100 if np.isfinite(rnd) else np.nan,
        "P/E": mcap / ni if np.isfinite(ni) and ni > 0 else np.nan,
        "EV/S": ev / rev, "EV/GP": ev / gp if np.isfinite(gp) else np.nan,
        "EV/EBIT": ev / oi if np.isfinite(oi) and oi > 0 else np.nan,
    })

t = pd.DataFrame(rows)
print("=" * 250)
print("PEER TABLE -- ALL GAAP, ALL TTM VIA (annual + latest YTD - prior YTD), prices are the latest close")
print("=" * 250)
print(t.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

print("\n--- TTM construction method actually used per name (audit trail)")
for s in PEERS:
    v, a, m = ttm(s, "revenue")
    print(f"  {s:6s} rev_ttm={v/1e9:>9,.1f}B  as_of={a}  method: {m}")

print("\n--- multiple vs growth: EV/S per point of YTD revenue growth, and P/E per point")
t["EVS_per_g"] = t["EV/S"] / t["YTD_rev_g_%"]
t["PE_per_g"] = t["P/E"] / t["YTD_rev_g_%"]
print(t[["sym", "EV/S", "P/E", "YTD_rev_g_%", "GM_%", "OM_%", "EVS_per_g", "PE_per_g"]]
      .sort_values("PE_per_g").to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
