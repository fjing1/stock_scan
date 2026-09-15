"""_fd_AAPL_history_percentile.py -- where does today's multiple sit in AAPL's OWN trading history?

Point-in-time by construction: for each trading day the TTM EPS / TTM revenue used is only the one
that had actually been FILED by that date (XBRL `filed` field), so no figure is applied before the
market could have known it. Q4 quarters, which Apple never tags as a ~90-day duration, are handled
by building TTM from cumulative YTD facts (FY + latest YTD - prior-year YTD) rather than by summing
four quarterly rows.
"""
import numpy as np
import pandas as pd
import yfinance as yf

import _fund_data as F

pd.set_option("display.width", 220)
SYM, PRICE_TODAY, EPS_TTM_NOW = "AAPL", 333.08, 8.71
panel = F.load()


def dur(concept):
    d = panel[(panel.symbol == SYM) & (panel.concept == concept)].copy()
    d = d[d["start"].notna()]
    d["days"] = (d["end"] - d["start"]).dt.days
    return d.sort_values(["end", "days", "filed"])


def ttm_timeline(concept):
    """DataFrame indexed by FILED date: the TTM value that became public on that date."""
    d = dur(concept)
    ann = d[(d.days >= 340) & (d.days <= 400)].drop_duplicates("end", keep="first").sort_values("end")
    events = [(r.filed, float(r.val), r.end, "FY") for _, r in ann.iterrows()]
    interim = d[(d.days >= 60) & (d.days <= 320)].drop_duplicates(["start", "end"], keep="first")
    for _, r in interim.iterrows():
        prior_fy = ann[ann.end < r.start]
        if prior_fy.empty:
            continue
        fy = prior_fy.iloc[-1]
        pool = d[(d.start.between(fy.start - pd.Timedelta(days=7), fy.start + pd.Timedelta(days=7))) &
                 (d.days.between(r.days - 12, r.days + 12))].drop_duplicates(["start", "end"], keep="first")
        if pool.empty:
            continue
        events.append((r.filed, float(fy.val) + float(r.val) - float(pool.val.iloc[-1]),
                       r.end, f"{r.days}d YTD"))
    e = pd.DataFrame(events, columns=["filed", "ttm", "period_end", "how"])
    e = e.sort_values(["filed", "period_end"]).drop_duplicates("filed", keep="last")
    return e.set_index("filed")


eps = ttm_timeline("eps_diluted")
rev = ttm_timeline("revenue")

raw = yf.download(SYM, start="2010-01-01", progress=False, auto_adjust=False)
close = raw["Close"][SYM] if isinstance(raw["Close"], pd.DataFrame) else raw["Close"]
close.index = close.index.tz_localize(None)

hist = pd.DataFrame({"close": close})
hist["eps_ttm"] = eps["ttm"].reindex(hist.index, method="ffill")
hist["rev_ttm"] = rev["ttm"].reindex(hist.index, method="ffill")
hist = hist.dropna()
hist["pe"] = hist.close / hist.eps_ttm
hist = hist[(hist.pe > 0) & (hist.pe < 200)]

pe_now = PRICE_TODAY / EPS_TTM_NOW
print("=" * 120)
print("AAPL TRAILING GAAP P/E, POINT-IN-TIME  (unadjusted close / TTM diluted EPS already filed)")
print("=" * 120)
for lbl, w in [("full history since 2010", hist), ("last 10 years", hist.last("3650D")),
               ("last 5 years", hist.last("1825D")), ("last 3 years", hist.last("1095D"))]:
    pct = (w.pe < pe_now).mean() * 100
    print(f"  {lbl:26s} n={len(w):5d}  median {w.pe.median():5.1f}x  p10 {w.pe.quantile(.10):5.1f}x  "
          f"p90 {w.pe.quantile(.90):5.1f}x  max {w.pe.max():5.1f}x   "
          f"today {pe_now:.1f}x = {pct:5.1f}th pctile")

print("\n  median trailing P/E by calendar year:")
ym = hist.groupby(hist.index.year).pe.median()
print("   " + "   ".join(f"{y}:{v:.0f}x" for y, v in ym.items()))

print("\n" + "=" * 120)
print("EPS TIMELINE AUDIT (the last 10 filing events used)")
print("=" * 120)
print(eps.tail(10).to_string(float_format=lambda x: f"{x:,.2f}"))
print(f"\n  latest TTM EPS in the timeline: {eps['ttm'].iloc[-1]:.2f}  "
      f"(period end {eps['period_end'].iloc[-1].date()}, filed {eps.index[-1].date()})")
print(f"  independent derivation in _fd_AAPL_valuation.py gave {EPS_TTM_NOW:.2f}  -> "
      f"{'AGREE' if abs(eps['ttm'].iloc[-1]-EPS_TTM_NOW) < 0.05 else 'DISAGREE, investigate'}")

print("\n" + "=" * 120)
print("PRICE CONTEXT")
print("=" * 120)
for lbl, w in [("3 years", close.last("1095D")), ("5 years", close.last("1825D")),
               ("10 years", close.last("3650D"))]:
    print(f"  close {PRICE_TODAY:.2f} vs {lbl}: min {w.min():7.2f}  max {w.max():7.2f}  "
          f"pctile {(w < PRICE_TODAY).mean()*100:5.1f}   from high {PRICE_TODAY/w.max()-1:+.1%}")

hist.to_pickle("_fd_AAPL_pe_history.pkl")
print("\n  saved _fd_AAPL_pe_history.pkl")
