"""ADVERSARIAL VERIFY #7 -- clean the point-in-time TTM EPS timeline, then re-run the percentile.

Why this file exists. Both the claim's _fd_AAPL_pe_percentile.py and my first reproduction build the
TTM timeline as  FY + current-period - prior-year-period, where 'period' is ANY duration fact of
60-320 days. That admits a NON-YTD fact: for the Q3 FY2026 filing it pairs the 90-day Q3 (start
2026-03-29) with the 90-day Q1 FY2025 (start 2024-09-29, which IS the fiscal-year start and so
passes the pool test), producing FY2025 + Q3FY26 - Q1FY25 = 7.08 -- not a TTM at all. Whether that
garbage row or the correct 8.72 row survives depends only on the pre-sort iteration order feeding
drop_duplicates(keep='last'). The repo script happens to sort by ['end','days'] and therefore
happens to keep the right one. That is luck, not logic.

Fix: a period may only enter the construction if its START is the fiscal-year start (i.e. it is a
genuine year-to-date cumulative), and the prior-year period must match on BOTH fiscal-year start and
duration. Then re-run every percentile window.
"""
from __future__ import annotations
import json, warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)

PRICE, EPS_TTM = 333.08, 8.71
CF = json.load(open("_fd_AAPL_verify_cf.json"))
US = CF["facts"]["us-gaap"]
SPLITS = [(pd.Timestamp("2014-06-09"), 7.0), (pd.Timestamp("2020-08-31"), 4.0)]


def rows(concept, per_share):
    out = []
    for u, arr in US[concept]["units"].items():
        for f in arr:
            out.append({"start": f.get("start"), "end": f["end"], "val": float(f["val"]),
                        "filed": f["filed"], "form": f.get("form")})
    d = pd.DataFrame(out)
    for c in ("start", "end", "filed"):
        d[c] = pd.to_datetime(d[c])
    d = d[d.start.notna()].copy()
    d["days"] = (d.end - d.start).dt.days
    if per_share:
        d["val"] = d.val / d.filed.map(lambda f: float(np.prod([r for dt, r in SPLITS if f < dt]) or 1.0))
    return d.sort_values("filed").drop_duplicates(["start", "end"], keep="first")   # first-filed only


def ttm_timeline(concept, per_share=False):
    d = rows(concept, per_share)
    ann = d[(d.days >= 340) & (d.days <= 400)].sort_values("end").reset_index(drop=True)
    # A fiscal year that has not yet been reported has no annual fact, so its start is not in
    # ann.start. Add (prior FY end + 1 day) so the CURRENT year's YTD facts are recognised too --
    # without this the timeline stops at the last 10-K and today's TTM is the stale FY figure,
    # which is exactly the failure mode in fund_metrics.py.
    fy_starts = list(ann.start) + list(ann.end + pd.Timedelta(days=1))
    ev = [{"filed": r.filed, "ttm": r.val, "period_end": r.end, "win_start": r.start, "how": "FY"}
          for _, r in ann.iterrows()]
    # a genuine YTD fact: start within +-7d of SOME fiscal-year start, and shorter than a year
    def is_ytd(s):
        return any(abs((s - f).days) <= 7 for f in fy_starts)
    ytd = d[(d.days >= 60) & (d.days <= 320) & d.start.map(is_ytd)]
    for _, r in ytd.iterrows():
        pf = ann[ann.end < r.start]
        if pf.empty:
            continue
        fy = pf.iloc[-1]                                     # last COMPLETED fiscal year
        prior = ytd[(abs((ytd.start - fy.start).dt.days) <= 7) & (abs(ytd.days - r.days) <= 12)]
        if prior.empty:
            continue
        ev.append({"filed": r.filed, "ttm": fy.val + r.val - float(prior.val.iloc[-1]),
                   "period_end": r.end, "win_start": r.end - pd.Timedelta(days=364),
                   "how": f"FY{fy.end.year}+{r.days}dYTD-prior"})
    E = pd.DataFrame(ev).sort_values(["filed", "period_end", "how"])
    # among rows sharing a filed date, keep the one with the LATEST period_end (most current info)
    E = E.sort_values(["filed", "period_end"]).drop_duplicates("filed", keep="last")
    return E.set_index("filed")


E = ttm_timeline("EarningsPerShareDiluted", per_share=True)
print("=" * 132)
print("CLEAN point-in-time TTM diluted EPS timeline -- last 12 filings")
print("=" * 132)
print(E.tail(12).to_string(float_format=lambda x: f"{x:,.2f}"))
print(f"\n  latest: {E.ttm.iloc[-1]:.2f} for period ending {E.period_end.iloc[-1].date()}, "
      f"filed {E.index[-1].date()}   (independent target from the four 8-Ks: 8.72 / XBRL 8.71)")
assert abs(E.ttm.iloc[-1] - EPS_TTM) < 0.02, "timeline still does not tie to the verified TTM"
print("  -> ties. The timeline is now clean.")

ONE_OFF_PS, ONE_OFF_Q = 0.67, pd.Timestamp("2024-09-28")   # Q4 FY2025 8-K, accn 0000320193-25-000077
E["ttm_n"] = np.where((E.win_start <= ONE_OFF_Q) & (E.period_end >= ONE_OFF_Q), E.ttm + ONE_OFF_PS, E.ttm)

px = yf.download("AAPL", start="2009-06-01", progress=False, auto_adjust=False)["Close"]
px = px.iloc[:, 0] if isinstance(px, pd.DataFrame) else px
px.index = px.index.tz_localize(None)
H = pd.DataFrame({"close": px})
H["eps"] = E.ttm.reindex(H.index, method="ffill")
H["eps_n"] = E.ttm_n.reindex(H.index, method="ffill")
H = H.dropna()
H["pe"], H["pe_n"] = H.close / H.eps, H.close / H.eps_n
H = H[(H.pe > 0) & (H.pe < 200)]
tnx = yf.download("^TNX", start="2009-06-01", progress=False, auto_adjust=False)["Close"]
tnx = tnx.iloc[:, 0] if isinstance(tnx, pd.DataFrame) else tnx
tnx.index = tnx.index.tz_localize(None)
H["y10"] = tnx.reindex(H.index).ffill()
H["eyp"] = 100.0 / H.pe - H.y10

pe_now = PRICE / EPS_TTM
eyp_now = 100.0 / pe_now - float(H.y10.iloc[-1])
print("\n" + "=" * 132)
print(f"TODAY  P/E {pe_now:.2f}x   10y {float(H.y10.iloc[-1]):.2f}%   earnings-yield premium {eyp_now:+.2f}pts")
print("=" * 132)
print(f"  {'window':<28} {'n':>6} {'GAAP P/E pctile':>17} {'tax-normd':>11} {'median':>9} "
      f"{'p90':>8} {'max':>8} {'EYprem pctile':>15}")
out = {}
for start, lbl in (("2010-01-01", "since 2010 (claim's window)"), ("2016-09-14", "last 10 years"),
                   ("2019-01-01", "since 2019 (claim's label)"), ("2021-09-14", "last 5 years"),
                   ("2023-09-14", "last 3 years")):
    w = H[H.index >= start]
    r = (len(w), float((w.pe < pe_now).mean() * 100), float((w.pe_n < pe_now).mean() * 100),
         w.pe.median(), w.pe.quantile(.9), w.pe.max(), float((w.eyp > eyp_now).mean() * 100))
    out[lbl] = r
    print(f"  {lbl:<28} {r[0]:>6} {r[1]:>16.1f}% {r[2]:>10.1f}% {r[3]:>8.1f}x {r[4]:>7.1f}x "
          f"{r[5]:>7.1f}x {r[6]:>14.1f}%")

print("\n  median P/E by calendar year (sanity vs the known re-rating story):")
print(H.groupby(H.index.year)[["pe", "close"]].median().to_string(float_format=lambda x: f"{x:,.1f}"))
H.to_csv("_fd_AAPL_verify_pe_clean.csv")
print("\nsaved _fd_AAPL_verify_pe_clean.csv")
