"""ADVERSARIAL VERIFY #6 -- the 'already priced' pillar, rebuilt independently.

Three attacks:
  (1) WINDOW LABEL. The claim says '97th percentile of 4,124 trading days since 2010 (94th since
      2019)'. The repo script it cites (_fd_AAPL_pe_history.py) is the BUGGY first pass; the fixed
      one (_fd_AAPL_pe_percentile.py) has windows of full/10y/5y/3y and NO 'since 2019' window.
      Compute the 2019 window explicitly, and report the as-reported GAAP number for every window
      so the reader can see the whole range rather than the top of it.
  (2) NORMALISATION CHOICE. The 97th figure uses the 'one-off tax added back' variant, which lowers
      only ~1 year of HISTORICAL P/Es (the window containing Q4 FY2024's $10.2bn State Aid charge)
      while today's number stays GAAP. Show both side by side per window.
  (3) RATE / GROWTH REGIME (the steelman for the OTHER side). A raw P/E percentile across 2010-2026
      compares today with the 2013-2016 'value Apple' era of 10-12x, when the 10-year Treasury was
      lower AND Apple's growth was negative. Recompute the percentile of (i) earnings yield minus
      the 10-year Treasury, and (ii) P/E divided by trailing EPS growth, so the level is judged
      against what it is being paid for.

Plus the event study: the four earnings-reaction days, AAPL vs SPY.
"""
from __future__ import annotations
import json, warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
pd.set_option("display.width", 230)

PRICE, EPS_TTM = 333.08, 8.71          # verified in _fd_AAPL_verify_ttm.py and the four 8-Ks
CF = json.load(open("_fd_AAPL_verify_cf.json"))
US = CF["facts"]["us-gaap"]
SPLITS = [(pd.Timestamp("2014-06-09"), 7.0), (pd.Timestamp("2020-08-31"), 4.0)]


def rows(concept):
    out = []
    for u, arr in US[concept]["units"].items():
        for f in arr:
            out.append({"start": f.get("start"), "end": f["end"], "val": float(f["val"]),
                        "filed": f["filed"], "accn": f["accn"], "form": f.get("form")})
    d = pd.DataFrame(out)
    for c in ("start", "end", "filed"):
        d[c] = pd.to_datetime(d[c])
    d = d[d.start.notna()].copy()
    d["days"] = (d.end - d.start).dt.days
    # POINT IN TIME: keep the FIRST filing that ever disclosed each period (no restatement lookahead)
    return d.sort_values("filed").drop_duplicates(["start", "end"], keep="first")


def split_adj(filed):
    return float(np.prod([r for d, r in SPLITS if filed < d]) or 1.0)


# ---- TTM EPS timeline: FY + latest YTD - prior-year same YTD. Immune to the missing Q4.
eps = rows("EarningsPerShareDiluted")
eps["val"] = eps.val / eps.filed.map(split_adj)
ann = eps[(eps.days >= 340) & (eps.days <= 400)].sort_values("end")
ev = [{"filed": r.filed, "ttm": r.val, "win_start": r.start, "period_end": r.end, "how": "FY"}
      for _, r in ann.iterrows()]
for _, r in eps[(eps.days >= 60) & (eps.days <= 320)].iterrows():
    pf = ann[ann.end < r.start]
    if pf.empty:
        continue
    fy = pf.iloc[-1]
    pool = eps[(eps.start.between(fy.start - pd.Timedelta(days=7), fy.start + pd.Timedelta(days=7)))
               & (eps.days.between(r.days - 12, r.days + 12))]
    if pool.empty:
        continue
    ev.append({"filed": r.filed, "ttm": fy.val + r.val - float(pool.val.iloc[-1]),
               "win_start": r.end - pd.Timedelta(days=364), "period_end": r.end, "how": f"{r.days}d YTD"})
E = (pd.DataFrame(ev).sort_values(["filed", "period_end"])
     .drop_duplicates("filed", keep="last").set_index("filed"))
print(f"point-in-time TTM EPS timeline: n={len(E)}  {E.index.min().date()} -> {E.index.max().date()}")
print(f"  latest row: TTM EPS {E.ttm.iloc[-1]:.2f} for period ending {E.period_end.iloc[-1].date()}"
      f"  (independent target {EPS_TTM})")

# one-off State Aid tax: $10,246M per the Q4 FY2025 8-K reconciliation (accn 0000320193-25-000077),
# $0.67/diluted share, in the quarter ended 2024-09-28.
ONE_OFF_PS, ONE_OFF_Q = 0.67, pd.Timestamp("2024-09-28")
E["ttm_n"] = np.where((E.win_start <= ONE_OFF_Q) & (E.period_end >= ONE_OFF_Q),
                      E.ttm + ONE_OFF_PS, E.ttm)

px = yf.download("AAPL", start="2009-06-01", progress=False, auto_adjust=False)
c = px["Close"]
c = c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c
c.index = c.index.tz_localize(None)
H = pd.DataFrame({"close": c})
H["eps"] = E.ttm.reindex(H.index, method="ffill")
H["eps_n"] = E.ttm_n.reindex(H.index, method="ffill")
H = H.dropna()
H["pe"], H["pe_n"] = H.close / H.eps, H.close / H.eps_n
H["eps_g"] = H.eps / H.eps.shift(252) - 1
H = H[(H.pe > 0) & (H.pe < 200)]

tnx = yf.download("^TNX", start="2009-06-01", progress=False, auto_adjust=False)["Close"]
tnx = tnx.iloc[:, 0] if isinstance(tnx, pd.DataFrame) else tnx
tnx.index = tnx.index.tz_localize(None)
H["y10"] = tnx.reindex(H.index).ffill()          # ^TNX quotes the yield in percent already
H["eyp"] = 100.0 / H.pe - H.y10                  # earnings-yield premium over the 10y, in points
pe_now = PRICE / EPS_TTM
y10_now = float(H.y10.iloc[-1])
eyp_now = 100.0 / pe_now - y10_now

print("\n" + "=" * 132)
print(f"TODAY: price {PRICE:.2f}  TTM GAAP EPS {EPS_TTM:.2f}  P/E {pe_now:.2f}x   "
      f"10y {y10_now:.2f}%   earnings-yield premium {eyp_now:+.2f}pts")
print("=" * 132)
print(f"  {'window':<26} {'n':>6}  {'GAAP P/E pctile':>16} {'tax-normd pctile':>17} "
      f"{'median P/E':>11} {'EY-premium pctile':>19}")
for start, lbl in (("2010-01-01", "since 2010 (claim's)"), ("2016-09-14", "last 10y (script's)"),
                   ("2019-01-01", "since 2019 (claim's label)"), ("2021-09-14", "last 5y"),
                   ("2023-09-14", "last 3y")):
    w = H[H.index >= start]
    p_raw = float((w.pe < pe_now).mean() * 100)
    p_nrm = float((w.pe_n < pe_now).mean() * 100)
    # for the earnings-yield premium a HIGH percentile = cheap, so report the "expensive" side
    p_eyp = float((w.eyp > eyp_now).mean() * 100)
    print(f"  {lbl:<26} {len(w):>6}  {p_raw:>15.1f}% {p_nrm:>16.1f}% {w.pe.median():>10.1f}x "
          f"{p_eyp:>18.1f}%")
print("  reading: last column = share of days on which the stock offered a BIGGER earnings yield")
print("           over the 10-year Treasury than today, i.e. a rate-adjusted 'more expensive than")
print("           today' count. Same direction as the P/E percentile if rates are not the story.")

print("\n  P/E relative to what it is buying (trailing 1y EPS growth on the same point-in-time series):")
print(f"    today: TTM EPS growth over 252 trading days = {float(H.eps_g.iloc[-1]):+.1%}, "
      f"P/E {pe_now:.1f}x  ->  PEG {pe_now/max(float(H.eps_g.iloc[-1])*100, 1e-9):.2f}")
sub = H[(H.index >= "2010-01-01") & (H.eps_g > 0.02)]
peg = sub.pe / (sub.eps_g * 100)
print(f"    percentile of today's PEG among the {len(sub)} days since 2010 with positive EPS growth:"
      f" {float((peg < pe_now/(float(H.eps_g.iloc[-1])*100)).mean()*100):.1f}%")

print("\n" + "=" * 132)
print("EVENT STUDY -- the four earnings-reaction days. 8-Ks are filed after the close, so the")
print("reaction is the NEXT session. Abnormal = AAPL return minus SPY return that day.")
print("=" * 132)
spy = yf.download("SPY", start="2025-09-01", progress=False, auto_adjust=True)["Close"]
spy = spy.iloc[:, 0] if isinstance(spy, pd.DataFrame) else spy
spy.index = spy.index.tz_localize(None)
aapl = yf.download("AAPL", start="2025-09-01", progress=False, auto_adjust=True)["Close"]
aapl = aapl.iloc[:, 0] if isinstance(aapl, pd.DataFrame) else aapl
aapl.index = aapl.index.tz_localize(None)
ra, rs = aapl.pct_change(), spy.pct_change()
FILED = [("2025-10-30", "Q4 FY2025  rev +8%,  EPS 1.85"),
         ("2026-01-29", "Q1 FY2026  rev +16%, EPS 2.84"),
         ("2026-04-30", "Q2 FY2026  rev +17%, EPS 2.01"),
         ("2026-07-30", "Q3 FY2026  rev +16%, EPS 2.02  <- the '+16.4% quarter'")]
tot_a = tot_ab = 1.0
print(f"  {'8-K filed':<12} {'reaction day':<13} {'AAPL':>8} {'SPY':>8} {'abnormal':>10}   label")
for f, lab in FILED:
    nxt = ra.index[ra.index > pd.Timestamp(f)]
    if len(nxt) == 0:
        continue
    d = nxt[0]
    a, s = float(ra[d]), float(rs[d])
    tot_a *= 1 + a
    tot_ab *= 1 + (a - s)
    print(f"  {f:<12} {str(d.date()):<13} {a:>+7.2%} {s:>+7.2%} {a-s:>+9.2%}   {lab}")
print(f"\n  cumulative over the four reaction days:  raw {tot_a-1:+.2%}   abnormal {tot_ab-1:+.2%}"
      f"    (claim: -4.4% cumulative, -8.4% abnormal on the last one)")
yr = aapl[aapl.index >= aapl.index[-1] - pd.Timedelta(days=366)]
print(f"  AAPL over the same trailing year: {float(yr.iloc[-1]/yr.iloc[0]-1):+.1%} "
      f"({yr.index[0].date()} -> {yr.index[-1].date()})")
print(f"  n = 4 earnings days. Any inference from this is n=4; it cannot separate 'priced in' from")
print(f"  'four idiosyncratic reactions'. State it as an observation, not a finding.")
