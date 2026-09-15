"""_fd_AAPL_history2.py -- a clean historical valuation percentile, fixing two contaminations in the
first pass, plus the reverse-DCF that asks what growth $333.08 actually requires.

CONTAMINATION 1 -- SPLITS. yfinance's unadjusted Close is split-adjusted; XBRL EPS is as-originally
reported. Apple split 7:1 (Jun 2014) and 4:1 (Aug 2020), so any P/E built from filings older than
the Aug-2020 split is off by up to 28x. Pre-2021 P/E values from the first pass (1x, 0x, 3x) are
artefacts. The usable window starts 2020-09-01.

CONTAMINATION 2 -- THE ONE-OFF TAX CHARGE. The 42.6x "historical maximum" P/E is not a valuation
peak: it is the $10,175M one-time tax charge in the quarter ending 2024-09-28 shrinking the TTM EPS
denominator. Every TTM window containing that quarter is restated here.

A tax-immune cross-check (price / sales per share) is computed alongside, because revenue cannot be
distorted by a tax item.
"""
import numpy as np
import pandas as pd
import yfinance as yf

import _fund_data as F

pd.set_option("display.width", 220)
SYM, PRICE_TODAY = "AAPL", 333.08
EPS_TTM, REV_TTM, NI_TTM = 8.71, 466_823e6, 128_930e6
SH_NOW, NET_CASH = 14_594_180_000, 62_220e6
ONE_OFF_TAX = 10_175e6                     # derived in _fd_AAPL_tax_normalize.py
ONE_OFF_Q_END = pd.Timestamp("2024-09-28")
panel = F.load()


def dur(concept):
    d = panel[(panel.symbol == SYM) & (panel.concept == concept)].copy()
    d = d[d["start"].notna()]
    d["days"] = (d["end"] - d["start"]).dt.days
    return d.sort_values(["end", "days", "filed"])


def ttm_timeline(concept):
    d = dur(concept)
    ann = d[(d.days >= 340) & (d.days <= 400)].drop_duplicates("end", keep="first").sort_values("end")
    ev = [(r.filed, float(r.val), r.start, r.end, "FY") for _, r in ann.iterrows()]
    itm = d[(d.days >= 60) & (d.days <= 320)].drop_duplicates(["start", "end"], keep="first")
    for _, r in itm.iterrows():
        pf = ann[ann.end < r.start]
        if pf.empty:
            continue
        fy = pf.iloc[-1]
        pool = d[(d.start.between(fy.start - pd.Timedelta(days=7), fy.start + pd.Timedelta(days=7))) &
                 (d.days.between(r.days - 12, r.days + 12))].drop_duplicates(["start", "end"], keep="first")
        if pool.empty:
            continue
        ev.append((r.filed, float(fy.val) + float(r.val) - float(pool.val.iloc[-1]),
                   r.end - pd.Timedelta(days=364), r.end, f"{r.days}d YTD"))
    e = pd.DataFrame(ev, columns=["filed", "ttm", "win_start", "period_end", "how"])
    return e.sort_values(["filed", "period_end"]).drop_duplicates("filed", keep="last").set_index("filed")


eps_t, rev_t, sh_t = ttm_timeline("eps_diluted"), ttm_timeline("revenue"), None
shq = dur("shares_diluted")
shq = shq[(shq.days >= 60) & (shq.days <= 120)].drop_duplicates("end", keep="first")
sh_t = shq.sort_values(["filed", "end"]).drop_duplicates("filed", keep="last").set_index("filed")["val"]

# restate any TTM EPS window that contains the one-off tax quarter
eps_t["contains_oneoff"] = (eps_t.win_start <= ONE_OFF_Q_END) & (eps_t.period_end >= ONE_OFF_Q_END)
eps_t["sh"] = sh_t.reindex(eps_t.index, method="ffill")
eps_t["ttm_norm"] = np.where(eps_t.contains_oneoff, eps_t.ttm + ONE_OFF_TAX / eps_t.sh, eps_t.ttm)

raw = yf.download(SYM, start="2015-01-01", progress=False, auto_adjust=False)
close = raw["Close"][SYM] if isinstance(raw["Close"], pd.DataFrame) else raw["Close"]
close.index = close.index.tz_localize(None)

h = pd.DataFrame({"close": close})
h["eps"] = eps_t["ttm"].reindex(h.index, method="ffill")
h["eps_n"] = eps_t["ttm_norm"].reindex(h.index, method="ffill")
h["rev"] = rev_t["ttm"].reindex(h.index, method="ffill")
h["sh"] = sh_t.reindex(h.index, method="ffill")
h = h.dropna()
h = h[h.index >= "2020-09-01"]              # post 4:1 split, so price and EPS share one basis
h["pe"] = h.close / h.eps
h["pe_n"] = h.close / h.eps_n
h["ps"] = h.close / (h.rev / h.sh)

print("=" * 122)
print(f"AAPL VALUATION PERCENTILES, POST-SPLIT WINDOW ONLY ({h.index[0].date()} -> {h.index[-1].date()}, "
      f"n={len(h)})")
print("=" * 122)
now = {"pe": PRICE_TODAY / EPS_TTM, "pe_n": PRICE_TODAY / EPS_TTM,
       "ps": PRICE_TODAY / (REV_TTM / SH_NOW)}
labels = {"pe": "trailing GAAP P/E (as reported)",
          "pe_n": "trailing GAAP P/E (one-off tax added back to every affected window)",
          "ps": "price / sales per share (tax-immune)"}
for k in ["pe", "pe_n", "ps"]:
    s = h[k]
    print(f"\n  {labels[k]}")
    print(f"    today {now[k]:6.2f}   median {s.median():6.2f}   p10 {s.quantile(.10):6.2f}   "
          f"p25 {s.quantile(.25):6.2f}   p75 {s.quantile(.75):6.2f}   p90 {s.quantile(.90):6.2f}   "
          f"max {s.max():6.2f}")
    print(f"    percentile of today = {(s < now[k]).mean()*100:.1f}   "
          f"vs median = {now[k]/s.median()-1:+.1%}   date of max = {s.idxmax().date()}")

print("\n  median by calendar year:")
print(h.groupby(h.index.year)[["pe", "pe_n", "ps"]].median().to_string(float_format=lambda x: f"{x:,.1f}"))

# ---------------------------------------------------------------- FCF inputs for the reverse DCF
def ttm_val(concept):
    d = dur(concept)
    a = d[(d.days >= 340) & (d.days <= 400)].sort_values("end")
    fy, fs, fe = float(a.val.iloc[-1]), a.start.iloc[-1], a.end.iloc[-1]
    y = d[(d.start > fe) & (d.days >= 60)].sort_values("days")
    if y.empty:
        return fy
    c = y.iloc[-1]
    p = d[(d.start.between(fs - pd.Timedelta(days=7), fs + pd.Timedelta(days=7))) &
          (d.days.between(int(c.days) - 12, int(c.days) + 12))]
    return fy + float(c.val) - float(p.val.iloc[-1]) if not p.empty else fy


ocf, capex, sbc = ttm_val("ocf"), ttm_val("capex"), ttm_val("sbc")
fcf = ocf - capex
print("\n" + "=" * 122)
print("CASH FLOW, TTM ENDING 2026-06-27  (all GAAP, from XBRL)")
print("=" * 122)
print(f"  operating cash flow  {ocf/1e6:>12,.0f} $M   ({ocf/REV_TTM:.1%} of revenue)")
print(f"  capex                {capex/1e6:>12,.0f} $M   ({capex/REV_TTM:.1%} of revenue)")
print(f"  free cash flow       {fcf/1e6:>12,.0f} $M   ({fcf/REV_TTM:.1%} of revenue)")
print(f"  SBC                  {sbc/1e6:>12,.0f} $M   ({sbc/REV_TTM:.1%} of revenue)")
print(f"  FCF after SBC add-back removed (SBC is a real cost already inside OCF as a non-cash")
print(f"    add-back, so cash FCF overstates economic FCF by SBC): {(fcf-sbc)/1e6:,.0f} $M "
      f"({(fcf-sbc)/REV_TTM:.1%} of revenue)")
print(f"  net income           {NI_TTM/1e6:>12,.0f} $M   FCF/NI = {fcf/NI_TTM:.2f}x")

# ---------------------------------------------------------------- reverse DCF
EV = PRICE_TODAY * SH_NOW - NET_CASH
print("\n" + "=" * 122)
print("REVERSE DCF -- WHAT REVENUE CAGR DOES THE CURRENT EV REQUIRE?")
print("=" * 122)
print(f"  EV = {PRICE_TODAY:.2f} x {SH_NOW/1e9:.3f}bn shares - {NET_CASH/1e9:.1f}bn net cash "
      f"= ${EV/1e9:,.0f}bn")
print("  Model: 10-year explicit horizon, revenue grows at CAGR c, FCF margin m held at the TTM")
print("  level, terminal value = FCF_10 x (1+g) / (r-g). Solved for c by bisection.")
print(f"  TTM revenue {REV_TTM/1e9:,.1f}bn, TTM FCF margin {fcf/REV_TTM:.1%}\n")


def pv(c, m, r, g, n=10):
    tot = sum(REV_TTM * (1 + c) ** t * m / (1 + r) ** t for t in range(1, n + 1))
    fcf_n = REV_TTM * (1 + c) ** n * m
    return tot + fcf_n * (1 + g) / (r - g) / (1 + r) ** n


def solve_c(m, r, g, target=EV):
    lo, hi = -0.30, 0.60
    for _ in range(200):
        mid = (lo + hi) / 2
        if pv(mid, m, r, g) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


m_cash = fcf / REV_TTM
m_econ = (fcf - sbc) / REV_TTM
rows = []
for label, m in [("FCF margin as reported", m_cash), ("FCF margin net of SBC", m_econ)]:
    for r in [0.08, 0.09, 0.10, 0.11]:
        for g in [0.025, 0.030]:
            rows.append({"FCF margin basis": label, "margin_%": m * 100, "r_%": r * 100,
                         "g_%": g * 100, "required_rev_CAGR_%": solve_c(m, r, g) * 100})
req = pd.DataFrame(rows)
print(req.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

print("\n" + "=" * 122)
print("THE CONTINUATION CASE PRICED EXPLICITLY -- what is AAPL worth if the DEMONSTRATED trend just")
print("continues, rather than a step-change?")
print("=" * 122)
hist_cagr = {"FY2020->FY2025 revenue CAGR": (416_161 / 274_515) ** (1 / 5) - 1,
             "FY2023->FY2025 revenue CAGR": (416_161 / 383_285) ** (1 / 2) - 1,
             "TTM revenue YoY (current run rate)": 0.1424,
             "latest quarter YoY": 0.1636}
for k, v in hist_cagr.items():
    print(f"  {k:38s} {v:+.2%}")
print()
out = []
for lbl, c in [("5yr history 8.7%", (416_161 / 274_515) ** (1 / 5) - 1),
               ("2yr history 4.2%", (416_161 / 383_285) ** (1 / 2) - 1),
               ("current TTM run rate 14.2%", 0.1424),
               ("current quarter run rate 16.4%", 0.1636)]:
    for m, mlbl in [(m_cash, "as-rptd"), (m_econ, "net SBC")]:
        for r in [0.09, 0.10]:
            v = pv(c, m, r, 0.03)
            eq = v + NET_CASH
            out.append({"growth case": lbl, "margin": mlbl, "r_%": r * 100,
                        "implied_EV_$B": v / 1e9, "implied_price": eq / SH_NOW,
                        "vs_333.08_%": (eq / SH_NOW) / PRICE_TODAY * 100 - 100})
print(pd.DataFrame(out).to_string(index=False, float_format=lambda x: f"{x:,.1f}"))
