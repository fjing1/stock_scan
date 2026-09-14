"""(1) Options data-quality check via put-call parity. (2) Quarterly royalty/licence series
from the 6-K shareholder letters, to set falsifiable thresholds for 2026-11-04.
"""
import re

import numpy as np
import pandas as pd
import yfinance as yf
from lxml import html

pd.set_option("display.width", 240)
PX = 239.01
t = yf.Ticker("ARM")

print("=" * 100)
print("OPTIONS DATA QUALITY: put-call parity, C - P vs S - K*exp(-rT)")
print("=" * 100)
r_f = 0.040
for exp in ("2026-10-09", "2026-11-20", "2026-12-18"):
    T = (pd.Timestamp(exp) - pd.Timestamp("2026-09-14")).days / 365
    oc = t.option_chain(exp)
    ca, pu = oc.calls.copy(), oc.puts.copy()
    for df in (ca, pu):
        df["mid"] = np.where((df.bid > 0) & (df.ask > 0), (df.bid + df.ask) / 2, np.nan)
        df["spr"] = np.where((df.bid > 0) & (df.ask > 0), (df.ask - df.bid) / ((df.ask + df.bid) / 2), np.nan)
    m = ca[["strike", "mid", "spr", "openInterest", "impliedVolatility"]].merge(
        pu[["strike", "mid", "spr", "openInterest", "impliedVolatility"]], on="strike",
        suffixes=("_c", "_p"))
    m = m[(m.strike >= PX * 0.8) & (m.strike <= PX * 1.3)].dropna(subset=["mid_c", "mid_p"])
    m["synth_fwd"] = m.mid_c - m.mid_p + m.strike * np.exp(-r_f * T)
    m["parity_err%"] = 100 * (m.synth_fwd / PX - 1)
    print(f"\n  {exp}  T={T:.3f}y   spot {PX}")
    print(m[["strike", "mid_c", "mid_p", "spr_c", "spr_p", "openInterest_c", "openInterest_p",
             "synth_fwd", "parity_err%", "impliedVolatility_c", "impliedVolatility_p"]].round(3).to_string(index=False))
    print(f"   median synthetic forward from parity: {m.synth_fwd.median():.2f} "
          f"({100 * (m.synth_fwd.median() / PX - 1):+.2f}% vs spot)")
    print(f"   median bid-ask width: calls {100 * m.spr_c.median():.1f}%  puts {100 * m.spr_p.median():.1f}% of mid")

print("\n" + "=" * 100)
print("EARNINGS EVENT PREMIUM: Oct 9 (pre-earnings) vs Nov 20 (spans 2026-11-04 earnings)")
print("=" * 100)


def atm_straddle(exp):
    oc = t.option_chain(exp)
    ca, pu = oc.calls, oc.puts
    k = min(ca.strike, key=lambda x: abs(x - PX))
    cm = ca.loc[ca.strike == k, ["bid", "ask"]].iloc[0]
    pm = pu.loc[pu.strike == k, ["bid", "ask"]].iloc[0]
    c = (cm.bid + cm.ask) / 2
    p = (pm.bid + pm.ask) / 2
    return k, c + p


for exp in ("2026-10-09", "2026-10-30", "2026-11-20", "2026-12-18"):
    try:
        k, sd = atm_straddle(exp)
        dte = (pd.Timestamp(exp) - pd.Timestamp("2026-09-14")).days
        spans = "SPANS Nov-4 earnings" if pd.Timestamp(exp) > pd.Timestamp("2026-11-04") else "pre-earnings"
        print(f"  {exp} ({dte:>3}d) ATM K={k:.0f} straddle ${sd:6.2f} = {100 * sd / PX:5.2f}% of spot   "
              f"implied +/- move to expiry {100 * sd / PX * 0.8:5.2f}%   [{spans}]")
    except Exception as e:
        print(f"  {exp}: {e}")

print("\n" + "=" * 100)
print("QUARTERLY ROYALTY / LICENCE HISTORY from the 6-K Ex-99.2 shareholder letters ($m)")
print("=" * 100)
q = pd.DataFrame(
    [
        # (fiscal quarter, quarter-end, total revenue, royalty, license)
        # ALL taken from the 6-K Ex-99.2 "Key Financial and Operating Metrics" tables.
        # FY26 and FY25 quarterly sums reconcile EXACTLY to the 20-F annual figures (checked below).
        ("Q3 FY24", "2023-12-31", 824, 470, 354),
        ("Q4 FY24", "2024-03-31", 928, 514, 414),
        ("Q1 FY25", "2024-06-30", 939, 467, 472),
        ("Q2 FY25", "2024-09-30", 844, 514, 330),
        ("Q3 FY25", "2024-12-31", 983, 580, 403),
        ("Q4 FY25", "2025-03-31", 1241, 607, 634),
        ("Q1 FY26", "2025-06-30", 1053, 585, 468),
        ("Q2 FY26", "2025-09-30", 1135, 620, 515),
        ("Q3 FY26", "2025-12-31", 1242, 737, 505),
        ("Q4 FY26", "2026-03-31", 1490, 671, 819),
        ("Q1 FY27", "2026-06-30", 1289, 715, 574),
    ],
    columns=["q", "end", "total", "royalty", "license"],
)
q["roy_qoq%"] = q.royalty.pct_change() * 100
q["roy_yoy%"] = q.royalty.pct_change(4) * 100
q["tot_yoy%"] = q.total.pct_change(4) * 100
q["lic_yoy%"] = q.license.pct_change(4) * 100
print(q.round(1).to_string(index=False))
print("\n  RECONCILIATION to the audited 20-F:")
for fy, tgt, rtgt, ltgt in [("FY26", 4920, 2613, 2307), ("FY25", 4007, 2168, 1839)]:
    sub = q[q.q.str.contains(fy)]
    print(f"   {fy}: total {sub.total.sum():,} (20-F {tgt:,})  royalty {sub.royalty.sum():,} "
          f"(20-F {rtgt:,})  licence {sub.license.sum():,} (20-F {ltgt:,})"
          f"  {'MATCH' if (sub.total.sum() == tgt and sub.royalty.sum() == rtgt and sub.license.sum() == ltgt) else 'MISMATCH'}")

print("\n  *** THE ROYALTY LINE HAS NOT MADE A NEW HIGH IN THREE QUARTERS ***")
print(f"   Q3 FY26 (Dec-25) royalty ${q.loc[q.q == 'Q3 FY26', 'royalty'].iloc[0]}m  <- peak")
print(f"   Q4 FY26 (Mar-26) royalty ${q.loc[q.q == 'Q4 FY26', 'royalty'].iloc[0]}m  "
      f"({100 * (671 / 737 - 1):+.1f}% q/q, only +11% y/y)")
print(f"   Q1 FY27 (Jun-26) royalty ${q.loc[q.q == 'Q1 FY27', 'royalty'].iloc[0]}m  "
      f"({100 * (715 / 737 - 1):+.1f}% vs the Dec-25 peak)")
print("   ...and this is the SAME period in which the letters say 'data center royalties more than")
print("   doubled year over year'. Both can be true only if the non-data-centre royalty base")
print("   (which is mostly smartphones) is FLAT TO DOWN.")

print("\n" + "=" * 100)
print("THRESHOLDS FOR 2026-11-04 (Q2 FYE27)")
print("=" * 100)
base_roy = 620   # Q2 FY26 royalty
base_tot = 1135
guide = 1380
print(f"  Company guidance for Q2 FYE27: revenue ${guide}m +/- $50m (6-K 2026-07-29 letter)")
print(f"  Consensus non-GAAP EPS $0.47 (yfinance) = exactly the company's guidance midpoint")
print(f"  Q2 FY26 comparables: total ${base_tot}m, royalty ${base_roy}m")
print(f"  Guidance implies total revenue y/y +{100 * (guide / base_tot - 1):.1f}%")
for g in (0.15, 0.20, 0.25, 0.30):
    print(f"    royalty at +{g:.0%} y/y = ${base_roy * (1 + g):,.0f}m")
print("\n  Q1 FYE27 royalty was +22% y/y with 'data center royalties more than doubling'.")
print("  Falsification threshold: if Q2 FYE27 royalty grows < +15% y/y (i.e. < $713m) while")
print("  the data-centre narrative is intact, the royalty engine is decelerating on a rising base.")
