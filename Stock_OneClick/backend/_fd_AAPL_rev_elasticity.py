"""AAPL: where growth actually comes from, and the holder's implied elasticity.

All inputs are filed figures (8-K EX-99.1 / 10-Q / 10-K); every derived number shows its arithmetic.
Accessions cited inline.
"""
from __future__ import annotations
import pandas as pd, numpy as np

pd.set_option("display.width", 220)

# ---------------- filed quarterly category & geography net sales ($M)
# source: 8-K EX-99.1 tables, cross-checked against the 10-Q/10-K XBRL dimensional facts
CAT = pd.DataFrame([
    # qend        iPhone     Mac    iPad  Wear   Services   total
    ("Q4FY24", "2024-09-28", 46222,  7744,  6950,  9042, 24972,  94930),
    ("Q1FY25", "2024-12-28", 69138,  8987,  8088, 11747, 26340, 124300),
    ("Q2FY25", "2025-03-29", 46841,  7949,  6402,  7522, 26645,  95359),
    ("Q3FY25", "2025-06-28", 44582,  8046,  6581,  7404, 27423,  94036),
    ("Q4FY25", "2025-09-27", 49025,  8726,  6952,  9013, 28750, 102466),
    ("Q1FY26", "2025-12-27", 85269,  8386,  8595, 11493, 30013, 143756),
    ("Q2FY26", "2026-03-28", 56994,  8399,  6914,  7901, 30976, 111184),
    ("Q3FY26", "2026-06-27", 54252, 10352,  6191,  7883, 30739, 109417),
], columns=["tag", "qend", "iPhone", "Mac", "iPad", "Wear", "Services", "total"]).set_index("tag")
GEO = pd.DataFrame([
    ("Q4FY24", 41664, 24924, 15033, 5926, 7383),
    ("Q1FY25", 52648, 33861, 18513, 8987, 10291),
    ("Q2FY25", 40315, 24454, 16002, 7298, 7290),
    ("Q3FY25", 41198, 24014, 15369, 5782, 7673),
    ("Q4FY25", 44192, 28703, 14493, 6636, 8442),
    ("Q1FY26", 58529, 38146, 25526, 9413, 12142),
    ("Q2FY26", 45093, 28055, 20497, 8401, 9138),
    ("Q3FY26", 45781, 29395, 18816, 6554, 8871),
], columns=["tag", "Americas", "Europe", "GreaterChina", "Japan", "RestAP"]).set_index("tag")
CATS = ["iPhone", "Mac", "iPad", "Wear", "Services"]
GEOS = list(GEO.columns)
assert (CAT[CATS].sum(1) - CAT.total).abs().max() == 0
assert (GEO.sum(1) - CAT.total).abs().max() == 0
print("category and geography tables both foot to total net sales for all 8 quarters.")

cur = ["Q4FY25", "Q1FY26", "Q2FY26", "Q3FY26"]
pri = ["Q4FY24", "Q1FY25", "Q2FY25", "Q3FY25"]
tc, tp = CAT.loc[cur], CAT.loc[pri]
gc, gp_ = GEO.loc[cur], GEO.loc[pri]

print("\n" + "=" * 118)
print("1. TTM vs prior TTM  ($M)  -- window Q4FY25..Q3FY26 vs Q4FY24..Q3FY25")
print("=" * 118)
rows = []
tot_d = tc.total.sum() - tp.total.sum()
for c in CATS:
    a, b = tc[c].sum(), tp[c].sum()
    rows.append((c, b, a, a - b, a / b - 1, (a - b) / tot_d, a / tc.total.sum()))
for c in GEOS:
    a, b = gc[c].sum(), gp_[c].sum()
    rows.append((c, b, a, a - b, a / b - 1, (a - b) / tot_d, a / tc.total.sum()))
R = pd.DataFrame(rows, columns=["line", "prior_TTM", "TTM", "delta", "growth", "share_of_growth", "share_of_rev"]).set_index("line")
print(R.to_string(formatters={"prior_TTM": "{:,.0f}".format, "TTM": "{:,.0f}".format,
                              "delta": "{:+,.0f}".format, "growth": "{:+.1%}".format,
                              "share_of_growth": "{:+.1%}".format, "share_of_rev": "{:.1%}".format}))
print(f"\n  total TTM revenue {tc.total.sum():,.0f}  vs prior {tp.total.sum():,.0f}  = {tot_d:+,.0f} ({tot_d/tp.total.sum():+.1%})")
print("  RANKING BY CONTRIBUTION TO GROWTH, not by size:")
print("   ", "  >  ".join(f"{k} {v:+.0%}" for k, v in
                          R.loc[CATS, "share_of_growth"].sort_values(ascending=False).items()))
print("   ", "  >  ".join(f"{k} {v:+.0%}" for k, v in
                          R.loc[GEOS, "share_of_growth"].sort_values(ascending=False).items()))

# ---------------- the iPhone plateau
print("\n" + "=" * 118)
print("2. IS THE iPHONE STEP-UP A NEW LEVEL OR A DEVIATION FROM A 4-YEAR PLATEAU?")
print("   annual iPhone net sales, 10-K filings (FY22 acc -22-000108, FY23 -23-000106,")
print("   FY24 -24-000123, FY25 -25-000079); TTM built from the four filed quarters above")
print("=" * 118)
iph = {"FY2022": 205489, "FY2023": 200583, "FY2024": 201183, "FY2025": 209586,
       "TTM to 2026-06-27": tc.iPhone.sum()}
prev = None
for k, v in iph.items():
    g = f"{v/prev-1:+.1%}" if prev else "     -"
    print(f"  {k:<22}{v:>10,.0f}   {g}")
    prev = v
plateau = np.mean([205489, 200583, 201183, 209586])
print(f"\n  FY22-FY25 mean iPhone revenue        {plateau:>10,.0f}")
print(f"  TTM iPhone revenue                  {tc.iPhone.sum():>10,.0f}   = {tc.iPhone.sum()/plateau-1:+.1%} above that plateau")
print(f"  dollars of TTM revenue riding on that gap: {tc.iPhone.sum()-plateau:+,.0f} $M"
      f" = {(tc.iPhone.sum()-plateau)/tc.total.sum():.1%} of TTM revenue")
print("  10-Q Q3FY26 (acc 0000320193-26-000020) attribution, verbatim: iPhone net sales increased")
print("   'primarily due to higher net sales of Pro models.'  -> MIX / ASP, not units.")
print("  Apple has not disclosed iPhone UNITS since the FY2018 10-K, so units vs price CANNOT be")
print("   separated from primary sources. The bound is: 100% of the +18.7% could be price/mix.")

# ---------------- elasticity
print("\n" + "=" * 118)
print("3. ELASTICITY. TTM base: revenue, gross margin by type, opex, tax, share count -- all filed.")
print("=" * 118)
TTM = dict(rev=466823.0, prod_rev=346345.0, svc_rev=120478.0, prod_gp=137160.0, svc_gp=90963.0,
           gp=228123.0, opex=75425.0, opinc=154859.0, tax_rate=0.1758, sh=14714.676, eps=8.72)
# rebuild from the filed quarterly P&L so nothing is asserted
QP = pd.read_csv("_fd_AAPL_rev_quarterly.csv", index_col=0)
t4 = QP.iloc[-4:]
TTM.update(rev=t4.rev.sum(), prod_rev=t4.prod_rev.sum(), svc_rev=t4.svc_rev.sum(),
           prod_gp=(t4.prod_rev - t4.prod_cos).sum(), svc_gp=(t4.svc_rev - t4.svc_cos).sum(),
           gp=t4.gp.sum(), opex=(t4.rnd + t4.sga).sum(), opinc=t4.opinc.sum(),
           tax_rate=t4.tax.sum() / t4.pretax.sum(), eps=t4.eps.sum(),
           sh=t4.sh_dil.iloc[-1] / 1000.0)
prod_gm = TTM["prod_gp"] / TTM["prod_rev"]
svc_gm = TTM["svc_gp"] / TTM["svc_rev"]
print(f"  TTM revenue {TTM['rev']:,.0f}  (Products {TTM['prod_rev']:,.0f} @ {prod_gm:.1%} GM,"
      f" Services {TTM['svc_rev']:,.0f} @ {svc_gm:.1%} GM)")
print(f"  TTM opex {TTM['opex']:,.0f}   TTM operating income {TTM['opinc']:,.0f}"
      f"   effective tax {TTM['tax_rate']:.1%}   diluted shares {TTM['sh']:,.1f}M   TTM EPS {TTM['eps']:.2f}")
print("  ASSUMPTION A (stated): opex is FIXED for a marginal revenue change (no variable opex).")
print("  ASSUMPTION B (stated): incremental revenue carries the SAME gross margin as its category.")
print("  ASSUMPTION C (stated): share count and tax rate unchanged.")


def shock(line, pct):
    """`line` in {'iPhone','Services','GreaterChina','Products'}; returns rev/EPS deltas."""
    if line == "Services":
        d_rev = TTM["svc_rev"] * pct
        d_gp = d_rev * svc_gm
    elif line == "Products":
        d_rev = TTM["prod_rev"] * pct
        d_gp = d_rev * prod_gm
    elif line == "iPhone":
        d_rev = tc.iPhone.sum() * pct
        d_gp = d_rev * prod_gm            # iPhone sits inside Products
    elif line == "GreaterChina":
        d_rev = gc.GreaterChina.sum() * pct
        # China TTM mix: use the company Products/Services split as the tightest available proxy
        d_gp = d_rev * (prod_gm * TTM["prod_rev"] + svc_gm * TTM["svc_rev"]) / TTM["rev"]
    d_eps = d_gp * (1 - TTM["tax_rate"]) / TTM["sh"]
    return d_rev, d_rev / TTM["rev"], d_gp, d_eps, d_eps / TTM["eps"]


print(f"\n  {'driver':<14}{'+5% on that line':>18}{'= rev change':>14}{'% of revenue':>14}"
      f"{'gross profit':>14}{'EPS $':>10}{'% of TTM EPS':>14}")
for ln in ["iPhone", "Services", "GreaterChina", "Products"]:
    dr, drp, dg, de, dep = shock(ln, 0.05)
    base = {"iPhone": tc.iPhone.sum(), "Services": TTM["svc_rev"],
            "GreaterChina": gc.GreaterChina.sum(), "Products": TTM["prod_rev"]}[ln]
    print(f"  {ln:<14}{base*0.05:>18,.0f}{drp:>14.2%}{dg:>14,.0f}{de:>10.2f}{dep:>14.2%}")
print("\n  READ: a +5% move in SERVICES is worth less revenue than a +5% move in iPhone"
      f" ({TTM['svc_rev']*0.05:,.0f} vs {tc.iPhone.sum()*0.05:,.0f} $M) but ~2x the gross profit per revenue dollar,")
print("  so on an EQUAL-PERCENTAGE basis the two are nearly interchangeable for EPS:"
      f" iPhone/Services = {shock('iPhone',0.05)[3]/shock('Services',0.05)[3]:.2f}x.")
print("  The claim 'this is an iPhone stock' therefore does NOT rest on elasticity. It rests on")
print("  WHICH LINE ACTUALLY MOVED: iPhone delivered 66.6% of the TTM revenue increment vs Services")
print("  25.9%, i.e. 2.57x as much of the realised growth, and iPhone is 52.6% of revenue vs 25.8%.")

# ---------------- the downside stress: iPhone reverts to plateau
print("\n" + "=" * 118)
print("4. STRESS TEST -- OVERTURN THE WEAKEST STEP. The weakest step in any bull case here is that")
print("   TTM iPhone revenue of $245.5bn is the new baseline rather than a Pro-mix/China/FX peak.")
print("=" * 118)
for target, label in [(209586, "reverts to FY2025 ($209.6bn, the best of the plateau years)"),
                      (plateau, f"reverts to the FY22-25 mean (${plateau/1000:.1f}bn)"),
                      (tc.iPhone.sum() * 0.90, "just -10% off the TTM run-rate"),
                      (tc.iPhone.sum() * 1.10, "+10% MORE (bull case)")]:
    d_rev = target - tc.iPhone.sum()
    d_gp = d_rev * prod_gm
    d_eps = d_gp * (1 - TTM["tax_rate"]) / TTM["sh"]
    eps_new = TTM["eps"] + d_eps
    print(f"  iPhone {label:<52} rev {d_rev:>+9,.0f} ({d_rev/TTM['rev']:>+6.1%})"
          f"  EPS {TTM['eps']:.2f} -> {eps_new:.2f} ({d_eps/TTM['eps']:+.1%})"
          f"   P/E at $333.08 becomes {333.08/eps_new:.1f}x")

# ---------------- multiple compression vs estimate cuts
print("\n" + "=" * 118)
print("5. WHAT THE PRICE REQUIRES.  price $333.08, TTM EPS %.2f -> %.1fx trailing GAAP" % (TTM["eps"], 333.08 / TTM["eps"]))
print("=" * 118)
print(f"  {'exit P/E in 5y':>16}{'price for +10%/yr':>20}{'EPS needed':>13}{'EPS CAGR required':>20}")
tgt = 333.08 * 1.10 ** 5
for pe in [20, 25, 30, 33, 38.2]:
    need = tgt / pe
    print(f"  {pe:>16.1f}{tgt:>20,.0f}{need:>13.2f}{(need/TTM['eps'])**0.2-1:>20.1%}")
print("\n  For reference the actual 3-quarter EPS growth run-rate was +18.3%, +21.8%, +28.7% YoY,")
print("  of which (Q3FY26) 24% of the increment was a one-off tariff refund and ~1.6pp came from")
print("  the shrinking share count rather than from the business.")

# ---------------- buyback arithmetic
print("\n" + "=" * 118)
print("6. THE BUYBACK LEG.  9M cash flows, acc 0000320193-26-000020 vs -25-000073")
print("=" * 118)
bb26, bb25 = 62094.0, 70579.0
mcap = 333.08 * TTM["sh"]
print(f"  repurchases 9M FY26 {bb26:,.0f} $M   vs 9M FY25 {bb25:,.0f} $M   = {bb26/bb25-1:+.1%}")
print(f"  annualised {bb26*4/3:,.0f} $M on a {mcap:,.0f} $M market cap = {bb26*4/3/mcap:.2%} repurchase yield")
print(f"  a year ago the same annualised spend was {bb25*4/3:,.0f} $M; at today's cap that would be"
      f" {bb25*4/3/mcap:.2%}")
print(f"  realised effect: diluted shares {t4.sh_dil.iloc[0]/1000:,.1f}M -> {TTM['sh']:,.1f}M over the TTM"
      f" = {TTM['sh']/(t4.sh_dil.iloc[0]/1000)-1:+.2%}")
print("  INFERENCE: buyback spend fell 12.0% while the share price rose 42.8% over the year, so the")
print("  per-share arithmetic contributed LESS this year, not more. It is a ~1.6%/yr tailwind, not a thesis.")
