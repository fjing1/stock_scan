"""_fd_AAPL_mix.py -- decompose the acceleration. Which lines and which geographies actually produced
the +16.4% quarter, and is the mix moving the way a "Services shift / AI supercycle" thesis assumes?

Sources, all primary:
  FY2025 10-K   accn 0000320193-25-000079, R38.htm  (FY2025/2024/2023 product revenue)
  Q3 FY2026 10-Q accn 0000320193-26-000020, R28.htm (Q3 and 9M FY2026 vs FY2025 product revenue)
                                            R46.htm (segment net sales, cost of sales, R&D)
Product-line revenue is dimensional XBRL (srt:ProductOrServiceAxis) and is therefore NOT in the
companyfacts API; these R-files rendered from the same instance documents are the primary source.
"""
import pandas as pd

pd.set_option("display.width", 200)

# ---- FY totals, 10-K R38 (accn 0000320193-25-000079)
FY = pd.DataFrame({
    "FY2023": {"iPhone": 200583, "Mac": 29357, "iPad": 28300, "Wearables": 39845, "Services": 85200},
    "FY2024": {"iPhone": 201183, "Mac": 29984, "iPad": 26694, "Wearables": 37005, "Services": 96169},
    "FY2025": {"iPhone": 209586, "Mac": 33708, "iPad": 28023, "Wearables": 35686, "Services": 109158},
})
# ---- Q3 and 9M, 10-Q R28 (accn 0000320193-26-000020)
Q3 = pd.DataFrame({
    "Q3FY25": {"iPhone": 44582, "Mac": 8046, "iPad": 6581, "Wearables": 7404, "Services": 27423},
    "Q3FY26": {"iPhone": 54252, "Mac": 10352, "iPad": 6191, "Wearables": 7883, "Services": 30739},
})
NM = pd.DataFrame({
    "9M_FY25": {"iPhone": 160561, "Mac": 24982, "iPad": 21071, "Wearables": 26673, "Services": 80408},
    "9M_FY26": {"iPhone": 196515, "Mac": 27137, "iPad": 21700, "Wearables": 27277, "Services": 91728},
})


def decomp(df, a, b, label):
    t = df.copy()
    t["growth_$M"] = t[b] - t[a]
    t["growth_%"] = (t[b] / t[a] - 1) * 100
    tot_a, tot_b = t[a].sum(), t[b].sum()
    t["share_of_total_growth_%"] = t["growth_$M"] / (tot_b - tot_a) * 100
    t[f"mix_{a}_%"] = t[a] / tot_a * 100
    t[f"mix_{b}_%"] = t[b] / tot_b * 100
    t["mix_change_pp"] = t[f"mix_{b}_%"] - t[f"mix_{a}_%"]
    print("\n" + "=" * 132)
    print(f"{label}   total {tot_a:,.0f} -> {tot_b:,.0f} = {(tot_b/tot_a-1)*100:+.2f}%")
    print("=" * 132)
    print(t.sort_values("growth_$M", ascending=False).to_string(float_format=lambda x: f"{x:,.2f}"))
    return t


decomp(FY, "FY2024", "FY2025", "FISCAL 2025 vs 2024 (the +6.4% year the 39.5x multiple was quoted against)")
decomp(NM, "9M_FY25", "9M_FY26", "NINE MONTHS FY2026 vs FY2025 (the current acceleration)")
decomp(Q3, "Q3FY25", "Q3FY26", "LATEST QUARTER (the +16.4% headline)")

print("\n" + "=" * 132)
print("THE TWO CLAIMS A 'SERVICES MIX SHIFT' THESIS MAKES, TESTED")
print("=" * 132)
for lbl, a, b, df in [("FY2024->FY2025", "FY2024", "FY2025", FY),
                      ("9M FY25->9M FY26", "9M_FY25", "9M_FY26", NM),
                      ("Q3 FY25->Q3 FY26", "Q3FY25", "Q3FY26", Q3)]:
    sa, sb = df.loc["Services", a] / df[a].sum() * 100, df.loc["Services", b] / df[b].sum() * 100
    ia, ib = df.loc["iPhone", a] / df[a].sum() * 100, df.loc["iPhone", b] / df[b].sum() * 100
    print(f"  {lbl:18s} Services share {sa:5.2f}% -> {sb:5.2f}% ({sb-sa:+5.2f}pp)   "
          f"iPhone share {ia:5.2f}% -> {ib:5.2f}% ({ib-ia:+5.2f}pp)")

print("\n  iPhone revenue growth by period:")
print(f"    FY2024 {(FY.loc['iPhone','FY2024']/FY.loc['iPhone','FY2023']-1)*100:+6.2f}%   "
      f"FY2025 {(FY.loc['iPhone','FY2025']/FY.loc['iPhone','FY2024']-1)*100:+6.2f}%   "
      f"9M FY2026 {(NM.loc['iPhone','9M_FY26']/NM.loc['iPhone','9M_FY25']-1)*100:+6.2f}%   "
      f"Q3 FY2026 {(Q3.loc['iPhone','Q3FY26']/Q3.loc['iPhone','Q3FY25']-1)*100:+6.2f}%")
print("  Services revenue growth by period:")
print(f"    FY2024 {(FY.loc['Services','FY2024']/FY.loc['Services','FY2023']-1)*100:+6.2f}%   "
      f"FY2025 {(FY.loc['Services','FY2025']/FY.loc['Services','FY2024']-1)*100:+6.2f}%   "
      f"9M FY2026 {(NM.loc['Services','9M_FY26']/NM.loc['Services','9M_FY25']-1)*100:+6.2f}%   "
      f"Q3 FY2026 {(Q3.loc['Services','Q3FY26']/Q3.loc['Services','Q3FY25']-1)*100:+6.2f}%")

# ---- geography and cost structure, R46
GEO = pd.DataFrame({
    "9M_FY25_sales": {"Americas": 134161, "Europe": 82329, "GreaterChina": 49884, "Japan": 22067, "RestAsiaPac": 25254},
    "9M_FY26_sales": {"Americas": 149403, "Europe": 95596, "GreaterChina": 64839, "Japan": 24368, "RestAsiaPac": 30151},
    "9M_FY25_cos": {"Americas": 71763, "Europe": 43447, "GreaterChina": 27523, "Japan": 10698, "RestAsiaPac": 13404},
    "9M_FY26_cos": {"Americas": 76470, "Europe": 47549, "GreaterChina": 34434, "Japan": 12088, "RestAsiaPac": 15034},
})
GEO["sales_g_%"] = (GEO["9M_FY26_sales"] / GEO["9M_FY25_sales"] - 1) * 100
GEO["growth_$M"] = GEO["9M_FY26_sales"] - GEO["9M_FY25_sales"]
GEO["share_of_growth_%"] = GEO["growth_$M"] / GEO["growth_$M"].sum() * 100
GEO["GM_FY25_%"] = (1 - GEO["9M_FY25_cos"] / GEO["9M_FY25_sales"]) * 100
GEO["GM_FY26_%"] = (1 - GEO["9M_FY26_cos"] / GEO["9M_FY26_sales"]) * 100
GEO["GM_chg_pp"] = GEO["GM_FY26_%"] - GEO["GM_FY25_%"]
print("\n" + "=" * 132)
print("GEOGRAPHY, 9 MONTHS FY2026 vs FY2025 (segment note, R46) -- where the growth and the margin came from")
print("=" * 132)
print(GEO[["9M_FY25_sales", "9M_FY26_sales", "sales_g_%", "growth_$M", "share_of_growth_%",
           "GM_FY25_%", "GM_FY26_%", "GM_chg_pp"]]
      .sort_values("growth_$M", ascending=False).to_string(float_format=lambda x: f"{x:,.2f}"))

print("\n" + "=" * 132)
print("COST STRUCTURE: is the margin gain being reinvested? (R46, 9 months)")
print("=" * 132)
cost = {"Net sales": (313695, 364357), "Cost of sales": (166835, 185575),
        "Research and development": (25684, 34035), "Selling and marketing": (14601, 15632),
        "General and administrative": (5952, 6683), "Operating income": (100623, 122432)}
for k, (a, b) in cost.items():
    print(f"  {k:28s} {a:>9,.0f} -> {b:>9,.0f}  {(b/a-1)*100:+7.2f}%   "
          f"% of sales {a/313695*100:5.2f}% -> {b/364357*100:5.2f}%  "
          f"({b/364357*100 - a/313695*100:+5.2f}pp)")
print("\n  R&D is growing at 2.0x the rate of revenue (+32.5% vs +16.2%), lifting R&D from 8.19% to")
print("  9.34% of sales (+1.15pp). Gross margin gained 2.25pp over the same nine months (cost of")
print("  sales 53.18% -> 50.93%), so ~51% of the gross-margin gain is consumed by R&D reinvestment")
print("  before it reaches operating income. Operating margin gained only 1.53pp of the 2.25pp.")
