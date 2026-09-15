"""
_fd_AAPL_margin_mix.py — the price-vs-units question, bounded.

Apple stopped disclosing iPhone UNITS in FY2019 (accn 0000320193-18-000145 was the last 10-K with
them), so "is the +21.7% iPhone growth price or volume" is NOT DISCLOSED and cannot be computed. What
CAN be computed is a bound, from the one thing Apple does still split: cost of sales by
Products vs Services.

  * If the acceleration were mostly PRICE (mix shift to higher-priced models, or list-price
    increases) with roughly unchanged unit cost, Products gross margin would EXPAND.
  * If it were mostly UNITS at similar prices, Products gross margin would be roughly FLAT.

That is a real, falsifiable discriminator built entirely from filed facts. It does not identify the
split, but it bounds which story is consistent with the margins.

Also computed: Products vs Services contribution to GROSS PROFIT growth, which is what actually
drives EPS -- revenue mix and gross-profit mix are different things and the second one is the one
that matters.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)
HERE = Path(__file__).resolve().parent
D = pd.read_csv(HERE / "_fd_AAPL_dim_facts.csv", parse_dates=["start", "end"])
D["dims"] = D["dims"].fillna("")
REV = "RevenueFromContractWithCustomerExcludingAssessedTax"
COGS = "CostOfGoodsAndServicesSold"


def pan(tag, axis="ProductOrServiceAxis", lo=80, hi=100):
    d = D[(D.tag == tag) & D.dims.str.contains(axis) & D.days.between(lo, hi)].copy()
    d = d[d.dims.str.count("=") == 1]
    d["k"] = d.dims.str.extract(rf"{axis}=([A-Za-z]+)")
    return (d.groupby(["end", "k"]).val.max().unstack() / 1e6).sort_index()


R, Cg = pan(REV), pan(COGS)
GP = (R[["ProductMember", "ServiceMember"]] - Cg[["ProductMember", "ServiceMember"]]).dropna()
GM = GP / R[["ProductMember", "ServiceMember"]]
tot_r = R["ProductMember"] + R["ServiceMember"]
tot_gp = GP.sum(axis=1)

print("=" * 118)
print("季度毛利率：硬件 vs 服务（全部来自 10-Q/10-K 的 ProductOrServiceAxis 维度事实）")
print("=" * 118)
out = pd.DataFrame({
    "硬件营收": R["ProductMember"], "硬件毛利": GP["ProductMember"], "硬件毛利率": GM["ProductMember"],
    "服务营收": R["ServiceMember"], "服务毛利": GP["ServiceMember"], "服务毛利率": GM["ServiceMember"],
    "合并毛利率": tot_gp / tot_r,
}).dropna()
print(out.to_string(float_format=lambda x: f"{x:,.1%}" if abs(x) < 5 else f"{x:,.0f}"))

print("\n" + "=" * 118)
print("同比：毛利率的变化（这是「涨价 vs 卖更多台」的判别式）")
print("=" * 118)
for dt in out.index:
    tgt = dt - pd.Timedelta(days=365)
    cand = [x for x in out.index if abs((x - tgt).days) <= 25]
    if not cand:
        continue
    b = cand[0]
    a_, b_ = out.loc[dt], out.loc[b]
    print(f"  {dt.date()} vs {b.date()}   硬件毛利率 {b_['硬件毛利率']:.2%}→{a_['硬件毛利率']:.2%} "
          f"({(a_['硬件毛利率']-b_['硬件毛利率'])*100:+.2f}pts)   "
          f"服务毛利率 {b_['服务毛利率']:.2%}→{a_['服务毛利率']:.2%} "
          f"({(a_['服务毛利率']-b_['服务毛利率'])*100:+.2f}pts)   "
          f"合并 {b_['合并毛利率']:.2%}→{a_['合并毛利率']:.2%} "
          f"({(a_['合并毛利率']-b_['合并毛利率'])*100:+.2f}pts)")

print("\n" + "=" * 118)
print("毛利润的增量归因（EPS 靠的是毛利，不是营收）")
print("=" * 118)
for dt in out.index:
    cand = [x for x in out.index if abs((x - (dt - pd.Timedelta(days=365))).days) <= 25]
    if not cand:
        continue
    b = cand[0]
    dgp_p = out.loc[dt, "硬件毛利"] - out.loc[b, "硬件毛利"]
    dgp_s = out.loc[dt, "服务毛利"] - out.loc[b, "服务毛利"]
    tot = dgp_p + dgp_s
    dr_p = out.loc[dt, "硬件营收"] - out.loc[b, "硬件营收"]
    dr_s = out.loc[dt, "服务营收"] - out.loc[b, "服务营收"]
    print(f"  {dt.date()}  毛利增量 {tot:+,.0f}M ({tot/(out.loc[b,'硬件毛利']+out.loc[b,'服务毛利']):+.1%})"
          f"   硬件 {dgp_p:+,.0f}M ({dgp_p/tot:+.0%})   服务 {dgp_s:+,.0f}M ({dgp_s/tot:+.0%})"
          f"   |  营收增量占比 硬件 {dr_p/(dr_p+dr_s):+.0%} 服务 {dr_s/(dr_p+dr_s):+.0%}")

print("\n  服务占毛利的比重 vs 占营收的比重")
sh = pd.DataFrame({"服务占营收": out["服务营收"] / (out["硬件营收"] + out["服务营收"]),
                   "服务占毛利": out["服务毛利"] / (out["硬件毛利"] + out["服务毛利"])})
print(sh.to_string(float_format=lambda x: f"{x:.1%}"))

print("\n" + "=" * 118)
print("单位数披露的检验：Apple 现在还披露 iPhone 台数吗")
print("=" * 118)
print("  FACT: 本次抓取的 8 份 10-Q/10-K 的 XBRL 实例文件中，不存在任何 iPhone 单位数（units）事实。")
print("  Apple 自 FY2019 起停止披露分产品单位数与 ASP。因此「+21.7% 里多少是涨价、多少是台数」")
print("  在申报文件中 NOT DISCLOSED，任何拆分都是估算。可得的最紧界就是上面的毛利率变化。")

print("\n" + "=" * 118)
print("经营杠杆：营业利润率与 R&D")
print("=" * 118)
oi = D[(D.tag == "OperatingIncomeLoss") & (D.dims == "") & D.days.between(80, 100)]
rd = D[(D.tag == "ResearchAndDevelopmentExpense") & (D.dims == "") & D.days.between(80, 100)]
o = (oi.groupby("end").val.max() / 1e6).sort_index()
r_ = (rd.groupby("end").val.max() / 1e6).sort_index()
t = pd.DataFrame({"营收": tot_r, "营业利润": o, "营业利润率": o / tot_r,
                  "R&D": r_, "R&D/营收": r_ / tot_r}).dropna()
print(t.to_string(float_format=lambda x: f"{x:.1%}" if abs(x) < 5 else f"{x:,.0f}"))
for dt in t.index:
    cand = [x for x in t.index if abs((x - (dt - pd.Timedelta(days=365))).days) <= 25]
    if cand:
        b = cand[0]
        print(f"  {dt.date()} 营业利润率 {t.loc[b,'营业利润率']:.2%}→{t.loc[dt,'营业利润率']:.2%} "
              f"({(t.loc[dt,'营业利润率']-t.loc[b,'营业利润率'])*100:+.2f}pts)   "
              f"营业利润同比 {t.loc[dt,'营业利润']/t.loc[b,'营业利润']-1:+.1%}   "
              f"营收同比 {t.loc[dt,'营收']/t.loc[b,'营收']-1:+.1%}   "
              f"R&D/营收 {t.loc[b,'R&D/营收']:.2%}→{t.loc[dt,'R&D/营收']:.2%}")
