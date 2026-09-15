"""
_fd_AAPL_segments2.py — clean decomposition of AAPL's revenue acceleration.

Fixes two things in _fd_AAPL_segments.py:
  1. DOUBLE COUNTING. `ProductMember` IS iPhone+Mac+iPad+Wearables (verified to the dollar), so
     summing all six ProductOrServiceAxis members counts hardware twice and every "% of increment"
     printed there is roughly halved. Total revenue = ProductMember + ServiceMember, full stop.
  2. The FY2026 10-Qs tag geographic segments with TWO dimensions
     (ConsolidationItemsAxis=OperatingSegmentsMember ; StatementBusinessSegmentsAxis=...), so a
     "single dimension only" filter silently drops the three most recent quarters.

Then: proper quarterly OCF from the cumulative facts, and a 2-year stacked growth rate to separate
genuine acceleration from lapping a weak base.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 260)
HERE = Path(__file__).resolve().parent

D = pd.read_csv(HERE / "_fd_AAPL_dim_facts.csv", parse_dates=["start", "end"])
D["dims"] = D["dims"].fillna("")
REV = "RevenueFromContractWithCustomerExcludingAssessedTax"


def q_panel(axis: str, pat: str) -> pd.DataFrame:
    d = D[(D.tag == REV) & D.dims.str.contains(axis, na=False) & D.days.between(80, 100)].copy()
    d["k"] = d.dims.str.extract(rf"{axis}=([A-Za-z]+)")
    d = d.dropna(subset=["k"])
    return (d.groupby(["end", "k"]).val.max().unstack() / 1e6).sort_index()


def yoy_tbl(piv: pd.DataFrame, lag_days=365, tol=25) -> pd.DataFrame:
    out = pd.DataFrame(index=piv.index, columns=piv.columns, dtype=float)
    for dt in piv.index:
        tgt = dt - pd.Timedelta(days=lag_days)
        cand = [x for x in piv.index if abs((x - tgt).days) <= tol]
        if cand:
            out.loc[dt] = piv.loc[dt] / piv.loc[cand[0]] - 1
    return out.dropna(how="all")


prod = q_panel("ProductOrServiceAxis", "")
print("=" * 130)
print("完整性校验：ProductMember 是否等于 iPhone+Mac+iPad+Wearables，以及 Product+Service 是否等于合并营收")
comp = ["IPhoneMember", "MacMember", "IPadMember", "WearablesHomeandAccessoriesMember"]
chk = pd.DataFrame({
    "ProductMember": prod["ProductMember"],
    "四项之和": prod[comp].sum(axis=1),
    "差": prod["ProductMember"] - prod[comp].sum(axis=1),
    "Product+Service": prod["ProductMember"] + prod["ServiceMember"],
})
print(chk.to_string(float_format=lambda x: f"{x:,.0f}"))

TOT = prod["ProductMember"] + prod["ServiceMember"]
print("\n" + "=" * 130)
print("正确的两分法：硬件 vs 服务（$M，季度）")
two = pd.DataFrame({"合并营收": TOT, "硬件Product": prod["ProductMember"],
                    "服务Services": prod["ServiceMember"],
                    "服务占比": prod["ServiceMember"] / TOT})
print(two.to_string(float_format=lambda x: f"{x:,.3f}" if x < 1 else f"{x:,.0f}"))

print("\n服务占营收比重（%）—— 论点「Services 占比提升」的直接检验")
print((prod["ServiceMember"] / TOT * 100).round(2).to_string())

print("\n" + "=" * 130)
print("同比增速（date-matched）")
allc = prod.copy()
allc["合并"] = TOT
print(yoy_tbl(allc).to_string(float_format=lambda x: f"{x*100:+.1f}%"))

print("\n" + "=" * 130)
print("对合并营收同比增量的贡献（不重复计数：iPhone/Mac/iPad/Wearables/Services 五项，和 = 100%）")
lines = ["IPhoneMember", "MacMember", "IPadMember", "WearablesHomeandAccessoriesMember", "ServiceMember"]
for dt in TOT.index:
    tgt = dt - pd.Timedelta(days=365)
    cand = [x for x in TOT.index if abs((x - tgt).days) <= 25]
    if not cand:
        continue
    b = cand[0]
    tot_d = TOT[dt] - TOT[b]
    d_abs = prod.loc[dt, lines] - prod.loc[b, lines]
    assert abs(d_abs.sum() - tot_d) < 1.0, (d_abs.sum(), tot_d)
    print(f"  {dt.date()} vs {b.date()}  合并 {TOT[b]:,.0f}→{TOT[dt]:,.0f} = {tot_d:+,.0f}M ({tot_d/TOT[b]*100:+.2f}%)")
    for k in lines:
        v = d_abs[k]
        print(f"      {k:<36}{v:>+9,.0f}M   占增量 {v/tot_d*100:>+6.1f}%   "
              f"该项同比 {prod.loc[dt,k]/prod.loc[b,k]-1:>+7.2%}")

print("\n" + "=" * 130)
print("两年叠加增速（年化）—— 分离「真加速」与「上年基数低」")
st = pd.DataFrame(index=TOT.index, columns=["合并1y", "合并2y年化", "iPhone1y", "iPhone2y年化",
                                            "Services1y", "Services2y年化"], dtype=float)
for dt in TOT.index:
    for lag, suf in ((365, "1y"), (730, "2y年化")):
        cand = [x for x in TOT.index if abs((x - (dt - pd.Timedelta(days=lag))).days) <= 25]
        if not cand:
            continue
        b = cand[0]
        p = 1 if lag == 365 else 2
        st.loc[dt, f"合并{suf}"] = (TOT[dt] / TOT[b]) ** (1 / p) - 1
        st.loc[dt, f"iPhone{suf}"] = (prod.loc[dt, "IPhoneMember"] / prod.loc[b, "IPhoneMember"]) ** (1 / p) - 1
        st.loc[dt, f"Services{suf}"] = (prod.loc[dt, "ServiceMember"] / prod.loc[b, "ServiceMember"]) ** (1 / p) - 1
print(st.dropna(how="all").to_string(float_format=lambda x: f"{x*100:+.1f}%"))

# ---------------- geography, both 1-dim and 2-dim taggings
g = D[(D.tag == REV) & D.dims.str.contains("StatementBusinessSegmentsAxis", na=False)
      & D.days.between(80, 100)].copy()
g["seg"] = g.dims.str.extract(r"StatementBusinessSegmentsAxis=([A-Za-z]+)")
gp = (g.groupby(["end", "seg"]).val.max().unstack() / 1e6).sort_index()
print("\n" + "=" * 130)
print("地区分部 季度营收 ($M)  —— 含双维度标记的 FY2026 各季")
print(gp.to_string(float_format=lambda x: f"{x:,.0f}"))
print("\n地区同比")
print(yoy_tbl(gp).to_string(float_format=lambda x: f"{x*100:+.1f}%"))
print("\n地区占比")
print((gp.div(gp.sum(axis=1), axis=0) * 100).round(1).to_string())
print("\n各地区对合并增量的贡献")
for dt in gp.index:
    cand = [x for x in gp.index if abs((x - (dt - pd.Timedelta(days=365))).days) <= 25]
    if not cand:
        continue
    b = cand[0]
    d_abs = (gp.loc[dt] - gp.loc[b]).dropna()
    tot_d = d_abs.sum()
    print(f"  {dt.date()} vs {b.date()} 总 {tot_d:+,.0f}M  " +
          "  ".join(f"{k.replace('SegmentMember','')}:{v:+,.0f}({v/tot_d*100:+.0f}%)" for k, v in d_abs.items()))

# ---------------- OCF quarterly from cumulative
o = D[(D.tag == "NetCashProvidedByUsedInOperatingActivities") & (D.dims == "")].dropna(subset=["days"])
o = o.drop_duplicates(subset=["start", "end", "days"]).sort_values(["start", "days"])
rows = []
for fs, grp in o.groupby("start"):
    prev = 0.0
    for r in grp.itertuples():
        rows.append({"q_end": r.end, "ocf_q": r.val - prev, "cum_days": r.days})
        prev = r.val
q = pd.DataFrame(rows).sort_values("q_end").drop_duplicates(subset=["q_end"], keep="last")
print("\n" + "=" * 130)
print("季度 OCF（由累计事实差分推导，$M）")
print(q.assign(ocf_q=lambda x: x.ocf_q / 1e6).to_string(index=False, float_format=lambda x: f"{x:,.0f}"))
# TTM OCF: FY2025 annual + (FY2026 9M - FY2025 9M) -- the arithmetically safe route
fy25 = float(o[(o.days == 363) & (o.end == pd.Timestamp("2025-09-27"))].val.iloc[0])
m9_26 = float(o[(o.days == 272) & (o.end == pd.Timestamp("2026-06-27"))].val.iloc[0])
m9_25 = float(o[(o.days == 272) & (o.end == pd.Timestamp("2025-06-28"))].val.iloc[0])
ttm_ocf = fy25 + m9_26 - m9_25
print(f"\n  TTM OCF = FY2025({fy25/1e6:,.0f}) + 9M FY26({m9_26/1e6:,.0f}) - 9M FY25({m9_25/1e6:,.0f})"
      f" = ${ttm_ocf/1e6:,.0f}M")
print(f"  9个月 OCF 同比 {m9_26/m9_25-1:+.1%}")
TTM_REV = float(TOT.iloc[-4:].sum())
print(f"  TTM 营收(由季度加总) ${TTM_REV:,.0f}M   TTM OCF/营收 {ttm_ocf/1e6/TTM_REV:.1%}")

# ---------------- capex / buyback / dividend, 9M comparisons
for tag, cn in (("PaymentsToAcquirePropertyPlantAndEquipment", "capex"),
                ("PaymentsForRepurchaseOfCommonStock", "回购"),
                ("PaymentsOfDividendsCommonStock", "股息"),
                ("ShareBasedCompensation", "SBC")):
    s = D[(D.tag == tag) & (D.dims == "")].dropna(subset=["days"]).drop_duplicates(subset=["start", "end", "days"])
    if s.empty:
        continue
    a = s[s.days == 272].sort_values("end")
    print(f"\n{cn} ({tag}) 9个月口径")
    print(a[["start", "end", "val"]].assign(val=lambda x: x.val / 1e6)
          .to_string(index=False, float_format=lambda x: f"{x:,.0f}"))
    if len(a) >= 2:
        print(f"  9M 同比 {float(a.val.iloc[-1])/float(a.val.iloc[-2])-1:+.1%}")
