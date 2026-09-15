"""
_fd_AAPL_checkpoints.py — build the DATED, numeric checkpoints, and stress the covered-call
conclusion against the one assumption it rests on (the model's drift).

Two jobs:
  1. Derive Q4 FY2025 product-category revenue by subtraction (FY total minus Q1+Q2+Q3), because the
     10-K discloses product-category revenue only at the FISCAL YEAR level -- so the quarterly
     segment panel is missing every Q4 and there is otherwise no baseline for the October print.
  2. The covered-call verdict ("gives up 0.5-1.5% of expected 21-day return") depends on move_prob's
     fitted drift of +1.04% per 21 days (~13%/yr). That is a large assumption. Re-price the same
     structures with the drift zeroed out and with it halved, and report the break-even drift.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import _fund_data as F
import move_prob as M

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)
HERE = Path(__file__).resolve().parent
REV = "RevenueFromContractWithCustomerExcludingAssessedTax"

D = pd.read_csv(HERE / "_fd_AAPL_dim_facts.csv", parse_dates=["start", "end"])
D["dims"] = D["dims"].fillna("")

# ---------------- FY-level product revenue
fy = D[(D.tag == REV) & D.dims.str.contains("ProductOrServiceAxis") & D.days.between(355, 375)].copy()
fy["k"] = fy.dims.str.extract(r"ProductOrServiceAxis=([A-Za-z]+)")
fy = fy[fy.dims.str.count("=") == 1]
FY = (fy.groupby(["end", "k"]).val.max().unstack() / 1e6).sort_index()
print("=" * 120)
print("产品类别 财年营收 ($M)  —— 10-K 只在财年层面披露产品线，这是推导 Q4 的唯一来源")
print(FY.to_string(float_format=lambda x: f"{x:,.0f}"))

q = D[(D.tag == REV) & D.dims.str.contains("ProductOrServiceAxis") & D.days.between(80, 100)].copy()
q["k"] = q.dims.str.extract(r"ProductOrServiceAxis=([A-Za-z]+)")
q = q[q.dims.str.count("=") == 1]
Q = (q.groupby(["end", "k"]).val.max().unstack() / 1e6).sort_index()

print("\n" + "=" * 120)
print("推导 Q4（9月季）产品线营收 = 财年 − 该财年 Q1..Q3")
for y_end in FY.index:
    y_start = y_end - pd.Timedelta(days=370)
    ins = Q[(Q.index > y_start) & (Q.index <= y_end)]
    if len(ins) != 3:
        print(f"  {y_end.date()}: 只有 {len(ins)} 个季度可用，跳过")
        continue
    q4 = FY.loc[y_end] - ins.sum()
    print(f"  Q4 截至 {y_end.date()}: " + "  ".join(f"{k}={v:,.0f}" for k, v in q4.items()
                                                   if pd.notna(v)))
    globals()[f"Q4_{y_end.year}"] = q4

# ---------------- consolidated Q4 baselines from the panel
panel = F.load()
rev_a = F.series(panel, "AAPL", "revenue", annual=True)
rev_q = F.series(panel, "AAPL", "revenue", annual=False)
eps_a = F.series(panel, "AAPL", "eps_diluted", annual=True)
eps_q = F.series(panel, "AAPL", "eps_diluted", annual=False)


def q4_of(a, qq, y_end):
    ins = qq[(qq.end > y_end - pd.Timedelta(days=370)) & (qq.end <= y_end)]
    av = a[a.end == y_end]
    if len(ins) != 3 or av.empty:
        return np.nan
    return float(av.val.iloc[0]) - float(ins.val.sum())


for y in ("2024-09-28", "2025-09-27"):
    ye = pd.Timestamp(y)
    print(f"\n  合并 Q4 截至 {y}: 营收 ${q4_of(rev_a, rev_q, ye)/1e6:,.0f}M   "
          f"摊薄EPS ${q4_of(eps_a, eps_q, ye):.2f}   "
          f"(来源 10-K {rev_a[rev_a.end==ye].accn.iloc[0]})")

# ---------------- consensus anchor
import yfinance as yf
cal = yf.Ticker("AAPL").calendar
q4_25_rev = q4_of(rev_a, rev_q, pd.Timestamp("2025-09-27"))
q4_25_eps = q4_of(eps_a, eps_q, pd.Timestamp("2025-09-27"))
print("\n" + "=" * 120)
print("下一次财报的预期锚（yfinance 聚合的分析师一致预期 —— 标注为 commentary, unverified，"
      "\n不是申报文件；但它是「预期变化」的唯一可量化锚点）")
print(f"  日期 {cal['Earnings Date']}")
print(f"  营收 一致预期 ${cal['Revenue Average']/1e6:,.0f}M  (区间 "
      f"${cal['Revenue Low']/1e6:,.0f}–{cal['Revenue High']/1e6:,.0f}M)")
print(f"  对应同比 = {cal['Revenue Average']/1e6/q4_25_rev*1e6*0+cal['Revenue Average']/(q4_25_rev)-1:+.1%}"
      f"   （基期 = 推导的 FQ4 FY2025 ${q4_25_rev/1e6:,.0f}M）")
print(f"  EPS 一致预期 ${cal['Earnings Average']:.2f} (区间 {cal['Earnings Low']:.2f}–"
      f"{cal['Earnings High']:.2f})  对应同比 {cal['Earnings Average']/q4_25_eps-1:+.1%}")
print(f"  ⇒ 市场已经在预期营收增速从 +16.4% 减速到 {cal['Revenue Average']/q4_25_rev-1:+.1%}。")
print("    这很关键：+16.4% 并没有被外推。要「超预期」，10/29 的营收必须高于 "
      f"${cal['Revenue Average']/1e6:,.0f}M。")

# ---------------- drift sensitivity on the covered call
print("\n" + "=" * 120)
print("备兑结论的漂移敏感性（结论原本依赖 move_prob 拟合的 21日 +1.04% 漂移 ≈ 13%/年）")
d = yf.download("AAPL", period="max", progress=False, auto_adjust=True)
if isinstance(d.columns, pd.MultiIndex):
    d.columns = d.columns.get_level_values(0)
d = d[d.Close > 0].dropna(subset=["Close"])
C, H, L, V = d.Close, d.High, d.Low, d.Volume
px = float(C.iloc[-1])
model = pd.read_pickle(M.MODEL_PATH)
gsp = model["groups"][M.asset_class("AAPL")]["per_h"][21]
ft = M.build_features(C, H, L, None, V)
sig = float(np.exp(M._design(ft.iloc[[-1]], gsp["cols"]) @ gsp["beta"]).item()) * np.sqrt(21)
z = np.asarray(gsp["z"])
rng = np.random.default_rng(11)
zz = rng.choice(z, 400000, replace=True)
base_lr = zz * sig
mu0 = base_lr.mean()
print(f"  经验 z 表在 21日 sigma={sig:.4f} 下的隐含漂移 = {np.exp(mu0)-1:+.3%} / 21日 "
      f"= {(np.exp(mu0)**(252/21)-1)*100:+.1f}%/年")
CALLS = {335: 9.40, 340: 7.08, 350: 3.72, 360: 1.80}     # 2026-10-16 mids, 32d, scaled below
print(f"\n  {'情形':<22}{'纯持股E':>10}" + "".join(f"{'CC K='+str(k):>12}" for k in CALLS))
for label, shift in (("模型原样", 0.0), ("漂移减半", -mu0 / 2), ("漂移=0", -mu0),
                     ("漂移=-5%/年", -mu0 + np.log(0.95) * 21 / 252)):
    lrs = base_lr + shift
    ST = px * np.exp(lrs)
    ev_s = np.mean(ST / px - 1)
    row = f"  {label:<22}{ev_s*100:>9.2f}%"
    for k, prem in CALLS.items():
        p = prem * 21 / 32
        row += f"{(np.mean(np.minimum(ST, k) + p) / px - 1)*100:>11.2f}%"
    print(row)
print("\n  break-even 漂移（备兑期望 = 纯持股期望）")
for k, prem in CALLS.items():
    p = prem * 21 / 32
    lo, hi = -0.20, 0.20
    for _ in range(60):
        mid_s = (lo + hi) / 2
        ST = px * np.exp(base_lr - mu0 + mid_s)
        diff = np.mean(np.minimum(ST, k) + p) - np.mean(ST)
        if diff > 0:
            lo = mid_s
        else:
            hi = mid_s
    ann = (np.exp(mid_s) ** (252 / 21) - 1) * 100
    print(f"    K={k}: 当 21日漂移 < {np.exp(mid_s)-1:+.2%} (年化 {ann:+.1f}%) 时备兑才不吃亏")
