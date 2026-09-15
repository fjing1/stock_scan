"""
_fd_AAPL_forward_matrix.py — the last three pieces.

1. FORWARD-RETURN MATRIX. Invert the "required growth" table into the form a holder actually thinks
   in: given an EPS growth path and an exit multiple, what is the annualised price return? The
   anchors are AAPL's OWN measured point-in-time multiple distribution (5-year median 30.6x, range
   20.1-41.6x from _fd_AAPL_pe_history2.py), not a guess.

2. SHARE COUNT AND CAPITAL RETURN, verified. The brief asserts "buybacks shrink the count 1.7%/yr".
   Check it against the diluted-share series and against the actual cash spent, because the buyback
   run-rate is DOWN 12% year over year on a 9-month basis (_fd_AAPL_segments2.py) and that changes
   the forward shrink rate.

3. RELATIVE STRENGTH, recomputed rather than inherited, including the vs-MSFT claim.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import yfinance as yf

import _fund_data as F

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)
SYM = "AAPL"
panel = F.load()
PE_NOW, EPS_NOW, PX_NOW = 38.24, 8.71, 333.08
PE5_MED, PE5_LO, PE5_HI = 30.6, 20.1, 41.6
PE1_MED = 35.9

print("=" * 116)
print("【1】三年后的价格年化回报 = EPS 增速 × 倍数变化   （起点 PE 38.2x，TTM EPS 8.71）")
print(f"  倍数锚点全部来自本仓库重算的 point-in-time 序列：近5年中位 {PE5_MED}x，"
      f"区间 {PE5_LO}–{PE5_HI}x，近1年中位 {PE1_MED}x")
print("=" * 116)
exits = [(PE5_LO, "5年最低 20.1x"), (24.0, "24x"), (PE5_MED, f"5年中位 {PE5_MED}x"),
         (33.6, "3年中位 33.6x"), (PE1_MED, f"近1年中位 {PE1_MED}x"),
         (PE_NOW, "不变 38.2x"), (PE5_HI, f"5年最高 {PE5_HI}x")]
growths = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
print(f"  {'3年后PE':<16}" + "".join(f"{'EPS'+f'{g:+.0%}':>11}" for g in growths))
for pe_e, lbl in exits:
    row = f"  {lbl:<16}"
    for g in growths:
        tot = (1 + g) ** 3 * (pe_e / PE_NOW)
        row += f"{(tot ** (1/3) - 1)*100:>10.1f}%"
    print(row)
print("\n  同一张表的总回报（3年累计，不年化）")
print(f"  {'3年后PE':<16}" + "".join(f"{'EPS'+f'{g:+.0%}':>11}" for g in growths))
for pe_e, lbl in exits:
    row = f"  {lbl:<16}"
    for g in growths:
        row += f"{((1+g)**3*(pe_e/PE_NOW)-1)*100:>10.1f}%"
    print(row)
print("\n  参照坐标：TTM EPS 同比 +32.2%（拆股调整、含推导Q4，来自 10-Q 0000320193-26-000020 及"
      "\n  10-K 0000320193-25-000079）。分析师对 FQ4 一致预期隐含 EPS +7.7% / 营收 +10.9%。")
print("  ⇒ 若倍数只回到近5年中位 30.6x，即使 EPS 三年年复合 +12%，价格年化只有 "
      f"{((1.12**3*(PE5_MED/PE_NOW))**(1/3)-1)*100:.1f}%。")
print(f"  ⇒ 要跑出 +12%/年，在倍数不变的前提下 EPS 也必须 +12%/年；倍数回中位则需要 "
      f"{(((1.12)**3*PE_NOW/PE5_MED)**(1/3)-1)*100:.1f}%/年。")

# ---------------- share count and capital return
print("\n" + "=" * 116)
print("【2】股数与资本回报（核实「回购使股数年减 1.7%」）")
print("=" * 116)
sh = panel[(panel.symbol == SYM) & (panel.concept == "shares_diluted") & (panel.kind == "duration")]
sh = sh[sh.days.between(60, 120)]
s = (sh.groupby("end").agg(val=("val", "last"), tag=("tag", "last"),
                           accn=("accn", "first")).reset_index().sort_values("end"))
s = s.tail(14)
print("  季度加权摊薄/基本股数（百万）")
prev = {}
for r in s.itertuples():
    tgt = r.end - pd.Timedelta(days=365)
    m = s[(s.end - tgt).abs().dt.days <= 20]
    yy = f"{r.val/float(m.val.iloc[0])-1:+.2%}" if len(m) else "—"
    print(f"    {r.end.date()}  {r.val/1e6:>9,.1f}   同比 {yy:>8}   {r.tag[:34]:<36}{r.accn}")
print("\n  ⚠️ 口径提醒：最近两期的 tag 是 WeightedAverageNumberOfSharesOutstandingBasic（基本），"
      "\n     而不是 Diluted。_fund_data 的阶梯里 Diluted 优先，但 F.series 每个 end 只留一行、"
      "\n     按 filed 取最后一版，于是基本股数覆盖了摊薄股数。摊薄 > 基本，所以用基本股数算出的"
      "\n     市值和 EPS 会略偏低。差额见下：")
bd = panel[(panel.symbol == SYM) & (panel.concept == "shares_diluted") & (panel.days.between(60, 120))]
for e in ("2026-06-27", "2026-03-28"):
    k = bd[bd.end == pd.Timestamp(e)]
    d_ = k[k.tag.str.contains("Diluted")].val.max()
    b_ = k[k.tag.str.contains("Basic")].val.max()
    if np.isfinite(d_) and np.isfinite(b_):
        print(f"     {e}: 摊薄 {d_/1e6:,.1f}M  基本 {b_/1e6:,.1f}M  差 {d_/b_-1:+.2%}"
              f"  → 用摊薄股数市值 ${PX_NOW*d_/1e9:,.0f}bn（而非 ${PX_NOW*b_/1e9:,.0f}bn）")

# capital return run rate
print("\n  资本回报的现金口径（9个月，来自 10-Q 0000320193-26-000020 / 0000320193-25-000073）")
print("    回购 9M FY2026 $62,094M vs 9M FY2025 $70,579M  = -12.0%")
print("    股息 见下")
D = pd.read_csv("_fd_AAPL_dim_facts.csv", parse_dates=["start", "end"])
D["dims"] = D["dims"].fillna("")
div = D[(D.tag == "PaymentsOfDividendsCommonStock") & (D.dims == "")].dropna(subset=["days"])
if not div.empty:
    print(div.drop_duplicates(subset=["start", "end", "days"]).sort_values(["end", "days"])
          [["start", "end", "days", "val"]].assign(val=lambda x: x.val / 1e6)
          .to_string(index=False, float_format=lambda x: f"{x:,.0f}"))
mcap_d = PX_NOW * 14714.676e6
print(f"\n  按 9M 折年：回购 ${62094/0.75/1e3:,.1f}bn/年 → 相对市值 ${mcap_d/1e9:,.0f}bn 的回购收益率 "
      f"{62094e6/0.75/mcap_d:.2%}")
print(f"  这就是「股数年减」的上限：如果股价不变、SBC 不抵消，回购收益率 {62094e6/0.75/mcap_d:.2%} "
      "≈ 股数年减速度。\n  实测同比股数变化见上表。回购金额在缩，且股价涨了 43.8%，"
      "同样的美元买回的股数更少 —— 这两者叠加\n  意味着未来的每股增益贡献比过去几年低。")
sbc9 = 10523e6
print(f"  SBC 9M FY2026 ${sbc9/1e6:,.0f}M（折年 ${sbc9/0.75/1e9:.1f}bn）抵消了回购的 "
      f"{sbc9/0.75/(62094e6/0.75):.0%}。净回购 ≈ ${(62094e6-sbc9)/0.75/1e9:.1f}bn/年"
      f" = 市值的 {(62094e6-sbc9)/0.75/mcap_d:.2%}")

# ---------------- relative strength, recomputed
print("\n" + "=" * 116)
print("【3】相对强弱（自行重算，不沿用）")
print("=" * 116)
peers = ["AAPL", "SPY", "QQQ", "XLK", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]
p = yf.download(peers, period="2y", progress=False, auto_adjust=True)["Close"].dropna(how="all")
print(f"  数据截至 {p.index[-1].date()}")
print(f"  {'':<7}{'近1月':>9}{'近3月':>9}{'近6月':>9}{'近1年':>9}"
      f"{'相对SPY 1月':>13}{'相对SPY 3月':>13}{'距1年高点':>11}")
base = {}
for s_ in peers:
    if s_ not in p.columns:
        continue
    x = p[s_].dropna()
    r = {n: float(x.iloc[-1] / x.iloc[-1 - n] - 1) for n in (21, 63, 126, 252) if len(x) > n}
    base[s_] = r
for s_ in peers:
    if s_ not in base:
        continue
    r = base[s_]
    x = p[s_].dropna()
    dd = float(x.iloc[-1] / x.tail(252).max() - 1)
    rs1 = r[21] - base["SPY"][21]
    rs3 = r[63] - base["SPY"][63]
    print(f"  {s_:<7}{r[21]*100:>8.1f}%{r[63]*100:>8.1f}%{r[126]*100:>8.1f}%{r[252]*100:>8.1f}%"
          f"{rs1*100:>12.1f}%{rs3*100:>12.1f}%{dd*100:>10.1f}%")
a, m = base["AAPL"], base["MSFT"]
print(f"\n  AAPL 相对 MSFT：近1月 {(a[21]-m[21])*100:+.1f}%   近3月 {(a[63]-m[63])*100:+.1f}%"
      f"   近6月 {(a[126]-m[126])*100:+.1f}%   近1年 {(a[252]-m[252])*100:+.1f}%")
print(f"  AAPL 相对 XLK：近1月 {(a[21]-base['XLK'][21])*100:+.1f}%   "
      f"近3月 {(a[63]-base['XLK'][63])*100:+.1f}%   近1年 {(a[252]-base['XLK'][252])*100:+.1f}%")
