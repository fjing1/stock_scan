"""
_fd_AAPL_pe_history.py — where does today's multiple sit in AAPL's OWN history, computed
point-in-time so there is no lookahead.

This is the arithmetic that answers "is the quality already priced". Quality that has been true for
twenty years cannot explain a future return; a MULTIPLE that has re-rated can. So:

  * at every trading day, TTM EPS is assembled from ONLY the XBRL facts whose `filed` date is <= that
    day (with Q4 derived from the 10-K, since Apple's quarterly EPS facts omit Q4)
  * P/E = that day's close / that TTM EPS
  * same for EV/Sales, using TTM revenue and the balance sheet known on that day

Then decompose the last 1 / 3 / 5 years of price return into EARNINGS GROWTH vs MULTIPLE CHANGE,
because for a holder those are completely different situations.
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


def full_quarters(concept: str) -> pd.DataFrame:
    """Quarterly series with Q4 derived from (annual - Q1..Q3), each row stamped with the date the
    information became public (`filed`). For a derived Q4 that is the 10-K's filed date."""
    a = F.series(panel, SYM, concept, annual=True)
    q = F.series(panel, SYM, concept, annual=False)
    rows = q[["end", "val", "filed", "accn"]].to_dict("records")
    for i in range(1, len(a)):
        ye, ys = a.end.iloc[i], a.end.iloc[i - 1]
        ins = q[(q.end > ys) & (q.end <= ye)]
        if len(ins) != 3:
            continue
        rows.append({"end": ye, "val": float(a.val.iloc[i]) - float(ins.val.sum()),
                     "filed": a.filed.iloc[i], "accn": a.accn.iloc[i]})
    d = pd.DataFrame(rows).sort_values(["end", "filed"]).drop_duplicates("end", keep="first")
    return d.reset_index(drop=True)


eps = full_quarters("eps_diluted")
rev = full_quarters("revenue")
print(f"季度 EPS 序列 n={len(eps)}  {eps.end.min().date()}→{eps.end.max().date()}")
print(f"季度 营收 序列 n={len(rev)}  {rev.end.min().date()}→{rev.end.max().date()}")

d = yf.download(SYM, period="max", progress=False, auto_adjust=True)
if isinstance(d.columns, pd.MultiIndex):
    d.columns = d.columns.get_level_values(0)
C = d[d.Close > 0].Close.dropna()
# split factor: XBRL EPS is as-reported, prices are split-adjusted. Rebase EPS onto the adjusted
# axis using the ratio of adjusted to unadjusted close today (AAPL last split 2020-08-31 4:1, all
# XBRL facts here are post-split, so this is a no-op check rather than a correction).
raw = yf.download(SYM, period="10d", progress=False, auto_adjust=False)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)
adj_ratio = float(C.iloc[-1]) / float(raw["Close"].dropna().iloc[-1])
print(f"调整后/未调整 收盘比 = {adj_ratio:.4f}  （=1 表示无需分红/拆股再调整 EPS 轴）")


def ttm_asof(s: pd.DataFrame, day: pd.Timestamp):
    """TTM sum using only facts filed on or before `day`, requiring 4 consecutive quarters."""
    k = s[s.filed <= day]
    if len(k) < 4:
        return np.nan, None
    k = k.sort_values("end").tail(4)
    if not (255 <= (k.end.iloc[-1] - k.end.iloc[0]).days <= 290):
        return np.nan, None
    return float(k.val.sum()), k.end.iloc[-1]


days = C.index[C.index >= "2010-01-01"]
rows = []
for day in days[::5]:                    # weekly grid is plenty and keeps this fast
    e, e_end = ttm_asof(eps, day)
    r, r_end = ttm_asof(rev, day)
    if not np.isfinite(e) or e <= 0:
        continue
    rows.append({"date": day, "close": float(C[day]), "eps_ttm": e, "pe": float(C[day]) / e,
                 "rev_ttm": r, "q_end": e_end})
P = pd.DataFrame(rows).set_index("date")
print(f"\n逐日(周度网格) point-in-time PE 序列 n={len(P)}  {P.index[0].date()}→{P.index[-1].date()}")

pe_now = float(P.pe.iloc[-1])
print("\n" + "=" * 110)
print(f"今天的 point-in-time GAAP PE = {pe_now:.1f}x   "
      f"(收盘 {P.close.iloc[-1]:.2f} / TTM EPS {P.eps_ttm.iloc[-1]:.2f}, 最新季 {P.q_end.iloc[-1].date()})")
print("=" * 110)
for lbl, sub in (("2010年至今", P), ("近10年", P.tail(522)), ("近5年", P.tail(261)),
                 ("近3年", P.tail(157)), ("近1年", P.tail(53))):
    print(f"  {lbl:<10} PE 中位 {sub.pe.median():5.1f}x  区间 {sub.pe.min():.1f}–{sub.pe.max():.1f}x"
          f"   今天所处百分位 {float((sub.pe <= pe_now).mean())*100:5.1f}%")

print("\n  按年（每年末的 point-in-time PE 与该年 TTM EPS）")
yr = P.groupby(P.index.year).last()
print(yr[["close", "eps_ttm", "pe"]].to_string(float_format=lambda x: f"{x:,.2f}"))

print("\n" + "=" * 110)
print("回报归因：价格变化 = EPS 变化 × 倍数变化   （对持有者这是最有决策价值的一个拆解）")
print("=" * 110)
for yrs, n in ((1, 53), (2, 105), (3, 157), (5, 261), (10, 522)):
    if len(P) <= n:
        continue
    a, b = P.iloc[-1], P.iloc[-1 - n]
    pr = a.close / b.close - 1
    eg = a.eps_ttm / b.eps_ttm - 1
    mg = a.pe / b.pe - 1
    print(f"  近{yrs:>2}年 ({b.name.date()}→{a.name.date()})  价格 {pr:+7.1%}"
          f" = EPS {eg:+7.1%}  ×  倍数 {mg:+7.1%}"
          f"    (PE {b.pe:.1f}x→{a.pe:.1f}x, EPS {b.eps_ttm:.2f}→{a.eps_ttm:.2f})"
          f"   倍数贡献占比 {np.log1p(mg)/np.log1p(pr)*100 if pr>0 else float('nan'):.0f}%")

print("\n" + "=" * 110)
print("「已被定价」的算术：在今天的倍数上，未来回报需要什么")
print("=" * 110)
eps0 = float(P.eps_ttm.iloc[-1])
px0 = float(P.close.iloc[-1])
print(f"  盈利收益率 = 1/{pe_now:.1f} = {1/pe_now:.2%}")
try:
    tnx = yf.download("^TNX", period="1mo", progress=False)["Close"].dropna()
    y10 = float(tnx.iloc[-1]) / 10
    print(f"  10年美债收益率 {y10:.2f}%（^TNX，市场数据）→ 盈利收益率溢价 "
          f"{(1/pe_now*100 - y10):+.2f}pts")
except Exception as e:
    print(f"  ^TNX 取数失败: {e}")
print(f"\n  三年后若倍数回到各水平，需要多少 EPS 复合增速才能维持 0% / +8% / +15% 年化价格回报")
print(f"  {'3年后PE':>9}" + "".join(f"{'价格'+t:>12}" for t in ("+0%/年", "+8%/年", "+15%/年")))
for pe_end in (25, 28, 30, 33, 35, 38.2, 42):
    row = f"  {pe_end:>9.1f}"
    for tgt in (0.0, 0.08, 0.15):
        need = ((1 + tgt) ** 3 * pe_now / pe_end) ** (1 / 3) - 1
        row += f"{need*100:>11.1f}%"
    print(row)
print("\n  读法：表里每个数字是「EPS 年复合增速要求」。当前 TTM EPS 同比 "
      f"{float(P.eps_ttm.iloc[-1]/P.eps_ttm.iloc[-53]-1)*100:+.1f}%。")

print("\n" + "=" * 110)
print("EV/Sales 与营收的同一拆解（TTM 营收，point-in-time）")
Pr = P.dropna(subset=["rev_ttm"])
for yrs, n in ((1, 53), (3, 157), (5, 261)):
    if len(Pr) <= n:
        continue
    a, b = Pr.iloc[-1], Pr.iloc[-1 - n]
    print(f"  近{yrs}年  营收TTM {b.rev_ttm/1e6:,.0f}→{a.rev_ttm/1e6:,.0f}M ({a.rev_ttm/b.rev_ttm-1:+.1%})"
          f"   价格 {a.close/b.close-1:+.1%}   ⇒ 市销倍数变化 "
          f"{(a.close/b.close)/(a.rev_ttm/b.rev_ttm)-1:+.1%}")
