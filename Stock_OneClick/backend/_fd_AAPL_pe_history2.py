"""
_fd_AAPL_pe_history2.py — point-in-time GAAP P/E and EV/Sales history for AAPL, with the two bugs
from the first pass fixed.

BUG 1, SPLITS. XBRL EPS is AS REPORTED; yfinance prices are SPLIT ADJUSTED. AAPL split 7:1 on
2014-06-09 and 4:1 on 2020-08-31, so an unadjusted EPS from FY2013 is 28x too large relative to an
adjusted price, and the first pass produced a "P/E" of 0.39x for 2013 and a 10-year attribution
claiming the multiple went from 1.1x to 36.2x. Fixed by dividing each EPS fact by the cumulative
split factor applied AFTER its period end.

BUG 2, FILED DATES. _fund_data.F.series() keeps the LAST filed version of each period, which for a
quarter is the comparative column of a 10-Q filed a year later. Using that as the point-in-time key
delays every fact by up to 12 months and mis-dates the whole series. Fixed by taking min(filed) over
all versions of each (concept, period), which is the original filing.

Also: 1/PE against the actual 10-year yield (^TNX quotes the yield directly, ~4.96, not yield x 10 --
the first pass divided by 10 and printed 0.50%).
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
SPLITS = {pd.Timestamp("2014-06-09"): 7.0, pd.Timestamp("2020-08-31"): 4.0}


def split_factor(period_end: pd.Timestamp) -> float:
    """Cumulative factor by which shares multiplied AFTER `period_end`. Divide as-reported per-share
    figures by this to put them on today's split-adjusted axis."""
    return float(np.prod([v for k, v in SPLITS.items() if k > period_end]) or 1.0)


def orig_quarters(concept: str, per_share: bool) -> pd.DataFrame:
    """One row per fiscal quarter, Q4 derived from the 10-K, value on the split-adjusted axis, and
    `filed` = the EARLIEST filing that carried the fact (true point-in-time)."""
    d = panel[(panel.symbol == SYM) & (panel.concept == concept) & (panel.kind == "duration")].copy()
    q = d[d.days.between(60, 120)]
    a = d[d.days.between(300, 400)]
    qf = (q.groupby("end").agg(val=("val", "last"), filed=("filed", "min"),
                               accn=("accn", "first")).reset_index().sort_values("end"))
    af = (a.groupby("end").agg(val=("val", "last"), filed=("filed", "min"),
                               accn=("accn", "first")).reset_index().sort_values("end"))
    rows = qf.to_dict("records")
    for i in range(1, len(af)):
        ye, ys = af.end.iloc[i], af.end.iloc[i - 1]
        ins = qf[(qf.end > ys) & (qf.end <= ye)]
        if len(ins) != 3:
            continue
        rows.append({"end": ye, "val": float(af.val.iloc[i]) - float(ins.val.sum()),
                     "filed": af.filed.iloc[i], "accn": af.accn.iloc[i]})
    o = pd.DataFrame(rows).sort_values(["end", "filed"]).drop_duplicates("end", keep="first")
    if per_share:
        o["val"] = o.val / o.end.map(split_factor)
    return o.reset_index(drop=True)


eps = orig_quarters("eps_diluted", True)
rev = orig_quarters("revenue", False)
print(f"EPS 季度序列 n={len(eps)}  {eps.end.min().date()}→{eps.end.max().date()}")
print("  拆股调整抽样：")
for e in ("2013-06-29", "2013-12-28", "2019-06-29", "2020-06-27", "2021-03-27", "2026-06-27"):
    r = eps[eps.end == pd.Timestamp(e)]
    if len(r):
        r = r.iloc[0]
        print(f"    {e}  调整后EPS {r.val:.4f}  (÷{split_factor(pd.Timestamp(e)):.0f})  "
              f"首次报送 {r.filed.date()}  {r.accn}")

d = yf.download(SYM, period="max", progress=False, auto_adjust=True)
if isinstance(d.columns, pd.MultiIndex):
    d.columns = d.columns.get_level_values(0)
C = d[d.Close > 0].Close.dropna()


def ttm_asof(s, day):
    k = s[s.filed <= day]
    if len(k) < 4:
        return np.nan, None
    k = k.sort_values("end").tail(4)
    if not (255 <= (k.end.iloc[-1] - k.end.iloc[0]).days <= 290):
        return np.nan, None
    return float(k.val.sum()), k.end.iloc[-1]


grid = C.index[C.index >= "2011-01-01"][::5]
rows = []
for day in grid:
    e, qe = ttm_asof(eps, day)
    r, _ = ttm_asof(rev, day)
    if not np.isfinite(e) or e <= 0:
        continue
    rows.append({"date": day, "close": float(C[day]), "eps": e, "pe": float(C[day]) / e,
                 "rev": r, "q_end": qe})
P = pd.DataFrame(rows).set_index("date")
step_days = int(np.median(np.diff(P.index.values).astype("timedelta64[D]").astype(int)))
per_yr = int(round(365 / step_days))
print(f"\n序列 n={len(P)}  {P.index[0].date()}→{P.index[-1].date()}  "
      f"网格步长 ~{step_days}天 → 每年 ~{per_yr} 个点")
# today's live value, not the last grid point
px_now = float(C.iloc[-1])
eps_now = float(eps.sort_values("end").val.tail(4).sum())
pe_now = px_now / eps_now
print(f"\n{'='*112}")
print(f"今天  收盘 {px_now:.2f}   TTM EPS(拆股调整, 含推导Q4) {eps_now:.2f}   GAAP PE {pe_now:.1f}x")
print(f"{'='*112}")
for lbl, n in (("2011年至今", len(P)), ("近10年", 10 * per_yr), ("近5年", 5 * per_yr),
               ("近3年", 3 * per_yr), ("近1年", per_yr)):
    sub = P.tail(n)
    print(f"  {lbl:<10} PE 中位 {sub.pe.median():5.1f}x  区间 {sub.pe.min():.1f}–{sub.pe.max():.1f}x"
          f"   今天({pe_now:.1f}x) 所处百分位 {float((sub.pe <= pe_now).mean())*100:5.1f}%")

print("\n  逐年末的 point-in-time PE")
yr = P.groupby(P.index.year).last()
print(yr[["close", "eps", "pe", "q_end"]].to_string(float_format=lambda x: f"{x:,.2f}"))

print("\n" + "=" * 112)
print("回报归因：价格 = EPS × 倍数")
print("=" * 112)
for yrs in (1, 2, 3, 5, 10):
    n = yrs * per_yr
    if len(P) <= n:
        continue
    b = P.iloc[-1 - n]
    pr, eg, mg = px_now / b.close - 1, eps_now / b.eps - 1, pe_now / b.pe - 1
    share = np.log1p(mg) / np.log1p(pr) * 100 if pr > 0 else np.nan
    print(f"  近{yrs:>2}年 {b.name.date()}→{C.index[-1].date()}   价格 {pr:+8.1%} = "
          f"EPS {eg:+8.1%} × 倍数 {mg:+8.1%}   (PE {b.pe:.1f}→{pe_now:.1f}x, "
          f"EPS {b.eps:.2f}→{eps_now:.2f})   倍数贡献 {share:.0f}%")

print("\n" + "=" * 112)
print("盈利收益率 vs 无风险利率")
tnx = yf.download("^TNX", period="5d", progress=False)["Close"].dropna()
y10 = float(tnx.iloc[-1])
print(f"  GAAP 盈利收益率 1/{pe_now:.1f} = {1/pe_now:.2%}   10年美债 {y10:.2f}%（^TNX 直接报收益率）")
print(f"  差 = {1/pe_now*100-y10:+.2f}pts  ← 负值表示按 GAAP 盈利收益率算，"
      "现价相对无风险利率没有静态补偿；\n     全部回报必须来自盈利增长。")
ocf_ttm = 146724e6      # from _fd_AAPL_segments2.py, 10-Q accn 0000320193-26-000020
capex_ttm = None
cx = F.series(panel, SYM, "capex", annual=True)
fy25_capex = float(cx.val.iloc[-1])
print(f"\n  自由现金流口径：TTM OCF ${ocf_ttm/1e9:.1f}bn（10-Q 0000320193-26-000020 推导）"
      f"   FY2025 capex ${fy25_capex/1e9:.1f}bn")
sh = float(F.latest(panel, SYM, "shares_diluted", kind="duration", min_days=60, max_days=120)["val"])
mcap = px_now * sh
print(f"  市值 ${mcap/1e9:,.0f}bn（{sh/1e6:,.0f}M 加权摊薄股）")
print(f"  P/OCF(TTM) = {mcap/ocf_ttm:.1f}x    OCF 收益率 {ocf_ttm/mcap:.2%}")
fcf = ocf_ttm - 8.9e9   # TTM capex, derived below
print(f"  TTM capex（FY2025 + 9M FY26 − 9M FY25 = {12.9:.1f}+6.8−9.5）= ~$10.2bn "
      f"→ FCF ~${(ocf_ttm-10.2e9)/1e9:.0f}bn，FCF 收益率 {(ocf_ttm-10.2e9)/mcap:.2%}")

print("\n" + "=" * 112)
print("「今天的倍数已经要求什么」：3年后倍数 × 需要的 EPS 年复合增速")
print("=" * 112)
print(f"  {'3年后PE':>8}" + "".join(f"{'价格'+t:>13}" for t in ("+0%/年", "+8%/年", "+12%/年", "+15%/年")))
for pe_end in (24, 27, 30, 33, 36, pe_now, 42):
    row = f"  {pe_end:>8.1f}"
    for tgt in (0.0, 0.08, 0.12, 0.15):
        row += f"{(((1+tgt)**3*pe_now/pe_end)**(1/3)-1)*100:>12.1f}%"
    print(row)
print(f"\n  参照：TTM EPS 同比 {eps_now/float(P.iloc[-1-per_yr].eps)-1:+.1%}；"
      f"最近一季营收同比 +16.4%；分析师对 FQ4 的一致预期隐含营收 +10.9%。")

print("\n" + "=" * 112)
print("EV/Sales 的同一拆解")
Pr = P.dropna(subset=["rev"])
rev_now = 466823e6
for yrs in (1, 3, 5):
    n = yrs * per_yr
    if len(Pr) <= n:
        continue
    b = Pr.iloc[-1 - n]
    print(f"  近{yrs}年  TTM营收 {b.rev/1e6:,.0f}→{rev_now/1e6:,.0f}M ({rev_now/b.rev-1:+.1%})"
          f"   价格 {px_now/b.close-1:+.1%}   ⇒ 市销倍数 {(px_now/b.close)/(rev_now/b.rev)-1:+.1%}")
