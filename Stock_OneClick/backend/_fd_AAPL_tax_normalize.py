"""
_fd_AAPL_tax_normalize.py — normalise the FY2024 one-off tax charge, because it inflates every
"EPS growth" number computed off the reported TTM base.

FACT trail: AAPL's effective tax rate was 14.7% (FY2023), 24.1% (FY2024), 15.6% (FY2025)
[us-gaap:EffectiveIncomeTaxRateContinuingOperations, 10-K accn 0000320193-25-000079]. FY2024's spike
is the EU State Aid decision, and it lands entirely in the SEPTEMBER 2024 quarter -- which is exactly
the quarter sitting in the year-ago TTM base for the June-2026 TTM. So the headline "TTM EPS
+32.2%" is partly the lapping of a one-time tax charge, not operating improvement.

Derive the charge from the filed facts only:
    Q4 FY2024 pretax = FY2024 pretax − (Q1+Q2+Q3 pretax)
    Q4 FY2024 tax    = FY2024 tax    − (Q1+Q2+Q3 tax)
    excess           = Q4 tax − Q4 pretax x normal rate
then restate the 1-year return attribution on the normalised base.
"""
from __future__ import annotations

import gzip
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)
cf = json.load(gzip.open("_fund_cache/CIK0000320193.json.gz", "rt"))
G = cf["facts"]["us-gaap"]


def ser(tag, lo, hi):
    rows = []
    for u, arr in G[tag]["units"].items():
        for x in arr:
            if not x.get("start"):
                continue
            rows.append({"start": x["start"], "end": x["end"], "val": x["val"],
                         "form": x.get("form"), "filed": x["filed"], "accn": x.get("accn")})
    d = pd.DataFrame(rows)
    d["start"], d["end"] = pd.to_datetime(d.start), pd.to_datetime(d.end)
    d["days"] = (d.end - d.start).dt.days
    return (d[d.days.between(lo, hi)].sort_values(["end", "filed"])
            .drop_duplicates("end", keep="last").set_index("end"))


PT = "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"
pt_q, pt_a = ser(PT, 80, 100), ser(PT, 355, 375)
tx_q, tx_a = ser("IncomeTaxExpenseBenefit", 80, 100), ser("IncomeTaxExpenseBenefit", 355, 375)
ni_q, ni_a = ser("NetIncomeLoss", 80, 100), ser("NetIncomeLoss", 355, 375)
eps_q, eps_a = ser("EarningsPerShareDiluted", 80, 100), ser("EarningsPerShareDiluted", 355, 375)
etr = ser("EffectiveIncomeTaxRateContinuingOperations", 355, 375)

print("=" * 112)
print("有效税率（年度，10-K 申报事实）")
for e in ("2022-09-24", "2023-09-30", "2024-09-28", "2025-09-27"):
    r = etr.loc[pd.Timestamp(e)]
    print(f"  FY 截至 {e}  ETR {r.val:.1%}   accn {r.accn}")

print("\n" + "=" * 112)
print("推导 FY2024 第四财季（2024-09-28）的税负异常")
FY = pd.Timestamp("2024-09-28")
q1q3_pt = pt_q.loc[["2023-12-30", "2024-03-30", "2024-06-29"]].val.sum()
q1q3_tx = tx_q.loc[["2023-12-30", "2024-03-30", "2024-06-29"]].val.sum()
q1q3_ni = ni_q.loc[["2023-12-30", "2024-03-30", "2024-06-29"]].val.sum()
q1q3_eps = eps_q.loc[["2023-12-30", "2024-03-30", "2024-06-29"]].val.sum()
fy_pt, fy_tx, fy_ni, fy_eps = (float(pt_a.loc[FY].val), float(tx_a.loc[FY].val),
                               float(ni_a.loc[FY].val), float(eps_a.loc[FY].val))
q4_pt, q4_tx, q4_ni, q4_eps = fy_pt - q1q3_pt, fy_tx - q1q3_tx, fy_ni - q1q3_ni, fy_eps - q1q3_eps
print(f"  FY2024 税前 ${fy_pt/1e6:,.0f}M  税 ${fy_tx/1e6:,.0f}M  净利 ${fy_ni/1e6:,.0f}M  "
      f"EPS ${fy_eps:.2f}   (10-K accn {pt_a.loc[FY].accn})")
print(f"  Q1-Q3     税前 ${q1q3_pt/1e6:,.0f}M  税 ${q1q3_tx/1e6:,.0f}M  净利 ${q1q3_ni/1e6:,.0f}M  "
      f"EPS ${q1q3_eps:.2f}")
print(f"  ⇒ Q4FY24  税前 ${q4_pt/1e6:,.0f}M  税 ${q4_tx/1e6:,.0f}M  净利 ${q4_ni/1e6:,.0f}M  "
      f"EPS ${q4_eps:.2f}")
print(f"  ⇒ Q4FY24 有效税率 = {q4_tx/q4_pt:.1%}   （FY2025 全年 {float(etr.loc['2025-09-27'].val):.1%}，"
      f"FY2023 {float(etr.loc['2023-09-30'].val):.1%}）")
for nm, rate in (("FY2025 15.6%", float(etr.loc["2025-09-27"].val)),
                 ("FY2023 14.7%", float(etr.loc["2023-09-30"].val)),
                 ("FY22-25均值", float(etr.loc[["2022-09-24", "2023-09-30", "2025-09-27"]].val.mean()))):
    excess = q4_tx - q4_pt * rate
    sh = q4_ni / q4_eps                    # implied diluted share count for that quarter
    print(f"     按 {nm} 常态税率 → 一次性超额税负 ${excess/1e6:,.0f}M，"
          f"正常化 Q4FY24 净利 ${(q4_ni+excess)/1e6:,.0f}M，EPS ${q4_eps + excess/sh:.2f}")

rate = float(etr.loc["2025-09-27"].val)
excess = q4_tx - q4_pt * rate
sh = q4_ni / q4_eps
q4_eps_norm = q4_eps + excess / sh

print("\n" + "=" * 112)
print("对 TTM EPS 增速与一年回报归因的影响")
print("=" * 112)
ttm_now = 1.84 + 2.84 + 2.01 + 2.02        # Sep-25, Dec-25, Mar-26, Jun-26 (Q4 由 10-K 推导)
ttm_ago_rep = q4_eps + 2.40 + 1.65 + 1.57  # Sep-24(报告), Dec-24, Mar-25, Jun-25
ttm_ago_norm = q4_eps_norm + 2.40 + 1.65 + 1.57
px_now, px_ago = 333.08, 231.66            # 2026-09-14 与 2025-09-11 前后一年的收盘（yfinance 调整后）
print(f"  当前 TTM EPS                    ${ttm_now:.2f}")
print(f"  一年前 TTM EPS（报告值）          ${ttm_ago_rep:.2f}  → 同比 {ttm_now/ttm_ago_rep-1:+.1%}")
print(f"  一年前 TTM EPS（税负正常化后）      ${ttm_ago_norm:.2f}  → 同比 {ttm_now/ttm_ago_norm-1:+.1%}")
print(f"\n  一年价格回报 {px_now/px_ago-1:+.1%}")
for lbl, base in (("按报告 EPS", ttm_ago_rep), ("按正常化 EPS", ttm_ago_norm)):
    pe0, pe1 = px_ago / base, px_now / ttm_now
    eg, mg = ttm_now / base - 1, pe1 / pe0 - 1
    print(f"  {lbl:<14} PE {pe0:.1f}x→{pe1:.1f}x  倍数 {mg:+.1%}   EPS {eg:+.1%}   "
          f"倍数贡献占比 {np.log1p(mg)/np.log1p(px_now/px_ago-1)*100:.0f}%")
print("\n  ⇒ 这是关键修正：用报告 EPS 看，过去一年 +43.8% 里只有 23% 来自倍数；")
print("    剔除 FY2024 那笔一次性税负后，倍数贡献约一半。市场确实在重新定价，不只是盈利在长。")
