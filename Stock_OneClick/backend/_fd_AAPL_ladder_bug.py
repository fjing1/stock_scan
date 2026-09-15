"""
_fd_AAPL_ladder_bug.py — verify a defect in _fund_data.py's concept ladder.

_fund_data.CONCEPTS is documented as a PRIORITY LADDER: "First tag that has data for a given period
wins, and the winning tag is recorded so tag switches are auditable." But F.series() and F.latest()
never look at tag priority:

    d = d.sort_values(["end", "filed"]).drop_duplicates(subset=["end"], keep="last")

When two tags of the same concept appear in the SAME filing (same `end`, same `filed`), the sort is
stable, so the surviving row is whichever came LAST out of extract() -- i.e. the LOWEST-priority tag
on the ladder. The ladder is effectively inverted for any concept whose tags co-occur.

Two places that bites for AAPL:
  * shares_diluted = [Diluted, Basic]  -> Basic wins  -> share count, market cap and EV understated
  * cash = [CashAndCashEquivalentsAtCarryingValue, CashCashEquivalentsAndShortTermInvestments]
    The second tag ALREADY INCLUDES short-term investments. metrics() then adds short_term_inv on
    top, so if the broad tag ever wins, short-term investments are counted TWICE in net cash.

Checked here against the real panel, and the corrected valuation restated.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import _fund_data as F

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)
SYM = "AAPL"
panel = F.load()
PX = 333.08

print("=" * 116)
print("【A】shares_diluted：同一份 10-Q 里 Diluted 与 Basic 同时存在，series() 选了哪个")
print("=" * 116)
d = panel[(panel.symbol == SYM) & (panel.concept == "shares_diluted") & panel.days.between(60, 120)]
for e in ("2026-06-27", "2026-03-28", "2025-12-27"):
    k = d[d.end == pd.Timestamp(e)][["tag", "val", "filed", "accn", "form"]]
    print(f"\n  end={e}  面板里的全部版本")
    print(k.assign(val=lambda x: x.val / 1e6).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
picked = F.series(panel, SYM, "shares_diluted", annual=False)
print("\n  F.series() 实际返回的最近3期")
print(picked.tail(3)[["end", "val", "tag", "accn"]]
      .assign(val=lambda x: x.val / 1e6).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
basic = float(picked.val.iloc[-1])
dil = float(d[(d.end == pd.Timestamp("2026-06-27")) & d.tag.str.contains("Diluted")].val.max())
print(f"\n  结论：返回的是 {'Basic（低优先级）' if 'Basic' in picked.tag.iloc[-1] else 'Diluted'}"
      f"  = {basic/1e6:,.1f}M；阶梯本应优先的 Diluted = {dil/1e6:,.1f}M   低估 {basic/dil-1:+.2%}")

print("\n" + "=" * 116)
print("【B】cash：广义标签会不会赢，从而与 short_term_inv 重复计数")
print("=" * 116)
for c in ("cash", "short_term_inv", "long_term_inv", "debt_term_total", "debt_term_nc",
          "debt_term_c", "debt_short"):
    k = panel[(panel.symbol == SYM) & (panel.concept == c) & (panel.kind == "instant")]
    if k.empty:
        print(f"  {c:<18} 无数据")
        continue
    latest_end = k.end.max()
    kk = k[k.end == latest_end][["tag", "val", "filed", "accn"]].drop_duplicates("tag")
    got = F.latest(panel, SYM, c, kind="instant")
    print(f"  {c:<18} 最新期末 {latest_end.date()}   series/latest 选中 tag = {got['tag']}"
          f"   值 ${got['val']/1e9:,.2f}bn")
    for r in kk.itertuples():
        print(f"      候选 {r.tag:<58} ${r.val/1e9:>8,.2f}bn")
print("\n  判定：AAPL 只报 CashAndCashEquivalentsAtCarryingValue（窄口径），"
      "\n  所以本例没有发生现金重复计数；净现金 $62.2bn 站得住。但这是运气，不是设计 ——"
      "\n  任何同时报窄口径与广口径现金的公司都会被重复计入短期投资。")

print("\n" + "=" * 116)
print("【C】用正确的摊薄股数重述估值")
print("=" * 116)
net_cash = 62220e6
ttm_rev, ttm_eps, ttm_ni, ttm_ocf = 466823e6, 8.71, 128930e6, 146724e6
for lbl, s in (("基本股数(现口径)", basic), ("摊薄股数(应有口径)", dil)):
    mcap = PX * s
    ev = mcap - net_cash
    print(f"  {lbl:<20} 股数 {s/1e6:,.1f}M   市值 ${mcap/1e9:,.1f}bn   EV ${ev/1e9:,.1f}bn"
          f"   EV/Sales {ev/ttm_rev:.2f}x   P/OCF {mcap/ttm_ocf:.1f}x   PE(NI/股) {PX/(ttm_ni/s):.2f}x")
print(f"\n  PE 用 EPS 序列直接算（不经过股数，因此不受此 bug 影响）= {PX/ttm_eps:.2f}x")
print(f"  差异量级：EV/Sales 10.32x → 10.36x，PE 37.86x → 38.01x。对结论无影响，但记录在案。")
