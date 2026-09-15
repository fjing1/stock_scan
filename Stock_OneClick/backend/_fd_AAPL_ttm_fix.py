"""
_fd_AAPL_ttm_fix.py — recompute AAPL's TTM revenue / TTM EPS / TTM net income from the SEC XBRL
panel, correcting the missing-Q4 problem, and re-derive EV/Sales and GAAP P/E.

WHY: fund_metrics.py reports revenue_ttm == revenue_fy for AAPL. That is not TTM. Its own guard
(`250 <= span <= 300`) rejects the naive 4-row quarterly sum because Apple's XBRL quarterly duration
facts are MISSING Q4 (the 10-K carries the full year instead), so the last four *rows* span ~15
months. The guard correctly refuses to sum them -- but then falls back to the FISCAL-YEAR figure,
which as of 2026-09 is 11.5 months stale and, with quarterly YoY at +16%, materially understates
the denominator of EV/Sales.

Worse: `pe_gaap` has NO such guard. It does `eps.val.iloc[-4:].sum()` on the same Q4-less quarterly
series, so "TTM EPS" is actually Q3FY25 + Q1FY26 + Q2FY26 + Q3FY26 -- it substitutes the June-2025
quarter for the September-2025 quarter and skips a quarter entirely.

FIX: derive Q4 as (annual - sum of that year's Q1..Q3), then take a true trailing four quarters.
Every number printed here is traceable to an accession number, printed alongside.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import _fund_data as F

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

SYM = "AAPL"


def show(s: pd.DataFrame, name: str, n: int = 14, scale: float = 1e6, unit: str = "$M"):
    print(f"\n--- {name}  (n={len(s)})")
    t = s.tail(n).copy()
    t["val_disp"] = t["val"] / scale
    print(t[["end", "val_disp", "days", "tag", "form", "filed", "accn"]]
          .rename(columns={"val_disp": unit}).to_string(index=False))


def derive_q4(annual: pd.DataFrame, quarterly: pd.DataFrame) -> pd.DataFrame:
    """Return quarterly series with derived Q4 rows inserted.

    A fiscal year's Q4 = annual - (the three quarterly facts whose end dates fall inside that
    fiscal year). Fiscal-year window is (prior_annual_end, annual_end].
    """
    rows = []
    a = annual.sort_values("end").reset_index(drop=True)
    q = quarterly.sort_values("end").reset_index(drop=True)
    for i in range(1, len(a)):
        y_end, y_start = a.end.iloc[i], a.end.iloc[i - 1]
        inside = q[(q.end > y_start) & (q.end <= y_end)]
        if len(inside) != 3:
            continue                      # can't derive safely
        q4 = float(a.val.iloc[i]) - float(inside.val.sum())
        rows.append({"end": y_end, "val": q4, "days": 91, "tag": "DERIVED(FY-Q1..Q3)",
                     "form": a.form.iloc[i], "filed": a.filed.iloc[i], "accn": a.accn.iloc[i]})
    out = pd.concat([q, pd.DataFrame(rows)], ignore_index=True) if rows else q
    return out.sort_values("end").drop_duplicates(subset=["end"], keep="first").reset_index(drop=True)


def ttm(series: pd.DataFrame, label: str):
    """True trailing 4 quarters, with a span assertion."""
    s = series.tail(4)
    span = (s.end.iloc[-1] - s.end.iloc[0]).days
    val = float(s.val.sum())
    print(f"\n  {label}: TTM = {val:,.2f}   四季末: "
          + ", ".join(f"{r.end.date()}={r.val:,.2f}" for r in s.itertuples())
          + f"   首末间隔 {span}d {'OK' if 260 <= span <= 290 else '<< 有问题'}")
    return val, span


def main():
    panel = F.load()
    rev_a = F.series(panel, SYM, "revenue", annual=True)
    rev_q = F.series(panel, SYM, "revenue", annual=False)
    eps_a = F.series(panel, SYM, "eps_diluted", annual=True)
    eps_q = F.series(panel, SYM, "eps_diluted", annual=False)
    ni_a = F.series(panel, SYM, "net_income", annual=True)
    ni_q = F.series(panel, SYM, "net_income", annual=False)
    ocf_a = F.series(panel, SYM, "ocf", annual=True)
    ocf_q = F.series(panel, SYM, "ocf", annual=False)

    print("=" * 110)
    print("AAPL — TTM 重算（修正 XBRL 缺 Q4 的问题）")
    print("=" * 110)
    show(rev_q, "季度营收 原始（注意 end 日期的跳档 = 缺 Q4）")
    show(rev_a, "年度营收")
    show(eps_q, "季度摊薄EPS 原始", scale=1.0, unit="$/sh")

    print("\n" + "=" * 110)
    print("缺 Q4 的直接证据：相邻季度 end 的间隔")
    g = rev_q.copy()
    g["gap_days"] = g.end.diff().dt.days
    print(g.tail(13)[["end", "gap_days", "val", "form", "accn"]]
          .assign(val=lambda d: d.val / 1e6).to_string(index=False))

    print("\n" + "=" * 110)
    print("补齐 Q4 后")
    for name, qa, aa, sc, un in (("revenue", rev_q, rev_a, 1e6, "$M"),
                                 ("eps_diluted", eps_q, eps_a, 1.0, "$/sh"),
                                 ("net_income", ni_q, ni_a, 1e6, "$M"),
                                 ("ocf", ocf_q, ocf_a, 1e6, "$M")):
        full = derive_q4(aa, qa)
        show(full, f"{name} 补齐后", n=9, scale=sc, unit=un)
        v, span = ttm(full, f"{name}")
        # naive (what fund_metrics does for EPS)
        nv = float(qa.tail(4).val.sum())
        nspan = (qa.end.iloc[-1] - qa.end.iloc[-4]).days
        print(f"  {name}: 朴素后4行求和 = {nv:,.2f} (间隔 {nspan}d)  "
              f"误差 {nv - v:+,.2f} = {(nv/v-1)*100:+.2f}%")
        globals()[f"TTM_{name}"] = v

    # ---- balance sheet / share count, same as fund_metrics
    import fund_metrics as FM
    m = FM.metrics(panel, SYM, price=None)
    print("\n" + "=" * 110)
    print("资产负债表与股数（沿用 fund_metrics 的口径）")
    for k in ("cash_and_inv", "debt_total", "net_cash", "shares", "bs_as_of"):
        print(f"  {k:<14} {m.get(k)}")

    # shares OUTSTANDING (dei) matters more than weighted-average diluted for market cap
    d = panel[(panel.symbol == SYM) & (panel.concept == "shares_diluted")]
    print("\n  shares_diluted 明细（最近6条，用于确认是加权平均摊薄股数）")
    print(d.sort_values(["end", "filed"]).tail(6)[["end", "val", "tag", "days", "form", "accn"]]
          .assign(val=lambda x: x.val / 1e6).to_string(index=False))

    # ---- valuation on the corrected TTM
    import yfinance as yf
    px = float(yf.download(SYM, period="5d", progress=False, auto_adjust=True)["Close"].dropna().iloc[-1])
    sh = m["shares"]
    mcap = px * sh
    ev = mcap - m["net_cash"]
    print("\n" + "=" * 110)
    print(f"估值重算  价格 {px:.2f}  股数 {sh/1e6:,.0f}M")
    print(f"  市值 ${mcap/1e9:,.1f}bn   净现金 ${m['net_cash']/1e9:+,.1f}bn   EV ${ev/1e9:,.1f}bn")
    print(f"  EV/Sales  用 FY2025 营收 (fund_metrics 现口径) = {ev/m['revenue_fy']:.2f}x")
    print(f"  EV/Sales  用 真实TTM 营收                      = {ev/TTM_revenue:.2f}x")
    print(f"  PE(GAAP)  用 朴素后4行EPS (fund_metrics 现口径) = {px/float(eps_q.tail(4).val.sum()):.2f}x")
    print(f"  PE(GAAP)  用 真实TTM EPS                       = {px/TTM_eps_diluted:.2f}x")
    print(f"  PE(GAAP)  用 TTM净利/加权股数                  = {px/(TTM_net_income/sh):.2f}x")
    print(f"  EV/OCF    用 真实TTM OCF                       = {ev/TTM_ocf:.2f}x")
    print(f"  TTM 营收 vs FY2025: {TTM_revenue/m['revenue_fy']-1:+.2%}")
    print(f"  TTM 净利率 {TTM_net_income/TTM_revenue:.1%}   TTM OCF/营收 {TTM_ocf/TTM_revenue:.1%}"
          f"   TTM 应计 (NI-OCF)/营收 {(TTM_net_income-TTM_ocf)/TTM_revenue:+.2%}")

    # ---- TTM revenue growth: this quarter's TTM vs the TTM one year ago
    full_rev = derive_q4(rev_a, rev_q)
    if len(full_rev) >= 8:
        t0 = float(full_rev.val.iloc[-4:].sum())
        t1 = float(full_rev.val.iloc[-8:-4].sum())
        print(f"\n  TTM 营收同比: {t0/1e6:,.0f} vs {t1/1e6:,.0f} = {t0/t1-1:+.2%}"
              f"   (窗口 {full_rev.end.iloc[-8].date()}→{full_rev.end.iloc[-1].date()})")
        print("\n  近8个季度 同比（按补齐后的真实季度序列，date-matched）")
        for i in range(len(full_rev) - 8, len(full_rev)):
            cur = full_rev.iloc[i]
            tgt = cur.end - pd.Timedelta(days=365)
            prior = full_rev.iloc[:i].copy()
            prior["gap"] = (prior.end - tgt).abs().dt.days
            c = prior.nsmallest(1, "gap").iloc[0]
            flag = "" if c["gap"] <= 20 else f"  (基期偏差 {c['gap']}d!)"
            print(f"    {cur.end.date()}  {cur.val/1e6:>10,.0f}  vs {c.end.date()} "
                  f"{c.val/1e6:>10,.0f}   {cur.val/c.val-1:+7.2%}   tag={cur.tag[:26]}{flag}")


if __name__ == "__main__":
    main()
