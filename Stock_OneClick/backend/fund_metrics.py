"""
fund_metrics.py — deterministic fundamental metrics and flags from the point-in-time SEC panel.

Layer 2 of the fundamentals stack. NO LLM: everything here is arithmetic on XBRL facts, so it runs
over a whole watchlist in seconds and costs nothing. The expensive language-model layer only gets
invoked later, and only for what XBRL cannot tag (risk-factor deltas, named-entity counts,
litigation status, disclosure withdrawal).

DESIGN RULE THAT MAKES THIS TRUSTWORTHY: numbers come from here, never from a language model.
The ARM workup produced a wrong EV/Sales (54.2x, from a stale hard-coded figure) that only got
caught because a verification pass recomputed it. Structured data is the ground truth; prose is
commentary on top of it.

WHAT THE FLAGS ARE FOR, AND WHAT THEY ARE NOT: these are RISK MARKERS, not a score, and they do not
feed 观海买点分 or the scan's risk score. Fundamental "quality" has never been shown to have a
forward edge in this repo, and six plausible-sounding technical signals here have already died on
measurement. Every flag below is chosen because it was informative in a real workup, not because it
is validated as predictive. Treat them as "go read this filing", not "sell".

The flags that earned their place in the ARM analysis, all mechanically detectable:
  * RPO shrinking while revenue grows  (ARM: RPO -16.6% over two years, revenue +52.2%)
  * capex/revenue accelerating         (ARM: 1.3% -> 11.1% of revenue in four years)
  * related-party revenue carrying growth (ARM: 61% of FY2026 revenue growth)
  * SBC as a share of revenue          (the GAAP-vs-adjusted gap in one number)
  * accruals divergence: net income growing while operating cash flow does not

Usage:
    ../../vcp_env/bin/python fund_metrics.py --watchlist
    ../../vcp_env/bin/python fund_metrics.py --symbols ARM AAPL NVDA --verbose
    ../../vcp_env/bin/python fund_metrics.py --watchlist --out ../reports/fund_screen_2026-09-14.csv
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import _fund_data as F

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent


def _g(a, b):
    """Growth from b to a, guarding zero/negative denominators (common on loss-making lines)."""
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return np.nan
    return a / abs(b) - 1.0


def _val(row):
    return float(row["val"]) if row is not None else np.nan


def _yoy(s: pd.DataFrame, tol_days: int = 45):
    """Year-over-year growth matched on DATE, not on positional index.

    Positional matching (iloc[-5]) is wrong and silently so: XBRL quarterly duration facts are
    MISSING Q4 for most filers, because the 10-K reports the full year and Q4 has to be derived.
    Verified on NVDA -- the quarterly revenue series jumps 2024-10-27 -> 2025-04-27, so iloc[-5]
    reaches back 15 months, not 12, and every "YoY" computed that way is comparing the wrong
    periods. Here we find the observation closest to exactly 365 days before the latest one and
    reject it if it is more than tol_days away."""
    if len(s) < 2:
        return np.nan, None
    last = s.iloc[-1]
    target = last["end"] - pd.Timedelta(days=365)
    prior = s.iloc[:-1].copy()
    prior["gap"] = (prior["end"] - target).abs().dt.days
    cand = prior.nsmallest(1, "gap").iloc[0]
    if cand["gap"] > tol_days:
        return np.nan, None
    return _g(last["val"], cand["val"]), cand["end"].date()


def metrics(panel: pd.DataFrame, symbol: str, when=None, price: float | None = None) -> dict:
    """One symbol's metrics using only filings available at `when` (None = everything filed)."""
    S = lambda c, annual: F.series(panel, symbol, c, annual=annual, when=when)
    m = {"symbol": symbol}

    rev_a = S("revenue", True)
    rev_q = S("revenue", False)
    m["n_annual"] = len(rev_a)
    m["n_quarter"] = len(rev_q)
    if rev_a.empty and rev_q.empty:
        m["note"] = "无营收数据（可能是 ETF / 无 XBRL 申报）"
        return m

    # ---- period anchors
    if not rev_a.empty:
        m["fy_end"] = rev_a.end.iloc[-1].date()
        m["fy_filed"] = rev_a.filed.iloc[-1].date()
        m["revenue_fy"] = float(rev_a.val.iloc[-1])
        m["rev_g_fy"] = _g(rev_a.val.iloc[-1], rev_a.val.iloc[-2]) if len(rev_a) >= 2 else np.nan
        m["rev_cagr_3y"] = ((rev_a.val.iloc[-1] / rev_a.val.iloc[-4]) ** (1 / 3) - 1
                            if len(rev_a) >= 4 and rev_a.val.iloc[-4] > 0 else np.nan)
        m["tag_switches"] = int(rev_a.tag.nunique())     # >1 means the concept was re-tagged
    # latest quarter, and the same quarter a year earlier (4 periods back)
    if len(rev_q) >= 2:
        m["q_end"] = rev_q.end.iloc[-1].date()
        m["revenue_q"] = float(rev_q.val.iloc[-1])
        m["rev_g_q_yoy"], m["rev_q_yoy_base"] = _yoy(rev_q)
        # TTM only if the last 4 quarters really span ~a year; the missing-Q4 gap makes a naive
        # 4-row sum understate TTM revenue badly for filers whose Q4 is absent.
        if len(rev_q) >= 4:
            span = (rev_q.end.iloc[-1] - rev_q.end.iloc[-4]).days
            m["ttm_span_days"] = span
            m["revenue_ttm"] = (float(rev_q.val.iloc[-4:].sum()) if 250 <= span <= 300
                                else (m.get("revenue_fy", np.nan)))
        else:
            m["revenue_ttm"] = m.get("revenue_fy", np.nan)
    elif not rev_a.empty:
        m["revenue_ttm"] = m["revenue_fy"]

    # ---- margins and accounting quality
    def annual_latest(c):
        s = S(c, True)
        return float(s.val.iloc[-1]) if not s.empty else np.nan

    gp, cor = annual_latest("gross_profit"), annual_latest("cost_of_revenue")
    rev = m.get("revenue_fy", np.nan)
    if not np.isfinite(gp) and np.isfinite(cor) and np.isfinite(rev):
        gp = rev - cor
    m["gross_margin"] = gp / rev if np.isfinite(gp) and rev else np.nan
    oi, ni, ocf = annual_latest("operating_income"), annual_latest("net_income"), annual_latest("ocf")
    m["op_margin"] = oi / rev if np.isfinite(oi) and rev else np.nan
    m["net_margin"] = ni / rev if np.isfinite(ni) and rev else np.nan
    m["ocf_margin"] = ocf / rev if np.isfinite(ocf) and rev else np.nan
    # accruals: earnings not backed by cash. (NI - OCF)/revenue; positive = accrual-heavy
    m["accruals"] = (ni - ocf) / rev if np.isfinite(ni) and np.isfinite(ocf) and rev else np.nan

    sbc = annual_latest("sbc")
    m["sbc_pct_rev"] = sbc / rev if np.isfinite(sbc) and rev else np.nan
    m["rnd_pct_rev"] = annual_latest("rnd") / rev if rev else np.nan

    capex = annual_latest("capex")
    m["capex_pct_rev"] = capex / rev if np.isfinite(capex) and rev else np.nan
    cx = S("capex", True)
    rv = S("revenue", True)
    if len(cx) >= 4 and len(rv) >= 4:
        j = rv.set_index("end").val.reindex(cx.set_index("end").index)
        ratio = (cx.set_index("end").val / j).dropna()
        if len(ratio) >= 4:
            m["capex_pct_rev_4y_ago"] = float(ratio.iloc[-4])
            m["capex_accel"] = float(ratio.iloc[-1] - ratio.iloc[-4])

    # ---- related-party revenue: how much of growth is not arm's-length
    rp = S("revenue_related_party", True)
    if len(rp) >= 2 and not rev_a.empty and len(rev_a) >= 2:
        d_rp = rp.val.iloc[-1] - rp.val.iloc[-2]
        d_rev = rev_a.val.iloc[-1] - rev_a.val.iloc[-2]
        m["related_party_rev"] = float(rp.val.iloc[-1])
        m["related_party_pct_rev"] = float(rp.val.iloc[-1]) / rev if rev else np.nan
        m["related_party_pct_growth"] = float(d_rp / d_rev) if d_rev else np.nan

    # ---- forward indicators
    rpo = S("rpo", False)
    if rpo.empty:
        rpo = S("rpo", True)
    if len(rpo) >= 2:
        m["rpo"] = float(rpo.val.iloc[-1])
        m["rpo_g_yoy"], m["rpo_yoy_base"] = _yoy(rpo)
    ca = S("contract_asset", False)
    if len(ca) >= 2:
        m["contract_asset_g_yoy"], _ = _yoy(ca)

    # ---- dilution
    sh = S("shares_diluted", False)
    if len(sh) >= 1:
        m["shares"] = float(sh.val.iloc[-1])
        m["dilution_yoy"], _ = _yoy(sh)

    # ---- balance sheet
    def inst_latest(c):
        r = F.latest(panel, symbol, c, when=when, kind="instant")
        return _val(r)
    cash, sti, debt = inst_latest("cash"), inst_latest("short_term_inv"), inst_latest("debt_total")
    m["net_cash"] = np.nansum([cash, sti]) - (debt if np.isfinite(debt) else 0.0)

    # ---- valuation (needs a live price and share count)
    if price and np.isfinite(m.get("shares", np.nan)):
        mcap = price * m["shares"]
        ev = mcap - (m["net_cash"] if np.isfinite(m["net_cash"]) else 0.0)
        m["price"], m["mcap"], m["ev"] = price, mcap, ev
        ttm = m.get("revenue_ttm", np.nan)
        m["ev_sales"] = ev / ttm if np.isfinite(ttm) and ttm else np.nan
        if np.isfinite(gp) and rev:
            m["ev_gp"] = ev / (gp / rev * ttm) if np.isfinite(ttm) and ttm else np.nan
        eps = S("eps_diluted", False)
        if len(eps) >= 4:
            e_ttm = float(eps.val.iloc[-4:].sum())
            m["eps_ttm_gaap"] = e_ttm
            m["pe_gaap"] = price / e_ttm if e_ttm > 0 else np.nan
    return m


# ---------------------------------------------------------------- flags
def flags(m: dict) -> list[str]:
    """Risk markers, each with the threshold inline so a reader can disagree with it.
    Thresholds are judgement calls anchored on what was informative in real workups -- they are
    NOT calibrated against forward returns, because that study has not been run."""
    f = []
    g = lambda k: m.get(k, np.nan)

    if np.isfinite(g("sbc_pct_rev")) and g("sbc_pct_rev") > 0.15:
        f.append(f"SBC占营收{g('sbc_pct_rev')*100:.0f}%")
    if np.isfinite(g("related_party_pct_growth")) and g("related_party_pct_growth") > 0.30:
        f.append(f"关联方占增量{g('related_party_pct_growth')*100:.0f}%")
    if np.isfinite(g("related_party_pct_rev")) and g("related_party_pct_rev") > 0.10:
        f.append(f"关联方占营收{g('related_party_pct_rev')*100:.0f}%")
    # the ARM tell: forward book shrinking while reported revenue grows
    if (np.isfinite(g("rpo_g_yoy")) and np.isfinite(g("rev_g_fy"))
            and g("rpo_g_yoy") < 0 and g("rev_g_fy") > 0.10):
        f.append(f"RPO {g('rpo_g_yoy')*100:+.0f}% 而营收 {g('rev_g_fy')*100:+.0f}%")
    if np.isfinite(g("capex_accel")) and g("capex_accel") > 0.05:
        f.append(f"capex/营收 4年 {g('capex_pct_rev_4y_ago')*100:.0f}%→{g('capex_pct_rev')*100:.0f}%")
    if np.isfinite(g("accruals")) and g("accruals") > 0.05:
        f.append(f"应计项目{g('accruals')*100:+.0f}%营收(利润未落现金)")
    if np.isfinite(g("dilution_yoy")) and g("dilution_yoy") > 0.03:
        f.append(f"股数年增{g('dilution_yoy')*100:+.1f}%")
    if np.isfinite(g("contract_asset_g_yoy")) and np.isfinite(g("rev_g_q_yoy")) \
            and g("contract_asset_g_yoy") > g("rev_g_q_yoy") + 0.30:
        f.append("contract asset 增速远超营收")
    if np.isfinite(g("ev_sales")) and g("ev_sales") > 20:
        f.append(f"EV/Sales {g('ev_sales'):.0f}x")
    if np.isfinite(g("net_margin")) and g("net_margin") < 0:
        f.append(f"净利率{g('net_margin')*100:.0f}%")
    if np.isfinite(g("tag_switches")) and g("tag_switches") > 1:
        f.append(f"营收标签换过{g('tag_switches')}次(已按阶梯合并)")
    return f


def prices(symbols: list[str]) -> dict[str, float]:
    import yfinance as yf
    try:
        d = yf.download(symbols, period="5d", progress=False, auto_adjust=True)["Close"]
        if isinstance(d, pd.Series):
            return {symbols[0]: float(d.dropna().iloc[-1])}
        return {s: float(d[s].dropna().iloc[-1]) for s in d.columns if d[s].notna().any()}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--watchlist", action="store_true")
    ap.add_argument("--fetch", action="store_true", help="先抓取缺失标的的 SEC 数据")
    ap.add_argument("--out", default=None, help="写出 CSV")
    ap.add_argument("--verbose", action="store_true", help="逐标的展开明细")
    a = ap.parse_args()

    syms = [s.upper() for s in (a.symbols or (F.watchlist_symbols() if a.watchlist else ["ARM"]))]
    if a.fetch:
        print(f"抓取 {len(syms)} 个标的 ...")
        F.build(syms)
    panel = F.load()
    have = set(panel.symbol.unique())
    todo = [s for s in syms if s not in have]
    if todo:
        print(f"⚠️ 面板中缺少 {len(todo)} 个标的（加 --fetch 抓取）: {', '.join(todo[:10])}"
              + (" ..." if len(todo) > 10 else ""))
    syms = [s for s in syms if s in have]
    if not syms:
        print("没有可分析的标的。"); return

    px = prices(syms)
    rows = [metrics(panel, s, price=px.get(s)) for s in syms]
    for r in rows:
        r["flags"] = " | ".join(flags(r))
    df = pd.DataFrame(rows)

    show = ["symbol", "fy_end", "rev_g_fy", "rev_g_q_yoy", "gross_margin", "op_margin",
            "sbc_pct_rev", "capex_pct_rev", "accruals", "dilution_yoy", "rpo_g_yoy",
            "ev_sales", "pe_gaap", "flags"]
    show = [c for c in show if c in df.columns]
    out = df[show].copy()
    pct = ["rev_g_fy", "rev_g_q_yoy", "gross_margin", "op_margin", "sbc_pct_rev",
           "capex_pct_rev", "accruals", "dilution_yoy", "rpo_g_yoy"]
    for c in pct:
        if c in out:
            out[c] = out[c].map(lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "—")
    for c in ("ev_sales", "pe_gaap"):
        if c in out:
            out[c] = out[c].map(lambda x: f"{x:.1f}x" if pd.notna(x) else "—")

    hdr = {"symbol": "代码", "fy_end": "财年末", "rev_g_fy": "营收YoY", "rev_g_q_yoy": "季度YoY",
           "gross_margin": "毛利率", "op_margin": "营业利润率", "sbc_pct_rev": "SBC/营收",
           "capex_pct_rev": "capex/营收", "accruals": "应计", "dilution_yoy": "稀释",
           "rpo_g_yoy": "RPO YoY", "ev_sales": "EV/Sales", "pe_gaap": "PE(GAAP)", "flags": "标记"}
    print(f"\n{'='*150}")
    print(f"基本面确定性指标  {len(syms)} 标的  （point-in-time，来源 SEC XBRL；标记为风险提示，不进买卖分）")
    print("=" * 150)
    print(out.rename(columns=hdr).to_string(index=False))

    n_flag = int((df["flags"].str.len() > 0).sum())
    print(f"\n有标记 {n_flag}/{len(df)} 个标的")
    if a.verbose:
        for r in rows:
            if not r.get("flags"):
                continue
            print(f"\n--- {r['symbol']}  (年报 {r.get('n_annual')} 期 / 季报 {r.get('n_quarter')} 期, "
                  f"最新年报报送 {r.get('fy_filed')})")
            for k in ("revenue_fy", "revenue_ttm", "net_cash", "related_party_rev", "rpo", "shares"):
                if np.isfinite(r.get(k, np.nan)):
                    print(f"    {k:<22}{r[k]/1e6:>14,.0f} 百万")
            print(f"    标记: {r['flags']}")

    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        print(f"\n已写出 {p}")
    print("\n注意：这些标记是「去读一下这份财报」，不是「卖出」。基本面质量在本仓库从未被")
    print("      测量过有前瞻边际，阈值是判断而非校准结果，因此不进 观海买点分 也不进风险分。")


if __name__ == "__main__":
    main()
