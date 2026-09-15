"""
fund_forward.py — the ledger that makes the fundamental layer falsifiable.

Layer 4. Every verdict the fundamental stack produces gets registered here with DATED, NUMERIC,
FALSIFIABLE checkpoints, plus the price and benchmark level on the day it was made. Later, this
same module resolves those checkpoints against filings and computes the realised excess return of
each call. In 12 months it answers the only question that matters about an analysis layer:
"did its verdicts predict anything?"

WHY THIS FILE EXISTS BEFORE THE AGENT DOES ANY MORE WORK: this repo has already made exactly this
mistake once. From its own notes -- the 905 enable=0 research-pool names have no forward returns
attached, so none of the opinions formed about them can ever be backtested. An analysis layer with
no ledger is indistinguishable from a layer that generates pleasant narratives, and the
indistinguishability is permanent: you cannot reconstruct point-in-time conviction after the fact.

DESIGN CHOICES THAT KEEP IT HONEST

  * APPEND-ONLY JSONL. A record is never edited in place except to fill resolution fields. If a
    view changes, register a NEW record that supersedes the old one by id -- so the ledger keeps the
    fact that the view changed, which is itself data about the layer's stability.
  * PRICE AND BENCHMARK CAPTURED AT CALL TIME. Not looked up later. A verdict without the
    contemporaneous price is unscoreable, and "what was the price on that date" is exactly the kind
    of thing that gets silently mis-fetched.
  * EXCESS RETURN, NOT RAW. A bearish call on a stock that fell 20% in a market that fell 25% was
    wrong. Scored against SPY and, where given, a sector benchmark.
  * TWO RESOLVER KINDS, HONESTLY LABELLED. `auto` checkpoints map to a key that fund_metrics.py
    computes from XBRL and resolve themselves. `manual` checkpoints are things XBRL does not tag --
    segment royalty revenue, litigation outcomes, whether a disclosure was withdrawn -- and they
    stay pending until a human or an agent fills them in. Pretending the second kind is automatic
    is how a ledger quietly fills with fiction.
  * MINIMUM SAMPLE BEFORE ANY CONCLUSION. scorecard() refuses to report hit rates below MIN_N
    resolved verdicts. With n=1 the number would be 0% or 100% and both are meaningless; this repo
    has killed six signals that looked real at small n.

Usage:
    ../../vcp_env/bin/python fund_forward.py --backfill-arm        # seed the ARM verdict
    ../../vcp_env/bin/python fund_forward.py --list
    ../../vcp_env/bin/python fund_forward.py --resolve             # settle anything now due
    ../../vcp_env/bin/python fund_forward.py --score
"""
from __future__ import annotations

import argparse
import json
import uuid
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
LEDGER = HERE / "_fund_forward.jsonl"
MIN_N = 8                      # below this, no hit rates are reported
HORIZONS = (21, 63, 126, 252)  # trading days at which excess return is scored


# ---------------------------------------------------------------- storage
def _read() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]


def _write(recs: list[dict]):
    LEDGER.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in recs))


def _px(symbols: list[str], on: str) -> dict[str, float]:
    """Closing price on (or the last close before) `on`. Captured at registration so the record is
    self-contained and never depends on a later re-fetch."""
    import yfinance as yf
    end = (pd.Timestamp(on) + pd.Timedelta(days=6)).strftime("%Y-%m-%d")
    start = (pd.Timestamp(on) - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    d = yf.download(symbols, start=start, end=end, progress=False, auto_adjust=True)["Close"]
    if isinstance(d, pd.Series):
        d = d.to_frame(symbols[0])
    d = d[d.index <= pd.Timestamp(on)]
    return {s: float(d[s].dropna().iloc[-1]) for s in d.columns if d[s].notna().any()}


# ---------------------------------------------------------------- register
def register(symbol: str, verdict: str, thesis: str, checkpoints: list[dict],
             *, date: str, source: str = "agent", benchmark: str = "SPY",
             sector: str | None = None, notes: str = "", supersedes: str | None = None) -> dict:
    """Append one verdict.

    verdict: supports / mixed / undermines / irrelevant -- the same vocabulary the research
             workflow emits, so agent output maps in without translation.
    checkpoints: [{due, metric, threshold, direction, meaning, resolver}] where
             direction is 'above'|'below', resolver is 'auto'|'manual'.
    """
    if verdict not in ("supports", "mixed", "undermines", "irrelevant"):
        raise ValueError(f"unknown verdict {verdict!r}")
    for c in checkpoints:
        missing = {"due", "metric", "direction", "meaning"} - set(c)
        if missing:
            raise ValueError(f"checkpoint missing {missing}: {c}")
        if c["direction"] not in ("above", "below"):
            raise ValueError(f"direction must be above/below: {c}")
        c.setdefault("resolver", "manual")
        c.setdefault("status", "pending")
        c.setdefault("resolved_value", None)
        c.setdefault("resolved_on", None)

    syms = [symbol, benchmark] + ([sector] if sector else [])
    prices = _px(syms, date)
    rec = {
        "id": uuid.uuid4().hex[:12], "date": date, "symbol": symbol.upper(),
        "source": source, "verdict": verdict, "thesis": thesis, "notes": notes,
        "price": prices.get(symbol.upper()), "benchmark": benchmark,
        "benchmark_price": prices.get(benchmark),
        "sector": sector, "sector_price": prices.get(sector) if sector else None,
        "checkpoints": checkpoints, "supersedes": supersedes,
    }
    recs = _read()
    recs.append(rec)
    _write(recs)
    return rec


# ---------------------------------------------------------------- resolve
# checkpoint metric -> key produced by fund_metrics.metrics()
AUTO_METRICS = {
    "revenue_growth_fy": "rev_g_fy", "revenue_growth_q_yoy": "rev_g_q_yoy",
    "rpo_growth_yoy": "rpo_g_yoy", "sbc_pct_revenue": "sbc_pct_rev",
    "capex_pct_revenue": "capex_pct_rev", "accruals": "accruals",
    "dilution_yoy": "dilution_yoy", "gross_margin": "gross_margin",
    "op_margin": "op_margin", "net_margin": "net_margin",
    "related_party_pct_growth": "related_party_pct_growth",
    "related_party_pct_revenue": "related_party_pct_rev",
    "ev_sales": "ev_sales", "revenue_ttm": "revenue_ttm",
}


def resolve(today: str | None = None, refetch: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Settle every checkpoint whose due date has passed.

    `auto` checkpoints are recomputed from the XBRL panel. Note the deliberate asymmetry: we do NOT
    restrict to filings available on the due date -- we are SCORING, so the latest filed figure is
    the right one. Point-in-time discipline applies to prediction, not to marking the exam."""
    import _fund_data as F
    import fund_metrics as FM

    today = pd.Timestamp(today or pd.Timestamp.now().date())
    recs = _read()
    if not recs:
        print("台账为空。")
        return pd.DataFrame()

    try:
        panel = F.load()
    except FileNotFoundError:
        panel = None
    if refetch and panel is not None:
        syms = sorted({r["symbol"] for r in recs})
        F.build(syms, verbose=False)
        panel = F.load()

    cache: dict[str, dict] = {}
    rows = []
    for r in recs:
        for c in r["checkpoints"]:
            due = pd.Timestamp(c["due"])
            if c["status"] != "pending" or due > today:
                rows.append({**_flat(r, c), "action": "—"})
                continue
            if c.get("resolver") == "auto" and panel is not None:
                key = AUTO_METRICS.get(c["metric"])
                if key is None:
                    rows.append({**_flat(r, c), "action": f"auto 但无映射: {c['metric']}"})
                    continue
                if r["symbol"] not in cache:
                    cache[r["symbol"]] = FM.metrics(panel, r["symbol"])
                val = cache[r["symbol"]].get(key, np.nan)
                if not np.isfinite(val):
                    rows.append({**_flat(r, c), "action": "数据缺失，仍 pending"})
                    continue
                # FRESHNESS GUARD. A due date passing does not mean new data exists. Without this,
                # the resolver compares the SAME figure the verdict was written against to its own
                # threshold and emits a meaningless hit/miss -- observed: an FY2026 gross-margin
                # checkpoint "resolved" as miss at 0.4691 vs a 0.4691 threshold, because only
                # FY2025 had been filed. Require a filing dated after the verdict.
                newest = panel[(panel.symbol == r["symbol"])].filed.max()
                if pd.isna(newest) or newest <= pd.Timestamp(r["date"]):
                    rows.append({**_flat(r, c),
                                 "action": f"到期但无新申报（最新报送 "
                                           f"{newest.date() if pd.notna(newest) else '无'}"
                                           f" <= 判断日 {r['date']}），仍 pending"})
                    continue
                thr = c.get("threshold")
                hit = (val > thr) if c["direction"] == "above" else (val < thr)
                c["status"] = "hit" if hit else "miss"
                c["resolved_value"] = float(val)
                c["resolved_on"] = str(today.date())
                rows.append({**_flat(r, c), "action": f"已判定 {c['status']} (实际 {val:.4g})"})
            else:
                rows.append({**_flat(r, c), "action": "到期待人工核对（XBRL 无此 tag）"})
    _write(recs)
    df = pd.DataFrame(rows)
    if verbose and not df.empty:
        print(df.to_string(index=False))
    return df


def _flat(r: dict, c: dict) -> dict:
    return {"symbol": r["symbol"], "date": r["date"], "verdict": r["verdict"],
            "due": c["due"], "metric": c["metric"], "dir": c["direction"],
            "threshold": c.get("threshold"), "resolver": c.get("resolver"),
            "status": c["status"]}


# ---------------------------------------------------------------- score
def scorecard(today: str | None = None) -> dict:
    """Realised excess return of each verdict, plus checkpoint hit rate once n is large enough."""
    import yfinance as yf

    recs = _read()
    if not recs:
        print("台账为空。")
        return {}
    today = pd.Timestamp(today or pd.Timestamp.now().date())
    syms = sorted({r["symbol"] for r in recs} | {r["benchmark"] for r in recs}
                  | {r["sector"] for r in recs if r.get("sector")})
    start = min(pd.Timestamp(r["date"]) for r in recs) - pd.Timedelta(days=10)
    d = yf.download(syms, start=start.strftime("%Y-%m-%d"), progress=False,
                    auto_adjust=True)["Close"]
    if isinstance(d, pd.Series):
        d = d.to_frame(syms[0])

    rows = []
    for r in recs:
        s, b = r["symbol"], r["benchmark"]
        if s not in d.columns:
            continue
        ser_all = d[s].dropna()
        if ser_all.empty:
            continue
        ser = ser_all[ser_all.index >= pd.Timestamp(r["date"])]
        if ser.empty:
            # Registered today, or the provider has not published the bar for the record date yet.
            # Fall back to the newest available close so the row shows 0% instead of disappearing:
            # a scorecard that silently drops records is worse than one that reports zero elapsed.
            ser = ser_all.tail(1)
        row = {"id": r["id"], "date": r["date"], "symbol": s, "verdict": r["verdict"],
               "days_elapsed": int((today - pd.Timestamp(r["date"])).days),
               "price_at_call": r["price"], "price_now": float(ser.iloc[-1])}
        row["ret"] = row["price_now"] / row["price_at_call"] - 1 if r["price"] else np.nan
        for name, bench in (("spy", b), ("sec", r.get("sector"))):
            if not bench or bench not in d.columns:
                continue
            bs = d[bench].dropna()
            bs = bs[bs.index >= pd.Timestamp(r["date"])]
            base = r["benchmark_price"] if name == "spy" else r.get("sector_price")
            if bs.empty or not base:
                continue
            row[f"{name}_ret"] = float(bs.iloc[-1]) / base - 1
            row[f"excess_{name}"] = row["ret"] - row[f"{name}_ret"]
        for h in HORIZONS:
            if len(ser) > h:
                row[f"ret_d{h}"] = float(ser.iloc[h]) / row["price_at_call"] - 1
        cps = r["checkpoints"]
        row["cp_total"] = len(cps)
        row["cp_hit"] = sum(1 for c in cps if c["status"] == "hit")
        row["cp_miss"] = sum(1 for c in cps if c["status"] == "miss")
        row["cp_pending"] = sum(1 for c in cps if c["status"] == "pending")
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"台账有 {len(recs)} 条记录，但没有一条能取到价格序列（数据源问题）。"
              f"涉及标的: {', '.join(syms)}")
        return {"df": df, "n": 0, "resolved": 0}
    print("=" * 112)
    print(f"基本面判断台账  {len(df)} 条  （每条判断都带成交价与基准，可计算超额收益）")
    print("=" * 112)
    show = [c for c in ["date", "symbol", "verdict", "days_elapsed", "price_at_call",
                        "price_now", "ret", "excess_spy", "excess_sec",
                        "cp_hit", "cp_miss", "cp_pending"] if c in df.columns]
    o = df[show].copy()
    for c in ("ret", "excess_spy", "excess_sec"):
        if c in o:
            o[c] = o[c].map(lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "—")
    print(o.to_string(index=False))

    resolved = int(df["cp_hit"].sum() + df["cp_miss"].sum()) if "cp_hit" in df.columns else 0
    pending = int(df["cp_pending"].sum()) if "cp_pending" in df.columns else 0
    n_verdicts = len(df)
    print(f"\n已判定 checkpoint {resolved} 个，待判定 {pending} 个")
    if n_verdicts < MIN_N:
        print(f"⚠️ 判断数 {n_verdicts} < {MIN_N}，**不报命中率**。")
        print("   n=1 时命中率只能是 0% 或 100%，两者都无意义；本仓库已有六个在小样本上")
        print("   看起来成立、实测归零的信号。等样本够了这里才会出数字。")
    else:
        for v in df["verdict"].unique():
            sub = df[df["verdict"] == v]
            if "excess_spy" in sub.columns and sub["excess_spy"].notna().any():
                print(f"  {v:<12} n={len(sub):>3}  中位超额(vs SPY) "
                      f"{sub['excess_spy'].median()*100:+.1f}%")
    return {"df": df, "n": n_verdicts, "resolved": resolved}


# ---------------------------------------------------------------- ARM backfill
def backfill_arm(date: str = "2026-09-14") -> dict:
    """Seed the ledger with the ARM verdict from the 2026-09-14 deep research.

    Every checkpoint below came out of that workup with a number attached. Note that three of the
    five are `manual`: segment royalty revenue, the SoftBank-affiliate share of growth, and the
    external-vs-related-party license split are all disclosed in prose and note tables, NOT in a
    standard us-gaap tag, so XBRL cannot settle them. Labelling them auto would make the ledger
    look self-maintaining while quietly never resolving."""
    return register(
        symbol="ARM", verdict="undermines", date=date, source="workflow:arm-thesis-research",
        benchmark="SPY", sector="SMH",
        thesis="持有者论点为「ARM 架构无处不在 + 苹果用它 + 苹果发新机 ⇒ 长期看多」。"
               "研究裁定因果链断裂：苹果在 FY2026 20-F 全文出现 1 次（1990 年历史段落），"
               "iPhone 出现 0 次；iPhone 台数 +5% 对 ARM 营收的影响上限 +0.41%；"
               "手机 AP 占 royalty 池比例在下降（46%→43%），FY2026 royalty 增量 72% 来自非手机。"
               "真正的问题是估值：EV/Sales 48.8x，反向 DCF 需要 10 年 27-34% CAGR。",
        notes="成本 257 位于上市以来收盘价第 92 百分位。−45.6% 回撤是纯估值压缩，非预期下调。"
              "证据文件 _arm_*（383 个），主源 20-F acc 0001973239-26-000097。",
        checkpoints=[
            {"due": "2026-11-04", "metric": "royalty_revenue_quarterly", "threshold": 737e6,
             "direction": "above", "resolver": "manual",
             "meaning": "royalty 能否突破三个季度以来的 $737M 高点。若再次低于，"
                        "则「非数据中心基本盘走平」被确认（分部数据只在股东信/附注，XBRL 无 tag）"},
            {"due": "2026-11-04", "metric": "softbank_pct_of_growth", "threshold": 0.20,
             "direction": "below", "resolver": "manual",
             "meaning": "SoftBank 关联方 consulting 占营收增量的比例。<20% 说明 FY2026 的 61.2% "
                        "是一次性；再次 >50% 说明依赖已固化"},
            {"due": "2026-11-04", "metric": "external_license_growth_yoy", "threshold": 0.0,
             "direction": "above", "resolver": "manual",
             "meaning": "外部 license & other 同比。Q1 FY27 已从 −8.7% 恢复到 +21.6%，"
                        "若回到负值说明 FY2026 的下滑不是时点波动"},
            {"due": "2026-11-04", "metric": "rpo_growth_yoy", "threshold": 0.0,
             "direction": "above", "resolver": "auto",
             "meaning": "RPO 能否转正增长。两年 −16.6% 而同期营收 +52.2%；"
                        "ARM 已把 RPO 从股东信撤下，但仍留在中期报表附注（XBRL 可取）"},
            {"due": "2026-12-31", "metric": "qualcomm_countersuit_outcome", "threshold": 0.0,
             "direction": "above", "resolver": "manual",
             "meaning": "高通反诉（含反竞争指控）预计 2026 Q4 开庭。高通是 ARM 营收的 9%，"
                        "而 ARM 未计提任何诉讼准备"},
        ])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill-arm", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--resolve", action="store_true")
    ap.add_argument("--refetch", action="store_true", help="resolve 前重新抓取 SEC 数据")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()

    if a.backfill_arm:
        existing = [r for r in _read() if r["symbol"] == "ARM"]
        if existing:
            print(f"ARM 已在台账中（{len(existing)} 条），跳过。要重记请先手动编辑 {LEDGER.name}")
        else:
            r = backfill_arm()
            print(f"已登记 {r['symbol']} {r['verdict']}  价格 {r['price']:.2f}  "
                  f"基准 {r['benchmark']} {r['benchmark_price']:.2f}  "
                  f"行业 {r['sector']} {r['sector_price']:.2f}")
            print(f"  {len(r['checkpoints'])} 个 checkpoint，其中 "
                  f"{sum(1 for c in r['checkpoints'] if c['resolver']=='auto')} 个可自动判定")
    if a.list:
        for r in _read():
            print(f"\n[{r['date']}] {r['symbol']} {r['verdict']}  @{r['price']}  ({r['source']})")
            print(f"  {r['thesis'][:150]}...")
            for c in r["checkpoints"]:
                mark = {"pending": "·", "hit": "✓", "miss": "✗"}[c["status"]]
                print(f"   {mark} {c['due']}  {c['metric']} {c['direction']} {c['threshold']}"
                      f"  [{c['resolver']}]")
    if a.resolve:
        resolve(refetch=a.refetch)
    if a.score:
        scorecard()
    if not any([a.backfill_arm, a.list, a.resolve, a.score]):
        ap.print_help()
