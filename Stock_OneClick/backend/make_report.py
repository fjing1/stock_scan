#!/usr/bin/env python3
"""Generate dated Markdown scan reports from a scan_result workbook.

Reads ``scan_result_latest.xlsx`` (or ``--file``), which holds one sheet per
signal date, and writes one report per date into ``../reports/`` named
``scan_report_<YYYY-MM-DD>.md`` (date-sortable), plus a ``reports/README.md``
index sorted newest-first. Re-running overwrites cleanly, so it is safe to call
after every scan to keep the committed report history current.

Read-only on the workbook; no network. Run with the project venv:

    ../../vcp_env/bin/python make_report.py
    ../../vcp_env/bin/python make_report.py --file ../history/scan_result_20260605_143536.xlsx
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
import scan_stocks as scan  # noqa: E402  — canonical buy/sell scorers (single source of truth)
from gate_calc import gate  # noqa: E402  — regime/breadth exposure gate

BASE_DIR = BACKEND_DIR.parent
DEFAULT_WORKBOOK = BASE_DIR / "scan_result_latest.xlsx"
HISTORY_DIR = BASE_DIR / "history"
REPORTS_DIR = BASE_DIR / "reports"

CTX_LABELS = ["市场环境", "日线判断", "4H提示", "轮动判断", "指数快照", "策略提示"]
STOP_MARKERS = ("Top5统计", "排名", "触发样本", "市场环境", "No signals", "信号快照")


def _is_stop(cell) -> bool:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return True
    s = str(cell).strip()
    return (not s) or s.lower() == "nan" or any(m in s for m in STOP_MARKERS)


def _last_pct(row_vals, pct_cols):
    fwd, day = np.nan, 0
    for ci, dnum in pct_cols:
        v = pd.to_numeric(row_vals[ci], errors="coerce") if ci < len(row_vals) else np.nan
        if pd.notna(v):
            fwd, day = float(v), dnum
    return fwd, day


def _dnum(h):
    try:
        return int(str(h)[1:].split("_", 1)[0])
    except Exception:
        return 0


def _build_score_lookup() -> dict:
    """Map (symbol, signal_date_iso, side) -> score, from the indicator fields
    in RawSignals across the latest workbook + recent history runs. Sells get a
    score even though the per-date follow-up sheets don't carry one (those sheets
    lack the rank120/RSI/etc. columns the scorer needs). Uses the engine's
    canonical scorers so the logic stays in one place."""
    lut: dict = {}
    paths = [DEFAULT_WORKBOOK] if DEFAULT_WORKBOOK.exists() else []
    pat = re.compile(r"scan_result_(\d{8})_")
    for p in sorted(HISTORY_DIR.glob("scan_result_*.xlsx")):
        m = pat.search(p.name)
        if m and m.group(1) >= "20260520":   # recent (current-format) runs only
            paths.append(p)
    for p in paths:
        for side, fn in (("BUY", scan.score_buy_signal_row), ("SELL", scan.score_sell_signal_row)):
            try:
                rows = scan._read_signal_rows_from_result(p, side)
            except Exception:
                continue
            if rows is None or rows.empty or not {"symbol", "signal_date"}.issubset(rows.columns):
                continue
            rows = rows.copy()
            rows["symbol"] = rows["symbol"].astype(str).str.strip().str.upper()
            rows["signal_date"] = pd.to_datetime(rows["signal_date"], errors="coerce").dt.date
            for _, r in rows.iterrows():
                d = r["signal_date"]
                if pd.isna(d):
                    continue
                sc = fn(r)
                if pd.isna(sc):
                    continue
                key = (r["symbol"], d.isoformat(), side)
                lut[key] = sc if key not in lut else max(lut[key], sc)
    return lut


def _parse_date_sheet(raw: pd.DataFrame):
    """Return (context: dict, buys: list[dict], sells: list[dict])."""
    ctx, buys, sells = {}, [], []
    section = None
    for r in range(len(raw)):
        rowvals = raw.iloc[r].tolist()
        c0 = str(rowvals[0]) if pd.notna(rowvals[0]) else ""
        if c0 in CTX_LABELS and len(rowvals) > 1 and pd.notna(rowvals[1]):
            ctx[c0] = str(rowvals[1])
            continue
        if "买入触发样本" in c0:
            section = "buy"; continue
        if "卖出触发样本" in c0:
            section = "sell"; continue
        header = [str(x) if pd.notna(x) else "" for x in rowvals]
        if "symbol" in header and "D0_date" in header:
            col = {h: header.index(h) for h in
                   ["symbol", "观海买点分", "板块", "D0_rule", "D0_close"] if h in header}
            pct_cols = sorted([(i, _dnum(header[i])) for i in range(len(header))
                               if str(header[i]).endswith("_pct_vs_D0")], key=lambda t: t[1])
            bucket = buys if section == "buy" else sells if section == "sell" else None
            if bucket is None:
                continue
            for rr in range(r + 1, len(raw)):
                vals = raw.iloc[rr].tolist()
                sym = vals[col["symbol"]] if col.get("symbol", 0) < len(vals) else None
                if _is_stop(sym):
                    break
                fwd, day = _last_pct(vals, pct_cols)
                bucket.append({
                    "symbol": str(sym).strip().upper(),
                    "score": pd.to_numeric(vals[col["观海买点分"]], errors="coerce") if "观海买点分" in col else np.nan,
                    "sector": str(vals[col["板块"]]) if "板块" in col and pd.notna(vals[col["板块"]]) else "",
                    "rule": str(vals[col["D0_rule"]]) if "D0_rule" in col and pd.notna(vals[col["D0_rule"]]) else "",
                    "d0_close": pd.to_numeric(vals[col["D0_close"]], errors="coerce") if "D0_close" in col else np.nan,
                    "fwd": fwd, "day": day,
                })
    return ctx, buys, sells


def _fmt_pct(v):
    return "—" if pd.isna(v) else f"{v:+.2%}"


def _fmt_score(v):
    return "—" if pd.isna(v) else f"{v:.0f}"


def _state_of(ctx) -> str:
    raw = ctx.get("市场环境", "")
    return raw.split("（")[0].strip() or "—"


def _write_report(date: str, ctx: dict, buys: list, sells: list, run_stamp: str, score_lookup: dict) -> str:
    L = []
    L.append(f"# 📊 Scan Report — {date}")
    L.append("")
    L.append(f"*Source: `scan_result_latest.xlsx` · scan run {run_stamp} · generated by `backend/make_report.py`*")
    L.append("")
    # market environment
    L.append("## 🌡️ Market environment")
    L.append("")
    L.append("| | |")
    L.append("|---|---|")
    for k in CTX_LABELS:
        if k in ctx:
            L.append(f"| **{k}** | {ctx[k]} |")
    L.append("")
    # exposure gate (from gate_calc) — the actionable exposure decision for this date
    buy_n, sell_n = len(buys), len(sells)
    sell_share = round(sell_n / (buy_n + sell_n), 3) if (buy_n + sell_n) else float("nan")
    state = _state_of(ctx)
    tg, hedge, action = gate(state, sell_share)
    L.append(f"**📐 Exposure gate:** state `{state}` · SELL_share `{sell_share}` "
             f"→ **target long gross {tg}%** · hedge: {hedge}")
    L.append(f"> {action}")
    L.append("")
    # buys
    scored = [b for b in buys if pd.notna(b["score"])]
    scored.sort(key=lambda b: (-(b["score"] if pd.notna(b["score"]) else -1), b["symbol"]))
    L.append(f"## 🟢 Buy samples ({len(buys)})")
    L.append("")
    if buys:
        with_fwd = [b for b in buys if pd.notna(b["fwd"])]
        if with_fwd:
            mean_fwd = float(np.mean([b["fwd"] for b in with_fwd]))
            hit = np.mean([1.0 if b["fwd"] > 0 else 0.0 for b in with_fwd])
            L.append(f"*Forward (close-to-close from D0, last tracked day): mean {mean_fwd:+.2%}, "
                     f"hit-rate {hit:.0%} over {len(with_fwd)} with data.*")
            L.append("")
        L.append("| 观海买点分 | symbol | sector | rule | D0 close | fwd % (D) |")
        L.append("|:--:|--------|--------|------|--------:|:--:|")
        for b in (scored if scored else buys):
            d = f"{_fmt_pct(b['fwd'])} (D{b['day']})" if pd.notna(b["fwd"]) else "—"
            L.append(f"| {_fmt_score(b['score'])} | **{b['symbol']}** | {b['sector']} | "
                     f"{b['rule'].replace('|','+')} | {b['d0_close']:.2f} | {d} |")
    else:
        L.append("_No buy triggers this date._")
    L.append("")
    # sells — attach the sell-conviction score (卖出分) from the indicator data
    for s in sells:
        s["score"] = score_lookup.get((s["symbol"], date, "SELL"), np.nan)
    sells_sorted = sorted(sells, key=lambda s: (-(s["score"] if pd.notna(s["score"]) else -1.0), s["symbol"]))
    L.append(f"## 🔴 Sell samples ({len(sells)})")
    L.append("")
    if sells:
        with_fwd = [s for s in sells if pd.notna(s["fwd"])]
        if with_fwd:
            mean_fwd = float(np.mean([s["fwd"] for s in with_fwd]))
            down = np.mean([1.0 if s["fwd"] < 0 else 0.0 for s in with_fwd])
            L.append(f"*Forward (close-to-close from D0, last tracked day): mean {mean_fwd:+.2%}; "
                     f"price fell on {down:.0%} of {len(with_fwd)} with data (for a sell, lower = correct).*")
            L.append("")
        L.append("| 卖出分 | symbol | sector | rule | D0 close | fwd % (D) |")
        L.append("|:--:|--------|--------|------|--------:|:--:|")
        for s in sells_sorted:
            d = f"{_fmt_pct(s['fwd'])} (D{s['day']})" if pd.notna(s["fwd"]) else "—"
            L.append(f"| {_fmt_score(s['score'])} | **{s['symbol']}** | {s['sector']} | "
                     f"{s['rule'].replace('|','+')} | {s['d0_close']:.2f} | {d} |")
    else:
        L.append("_No sell triggers this date._")
    L.append("")
    L.append("---")
    L.append("*观海买点分 = 0–100 **buy** score · 卖出分 = 0–100 **sell**-conviction score "
             "(both: higher = stronger). fwd % = forward return from the D0 anchor close, "
             "no costs. Not investment advice.*")
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=str(DEFAULT_WORKBOOK), help="scan_result workbook to read")
    args = ap.parse_args()

    wb = Path(args.file)
    if not wb.exists():
        print(f"Workbook not found: {wb}")
        return 1
    xls = pd.ExcelFile(wb)

    run_stamp = "?"
    try:
        rs = pd.read_excel(xls, "RawSignals")
        if not rs.empty and {"run_date", "run_time"}.issubset(rs.columns):
            rd = pd.to_datetime(rs["run_date"].dropna().iloc[0], errors="coerce")
            rd = rd.date() if pd.notna(rd) else str(rs["run_date"].dropna().iloc[0])
            run_stamp = f"{rd} {str(rs['run_time'].dropna().iloc[0])}"
    except Exception:
        pass

    date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    date_sheets = [s for s in xls.sheet_names if date_re.match(s)]
    date_sheets.sort(reverse=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    score_lookup = _build_score_lookup()
    index_rows = []
    written = 0
    for date in date_sheets:
        raw = pd.read_excel(xls, date, header=None)
        ctx, buys, sells = _parse_date_sheet(raw)
        md = _write_report(date, ctx, buys, sells, run_stamp, score_lookup)
        (REPORTS_DIR / f"scan_report_{date}.md").write_text(md, encoding="utf-8")
        written += 1
        top = ""
        scored = [b for b in buys if pd.notna(b["score"])]
        if scored:
            t = max(scored, key=lambda b: b["score"])
            top = f"{t['symbol']} ({t['score']:.0f})"
        top_sell = ""
        sell_sc = [(score_lookup.get((s["symbol"], date, "SELL"), np.nan), s["symbol"]) for s in sells]
        sell_sc = [(sc, sym) for sc, sym in sell_sc if pd.notna(sc)]
        if sell_sc:
            sc, sym = max(sell_sc)
            top_sell = f"{sym} ({sc:.0f})"
        index_rows.append((date, _state_of(ctx), len(buys), len(sells), top, top_sell))

    # index
    idx = ["# Scan Reports",
           "",
           "Dated Markdown reports generated from `scan_result_latest.xlsx` by "
           "`backend/make_report.py`. One file per signal date, newest first. "
           "Filenames are ISO-dated so they sort chronologically.",
           "",
           f"*Latest scan run: {run_stamp} · {written} dated reports.*",
           "",
           "| Date | Market state | Buys | Sells | Top buy | Top sell | Report |",
           "|------|--------------|:----:|:-----:|---------|----------|--------|"]
    for date, state, nb, ns, top, tops in index_rows:
        idx.append(f"| {date} | {state} | {nb} | {ns} | {top} | {tops} | [report](scan_report_{date}.md) |")
    idx.append("")
    (REPORTS_DIR / "README.md").write_text("\n".join(idx), encoding="utf-8")

    print(f"Wrote {written} reports + index to {REPORTS_DIR}")
    for date, state, nb, ns, top, tops in index_rows:
        print(f"  {date}  {state:<10}  buys={nb:<3} sells={ns:<3} top_buy={top:<14} top_sell={tops}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
