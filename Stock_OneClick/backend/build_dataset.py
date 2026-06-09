#!/usr/bin/env python3
"""Build a clean, flat, labeled signal dataset for strategy analysis.

Reads the per-date follow-up sheets across `scan_result_latest.xlsx` + recent
`history/scan_result_*.xlsx` and emits ONE row per (symbol, signal_date, side)
with: market-regime state, sector, signal_type, score (观海买点分 for buys /
卖出分 for sells), the scorer SUB-FEATURES (rank120, RSI, L2_trend, H4_RSI,
H4_FJ — joined from RawSignals), D0 close, and forward returns
(fwd_d1/d3/d5/d10/d14 + fwd_last + days). Forward returns are fractions
(0.05 = +5%) close-to-close from the D0 anchor.

This is the single source of truth for `score_calibration.py` and `gate_calc.py`.
Read-only on the workbooks; no network. Run with the project venv:

    ../../vcp_env/bin/python build_dataset.py                 # -> ../reports/strategy_dataset.csv
    ../../vcp_env/bin/python build_dataset.py --out /tmp/x.csv --min-run-date 20260520
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
import scan_stocks as scan  # noqa: E402  — canonical scorers + RawSignals reader

BASE_DIR = BACKEND_DIR.parent
LATEST = BASE_DIR / "scan_result_latest.xlsx"
HIST = BASE_DIR / "history"
DEFAULT_OUT = BASE_DIR / "reports" / "strategy_dataset.csv"

CTX_LABELS = {"市场环境", "日线判断", "4H提示", "轮动判断", "指数快照", "策略提示"}
STOP = ("Top5统计", "排名", "触发样本", "市场环境", "信号快照", "No signals")
SUBFEATS = ["rank120", "RSI", "L2_trend", "L2_pump", "H4_RSI", "H4_FJ"]


def _is_stop(c) -> bool:
    if c is None or (isinstance(c, float) and pd.isna(c)):
        return True
    s = str(c).strip()
    return (not s) or s.lower() == "nan" or any(m in s for m in STOP)


def _dnum(h) -> int:
    try:
        return int(str(h)[1:].split("_", 1)[0])
    except Exception:
        return 0


def _run_files(min_run_date: str):
    pat = re.compile(r"scan_result_(\d{8})_")
    paths = [str(LATEST)] if LATEST.exists() else []
    for p in sorted(glob.glob(str(HIST / "scan_result_*.xlsx"))):
        m = pat.search(os.path.basename(p))
        if m and m.group(1) >= min_run_date:
            paths.append(p)
    return paths


def build_lut(min_run_date: str) -> dict:
    """(symbol, date_iso, side) -> {score, types:set, <subfeatures>} from RawSignals."""
    lut: dict = {}
    for p in _run_files(min_run_date):
        for side, fn in (("BUY", scan.score_buy_signal_row), ("SELL", scan.score_sell_signal_row)):
            try:
                rows = scan._read_signal_rows_from_result(Path(p), side)
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
                k = (r["symbol"], d.isoformat(), side)
                rec = lut.setdefault(k, {"score": np.nan, "types": set(), **{f: np.nan for f in SUBFEATS}})
                if pd.notna(sc):
                    rec["score"] = sc if pd.isna(rec["score"]) else max(rec["score"], sc)
                st = str(r.get("signal_type", "") or "")
                if st:
                    rec["types"].add(st)
                for f in SUBFEATS:
                    v = pd.to_numeric(r.get(f, np.nan), errors="coerce")
                    if pd.notna(v) and pd.isna(rec[f]):
                        rec[f] = float(v)
    return lut


def parse_date_sheet(xls, sheet):
    """Return (state, rows[]) where each row has side/symbol/sector/rule/d0_close/
    buy_score_sheet and a horizon dict h={D-day: pct_vs_D0}."""
    raw = pd.read_excel(xls, sheet, header=None)
    state = ""
    for _, r in raw.iterrows():
        if str(r.iloc[0]) == "市场环境" and len(r) > 1 and pd.notna(r.iloc[1]):
            state = str(r.iloc[1]).split("（")[0].strip()
            break
    out, section, rr, n = [], None, 0, len(raw)
    while rr < n:
        hdr = [str(x) if pd.notna(x) else "" for x in raw.iloc[rr].tolist()]
        c0 = hdr[0]
        if "买入触发样本" in c0:
            section = "BUY"
        elif "卖出触发样本" in c0:
            section = "SELL"
        if "symbol" in hdr and "D0_date" in hdr and section:
            ci = {h: hdr.index(h) for h in ["symbol", "观海买点分", "板块", "D0_rule", "D0_close"] if h in hdr}
            pcs = sorted([(i, _dnum(hdr[i])) for i in range(len(hdr)) if str(hdr[i]).endswith("_pct_vs_D0")],
                         key=lambda t: t[1])
            j = rr + 1
            while j < n:
                vals = raw.iloc[j].tolist()
                sym = vals[ci["symbol"]] if ci["symbol"] < len(vals) else None
                if _is_stop(sym):
                    break
                hz = {dn: (pd.to_numeric(vals[i], errors="coerce") if i < len(vals) else np.nan) for i, dn in pcs}
                out.append({
                    "side": section, "symbol": str(sym).strip().upper(), "state": state,
                    "sector": str(vals[ci["板块"]]) if "板块" in ci and pd.notna(vals[ci["板块"]]) else "",
                    "rule": str(vals[ci["D0_rule"]]) if "D0_rule" in ci and pd.notna(vals[ci["D0_rule"]]) else "",
                    "d0_close": pd.to_numeric(vals[ci["D0_close"]], errors="coerce") if "D0_close" in ci else np.nan,
                    "buy_score_sheet": pd.to_numeric(vals[ci["观海买点分"]], errors="coerce") if "观海买点分" in ci else np.nan,
                    "h": hz,
                })
                j += 1
            rr = j
            continue
        rr += 1
    return state, out


def build(min_run_date: str) -> pd.DataFrame:
    lut = build_lut(min_run_date)
    best: dict = {}
    for p in _run_files(min_run_date):
        try:
            xls = pd.ExcelFile(p)
        except Exception:
            continue
        for s in xls.sheet_names:
            if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
                _, rows = parse_date_sheet(xls, s)
                for row in rows:
                    row["date"] = s
                    ndays = sum(1 for v in row["h"].values() if pd.notna(v))
                    k = (row["symbol"], s, row["side"])
                    if k not in best or ndays > best[k][0]:
                        best[k] = (ndays, row)

    recs = []
    for (sym, date, side), (nd, row) in best.items():
        h = row["h"]
        fv, fd = np.nan, 0
        for d in sorted(h):
            if pd.notna(h[d]):
                fv, fd = float(h[d]), d
        rtypes = [x.strip() for x in row["rule"].split("|") if x.strip() and "跟踪" not in x]
        rec = lut.get((sym, date, side), {})
        stype = " + ".join(sorted(set(rtypes) | rec.get("types", set())))
        score = row["buy_score_sheet"] if side == "BUY" else rec.get("score", np.nan)
        out = {
            "date": date, "symbol": sym, "side": side, "state": row["state"], "sector": row["sector"],
            "signal_type": stype, "score": score, "d0_close": row["d0_close"],
            "fwd_d1": h.get(1, np.nan), "fwd_d3": h.get(3, np.nan), "fwd_d5": h.get(5, np.nan),
            "fwd_d10": h.get(10, np.nan), "fwd_d14": h.get(14, np.nan), "fwd_last": fv, "days": fd,
        }
        for f in SUBFEATS:
            out[f] = rec.get(f, np.nan)
        recs.append(out)
    df = pd.DataFrame(recs)
    if not df.empty:
        df = df.sort_values(["date", "side", "symbol"]).reset_index(drop=True)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--min-run-date", default="20260520", help="YYYYMMDD; only read history runs on/after this")
    args = ap.parse_args()
    df = build(args.min_run_date)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"WROTE {args.out}  shape={df.shape}")
    if not df.empty:
        print("\nby side:\n" + df["side"].value_counts().to_string())
        print("\nstate x side:\n" + pd.crosstab(df["state"], df["side"]).to_string())
        print(f"\nrows with >=1 forward day: {int(df['fwd_last'].notna().sum())}"
              f" | days median {int(df['days'].median())} max {int(df['days'].max())}")
        print(f"sub-feature coverage (non-null): " +
              ", ".join(f"{f}={int(df[f].notna().sum())}" for f in SUBFEATS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
