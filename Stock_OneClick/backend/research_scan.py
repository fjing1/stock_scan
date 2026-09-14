#!/usr/bin/env python3
"""Isolated, READ-ONLY research scan over the WIDE universe (enable ignored).

Runs the live signal engine (scan_stocks.scan_one_symbol) across a shard of the
full Sheet2 universe and writes ONLY to reports/ — it never touches the live
workbook, scan_result_latest.xlsx, history/, or the lifecycle tracking. This is
how we get a bias-reduced cross-sectional signal read on the expanded 1,067-name
set without contaminating the curated live book.

Throttle-safety: yfinance failures are logged per-symbol (a failed fetch is NOT
"no signal" — conflating them would bias breadth). Use modest download workers.

    ../../vcp_env/bin/python research_scan.py --shard 1/3 --out ../reports/research_scan_1.csv
    ../../vcp_env/bin/python research_scan.py --limit 5           # smoke test
    STOCK_ONECLICK_DOWNLOAD_WORKERS=8 ../../vcp_env/bin/python research_scan.py --shard 2/3 ...
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
import scan_stocks as scan  # noqa: E402


def _universe() -> pd.DataFrame:
    """All Sheet2 rows (enable ignored), passed through the same scannable filter
    the live run uses (drops market-context / index groups), so we score equities."""
    df_in, df_meta = scan.load_input_and_meta(scan.INPUT_FILE)
    if df_meta is None or df_meta.empty:
        df_meta = pd.DataFrame({"symbol": df_in["symbol"], "name": "", "group": ""})
    df_meta = df_meta.copy()
    if "enable" in df_meta.columns:
        df_meta["enable"] = 1  # treat everything as active for research
    df_run, _dropped = scan.filter_scannable_universe(df_meta)
    return df_run


def _shard(df: pd.DataFrame, spec: str) -> pd.DataFrame:
    if not spec:
        return df
    k, n = (int(x) for x in spec.split("/"))
    syms = df.reset_index(drop=True)
    return syms.iloc[[i for i in range(len(syms)) if i % n == (k - 1)]].reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", default="", help="k/N, e.g. 1/3 (every Nth symbol)")
    ap.add_argument("--out", default="../reports/research_scan.csv")
    ap.add_argument("--limit", type=int, default=0, help="scan only first N of the shard (smoke test)")
    args = ap.parse_args()

    df_run = _shard(_universe(), args.shard)
    if args.limit:
        df_run = df_run.head(args.limit)
    if df_run.empty:
        print("empty shard"); return 0

    sector_map = (df_run.drop_duplicates("symbol")
                  .assign(板块=lambda x: x["group"].apply(scan._normalize_sector_with_code))
                  .set_index("symbol")["板块"].to_dict())

    total = len(df_run)
    print(f"research scan: {total} symbols  shard={args.shard or 'all'}  -> {args.out}", flush=True)
    try:
        scan.prefetch_bars(df_run["symbol"].tolist(), daily_period="1y", h4_period="90d")
    except Exception as e:
        print(f"prefetch warning: {e}", flush=True)

    xl = scan.XunLongIndicator()
    rows, failures = [], []
    t0 = time.time()
    for i, (_, r) in enumerate(df_run.iterrows(), start=1):
        sym, name = r["symbol"], r.get("name", "")
        try:
            df_sig = scan.scan_one_symbol(sym, name, xl)
        except Exception as e:
            failures.append((sym, f"{type(e).__name__}: {str(e)[:80]}"))
            continue
        if df_sig is not None and not df_sig.empty:
            df_sig = df_sig.copy()
            df_sig["板块"] = sector_map.get(sym, "99 未分组")
            rows.append(df_sig)
        if i % 50 == 0 or i == total:
            print(f"  [{i}/{total}] {time.time()-t0:.0f}s  signals so far={sum(len(x) for x in rows)}"
                  f"  fails={len(failures)}", flush=True)

    out = Path(args.out)
    if not out.is_absolute():
        out = (BACKEND_DIR / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        df_all = pd.concat(rows, ignore_index=True)
        df_all["buy_score"] = df_all.apply(scan.score_buy_signal_row, axis=1)
        df_all["buy_score_raw"] = df_all.apply(scan.score_buy_signal_row_raw, axis=1)
        df_all["sell_score"] = df_all.apply(scan.score_sell_signal_row, axis=1)
        df_all.to_csv(out, index=False)
    else:
        df_all = pd.DataFrame()
        out.write_text("")  # empty marker

    fail_path = out.with_name(out.stem + "_fail.txt")
    fail_path.write_text("\n".join(f"{s}\t{m}" for s, m in failures) + ("\n" if failures else ""))

    scanned_ok = total - len(failures)
    print(f"\n✅ shard done: scanned_ok={scanned_ok}/{total}  fails={len(failures)}  "
          f"signal_rows={len(df_all)}  {time.time()-t0:.0f}s")
    print(f"   wrote {out.name}  +  {fail_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
