#!/usr/bin/env python3
"""Backtest / calibrate the 观海买点分 (buy_score) against realized forward returns.

The nightly scanner records D0..D14 forward closes for every buy signal (the
per-date sheets in ``scan_result_latest.xlsx`` / ``history/scan_result_*.xlsx``,
and the graduated ``history/completed_14d/`` archives). That is a ready-made
labeled dataset: (buy_score, forward_return). This tool reads it and reports
whether the score is actually predictive — the feedback loop the project was
collecting data for but never closed.

Read-only. No network. Run with the project venv:

    ../../vcp_env/bin/python backtest_score.py                 # latest workbook
    ../../vcp_env/bin/python backtest_score.py --source both   # + history + completed
    ../../vcp_env/bin/python backtest_score.py --horizon 5

Notes:
  * 观海买点分 was added in the 2026-06 rewrite, so pre-June archives (and any
    completed_14d batch from before then) have no score and are skipped.
  * Because the lifecycle epoch is recent, scored signals may have only a few
    forward days filled — the tool reports the days-tracked distribution and
    uses the last available return within the requested horizon.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
BASE_DIR = BACKEND_DIR.parent
HISTORY_DIR = BASE_DIR / "history"
COMPLETED_DIR = HISTORY_DIR / "completed_14d"
LATEST_FILE = BASE_DIR / "scan_result_latest.xlsx"

SCORE_HEADER = "观海买点分"
STOP_MARKERS = ("Top5统计", "排名", "触发样本", "市场环境", "No signals", "信号快照")


def _iter_workbooks(source: str):
    if source in ("latest", "both") and LATEST_FILE.exists():
        yield LATEST_FILE
    if source in ("completed", "both"):
        yield from sorted(COMPLETED_DIR.glob("*.xlsx"))
    if source in ("results", "both"):
        yield from sorted(HISTORY_DIR.glob("scan_result_*.xlsx"))


def _dnum(header_cell: str) -> int:
    try:
        return int(str(header_cell)[1:].split("_", 1)[0])
    except Exception:
        return 0


def _is_stop(cell) -> bool:
    s = str(cell or "").strip()
    if not s:
        return True
    return any(m in s for m in STOP_MARKERS)


def _extract(path: Path, horizon: int) -> list[dict]:
    out: list[dict] = []
    try:
        xls = pd.ExcelFile(path)
    except Exception:
        return out
    for sheet in xls.sheet_names:
        try:
            raw = pd.read_excel(xls, sheet_name=sheet, header=None)
        except Exception:
            continue
        if raw.empty:
            continue
        ncol = raw.shape[1]
        for r in range(len(raw)):
            header = [str(x) if pd.notna(x) else "" for x in raw.iloc[r].tolist()]
            if "symbol" not in header or SCORE_HEADER not in header:
                continue  # not a scored follow-up header row
            sym_col = header.index("symbol")
            score_col = header.index(SCORE_HEADER)
            d0_col = header.index("D0_date") if "D0_date" in header else None
            pct_cols = [(i, _dnum(header[i])) for i in range(len(header))
                        if str(header[i]).endswith("_pct_vs_D0")]
            pct_cols = [(i, d) for i, d in pct_cols if 0 < d <= horizon]
            pct_cols.sort(key=lambda t: t[1])
            for rr in range(r + 1, len(raw)):
                first = raw.iat[rr, sym_col] if sym_col < ncol else None
                if _is_stop(first):
                    break
                score = pd.to_numeric(raw.iat[rr, score_col], errors="coerce") if score_col < ncol else np.nan
                if pd.isna(score):
                    continue  # sell-side rows / unscored -> skip
                fwd, days = np.nan, 0
                for i, d in pct_cols:
                    v = pd.to_numeric(raw.iat[rr, i], errors="coerce") if i < ncol else np.nan
                    if pd.notna(v):
                        fwd, days = float(v), d
                d0 = raw.iat[rr, d0_col] if (d0_col is not None and d0_col < ncol) else sheet
                out.append({
                    "symbol": str(raw.iat[rr, sym_col]).strip().upper(),
                    "d0_date": str(d0),
                    "score": float(score),
                    "fwd_return": fwd,
                    "days": days,
                })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["latest", "completed", "results", "both"], default="latest")
    ap.add_argument("--horizon", type=int, default=14, help="forward horizon in tracked days (<=14)")
    args = ap.parse_args()

    rows, files = [], 0
    for path in _iter_workbooks(args.source):
        got = _extract(path, args.horizon)
        if got:
            files += 1
            rows.extend(got)

    if not rows:
        print(f"No scored follow-up rows found in source '{args.source}'.")
        print("观海买点分 only exists in 2026-06+ workbooks; try --source both.")
        return 0

    df = pd.DataFrame(rows)
    # Dedup the same (symbol, signal date) across re-emitted runs, keeping the
    # observation with the most forward days filled.
    df = (df.sort_values("days")
            .drop_duplicates(subset=["symbol", "d0_date"], keep="last")
            .reset_index(drop=True))

    scored = len(df)
    have_ret = df[df["fwd_return"].notna()].copy()
    print(f"Source: {args.source}   workbooks with data: {files}")
    print(f"Unique scored signals: {scored}   with >=1 forward day: {len(have_ret)}")
    if scored:
        dd = df["days"]
        print(f"Forward days tracked: min {int(dd.min())}, median {int(dd.median())}, max {int(dd.max())}")

    if len(have_ret) < 10:
        print("\nNot enough signals with realized returns yet (need >=10).")
        print("This is expected so soon after the 2026-05-22 lifecycle epoch — the")
        print("tool will produce a real IC once more batches accrue forward days.")
        if scored:
            print("\nScore distribution so far:")
            print(df["score"].describe().to_string())
        return 0

    ret = have_ret["fwd_return"]
    print(f"\n=== Forward return (fraction; 0.05 = +5%), last day <= D{args.horizon} ===")
    print(f"  mean {ret.mean():+.4f}   median {ret.median():+.4f}   "
          f"stdev {ret.std():.4f}   hit-rate(>0) {(ret > 0).mean():.1%}")

    print("\n=== Information Coefficient: does score predict return? ===")
    pear = have_ret["score"].corr(ret, method="pearson")
    spear = have_ret["score"].rank().corr(ret.rank())  # Spearman == Pearson on ranks (no scipy dep)
    print(f"  Pearson  = {pear:+.3f}     Spearman = {spear:+.3f}")
    verdict = ("positive — higher score → higher return (as intended)" if spear > 0.05
               else "INVERTED — higher score → lower return" if spear < -0.05
               else "~zero — score shows little edge at this horizon")
    print(f"  -> {verdict}")

    print("\n=== By score bucket ===")
    bins = [0, 60, 70, 80, 90, 100.0001]
    labels = ["<60", "60-70", "70-80", "80-90", "90-100"]
    have_ret["bucket"] = pd.cut(have_ret["score"], bins=bins, labels=labels, right=False)
    grp = have_ret.groupby("bucket", observed=True)["fwd_return"]
    print(f"  {'bucket':<8}{'n':>5}{'mean ret':>11}{'hit-rate':>11}")
    for b in labels:
        if b in grp.groups:
            s = grp.get_group(b)
            print(f"  {b:<8}{len(s):>5}{s.mean():>+11.4f}{(s > 0).mean():>10.1%}")

    print("\nNote: close-to-close from the D0 anchor, no transaction costs. "
          "A relative ranking check, not a P&L statement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
