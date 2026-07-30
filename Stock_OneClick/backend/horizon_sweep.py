#!/usr/bin/env python3
"""Horizon win-rate sweep: for each BUY signal type, which forward holding period
(D1..D5) has the highest win rate / best mean return?

Motivation (user, 2026-07): the signals look like a short "cycle" edge (~1-5 days)
that fades by D10 (escalation_backtest.py shows every cohort negative at D10). This
sweeps D1..D5 and picks the best horizon per signal type.

The flat dataset (build_dataset.py) only kept D1/D3/D5, so this reads the raw
per-date follow-up sheets directly (which store D1..D14 pct_vs_D0) and keeps every
day 1..5. Read-only, no network. Reuses build_dataset's sheet parser + type LUT.

    ../../vcp_env/bin/python horizon_sweep.py
    ../../vcp_env/bin/python horizon_sweep.py --min-run-date 20260101 --max-h 5

Returns are close-to-close fractions from the D0 anchor, no costs. Win rate = share
of signals with fwd return > 0 at that horizon.
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
import build_dataset as bd  # noqa: E402

PREALERT, FORMAL = "预警买入", "正式买入"
SECOND, FIRST = "二进宫买入点", "第一观察点"


def build_full(min_run_date: str, max_h: int) -> pd.DataFrame:
    """One row per (symbol, date) BUY signal with signal_type + fwd_d1..fwd_d{max_h}.
    Dedups re-emitted runs keeping the row with the most forward days filled."""
    lut = bd.build_lut(min_run_date)
    best: dict = {}
    for p in bd._run_files(min_run_date):
        try:
            xls = pd.ExcelFile(p)
        except Exception:
            continue
        for s in xls.sheet_names:
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", s):
                continue
            _, rows = bd.parse_date_sheet(xls, s)
            for row in rows:
                if row["side"] != "BUY":
                    continue
                row["date"] = s
                ndays = sum(1 for v in row["h"].values() if pd.notna(v))
                k = (row["symbol"], s)
                if k not in best or ndays > best[k][0]:
                    best[k] = (ndays, row)

    recs = []
    for (sym, date), (nd, row) in best.items():
        h = row["h"]
        rtypes = [x.strip() for x in row["rule"].split("|") if x.strip() and "跟踪" not in x]
        rec = lut.get((sym, date, "BUY"), {})
        stype = " + ".join(sorted(set(rtypes) | rec.get("types", set())))
        out = {
            "date": date, "symbol": sym, "signal_type": stype,
            "state": row["state"], "score": row["buy_score_sheet"],
        }
        for d in range(1, max_h + 1):
            out[f"fwd_d{d}"] = h.get(d, np.nan)
        recs.append(out)
    df = pd.DataFrame(recs)
    if not df.empty:
        df["date_d"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _has(t, tok):
    return tok in str(t or "")


def sweep(df: pd.DataFrame, cohort_mask: pd.Series, max_h: int) -> pd.DataFrame:
    sub = df[cohort_mask]
    rows = []
    for d in range(1, max_h + 1):
        x = pd.to_numeric(sub[f"fwd_d{d}"], errors="coerce").dropna()
        n = len(x)
        if n == 0:
            rows.append({"H": f"D{d}", "n": 0, "win_rate": np.nan, "mean": np.nan,
                         "median": np.nan, "t": np.nan})
            continue
        sd = x.std(ddof=1) if n > 1 else np.nan
        t = (x.mean() / (sd / np.sqrt(n))) if (n > 1 and sd) else np.nan
        rows.append({"H": f"D{d}", "n": n, "win_rate": (x > 0).mean(),
                     "mean": x.mean(), "median": x.median(), "t": t})
    return pd.DataFrame(rows)


def _print_sweep(name: str, tbl: pd.DataFrame) -> None:
    print(f"\n=== {name} ===")
    print(f"  {'H':<4}{'n':>6}{'win_rate':>11}{'mean':>10}{'median':>10}{'t-stat':>9}")
    best_h, best_wr = None, -1
    for _, r in tbl.iterrows():
        if r["n"] == 0:
            print(f"  {r['H']:<4}{0:>6}{'—':>11}{'—':>10}{'—':>10}{'—':>9}")
            continue
        star = ""
        if r["win_rate"] > best_wr:
            best_wr, best_h = r["win_rate"], r["H"]
        print(f"  {r['H']:<4}{int(r['n']):>6}{r['win_rate']:>10.1%}"
              f"{r['mean']:>+10.4f}{r['median']:>+10.4f}{r['t']:>+9.2f}")
    if best_h is not None:
        br = tbl[tbl["H"] == best_h].iloc[0]
        print(f"  -> best win-rate horizon: {best_h}  ({best_wr:.1%}, mean {br['mean']:+.4f}, n={int(br['n'])})")


def _walk_forward_h(df: pd.DataFrame, cohort_mask: pd.Series, horizon_col: str) -> None:
    sub = df[cohort_mask].copy()
    sub["fold"] = sub["date_d"].dt.to_period("M").astype(str)
    print(f"\n  walk-forward win-rate at {horizon_col} by month:")
    wrs = []
    for f in sorted(sub["fold"].dropna().unique()):
        x = pd.to_numeric(sub.loc[sub["fold"] == f, horizon_col], errors="coerce").dropna()
        if len(x):
            wr = (x > 0).mean()
            wrs.append(wr)
            print(f"    {f}: {wr:.1%}  (n={len(x)}, mean {x.mean():+.4f})")
    if wrs:
        a = np.array(wrs)
        print(f"    cross-fold: mean {a.mean():.1%}  folds>50% {int((a > 0.5).sum())}/{len(a)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-run-date", default="20260101")
    ap.add_argument("--max-h", type=int, default=5, help="max forward horizon in tracked days")
    args = ap.parse_args()

    df = build_full(args.min_run_date, args.max_h)
    if df.empty:
        print("No BUY signals in dataset.")
        return 0
    print(f"BUY signals: {len(df)}   date range {df['date'].min()} .. {df['date'].max()}")
    cov = ", ".join(f"D{d}={int(df[f'fwd_d{d}'].notna().sum())}" for d in range(1, args.max_h + 1))
    print(f"forward-day coverage: {cov}")

    cohorts = {
        "ALL BUY": pd.Series(True, index=df.index),
        "预警买入 (pre-alert)": df["signal_type"].map(lambda t: _has(t, PREALERT)),
        "正式买入 (formal)": df["signal_type"].map(lambda t: _has(t, FORMAL)),
        "二进宫买入点 (2nd entry)": df["signal_type"].map(lambda t: _has(t, SECOND)),
        "第一观察点 (1st obs)": df["signal_type"].map(lambda t: _has(t, FIRST)),
        "score>=90": pd.to_numeric(df["score"], errors="coerce") >= 90,
    }

    best_overall = {}
    for name, mask in cohorts.items():
        tbl = sweep(df, mask, args.max_h)
        _print_sweep(name, tbl)
        valid = tbl.dropna(subset=["win_rate"])
        if not valid.empty:
            best_overall[name] = valid.loc[valid["win_rate"].idxmax()]

    print("\n=== Best win-rate horizon per cohort (highest win rate) ===")
    print(f"  {'cohort':<26}{'best H':>8}{'win_rate':>11}{'mean':>10}{'n':>7}")
    for name, r in best_overall.items():
        print(f"  {name:<26}{r['H']:>8}{r['win_rate']:>10.1%}{r['mean']:>+10.4f}{int(r['n']):>7}")

    # walk-forward stability for the two most-used tradeable cohorts at their best H
    for name in ["ALL BUY", "正式买入 (formal)"]:
        if name in best_overall:
            hcol = f"fwd_{best_overall[name]['H'].lower()}"
            print(f"\n--- stability: {name} at {best_overall[name]['H']} ---")
            _walk_forward_h(df, cohorts[name], hcol)

    print("\nCaveats: close-to-close from D0, no costs/slippage. Win rate ignores payoff")
    print("asymmetry — check mean too (a 55% win rate with tiny wins/big losses still loses).")
    print("Recent epoch: prefer horizons whose win rate holds >50% across folds, not one window.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
