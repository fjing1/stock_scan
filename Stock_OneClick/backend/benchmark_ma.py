#!/usr/bin/env python3
"""Benchmark: the 'dumb' moving-average de-risk rule the regime gate must beat.

Per docs/STRATEGY_PROPOSAL.md §8, before the scanner's regime gate earns capital
it must beat a transparent price rule: hold the index when close >= its N-day MA,
go to cash otherwise (acting the NEXT day — no look-ahead). This tool implements
that rule and compares it to buy-and-hold (total return, max drawdown, time in
market, switches).

Price source (in priority order):
  1. --prices CSV with columns date,close  (use a real multi-month SPX/SPY series)
  2. fallback: reconstruct a short SPX index from the '指数快照' daily % printed in
     the scan_result date sheets (only ~2-3 weeks here -> NOT enough for a 20-day
     MA; use --window small, and treat the output as a demo, not a verdict).

No scipy, no network. Run:
    ../../vcp_env/bin/python benchmark_ma.py --window 5
    ../../vcp_env/bin/python benchmark_ma.py --prices spx.csv --window 20
"""
from __future__ import annotations

import argparse
import glob
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
LATEST = BASE_DIR / "scan_result_latest.xlsx"
HIST = BASE_DIR / "history"


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def ma_derisk(closes: pd.Series, window: int) -> dict:
    """Hold the index when close >= N-day MA, else cash; position applied NEXT day."""
    closes = closes.astype(float).sort_index()
    ma = closes.rolling(window).mean()
    in_mkt = (closes >= ma)
    position = in_mkt.shift(1).fillna(False).astype(float)   # act next day -> no look-ahead
    ret = closes.pct_change().fillna(0.0)
    strat_ret = position * ret
    bench_eq = (1 + ret).cumprod()
    strat_eq = (1 + strat_ret).cumprod()
    switches = int((position.diff().abs() > 0).sum())
    return {
        "bench_total": float(bench_eq.iloc[-1] - 1) if len(bench_eq) else np.nan,
        "strat_total": float(strat_eq.iloc[-1] - 1) if len(strat_eq) else np.nan,
        "bench_maxdd": max_drawdown(bench_eq),
        "strat_maxdd": max_drawdown(strat_eq),
        "pct_in_market": float(position.mean()),
        "switches": switches,
        "n": int(len(closes)),
        "bench_eq": bench_eq, "strat_eq": strat_eq, "position": position,
    }


def extract_index_from_sheets(symbol: str = "SPX") -> pd.Series:
    """Reconstruct a daily index level from the '指数快照' daily % across date sheets."""
    rx = re.compile(rf"{re.escape(symbol)}\s*([+-]?[0-9]+(?:\.[0-9]+)?)%")
    rets: dict[str, float] = {}
    paths = [str(LATEST)] if LATEST.exists() else []
    paths += sorted(glob.glob(str(HIST / "scan_result_*.xlsx")))
    for p in paths:
        try:
            xls = pd.ExcelFile(p)
        except Exception:
            continue
        for s in xls.sheet_names:
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", s):
                continue
            try:
                raw = pd.read_excel(xls, s, header=None)
            except Exception:
                continue
            for _, r in raw.iterrows():
                if str(r.iloc[0]) == "指数快照" and len(r) > 1 and pd.notna(r.iloc[1]):
                    m = rx.search(str(r.iloc[1]))
                    if m:
                        rets.setdefault(s, float(m.group(1)) / 100.0)
                    break
    if not rets:
        return pd.Series(dtype=float)
    sr = pd.Series(rets).sort_index()
    level = 100.0 * (1.0 + sr).cumprod()
    return level


def _report(stats: dict, label: str, window: int):
    print(f"\n=== {label} (MA window={window}, n={stats['n']}) ===")
    print(f"  buy & hold:   total {stats['bench_total']:+.2%}   maxDD {stats['bench_maxdd']:+.2%}")
    print(f"  MA de-risk:   total {stats['strat_total']:+.2%}   maxDD {stats['strat_maxdd']:+.2%}")
    print(f"  time in market: {stats['pct_in_market']:.0%}   switches: {stats['switches']}")
    better_dd = stats["strat_maxdd"] >= stats["bench_maxdd"]   # less negative = better
    print(f"  -> MA rule {'REDUCED' if better_dd else 'did NOT reduce'} max drawdown vs hold "
          f"({stats['strat_maxdd']:+.2%} vs {stats['bench_maxdd']:+.2%}).")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prices", help="CSV with columns date,close (real SPX/SPY series)")
    ap.add_argument("--symbol", default="SPX", help="index ticker to extract from sheets (fallback)")
    ap.add_argument("--window", type=int, default=20, help="MA window in trading days")
    args = ap.parse_args()

    if args.prices:
        df = pd.read_csv(args.prices)
        df.columns = [c.lower() for c in df.columns]
        closes = pd.Series(df["close"].values, index=df["date"].values).astype(float)
        src = f"prices CSV {args.prices}"
    else:
        closes = extract_index_from_sheets(args.symbol)
        src = f"reconstructed {args.symbol} from scan date sheets"
        if closes.empty:
            print("No price data: pass --prices a CSV (date,close), or ensure date sheets carry 指数快照.")
            return 1

    win = args.window
    reconstructed = not args.prices
    print(f"Source: {src}  ({len(closes)} points)")
    if reconstructed:
        print("⚠️  DEMO ONLY — this index is reconstructed from a handful of scan snapshots, NOT a "
              "real price history. A meaningful N-day MA needs >=60 daily closes; pass --prices a real "
              "SPX/SPY CSV (date,close) for an actual benchmark. Do NOT read the numbers below as a verdict.")
    if len(closes) < win + 2:
        eff = max(3, len(closes) // 3)
        print(f"   (only {len(closes)} points; reducing MA window {win} -> {eff} so it computes at all)")
        win = eff

    stats = ma_derisk(closes, win)
    _report(stats, f"{args.symbol} MA de-risk vs buy-and-hold", win)
    print("\nGate-vs-benchmark check (for the forward test): compare reports/gate_log.csv "
          "target_gross against this MA position over the SAME dates — the gate must avoid "
          "drawdowns at least as well, net of costs, to justify using it over this dumb rule.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
