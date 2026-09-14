#!/usr/bin/env python3
"""Executable-entry test: the true "buy next-morning open" expectancy.

The close-to-close study prices entry at the D0 signal-day CLOSE (+0.18% D0->D1).
But the daily signal only completes at the D0 close and the scan runs after hours,
so the earliest executable entry is the NEXT session's OPEN (D1 open). This splits
the D0close->D1close move into the overnight gap (which you MISS by entering at the
open) and the D1 intraday move (what you actually capture), then prices realistic
holds from the D1 open.

Needs network (yfinance opens). Reuses scan_stocks.download_daily + prefetch_bars
and horizon_sweep.build_full for the signal list. Close/open, no costs.

    ../../vcp_env/bin/python entry_open.py
    ../../vcp_env/bin/python entry_open.py --min-run-date 20260101
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
import horizon_sweep as hs  # noqa: E402
import scan_stocks as scan  # noqa: E402

FORMAL = "正式买入"


def _stat(x: np.ndarray) -> tuple:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return 0, np.nan, np.nan, np.nan
    sd = x.std(ddof=1) if n > 1 else np.nan
    t = (x.mean() / (sd / np.sqrt(n))) if (n > 1 and sd) else np.nan
    return n, (x > 0).mean(), x.mean(), t


def _row(label, x):
    n, win, mean, t = _stat(x)
    med = np.nanmedian(x) if n else np.nan
    if n == 0:
        return f"  {label:<32}{0:>6}{'—':>9}{'—':>10}{'—':>10}{'—':>8}"
    return f"  {label:<32}{n:>6}{win:>8.1%}{mean:>+10.4f}{med:>+10.4f}{t:>+8.2f}"


def collect(df: pd.DataFrame) -> pd.DataFrame:
    """For each (symbol, D0 date) attach c0, o1, c1, c2, c3 from real OHLC."""
    syms = sorted(df["symbol"].astype(str).str.strip().str.upper().unique().tolist())
    try:
        scan.prefetch_bars(syms, daily_period="1y", h4_period="90d")
    except Exception:
        pass
    ohlc: dict[str, pd.DataFrame] = {}
    for s in syms:
        try:
            d = scan.download_daily(s, period="1y")
        except Exception:
            d = None
        if d is None or d.empty or "Open" not in d.columns:
            continue
        dd = d.copy()
        dd.index = pd.to_datetime(dd.index).date
        ohlc[s] = dd

    recs = []
    for _, r in df.iterrows():
        sym = str(r["symbol"]).strip().upper()
        d = ohlc.get(sym)
        if d is None:
            continue
        d0 = pd.to_datetime(r["date"], errors="coerce")
        if pd.isna(d0):
            continue
        d0 = d0.date()
        idx = list(d.index)
        if d0 not in d.index:
            continue
        p = idx.index(d0)

        def close(k):
            return float(d["Close"].iloc[p + k]) if 0 <= p + k < len(idx) else np.nan

        def openp(k):
            return float(d["Open"].iloc[p + k]) if 0 <= p + k < len(idx) else np.nan

        recs.append({
            "symbol": sym, "date": r["date"], "signal_type": r.get("signal_type", ""),
            "c0": close(0), "o1": openp(1), "c1": close(1), "c2": close(2), "c3": close(3),
        })
    return pd.DataFrame(recs)


def _rr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = b / a - 1
    r[(a == 0) | np.isnan(a) | np.isnan(b)] = np.nan
    return r


def report(px: pd.DataFrame, label: str) -> None:
    c0, o1, c1, c2, c3 = (px[c].to_numpy(float) for c in ["c0", "o1", "c1", "c2", "c3"])
    print(f"\n=== {label} (n={len(px)}) ===")
    print(f"  {'entry -> exit':<32}{'n':>6}{'win':>9}{'mean':>10}{'median':>10}{'t':>8}")
    print(_row("D0 close -> D1 close (optimistic)", _rr(c0, c1)))
    print(_row("  of which: overnight gap D0c->D1o", _rr(c0, o1)))
    print(_row("EXECUTABLE: D1 open -> D1 close", _rr(o1, c1)))
    print(_row("EXECUTABLE: D1 open -> D2 close", _rr(o1, c2)))
    print(_row("EXECUTABLE: D1 open -> D3 close", _rr(o1, c3)))


def walk_forward(px: pd.DataFrame) -> None:
    d = px.copy()
    d["fold"] = pd.to_datetime(d["date"], errors="coerce").dt.to_period("M").astype(str)
    print("\n--- stability: EXECUTABLE D1 open -> D2 close, by month ---")
    wins = []
    for f in sorted(d["fold"].dropna().unique()):
        sub = d[d["fold"] == f]
        r = _rr(sub["o1"].to_numpy(float), sub["c2"].to_numpy(float))
        n, win, mean, _ = _stat(r)
        if n:
            wins.append(win)
            print(f"    {f}: win {win:.1%}  mean {mean:+.4f}  (n={n})")
    if wins:
        a = np.array(wins)
        print(f"    cross-fold: mean win {a.mean():.1%}  folds>50% {int((a > 0.5).sum())}/{len(a)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-run-date", default="20260101")
    args = ap.parse_args()

    sig = hs.build_full(args.min_run_date, 5)
    if sig.empty:
        print("No BUY signals.")
        return 0
    print(f"BUY signals: {len(sig)}   fetching real OHLC opens ...")
    px = collect(sig)
    px = px.dropna(subset=["o1"])
    print(f"signals with a D1 open price: {len(px)}")
    if px.empty:
        print("No D1 opens resolved (dates may be the latest bar). Try --min-run-date earlier.")
        return 0

    report(px, "ALL BUY")
    formal = px[px["signal_type"].map(lambda t: FORMAL in str(t))]
    if len(formal) >= 20:
        report(formal, "正式买入 (formal)")
    walk_forward(px)

    print("\nRead: 'D0 close -> D1 close' is the optimistic backtest entry. It splits into the")
    print("overnight GAP (missed if you enter at the open) + the D1 intraday move (captured).")
    print("If the gap holds most of the edge and D1-open->D2-close is <=0, the executable")
    print("entry has no edge before costs — the near-close-entry idea is moot. No costs/slippage here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
