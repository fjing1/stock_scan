#!/usr/bin/env python3
"""Entry-timing test: enter at D0 (signal close) vs wait and enter at D1/D2 —
and does a "green-D1 confirmation" entry beat entering at the signal close?

Everything so far measures returns from the D0 signal-day close (entry at D0).
This flips it to the entry side. Because the follow-up sheets store cumulative
pct_vs_D0 (c_j = close[j]/close[0]-1), the return of entering at day a and exiting
at day b is simply ret(a,b) = (1+c_b)/(1+c_a) - 1, with c_0 = 0. So D0..D5 is
enough to price any entry/exit pair up to D5.

Reuses horizon_sweep.build_full. Read-only, no network, close-to-close, no costs.

    ../../vcp_env/bin/python entry_timing.py
    ../../vcp_env/bin/python entry_timing.py --min-run-date 20260101

Builds on the finding (horizon_combo.py) that D1 sign predicts continuation:
P(DN>0|D1>0)=73% vs 24% when D1<=0. The tradeable read of that is an ENTRY filter:
wait for the D1 close, enter only if D1 is green (momentum), hold H days.
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

FORMAL, FIRST = "正式买入", "第一观察点"


def _cum(df: pd.DataFrame, d: int) -> np.ndarray:
    if d == 0:
        return np.zeros(len(df))
    return pd.to_numeric(df.get(f"fwd_d{d}", np.nan), errors="coerce").to_numpy()


def ret(df: pd.DataFrame, a: int, b: int) -> np.ndarray:
    """Return of entering at day a, exiting at day b (fractions). NaN if either
    cumulative point is missing."""
    ca, cb = _cum(df, a), _cum(df, b)
    with np.errstate(invalid="ignore"):
        r = (1 + cb) / (1 + ca) - 1
    r[np.isnan(ca) | np.isnan(cb)] = np.nan
    return r


def _stat(x: np.ndarray) -> tuple:
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
        return f"  {label:<34}{0:>6}{'—':>9}{'—':>10}{'—':>10}{'—':>8}"
    return f"  {label:<34}{n:>6}{win:>8.1%}{mean:>+10.4f}{med:>+10.4f}{t:>+8.2f}"


def entry_grid(df: pd.DataFrame, max_h: int) -> None:
    print("\n=== Entry-timing grid: enter at day A, hold H days (all signals) ===")
    print(f"  {'entry@A, hold H (exit A+H)':<34}{'n':>6}{'win':>9}{'mean':>10}{'median':>10}{'t':>8}")
    for a in range(0, 4):
        for h in (1, 2, 3):
            b = a + h
            if b > max_h:
                continue
            print(_row(f"enter D{a}, hold {h}d (-> D{b})", ret(df, a, b)))
        print()


def confirmation_entry(df: pd.DataFrame, max_h: int) -> None:
    d1 = _cum(df, 1)
    green = d1 > 0
    red = (d1 <= 0) & ~np.isnan(d1)
    print("=== Confirmation entry: wait for the D1 close, then enter ===")
    print("  (compare on the SAME exit day so timing, not horizon, is the difference)")
    print(f"  {'rule':<34}{'n':>6}{'win':>9}{'mean':>10}{'median':>10}{'t':>8}")
    for h in (1, 2):
        b = 1 + h
        if b > max_h:
            continue
        print(f"\n  -- exit at D{b} --")
        # baseline: enter at D0, hold to D{b}, all signals
        print(_row(f"enter D0 (all), exit D{b}", ret(df, 0, b)))
        # momentum: enter D1 only if D1 green, hold to D{b}
        rm = ret(df, 1, b).copy(); rm[~green] = np.nan
        print(_row(f"enter D1 if GREEN, exit D{b}", rm))
        # same green names but had you entered D0 instead (does waiting help vs D0 on same names?)
        r0g = ret(df, 0, b).copy(); r0g[~green] = np.nan
        print(_row(f"  (same green names, enter D0)", r0g))
        # reversion: enter D1 only if D1 red (buy the dip)
        rr = ret(df, 1, b).copy(); rr[~red] = np.nan
        print(_row(f"enter D1 if RED (dip), exit D{b}", rr))
    frac_green = np.nanmean(green.astype(float)) if len(green) else np.nan
    print(f"\n  D1-green share of signals: {frac_green:.1%}  "
          f"(the confirmation filter only trades these)")


def walk_forward_conf(df: pd.DataFrame, b: int) -> None:
    d = df.copy()
    d["fold"] = pd.to_datetime(d["date"], errors="coerce").dt.to_period("M").astype(str)
    green = _cum(d, 1) > 0
    print(f"\n--- stability: enter D1-if-green, exit D{b}, by month ---")
    wins = []
    for f in sorted(d["fold"].dropna().unique()):
        mask = (d["fold"] == f).to_numpy()
        r = ret(d, 1, b).copy()
        r[~(green & mask)] = np.nan
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
    ap.add_argument("--max-h", type=int, default=5)
    args = ap.parse_args()

    df = hs.build_full(args.min_run_date, args.max_h)
    if df.empty:
        print("No BUY signals.")
        return 0
    print(f"BUY signals: {len(df)}   {df['date'].min()} .. {df['date'].max()}")

    entry_grid(df, args.max_h)
    confirmation_entry(df, args.max_h)
    walk_forward_conf(df, 3)

    print("\nCaveats: close-to-close, no costs. Entering at D1 forgoes the D0->D1 move and")
    print("trades ~1 day later on fewer names (only D1-green). 'enter D1 if green' is the")
    print("tradeable read of the momentum-persistence find; judge it on cross-fold sign, and")
    print("remember waiting a day adds one more close of slippage/gap risk not modeled here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
