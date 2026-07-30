#!/usr/bin/env python3
"""Combination-exit test: does pairing D1 with a later day (D2..D5) beat a plain
fixed-horizon hold?

Motivation (user, 2026-07): D1 has the best fixed-horizon win rate but the edge is
marginal and decays by D5. Maybe a *path-dependent* rule that uses the D1 outcome
to decide whether to hold longer does better than any single day.

Reuses horizon_sweep.build_full (per-signal D1..D5 forward returns from the raw
follow-up sheets). Read-only, no network, close-to-close, no costs.

    ../../vcp_env/bin/python horizon_combo.py
    ../../vcp_env/bin/python horizon_combo.py --min-run-date 20260101

Rules tested, for each partner horizon N in {2,3,4,5} (baseline days shown too):
  EXIT_D1        exit at D1 close (baseline)                    -> return d1
  HOLD_DN        hold to DN close (baseline)                    -> return dN
  BESTOF_1N      exit at whichever of D1/DN is higher           -> max(d1,dN)  [LOOKAHEAD ceiling]
  FIRSTGREEN_1N  exit first day in 1..N that is green, else DN  -> tradeable
  CUTLOSER_1N    if d1>0 ride to DN else cut at D1              -> "let winners run" (momentum)
  TAKEPOP_1N     if d1>0 take the pop at D1 else ride to DN     -> "take pop, sit dips" (reversion)
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
import horizon_sweep as hs  # noqa: E402  — reuses build_full()

PREALERT, FORMAL, FIRST = "预警买入", "正式买入", "第一观察点"


def _has(t, tok):
    return tok in str(t or "")


def _series(df: pd.DataFrame, d: int) -> pd.Series:
    return pd.to_numeric(df.get(f"fwd_d{d}", np.nan), errors="coerce")


def _stat(x: np.ndarray) -> tuple:
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return 0, np.nan, np.nan, np.nan
    sd = x.std(ddof=1) if n > 1 else np.nan
    t = (x.mean() / (sd / np.sqrt(n))) if (n > 1 and sd) else np.nan
    return n, (x > 0).mean(), x.mean(), t


def combo_returns(df: pd.DataFrame, N: int) -> dict:
    """Return {rule: np.array of per-signal realized returns} for partner day N."""
    r = {d: _series(df, d).to_numpy() for d in range(1, N + 1)}
    d1, dN = r[1], r[N]
    both = ~np.isnan(d1) & ~np.isnan(dN)

    exit_d1 = np.where(~np.isnan(d1), d1, np.nan)
    hold_dN = np.where(~np.isnan(dN), dN, np.nan)
    bestof = np.where(both, np.maximum(d1, dN), np.nan)
    cutloser = np.where(both, np.where(d1 > 0, dN, d1), np.nan)   # winners ride, losers cut at D1
    takepop = np.where(both, np.where(d1 > 0, d1, dN), np.nan)    # take pop at D1, sit dips to DN

    # first-green: exit on first day 1..N with a positive return; else terminal (last non-nan <=N)
    firstgreen = np.full(len(df), np.nan)
    for i in range(len(df)):
        chosen = np.nan
        terminal = np.nan
        for d in range(1, N + 1):
            v = r[d][i]
            if not np.isnan(v):
                terminal = v
                if v > 0:
                    chosen = v
                    break
        firstgreen[i] = chosen if not np.isnan(chosen) else terminal

    return {
        "EXIT_D1": exit_d1, "HOLD_DN": hold_dN, "BESTOF_1N": bestof,
        "FIRSTGREEN_1N": firstgreen, "CUTLOSER_1N": cutloser, "TAKEPOP_1N": takepop,
    }


def _print_combo(name: str, df: pd.DataFrame, max_h: int) -> dict:
    print(f"\n=== {name} (n={len(df)}) ===")
    best = None
    for N in range(2, max_h + 1):
        rules = combo_returns(df, N)
        print(f"\n  partner DN = D{N}")
        print(f"    {'rule':<16}{'n':>6}{'win':>9}{'mean':>10}{'median':>10}{'t':>8}")
        for rule, arr in rules.items():
            n, win, mean, t = _stat(arr)
            med = np.nanmedian(arr) if n else np.nan
            tag = "  (lookahead)" if rule == "BESTOF_1N" else ""
            print(f"    {rule:<16}{n:>6}{win:>8.1%}{mean:>+10.4f}{med:>+10.4f}{t:>+8.2f}{tag}")
            # track best tradeable (exclude lookahead BESTOF) by mean return
            if rule != "BESTOF_1N" and n >= 20 and not np.isnan(mean):
                if best is None or mean > best["mean"]:
                    best = {"rule": rule, "N": N, "n": n, "win": win, "mean": mean}
    if best:
        print(f"\n  -> best tradeable combo: {best['rule']} with DN=D{best['N']}  "
              f"(win {best['win']:.1%}, mean {best['mean']:+.4f}, n={best['n']})")
    return best or {}


def _conditional(df: pd.DataFrame, max_h: int) -> None:
    """Does the D1 sign predict the DN sign? (momentum vs reversion)"""
    print("\n=== Does D1 predict DN? P(DN>0 | D1 sign), mean DN in each split ===")
    print(f"  {'DN':<5}{'P(DN>0|D1>0)':>16}{'meanDN|D1>0':>14}{'P(DN>0|D1<=0)':>16}{'meanDN|D1<=0':>14}")
    d1 = _series(df, 1).to_numpy()
    for N in range(2, max_h + 1):
        dN = _series(df, N).to_numpy()
        m = ~np.isnan(d1) & ~np.isnan(dN)
        up, dn = m & (d1 > 0), m & (d1 <= 0)
        pu = (dN[up] > 0).mean() if up.sum() else np.nan
        mu = dN[up].mean() if up.sum() else np.nan
        pd_ = (dN[dn] > 0).mean() if dn.sum() else np.nan
        md = dN[dn].mean() if dn.sum() else np.nan
        print(f"  D{N:<4}{pu:>15.1%}{mu:>+14.4f}{pd_:>15.1%}{md:>+14.4f}")


def _walk_forward(df: pd.DataFrame, rule: str, N: int) -> None:
    d = df.copy()
    d["fold"] = pd.to_datetime(d["date"], errors="coerce").dt.to_period("M").astype(str)
    print(f"\n--- stability: {rule} @ DN=D{N} by month ---")
    wins = []
    for f in sorted(d["fold"].dropna().unique()):
        sub = d[d["fold"] == f]
        arr = combo_returns(sub, N)[rule]
        n, win, mean, _ = _stat(arr)
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

    cohorts = {
        "ALL BUY": pd.Series(True, index=df.index),
        "正式买入 (formal)": df["signal_type"].map(lambda t: _has(t, FORMAL)),
        "第一观察点 (1st obs)": df["signal_type"].map(lambda t: _has(t, FIRST)),
    }
    best_all = None
    for name, mask in cohorts.items():
        sub = df[mask].reset_index(drop=True)
        b = _print_combo(name, sub, args.max_h)
        if name == "ALL BUY":
            _conditional(sub, args.max_h)
            best_all = (sub, b)

    if best_all and best_all[1]:
        _walk_forward(best_all[0], best_all[1]["rule"], best_all[1]["N"])

    print("\nCaveats: close-to-close from D0, no costs. BESTOF_1N is lookahead (a ceiling,")
    print("not tradeable). FIRSTGREEN/CUTLOSER/TAKEPOP are end-of-day rules. 'Best by mean'")
    print("ignores payoff shape — read win + mean + t together, and trust cross-fold sign.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
