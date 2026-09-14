#!/usr/bin/env python3
"""Backtest the "预警买入 → 正式买入" escalation edge before wiring it live.

The daily scanner emits two independent BUY signals: a 4H pre-alert (预警买入,
4H 0出) and a daily formal buy (正式买入, 日线 0出). They are NOT laddered today.
The open question (user asked 2026-07): if we PROMOTE a pre-alert to a buy when
it later escalates to a formal buy, is there a realized forward-return edge — or
is this just another multi-timeframe-alignment rule with no edge (see
memory: mtf-signal-alignment-no-edge)?

This tool answers that from the labeled dataset build_dataset.py already emits
(one row per symbol/date/side with signal_type + fwd returns). It is descriptive
+ walk-forward, read-only, no network. It does NOT change any score or signal —
per the user's "backtest first, then decide" choice.

    ../../vcp_env/bin/python escalation_backtest.py
    ../../vcp_env/bin/python escalation_backtest.py --horizon 5 --escal-window 5
    ../../vcp_env/bin/python escalation_backtest.py --min-run-date 20260101

Cohorts (BUY side only):
  PA_ONLY       pre-alert fired, no formal buy that same day (entry at pre-alert)
  PA_ESCALATED  PA_ONLY that DID get a formal buy within --escal-window trading
                days (entry at the pre-alert). NOTE: uses future info to label —
                descriptive signal-quality check, not a tradeable rule.
  PA_FIZZLED    PA_ONLY that did NOT escalate within the window (entry at pre-alert)
  FORMAL_ALL    any day with a formal buy (entry at the formal buy)
  FORMAL_WARM   formal buy preceded by a pre-alert within the window (the tradeable
                "wait for confirmation after the pre-alert" entry)
  FORMAL_COLD   formal buy with no preceding pre-alert in the window

Key questions:
  Q1  Does eventual escalation mark better pre-alerts?  PA_ESCALATED vs PA_FIZZLED
  Q2  Enter early vs wait?  PA_ESCALATED (from pre-alert) vs FORMAL_WARM (from formal)
  Q3  Does a pre-alert warm-up improve a formal buy?  FORMAL_WARM vs FORMAL_COLD

Returns are close-to-close fractions from the D0 anchor, no costs — a relative
ranking check, not a P&L statement.
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
import build_dataset as bd  # noqa: E402  — reuses the canonical flat dataset

PREALERT = "预警买入"
FORMAL = "正式买入"
HORIZON_COLS = {1: "fwd_d1", 3: "fwd_d3", 5: "fwd_d5", 10: "fwd_d10", 14: "fwd_d14"}


def _has(types_str: str, token: str) -> bool:
    return token in str(types_str or "")


def _fwd(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Forward return at the requested horizon, falling back to the last filled
    horizon column <= horizon, then to fwd_last."""
    cols = [c for h, c in sorted(HORIZON_COLS.items()) if h <= horizon and c in df.columns]
    out = pd.Series(np.nan, index=df.index)
    for c in cols:  # later (larger-horizon) non-null overwrites earlier
        v = pd.to_numeric(df[c], errors="coerce")
        out = v.where(v.notna(), out)
    if "fwd_last" in df.columns:
        fl = pd.to_numeric(df["fwd_last"], errors="coerce")
        out = out.where(out.notna(), fl)
    return out


def label_escalation(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Add is_pa_only / has_formal / escalated / formal_warm flags per row using a
    trading-day (business-day proxy) window between a pre-alert and a later formal
    buy on the same symbol."""
    df = df[df["side"] == "BUY"].copy()
    df["date_d"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date_d"])
    df["has_pa"] = df["signal_type"].map(lambda t: _has(t, PREALERT))
    df["has_formal"] = df["signal_type"].map(lambda t: _has(t, FORMAL))
    df["is_pa_only"] = df["has_pa"] & ~df["has_formal"]

    # per-symbol arrays of formal-buy dates and pre-alert dates
    formal_dates: dict[str, list] = {}
    prealert_dates: dict[str, list] = {}
    for sym, g in df.groupby("symbol"):
        formal_dates[sym] = sorted(g.loc[g["has_formal"], "date_d"].dt.date.tolist())
        prealert_dates[sym] = sorted(g.loc[g["has_pa"], "date_d"].dt.date.tolist())

    def busgap(a, b) -> int:
        return int(np.busday_count(a, b))

    escalated, formal_warm = [], []
    for _, r in df.iterrows():
        d = r["date_d"].date()
        sym = r["symbol"]
        # escalation: a formal buy strictly AFTER this pre-alert, within `window` trading days
        esc = False
        if r["is_pa_only"]:
            for fd in formal_dates.get(sym, []):
                g = busgap(d, fd)
                if 0 < g <= window:
                    esc = True
                    break
        escalated.append(esc)
        # warm formal: a pre-alert strictly BEFORE this formal buy, within `window` trading days
        warm = False
        if r["has_formal"]:
            for pd_ in prealert_dates.get(sym, []):
                g = busgap(pd_, d)
                if 0 < g <= window:
                    warm = True
                    break
        formal_warm.append(warm)
    df["escalated"] = escalated
    df["formal_warm"] = formal_warm
    return df


def _stats(fwd: pd.Series) -> dict:
    x = pd.to_numeric(fwd, errors="coerce").dropna()
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "hit": np.nan, "t": np.nan}
    sd = x.std(ddof=1) if n > 1 else np.nan
    t = (x.mean() / (sd / np.sqrt(n))) if (n > 1 and sd and not np.isnan(sd)) else np.nan
    return {"n": n, "mean": x.mean(), "median": x.median(), "hit": (x > 0).mean(), "t": t}


def _print_cohorts(cohorts: dict[str, pd.Series], title: str) -> None:
    print(f"\n=== {title} ===")
    print(f"  {'cohort':<14}{'n':>5}{'mean':>10}{'median':>10}{'hit>0':>9}{'t-stat':>9}")
    for name, fwd in cohorts.items():
        s = _stats(fwd)
        if s["n"] == 0:
            print(f"  {name:<14}{0:>5}{'—':>10}{'—':>10}{'—':>9}{'—':>9}")
            continue
        print(f"  {name:<14}{s['n']:>5}{s['mean']:>+10.4f}{s['median']:>+10.4f}"
              f"{s['hit']:>8.1%}{s['t']:>+9.2f}")


def _bootstrap_diff(a: pd.Series, b: pd.Series, iters: int = 5000, seed: int = 7) -> tuple:
    """Bootstrap 95% CI for mean(a) - mean(b). Returns (diff, lo, hi, p_sign)."""
    a = pd.to_numeric(a, errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(b, errors="coerce").dropna().to_numpy()
    if len(a) < 5 or len(b) < 5:
        return (np.nan, np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    diffs = np.empty(iters)
    for i in range(iters):
        da = rng.choice(a, size=len(a), replace=True).mean()
        db = rng.choice(b, size=len(b), replace=True).mean()
        diffs[i] = da - db
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    obs = a.mean() - b.mean()
    # fraction of bootstrap diffs with the same sign as observed -> confidence the sign is real
    p_sign = (diffs > 0).mean() if obs >= 0 else (diffs < 0).mean()
    return (obs, lo, hi, p_sign)


def _walk_forward(df: pd.DataFrame, horizon: int, cohort_masks: dict) -> None:
    """Per calendar-month fold: cohort mean forward return + cross-fold stability."""
    df = df.copy()
    df["fold"] = df["date_d"].dt.to_period("M").astype(str)
    df["_fwd"] = _fwd(df, horizon)
    folds = sorted(df["fold"].unique())
    print(f"\n=== Walk-forward by month (mean fwd_d{horizon}; n in parens) ===")
    names = list(cohort_masks.keys())
    print("  " + "fold".ljust(9) + "".join(f"{nm:>18}" for nm in names))
    per_fold_means = {nm: [] for nm in names}
    for f in folds:
        sub = df[df["fold"] == f]
        cells = []
        for nm, fn in cohort_masks.items():
            m = fn(sub)
            x = sub.loc[m, "_fwd"].dropna()
            if len(x):
                per_fold_means[nm].append(x.mean())
                cells.append(f"{x.mean():+.3f}({len(x)})".rjust(18))
            else:
                cells.append("—".rjust(18))
        print("  " + f.ljust(9) + "".join(cells))
    print("\n  cross-fold mean-of-means (stability; want consistent sign):")
    for nm in names:
        vals = per_fold_means[nm]
        if vals:
            arr = np.array(vals)
            print(f"    {nm:<14} mean {arr.mean():+.4f}  sd {arr.std(ddof=1) if len(arr) > 1 else float('nan'):.4f}"
                  f"  folds+ {int((arr > 0).sum())}/{len(arr)}")
        else:
            print(f"    {nm:<14} (no data)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-run-date", default="20260101", help="YYYYMMDD; only read history runs on/after this")
    ap.add_argument("--horizon", type=int, default=10, help="forward horizon in tracked days (<=14)")
    ap.add_argument("--escal-window", type=int, default=5, help="trading-day window for pre-alert->formal escalation")
    args = ap.parse_args()

    raw = bd.build(args.min_run_date)
    if raw.empty:
        print("No dataset rows. Run build_dataset.py first / check --min-run-date.")
        return 0
    df = label_escalation(raw, args.escal_window)
    df["_fwd"] = _fwd(df, args.horizon)

    have = int(df["_fwd"].notna().sum())
    print(f"BUY rows: {len(df)}   with realized fwd return: {have}   "
          f"escal-window: {args.escal_window} trading days   horizon: D{args.horizon}")
    n_pa = int(df["is_pa_only"].sum())
    n_esc = int((df["is_pa_only"] & df["escalated"]).sum())
    n_formal = int(df["has_formal"].sum())
    n_warm = int((df["has_formal"] & df["formal_warm"]).sum())
    print(f"pre-alert-only days: {n_pa} (escalated within window: {n_esc}, "
          f"rate {n_esc / n_pa:.1%})" if n_pa else "pre-alert-only days: 0")
    print(f"formal-buy days: {n_formal} (warm/had pre-alert: {n_warm})")

    if have < 20:
        print("\n⚠ Fewer than 20 realized-return rows — too thin for a verdict. This is")
        print("  expected soon after the lifecycle epoch; re-run as batches accrue D1..D14.")

    cohorts = {
        "PA_ONLY": df.loc[df["is_pa_only"], "_fwd"],
        "PA_ESCALATED": df.loc[df["is_pa_only"] & df["escalated"], "_fwd"],
        "PA_FIZZLED": df.loc[df["is_pa_only"] & ~df["escalated"], "_fwd"],
        "FORMAL_ALL": df.loc[df["has_formal"], "_fwd"],
        "FORMAL_WARM": df.loc[df["has_formal"] & df["formal_warm"], "_fwd"],
        "FORMAL_COLD": df.loc[df["has_formal"] & ~df["formal_warm"], "_fwd"],
    }
    _print_cohorts(cohorts, f"Cohort forward returns (D{args.horizon}, fraction; 0.05=+5%)")

    print("\n=== Head-to-head (bootstrap 95% CI on mean diff; p_sign=confidence sign is real) ===")
    tests = [
        ("Q1 PA_ESCALATED - PA_FIZZLED  (does escalation mark better pre-alerts?)",
         cohorts["PA_ESCALATED"], cohorts["PA_FIZZLED"]),
        ("Q2 PA_ESCALATED - FORMAL_WARM (enter early at pre-alert vs wait for formal?)",
         cohorts["PA_ESCALATED"], cohorts["FORMAL_WARM"]),
        ("Q3 FORMAL_WARM - FORMAL_COLD  (does a pre-alert warm-up improve a formal buy?)",
         cohorts["FORMAL_WARM"], cohorts["FORMAL_COLD"]),
    ]
    n_tests = len(tests)
    for label, a, b in tests:
        diff, lo, hi, p_sign = _bootstrap_diff(a, b)
        if np.isnan(diff):
            print(f"  {label}\n     insufficient data (need >=5 per side)")
            continue
        # Multiple-testing haircut (PBO-lite): require the CI to exclude 0 AND survive
        # a Bonferroni-style bar on sign confidence across the n tests tried.
        excl0 = (lo > 0) or (hi < 0)
        bar = 1 - 0.05 / n_tests
        survives = excl0 and (p_sign >= bar)
        verdict = "EDGE (survives multi-test haircut)" if survives else \
                  "excludes 0 but fails haircut" if excl0 else "no edge (CI spans 0)"
        print(f"  {label}\n     diff {diff:+.4f}  95%CI [{lo:+.4f}, {hi:+.4f}]  "
              f"p_sign {p_sign:.2f}  -> {verdict}")

    _walk_forward(df, args.horizon, {
        "PA_ESCALATED": lambda s: s["is_pa_only"] & s["escalated"],
        "FORMAL_WARM": lambda s: s["has_formal"] & s["formal_warm"],
        "FORMAL_COLD": lambda s: s["has_formal"] & ~s["formal_warm"],
    })

    print("\nCaveats: PA_ESCALATED uses future info to label (descriptive, not tradeable).")
    print("FORMAL_WARM is the tradeable read of the escalation idea. Close-to-close, no")
    print("costs, recent epoch -> treat any single-window 'EDGE' as provisional until it")
    print("holds across folds. This tool decides nothing live; it informs the go/no-go.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
