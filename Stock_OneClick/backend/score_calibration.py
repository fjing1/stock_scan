#!/usr/bin/env python3
"""Score calibration: does 观海买点分 / 卖出分 actually predict forward returns?

Reads the labeled dataset from build_dataset.py and reports, for BUY (观海买点分)
and SELL (卖出分) separately:
  * overall IC (Pearson + Spearman-via-rank) at horizons d1/d3/d5/last
  * DAY-DEMEANED IC (within-date, removing the regime/beta/which-day confound)
  * per-date IC distribution (sign-consistency across dates — the honest test)
  * winsorized score-bucket means / hit-rates (so single gappers don't dominate)
  * SUB-FEATURE univariate IC (rank120/RSI/L2_trend/H4_RSI/H4_FJ vs day-demeaned
    return) — the basis for rebuilding the score from features that survive
  * a GO/NO-GO line vs the "OOS Spearman >= +0.10, monotonic, >=3 dates" bar

No scipy (Spearman = Pearson on ranks). No network. Run:
    ../../vcp_env/bin/python build_dataset.py
    ../../vcp_env/bin/python score_calibration.py
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DS = BASE_DIR / "reports" / "strategy_dataset.csv"
SUBFEATS = ["rank120", "RSI", "L2_trend", "L2_pump", "H4_RSI", "H4_FJ"]


def _pair(a: pd.Series, b: pd.Series):
    m = a.notna() & b.notna()
    return a[m].astype(float), b[m].astype(float)


def ic(a: pd.Series, b: pd.Series):
    a, b = _pair(a, b)
    n = len(a)
    if n < 3 or a.std() == 0 or b.std() == 0:
        return n, np.nan, np.nan
    return n, round(a.corr(b), 3), round(a.rank().corr(b.rank()), 3)


def day_demean(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] - df.groupby("date")[col].transform("mean")


def section(df: pd.DataFrame, side: str, score_label: str):
    d = df[df["side"] == side].copy()
    print(f"\n{'='*72}\n{side}  (score = {score_label}; n={len(d)}, "
          f"scored+fwd={int((d['score'].notna() & d['fwd_last'].notna()).sum())})\n{'='*72}")
    if d["score"].notna().sum() < 5:
        print("  too few scored rows to analyze."); return

    print("  Overall IC of score vs forward return (Pearson / Spearman):")
    for h in ["fwd_d1", "fwd_d3", "fwd_d5", "fwd_last"]:
        n, p, s = ic(d["score"], d[h])
        print(f"    {h:<9} n={n:<4} Pearson={p}   Spearman={s}")

    # day-demeaned: remove the which-day/regime confound
    print("  DAY-DEMEANED IC (score vs date-demeaned return — confound removed):")
    for h in ["fwd_d3", "fwd_last"]:
        dd = d.dropna(subset=["score", h]).copy()
        if len(dd) >= 5:
            dd["dm"] = day_demean(dd, h)
            n, p, s = ic(dd["score"], dd["dm"])
            print(f"    {h:<9} n={n:<4} Pearson={p}   Spearman={s}")

    # per-date IC distribution
    print("  Per-date IC (Spearman of score vs fwd_last, dates with n>=4):")
    rows = []
    for dt, g in d.dropna(subset=["score", "fwd_last"]).groupby("date"):
        if len(g) >= 4:
            _, _, s = ic(g["score"], g["fwd_last"])
            rows.append((dt, len(g), s))
    if rows:
        ss = pd.Series([r[2] for r in rows]).dropna()
        for dt, ng, s in rows:
            print(f"    {dt}  n={ng:<4} Spearman={s}")
        if len(ss):
            print(f"    -> dates={len(ss)}  mean={ss.mean():+.3f}  median={ss.median():+.3f}  "
                  f"%positive={(ss>0).mean():.0%}")
    else:
        print("    (no date has n>=4)")

    # winsorized buckets
    print("  Winsorized (5/95%) fwd_last by score bucket:")
    b = d.dropna(subset=["score", "fwd_last"]).copy()
    if len(b):
        lo, hi = b["fwd_last"].quantile([0.05, 0.95])
        b["w"] = b["fwd_last"].clip(lo, hi)
        b["bk"] = pd.cut(b["score"], [0, 70, 80, 90, 100.01], right=False,
                         labels=["<70", "70-80", "80-90", "90-100"])
        agg = b.groupby("bk", observed=True).agg(n=("w", "size"), mean=("w", "mean"),
                                                  median=("w", "median"),
                                                  hit=("fwd_last", lambda s: (s > 0).mean()))
        for bk, r in agg.iterrows():
            print(f"    {str(bk):<7} n={int(r['n']):<4} mean={r['mean']:+.3%}  "
                  f"median={r['median']:+.3%}  hit>0={r['hit']:.0%}")


def subfeatures(df: pd.DataFrame):
    print(f"\n{'='*72}\nSUB-FEATURE univariate IC (BUY; feature vs DAY-DEMEANED fwd_last)\n"
          f"{'='*72}\n  Keep features with stable, non-trivial sign; drop/zero the rest.")
    d = df[(df["side"] == "BUY")].dropna(subset=["fwd_last"]).copy()
    if len(d) < 8:
        print("  too few rows."); return
    d["dm"] = day_demean(d, "fwd_last")
    for f in SUBFEATS:
        if d[f].notna().sum() >= 8:
            n, p, s = ic(d[f], d["dm"])
            flag = "  <- candidate" if (pd.notna(s) and abs(s) >= 0.10) else ""
            print(f"    {f:<10} n={n:<4} Pearson={p}   Spearman={s}{flag}")
        else:
            print(f"    {f:<10} (insufficient coverage: n={int(d[f].notna().sum())})")


def verdict(df: pd.DataFrame):
    print(f"\n{'='*72}\nGO / NO-GO\n{'='*72}")
    # BUY: higher score should mean higher forward return -> bar +0.10
    b = df[df["side"] == "BUY"].dropna(subset=["score", "fwd_last"])
    _, _, sb = ic(b["score"], b["fwd_last"])
    print(f"  BUY  观海买点分  Spearman(score, fwd_last) = {sb}  (bar to size: >= +0.10 OOS, monotonic, >=3 dates).")
    if pd.isna(sb) or sb < 0.10:
        print("       -> NOT USABLE for ranking/sizing. Trade equal-weight; do NOT weight by 观海买点分.")
    else:
        print("       -> in-sample bar met, but NOT out-of-sample. Re-test on dates > T before sizing.")
    # SELL: for a short, a NEGATIVE forward return is the win -> bar <= -0.10
    s = df[df["side"] == "SELL"].dropna(subset=["score", "fwd_last"])
    _, _, ss = ic(s["score"], s["fwd_last"])
    print(f"  SELL 卖出分    Spearman(score, fwd_last) = {ss}  (for a short, NEGATIVE is good; bar: <= -0.10 OOS, monotonic, >=3 dates).")
    if pd.isna(ss) or ss > -0.10:
        print("       -> NOT USABLE for sizing.")
    else:
        print("       -> directionally promising IN-SAMPLE only; buckets are non-monotonic and n is tiny -> do NOT size by it until OOS validation.")
    print("  Reminder: this dataset is ~one regime episode; treat ALL results as descriptive, not inferential.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ds", default=str(DEFAULT_DS))
    args = ap.parse_args()
    p = Path(args.ds)
    if not p.exists():
        print(f"Dataset not found: {p}\nRun build_dataset.py first."); return 1
    df = pd.read_csv(p)
    print(f"Dataset: {p}  shape={df.shape}  dates={df['date'].nunique()}")
    section(df, "BUY", "观海买点分")
    section(df, "SELL", "卖出分  [for a short, a NEGATIVE forward return = good]")
    subfeatures(df)
    verdict(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
