#!/usr/bin/env python3
"""Isolated forward-tracking harness for the RESEARCH pool — makes a bias-reduced
edge test possible.

The research names (enable=0) never enter the live daily lifecycle, so they have no
forward returns and can't be backtested. This harness maintains a SEPARATE,
point-in-time signal ledger and fills realized forward returns as calendar time
passes — the honest way to accrue a bias-reduced (current-membership) dataset.

Design / integrity guarantees:
  * one row per (symbol, signal_date); ENTRY fields (score, RSI, rank120, d0_close,
    signal_types...) are FROZEN once recorded — re-ingest never rewrites them. This
    is the anti-lookahead property: the signal is captured as it was on that bar.
  * forward fields (fwd_d1..fwd_d14 close-pct-vs-D0, plus o1_vs_d0 = the D1 OPEN vs
    D0 close, for executable next-open entry math) are (re)filled from ACTUAL daily
    closes, only for days that have already elapsed (never future).
  * fully isolated under reports/research_forward/ — never touches the live workbook,
    scan_result_latest.xlsx, history/, or the live lifecycle.

Downstream: once rows accrue forward days, entry_open.py / horizon_sweep.py math
applies directly (executable enter-D1-open -> exit-DN = (1+fwd_dN)/(1+o1_vs_d0)-1).

    # daily driver: fresh isolated scan -> ingest -> fill realized returns -> status
    ../../vcp_env/bin/python research_scan.py --out ../reports/research_scan_all.csv   # (or shards)
    ../../vcp_env/bin/python research_forward.py update --from-csv ../reports/research_scan_all.csv

    ../../vcp_env/bin/python research_forward.py ingest --from-csv ../reports/research_scan_all.csv
    ../../vcp_env/bin/python research_forward.py fill        # backfill elapsed forward days
    ../../vcp_env/bin/python research_forward.py report      # coverage + (once ready) horizon read
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
import scan_stocks as scan  # noqa: E402

BASE_DIR = BACKEND_DIR.parent
STORE_DIR = BASE_DIR / "reports" / "research_forward"
LEDGER = STORE_DIR / "ledger.csv"
MAX_FWD = 14

ENTRY_COLS = ["tier", "sector", "signal_types", "model", "d0_close",
              "buy_score", "buy_score_raw", "RSI", "rank120", "H4_RSI", "H4_FJ",
              "Gann_0", "gain_pct", "vol_ratio", "first_seen"]
FWD_COLS = ["o1_vs_d0"] + [f"fwd_d{i}" for i in range(1, MAX_FWD + 1)] + ["days_filled", "complete", "last_updated"]
KEY = ["symbol", "signal_date"]
ALL_COLS = KEY + ENTRY_COLS + FWD_COLS


def _today():
    return datetime.now().date()


def load_ledger() -> pd.DataFrame:
    if not LEDGER.exists():
        return pd.DataFrame(columns=ALL_COLS)
    df = pd.read_csv(LEDGER)
    for c in ALL_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["signal_date"] = df["signal_date"].astype(str)
    return df[ALL_COLS]


def save_ledger(df: pd.DataFrame) -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(["signal_date", "symbol"]).reset_index(drop=True)
    df.to_csv(LEDGER, index=False)


def _live_symbols() -> set[str]:
    try:
        _di, dm = scan.load_input_and_meta(scan.INPUT_FILE)
        if dm is None or dm.empty or "enable" not in dm.columns:
            return set()
        en = pd.to_numeric(dm["enable"], errors="coerce").fillna(1).astype(int)
        return set(dm.loc[en == 1, "symbol"].astype(str).str.strip().str.upper())
    except Exception:
        return set()


def ingest(csv_paths: list[str]) -> int:
    """Upsert BUY signals from research-scan CSV(s) as one frozen row per
    (symbol, signal_date). Existing rows are NOT modified."""
    frames = []
    for p in csv_paths:
        pth = Path(p)
        if not pth.is_absolute():
            pth = (BACKEND_DIR / pth).resolve()
        if pth.exists() and pth.stat().st_size:
            frames.append(pd.read_csv(pth))
    if not frames:
        print("ingest: no non-empty CSVs found"); return 0
    df = pd.concat(frames, ignore_index=True)
    df = df[df["signal_side"].astype(str).str.upper() == "BUY"].copy()
    if df.empty:
        print("ingest: no BUY rows"); return 0
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["signal_date"] = pd.to_datetime(df["signal_date"], errors="coerce").dt.date.astype(str)
    for c in ["buy_score", "buy_score_raw", "RSI", "rank120", "H4_RSI", "H4_FJ",
              "Gann_0", "Gann_gain_pct", "close", "volume", "vol_ma20"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    live = _live_symbols()
    today = _today().isoformat()
    ledger = load_ledger()
    have = set(zip(ledger["symbol"], ledger["signal_date"]))

    new_rows = []
    for (sym, sdate), g in df.groupby(["symbol", "signal_date"]):
        if (sym, sdate) in have:
            continue  # frozen — never rewrite an existing point-in-time record
        g = g.sort_values("buy_score_raw", ascending=False)
        best = g.iloc[0]
        vol = pd.to_numeric(best.get("volume"), errors="coerce")
        vma = pd.to_numeric(best.get("vol_ma20"), errors="coerce")
        new_rows.append({
            "symbol": sym, "signal_date": sdate,
            "tier": "live" if sym in live else "research",
            "sector": best.get("板块", ""),
            "signal_types": " + ".join(sorted(set(g["signal_type"].astype(str)))),
            "model": best.get("model", ""),
            "d0_close": pd.to_numeric(best.get("close"), errors="coerce"),
            "buy_score": pd.to_numeric(g["buy_score"], errors="coerce").max(),
            "buy_score_raw": pd.to_numeric(g["buy_score_raw"], errors="coerce").max(),
            "RSI": pd.to_numeric(best.get("RSI"), errors="coerce"),
            "rank120": pd.to_numeric(best.get("rank120"), errors="coerce"),
            "H4_RSI": pd.to_numeric(best.get("H4_RSI"), errors="coerce"),
            "H4_FJ": pd.to_numeric(best.get("H4_FJ"), errors="coerce"),
            "Gann_0": pd.to_numeric(best.get("Gann_0"), errors="coerce"),
            "gain_pct": pd.to_numeric(best.get("Gann_gain_pct"), errors="coerce"),
            "vol_ratio": (vol / vma) if (pd.notna(vol) and pd.notna(vma) and vma) else np.nan,
            "first_seen": today,
            "o1_vs_d0": np.nan, **{f"fwd_d{i}": np.nan for i in range(1, MAX_FWD + 1)},
            "days_filled": 0, "complete": False, "last_updated": np.nan,
        })
    if not new_rows:
        print(f"ingest: 0 new signals (all {len(df.groupby(['symbol','signal_date']))} already recorded)")
        return 0
    add = pd.DataFrame(new_rows)
    ledger = add if ledger.empty else pd.concat([ledger, add], ignore_index=True)
    save_ledger(ledger)
    print(f"ingest: +{len(new_rows)} new signal rows  (ledger now {len(ledger)})")
    return len(new_rows)


def fill() -> int:
    """Fill forward returns from actual daily closes for elapsed days. Idempotent —
    refreshes incomplete rows; frozen entry fields untouched."""
    ledger = load_ledger()
    if ledger.empty:
        print("fill: empty ledger — run ingest first"); return 0
    today = _today()
    todo = ledger[~ledger["complete"].fillna(False).astype(bool)]
    syms = sorted(todo["symbol"].unique().tolist())
    print(f"fill: {len(todo)} incomplete rows across {len(syms)} symbols", flush=True)

    updated = 0
    for n, sym in enumerate(syms, 1):
        try:
            d = scan.download_daily(sym, period="1y")
        except Exception:
            d = None
        if d is None or d.empty or "Close" not in d.columns:
            continue
        d = d.copy()
        d.index = pd.to_datetime(d.index).date
        idx = list(d.index)
        pos = {dt: i for i, dt in enumerate(idx)}
        rows = ledger[(ledger["symbol"] == sym) & (~ledger["complete"].fillna(False).astype(bool))]
        for li, r in rows.iterrows():
            try:
                sd = datetime.fromisoformat(str(r["signal_date"])).date()
            except Exception:
                continue
            if sd not in pos:
                continue
            p = pos[sd]
            c0 = float(d["Close"].iloc[p])
            if not c0:
                continue
            # D1 open (executable next-session entry) vs D0 close
            if p + 1 < len(idx) and idx[p + 1] <= today and "Open" in d.columns:
                o1 = float(d["Open"].iloc[p + 1])
                ledger.at[li, "o1_vs_d0"] = o1 / c0 - 1 if o1 else np.nan
            days = 0
            for k in range(1, MAX_FWD + 1):
                if p + k < len(idx) and idx[p + k] <= today:
                    ck = float(d["Close"].iloc[p + k])
                    ledger.at[li, f"fwd_d{k}"] = ck / c0 - 1
                    days = k
                # else leave NaN (future day)
            ledger.at[li, "days_filled"] = days
            ledger.at[li, "complete"] = bool(days >= MAX_FWD)
            ledger.at[li, "last_updated"] = today.isoformat()
            updated += 1
        if n % 100 == 0 or n == len(syms):
            print(f"  filled {n}/{len(syms)} symbols", flush=True)
    save_ledger(ledger)
    print(f"fill: updated {updated} rows")
    return updated


def _winrate_block(sub: pd.DataFrame, label: str) -> None:
    print(f"\n  {label} (n={len(sub)})")
    print(f"    {'H':<5}{'n':>6}{'win':>9}{'mean':>10}")
    for k in range(1, 6):
        x = pd.to_numeric(sub.get(f"fwd_d{k}"), errors="coerce").dropna()
        if len(x):
            print(f"    D{k:<4}{len(x):>6}{(x > 0).mean():>8.1%}{x.mean():>+10.4f}")
        else:
            print(f"    D{k:<4}{0:>6}{'—':>9}{'—':>10}")


def report() -> int:
    ledger = load_ledger()
    if ledger.empty:
        print("report: empty ledger"); return 0
    n = len(ledger)
    with_fwd = int((pd.to_numeric(ledger["days_filled"], errors="coerce").fillna(0) >= 1).sum())
    complete = int(ledger["complete"].fillna(False).astype(bool).sum())
    df = ledger.copy()
    df["dfil"] = pd.to_numeric(df["days_filled"], errors="coerce").fillna(0).astype(int)
    print(f"ledger: {n} signals | with >=1 fwd day: {with_fwd} | complete(D14): {complete}")
    print(f"date range: {df['signal_date'].min()} .. {df['signal_date'].max()}")
    print(f"tier: " + ", ".join(f"{k}={v}" for k, v in df['tier'].value_counts().items()))
    print(f"days_filled: median {int(df['dfil'].median())}, max {int(df['dfil'].max())}")

    ready = df[df["dfil"] >= 1]
    if with_fwd < 30:
        print(f"\n⚠ only {with_fwd} rows have realized forward days — too thin for a read.")
        print("  Re-run the daily driver for a few weeks; forward days accrue automatically.")
        return 0
    print("\n=== Horizon win-rate (executable close-to-close from D0; screen-not-edge) ===")
    _winrate_block(ready, "ALL research-pool BUY")
    formal = ready[ready["signal_types"].astype(str).str.contains("正式买入")]
    if len(formal) >= 20:
        _winrate_block(formal, "正式买入 (formal) only")
    hi = ready[pd.to_numeric(ready["buy_score"], errors="coerce") >= 95]
    if len(hi) >= 20:
        _winrate_block(hi, "buy_score >= 95")
    print("\nNote: point-in-time, current-membership (not fully survivorship-free), no costs.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("ingest"); pi.add_argument("--from-csv", required=True, help="comma-separated CSV path(s)")
    sub.add_parser("fill")
    sub.add_parser("report")
    pu = sub.add_parser("update"); pu.add_argument("--from-csv", required=True, help="ingest then fill then report")
    args = ap.parse_args()

    if args.cmd == "ingest":
        ingest([p.strip() for p in args.from_csv.split(",") if p.strip()])
    elif args.cmd == "fill":
        fill()
    elif args.cmd == "report":
        report()
    elif args.cmd == "update":
        ingest([p.strip() for p in args.from_csv.split(",") if p.strip()])
        fill()
        print()
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
