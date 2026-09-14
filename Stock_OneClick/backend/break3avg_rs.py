#!/usr/bin/env python3
"""Rank break SMA 8/22 signals by 12-month relative strength (RS) — does it sort quality?

Prior research (RESEARCH.md #22 / rank_test.py / mtf-signal-alignment-no-edge): among
dip signals, 12-month RELATIVE STRENGTH is the strongest forward predictor (high-RS
~58% OOS vs ~52% base); oversold DEPTH has no ranking power. RS is cross-sectional /
point-in-time, so unlike ma85/OBV/15m (same-bar collinear with the breakout) it can
add orthogonal information. This tests RS-ranking on the break-8/22 MOMENTUM entry.

RS at signal time = stock trailing-252d return MINUS SPY trailing-252d return (excess,
no lookahead). Pool all signals, split into RS terciles, compare forward D5/D10/D20.

    ../../vcp_env/bin/python break3avg_rs.py

Caveat: close-to-close, no costs. Survivor basket COMPRESSES RS dispersion (all are
long-run winners) — a broad universe would sharpen this; treat 42-name as a first read.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
import scan_stocks as scan  # noqa: E402
from rsi_ma_sweep import BASKET, INDEX  # noqa: E402
from break3avg_scan import break3avg_signal, load_all_market  # noqa: E402

HORIZONS = [5, 10, 20]
LOOKBACK = 252


def batch_daily(symbols, period):
    """Batched full-history daily download (avoids single-ticker rate limits)."""
    import yfinance as yf
    out = {}
    for i in range(0, len(symbols), 100):
        chunk = symbols[i:i + 100]
        try:
            raw = yf.download(chunk, period=period, interval="1d", auto_adjust=True,
                              progress=False, group_by="ticker", threads=True)
        except Exception:
            continue
        for s in chunk:
            try:
                d = raw[s].dropna(how="all")
            except Exception:
                continue
            if len(d):
                out[s] = d
        print(f"  ...{min(i+100, len(symbols))}/{len(symbols)} downloaded, {len(out)} with data")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", choices=["index", "basket", "both"], default="both")
    ap.add_argument("--all", action="store_true", help="broad ~1000-name universe (real RS dispersion, batched)")
    ap.add_argument("--period", default="max")
    args = ap.parse_args()

    if args.all:
        syms = load_all_market()
        print(f"Downloading broad universe ({len(syms)} names) + SPY, batched...")
        frames = batch_daily([scan.to_yfinance_symbol(s) for s in syms], args.period)
        names = list(frames.keys())
        get = lambda s: frames.get(s)
    else:
        names = {"index": INDEX, "basket": BASKET, "both": INDEX + BASKET}[args.universe]
        print(f"Downloading {len(names)} names + SPY ({args.universe})...")
        get = lambda s: scan.download_daily(s, period=args.period)
    spy = scan.download_daily("SPY", period=args.period)["Close"].astype(float)
    spy_ret = spy / spy.shift(LOOKBACK) - 1   # trailing 12m return, indexed by date

    recs = []   # one row per break-8/22 signal
    n_names = 0
    for sym in names:
        try:
            df = get(sym)
        except Exception:
            df = None
        if df is None or len(df) < LOOKBACK + max(HORIZONS) + 5:
            continue
        n_names += 1
        c = df["Close"].astype(float)
        e = break3avg_signal(df, 8, 22, "SMA", "SMA")
        idx = np.where(e.values)[0]
        for i in idx:
            if i < LOOKBACK or i + max(HORIZONS) >= len(c):
                continue
            d0 = c.index[i]
            stock_rs = c.iloc[i] / c.iloc[i - LOOKBACK] - 1
            spy_rs = spy_ret.asof(d0)
            if not np.isfinite(spy_rs):
                continue
            rec = {"symbol": sym, "date": d0, "rs_excess": stock_rs - spy_rs, "rs_abs": stock_rs}
            for hh in HORIZONS:
                rec[f"fwd{hh}"] = c.iloc[i + hh] / c.iloc[i] - 1
            recs.append(rec)
    d = pd.DataFrame(recs)
    print(f"got {n_names} names, {len(d)} break-8/22 signals with RS + forward\n")
    if len(d) < 60:
        print("Too few signals for tercile analysis.")
        return 0

    # tercile by RS excess vs SPY
    d["tercile"] = pd.qcut(d["rs_excess"], 3, labels=["bottom (weak RS)", "mid", "top (strong RS)"])
    print("=== break-8/22 forward returns by 12m RELATIVE-STRENGTH tercile (vs SPY) ===")
    print(f"  {'tercile':<20}{'n':>6}{'RS med':>8}"
          + "".join(f"{'win'+str(h):>7}" for h in HORIZONS)
          + "".join(f"{'mean'+str(h):>9}" for h in HORIZONS))
    for terc in ["bottom (weak RS)", "mid", "top (strong RS)"]:
        s = d[d["tercile"] == terc]
        line = f"  {terc:<20}{len(s):>6}{s['rs_excess'].median():>+8.0%}"
        for h in HORIZONS:
            line += f"{(s[f'fwd{h}']>0).mean():>7.1%}"
        for h in HORIZONS:
            line += f"{s[f'fwd{h}'].mean():>+9.2%}"
        print(line)

    top = d[d["tercile"] == "top (strong RS)"]
    bot = d[d["tercile"] == "bottom (weak RS)"]
    print("\n=== top vs bottom tercile spread ===")
    for h in HORIZONS:
        dw = ((top[f"fwd{h}"] > 0).mean() - (bot[f"fwd{h}"] > 0).mean()) * 100
        dm = top[f"fwd{h}"].mean() - bot[f"fwd{h}"].mean()
        # welch t on the mean difference
        a, b = top[f"fwd{h}"], bot[f"fwd{h}"]
        se = np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
        t = (a.mean() - b.mean()) / se if se else np.nan
        print(f"  D{h}: Δwin {dw:>+5.1f}pp   Δmean {dm:>+.2%}   t {t:>+.2f}")

    print("\nRead: if the TOP RS tercile has higher win rate & mean than the BOTTOM (and t>~2),")
    print("RS ranking sorts breakout quality → take the strongest-RS names among each day's")
    print("hits. If terciles are flat, RS adds nothing here (survivor RS compression is a risk).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
