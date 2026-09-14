#!/usr/bin/env python3
"""Mean-reversion DIP entry + long-MA trend filter: does sma85 lift the win rate?

This is the setup the ma85 filter was actually earned on (RESEARCH.md #22): buying
oversold WEAKNESS (Connors RSI2), where a long-MA uptrend filter is supposed to
separate real dips-in-uptrend from falling knives.

Dip entry = RSI2 < threshold (2-period Wilder RSI). We compare, per threshold:
  no filter            : all oversold dips
  & Close>SMA85        : dip inside an uptrend (hi-conv tier from #22)
  & SMA50>SMA200 (GC)  : dip inside golden-cross regime (deployed dip_scan gate)
  & Close>SMA200       : dip above the 200-day
  & Close<SMA200 KNIFE : dip in a DOWNtrend (the falling-knife contrast)

Forward-return edge vs the all-day baseline, same methodology as the break tests.

    ../../vcp_env/bin/python mr_dip_trendfilter.py
    ../../vcp_env/bin/python mr_dip_trendfilter.py --rank-h 5

Caveat: close-to-close, no costs. Survivor basket flatters absolute dip win rates
(survivors bounce) — trust the WITH-vs-WITHOUT-filter and uptrend-vs-knife contrast.
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
from rsi_ma_sweep import BASKET, INDEX, rsi_wilder  # noqa: E402

HORIZONS = [1, 5, 10, 20]
THRESHOLDS = [5, 10]


def prep(df: pd.DataFrame) -> dict:
    o, h, l, c = (df[k].astype(float) for k in ("Open", "High", "Low", "Close"))
    return {
        "close": c,
        "rsi2": rsi_wilder(c, 2),
        "sma85": c.rolling(85).mean(),
        "sma200": c.rolling(200).mean(),
        "sma50": c.rolling(50).mean(),
        "fwd": {hh: c.shift(-hh) / c - 1 for hh in HORIZONS},
    }


def variants(p: dict) -> dict:
    c = p["close"]
    r2 = p["rsi2"]
    up85 = c > p["sma85"]
    up200 = c > p["sma200"]
    gc = p["sma50"] > p["sma200"]
    out = {}
    for t in THRESHOLDS:
        dip = r2 < t
        out[f"RSI2<{t}"] = dip
        out[f"RSI2<{t} &C>SMA85"] = dip & up85
        out[f"RSI2<{t} &GC50>200"] = dip & gc
        out[f"RSI2<{t} &C>SMA200"] = dip & up200
        out[f"RSI2<{t} KNIFE(<200)"] = dip & (~up200)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", choices=["index", "basket", "both"], default="both")
    ap.add_argument("--rank-h", type=int, default=5, choices=HORIZONS,
                    help="horizon for edge (MR is short-horizon; D5 default)")
    ap.add_argument("--period", default="max")
    args = ap.parse_args()

    names = {"index": INDEX, "basket": BASKET, "both": INDEX + BASKET}[args.universe]
    print(f"Downloading {len(names)} names ({args.universe})...")
    preps, base_fwd = [], {hh: [] for hh in HORIZONS}
    for sym in names:
        try:
            df = scan.download_daily(sym, period=args.period)
        except Exception:
            df = None
        if df is None or len(df) < 300:
            continue
        p = prep(df)
        preps.append(p)
        for hh in HORIZONS:
            base_fwd[hh].append(p["fwd"][hh].dropna())
    print(f"got data for {len(preps)}/{len(names)} names\n")
    base_mean = {hh: pd.concat(base_fwd[hh]).mean() for hh in HORIZONS}
    base_win = {hh: (pd.concat(base_fwd[hh]) > 0).mean() for hh in HORIZONS}
    H = args.rank_h
    print(f"all-day baseline: D{H} win {base_win[H]:.1%}  mean {base_mean[H]:+.4f}\n")

    vnames = list(variants(preps[0]).keys())
    rows = []
    for name in vnames:
        n, fr = 0, {hh: [] for hh in HORIZONS}
        for p in preps:
            e = variants(p)[name].fillna(False)
            n += int(e.sum())
            for hh in HORIZONS:
                fr[hh].append(p["fwd"][hh][e].dropna())
        agg = {hh: pd.concat(fr[hh]) if fr[hh] else pd.Series(dtype=float) for hh in HORIZONS}
        rk = agg[H]
        sd = rk.std(ddof=1)
        t = (rk.mean() / (sd / np.sqrt(len(rk)))) if len(rk) > 1 and sd else np.nan
        rows.append({"variant": name, "n": n, f"win{H}": (rk > 0).mean(),
                     "win10": (agg[10] > 0).mean(), f"mean{H}": rk.mean(),
                     "edge": rk.mean() - base_mean[H], "t": t})
    tbl = pd.DataFrame(rows)

    print(f"=== oversold DIP entry + trend filter (D{H} forward) ===")
    print(f"  {'variant':<22}{'n':>7}{f'win{H}':>7}{'win10':>7}{f'mean{H}':>9}{'edge':>9}{'t':>7}")
    for _, r in tbl.iterrows():
        mark = ""
        if "KNIFE" in r["variant"]:
            mark = "  <- downtrend"
        elif "SMA85" in r["variant"]:
            mark = "  <- ma85 tier"
        print(f"  {r['variant']:<22}{int(r['n']):>7}{r[f'win{H}']:>7.1%}{r['win10']:>7.1%}"
              f"{r[f'mean{H}']:>+9.4f}{r['edge']:>+9.4f}{r['t']:>7.2f}{mark}")

    # explicit with/without-filter deltas per threshold
    print("\n=== does the ma85 / uptrend filter lift the dip? (vs no-filter, same threshold) ===")
    for t in THRESHOLDS:
        b = tbl[tbl["variant"] == f"RSI2<{t}"].iloc[0]
        print(f"  RSI2<{t} base: win{H} {b[f'win{H}']:.1%}, mean{H} {b[f'mean{H}']:+.4f}, n {int(b['n'])}")
        for suff in ["&C>SMA85", "&GC50>200", "&C>SMA200", "KNIFE(<200)"]:
            r = tbl[tbl["variant"] == f"RSI2<{t} {suff}"].iloc[0]
            dwin = r[f"win{H}"] - b[f"win{H}"]
            dmean = r[f"mean{H}"] - b[f"mean{H}"]
            print(f"    {suff:<14} Δwin{H} {dwin:>+6.1%}  Δmean{H} {dmean:>+.4f}  "
                  f"keeps {r['n']/b['n']:>4.0%}")
    print("\nRead: if the uptrend filters LIFT win/mean and KNIFE(<200) is much worse, the")
    print("ma85/regime filter is doing its job on this (weakness-buying) setup — unlike the")
    print("breakout, where it hurt. Mean-reversion is short-horizon; weight D5 over D20.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
