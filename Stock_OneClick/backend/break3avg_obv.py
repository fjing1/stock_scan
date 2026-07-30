#!/usr/bin/env python3
"""Does an OBV (volume) 8/22 confirmation improve the price break SMA 8/22 entry?

OBV is orthogonal to price (it's volume-flow), so unlike the ma85 trend filter and
15m confirmation (both redundant with the price breakout), an OBV confirmation might
add real information. The Pine script carried OBV MAs: ema8v=ema(obv,8), ma21v=
sma(obv,21), addwatch=crossover(obv,ema8v). We build OBV EMA8 / SMA22 (mirroring the
price lines) and test several confirmation conditions stacked on break SMA 8/22.

Variants (all = price break 8/22 AND <obv condition>):
  base              : no OBV condition
  +OBV>OBV_SMA22    : volume flow above its slow MA (volume uptrend)
  +OBV_EMA8>SMA22   : volume momentum up (fast above slow)
  +OBV rising       : obv > obv[1]
  +OBV break(8/22)  : OBV above EMA8 & EMA8>EMA8[1] & OBV>SMA22 (mirror of price break)
Also: standalone "OBV break 8/22" alone (no price), to see OBV's own edge.

    ../../vcp_env/bin/python break3avg_obv.py

Same forward-edge methodology as break3avg_tune.py (42-name universe, D10 default).
Caveat: close-to-close, no costs; per-event edge vs baseline; watch fire count.
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
from rsi_ma_sweep import BASKET, INDEX, ma  # noqa: E402
from trend_alert_backtest import crossover  # noqa: E402

HORIZONS = [1, 5, 10, 20]


def obv(close: pd.Series, vol: pd.Series) -> pd.Series:
    sign = np.sign(close.diff().fillna(0.0))
    return (sign * vol).cumsum()


def prep(df: pd.DataFrame) -> dict:
    o, h, l, c = (df[k].astype(float) for k in ("Open", "High", "Low", "Close"))
    v = df["Volume"].astype(float)
    ohlc4 = (o + h + l + c) / 4
    sma8 = ma(ohlc4, 8, "SMA"); sma22 = ma(ohlc4, 22, "SMA"); sma3 = ma(ohlc4, 3, "SMA")
    base = (crossover(c, sma8) & (sma8 > sma8.shift(1))
            & crossover(c, sma22) & (sma3 > sma22)).fillna(False)
    ob = obv(c, v)
    return {
        "close": c, "base": base, "obv": ob,
        "obv_e8": ma(ob, 8, "EMA"), "obv_s22": ma(ob, 22, "SMA"),
        "fwd": {hh: c.shift(-hh) / c - 1 for hh in HORIZONS},
    }


def variants(p: dict) -> dict:
    b = p["base"]
    ob, e8, s22 = p["obv"], p["obv_e8"], p["obv_s22"]
    obv_up = ob > s22
    obv_mom = e8 > s22
    obv_rising = ob > ob.shift(1)
    obv_break = (ob > e8) & (e8 > e8.shift(1)) & (ob > s22)
    return {
        "base (price 8/22)": b,
        "+OBV>OBV_SMA22":    b & obv_up,
        "+OBV_EMA8>SMA22":   b & obv_mom,
        "+OBV rising":       b & obv_rising,
        "+OBV break(8/22)":  b & obv_break,
        "OBV break ALONE":   obv_break,        # no price condition
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", choices=["index", "basket", "both"], default="both")
    ap.add_argument("--rank-h", type=int, default=10, choices=HORIZONS)
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
    H = args.rank_h
    print(f"all-day baseline: D{H} mean {base_mean[H]:+.4f}\n")

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
        rows.append({"variant": name, "n": n, "win5": (agg[5] > 0).mean(),
                     "win10": (agg[10] > 0).mean(), f"mean{H}": rk.mean(),
                     "edge": rk.mean() - base_mean[H], "t": t})
    tbl = pd.DataFrame(rows)

    print(f"=== price break 8/22 + OBV(volume) confirmation (D{H} forward) ===")
    print(f"  {'variant':<20}{'n':>7}{'win5':>7}{'win10':>7}{f'mean{H}':>9}{'edge':>9}{'t':>7}")
    for _, r in tbl.iterrows():
        print(f"  {r['variant']:<20}{int(r['n']):>7}{r['win5']:>7.1%}{r['win10']:>7.1%}"
              f"{r[f'mean{H}']:>+9.4f}{r['edge']:>+9.4f}{r['t']:>7.2f}")

    b = tbl[tbl["variant"] == "base (price 8/22)"].iloc[0]
    print(f"\n  base 8/22: win10 {b['win10']:.1%}, mean{H} {b[f'mean{H}']:+.4f}, n {int(b['n'])}")
    for _, r in tbl.iterrows():
        if r["variant"] in ("base (price 8/22)", "OBV break ALONE"):
            continue
        print(f"  {r['variant']:<20} Δwin10 {r['win10']-b['win10']:>+6.1%}  "
              f"Δmean{H} {r[f'mean{H}']-b[f'mean{H}']:>+.4f}  keeps {r['n']/b['n']:>4.0%}")
    print("\nRead: OBV is orthogonal to price, so if a volume confirmation LIFTS win/mean it's")
    print("adding real info (unlike ma85 / 15m, which were redundant). If it only cuts fires")
    print("without lifting edge, volume flow is already baked into the price breakout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
