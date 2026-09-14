#!/usr/bin/env python3
"""RSI/OBV DIVERGENCE as a reversal signal -- a different mechanism than every RSI-MA
crossover test so far (all of which are trend-following and all of which lost to
buy-and-hold: [[rsi-ma-crossover-no-edge]]). Divergence is a REVERSAL bet: price makes a
new pivot low/high that the indicator doesn't confirm.

Mirrors xunlong_panel.pine's (currently-off-by-default) divergence block exactly:
  - pivots via ta.pivotlow/pivothigh(leftbars=5, rightbars=5) -- a pivot at bar i is only
    CONFIRMED at bar i+rightbars, which is when the signal can actually fire. No lookahead:
    the confirmation window only ever looks at data up to the firing bar.
  - bullish divergence: new pivot LOW in price that is LOWER than the prior confirmed pivot
    low, while the indicator's value at that pivot is HIGHER than at the prior pivot low
    (price weaker, momentum/flow not confirming -> potential reversal up).
  - bearish divergence: mirror image on pivot highs.
  - gap between the two pivots' actual bars must be in [rangeLower=5, rangeUpper=60], same
    as the Pine defaults.

Tests the indicator as BOTH price-RSI (classic divergence) and OBV-RSI (the volume-based
version the panel now runs on by default) so the two are directly comparable, same as
rsi_obv_ma_compare.py did for the crossover mechanism.

    ../../vcp_env/bin/python rsi_obv_divergence.py --universe both
"""
from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import scan_stocks as scan
from rsi_ma_sweep import BASKET, INDEX, metrics  # noqa: F401  (metrics unused here, kept for parity)
from rsi_obv_ma_compare import obv_series, rsi_wilder_on

HORIZONS = [1, 5, 10, 20]
LEFT, RIGHT = 5, 5
RANGE_LOWER, RANGE_UPPER = 5, 60


def confirmed_pivots(series: pd.Series, left: int = LEFT, right: int = RIGHT) -> tuple[pd.Series, pd.Series]:
    """Returns (pivot_low_confirmed, pivot_high_confirmed): booleans TRUE at the CONFIRMATION
    bar (right bars after the actual pivot), matching ta.pivotlow/pivothigh's semantics.
    window_min/max at position j covers series[j-(left+right) : j+1] == series[i-left : i+right+1]
    where i = j-right is the candidate pivot bar -- exactly the pivot definition, and it only
    ever reads data up to j, so there is no lookahead."""
    win = left + right + 1
    window_min = series.rolling(win).min()
    window_max = series.rolling(win).max()
    candidate = series.shift(right)
    return (candidate == window_min), (candidate == window_max)


def divergence_signals(price: pd.Series, indicator: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Bullish/bearish divergence signals, TRUE at the bar they become tradeable (the
    confirmation bar of the SECOND pivot, i=j-RIGHT)."""
    plo, phi = confirmed_pivots(price)
    n = len(price)
    bull = pd.Series(False, index=price.index)
    bear = pd.Series(False, index=price.index)

    prev_low_i = None
    prev_high_i = None
    price_v = price.values
    ind_v = indicator.values
    plo_v = plo.values
    phi_v = phi.values
    for j in range(n):
        i = j - RIGHT
        if i < 0:
            continue
        if plo_v[j]:
            if prev_low_i is not None:
                gap = i - prev_low_i
                if RANGE_LOWER <= gap <= RANGE_UPPER:
                    if (price_v[i] < price_v[prev_low_i]) and (ind_v[i] > ind_v[prev_low_i]):
                        bull.iloc[j] = True
            prev_low_i = i
        if phi_v[j]:
            if prev_high_i is not None:
                gap = i - prev_high_i
                if RANGE_LOWER <= gap <= RANGE_UPPER:
                    if (price_v[i] > price_v[prev_high_i]) and (ind_v[i] < ind_v[prev_high_i]):
                        bear.iloc[j] = True
            prev_high_i = i
    return bull, bear


def prep(df: pd.DataFrame, rsi_len: int) -> dict:
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)
    price_rsi = rsi_wilder_on(close, rsi_len)
    obv_rsi = rsi_wilder_on(obv_series(close, vol), rsi_len)
    bull_p, bear_p = divergence_signals(close, price_rsi)
    bull_o, bear_o = divergence_signals(close, obv_rsi)
    fwd = {h: close.shift(-h) / close - 1 for h in HORIZONS}
    return {"close": close, "fwd": fwd,
            "bull_price": bull_p, "bear_price": bear_p,
            "bull_obv": bull_o, "bear_obv": bear_o}


def summarize(label: str, preps: list, key: str, base_mean: dict, rank_h: int):
    n = 0
    fr = {h: [] for h in HORIZONS}
    for p in preps:
        sig = p[key]
        n += int(sig.sum())
        for h in HORIZONS:
            fr[h].append(p["fwd"][h][sig].dropna())
    agg = {h: (pd.concat(fr[h]) if fr[h] else pd.Series(dtype=float)) for h in HORIZONS}
    rk = agg[rank_h]
    sd = rk.std(ddof=1)
    t = (rk.mean() / (sd / np.sqrt(len(rk)))) if len(rk) > 1 and sd else np.nan
    win5 = (agg[5] > 0).mean() if len(agg[5]) else np.nan
    win10 = (agg[10] > 0).mean() if len(agg[10]) else np.nan
    mean_h = rk.mean() if len(rk) else np.nan
    edge = mean_h - base_mean[rank_h] if pd.notna(mean_h) else np.nan
    return {"label": label, "n": n, "win5": win5, "win10": win10,
            f"mean{rank_h}": mean_h, "edge": edge, "t": t}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", choices=["index", "basket", "both"], default="both")
    ap.add_argument("--rsi-len", type=int, default=14)
    ap.add_argument("--rank-h", type=int, default=10, choices=HORIZONS)
    ap.add_argument("--period", default="max")
    args = ap.parse_args()

    names = {"index": INDEX, "basket": BASKET, "both": INDEX + BASKET}[args.universe]
    print(f"Downloading {len(names)} names ({args.universe})...")
    preps = []
    for sym in names:
        try:
            df = scan.download_daily(sym, period=args.period)
        except Exception:
            df = None
        if df is None or len(df) < 300 or "Volume" not in df.columns:
            continue
        preps.append(prep(df, args.rsi_len))
    print(f"got data for {len(preps)}/{len(names)} names\n")
    if not preps:
        print("no data")
        return 1

    base_fwd = {h: pd.concat([p["fwd"][h].dropna() for p in preps]) for h in HORIZONS}
    base_mean = {h: base_fwd[h].mean() for h in HORIZONS}
    H = args.rank_h
    print(f"all-day baseline: D{H} mean {base_mean[H]:+.4f}\n")

    rows = []
    for key, label in (
        ("bull_price", "BULLISH div (price-RSI)"),
        ("bull_obv",   "BULLISH div (OBV-RSI)"),
        ("bear_price", "BEARISH div (price-RSI)"),
        ("bear_obv",   "BEARISH div (OBV-RSI)"),
    ):
        rows.append(summarize(label, preps, key, base_mean, H))
    tbl = pd.DataFrame(rows)

    print(f"=== RSI/OBV divergence, forward D{H} return (mirrors xunlong_panel.pine's divergence block) ===")
    print(f"  {'signal':<26}{'n':>7}{'win5':>7}{'win10':>7}{f'mean{H}':>9}{'edge':>9}{'t':>7}")
    for _, r in tbl.iterrows():
        print(f"  {r['label']:<26}{int(r['n']):>7}{r['win5']:>7.1%}{r['win10']:>7.1%}"
              f"{r[f'mean{H}']:>+9.4f}{r['edge']:>+9.4f}{r['t']:>7.2f}")

    print(f"\n=== walk-forward by calendar year: BULLISH divergence (D5 win rate) vs all-day baseline ===")
    for key, label in (("bull_price", "price-RSI"), ("bull_obv", "OBV-RSI")):
        print(f"  -- {label} --")
        all_events = []
        for p in preps:
            sig = p[key]
            idx = sig[sig].index
            for dt in idx:
                v = p["fwd"][5].get(dt, np.nan)
                if pd.notna(v):
                    all_events.append((dt, v))
        if not all_events:
            print("    no events")
            continue
        edf = pd.DataFrame(all_events, columns=["date", "fwd5"]).set_index("date").sort_index()
        wins, n_yr = 0, 0
        for y in sorted(edf.index.year.unique()):
            sub = edf[edf.index.year == y]
            if len(sub) < 10:
                continue
            n_yr += 1
            win = (sub["fwd5"] > 0).mean()
            is_win = win > 0.5
            wins += int(is_win)
            print(f"    {y}: n={len(sub):>4}  win5={win:>6.1%}  {'Y' if is_win else 'n'}")
        print(f"    -> beat coin-flip in {wins}/{n_yr} years")

    print("\nRead: divergence is a REVERSAL bet, mechanically different from every RSI-MA")
    print("crossover tested so far (all of which lost to buy-and-hold). A positive, t>2,")
    print("year-consistent edge on BULLISH divergence would be a genuinely new finding, not")
    print("a restatement of rsi-ma-crossover-no-edge. Bearish divergence's 'edge' should be")
    print("NEGATIVE forward return to be useful (it's a reversal-down bet, not a buy signal).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
