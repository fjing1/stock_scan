"""Which features rank dip-in-uptrend signals best? (evidence for a DipRank score)

Takes every historical entry (Close>SMA200 & RSI<40 & %K<20), tags it with
candidate ranking features, and measures forward 10-day EXCESS-vs-SPY outcomes
by tercile. Tercile cut points are set on TRAIN (pre-2019) and applied to TEST
(2019-2026) so the ranking is validated out-of-sample, not fit to it.

A feature is useful for ranking only if its high vs low tercile separates
forward win rate / return monotonically on the TEST set.

    python rank_test.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cycle_patter_for_swing import compute_cycle_stoch  # noqa: E402

import yfinance as yf  # noqa: E402

BENCHMARK = "SPY"
BASKET = ["SPY", "QQQ", "DIA", "IWM", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP",
          "XLI", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "JPM", "XOM", "JNJ",
          "KO", "WMT", "CAT", "HD", "PG", "UNH", "DIS", "INTC", "CSCO", "ORCL",
          "MCD"]
H = 10
SPLIT = pd.Timestamp("2019-01-01")

# candidate ranking features (higher value -> expected better, except noted)
FEATURES = {
    "trend_str (C/MA200-1)":  "trend",
    "rsi (lower=oversold)":   "rsi",
    "stochK (lower=oversold)": "k",
    "cycle (lower=oversold)": "cycle",
    "mom126 (6m RS)":         "mom126",
    "mom252 (12m RS)":        "mom252",
    "pullback C/MA50-1":      "pb50",
    "atr% (lower=calm)":      "atrp",
    "dist 52w-high":          "dist52",
}


def fetch_batch(symbols, period, interval):
    raw = yf.download(symbols, period=period, interval=interval,
                      auto_adjust=False, progress=False, group_by="ticker",
                      threads=True)
    out = {}
    for s in symbols:
        try:
            d = raw[s].dropna(how="all").copy()
        except Exception:
            continue
        if len(d) == 0:
            continue
        if getattr(d.index, "tz", None) is not None:
            d.index = d.index.tz_localize(None)
        out[s] = d
    return out


def main():
    print("downloading ...", flush=True)
    data = fetch_batch(BASKET, "20y", "1d")
    bdf = data.get(BENCHMARK)
    spy = bdf["Close"].astype(float) if bdf is not None else None
    spy_fwd = spy.shift(-H) / spy - 1.0 if spy is not None else None

    rows = []
    for sym, df in data.items():
        if sym == BENCHMARK or len(df) < 300:
            continue
        try:
            cs = compute_cycle_stoch(df)
        except Exception:
            continue
        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()
        pc = close.shift(1)
        tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
        feat = pd.DataFrame(index=df.index)
        feat["trend"] = close / ma200 - 1
        feat["rsi"] = cs["rsi"].values
        feat["k"] = cs["stoch_k"].values
        feat["cycle"] = cs["cycle"].values
        feat["mom126"] = close / close.shift(126) - 1
        feat["mom252"] = close / close.shift(252) - 1
        feat["pb50"] = close / ma50 - 1
        feat["atrp"] = atr / close
        feat["dist52"] = close / high.rolling(252).max() - 1
        abs_fwd = close.shift(-H) / close - 1
        exc_fwd = abs_fwd - (spy_fwd.reindex(df.index) if spy_fwd is not None else 0)
        entry = ((close > ma200) & (cs["rsi"].values < 40) & (cs["stoch_k"].values < 20)).fillna(False)
        sub = feat[entry].copy()
        sub["exc"] = exc_fwd[entry].values
        sub["abs"] = abs_fwd[entry].values
        sub["train"] = (df.index[entry] < SPLIT)
        rows.append(sub)

    pool = pd.concat(rows).dropna(subset=["exc"])
    tr = pool[pool["train"]]
    te = pool[~pool["train"]]
    base_tr = (tr["exc"] > 0).mean() * 100
    base_te = (te["exc"] > 0).mean() * 100
    print(f"\nentries: train={len(tr)} test={len(te)}  | "
          f"baseline excess-win%  train {base_tr:.1f} / test {base_te:.1f}  (H={H}d)\n")

    print(f"{'feature':<26}{'tercile':<6}"
          f"{'tr_win%':>8}{'tr_exc%':>8}{'te_win%':>8}{'te_exc%':>8}{'teN':>6}")
    summary = []
    for label, col in FEATURES.items():
        q1, q2 = tr[col].quantile([1 / 3, 2 / 3])
        def bucket(s):
            return np.where(s[col] <= q1, "low", np.where(s[col] >= q2, "high", "mid"))
        tr_b, te_b = bucket(tr), bucket(te)
        spread = {}
        for name in ["low", "mid", "high"]:
            mtr = tr[tr_b == name]
            mte = te[te_b == name]
            twin = (mtr["exc"] > 0).mean() * 100 if len(mtr) else float("nan")
            texc = mtr["exc"].mean() * 100 if len(mtr) else float("nan")
            ewin = (mte["exc"] > 0).mean() * 100 if len(mte) else float("nan")
            eexc = mte["exc"].mean() * 100 if len(mte) else float("nan")
            spread[name] = ewin
            print(f"{label if name=='low' else '':<26}{name:<6}"
                  f"{twin:>8.1f}{texc:>8.2f}{ewin:>8.1f}{eexc:>8.2f}{len(mte):>6}")
        mono = spread["high"] - spread["low"]
        summary.append((label, mono))
        print(f"{'':<26}{'-> high-low test win% spread:':<30}{mono:>+6.1f}\n")

    print("RANK FEATURE USEFULNESS (test high-tercile minus low-tercile win%):")
    for label, mono in sorted(summary, key=lambda x: abs(x[1]), reverse=True):
        arrow = "useful" if abs(mono) >= 4 else "weak"
        print(f"  {label:<26}{mono:>+6.1f}  ({arrow})")


if __name__ == "__main__":
    main()
