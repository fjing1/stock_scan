"""Broad ENTRY-indicator sweep, same methodology as the rest of the project.

Tests a library of long-entry signals across families, each backtested as real
trades with a STANDARDIZED exit (close >= SMA20, the validated mean-revert exit,
capped 21 bars) so entries are directly comparable. Out-of-sample: train < 2019,
test 2019 -> present. Reports win%, avg, profit factor, and DETRENDED (vs-SPY)
win% per entry, train vs test. Each base signal is also tested with a >SMA200
trend filter (the ingredient that made the dip-in-uptrend entry work).

Families: mean-reversion/oversold, breakout/momentum, volume, candlestick.

    python entry_sweep.py
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
          "KO", "WMT", "CAT", "HD", "PG", "UNH", "DIS", "INTC", "CSCO", "ORCL", "MCD"]
SPLIT = pd.Timestamp("2019-01-01")
MAX_HOLD = 21


def fetch_batch(symbols, period, interval):
    raw = yf.download(symbols, period=period, interval=interval,
                      auto_adjust=False, progress=False, group_by="ticker", threads=True)
    out = {}
    for s in symbols:
        try:
            d = raw[s].dropna(how="all").copy()
        except Exception:
            continue
        if len(d):
            if getattr(d.index, "tz", None) is not None:
                d.index = d.index.tz_localize(None)
            out[s] = d
    return out


def rsi(close, n):
    ch = close.diff()
    up = ch.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-ch).clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50)


def build_entries(df, cs):
    """Return dict name -> boolean Series (long entry signals)."""
    o, h, l, c, v = (df["Open"].astype(float), df["High"].astype(float),
                     df["Low"].astype(float), df["Close"].astype(float),
                     df["Volume"].astype(float))
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    sma200 = c.rolling(200).mean()
    rsi2, rsi14 = rsi(c, 2), cs["rsi"]
    stochK = cs["stoch_k"]
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    ma10, ma50 = c.rolling(10).mean(), c.rolling(50).mean()
    rng = (h - l)
    body = (c - o).abs()
    ibs = ((c - l) / rng.replace(0, np.nan))
    down = c < c.shift(1)
    vol_ma = v.rolling(20).mean()

    e = {}
    # --- mean-reversion / oversold ---
    e["RSI2<5"] = rsi2 < 5
    e["RSI2<10"] = rsi2 < 10
    e["RSI14<30"] = rsi14 < 30
    e["StochK<20"] = stochK < 20
    e["close<BBlow"] = c < (sma20 - 2 * std20)
    e["3DownDays"] = down & down.shift(1) & down.shift(2)
    e["5DayLow"] = c <= c.rolling(5).min()
    e["IBS<0.2"] = ibs < 0.2
    # --- breakout / momentum ---
    e["20DayHighBO"] = c > h.rolling(20).max().shift(1)
    e["52wHigh"] = c >= c.rolling(252).max()
    e["MACDcross"] = (macd > macd_sig) & (macd.shift(1) <= macd_sig.shift(1))
    e["MA10xMA50"] = (ma10 > ma50) & (ma10.shift(1) <= ma50.shift(1))
    # --- volume / volatility ---
    e["NR7up"] = (rng <= rng.rolling(7).min()) & (c > o)
    e["VolSpikeUp"] = (v > 1.5 * vol_ma) & (c > o)
    # --- candlestick ---
    e["BullEngulf"] = (c.shift(1) < o.shift(1)) & (c > o) & (c > o.shift(1)) & (o < c.shift(1))
    lower_wick = (o.where(o < c, c) - l)
    upper_wick = (h - c.where(c > o, o))
    e["Hammer"] = (lower_wick >= 2 * body) & (upper_wick <= body) & (c >= o)

    base = {k: s.fillna(False) for k, s in e.items()}
    trend = (c > sma200).fillna(False)
    out = {}
    for k, s in base.items():
        out[k] = s
        out[k + " & >MA200"] = (s & trend)
    return out, sma20


def simulate(sig, close, sma20, dates):
    """One position at a time; exit when close>=SMA20 (cap MAX_HOLD). Rising-edge entry."""
    n = len(close)
    tr, te = [], []
    i = 1
    while i < n - 1:
        if sig[i] and not sig[i - 1]:
            ei = i
            ep = close[ei]
            end = min(n - 1, ei + MAX_HOLD)
            xi = end
            for j in range(ei + 1, end + 1):
                if close[j] >= sma20[j]:
                    xi = j
                    break
            ret = close[xi] / ep - 1.0
            (tr if dates[ei] < SPLIT else te).append((ret, ei, xi))
            i = xi + 1
        else:
            i += 1
    return tr, te


def main():
    print("downloading daily (20y) ...", flush=True)
    data = fetch_batch(BASKET, "20y", "1d")
    bdf = data.get(BENCHMARK)
    spy_close = bdf["Close"].astype(float) if bdf is not None else None

    agg = {}   # name -> {"tr":[(ret,exc)], "te":[...]}
    for sym, df in data.items():
        if len(df) < 260:
            continue
        try:
            cs = compute_cycle_stoch(df)
        except Exception:
            continue
        entries, sma20 = build_entries(df, cs)
        close = df["Close"].astype(float)
        spy = spy_close.reindex(df.index) if spy_close is not None else None
        c_arr, s_arr = close.values, sma20.values
        dates = df.index
        spy_arr = spy.values if spy is not None else None
        for name, sig in entries.items():
            tr, te = simulate(sig.values, c_arr, s_arr, dates)
            a = agg.setdefault(name, {"tr": [], "te": []})
            for bucket, rows in (("tr", tr), ("te", te)):
                for (ret, ei, xi) in rows:
                    exc = ret - (spy_arr[xi] / spy_arr[ei] - 1) if spy_arr is not None and not np.isnan(spy_arr[ei]) else np.nan
                    a[bucket].append((ret, exc))

    def stats(rows):
        if len(rows) < 20:
            return None
        r = np.array([x[0] for x in rows])
        ex = np.array([x[1] for x in rows]); ex = ex[~np.isnan(ex)]
        w, l = r[r > 0], r[r < 0]
        pf = w.sum() / abs(l.sum()) if l.sum() != 0 else float("inf")
        return dict(n=len(r), win=(r > 0).mean() * 100, avg=r.mean() * 100,
                    pf=pf, exc=(ex > 0).mean() * 100 if len(ex) else float("nan"))

    rows = []
    for name, d in agg.items():
        s_tr, s_te = stats(d["tr"]), stats(d["te"])
        if s_tr and s_te:
            rows.append((name, s_tr, s_te))
    rows.sort(key=lambda x: (x[2]["pf"] if x[2]["pf"] != float("inf") else 99), reverse=True)

    print(f"\nstandardized exit = close>=SMA20 (cap {MAX_HOLD}); test = 2019-present")
    print(f"{'entry':<24}{'teN':>6}{'teWin%':>8}{'teAvg%':>8}{'tePF':>7}{'teExc%':>8}"
          f"  | {'trPF':>6}{'trWin%':>8}")
    for name, s_tr, s_te in rows:
        star = " *" if (s_te["pf"] >= 1.8 and s_tr["pf"] >= 1.8 and s_te["exc"] >= 52) else ""
        print(f"{name:<24}{s_te['n']:>6}{s_te['win']:>8.1f}{s_te['avg']:>8.2f}"
              f"{s_te['pf']:>7.2f}{s_te['exc']:>8.1f}  | {s_tr['pf']:>6.2f}{s_tr['win']:>8.1f}{star}")
    print("\n* = robust: test PF>=1.8 AND train PF>=1.8 AND test detrended win%>=52")
    print("Detrended (teExc%) > ~52 and stable = real edge vs just being long.")


if __name__ == "__main__":
    main()
