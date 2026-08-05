"""Momentum/breakout ENTRIES x trend-following EXITS cross-study.

The entry sweep judged breakout entries under a mean-revert exit (unfair — a
breakout sells instantly above SMA20). Here each momentum entry is paired with
trend-following exits that let winners run, to see whether a SECOND, independent
(trend) strategy has edge — distinct from the mean-reversion dip-in-uptrend.

OOS: train < 2019, test 2019 -> present. One position at a time, max hold 252.
Trend systems have LOW win rate but high payoff — judge on profit factor / avg,
not win rate.

    python momentum_exit_sweep.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yfinance as yf  # noqa: E402

BENCHMARK = "SPY"
BASKET = ["SPY", "QQQ", "DIA", "IWM", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP",
          "XLI", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "JPM", "XOM", "JNJ",
          "KO", "WMT", "CAT", "HD", "PG", "UNH", "DIS", "INTC", "CSCO", "ORCL", "MCD"]
SPLIT = pd.Timestamp("2019-01-01")
MAX_HOLD = 252


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


def atr(high, low, close, n):
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# ---- trailing exit resolvers: (exit_index, exit_price) ----
def x_chandelier(A, ei, end, atrkey, k):
    high, low, close, atrN = A["high"], A["low"], A["close"], A[atrkey]
    hh, stop = high[ei], -np.inf
    for j in range(ei + 1, end + 1):
        hh = max(hh, high[j - 1])
        stop = max(stop, hh - k * atrN[j - 1])
        if low[j] <= stop:
            return j, min(stop, high[j])
    return end, close[end]


def x_ma_cross(A, ei, end, makey):
    close, ma = A["close"], A[makey]
    for j in range(ei + 1, end + 1):
        if close[j] < ma[j]:
            return j, close[j]
    return end, close[end]


def x_donchian(A, ei, end):
    close, ll = A["close"], A["low10_prev"]
    for j in range(ei + 1, end + 1):
        if close[j] < ll[j]:
            return j, close[j]
    return end, close[end]


def x_psar(A, ei, end):
    high, low, close = A["high"], A["low"], A["close"]
    af, ep_, sar = 0.02, high[ei], low[ei]
    for j in range(ei + 1, end + 1):
        sar = sar + af * (ep_ - sar)
        sar = min(sar, low[j - 1], low[max(ei, j - 2)])
        if low[j] <= sar:
            return j, sar
        if high[j] > ep_:
            ep_, af = high[j], min(0.2, af + 0.02)
    return end, close[end]


def x_time(A, ei, end, n):
    j = min(end, ei + n)
    return j, A["close"][j]


EXITS = {
    "Chand22": lambda A, ei, end: x_chandelier(A, ei, end, "atr22", 3.0),
    "Chand10": lambda A, ei, end: x_chandelier(A, ei, end, "atr10", 2.5),
    "close<MA20": lambda A, ei, end: x_ma_cross(A, ei, end, "ma20"),
    "close<MA50": lambda A, ei, end: x_ma_cross(A, ei, end, "ma50"),
    "Donch10exit": lambda A, ei, end: x_donchian(A, ei, end),
    "PSAR": lambda A, ei, end: x_psar(A, ei, end),
    "time63": lambda A, ei, end: x_time(A, ei, end, 63),
}


def build(df):
    o, h, l, c = (df["Open"].astype(float), df["High"].astype(float),
                  df["Low"].astype(float), df["Close"].astype(float))
    ma10, ma20, ma50 = c.rolling(10).mean(), c.rolling(20).mean(), c.rolling(50).mean()
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    entries = {
        "Donch20": c > h.rolling(20).max().shift(1),
        "Donch55": c > h.rolling(55).max().shift(1),
        "52wHigh": c >= c.rolling(252).max(),
        "MACDx": (macd > macd_sig) & (macd.shift(1) <= macd_sig.shift(1)),
        "MA10x50": (ma10 > ma50) & (ma10.shift(1) <= ma50.shift(1)),
    }
    A = {
        "high": h.values, "low": l.values, "close": c.values,
        "ma20": ma20.values, "ma50": ma50.values,
        "atr22": atr(h, l, c, 22).values, "atr10": atr(h, l, c, 10).values,
        "low10_prev": l.rolling(10).min().shift(1).values,
    }
    return {k: v.fillna(False).values for k, v in entries.items()}, A


def main():
    print("downloading daily (20y) ...", flush=True)
    data = fetch_batch(BASKET, "20y", "1d")
    bdf = data.get(BENCHMARK)
    spy_close = bdf["Close"].astype(float) if bdf is not None else None

    agg = {}   # (entry,exit) -> {"tr":[(ret,exc)],"te":[...]}
    for sym, df in data.items():
        if len(df) < 300:
            continue
        entries, A = build(df)
        dates = df.index
        close = A["close"]
        spy = spy_close.reindex(df.index).values if spy_close is not None else None
        n = len(close)
        for ename, sig in entries.items():
            for xname, xfn in EXITS.items():
                i = 1
                rows_tr, rows_te = [], []
                while i < n - 1:
                    if sig[i] and not sig[i - 1]:
                        ei = i
                        end = min(n - 1, ei + MAX_HOLD)
                        xi, xp = xfn(A, ei, end)
                        ret = xp / close[ei] - 1.0
                        exc = ret - (spy[xi] / spy[ei] - 1) if spy is not None and not np.isnan(spy[ei]) else np.nan
                        (rows_tr if dates[ei] < SPLIT else rows_te).append((ret, exc))
                        i = xi + 1
                    else:
                        i += 1
                a = agg.setdefault((ename, xname), {"tr": [], "te": []})
                a["tr"] += rows_tr
                a["te"] += rows_te

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
    for key, d in agg.items():
        s_tr, s_te = stats(d["tr"]), stats(d["te"])
        if s_tr and s_te:
            rows.append((key, s_tr, s_te))
    rows.sort(key=lambda x: (x[2]["pf"] if x[2]["pf"] != float("inf") else 99), reverse=True)

    print(f"\nmomentum entry x trend exit | test 2019-present | max hold {MAX_HOLD}")
    print(f"{'entry':<10}{'exit':<12}{'teN':>5}{'teWin%':>8}{'teAvg%':>8}{'tePF':>7}{'teExc%':>8}"
          f"  | {'trPF':>6}")
    for (e, x), s_tr, s_te in rows[:25]:
        star = " *" if (s_te["pf"] >= 1.5 and s_tr["pf"] >= 1.5) else ""
        print(f"{e:<10}{x:<12}{s_te['n']:>5}{s_te['win']:>8.1f}{s_te['avg']:>8.2f}"
              f"{s_te['pf']:>7.2f}{s_te['exc']:>8.1f}  | {s_tr['pf']:>6.2f}{star}")
    print("\n* = test PF>=1.5 AND train PF>=1.5. Trend systems: low win%, judge on PF/avg.")
    print("Compare to the mean-reversion system (StochK<20&>MA200 + SMA20 exit, test PF ~1.9).")


if __name__ == "__main__":
    main()
