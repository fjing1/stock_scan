"""Literature exit strategies vs the %K>=70 champion (dip-in-uptrend entry).

Implements named, mechanical exits from the trading literature and backtests each
against the locked entry over 20y daily data, train (<2019) / test (2019-2026),
ranked by profit factor. One position at a time; max hold 21 bars.

Exits (category in brackets):
  K70           Stoch %K >= 70                         [sell-into-strength, current champion]
  RSI2>70       Connors RSI(2) >= 70                    [sell-into-strength]
  close>SMA5    Connors ETF exit, close above 5-SMA     [sell-into-strength]
  close>SMA10   close above 10-SMA                      [sell-into-strength]
  BBmid         close >= SMA20 (Bollinger middle band)  [reversion target]
  BBupper       close >= SMA20 + 2*std20                [sell-into-strength]
  firstUpClose  first bar with close > prior close      [fast time/target]
  firstProfit   first bar closing above entry           [fast target, Connors]
  Chandelier22  highest_high_since_entry - 3*ATR(22)    [trailing volatility stop]
  Chandelier10  highest_high_since_entry - 2*ATR(10)    [tight trailing stop]
  PSAR          Parabolic SAR (0.02/0.2)                [trailing stop / reverse]
  RSI14>70      RSI(14) >= 70                           [slow sell-into-strength]
  time_10       exit after 10 bars                      [time stop baseline]

Sources (standard references; NOT live-verified this session — sandbox is
Yahoo-only): Connors & Alvarez "Short Term Trading Strategies That Work";
LeBeau Chandelier (StockCharts); Wilder ATR/SAR/RSI; Bollinger Bands.

    python exit_strategies_lit.py
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


def rsi(close, length):
    ch = close.diff()
    up = ch.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    dn = (-ch).clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def atr(high, low, close, length):
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


# ---- per-trade exit resolvers (return (exit_index, exit_price)) ----
def exit_close_signal(sig, close, ei, end):
    for j in range(ei + 1, end + 1):
        if sig[j]:
            return j, close[j]
    return end, close[end]


def exit_first_profit(close, ei, end, up_only):
    ep = close[ei]
    for j in range(ei + 1, end + 1):
        if (close[j] > close[j - 1]) if up_only else (close[j] > ep):
            return j, close[j]
    return end, close[end]


def exit_chandelier(high, low, close, atrN, ei, end, k):
    hh, stop = high[ei], -np.inf
    for j in range(ei + 1, end + 1):
        hh = max(hh, high[j - 1])
        stop = max(stop, hh - k * atrN[j - 1])
        if low[j] <= stop:
            return j, min(stop, high[j])      # filled at stop (approx)
    return end, close[end]


def exit_psar(high, low, close, ei, end):
    af, ep_, sar = 0.02, high[ei], low[ei]
    for j in range(ei + 1, end + 1):
        sar = sar + af * (ep_ - sar)
        sar = min(sar, low[j - 1], low[max(ei, j - 2)])
        if low[j] <= sar:
            return j, sar
        if high[j] > ep_:
            ep_, af = high[j], min(0.2, af + 0.02)
    return end, close[end]


RULES = ["K70", "RSI2>70", "close>SMA5", "close>SMA10", "BBmid", "BBupper",
         "firstUpClose", "firstProfit", "Chandelier22", "Chandelier10", "PSAR",
         "RSI14>70", "time_10"]


def resolve(rule, A, ei, end):
    if rule == "time_10":
        j = min(end, ei + 10)
        return j, A["close"][j]
    if rule == "firstProfit":
        return exit_first_profit(A["close"], ei, end, up_only=False)
    if rule == "firstUpClose":
        return exit_first_profit(A["close"], ei, end, up_only=True)
    if rule == "Chandelier22":
        return exit_chandelier(A["high"], A["low"], A["close"], A["atr22"], ei, end, 3.0)
    if rule == "Chandelier10":
        return exit_chandelier(A["high"], A["low"], A["close"], A["atr10"], ei, end, 2.0)
    if rule == "PSAR":
        return exit_psar(A["high"], A["low"], A["close"], ei, end)
    sig = {"K70": A["k70"], "RSI2>70": A["rsi2_ob"], "close>SMA5": A["gt_sma5"],
           "close>SMA10": A["gt_sma10"], "BBmid": A["ge_sma20"], "BBupper": A["ge_bbup"],
           "RSI14>70": A["rsi14_ob"]}[rule]
    return exit_close_signal(sig, A["close"], ei, end)


def main():
    print("downloading daily (20y) ...", flush=True)
    data = fetch_batch(BASKET, "20y", "1d")
    results = {r: {"tr": [], "te": []} for r in RULES}

    for sym, df in data.items():
        if len(df) < 260:
            continue
        try:
            cs = compute_cycle_stoch(df)
        except Exception:
            continue
        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        sma5, sma10, sma20 = (close.rolling(5).mean(), close.rolling(10).mean(),
                              close.rolling(20).mean())
        std20 = close.rolling(20).std()
        sma200 = close.rolling(200).mean()
        rsi2 = rsi(close, 2)
        A = {
            "close": close.values, "high": high.values, "low": low.values,
            "k70": (cs["stoch_k"] >= 70).fillna(False).values,
            "rsi2_ob": (rsi2 >= 70).fillna(False).values,
            "gt_sma5": (close > sma5).fillna(False).values,
            "gt_sma10": (close > sma10).fillna(False).values,
            "ge_sma20": (close >= sma20).fillna(False).values,
            "ge_bbup": (close >= sma20 + 2 * std20).fillna(False).values,
            "rsi14_ob": (cs["rsi"] >= 70).fillna(False).values,
            "atr22": atr(high, low, close, 22).values,
            "atr10": atr(high, low, close, 10).values,
        }
        entry = ((close > sma200) & (cs["rsi"].values < 40) & (cs["stoch_k"].values < 20)).fillna(False).values
        dates = df.index
        n = len(close)
        for rule in RULES:
            i = 0
            while i < n - 1:
                if not entry[i]:
                    i += 1
                    continue
                ei = i
                end = min(n - 1, i + MAX_HOLD)
                xi, xp = resolve(rule, A, ei, end)
                ret = xp / A["close"][ei] - 1.0
                bucket = "tr" if dates[ei] < SPLIT else "te"
                results[rule][bucket].append((ret, xi - ei))
                i = xi + 1

    def stats(rows):
        if not rows:
            return None
        r = np.array([x[0] for x in rows])
        h = np.array([x[1] for x in rows])
        w, l = r[r > 0], r[r < 0]
        pf = w.sum() / abs(l.sum()) if l.sum() != 0 else float("inf")
        return dict(n=len(r), win=(r > 0).mean() * 100, avg=r.mean() * 100,
                    med=np.median(r) * 100, pf=pf, hold=h.mean())

    rows = []
    for rule in RULES:
        s_tr, s_te = stats(results[rule]["tr"]), stats(results[rule]["te"])
        if s_tr and s_te:
            rows.append((rule, s_tr, s_te))
    rows.sort(key=lambda x: (x[2]["pf"] if x[2]["pf"] != float("inf") else 99), reverse=True)

    print("\n========= EXIT STRATEGIES — OUT-OF-SAMPLE (test 2019-2026), by profit factor =========")
    print(f"{'exit':<14}{'N':>5}{'win%':>7}{'avg%':>8}{'med%':>8}{'PF':>7}{'hold':>7}"
          f"   | {'trPF':>6}{'trWin%':>8}")
    for rule, s_tr, s_te in rows:
        print(f"{rule:<14}{s_te['n']:>5}{s_te['win']:>7.1f}{s_te['avg']:>8.2f}"
              f"{s_te['med']:>8.2f}{s_te['pf']:>7.2f}{s_te['hold']:>7.1f}"
              f"   | {s_tr['pf']:>6.2f}{s_tr['win']:>8.1f}")
    print("\nAbsolute numbers are regime-influenced; rank/compare relative to K70 (champion).")


if __name__ == "__main__":
    main()
