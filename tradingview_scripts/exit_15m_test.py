"""Does a 15m 'strong sell bar on volume' improve the dip-in-uptrend EXIT?

Compares exit rules on recent dip-in-uptrend trades (entry = daily close of the
signal day). 15m history is ~60 days, so this is one regime / modest N — read the
caveat. A 15m strong-sell bar = close<open, big body, closes near its low, on
above-average volume (mirror of the buy confirmation).

Exit variants:
  K70_daily : sell when daily Stoch %K >= 70            (current validated exit)
  sell15    : sell on the first 15m strong-sell bar
  combo     : whichever of K70_daily / sell15 comes first
  protect15 : sell on a 15m sell bar ONLY if in profit, else hold to K70_daily
All capped at 21 trading days.

    python exit_15m_test.py
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
LIQUID = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "ORCL",
    "CRM", "ADBE", "AMD", "INTC", "CSCO", "QCOM", "TXN", "MU", "AMAT", "NFLX",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP",
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY", "AMGN",
    "WMT", "HD", "COST", "PG", "KO", "PEP", "MCD", "NKE", "SBUX", "LOW", "TGT", "DIS",
    "CAT", "BA", "HON", "GE", "UPS", "RTX", "XOM", "CVX", "COP", "T", "VZ", "CMCSA",
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "SMH",
]
MAX_HOLD = 21
COOLDOWN = 3
BODY_MULT, VOL_MULT, UP_FRAC = 1.5, 1.5, 0.6


def fetch_batch(symbols, period, interval):
    raw = yf.download(symbols, period=period, interval=interval,
                      auto_adjust=False, progress=False, group_by="ticker", threads=True)
    out = {}
    for s in symbols:
        try:
            d = raw[s].dropna(how="all").copy() if len(symbols) > 1 else raw.copy()
        except Exception:
            continue
        if len(d):
            if getattr(d.index, "tz", None) is not None:
                d.index = d.index.tz_localize(None)
            out[s] = d
    return out


def sell15_series(df):
    o, h, l, c, v = (df["Open"], df["High"], df["Low"], df["Close"], df["Volume"])
    body = o - c
    avg_body = (c - o).abs().rolling(20).mean().shift(1)
    avg_vol = v.rolling(20).mean().shift(1)
    rng = (h - l).replace(0, np.nan)
    close_pos = (c - l) / rng
    sell = ((c < o) & (body >= BODY_MULT * avg_body)
            & (close_pos <= 1 - UP_FRAC) & (v >= VOL_MULT * avg_vol)).fillna(False)
    return sell


def first_sell_after(sell, close15, t0, t_cap):
    """First 15m strong-sell bar in (t0, t_cap]; return (time, price) or None."""
    mask = (sell.index > t0) & (sell.index <= t_cap) & sell.values
    idx = sell.index[mask]
    if len(idx) == 0:
        return None
    t = idx[0]
    return t, float(close15.loc[t])


def main():
    print("downloading daily + 15m ...", flush=True)
    daily = fetch_batch(LIQUID, "2y", "1d")
    intr = fetch_batch(LIQUID, "60d", "15m")
    sells = {s: (sell15_series(intr[s]), intr[s]["Close"]) for s in intr
             if {"Open", "High", "Low", "Close", "Volume"}.issubset(intr[s].columns)
             and len(intr[s]) > 25}

    variants = {"K70_daily": [], "sell15": [], "combo": [], "protect15": []}
    n_eval = 0
    for sym, df in daily.items():
        if sym == BENCHMARK or len(df) < 260 or sym not in sells:
            continue
        try:
            cs = compute_cycle_stoch(df)
        except Exception:
            continue
        close = df["Close"].astype(float)
        k = cs["stoch_k"]
        ma200 = close.rolling(200).mean()
        dip = ((close > ma200) & (cs["rsi"].values < 40) & (k.values < 20)).fillna(False)
        sell_s, c15 = sells[sym]
        cov_start, cov_end = sell_s.index[0], sell_s.index[-1]

        last = -(10 ** 9)
        for pos in np.where(dip.values)[0]:
            if pos - last < COOLDOWN:
                continue
            D = df.index[pos]
            entry_t = pd.Timestamp(D) + pd.Timedelta(hours=20)   # ~ daily close (UTC)
            # need entry + room for exit inside 15m coverage
            if entry_t < cov_start or entry_t > cov_end - pd.Timedelta(days=3):
                continue
            ep = float(close.iloc[pos])
            cap_pos = min(len(df) - 1, pos + MAX_HOLD)
            cap_t = pd.Timestamp(df.index[cap_pos]) + pd.Timedelta(hours=20)
            cap_t = min(cap_t, cov_end)

            # daily K70 exit
            kx = None
            for j in range(pos + 1, cap_pos + 1):
                if k.iloc[j] >= 70:
                    kx = (pd.Timestamp(df.index[j]) + pd.Timedelta(hours=20),
                          float(close.iloc[j]), j - pos)
                    break
            if kx is None:
                kx = (cap_t, float(close.iloc[cap_pos]), cap_pos - pos)

            # 15m sell exit
            sx = first_sell_after(sell_s, c15, entry_t, cap_t)
            if sx is None:
                hold15 = cap_pos - pos
                sx_ret, sx_t = float(close.iloc[cap_pos]) / ep - 1, cap_t
            else:
                sx_t, sx_px = sx
                sx_ret = sx_px / ep - 1
                hd = df.index[(df.index >= sx_t.normalize())]
                hold15 = (df.index.searchsorted(sx_t.normalize()) - pos) or 1

            last = pos
            n_eval += 1
            # record
            variants["K70_daily"].append((kx[1] / ep - 1, kx[2]))
            variants["sell15"].append((sx_ret, hold15))
            # combo: earliest timestamp
            if sx is not None and sx[0] <= kx[0]:
                variants["combo"].append((sx_ret, hold15))
            else:
                variants["combo"].append((kx[1] / ep - 1, kx[2]))
            # protect15: 15m sell only if in profit at that bar, else K70
            if sx is not None and sx[1] / ep - 1 > 0 and sx[0] <= kx[0]:
                variants["protect15"].append((sx_ret, hold15))
            else:
                variants["protect15"].append((kx[1] / ep - 1, kx[2]))

    print(f"\ntrades evaluated (recent 15m-covered window): {n_eval}\n")
    if n_eval == 0:
        print("no covered trades — intraday window too short this run.")
        return
    print(f"{'exit rule':<12}{'N':>5}{'win%':>7}{'avg%':>8}{'med%':>8}{'PF':>7}{'hold(d)':>9}")
    for name, rows in variants.items():
        r = np.array([x[0] for x in rows])
        holds = np.array([x[1] for x in rows])
        wins, losses = r[r > 0], r[r < 0]
        pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
        print(f"{name:<12}{len(r):>5}{(r>0).mean()*100:>7.1f}{r.mean()*100:>8.2f}"
              f"{np.median(r)*100:>8.2f}{pf:>7.2f}{holds.mean():>9.1f}")

    print("\n  CAVEAT: ~60-day 15m window, single regime, small N, overlapping "
          "windows.\n  Compare relative ranking, not absolute numbers.")


if __name__ == "__main__":
    main()
