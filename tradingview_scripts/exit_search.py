"""Exit-rule search for the locked dip-in-uptrend entry.

Entry (fixed): Close > SMA200  AND  RSI(14) < 40  AND  Stoch %K(10,EMA4) < 20.

Simulates real trades (one position at a time per symbol) for a library of exit
rules, splits trades by entry date (train < 2019 <= test), and ranks by
OUT-OF-SAMPLE performance. For exits, win rate alone is misleading (a tight
profit-target gives a high win rate but tiny winners and fat losers), so the
headline metric is PROFIT FACTOR and EXPECTANCY; win rate, avg hold, and excess
vs SPY are reported alongside.

Exit families tested:
  time       : exit after N bars
  osc target : exit when Stoch %K / RSI / Cycle crosses up through a level
  MA break   : exit when Close drops below SMA20 / SMA50 / SMA200
  %  TP/SL   : fixed percent target / stop (intrabar, stop checked first)
  ATR TP/SL  : ATR-multiple target / stop
  combo      : osc-target OR max-hold OR % stop  (whichever comes first)
Every rule has a max-hold cap so positions always resolve.

    python exit_search.py
    python exit_search.py --split 2019-01-01
"""
from __future__ import annotations

import argparse
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


def fetch_batch(symbols, interval, period):
    raw = yf.download(symbols, interval=interval, period=period,
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


def _atr(high, low, close, length=14):
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


# Exit-rule templates. `sig` builds an entry-independent close-exit boolean.
STRATS = [
    {"name": "time_3",            "maxhold": 3},
    {"name": "time_5",            "maxhold": 5},
    {"name": "time_8",            "maxhold": 8},
    {"name": "time_10",           "maxhold": 10},
    {"name": "time_15",           "maxhold": 15},
    {"name": "time_21",           "maxhold": 21},
    {"name": "K>=50",  "sig": lambda d: d["stochK"] >= 50, "maxhold": 30},
    {"name": "K>=70",  "sig": lambda d: d["stochK"] >= 70, "maxhold": 30},
    {"name": "K>=80",  "sig": lambda d: d["stochK"] >= 80, "maxhold": 30},
    {"name": "RSI>=50", "sig": lambda d: d["rsi"] >= 50,   "maxhold": 30},
    {"name": "RSI>=60", "sig": lambda d: d["rsi"] >= 60,   "maxhold": 30},
    {"name": "RSI>=70", "sig": lambda d: d["rsi"] >= 70,   "maxhold": 30},
    {"name": "Cyc>=50", "sig": lambda d: d["cycle"] >= 50, "maxhold": 30},
    {"name": "Cyc>=80", "sig": lambda d: d["cycle"] >= 80, "maxhold": 30},
    {"name": "close<MA20",  "sig": lambda d: d["close"] < d["ma20"],  "maxhold": 60},
    {"name": "close<MA50",  "sig": lambda d: d["close"] < d["ma50"],  "maxhold": 60},
    {"name": "close<MA200", "sig": lambda d: d["close"] < d["ma200"], "maxhold": 120},
    {"name": "TP3/SL3",  "tp": 0.03, "sl": 0.03, "maxhold": 21},
    {"name": "TP5/SL5",  "tp": 0.05, "sl": 0.05, "maxhold": 21},
    {"name": "TP5/SL3",  "tp": 0.05, "sl": 0.03, "maxhold": 21},
    {"name": "TP8/SL5",  "tp": 0.08, "sl": 0.05, "maxhold": 21},
    {"name": "TP5/SL8",  "tp": 0.05, "sl": 0.08, "maxhold": 21},
    {"name": "TP10/SL5", "tp": 0.10, "sl": 0.05, "maxhold": 30},
    {"name": "ATR tp2/sl2", "tp_atr": 2, "sl_atr": 2, "maxhold": 21},
    {"name": "ATR tp3/sl2", "tp_atr": 3, "sl_atr": 2, "maxhold": 21},
    {"name": "ATR tp2/sl3", "tp_atr": 2, "sl_atr": 3, "maxhold": 30},
    {"name": "K>=70|cap10|SL5",  "sig": lambda d: d["stochK"] >= 70, "sl": 0.05, "maxhold": 10},
    {"name": "K>=70|cap15|SL8",  "sig": lambda d: d["stochK"] >= 70, "sl": 0.08, "maxhold": 15},
    {"name": "RSI>=60|cap10|SL5", "sig": lambda d: d["rsi"] >= 60,   "sl": 0.05, "maxhold": 10},
    {"name": "K>=80|cap21|SL5",  "sig": lambda d: d["stochK"] >= 80, "sl": 0.05, "maxhold": 21},
]


def simulate(close, high, low, atr, spy, entry, sig, tp, sl, tp_atr, sl_atr, maxhold):
    """One position at a time; return list of (entry_i, exit_i, ret, excess)."""
    n = len(close)
    trades = []
    i = 0
    while i < n - 1:
        if not entry[i]:
            i += 1
            continue
        ep = close[i]
        a = atr[i]
        stop_p = ep * (1 - sl) if sl else (ep - sl_atr * a if sl_atr else None)
        targ_p = ep * (1 + tp) if tp else (ep + tp_atr * a if tp_atr else None)
        ex_i, ex_p = None, None
        end = min(n - 1, i + maxhold)
        for j in range(i + 1, end + 1):
            if stop_p is not None and low[j] <= stop_p:      # stop checked first
                ex_i, ex_p = j, stop_p
                break
            if targ_p is not None and high[j] >= targ_p:
                ex_i, ex_p = j, targ_p
                break
            if sig is not None and sig[j]:
                ex_i, ex_p = j, close[j]
                break
        if ex_i is None:
            ex_i, ex_p = end, close[end]
        ret = ex_p / ep - 1.0
        sret = (spy[ex_i] / spy[i] - 1.0) if (not np.isnan(spy[i]) and not np.isnan(spy[ex_i])) else np.nan
        trades.append((i, ex_i, ret, ret - sret))
        i = ex_i + 1
    return trades


def metrics(rets, excess, holds):
    r = np.array(rets)
    if len(r) == 0:
        return None
    wins = r[r > 0]
    losses = r[r < 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    exc = np.array(excess)
    exc = exc[~np.isnan(exc)]
    return {
        "n": len(r),
        "win%": (r > 0).mean() * 100,
        "avg%": r.mean() * 100,
        "med%": np.median(r) * 100,
        "pf": pf,
        "expR%": r.mean() * 100,                       # expectancy per trade
        "avgHold": float(np.mean(holds)),
        "exc_win%": (exc > 0).mean() * 100 if len(exc) else float("nan"),
        "perBar%": (r.mean() / max(np.mean(holds), 1e-9)) * 100,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="2019-01-01")
    args = ap.parse_args(argv)
    split = pd.Timestamp(args.split)

    print(f"downloading {len(BASKET)} symbols (1d, 20y) ...", flush=True)
    data = fetch_batch(BASKET, "1d", "20y")
    bdf = data.get(BENCHMARK)
    spy_close = bdf["Close"].astype(float) if bdf is not None else None

    # precompute per-symbol arrays + entry
    sym_arr = {}
    for sym, df in data.items():
        if sym == BENCHMARK or len(df) < 260:
            continue
        try:
            cs = compute_cycle_stoch(df)
        except Exception:
            continue
        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        d = pd.DataFrame(index=df.index)
        d["close"], d["high"], d["low"] = close, high, low
        d["ma20"] = close.rolling(20).mean()
        d["ma50"] = close.rolling(50).mean()
        d["ma200"] = close.rolling(200).mean()
        d["atr"] = _atr(high, low, close, 14)
        d["rsi"] = cs["rsi"].values
        d["stochK"] = cs["stoch_k"].values
        d["cycle"] = cs["cycle"].values
        d["spy"] = (spy_close.reindex(df.index).values if spy_close is not None
                    else np.nan)
        entry = ((close > d["ma200"]) & (d["rsi"] < 40) & (d["stochK"] < 20)).fillna(False)
        d["entry"] = entry.values
        d["date"] = df.index
        sym_arr[sym] = d

    print(f"symbols: {len(sym_arr)}   total entry bars: "
          f"{sum(int(d['entry'].sum()) for d in sym_arr.values())}")

    # evaluate each strategy
    results = []
    for st in STRATS:
        tr_r, tr_e, tr_h, te_r, te_e, te_h = [], [], [], [], [], []
        for sym, d in sym_arr.items():
            sig = st["sig"](d).fillna(False).values if "sig" in st else None
            trades = simulate(
                d["close"].values, d["high"].values, d["low"].values,
                d["atr"].values, d["spy"].values, d["entry"].values,
                sig, st.get("tp"), st.get("sl"), st.get("tp_atr"),
                st.get("sl_atr"), st["maxhold"])
            dates = d["date"].values
            for (ei, xi, ret, exc) in trades:
                if dates[ei] < np.datetime64(split):
                    tr_r.append(ret); tr_e.append(exc); tr_h.append(xi - ei)
                else:
                    te_r.append(ret); te_e.append(exc); te_h.append(xi - ei)
        mtr = metrics(tr_r, tr_e, tr_h)
        mte = metrics(te_r, te_e, te_h)
        if mtr and mte:
            results.append((st["name"], mtr, mte))

    # leaderboard by TEST profit factor
    def line(name, m):
        return (f"{name:<18}{m['n']:>5}{m['win%']:>7.1f}{m['avg%']:>8.2f}"
                f"{m['med%']:>8.2f}{m['pf']:>7.2f}{m['avgHold']:>8.1f}"
                f"{m['perBar%']:>8.3f}{m['exc_win%']:>9.1f}")

    hdr = (f"{'exit rule':<18}{'N':>5}{'win%':>7}{'avg%':>8}{'med%':>8}"
           f"{'PF':>7}{'hold':>8}{'/bar%':>8}{'excWin%':>9}")
    results.sort(key=lambda x: (x[2]["pf"] if x[2]["pf"] != float("inf") else 99), reverse=True)

    print("\n================ OUT-OF-SAMPLE (test 2019-2026), ranked by profit factor ================")
    print(hdr)
    for name, mtr, mte in results:
        print(line(name, mte))

    print("\n================ IN-SAMPLE (train pre-2019), same order ================")
    print(hdr)
    for name, mtr, mte in results:
        print(line(name, mtr))

    # robust pick: good on BOTH, ranked by min(train PF, test PF)
    print("\n================ ROBUST (ranked by min(train PF, test PF)) ================")
    rob = sorted(results, key=lambda x: min(
        x[1]["pf"] if x[1]["pf"] != float("inf") else 99,
        x[2]["pf"] if x[2]["pf"] != float("inf") else 99), reverse=True)
    print(f"{'exit rule':<18}{'trPF':>7}{'tePF':>7}{'trWin%':>8}{'teWin%':>8}"
          f"{'teAvg%':>8}{'teHold':>8}{'te/bar%':>9}")
    for name, mtr, mte in rob[:10]:
        print(f"{name:<18}{mtr['pf']:>7.2f}{mte['pf']:>7.2f}{mtr['win%']:>8.1f}"
              f"{mte['win%']:>8.1f}{mte['avg%']:>8.2f}{mte['avgHold']:>8.1f}"
              f"{mte['perBar%']:>9.3f}")


if __name__ == "__main__":
    main()
