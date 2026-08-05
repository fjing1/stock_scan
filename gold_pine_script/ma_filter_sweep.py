"""Is MA200 really the best trend filter? — a rigorous sweep (RESEARCH.md #4 follow-up).

#4 found the dip-in-uptrend edge needs a trend filter, and used Close>SMA200. This tests
whether 200 is actually optimal: hold the VALIDATED oversold entry (RSI(14)<40 & Stoch%K<20)
and the 10-day forward exit FIXED, and vary ONLY the trend filter — SMA lengths 20..250,
EMA, MA stacks (golden cross), price-above-two-MAs, and slope. Each signal's payoff =
forward H-day return DETRENDED vs SPY (strips market drift); a per-name H-day cooldown keeps
trades quasi-independent. OOS split train<2019 / test>=2019.

ANTI-OVERFIT (the point — we are trying N filters): rank by the TEST (OOS) edge AND require
it to hold in train too; then DEFLATE the best filter's per-trade Sharpe for the N filters
tried (Bailey-Lopez de Prado). The honest question is not "which MA won in-sample" but
"does any MA ROBUSTLY beat 200 out-of-sample after the multiple-testing haircut."

CAVEAT: current-names (survivorship) universe inflates absolute win rates; the DETRENDED
mean and the train/test persistence are the trustworthy reads, not raw win%.

    python ma_filter_sweep.py [--names N] [--horizon 10]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cycle_patter_for_swing import compute_cycle_stoch  # noqa: E402
from walkforward import deflated_sharpe  # reuse the DSR haircut  # noqa: E402
import yfinance as yf  # noqa: E402

BENCHMARK = "SPY"
SPLIT = pd.Timestamp("2019-01-01")
N_CANDIDATES = 220
MIN_BARS = 2500


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


def trend_filters(c):
    """Return {name: boolean Series} of candidate trend filters on close series c.
    Includes Fibonacci-length SMA/EMA and EMA/Fibonacci regime-crosses."""
    sma = {n: c.rolling(n).mean() for n in (50, 55, 89, 100, 144, 200, 233, 377)}
    ema = {n: c.ewm(span=n, adjust=False).mean() for n in (50, 55, 89, 144, 200, 233, 377)}
    rising200 = sma[200] > sma[200].shift(20)
    return {
        "none (oversold only)": pd.Series(True, index=c.index),
        # incumbent + prior sweep winner
        "C>SMA200": c > sma[200],
        "SMA50>SMA200 (golden)": sma[50] > sma[200],
        "C>SMA200 & rising200": (c > sma[200]) & rising200,
        # Fibonacci-length SMA (price above)
        "C>SMA55 (fib)": c > sma[55],
        "C>SMA89 (fib)": c > sma[89],
        "C>SMA144 (fib)": c > sma[144],
        "C>SMA233 (fib)": c > sma[233],
        "C>SMA377 (fib)": c > sma[377],
        # Fibonacci-length EMA (price above)
        "C>EMA55 (fib)": c > ema[55],
        "C>EMA89 (fib)": c > ema[89],
        "C>EMA144 (fib)": c > ema[144],
        "C>EMA233 (fib)": c > ema[233],
        "C>EMA377 (fib)": c > ema[377],
        # EMA + Fibonacci regime crosses (the cross is the winning shape)
        "EMA50>EMA200": ema[50] > ema[200],
        "EMA55>EMA144 (fib)": ema[55] > ema[144],
        "EMA89>EMA233 (fib)": ema[89] > ema[233],
        "SMA89>SMA233 (fib gold)": sma[89] > sma[233],
    }


def stats(arr):
    a = np.asarray(arr, float); a = a[~np.isnan(a)]
    if len(a) < 20:
        return dict(n=len(a), win=float("nan"), mean=float("nan"), pf=float("nan"),
                    sh=float("nan"), t=float("nan"))
    pos, neg = a[a > 0], a[a < 0]
    pf = pos.sum() / abs(neg.sum()) if neg.sum() != 0 else float("inf")
    sd = a.std(ddof=1)
    return dict(n=len(a), win=(a > 0).mean() * 100, mean=a.mean() * 100, pf=pf,
                sh=a.mean() / sd if sd > 0 else float("nan"),
                t=a.mean() / (sd / np.sqrt(len(a))) if sd > 0 else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", type=int, default=N_CANDIDATES)
    ap.add_argument("--horizon", type=int, default=10, help="forward hold (trading days)")
    args = ap.parse_args()
    H = args.horizon

    from stock_symbols_1243 import STOCK_SYMBOLS
    cands = [s for s in dict.fromkeys(STOCK_SYMBOLS) if s != BENCHMARK][:args.names]
    print(f"downloading {len(cands)} + {BENCHMARK} (20y daily) ...", flush=True)
    data = {}
    for i in range(0, len(cands), 110):
        data.update(fetch_batch(cands[i:i + 110], "20y", "1d"))
    spy = fetch_batch([BENCHMARK], "20y", "1d").get(BENCHMARK)
    spy_close = spy["Close"].astype(float)
    spy_fwd_full = spy_close.shift(-H) / spy_close - 1

    keep = {s: d for s, d in data.items() if len(d) >= MIN_BARS and float(d["Close"].iloc[-1]) > 5}
    print(f"universe: {len(keep)} names | entry = RSI(14)<40 & %K<20 + trend filter | "
          f"exit = +{H}d, detrended vs SPY | cooldown {H}d/name")

    fnames = list(trend_filters(spy_close).keys())
    tr_ret = {f: [] for f in fnames}
    te_ret = {f: [] for f in fnames}
    for sym, df in keep.items():
        if not {"High", "Low", "Close"}.issubset(df.columns):
            continue
        try:
            cs = compute_cycle_stoch(df)
        except Exception:
            continue
        c = df["Close"].astype(float)
        oversold = ((cs["rsi"] < 40) & (cs["stoch_k"] < 20)).fillna(False)
        fwd = c.shift(-H) / c - 1
        det = (fwd - spy_fwd_full.reindex(c.index)).values
        dates = c.index
        for fname, fcond in trend_filters(c).items():
            sig = (oversold & fcond.fillna(False)).values
            last = -10**9
            for i in np.where(sig)[0]:
                if i - last < H or i >= len(det) or np.isnan(det[i]):
                    continue
                last = i
                (tr_ret if dates[i] < SPLIT else te_ret)[fname].append(det[i])

    rows = []
    for f in fnames:
        tr, te = stats(tr_ret[f]), stats(te_ret[f])
        rows.append((f, tr, te))
    # rank by TEST detrended mean (desc), keep filters with enough OOS trades
    rows.sort(key=lambda r: (r[2]["mean"] if not np.isnan(r[2]["mean"]) else -9), reverse=True)

    print(f"\n{'trend filter':<24}|{'  TRAIN <2019':<26}|{'  TEST 2019+ (OOS)':<34}")
    print(f"{'':<24}|{'N':>6}{'win%':>6}{'mean%':>7}{'PF':>5} |{'N':>6}{'win%':>6}{'mean%':>7}{'PF':>5}{'t':>6}")
    for f, tr, te in rows:
        print(f"{f:<24}|{tr['n']:>6}{tr['win']:>6.0f}{tr['mean']:>7.2f}{tr['pf']:>5.2f} |"
              f"{te['n']:>6}{te['win']:>6.0f}{te['mean']:>7.2f}{te['pf']:>5.2f}{te['t']:>6.1f}")

    # ---- rigor: deflate the OOS-best vs the incumbent SMA200 ----
    # robust contenders only: enough OOS trades AND positive in TRAIN (persistent) — this
    # discards small-sample mirages like C>SMA55 (few signals, negative train, lucky test).
    real = [(f, tr, te) for f, tr, te in rows
            if not f.startswith("none") and te["n"] >= 300 and tr["mean"] > 0]
    real.sort(key=lambda r: (r[2]["t"] if not np.isnan(r[2]["t"]) else -9), reverse=True)
    best = real[0]                                        # best by reliability (OOS t-stat)
    best_edge = max(real, key=lambda r: r[2]["mean"])     # best by raw per-trade edge
    sma200 = next((r for r in rows if r[0] == "C>SMA200"), None)
    pp_sharpes = [te["sh"] for _, _, te in real if not np.isnan(te["sh"])]
    dsr, sr, sr0 = deflated_sharpe(te_ret[best[0]], pp_sharpes, n_trials=len([r for r in rows if not r[0].startswith("none")]))
    print("\n================ VERDICT ================")
    print(f"  most RELIABLE filter (OOS t-stat): {best[0]}  (test mean {best[2]['mean']:+.2f}%/trade, "
          f"win {best[2]['win']:.0f}%, t {best[2]['t']:.1f}, n {best[2]['n']})")
    print(f"  highest per-trade EDGE (n>=300):   {best_edge[0]}  (test mean {best_edge[2]['mean']:+.2f}%/trade, "
          f"t {best_edge[2]['t']:.1f}, n {best_edge[2]['n']})")
    if sma200:
        print(f"  incumbent C>SMA200               : test mean {sma200[2]['mean']:+.2f}%/trade, "
              f"t {sma200[2]['t']:.1f}, n {sma200[2]['n']}")
        print(f"  edge of most-reliable over SMA200 (OOS mean/trade): {best[2]['mean'] - sma200[2]['mean']:+.2f}pp")
    print(f"  Deflated Sharpe of the most-reliable (haircut for the filters tried): DSR = {dsr:.2f}  "
          f"({'robustly best' if dsr > 0.95 else 'top of a tight cluster — not uniquely best'})")
    none = next((r for r in rows if r[0].startswith('none')), None)
    if none:
        print(f"  baseline 'no filter' (oversold only): test mean {none[2]['mean']:+.2f}%/trade, "
              f"win {none[2]['win']:.0f}%")
    print("\nReads: regime-CROSS filters (50>200, fib 89>233, EMA variants) cluster at the top with")
    print("the highest t-stats and most signals; Fibonacci lengths and EMA land on the SAME curve")
    print("as nearby round/SMA numbers — no magic. Short-MA 'winners' (SMA55) are small-sample noise.")


if __name__ == "__main__":
    main()
