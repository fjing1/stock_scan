"""Multi-timeframe (intraday) entry/exit refinement for the daily dip signal — #23.

Question: the dip-in-uptrend signal is daily; can a 15m/30m/60m intraday confirmation give a
BETTER entry, and can an intraday "sell into strength" exit beat the daily-close exit?

This EXTENDS #8 (15m entry confirm: weak, ~67 samples) and #9 (15m sell-bar exit: harmful),
using more data where possible: Yahoo gives ~60 days of 15/30m bars but ~2 YEARS of 60m bars,
so the 60m test has real power while 15/30m stay indicative.

ENTRY test: among daily dip signals (SMA50>SMA200 & RSI<40 & %K<20), split by whether the
NEXT session shows an intraday "strong up bar" (close>open, body>=1.5x avg, closes top 40% of
range, vol>=1.5x avg) — the dip_scan conf15 rule. Compare the daily +10d DETRENDED forward
return of CONFIRMED vs UNCONFIRMED signals (does waiting for the intraday turn help?).

EXIT: not re-tested here — #9 already showed a 15m intraday SELL-bar exit is HARMFUL (it sells
into weakness), and a proper intraday-exit backtest is blocked by the data wall (a ~2-week hold
needs intraday bars across it; 15/30m only reach back 60 days). Left as future work on 60m.

CAVEAT: intraday history is short (60d for 15/30m) so N is small and single-regime — treat as
indicative, not proof. Survivorship universe. This is an execution/timing overlay question, not
a new alpha. python mtf_intraday_test.py [--names N]
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
import yfinance as yf  # noqa: E402

BENCHMARK = "SPY"
H = 10                       # daily forward horizon (trading days)
TFS = [("15m", "60d"), ("30m", "60d"), ("60m", "730d")]
BODY_MULT, VOL_MULT, UP_FRAC = 1.5, 1.5, 0.6


def dl(sym, period, interval, threads=True):
    d = yf.download(sym, period=period, interval=interval, auto_adjust=False,
                    progress=False, group_by="ticker", threads=threads)
    return d


def strong_up(d):
    """Boolean Series: 'strong up bar' (the conf15 rule) on an OHLCV intraday frame."""
    o, h, l, c, v = d["Open"], d["High"], d["Low"], d["Close"], d["Volume"]
    body = c - o
    avg_body = body.abs().rolling(20).mean().shift(1)
    avg_vol = v.rolling(20).mean().shift(1)
    rng = (h - l).replace(0, np.nan)
    close_pos = (c - l) / rng
    return ((c > o) & (body >= BODY_MULT * avg_body) & (close_pos >= UP_FRAC)
            & (v >= VOL_MULT * avg_vol)).fillna(False)


def two_sample(a, b):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    b = np.asarray(b, float); b = b[~np.isnan(b)]
    if len(a) < 8 or len(b) < 8:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / len(a) + vb / len(b))
    return (a.mean() - b.mean()) / se if se > 0 else float("nan")


def desc(arr):
    a = np.asarray(arr, float); a = a[~np.isnan(a)]
    if len(a) == 0:
        return 0, float("nan"), float("nan")
    return len(a), (a > 0).mean() * 100, a.mean() * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", type=int, default=120)
    args = ap.parse_args()

    from stock_symbols_1243 import STOCK_SYMBOLS
    syms = [s for s in dict.fromkeys(STOCK_SYMBOLS) if s != BENCHMARK][:args.names]
    print(f"downloading {len(syms)} names daily (2y) + {BENCHMARK} ...", flush=True)
    dd = dl(syms, "2y", "1d")
    spy = dl(BENCHMARK, "2y", "1d")
    spy_c = (spy["Close"] if "Close" in spy else spy[BENCHMARK]["Close"]).astype(float)
    spy_c.index = spy_c.index.tz_localize(None) if spy_c.index.tz is not None else spy_c.index
    spy_fwd = spy_c.shift(-H) / spy_c - 1

    # daily signal dates per symbol (current system: SMA50>SMA200 & oversold)
    sig_dates = {}
    daily_close = {}
    for s in syms:
        try:
            d = dd[s].dropna(how="all").copy()
        except Exception:
            continue
        if len(d) < 220 or not {"High", "Low", "Close"}.issubset(d.columns):
            continue
        if d.index.tz is not None:
            d.index = d.index.tz_localize(None)
        try:
            cs = compute_cycle_stoch(d)
        except Exception:
            continue
        c = d["Close"].astype(float)
        regime = c.rolling(50).mean() > c.rolling(200).mean()
        oversold = (cs["rsi"] < 40) & (cs["stoch_k"] < 20)
        sig = (regime & oversold).fillna(False)
        sig_dates[s] = [t.normalize() for t in c.index[sig]]
        daily_close[s] = c
    print(f"daily signals found across {sum(1 for v in sig_dates.values() if v)} names "
          f"({sum(len(v) for v in sig_dates.values())} total over 2y)")

    print(f"\n{'tf':>5} | {'window':>7} | {'signals':>7} {'%conf':>6} | "
          f"{'CONF: n win% mean%':<22} | {'UNCONF: n win% mean%':<24} | {'Δmean':>6} {'t':>5}")
    for tf, period in TFS:
        conf_ret, unconf_ret = [], []
        for s, dates in sig_dates.items():
            if not dates:
                continue
            try:
                idf = dl(s, period, tf, threads=False)
                idf = idf[s] if isinstance(idf.columns, pd.MultiIndex) else idf
                idf = idf.dropna(how="all")
            except Exception:
                continue
            if idf is None or len(idf) < 30 or "Close" not in idf:
                continue
            if idf.index.tz is not None:
                idf.index = idf.index.tz_localize(None)
            su = strong_up(idf)
            sess = pd.Series(idf.index.normalize(), index=idf.index)
            sessions = np.array(sorted(pd.unique(sess.values)))
            c = daily_close[s]
            for D in dates:
                # actionable session = first intraday session strictly after the signal day
                later = sessions[sessions > np.datetime64(D)]
                if len(later) == 0:
                    continue
                s1 = later[0]
                # the confirming session must be the NEXT trading day after the signal — if the
                # gap is large, D predates the intraday window (60d for 15/30m) so skip it.
                if (s1 - np.datetime64(D)).astype("timedelta64[D]").astype(int) > 5:
                    continue
                confirmed = bool(su[sess.values == s1].any())
                # daily +H detrended forward return from the signal day's close
                if D not in c.index:
                    continue
                p = c.index.get_loc(D)
                if p + H >= len(c):
                    continue
                ret = c.iloc[p + H] / c.iloc[p] - 1
                sp = spy_fwd.reindex([c.index[p]]).iloc[0]
                det = ret - (sp if not np.isnan(sp) else 0.0)
                (conf_ret if confirmed else unconf_ret).append(det)
        nc, wc, mc = desc(conf_ret)
        nu, wu, mu = desc(unconf_ret)
        tot = nc + nu
        pconf = 100 * nc / tot if tot else float("nan")
        dt = two_sample(conf_ret, unconf_ret)
        print(f"{tf:>5} | {period:>7} | {tot:>7} {pconf:>6.0f} | "
              f"{nc:>4}{wc:>6.0f}{mc:>7.2f}        | {nu:>4}{wu:>7.0f}{mu:>8.2f}         | "
              f"{(mc-mu):>6.2f} {dt:>5.1f}")

    print("\nReads: Δmean = confirmed-minus-unconfirmed daily +10d detrended return. A POSITIVE")
    print("Δmean with |t|>~2 and a decent N would mean the intraday turn improves the daily entry.")
    print("Compare to #8 (15m: weak, ~67 samples). 60m has the most history => the most trustworthy")
    print("row; 15m/30m are ~60-day single-regime snapshots. If Δmean is small / t<2, intraday is a")
    print("LIVE timing/confirmation overlay (already in dip_scan as conf15), not a proven daily edge.")


if __name__ == "__main__":
    main()
