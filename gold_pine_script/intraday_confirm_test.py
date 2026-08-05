"""Does an intraday "strong up bar on volume" confirmation improve the daily
dip-in-uptrend win rate, and which timeframe is best?

Tested timeframes: 1m, 5m, 15m, 30m (+2m). NOTE Yahoo limits 1m to ~7 days and
2m/5m/15m/30m to ~60 days, so this can only judge entries from the recent window
== one market regime, modest sample. Read the N and the caveat at the end.

For each daily dip-in-uptrend entry (Close>SMA200 & RSI<40 & %K<20) in the
covered window, on the signal day D we check whether a strong-up-volume bar fired
at timeframe T (same adaptive definition as the scanner). Entry = daily close of
D; forward 5-day return (abs and excess vs SPY). We compare win rate of confirmed
vs unconfirmed vs the unconditional baseline, per timeframe.

    python intraday_confirm_test.py
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
TIMEFRAMES = [("1m", "7d"), ("2m", "60d"), ("5m", "60d"), ("15m", "60d"), ("30m", "60d")]
HORIZON = 5            # forward trading days
COOLDOWN = 3          # min days between entries per symbol
BODY_MULT, VOL_MULT, UP_FRAC = 1.5, 1.5, 0.6


def fetch_batch(symbols, period, interval):
    raw = yf.download(symbols, period=period, interval=interval,
                      auto_adjust=False, progress=False, group_by="ticker",
                      threads=True)
    out = {}
    for s in symbols:
        try:
            d = raw[s].dropna(how="all").copy() if len(symbols) > 1 else raw.copy()
        except Exception:
            continue
        if len(d) == 0:
            continue
        if getattr(d.index, "tz", None) is not None:
            d.index = d.index.tz_localize(None)
        out[s] = d
    return out


def strong_and_coverage(df):
    """Return (set of dates with a strong-up-vol bar, set of all covered dates)."""
    o, h, l, c, v = (df["Open"], df["High"], df["Low"], df["Close"], df["Volume"])
    body = c - o
    avg_body = body.abs().rolling(20).mean().shift(1)
    avg_vol = v.rolling(20).mean().shift(1)
    rng = (h - l).replace(0, np.nan)
    close_pos = (c - l) / rng
    strong = ((c > o) & (body >= BODY_MULT * avg_body)
              & (close_pos >= UP_FRAC) & (v >= VOL_MULT * avg_vol)).fillna(False)
    sdates = set(pd.Series(df.index[strong.values]).dt.date)
    cdates = set(pd.Series(df.index).dt.date)
    return sdates, cdates


def main():
    print("downloading daily ...", flush=True)
    daily = fetch_batch(LIQUID, "2y", "1d")
    bdf = daily.get(BENCHMARK)
    spy = bdf["Close"].astype(float) if bdf is not None else None
    spy_fwd = spy.shift(-HORIZON) / spy - 1.0 if spy is not None else None

    # collect recent dip entries (deduped) per symbol
    entries = []   # (sym, date, abs_fwd, exc_fwd)
    for sym, df in daily.items():
        if sym == BENCHMARK or len(df) < 260:
            continue
        try:
            cs = compute_cycle_stoch(df)
        except Exception:
            continue
        close = df["Close"].astype(float)
        ma200 = close.rolling(200).mean()
        dip = ((close > ma200) & (cs["rsi"].values < 40) & (cs["stoch_k"].values < 20)).fillna(False)
        abs_fwd = close.shift(-HORIZON) / close - 1.0
        exc_fwd = abs_fwd - (spy_fwd.reindex(df.index) if spy_fwd is not None else 0)
        last_pos = -(10 ** 9)
        for pos in np.where(dip.values)[0]:
            if pos - last_pos < COOLDOWN:
                continue
            last_pos = pos
            d = df.index[pos]
            if np.isnan(abs_fwd.iloc[pos]):
                continue
            entries.append((sym, d.date(), float(abs_fwd.iloc[pos]), float(exc_fwd.iloc[pos])))

    if not entries:
        print("no dip entries in the daily window.")
        return
    ent_df = pd.DataFrame(entries, columns=["sym", "date", "abs", "exc"])
    ent_syms = sorted(ent_df["sym"].unique())
    print(f"dip entries (2y, deduped): {len(ent_df)} across {len(ent_syms)} symbols")

    base_win = (ent_df["abs"] > 0).mean() * 100
    base_exc = (ent_df["exc"] > 0).mean() * 100
    print(f"baseline {HORIZON}d win%: abs {base_win:.1f}  excess-vs-SPY {base_exc:.1f}\n")

    print(f"{'TF':>4}{'covEntries':>11}{'conf':>6}{'cWin%':>7}{'cExc%':>7}"
          f"{'cAvg%':>7}{'unconf':>8}{'uWin%':>7}{'lift(c-u)':>10}")
    summary = []
    for tf, per in TIMEFRAMES:
        intr = fetch_batch(ent_syms, per, tf)
        sd_cd = {s: strong_and_coverage(intr[s]) for s in intr
                 if {"Open", "High", "Low", "Close", "Volume"}.issubset(intr[s].columns)
                 and len(intr[s]) > 25}
        conf_r, conf_e, unconf_r, unconf_e = [], [], [], []
        covered = 0
        for _, row in ent_df.iterrows():
            sc = sd_cd.get(row["sym"])
            if sc is None:
                continue
            sdates, cdates = sc
            if row["date"] not in cdates:        # no intraday coverage for that day
                continue
            covered += 1
            if row["date"] in sdates:
                conf_r.append(row["abs"]); conf_e.append(row["exc"])
            else:
                unconf_r.append(row["abs"]); unconf_e.append(row["exc"])

        def wr(a):
            return (np.mean(np.array(a) > 0) * 100) if a else float("nan")
        cwin, cexc, cavg = wr(conf_r), wr(conf_e), (np.mean(conf_r) * 100 if conf_r else float("nan"))
        uwin = wr(unconf_r)
        lift = cwin - uwin if (conf_r and unconf_r) else float("nan")
        print(f"{tf:>4}{covered:>11}{len(conf_r):>6}{cwin:>7.1f}{cexc:>7.1f}"
              f"{cavg:>7.2f}{len(unconf_r):>8}{uwin:>7.1f}{lift:>10.1f}")
        summary.append((tf, len(conf_r), cwin, uwin, lift, base_win))

    print("\nVERDICT:")
    valid = [s for s in summary if s[1] >= 20]   # need >=20 confirmed for any read
    if not valid:
        print("  Too few confirmed entries at every timeframe (intraday history is"
              f"\n  only ~60 days) — NO reliable read. Largest confirmed N = "
              f"{max((s[1] for s in summary), default=0)}.")
        return
    best = max(valid, key=lambda s: s[2])
    print(f"  Highest confirmed win%: {best[0]} = {best[2]:.1f}% "
          f"(N={best[1]}) vs unconfirmed {best[3]:.1f}% / baseline {best[5]:.1f}%.")
    helps = [s for s in valid if not np.isnan(s[4]) and s[4] >= 5]
    if helps:
        print("  Confirmation HELPS (>=5pp over unconfirmed) at: "
              + ", ".join(f"{s[0]} (+{s[4]:.1f})" for s in helps))
    else:
        print("  No timeframe's confirmation beats 'no confirmation' by >=5pp — "
              "i.e. NO clear edge from the intraday filter in this window.")
    print(f"\n  CAVEAT: ~60-day (1m: ~7-day) sample, single regime, overlapping "
          "forward\n  windows across names — treat as indicative, not proven.")


if __name__ == "__main__":
    main()
