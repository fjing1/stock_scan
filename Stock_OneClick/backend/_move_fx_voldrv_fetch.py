"""
_move_fx_voldrv_fetch.py -- one-off fetch + CACHE of the volatility-derivatives complex.

WHAT IS FETCHED (source: Yahoo Finance via yfinance, period=max, auto_adjust=False, Close column):
    ^VIX9D   CBOE 9-day  volatility index
    ^VIX     CBOE 30-day volatility index          (cross-check vs the panel's own ^VIX column)
    ^VIX3M   CBOE 3-month volatility index         (formerly ^VXV)
    ^VIX6M   CBOE 6-month volatility index
    ^VVIX    CBOE VIX-of-VIX  (vol of vol)
    ^VXN     CBOE Nasdaq-100 volatility index      (robustness only)
    ^RVX     CBOE Russell-2000 volatility index    (robustness only)

Written to _move_fx_voldrv.pkl as a single DataFrame (dates x tickers) of raw index LEVELS.
NO forward-fill, NO interpolation: holes are left as NaN on purpose -- ^VIX3M in particular has
real multi-week holes in the Yahoo feed and stale-filling them would silently fabricate a
"3-month implied vol" that never existed.

Run once:  ../../vcp_env/bin/python _move_fx_voldrv_fetch.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE = Path(__file__).with_name("_move_fx_voldrv.pkl")
TICKERS = ["^VIX9D", "^VIX", "^VIX3M", "^VIX6M", "^VVIX", "^VXN", "^RVX"]


def build() -> pd.DataFrame:
    cols = {}
    for t in TICKERS:
        d = yf.download(t, period="max", progress=False, auto_adjust=False)
        if d is None or len(d) == 0:
            print(f"  {t}: EMPTY")
            continue
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        s = pd.to_numeric(d["Close"], errors="coerce")
        s = s[s > 0]
        cols[t] = s
        print(f"  {t}: n={s.notna().sum():>6,}  {s.index[0].date()} -> {s.index[-1].date()}  "
              f"mean={s.mean():.2f}")
    df = pd.concat(cols, axis=1)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    pd.to_pickle(df, CACHE)
    return df


def load() -> pd.DataFrame:
    if not CACHE.exists():
        return build()
    return pd.read_pickle(CACHE)


if __name__ == "__main__":
    df = build()
    print(f"\nsaved {CACHE.name}: {df.shape[0]} rows x {df.shape[1]} cols")
    print(df.tail(8).to_string(float_format=lambda v: f"{v:7.2f}"))
