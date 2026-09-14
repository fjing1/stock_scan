"""
_vix_data.py — one cached panel for all VIX-regime studies, so every analysis in the
_vix_* family reads identical bars instead of each re-downloading (and silently disagreeing).

Mirrors _btc_data.py / _gold_panel.pkl: build once, pickle, reuse.

Columns: vix, vix3m, vvix, spx, spy, qqq  (daily closes, outer-joined on the VIX calendar,
NaN where a series does not reach back that far — ^VIX 1990, ^GSPC 1927, SPY 1993, QQQ 1999,
^VIX3M 2006, ^VVIX 2007).

Partial-bar guard: if the last row is today and the US cash session has not closed, it is
dropped. Unclosed bars have silently corrupted three prior analyses in this repo.

Run directly to (re)build:  ../../vcp_env/bin/python _vix_data.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

PANEL = Path(__file__).with_name("_vix_panel.pkl")
TICKERS = {"vix": "^VIX", "vix3m": "^VIX3M", "vvix": "^VVIX",
           "spx": "^GSPC", "spy": "SPY", "qqq": "QQQ"}


def _close(ticker: str) -> pd.Series:
    d = yf.download(ticker, period="max", progress=False, auto_adjust=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return pd.to_numeric(d["Close"], errors="coerce").dropna()


def build() -> pd.DataFrame:
    cols = {k: _close(t) for k, t in TICKERS.items()}
    df = pd.DataFrame(cols)
    df.index = pd.to_datetime(df.index)
    df = df[df.vix.notna()]                      # VIX defines the calendar

    now_et = pd.Timestamp.now(tz="America/New_York")
    if df.index[-1].date() == now_et.date() and now_et.time() < pd.Timestamp("16:00").time():
        print(f"⚠️  dropping in-progress bar {df.index[-1].date()} (US session still open)")
        df = df.iloc[:-1]

    df.to_pickle(PANEL)
    return df


def load(rebuild: bool = False) -> pd.DataFrame:
    if rebuild or not PANEL.exists():
        return build()
    return pd.read_pickle(PANEL)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Shared feature set so every _vix_* study computes MA10 / bands the same way."""
    d = df.copy()
    d["ma10"] = d.vix.rolling(10).mean()
    d["stretch"] = d.vix / d.ma10 - 1.0
    for n in (10, 20):
        basis = d.vix.rolling(n).mean()
        sd = d.vix.rolling(n).std(ddof=0)
        for k in (1.5, 2.0, 2.5):
            t = f"bb{n}_{k}"
            d[f"{t}_up"], d[f"{t}_lo"] = basis + k * sd, basis - k * sd
            d[f"{t}_above"] = d.vix > d[f"{t}_up"]
            d[f"{t}_below"] = d.vix < d[f"{t}_lo"]
            d[f"{t}_reentry"] = d[f"{t}_above"].shift(1).fillna(False).astype(bool) & ~d[f"{t}_above"]
            d[f"{t}_exit_lo"] = d[f"{t}_below"].shift(1).fillna(False).astype(bool) & ~d[f"{t}_below"]
        d[f"bb{n}_pctb"] = (d.vix - (basis - 2 * sd)) / (4 * sd)
        d[f"bb{n}_width"] = (4 * sd) / basis
    # contextual controls used to test whether the BANDS add anything over the plain LEVEL
    d["vix_pct1y"] = d.vix.rolling(252).rank(pct=True)
    d["vix_pct2y"] = d.vix.rolling(504).rank(pct=True)
    d["vix_z1y"] = (d.vix - d.vix.rolling(252).mean()) / d.vix.rolling(252).std(ddof=0)
    d["term"] = d.vix / d.vix3m
    # forward index returns; g = entry at the NEXT close (VIX settles 16:15 ET, after the cash close)
    for h in (1, 3, 5, 10, 21):
        d[f"f{h}"] = d.spx.shift(-h) / d.spx - 1.0
        d[f"g{h}"] = d.spx.shift(-(h + 1)) / d.spx.shift(-1) - 1.0
    return d


if __name__ == "__main__":
    p = build()
    print(f"panel {PANEL.name}: {len(p):,} rows  {p.index[0].date()} → {p.index[-1].date()}")
    print(p.notna().sum().to_string())
    print(p.tail(3).to_string())
