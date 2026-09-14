"""
_btc_data.py — BTC price history loader/cache for the long-term-hold research engine.

Two sources, spliced:
  - yfinance BTC-USD: 2014-09-17 -> present. This is the PRODUCTION series — it's what
    scan_stocks.py already uses for every other ticker, so it's what btc_system.py trades on.
  - blockchain.info "market-price" chart: 2010-08-18 -> present (real trading; 2009 is all
    zero, no market yet). CONTEXT ONLY, for the 2011 bubble/crash and 2013 boom/bust that
    predate yfinance's series. The two sources disagree by a median ~1.5% and up to ~35-40% on
    single volatile days (different exchange-basket/timestamp conventions), so the pre-2014
    segment is rescaled to meet yfinance exactly at the splice date and used for qualitative
    drawdown/episode context, NOT spliced into the quantitative backtest.

Run: ../../vcp_env/bin/python _btc_data.py --refresh    # re-download and cache
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

CACHE = Path(__file__).resolve().parent / "_btc_panel.pkl"

SPLICE_DATE = "2014-09-17"   # first day of the yfinance BTC-USD series


def refresh_cache() -> dict:
    import json
    import urllib.request

    import yfinance as yf

    h = yf.Ticker("BTC-USD").history(period="max", interval="1d", auto_adjust=True)
    if getattr(h.index, "tz", None) is not None:
        h.index = h.index.tz_localize(None)
    yf_df = h[["Open", "High", "Low", "Close", "Volume"]].copy()
    print(f"  yfinance BTC-USD  {yf_df.index[0].date()} -> {yf_df.index[-1].date()}  {len(yf_df)} bars")

    url = "https://api.blockchain.info/charts/market-price?timespan=all&format=json&sampled=false"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.load(resp)
    bc = pd.Series(
        {pd.Timestamp(v["x"], unit="s").normalize(): v["y"] for v in data["values"]},
        name="Close",
    ).sort_index()
    bc = bc[bc > 0]   # drop the pre-market zero era (genesis block, Jan 2009)
    print(f"  blockchain.info   {bc.index[0].date()} -> {bc.index[-1].date()}  {len(bc)} bars")

    out = {"yfinance": yf_df, "blockchain_info": bc}
    with open(CACHE, "wb") as f:
        pickle.dump(out, f)
    print(f"cached -> {CACHE}")
    return out


def load_panel() -> dict:
    if not CACHE.exists():
        return refresh_cache()
    with open(CACHE, "rb") as f:
        return pickle.load(f)


def production_series(panel: dict | None = None) -> pd.Series:
    """The series to trade on: yfinance BTC-USD close, 2014-09-17 -> present."""
    panel = panel or load_panel()
    return panel["yfinance"]["Close"].dropna()


def extended_context_series(panel: dict | None = None) -> pd.Series:
    """
    Full history back to 2010-08-18 for qualitative episode context (2011 bubble/crash,
    2013 boom/bust). Pre-splice segment is rescaled to remove the level jump at the seam;
    do not use this series for the quantitative backtest — use production_series() for that.
    """
    panel = panel or load_panel()
    yf_close = panel["yfinance"]["Close"].dropna()
    bc = panel["blockchain_info"]
    splice = pd.Timestamp(SPLICE_DATE)
    pre = bc.loc[bc.index < splice]
    if splice in bc.index and splice in yf_close.index:
        scale = float(yf_close.loc[splice] / bc.loc[splice])
    else:
        scale = float(yf_close.iloc[0] / pre.iloc[-1])
    pre_scaled = pre * scale
    return pd.concat([pre_scaled, yf_close.loc[yf_close.index >= splice]]).sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    panel = refresh_cache() if args.refresh else load_panel()
    prod = production_series(panel)
    ext = extended_context_series(panel)
    print(f"\nproduction series: {prod.index[0].date()} -> {prod.index[-1].date()}  ({len(prod)} bars)")
    print(f"extended context:  {ext.index[0].date()} -> {ext.index[-1].date()}  ({len(ext)} bars)")


if __name__ == "__main__":
    main()
