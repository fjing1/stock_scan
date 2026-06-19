#!/usr/bin/env python3
"""Stage 1 of the exit-strategy backtest: build & cache the per-symbol panel.

For each symbol in the enabled scan universe, fetch ~6y daily OHLCV, run the
xunlong engine to get the formal-buy / formal-sell flags + Gann levels, and
precompute the indicator series the exit rules need (ATR, MAs, Donchian lows,
Parabolic SAR). Cache the result to a pickle so the backtest can iterate fast
without re-hitting yfinance.

Output: reports/exit_cache/panel.pkl  (dict[symbol] -> DataFrame)
"""
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import scan_stocks as scan
from xunlong import XunLongIndicator

PERIOD = os.environ.get("EXIT_BT_PERIOD", "6y")
UNIVERSE = os.environ.get("EXIT_BT_UNIVERSE", "scan")   # "scan" = 140 enabled list; "full" = stock_symbols_1243
OUT_DIR = scan.BASE_DIR / "reports" / "exit_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PKL = OUT_DIR / ("panel_full.pkl" if UNIVERSE == "full" else "panel.pkl")


def _load_universe():
    if UNIVERSE == "full":
        sys.path.insert(0, str(scan.BASE_DIR.parent))   # repo root holds stock_symbols_1243.py
        import stock_symbols_1243 as m
        syms = set()
        for a in dir(m):
            v = getattr(m, a)
            if isinstance(v, list):
                syms |= {str(s).strip().upper() for s in v if str(s).strip()}
        return sorted(syms)
    df_input, df_meta = scan.load_input_and_meta(scan.INPUT_FILE)
    df_enabled = df_meta[df_meta["enable"] == 1]
    df_run, _ = scan.filter_scannable_universe(df_enabled)
    return sorted(set(df_run["symbol"].astype(str).str.upper().tolist()))


def wilder_atr(high, low, close, n):
    h, l, c = high.values, low.values, close.values
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    tr = pd.Series(tr, index=high.index)
    # Wilder RMA = ewm with alpha=1/n, adjust=False
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def parabolic_sar(high, low, af_step=0.02, af_max=0.20):
    """Returns the SAR stop level that APPLIES TO each bar (computed from prior
    bars only -> no intrabar look-ahead). A separate running SAR carries state."""
    h = high.values
    l = low.values
    n = len(h)
    sar_stop = np.full(n, np.nan)   # level tested against bar i (known at its open)
    if n < 2:
        return pd.Series(sar_stop, index=high.index)
    up = True
    af = af_step
    ep = h[0]
    sar_run = l[0]
    sar_stop[0] = l[0]
    for i in range(1, n):
        cur = sar_run + af * (ep - sar_run)
        if up:
            cur = min(cur, l[i - 1], l[i - 2] if i >= 2 else l[i - 1])
        else:
            cur = max(cur, h[i - 1], h[i - 2] if i >= 2 else h[i - 1])
        sar_stop[i] = cur                       # stop for bar i (pre-flip, prior-bar info only)
        # advance running state using bar i
        if up:
            if l[i] < cur:                      # flip to down
                up = False
                sar_run = ep
                ep = l[i]
                af = af_step
            else:
                sar_run = cur
                if h[i] > ep:
                    ep = h[i]
                    af = min(af + af_step, af_max)
        else:
            if h[i] > cur:                      # flip to up
                up = True
                sar_run = ep
                ep = h[i]
                af = af_step
            else:
                sar_run = cur
                if l[i] < ep:
                    ep = l[i]
                    af = min(af + af_step, af_max)
    return pd.Series(sar_stop, index=high.index)


def build_symbol(sym, xl, want_4h_fallback=True):
    df_d = scan.download_daily(sym, period=PERIOD)
    if df_d is None or len(df_d) < 250:
        return None
    # daily Gann flags do not depend on 4h; try None first to skip the extra fetch
    try:
        df_xl = xl.compute(df_d, None)
        if "Gann_BUY_A" not in df_xl.columns:
            raise RuntimeError("no Gann_BUY_A with None 4h")
    except Exception:
        if not want_4h_fallback:
            return None
        df_4h = scan.download_4h(sym, period="60d")
        df_xl = xl.compute(df_d, df_4h)

    out = pd.DataFrame(index=df_xl.index)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        out[c] = df_d[c].reindex(df_xl.index)
    out["BUY_A"] = df_xl.get("Gann_BUY_A", False).fillna(False).astype(bool)
    out["SELL_1"] = df_xl.get("Gann_SELL_1_confirmed", pd.Series(False, index=df_xl.index)).fillna(False).astype(bool)
    out["Gann_0"] = pd.to_numeric(df_xl.get("Gann_0"), errors="coerce")
    out["Gann_1"] = pd.to_numeric(df_xl.get("Gann_1"), errors="coerce")
    # score sub-features (for cross-sectional 观海买点分 selection tests)
    out["Rank120"] = pd.to_numeric(df_xl.get("Rank120"), errors="coerce")
    out["RSI"] = pd.to_numeric(df_xl.get("RSI"), errors="coerce")
    out["L2_trend"] = pd.to_numeric(df_xl.get("L2_trend"), errors="coerce")
    # exit indicators (computed from OHLC, independent of the engine)
    out["ATR14"] = wilder_atr(out["High"], out["Low"], out["Close"], 14)
    out["ATR22"] = wilder_atr(out["High"], out["Low"], out["Close"], 22)
    out["EMA20"] = out["Close"].ewm(span=20, adjust=False).mean()
    out["SMA20"] = out["Close"].rolling(20).mean()
    out["SMA50"] = out["Close"].rolling(50).mean()
    out["SMA100"] = out["Close"].rolling(100).mean()
    out["DONCH10"] = out["Low"].rolling(10).min()
    out["DONCH20"] = out["Low"].rolling(20).min()
    out["SAR_STD"] = parabolic_sar(out["High"], out["Low"], 0.02, 0.20)
    out["SAR_SLOW"] = parabolic_sar(out["High"], out["Low"], 0.01, 0.10)
    return out


def main():
    syms = _load_universe()
    only = sys.argv[1:]
    if only:
        syms = [s for s in syms if s in {x.upper() for x in only}]
    print(f"Building panel for {len(syms)} symbols (universe={UNIVERSE}, period={PERIOD})...")

    xl = XunLongIndicator()
    panel = {}
    t0 = time.time()
    n_entries = 0
    for i, s in enumerate(syms, 1):
        try:
            df = build_symbol(s, xl)
        except Exception as e:
            print(f"  [{i}/{len(syms)}] {s}: ERROR {e}")
            continue
        if df is None:
            print(f"  [{i}/{len(syms)}] {s}: skipped (insufficient data)")
            continue
        panel[s] = df
        ne = int(df["BUY_A"].sum())
        n_entries += ne
        if i % 10 == 0 or i == len(syms):
            print(f"  [{i}/{len(syms)}] {s}: {len(df)} bars, {ne} entries  | total entries={n_entries}  ({time.time()-t0:.0f}s)")

    with open(OUT_PKL, "wb") as f:
        pickle.dump(panel, f)
    print(f"\nSaved {len(panel)} symbols, {n_entries} total formal-buy entries -> {OUT_PKL}")


if __name__ == "__main__":
    main()
