"""Python 3 port of "寻龙诀 Panel V1" (Xunlongjue Panel) — Pine v6, © bigbenv5 (MPL-2.0).

Source script: xunlong_panel.pine

Reproduces the panel's computed series on an OHLCV DataFrame shaped like
yfinance output (columns High/Low/Close):

    trend  -- L2 trend strength "t" (>0 only in an uptrend; Pine var6b-45)
    pump   -- L2 capitulation/pump "p"
    bbuy   -- green-bar trigger: ema(typical,6) crosses above ema(that,5)
    varr1  -- 0-100 up/abs momentum ratio (xsa-smoothed)
    red    -- red-bar trigger: varr1 crosses under red_level (default 82)
    rsi    -- TradingView RSI(14) (Wilder rma)

Plus panel_score(): the 0-10 confluence score used by the repo's scanner
(bbuy in last 3 bars +3; trend rising +2; trend>0 +2; RSI rising & 50-70 +2;
no red in last 3 bars +1). This is the same logic as _compute_xunlong_signals
in vcp_enhanced_obv_criteria.py, kept self-contained here.

Run as a script on live Yahoo Finance data::

    python xunlong_panel.py AAPL --period 1y
    python xunlong_panel.py NVDA --rows 8
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PANEL_CONFIG = {
    "K": 9,
    "D": 3,
    "MidPeriod": 58,
    "varr_len": 6,
    "red_level": 82.0,
    "recent_window": 3,
    "rsi_length": 14,
}


# --------------------------------------------------------------------------- #
# Pine built-in equivalents
# --------------------------------------------------------------------------- #
def _xsa(src, length, wei):
    """Pine ``xsa``: SMA seed then out[t] = (src*wei + out[t-1]*(len-wei))/len.

    Numpy-accelerated (the repo's reference uses .iloc); NaN inputs carry the
    previous value forward, matching the Pine recursion across na bars.
    """
    s = pd.Series(src, dtype=float).to_numpy()
    n = len(s)
    out = np.full(n, np.nan)
    if n < length:
        return pd.Series(out)
    seed = s[:length]
    out[length - 1] = np.nanmean(seed) if np.isfinite(seed).any() else np.nan
    a, b = wei / length, (length - wei) / length
    for i in range(length, n):
        prev = out[i - 1]
        cur = s[i]
        out[i] = prev if np.isnan(cur) else cur * a + prev * b
    return pd.Series(out)


def _ema(series, length):
    """Pine ``ema``: EMA with alpha = 2/(length+1)."""
    return series.ewm(span=length, adjust=False).mean()


def _rma(series, length):
    """Pine ``rma`` (Wilder): SMA seed then alpha = 1/length."""
    s = series.astype(float).to_numpy()
    n = len(s)
    out = np.full(n, np.nan)
    if n < length:
        return pd.Series(out)
    out[length - 1] = np.nanmean(s[:length])
    alpha = 1.0 / length
    for i in range(length, n):
        prev = out[i - 1]
        cur = s[i]
        out[i] = prev if np.isnan(cur) else alpha * cur + (1.0 - alpha) * prev
    return pd.Series(out)


# --------------------------------------------------------------------------- #
# Panel
# --------------------------------------------------------------------------- #
def compute_panel(df: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """Compute the panel series for an OHLCV DataFrame (yfinance shape)."""
    cfg = {**PANEL_CONFIG, **(cfg or {})}
    K, D, MidPeriod = cfg["K"], cfg["D"], cfg["MidPeriod"]
    varr_len, red_level = cfg["varr_len"], cfg["red_level"]

    high = df["High"].astype(float).reset_index(drop=True)
    low = df["Low"].astype(float).reset_index(drop=True)
    close = df["Close"].astype(float).reset_index(drop=True)

    # ---- Trend (t) ----
    high_K = high.rolling(K).max()
    low_K = low.rolling(K).min()
    denK = (high_K - low_K).clip(lower=1e-9)
    var1b = (high_K - close) / denK * 100 - 70
    var2b = _xsa(var1b, K, 1) + 100
    var3b = (close - low_K) / denK * 100
    var4b = _xsa(var3b, D, 1)
    var5b = _xsa(var4b, D, 1) + 100
    var6b = var5b - var2b
    trend = (var6b - 45).where(var6b > 45, 0.0)

    # ---- Pump (p) ----
    var2q = low.shift(1)
    abs_move = (low - var2q).abs()
    up_move = (low - var2q).clip(lower=0)
    s_abs = _xsa(abs_move, D, 1)
    s_up = _xsa(up_move, D, 1)
    var3q = (s_abs / s_up.replace(0, np.nan)).fillna(0) * 100.0
    chg = close.diff()
    val = pd.Series(np.where(chg > 0, var3q * 10.0, var3q / 10.0))
    var4q = val.ewm(span=D, adjust=False).mean()
    var5q = low.rolling(30).min()
    var6q = var4q.rolling(30).max()
    sma_mid = close.rolling(MidPeriod).mean()
    var7q = (~sma_mid.isna()).astype(float)
    inner = ((var4q + var6q * 2.0) / 2.0).where(low <= var5q, 0.0)
    var8q = inner.ewm(span=D, adjust=False).mean() / 999.0 * var7q
    pump = var8q.clip(upper=100.0)

    # ---- bbuy (green bar trigger) ----
    typical = (close + low + high) / 3.0
    d2 = typical.ewm(span=6, adjust=False).mean()
    d3 = d2.ewm(span=5, adjust=False).mean()
    bbuy = (d2 > d3) & (d2.shift(1) <= d3.shift(1))

    # ---- varr1 / red bar trigger ----
    chg_close = close.diff()
    up_part = _xsa(chg_close.clip(lower=0), varr_len, 1)
    abs_part = _xsa(chg_close.abs(), varr_len, 1)
    varr1 = (100.0 * up_part / abs_part.replace(0, np.nan)).fillna(0)
    red = (varr1 < red_level) & (varr1.shift(1) >= red_level)

    # ---- RSI(14) ----
    change = close.diff()
    rup = _rma(change.clip(lower=0.0), cfg["rsi_length"])
    rdn = _rma((-change).clip(lower=0.0), cfg["rsi_length"])
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100.0 - 100.0 / (1.0 + rup / rdn)

    out = pd.DataFrame({
        "trend": trend.fillna(0).to_numpy(),
        "pump": pump.fillna(0).to_numpy(),
        "bbuy": bbuy.fillna(False).to_numpy(),
        "varr1": varr1.to_numpy(),
        "red": red.fillna(False).to_numpy(),
        "rsi": rsi.to_numpy(),
    }, index=df.index)
    return out


def panel_score(panel: pd.DataFrame, cfg=None) -> pd.Series:
    """Per-bar 0-10 confluence score (repo's check_xunlongjue, vectorized)."""
    cfg = {**PANEL_CONFIG, **(cfg or {})}
    w = cfg["recent_window"]
    t = panel["trend"]
    bbuy = panel["bbuy"].astype(float)
    red = panel["red"].astype(float)
    rsi = panel["rsi"]

    bbuy_recent = bbuy.rolling(w, min_periods=1).max() > 0
    trend_rising = (t > t.shift(1)) & (t.shift(1) > t.shift(2))
    trend_active = t > 0
    rsi_rising = (rsi > rsi.shift(1)) & (rsi.shift(1) > rsi.shift(2))
    rsi_constructive = rsi_rising & (rsi >= 50) & (rsi <= 70)
    no_red_recent = red.rolling(w, min_periods=1).max() == 0

    score = (3 * bbuy_recent.astype(int) + 2 * trend_rising.astype(int)
             + 2 * trend_active.astype(int) + 2 * rsi_constructive.astype(int)
             + 1 * no_red_recent.astype(int))
    return score


# --------------------------------------------------------------------------- #
# Yahoo Finance helpers / CLI
# --------------------------------------------------------------------------- #
def fetch_yahoo(symbol, period="1y", interval="1d"):
    import yfinance as yf
    df = yf.download(symbol, period=period, interval=interval,
                     auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description='寻龙诀 Panel V1 (Pine v6 port)')
    p.add_argument("symbol")
    p.add_argument("--period", default="1y")
    p.add_argument("--interval", default="1d")
    p.add_argument("--rows", type=int, default=10)
    args = p.parse_args(argv)

    df = fetch_yahoo(args.symbol, period=args.period, interval=args.interval)
    if df.empty:
        raise SystemExit(f"No data for {args.symbol!r}")
    panel = compute_panel(df)
    panel["score"] = panel_score(panel)
    show = panel[["trend", "pump", "bbuy", "varr1", "red", "rsi", "score"]]
    print(show.tail(args.rows).round(2).to_string())


if __name__ == "__main__":
    main()
