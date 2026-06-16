"""Python 3 port of the TradingView Pine v3 study "Cycle and Stoch".

Source script: cycle_patter_for_swing.pine

The original plots four series in a sub-pane plus four horizontal guide
levels. This module reproduces the series on an OHLCV DataFrame shaped like
the output of ``yfinance.download`` (columns ``High``/``Low``/``Close``):

    cycle    -- double-smoothed stochastic "cycle" line   (Pine: cycleh)
    stoch_k  -- fast %K, EMA(4) smoothed                   (Pine: k2)
    stoch_d  -- %D, EMA(4) of EMA(3) of the raw %K         (Pine: d2)
    rsi      -- Wilder RSI(14)                             (Pine: rsi1)

The horizontal reference levels (19 / 39.5 / 60.5 / 81) are exposed as
``LEVELS`` for plotting or signal logic.

Indicator helpers are hand-ported to match Pine semantics exactly, in the
same style as the ``_xsa`` port in ``vcp_enhanced_obv_criteria.py``:
``ema`` -> ``ewm(span=n, adjust=False)``; ``rma`` -> SMA seed then
``alpha = 1/n``; ``stoch`` -> ``100 * (src - llv) / (hhv - llv)``.

Run as a script to compute on live Yahoo Finance data::

    python cycle_patter_for_swing.py AAPL --period 1y
    python cycle_patter_for_swing.py NVDA --interval 1wk --plot
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Plotted horizontal guide levels from the Pine script.
LEVELS = {
    "lower_band": 19.0,   # plot(19)   -- deep oversold
    "lower_mid": 39.5,    # plot(39.5)
    "upper_mid": 60.5,    # plot(60.5)
    "upper_band": 81.0,   # plot(81)   -- deep overbought
}


# --------------------------------------------------------------------------- #
# Pine built-in equivalents
# --------------------------------------------------------------------------- #
def _ema(series: pd.Series, length: int) -> pd.Series:
    """Pine ``ema()``: EMA with alpha = 2/(length+1), recursive seed."""
    return series.ewm(span=length, adjust=False).mean()


def _rma(series: pd.Series, length: int) -> pd.Series:
    """Pine ``rma()`` (Wilder's MA): SMA seed, then alpha = 1/length.

    NaNs in the input carry the previous value forward (matching how Pine
    holds the running average across ``na`` bars).
    """
    s = series.astype(float).reset_index(drop=True)
    n = len(s)
    out = np.full(n, np.nan)
    if n < length:
        return pd.Series(out)
    out[length - 1] = s.iloc[:length].mean()
    alpha = 1.0 / length
    for i in range(length, n):
        cur = s.iloc[i]
        prev = out[i - 1]
        out[i] = prev if np.isnan(cur) else alpha * cur + (1.0 - alpha) * prev
    return pd.Series(out)


def _stoch(source: pd.Series, high: pd.Series, low: pd.Series,
           length: int) -> pd.Series:
    """Pine ``stoch()``: ``100 * (source - llv(low,n)) / (hhv(high,n) - llv(low,n))``.

    A flat window (hhv == llv) yields ``na``, as it does in Pine.
    """
    llv = low.rolling(length).min()
    hhv = high.rolling(length).max()
    denom = (hhv - llv).replace(0, np.nan)  # 0/0 -> na
    return 100.0 * (source - llv) / denom


def _rsi(source: pd.Series, length: int) -> pd.Series:
    """Pine ``rsi()``: Wilder-smoothed ratio of up/down moves."""
    change = source.diff()
    up = _rma(change.clip(lower=0.0), length)
    down = _rma((-change).clip(lower=0.0), length)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = up / down  # x/0 -> inf -> rsi 100; 0/0 -> nan (na), as in Pine
        rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi


# --------------------------------------------------------------------------- #
# Indicator
# --------------------------------------------------------------------------- #
def compute_cycle_stoch(df: pd.DataFrame,
                        pds: int = 10,
                        period_k: int = 10,
                        rsi_length: int = 14) -> pd.DataFrame:
    """Compute the four Cycle-and-Stoch series for an OHLCV DataFrame.

    Args:
        df: DataFrame with ``High``, ``Low``, ``Close`` columns (yfinance shape).
        pds: stochastic length for the cycle (Pine ``PDS``, default 10).
        period_k: stochastic length for %K/%D (Pine ``periodK``, default 10).
        rsi_length: RSI length (Pine uses 14).

    Returns:
        DataFrame indexed like ``df`` with columns
        ``cycle``, ``stoch_k``, ``stoch_d``, ``rsi``.
    """
    high = df["High"].astype(float).reset_index(drop=True)
    low = df["Low"].astype(float).reset_index(drop=True)
    close = df["Close"].astype(float).reset_index(drop=True)

    # ---- Cycle: double-smoothed stochastic (Pine PreC -> xPreCalc -> xDSS) ----
    pre_c = _stoch(close, high, low, pds)        # raw fast %K
    x_pre = _ema(pre_c, 3)                        # Pine xPreCalc (also == k1)
    x_dss = _stoch(x_pre, x_pre, x_pre, pds)      # stochastic of the smoothed line
    x_dss_e = 0.5 * x_dss + 0.5 * x_pre.shift(1)  # Pine xPreCalc[1] = previous bar
    cycle = _ema(x_dss_e, 3)                      # Pine cycleh

    # ---- Stochastic %K / %D ----
    # When period_k == pds the raw %K equals pre_c above; recomputed so the
    # two lengths can differ without coupling, mirroring the Pine source.
    raw_k = _stoch(close, high, low, period_k)
    stoch_k = _ema(raw_k, 4)                      # Pine k2
    stoch_d = _ema(_ema(raw_k, 3), 4)             # Pine d2 = ema(k1, 4)

    # ---- RSI ----
    rsi = _rsi(close, rsi_length)                 # Pine rsi1

    out = pd.DataFrame({
        "cycle": cycle,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "rsi": rsi,
    })
    out.index = df.index
    return out


# --------------------------------------------------------------------------- #
# Yahoo Finance helpers / CLI
# --------------------------------------------------------------------------- #
def fetch_yahoo(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV from Yahoo Finance and flatten single-ticker columns."""
    import yfinance as yf

    df = yf.download(symbol, period=period, interval=interval,
                     auto_adjust=False, progress=False)
    # yfinance returns a MultiIndex (Price/Ticker) for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _plot(symbol: str, df: pd.DataFrame, ind: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib is required for --plot (pip install matplotlib)")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 8),
        gridspec_kw={"height_ratios": [2, 1]},
    )
    ax1.plot(df.index, df["Close"], color="black", lw=1)
    ax1.set_title(f"{symbol} — Close")
    ax1.grid(alpha=0.3)

    ax2.plot(ind.index, ind["cycle"], color="#ff8c00", lw=1, label="cycle")
    ax2.plot(ind.index, ind["rsi"], color="#800080", lw=2, label="rsi")
    ax2.plot(ind.index, ind["stoch_k"], color="#0095ff", lw=1, label="stoch K")
    ax2.plot(ind.index, ind["stoch_d"], color="#ff0000", lw=2, label="stoch D")
    for lvl in LEVELS.values():
        ax2.axhline(lvl, color="grey", lw=0.5, ls="--")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper left", ncol=4, fontsize=8)
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(
        description='"Cycle and Stoch" (Pine v3 port) on Yahoo Finance data')
    p.add_argument("symbol", help="Ticker, e.g. AAPL")
    p.add_argument("--period", default="1y", help="yfinance period (default 1y)")
    p.add_argument("--interval", default="1d", help="yfinance interval (default 1d)")
    p.add_argument("--rows", type=int, default=10,
                   help="Number of trailing rows to print (default 10)")
    p.add_argument("--plot", action="store_true", help="Show a matplotlib chart")
    args = p.parse_args(argv)

    df = fetch_yahoo(args.symbol, period=args.period, interval=args.interval)
    if df.empty:
        raise SystemExit(f"No data returned for {args.symbol!r}")

    ind = compute_cycle_stoch(df)
    print(ind.tail(args.rows).round(2).to_string())
    if args.plot:
        _plot(args.symbol, df, ind)


if __name__ == "__main__":
    main()
