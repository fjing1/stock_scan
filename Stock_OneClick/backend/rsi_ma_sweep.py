#!/usr/bin/env python3
"""Which moving-average LENGTH on the RSI gives the best daily strategy?

Models the 寻龙诀 Panel's RSI + yellow smoothing-MA line as a tradeable rule:
  long when RSI >= MA(RSI, len), cash when RSI < MA(RSI, len).
Position is applied the NEXT day (no look-ahead). 15 bps per turn (round-trip
= 2 turns). RSI is Wilder's (ta.rma), matching the Pine script.

We sweep the MA length (the "number" the user asked about) and, secondarily, the
MA type (SMA/EMA/RMA/WMA). Baseline = buy-and-hold the same name. Reported per
name then pooled equal-weight across the universe, with a walk-forward (per-year)
stability check because a single full-sample number over-fits (repo convention).

    ../../vcp_env/bin/python rsi_ma_sweep.py                 # SPY+QQQ index test
    ../../vcp_env/bin/python rsi_ma_sweep.py --universe basket
    ../../vcp_env/bin/python rsi_ma_sweep.py --rsi-len 14 --ma-type SMA

Caveats: long-only cash strategy; close-to-close; survivor basket flatters
absolutes (compare strategy-vs-hold on the SAME names, not absolute returns).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
import scan_stocks as scan  # noqa: E402

COST = 0.0015           # 15 bps per turn (entry OR exit)
TD = 252

# survivor large-cap basket (same as _swing_basket.py) — survivorship-biased, use
# for the strategy-vs-hold COMPARISON, not absolute returns.
BASKET = ["AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "BAC", "WFC", "XOM", "CVX", "JNJ",
          "PFE", "MRK", "PG", "KO", "PEP", "WMT", "HD", "MCD", "DIS", "NKE", "INTC",
          "CSCO", "ORCL", "IBM", "QCOM", "TXN", "CAT", "BA", "MMM", "UNH", "T", "VZ",
          "C", "GS", "COST", "LOW", "HON", "AMGN", "ADBE", "CRM"]
INDEX = ["SPY", "QQQ"]

MA_LENGTHS = [3, 5, 7, 9, 10, 12, 14, 18, 21, 30, 50]


def rsi_wilder(x: pd.Series, n: int) -> pd.Series:
    """Wilder's RSI (== Pine ta.rma smoothing)."""
    d = x.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    rs = up.ewm(alpha=1 / n, adjust=False).mean() / dn.ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + rs)


def ma(x: pd.Series, n: int, kind: str) -> pd.Series:
    if kind == "SMA":
        return x.rolling(n).mean()
    if kind == "EMA":
        return x.ewm(span=n, adjust=False).mean()
    if kind == "RMA":
        return x.ewm(alpha=1 / n, adjust=False).mean()
    if kind == "WMA":
        w = np.arange(1, n + 1, dtype=float)
        return x.rolling(n).apply(lambda v: np.dot(v, w) / w.sum(), raw=True)
    raise ValueError(kind)


def strat_returns(close: pd.Series, rsi_len: int, ma_len: int, ma_kind: str) -> pd.Series:
    """Daily net strategy returns for one name. Long when RSI>=MA(RSI), else cash."""
    close = close.astype(float)
    r = rsi_wilder(close, rsi_len)
    m = ma(r, ma_len, ma_kind)
    signal = (r >= m).astype(float)              # today's desired position
    return _apply(close, signal)


def strat_returns_dual(close: pd.Series, rsi_len: int, fast: int, slow: int, ma_kind: str) -> pd.Series:
    """Dual-MA crossover on the RSI: long when fastMA(RSI) >= slowMA(RSI), else cash.
    fast==1 degenerates to raw-RSI-crosses-MA (the single-MA rule)."""
    close = close.astype(float)
    r = rsi_wilder(close, rsi_len)
    f = r if fast <= 1 else ma(r, fast, ma_kind)
    s = ma(r, slow, ma_kind)
    signal = (f >= s).astype(float)
    return _apply(close, signal)


def _apply(close: pd.Series, signal: pd.Series) -> pd.Series:
    pos = signal.shift(1).fillna(0.0)            # act next day -> no look-ahead
    ret = close.pct_change().fillna(0.0)
    turn = pos.diff().abs().fillna(0.0)
    return pos * ret - turn * COST


def metrics(daily: pd.Series) -> dict:
    daily = daily.dropna()
    if len(daily) < 2:
        return {}
    eq = (1 + daily).cumprod()
    total = eq.iloc[-1] - 1
    yrs = len(daily) / TD
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    sharpe = (daily.mean() / daily.std(ddof=1) * np.sqrt(TD)) if daily.std(ddof=1) else np.nan
    peak = eq.cummax()
    maxdd = (eq / peak - 1).min()
    return {"total": total, "cagr": cagr, "sharpe": sharpe, "maxdd": maxdd}


def pooled_daily(closes: dict, fn) -> pd.Series:
    """Equal-weight average of per-name daily returns across the calendar."""
    per = {sym: fn(c) for sym, c in closes.items()}
    idx = sorted(set().union(*[s.index for s in per.values()]))
    idx = pd.DatetimeIndex(idx)
    mat = pd.DataFrame({sym: s.reindex(idx) for sym, s in per.items()})
    return mat.mean(axis=1)   # NaN-aware: names average in once they have data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", choices=["index", "basket", "both"], default="index")
    ap.add_argument("--rsi-len", type=int, default=14)
    ap.add_argument("--ma-type", default="SMA", choices=["SMA", "EMA", "RMA", "WMA"])
    ap.add_argument("--period", default="max")
    ap.add_argument("--dual", action="store_true",
                    help="sweep fast/slow dual-MA crossover on the RSI instead of single-MA")
    args = ap.parse_args()

    names = {"index": INDEX, "basket": BASKET, "both": INDEX + BASKET}[args.universe]
    print(f"Downloading {len(names)} names ({args.universe})...")
    closes: dict[str, pd.Series] = {}
    for sym in names:
        try:
            df = scan.download_daily(sym, period=args.period)
        except Exception:
            df = None
        if df is None or len(df) < 300:
            continue
        closes[sym] = df["Close"].astype(float)
    print(f"got data for {len(closes)}/{len(names)} names")
    if not closes:
        print("no data")
        return 1
    rng = pd.DatetimeIndex(sorted(set().union(*[c.index for c in closes.values()])))
    print(f"date range {rng.min().date()} .. {rng.max().date()}   RSI={args.rsi_len} MA type={args.ma_type}\n")

    # buy-and-hold pooled baseline
    bh = pooled_daily(closes, lambda c: c.pct_change().fillna(0.0))
    bhm = metrics(bh)
    print(f"{'BUY&HOLD (pooled)':<22} total {bhm['total']:>+8.1%}  CAGR {bhm['cagr']:>+7.2%}  "
          f"Sharpe {bhm['sharpe']:>5.2f}  maxDD {bhm['maxdd']:>+7.1%}")
    rows = []
    if args.dual:
        # dual-MA crossover: fast MA of RSI crossing slow MA of RSI
        pairs = [(f, s) for f in [1, 2, 3, 5, 7] for s in [9, 14, 21, 30, 50] if f < s]
        print("\n=== RSI dual-MA crossover: long when fastMA(RSI) >= slowMA(RSI), else cash ===")
        print("  (fast=1 == raw RSI crossing slow MA = the single-MA rule)")
        print(f"  {'fast':>5}{'slow':>6}{'total':>10}{'CAGR':>9}{'Sharpe':>8}{'maxDD':>9}{'vs B&H Sh':>11}")
        for f, s in pairs:
            pooled = pooled_daily(closes, lambda c, ff=f, ss=s:
                                  strat_returns_dual(c, args.rsi_len, ff, ss, args.ma_type))
            m = metrics(pooled)
            rows.append(((f, s), m, pooled))
            flag = "  <-- beats B&H" if m["sharpe"] > bhm["sharpe"] else ""
            print(f"  {f:>5}{s:>6}{m['total']:>+10.1%}{m['cagr']:>+9.2%}{m['sharpe']:>8.2f}"
                  f"{m['maxdd']:>+9.1%}{m['sharpe']-bhm['sharpe']:>+11.2f}{flag}")
        best = max(rows, key=lambda r: r[1]["sharpe"])
        print(f"\n  -> best (fast,slow) by Sharpe: {best[0]}  (Sharpe {best[1]['sharpe']:.2f}, "
              f"CAGR {best[1]['cagr']:+.2%}, maxDD {best[1]['maxdd']:+.1%})")
        best_label = f"fast/slow {best[0][0]}/{best[0][1]}"
    else:
        print("\n=== RSI-MA crossover: sweep MA length (long when RSI>=MA, else cash) ===")
        print(f"  {'MA len':>7}{'total':>10}{'CAGR':>9}{'Sharpe':>8}{'maxDD':>9}{'vs B&H (Sharpe)':>17}")
        for mlen in MA_LENGTHS:
            pooled = pooled_daily(closes, lambda c, ml=mlen: strat_returns(c, args.rsi_len, ml, args.ma_type))
            m = metrics(pooled)
            rows.append((mlen, m, pooled))
            d_sharpe = m["sharpe"] - bhm["sharpe"]
            flag = "  <-- beats B&H Sharpe" if d_sharpe > 0 else ""
            print(f"  {mlen:>7}{m['total']:>+10.1%}{m['cagr']:>+9.2%}{m['sharpe']:>8.2f}"
                  f"{m['maxdd']:>+9.1%}{d_sharpe:>+13.2f}{flag}")
        best = max(rows, key=lambda r: r[1]["sharpe"])
        print(f"\n  -> best MA length by Sharpe: {best[0]}  (Sharpe {best[1]['sharpe']:.2f}, "
              f"CAGR {best[1]['cagr']:+.2%}, maxDD {best[1]['maxdd']:+.1%})")
        best_label = f"MA len {best[0]}"

    # walk-forward per-year stability for the best config (repo convention: one
    # full-sample number over-fits; require the edge to hold across years)
    _, _, best_daily = best
    print(f"\n=== walk-forward: {best_label} vs B&H, by calendar year (Sharpe) ===")
    print(f"  {'year':>6}{'strat Sh':>10}{'B&H Sh':>9}{'strat ret':>11}{'B&H ret':>10}{'win?':>6}")
    yrs = sorted(set(best_daily.index.year))
    wins = 0
    n_yr = 0
    for y in yrs:
        sd = best_daily[best_daily.index.year == y].dropna()
        bd = bh[bh.index.year == y].dropna()
        if len(sd) < 30:
            continue
        n_yr += 1
        ss = sd.mean() / sd.std(ddof=1) * np.sqrt(TD) if sd.std(ddof=1) else np.nan
        bs = bd.mean() / bd.std(ddof=1) * np.sqrt(TD) if bd.std(ddof=1) else np.nan
        sr = (1 + sd).prod() - 1
        br = (1 + bd).prod() - 1
        win = ss > bs
        wins += int(win)
        print(f"  {y:>6}{ss:>+10.2f}{bs:>+9.2f}{sr:>+11.1%}{br:>+10.1%}{('Y' if win else 'n'):>6}")
    print(f"  -> strat Sharpe beat B&H in {wins}/{n_yr} years")

    print("\nCaveats: long-only cash, close-to-close, 15bps/turn. Basket is survivorship-")
    print("biased (absolutes flattered); trust the strategy-vs-B&H comparison, not the level.")
    print("A length that wins full-sample but loses most years is over-fit — prefer consistency.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
