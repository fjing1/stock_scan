"""Cross-sectional, market-neutral ensemble — the 'Citadel-style' alpha test.

Each rebalance (weekly), score every stock in a broad liquid universe with a
blend of cross-sectionally z-scored signals, go LONG the top quintile and SHORT
the bottom quintile (equal-weight, dollar-neutral), and measure the spread return.
Because long $ == short $, the spread is ~market-neutral: it isolates ALPHA, not
beta. The 2022 bear in the test window is the litmus test.

Alpha blend (each z-scored across names each day):
  reversal  : -5-day return        (recent losers bounce — short-term reversal)
  momentum  : 12-1 month return    (long-term winners — relative strength)
  trend     : close/SMA200 - 1     (uptrend names)
  oversold  : -RSI(2)              (oversold names)
  score = z(reversal) + z(momentum) + z(trend) + z(oversold)   (equal weight, not tuned)

OOS: train < 2019, test 2019 -> present. Reports gross/net (after turnover costs)
annualized return, Sharpe, % positive periods, beta-to-SPY, and the LONG-ONLY top
quintile vs SPY (the realistic retail, no-shorting view).

CAVEAT: universe = CURRENT names (survivorship bias) — inflates the long leg,
deflates the short leg. Treat long-short magnitude as optimistic; the long-only-
vs-SPY and the beta≈0 / OOS-consistency are the trustworthy reads.

    python market_neutral_ensemble.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yfinance as yf  # noqa: E402

BENCHMARK = "SPY"
SPLIT = pd.Timestamp("2019-01-01")
REBAL = 5                 # trading days between rebalances (weekly)
PPY = 252 / REBAL         # rebalance periods per year (~50.4)
COST_RT = 0.0010          # round-trip cost (10 bps) applied to turned-over fraction
N_CANDIDATES = 220        # how many universe names to consider
MIN_BARS = 2500           # ~10y history required


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


def rsi(close, n):
    ch = close.diff()
    up = ch.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-ch).clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50)


def zrow(df):
    """Cross-sectional z-score per row (date), across columns (names)."""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)


def annualize(series):
    s = np.asarray(series, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) < 5:
        return dict(n=len(s), ann=float("nan"), sharpe=float("nan"), pos=float("nan"),
                    cum=float("nan"))
    mean, std = s.mean(), s.std(ddof=1)
    ann = (1 + mean) ** PPY - 1
    sharpe = (mean / std) * np.sqrt(PPY) if std > 0 else float("nan")
    cum = np.prod(1 + s) - 1
    return dict(n=len(s), ann=ann * 100, sharpe=sharpe, pos=(s > 0).mean() * 100, cum=cum * 100)


def main():
    from stock_symbols_1243 import STOCK_SYMBOLS
    cands = list(dict.fromkeys(STOCK_SYMBOLS))[:N_CANDIDATES]
    print(f"downloading {len(cands)} candidates + {BENCHMARK} (20y daily) ...", flush=True)
    data = {}
    for i in range(0, len(cands), 110):
        data.update(fetch_batch(cands[i:i + 110], "20y", "1d"))
    spy = fetch_batch([BENCHMARK], "20y", "1d").get(BENCHMARK)

    # keep names with enough history and a real price
    keep = {s: d for s, d in data.items()
            if len(d) >= MIN_BARS and float(d["Close"].iloc[-1]) > 5}
    closes = pd.DataFrame({s: d["Close"].astype(float) for s, d in keep.items()})
    print(f"universe after history/price filter: {closes.shape[1]} names, "
          f"{closes.index[0].date()} -> {closes.index[-1].date()}")

    # features (dates x names)
    ret5 = closes.pct_change(5)
    mom = closes.shift(21) / closes.shift(252) - 1
    ma200 = closes.rolling(200).mean()
    trendr = closes / ma200 - 1
    rsi2 = closes.apply(lambda c: rsi(c, 2))
    fwd = closes.shift(-REBAL) / closes - 1
    spy_close = spy["Close"].astype(float).reindex(closes.index)
    spy_fwd = (spy_close.shift(-REBAL) / spy_close - 1)

    score = zrow(-ret5) + zrow(mom) + zrow(trendr) + zrow(-rsi2)

    dates = closes.index
    rebal_idx = range(252, len(dates) - REBAL, REBAL)   # start after warmup
    prev_long, prev_short = set(), set()
    rows = []
    for p in rebal_idx:
        t = dates[p]
        sc = score.iloc[p].dropna()
        fz = fwd.iloc[p]
        valid = sc.index[fz.reindex(sc.index).notna()]
        sc = sc.loc[valid]
        if len(sc) < 20:
            continue
        q = max(3, len(sc) // 5)
        ranked = sc.sort_values()
        shorts = list(ranked.index[:q])
        longs = list(ranked.index[-q:])
        lr = float(fz[longs].mean())
        sr = float(fz[shorts].mean())
        spread = lr - sr
        spy_f = float(spy_fwd.iloc[p]) if not np.isnan(spy_fwd.iloc[p]) else np.nan
        # turnover vs previous book
        turn = (len(set(longs) ^ prev_long) + len(set(shorts) ^ prev_short)) / (2 * q) if prev_long else 1.0
        prev_long, prev_short = set(longs), set(shorts)
        rows.append(dict(date=t, spread=spread, long=lr, short=sr, spy=spy_f,
                         net=spread - turn * COST_RT, turn=turn))

    R = pd.DataFrame(rows)
    tr, te = R[R["date"] < SPLIT], R[R["date"] >= SPLIT]

    def report(name, df):
        if len(df) < 5:
            print(f"  {name}: too few periods")
            return
        ls = annualize(df["spread"]); net = annualize(df["net"])
        lo = annualize(df["long"]); sp = annualize(df["spy"])
        loex = annualize(df["long"].values - df["spy"].values)
        # market beta of the long-short spread
        x, y = df["spy"].values, df["spread"].values
        m = ~(np.isnan(x) | np.isnan(y))
        beta = np.polyfit(x[m], y[m], 1)[0] if m.sum() > 5 else float("nan")
        print(f"\n  [{name}]  periods={ls['n']}  avg turnover={df['turn'].mean()*100:.0f}%")
        print(f"    Long-Short (alpha)  : ann {ls['ann']:+.1f}%  Sharpe {ls['sharpe']:.2f}  "
              f"%+ {ls['pos']:.0f}  beta-to-SPY {beta:+.2f}")
        print(f"    Long-Short net costs: ann {net['ann']:+.1f}%  Sharpe {net['sharpe']:.2f}")
        print(f"    Long-only (topQ)    : ann {lo['ann']:+.1f}%  Sharpe {lo['sharpe']:.2f}  "
              f"%+ {lo['pos']:.0f}")
        print(f"    SPY (same periods)  : ann {sp['ann']:+.1f}%  Sharpe {sp['sharpe']:.2f}")
        print(f"    Long-only ALPHA vs SPY: ann {loex['ann']:+.1f}%  Sharpe {loex['sharpe']:.2f}  "
              f"%+ {loex['pos']:.0f}")

    print("\n================ MARKET-NEUTRAL ENSEMBLE (weekly rebalance) ================")
    print("score = z(-5d ret) + z(12-1 mom) + z(close/MA200) + z(-RSI2); top/bottom quintile")
    report("TRAIN < 2019", tr)
    report("TEST 2019-present (incl. 2022 bear)", te)
    print("\nReads: Long-Short beta-to-SPY ~0 confirms market-neutral. Sharpe>0 & %+>50 in")
    print("BOTH train & test = persistent alpha. Long-only-ALPHA-vs-SPY = realistic no-short edge.")
    print("CAVEAT: current-names universe (survivorship) inflates long leg / deflates short leg.")


if __name__ == "__main__":
    main()
