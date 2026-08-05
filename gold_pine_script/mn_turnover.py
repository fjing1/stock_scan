"""Turnover-reduction pass on the market-neutral ensemble.

The base ensemble (market_neutral_ensemble.py) has ~120%/week turnover, which
halves net-of-cost Sharpe. This tests standard turnover cutters and reports
GROSS vs NET Sharpe + turnover + market beta, train/test, to find the config
with the best DEPLOYABLE (net) edge that stays market-neutral:

  - rebalance buffer (hysteresis): enter the long book in the top `enter`%,
    but only drop a name when it falls out of the wider `exit`% band
  - score smoothing: rank on a trailing mean of the daily score
  - rebalance frequency: weekly / biweekly / monthly

Same blend and universe as the base ensemble. 10bps round-trip cost on the
turned-over fraction.

    python mn_turnover.py
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
COST_RT = 0.0010
N_CANDIDATES = 220
MIN_BARS = 2500


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
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)


def run(closes, score_df, spy_fwd, rebal, enter, exit_, smooth):
    sdf = score_df.rolling(smooth).mean() if smooth > 1 else score_df
    fwd = closes.shift(-rebal) / closes - 1
    dates = closes.index
    prev_long, prev_short = [], []
    rows = []
    for p in range(252, len(dates) - rebal, rebal):
        sc = sdf.iloc[p].dropna()
        fz = fwd.iloc[p]
        sc = sc.loc[sc.index[fz.reindex(sc.index).notna()]]
        if len(sc) < 25:
            continue
        q = max(3, len(sc) // 5)
        pct = sc.rank(pct=True)
        if enter == exit_:                       # plain quintile (no buffer)
            ranked = sc.sort_values()
            longs, shorts = list(ranked.index[-q:]), list(ranked.index[:q])
        else:                                    # hysteresis buffer
            keep_l = [x for x in prev_long if x in pct.index and pct[x] >= 1 - exit_]
            add_l = [x for x in pct.sort_values(ascending=False).index
                     if pct[x] >= 1 - enter and x not in keep_l]
            longs = (keep_l + add_l)[:q]
            keep_s = [x for x in prev_short if x in pct.index and pct[x] <= exit_]
            add_s = [x for x in pct.sort_values().index
                     if pct[x] <= enter and x not in keep_s]
            shorts = (keep_s + add_s)[:q]
        lr, sr = float(fz[longs].mean()), float(fz[shorts].mean())
        turn = (len(set(longs) ^ set(prev_long)) + len(set(shorts) ^ set(prev_short))) / (2 * q) if prev_long else 1.0
        prev_long, prev_short = longs, shorts
        spy_f = float(spy_fwd.iloc[p]) if not np.isnan(spy_fwd.iloc[p]) else np.nan
        rows.append(dict(date=dates[p], spread=lr - sr, spy=spy_f, turn=turn,
                         net=(lr - sr) - turn * COST_RT))
    return pd.DataFrame(rows), 252 / rebal


def sharpe(x, ppy):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 5 or x.std(ddof=1) == 0:
        return float("nan"), float("nan")
    return (x.mean() / x.std(ddof=1)) * np.sqrt(ppy), ((1 + x.mean()) ** ppy - 1) * 100


def beta(df):
    x, y = df["spy"].values, df["spread"].values
    m = ~(np.isnan(x) | np.isnan(y))
    return np.polyfit(x[m], y[m], 1)[0] if m.sum() > 5 else float("nan")


def main():
    from stock_symbols_1243 import STOCK_SYMBOLS
    cands = list(dict.fromkeys(STOCK_SYMBOLS))[:N_CANDIDATES]
    print(f"downloading {len(cands)} candidates + {BENCHMARK} ...", flush=True)
    data = {}
    for i in range(0, len(cands), 110):
        data.update(fetch_batch(cands[i:i + 110], "20y", "1d"))
    spy = fetch_batch([BENCHMARK], "20y", "1d").get(BENCHMARK)

    keep = {s: d for s, d in data.items() if len(d) >= MIN_BARS and float(d["Close"].iloc[-1]) > 5}
    closes = pd.DataFrame({s: d["Close"].astype(float) for s, d in keep.items()})
    print(f"universe: {closes.shape[1]} names")

    ret5 = closes.pct_change(5)
    mom = closes.shift(21) / closes.shift(252) - 1
    trendr = closes / closes.rolling(200).mean() - 1
    rsi2 = closes.apply(lambda c: rsi(c, 2))
    score = zrow(-ret5) + zrow(mom) + zrow(trendr) + zrow(-rsi2)
    spy_close = spy["Close"].astype(float).reindex(closes.index)
    spy_fwd_w = spy_close.shift(-5) / spy_close - 1   # placeholder; recomputed per rebal

    configs = [
        ("weekly base",            dict(rebal=5,  enter=0.2, exit_=0.2, smooth=1)),
        ("weekly +buffer",         dict(rebal=5,  enter=0.2, exit_=0.4, smooth=1)),
        ("weekly +smooth5",        dict(rebal=5,  enter=0.2, exit_=0.2, smooth=5)),
        ("weekly +buffer+smooth5", dict(rebal=5,  enter=0.2, exit_=0.4, smooth=5)),
        ("biweekly +buffer",       dict(rebal=10, enter=0.2, exit_=0.4, smooth=1)),
        ("monthly base",           dict(rebal=21, enter=0.2, exit_=0.2, smooth=1)),
        ("monthly +buffer",        dict(rebal=21, enter=0.2, exit_=0.4, smooth=1)),
    ]

    print(f"\n{'config':<24}{'turn%':>7}{'  | ':<2}"
          f"{'trGrSh':>8}{'trNetSh':>8}{'teGrSh':>8}{'teNetSh':>8}{'teNet%':>8}{'teBeta':>8}")
    for name, cfg in configs:
        spy_fwd = spy_close.shift(-cfg["rebal"]) / spy_close - 1
        R, ppy = run(closes, score, spy_fwd, **cfg)
        tr, te = R[R["date"] < SPLIT], R[R["date"] >= SPLIT]
        tr_g, _ = sharpe(tr["spread"], ppy); tr_n, _ = sharpe(tr["net"], ppy)
        te_g, _ = sharpe(te["spread"], ppy); te_n, te_net = sharpe(te["net"], ppy)
        print(f"{name:<24}{R['turn'].mean()*100:>7.0f}{'  | ':<2}"
              f"{tr_g:>8.2f}{tr_n:>8.2f}{te_g:>8.2f}{te_n:>8.2f}{te_net:>8.1f}{beta(te):>8.2f}")
    print("\nGrSh=gross Sharpe, NetSh=after 10bps x turnover. Want: high NET Sharpe in BOTH")
    print("train & test, beta~0. Lower turnover -> less cost drag. (base weekly ~120% turn.)")


if __name__ == "__main__":
    main()
