"""Multi-alpha stack + sector-neutralization for the market-neutral ensemble.

The single market-neutral ensemble nets ~0.5 Sharpe. Elite books reach higher by
COMBINING many weak, low-correlation alphas (diversification raises Sharpe ~ sqrt of
the number of uncorrelated bets). This study:

  1. Defines 8 cross-sectional alphas from distinct anomaly families.
  2. Reports each one's standalone net Sharpe + market beta (weekly, buffer config).
  3. Prints the correlation matrix of their spread returns (uncorrelated = good).
  4. Combines them (equal-weight z-sum) and shows the Sharpe lift vs the best single
     and vs the original 4-alpha blend.
  5. Repeats the combined run SECTOR-NEUTRAL (demean each signal within sector) to
     remove accidental sector bets (#2).

Alphas (sign = higher -> expected outperform):
  rev5   -5d return            (short-term reversal)
  rev21  -21d return           (1-month reversal)
  mom    12-1 month return     (momentum / relative strength)
  trend  close/SMA200-1        (trend)
  osc    -RSI(2)               (oversold)
  lowvol -stdev(ret,20)        (low-volatility anomaly)
  maxret -max(daily ret,20)    (lottery / MAX effect — short high-max names)
  hi52   close/252-day-high    (52-week-high momentum, George-Hwang)

CAVEAT: still a current-names (survivorship) universe — see notes at end.

    python alpha_stack.py
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
REBAL, ENTER, EXIT = 5, 0.2, 0.4          # weekly + buffer (winning config)
PPY = 252 / REBAL


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


def sector_demean(df, groups):
    mean_map = {}
    for cols in groups.values():
        cols = [c for c in dict.fromkeys(cols) if c in df.columns]
        if len(cols) >= 2:
            sm = df[cols].mean(axis=1)
            for c in cols:
                mean_map[c] = sm
    glob = df.mean(axis=1)
    for c in df.columns:
        mean_map.setdefault(c, glob)
    means = pd.DataFrame(mean_map)[df.columns]
    return df.sub(means)


def run_alpha(score, closes, spy_fwd):
    fwd = closes.shift(-REBAL) / closes - 1
    dates = closes.index
    prev_l, prev_s = [], []
    rows = []
    for p in range(252, len(dates) - REBAL, REBAL):
        sc = score.iloc[p].dropna()
        fz = fwd.iloc[p]
        sc = sc.loc[sc.index[fz.reindex(sc.index).notna()]]
        if len(sc) < 25:
            continue
        q = max(3, len(sc) // 5)
        pct = sc.rank(pct=True)
        keep_l = [x for x in prev_l if x in pct.index and pct[x] >= 1 - EXIT]
        add_l = [x for x in pct.sort_values(ascending=False).index if pct[x] >= 1 - ENTER and x not in keep_l]
        longs = (keep_l + add_l)[:q]
        keep_s = [x for x in prev_s if x in pct.index and pct[x] <= EXIT]
        add_s = [x for x in pct.sort_values().index if pct[x] <= ENTER and x not in keep_s]
        shorts = (keep_s + add_s)[:q]
        lr, sr = float(fz[longs].mean()), float(fz[shorts].mean())
        turn = (len(set(longs) ^ set(prev_l)) + len(set(shorts) ^ set(prev_s))) / (2 * q) if prev_l else 1.0
        prev_l, prev_s = longs, shorts
        spy_f = float(spy_fwd.iloc[p]) if not np.isnan(spy_fwd.iloc[p]) else np.nan
        rows.append((dates[p], lr - sr, spy_f, (lr - sr) - turn * COST_RT, turn))
    return pd.DataFrame(rows, columns=["date", "spread", "spy", "net", "turn"])


def metrics(R):
    tr, te = R[R["date"] < SPLIT], R[R["date"] >= SPLIT]

    def sh(x):
        x = np.asarray(x, float); x = x[~np.isnan(x)]
        return (x.mean() / x.std(ddof=1)) * np.sqrt(PPY) if len(x) > 5 and x.std(ddof=1) > 0 else float("nan")
    x, y = te["spy"].values, te["spread"].values
    m = ~(np.isnan(x) | np.isnan(y))
    bta = np.polyfit(x[m], y[m], 1)[0] if m.sum() > 5 else float("nan")
    return dict(trNet=sh(tr["net"]), teNet=sh(te["net"]), teGr=sh(te["spread"]),
                beta=bta, turn=R["turn"].mean() * 100)


def main():
    import stock_symbols_1243 as ss
    cands = list(dict.fromkeys(ss.STOCK_SYMBOLS))[:N_CANDIDATES]
    sector_lists = {
        "Tech": ss.TECH_STOCKS, "Health": ss.HEALTHCARE_STOCKS,
        "Fin": ss.FINANCIAL_STOCKS, "Disc": ss.CONSUMER_DISCRETIONARY,
        "Staples": ss.CONSUMER_STAPLES, "Energy": ss.ENERGY_STOCKS,
        "Indu": ss.MATERIALS_INDUSTRIALS, "RE": ss.REAL_ESTATE_REITS,
        "Util": ss.UTILITIES, "Comm": ss.COMMUNICATION_SERVICES,
    }
    print(f"downloading {len(cands)} + {BENCHMARK} ...", flush=True)
    data = {}
    for i in range(0, len(cands), 110):
        data.update(fetch_batch(cands[i:i + 110], "20y", "1d"))
    spy = fetch_batch([BENCHMARK], "20y", "1d").get(BENCHMARK)
    keep = {s: d for s, d in data.items() if len(d) >= MIN_BARS and float(d["Close"].iloc[-1]) > 5}
    closes = pd.DataFrame({s: d["Close"].astype(float) for s, d in keep.items()})
    highs = pd.DataFrame({s: d["High"].astype(float) for s, d in keep.items()})
    lows = pd.DataFrame({s: d["Low"].astype(float) for s, d in keep.items()})
    rets = closes.pct_change()
    groups = {k: [s for s in v if s in closes.columns] for k, v in sector_lists.items()}
    print(f"universe: {closes.shape[1]} names across {sum(len(v)>0 for v in groups.values())} sectors")

    raw = {
        "rev5": -closes.pct_change(5),
        "rev21": -closes.pct_change(21),
        "mom": closes.shift(21) / closes.shift(252) - 1,
        "trend": closes / closes.rolling(200).mean() - 1,
        "osc": -closes.apply(lambda c: rsi(c, 2)),
        "lowvol": -rets.rolling(20).std(),
        "maxret": -rets.rolling(20).max(),
        "hi52": closes / closes.rolling(252).max(),
    }
    spy_close = spy["Close"].astype(float).reindex(closes.index)
    spy_fwd = spy_close.shift(-REBAL) / spy_close - 1

    # standalone
    print(f"\n--- standalone alphas (weekly + buffer; net = after 10bps x turnover) ---")
    print(f"{'alpha':<8}{'turn%':>7}{'trNetSh':>9}{'teNetSh':>9}{'teGrSh':>9}{'beta':>7}")
    spreads = {}
    meta = {}
    for name, r in raw.items():
        R = run_alpha(zrow(r), closes, spy_fwd)
        spreads[name] = R.set_index("date")["spread"]
        m = metrics(R)
        meta[name] = m
        print(f"{name:<8}{m['turn']:>7.0f}{m['trNet']:>9.2f}{m['teNet']:>9.2f}{m['teGr']:>9.2f}{m['beta']:>7.2f}")

    # correlation of spread returns (full sample)
    S = pd.DataFrame(spreads).dropna()
    print(f"\n--- alpha spread-return correlations (n={len(S)} periods) ---")
    corr = S.corr()
    print("        " + "".join(f"{c:>7}" for c in corr.columns))
    for r in corr.index:
        print(f"{r:<8}" + "".join(f"{corr.loc[r, c]:>7.2f}" for c in corr.columns))
    avg_off = (corr.values[np.triu_indices_from(corr.values, 1)]).mean()
    print(f"avg pairwise correlation: {avg_off:.2f}  (lower -> more diversification)")

    # combined
    combo = sum(zrow(r) for r in raw.values())
    combo4 = zrow(raw["rev5"]) + zrow(raw["mom"]) + zrow(raw["trend"]) + zrow(raw["osc"])
    combo_sn = sum(zrow(sector_demean(r, groups)) for r in raw.values())
    # train-Sharpe-weighted: only alphas positive in TRAIN, weighted by train Sharpe
    w = {n: max(0.0, meta[n]["trNet"]) for n in raw}
    combo_smart = sum(w[n] * zrow(raw[n]) for n in raw)
    kept = [n for n in raw if w[n] > 0]
    print(f"\n--- combined ensembles ---")
    print(f"train-Sharpe-weighted keeps: {', '.join(kept)}")
    print(f"{'ensemble':<24}{'turn%':>7}{'trNetSh':>9}{'teNetSh':>9}{'teGrSh':>9}{'beta':>7}")
    for label, sc in (("original 4-alpha", combo4), ("8-alpha equal", combo),
                      ("8-alpha trainSh-weighted", combo_smart),
                      ("8-alpha sector-neutral", combo_sn)):
        m = metrics(run_alpha(sc, closes, spy_fwd))
        print(f"{label:<24}{m['turn']:>7.0f}{m['trNet']:>9.2f}{m['teNet']:>9.2f}{m['teGr']:>9.2f}{m['beta']:>7.2f}")
    print("\nDiversification works if the 8-alpha net Sharpe > best single AND > 4-alpha,")
    print("with avg correlation low. Sector-neutral should keep Sharpe with cleaner beta.")
    print("CAVEAT (#3 survivorship): universe = CURRENT names. Long-short partially cancels")
    print("it (both legs same biased pool) and beta~0/OOS-consistency are robust reads, but")
    print("magnitudes stay optimistic. A true fix needs point-in-time constituents (CRSP/")
    print("Sharadar/Norgate) with delisted tickers — not available from Yahoo.")


if __name__ == "__main__":
    main()
