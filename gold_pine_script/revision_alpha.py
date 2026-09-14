"""Analyst-revision-momentum alpha (first NON-price signal in the project).

Built from yfinance `.upgrades_downgrades` (dated history of analyst rating
changes -> point-in-time backtestable, unlike the .info snapshot). For each stock,
the signal = trailing 63-day net revisions (upgrades −downgrades), defined only
where there has been recent analyst activity. Tested cross-sectionally with the
same weekly + rebalance-buffer machinery as the ensemble.

Questions:
  1. Is revision momentum individually positive out-of-sample (train AND test)?
  2. Is it UNCORRELATED with the price alphas (the point of adding it)?
  3. Does adding it to the 4-alpha ensemble improve OOS net Sharpe?

CAVEATs: current-names universe (survivorship); revision history depth varies by
name (coverage reported at runtime); one API call per ticker (slow).

    python revision_alpha.py
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
N_CANDIDATES = 200
MIN_BARS = 2500
REBAL, ENTER, EXIT = 5, 0.2, 0.4
PPY = 252 / REBAL
REV_WIN = 63              # trailing window (trading days) for net revisions
MIN_EVENTS = 5            # minimum total revision events to include a name


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


def run_alpha(score, closes, spy_fwd):
    fwd = closes.shift(-REBAL) / closes - 1
    dates = closes.index
    prev_l, prev_s = [], []
    rows = []
    for p in range(252, len(dates) - REBAL, REBAL):
        sc = score.iloc[p].dropna()
        fz = fwd.iloc[p]
        sc = sc.loc[sc.index[fz.reindex(sc.index).notna()]]
        if len(sc) < 20:
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
                beta=bta, turn=R["turn"].mean() * 100, ntr=len(tr), nte=len(te))


def build_revision_signal(symbols, calendar):
    """Wide DataFrame (calendar x names) of trailing net revisions; NaN where no
    recent analyst activity. Returns (signal_df, coverage_count, earliest_date)."""
    act_map = {"up": 1.0, "down": -1.0}
    cols = {}
    covered, earliest = 0, None
    for i, s in enumerate(symbols):
        try:
            ud = yf.Ticker(s).upgrades_downgrades
        except Exception:
            ud = None
        if ud is None or len(ud) < MIN_EVENTS or "Action" not in ud.columns:
            continue
        idx = ud.index
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        ev = pd.Series(ud["Action"].str.lower().map(act_map).fillna(0.0).values,
                       index=idx.normalize())
        cnt = pd.Series(1.0, index=idx.normalize())
        net_d = ev.groupby(level=0).sum().reindex(calendar, fill_value=0.0)
        cnt_d = cnt.groupby(level=0).sum().reindex(calendar, fill_value=0.0)
        trail_net = net_d.rolling(REV_WIN, min_periods=1).sum()
        trail_cnt = cnt_d.rolling(REV_WIN, min_periods=1).sum()
        sig = trail_net.where(trail_cnt > 0, np.nan)
        cols[s] = sig
        covered += 1
        e = idx.min()
        earliest = e if earliest is None else min(earliest, e)
        if (i + 1) % 40 == 0:
            print(f"   ...revisions {i+1}/{len(symbols)} ({covered} with data)", flush=True)
    return pd.DataFrame(cols).reindex(calendar), covered, earliest


def main():
    import stock_symbols_1243 as ss
    cands = list(dict.fromkeys(ss.STOCK_SYMBOLS))[:N_CANDIDATES]
    print(f"downloading prices for {len(cands)} + {BENCHMARK} ...", flush=True)
    data = {}
    for i in range(0, len(cands), 100):
        data.update(fetch_batch(cands[i:i + 100], "20y", "1d"))
    spy = fetch_batch([BENCHMARK], "20y", "1d").get(BENCHMARK)
    keep = {s: d for s, d in data.items() if len(d) >= MIN_BARS and float(d["Close"].iloc[-1]) > 5}
    closes = pd.DataFrame({s: d["Close"].astype(float) for s, d in keep.items()})
    print(f"price universe: {closes.shape[1]} names")

    print("fetching analyst revisions (one call per ticker; slow) ...", flush=True)
    rev, covered, earliest = build_revision_signal(list(closes.columns), closes.index)
    print(f"revision coverage: {covered}/{closes.shape[1]} names, earliest grade "
          f"{earliest.date() if earliest is not None else 'n/a'}")

    spy_close = spy["Close"].astype(float).reindex(closes.index)
    spy_fwd = spy_close.shift(-REBAL) / spy_close - 1

    raw = {
        "rev5": -closes.pct_change(5),
        "mom": closes.shift(21) / closes.shift(252) - 1,
        "trend": closes / closes.rolling(200).mean() - 1,
        "osc": -closes.apply(lambda c: rsi(c, 2)),
        "revision": rev,                                  # the new non-price alpha
    }

    print(f"\n--- standalone (weekly + buffer; net = after 10bps x turnover) ---")
    print(f"{'alpha':<10}{'trNetSh':>9}{'teNetSh':>9}{'teGrSh':>9}{'beta':>7}{'turn%':>7}{'teN':>5}")
    spreads = {}
    for name, r in raw.items():
        R = run_alpha(zrow(r), closes, spy_fwd)
        spreads[name] = R.set_index("date")["spread"]
        m = metrics(R)
        print(f"{name:<10}{m['trNet']:>9.2f}{m['teNet']:>9.2f}{m['teGr']:>9.2f}"
              f"{m['beta']:>7.2f}{m['turn']:>7.0f}{m['nte']:>5}")

    S = pd.DataFrame(spreads).dropna()
    print(f"\n--- spread-return correlation with 'revision' (n={len(S)}) ---")
    if "revision" in S.columns:
        for c in [x for x in S.columns if x != "revision"]:
            print(f"   revision vs {c:<8}: {S['revision'].corr(S[c]):+.2f}")

    combo4 = sum(zrow(raw[k]) for k in ["rev5", "mom", "trend", "osc"])
    combo5 = combo4 + zrow(raw["revision"])
    print(f"\n--- ensemble: does adding revision help? ---")
    print(f"{'ensemble':<22}{'trNetSh':>9}{'teNetSh':>9}{'teGrSh':>9}{'beta':>7}")
    for label, sc in (("4-alpha (price only)", combo4), ("5-alpha (+revision)", combo5)):
        m = metrics(run_alpha(sc, closes, spy_fwd))
        print(f"{label:<22}{m['trNet']:>9.2f}{m['teNet']:>9.2f}{m['teGr']:>9.2f}{m['beta']:>7.2f}")
    print("\nWin = revision standalone net Sharpe >0 (train&test), LOW correlation to price")
    print("alphas, and 5-alpha net Sharpe > 4-alpha. Else: honest negative / coverage-limited.")


if __name__ == "__main__":
    main()
