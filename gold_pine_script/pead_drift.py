"""Post-Earnings-Announcement Drift (PEAD) — the orthogonal, non-OHLCV alpha test.

WHY THIS STUDY (provenance): surfaced by the `research-ideas` agentic workflow and chosen
as the highest-value follow-up because RESEARCH.md #14/#15 concluded the binding constraint
is NEW data orthogonal to price — and PEAD is driven by the EARNINGS SURPRISE, a
fundamental/event signal. Long-documented (Ball & Brown 1968; Bernard & Thomas 1989):
stocks that beat earnings keep drifting UP for weeks; those that miss drift DOWN — the
market under-reacts to the surprise. yfinance exposes realized surprises deep enough
(AAPL to 2005, most liquid names 12-20y) for a real OOS test.

WHAT IT DOES
  1) EVENT STUDY — for every earnings event, measure the detrended (vs-SPY) cumulative
     drift over [+1, +H] trading days AFTER the announcement (entry the first session
     strictly after the announce date, so it is tradable — the initial reaction gap is
     excluded). Bucket by surprise sign and by surprise quintile; report mean drift, N,
     and t-stat for TRAIN (<2019) and TEST (2019->present). PEAD exists if the
     top-surprise bucket drifts up, the bottom drifts down, and the spread persists OOS.
  2) TRADABLE OVERLAY — a weekly cross-sectional long-short: a name is "in play" for W
     trading days after it reports; score = its earnings surprise; long top quintile /
     short bottom quintile of in-play names, dollar-neutral. Reports gross/net (after
     10bps turnover cost) annualized return, Sharpe, %+, beta-to-SPY, and the long-only
     top-quintile alpha vs SPY — i.e. is PEAD a DEPLOYABLE rank overlay for the scanner
     and is it orthogonal enough to STACK onto the #12 ensemble?

HONEST EVAL (house style): OOS split 2019-01-01; detrended vs SPY; t-stats + N reported;
costs modeled. CAVEAT: current-names universe (survivorship) — trust the surprise-quintile
MONOTONICITY, the OOS persistence, and beta~0, not the absolute long-leg magnitude.
Surprise here is analyst EPS surprise % (realized vs consensus), not the stronger
SUE/estimate-revision; treat as a lower bound on the true PEAD effect.

    python pead_drift.py [--names N] [--no-cache]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yfinance as yf  # noqa: E402

BENCHMARK = "SPY"
SPLIT = pd.Timestamp("2019-01-01")
REBAL = 5                 # trading days between rebalances (weekly)
PPY = 252 / REBAL
COST_RT = 0.0010          # round-trip cost (10 bps) on turned-over fraction
N_CANDIDATES = 220
MIN_BARS = 2500           # ~10y history
DRIFT_WINDOWS = [21, 63]  # post-announcement drift horizons (~1mo, ~1qtr)
INPLAY_W = 63             # trading days a name stays "in play" for the overlay (~1 quarter)
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pead_earnings_cache.json")


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


def load_earnings(symbols, use_cache=True):
    """Return {symbol: [(date, surprise_pct), ...]}. Cache to JSON (reruns are instant)."""
    cache = {}
    if use_cache and os.path.exists(CACHE):
        try:
            cache = json.load(open(CACHE))
        except Exception:
            cache = {}
    todo = [s for s in symbols if s not in cache]
    if todo:
        print(f"fetching earnings dates for {len(todo)} names (one-time; cached to {os.path.basename(CACHE)}) ...",
              flush=True)
    for i, s in enumerate(todo):
        rec = []
        try:
            ed = yf.Ticker(s).get_earnings_dates(limit=100)
            if ed is not None and len(ed):
                sur = next((c for c in ed.columns if "Surprise" in c), None)
                for ts, row in ed.iterrows():
                    sp = row.get(sur) if sur else None
                    if sp is None or (isinstance(sp, float) and np.isnan(sp)):
                        continue
                    d = ts.tz_localize(None) if getattr(ts, "tz", None) is not None else ts
                    rec.append([d.strftime("%Y-%m-%d"), float(sp)])
        except Exception:
            rec = []
        cache[s] = rec
        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(todo)}", flush=True)
            json.dump(cache, open(CACHE, "w"))
        time.sleep(0.25)
    if todo:
        json.dump(cache, open(CACHE, "w"))
    return {s: cache.get(s, []) for s in symbols}


def tstat(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


def annualize(series):
    s = np.asarray(series, dtype=float); s = s[~np.isnan(s)]
    if len(s) < 5:
        return dict(n=len(s), ann=float("nan"), sharpe=float("nan"), pos=float("nan"))
    mean, std = s.mean(), s.std(ddof=1)
    return dict(n=len(s), ann=((1 + mean) ** PPY - 1) * 100,
                sharpe=(mean / std) * np.sqrt(PPY) if std > 0 else float("nan"),
                pos=(s > 0).mean() * 100)


# ----------------------------- event study -----------------------------
def event_study(events_by_sym, closes, spy_close):
    """events_by_sym: {sym:[(date,surprise)]}. Build per-event detrended drift for each H."""
    dates = closes.index
    dvals = dates.values
    rows = []
    for sym, evs in events_by_sym.items():
        if sym not in closes.columns:
            continue
        c = closes[sym]
        for dstr, sp in evs:
            ad = np.datetime64(pd.Timestamp(dstr).normalize())
            entry = int(np.searchsorted(dvals, ad, side="right"))   # first session strictly after announce
            if entry < 1 or entry >= len(dates) - max(DRIFT_WINDOWS):
                continue
            c0, s0 = c.iloc[entry], spy_close.iloc[entry]
            if not (np.isfinite(c0) and c0 > 0 and np.isfinite(s0) and s0 > 0):
                continue
            row = dict(sym=sym, date=dates[entry], surprise=sp)
            ok = True
            for H in DRIFT_WINDOWS:
                cH, sH = c.iloc[entry + H], spy_close.iloc[entry + H]
                if not (np.isfinite(cH) and np.isfinite(sH)):
                    ok = False; break
                row[f"car{H}"] = (cH / c0 - 1) - (sH / s0 - 1)   # detrended cumulative drift
            if ok:
                rows.append(row)
    return pd.DataFrame(rows)


def report_event_study(E):
    print("\n================ PEAD EVENT STUDY (detrended drift after the announcement) ================")
    print(f"total tradable events: {len(E)}  ({E['date'].min().date()} -> {E['date'].max().date()})")
    for label, df in [("TRAIN < 2019", E[E["date"] < SPLIT]), ("TEST 2019-present", E[E["date"] >= SPLIT])]:
        print(f"\n  [{label}]  n={len(df)}")
        for H in DRIFT_WINDOWS:
            col = f"car{H}"
            d = df.dropna(subset=[col, "surprise"])
            if len(d) < 30:
                print(f"    +{H}d: too few"); continue
            pos = d[d["surprise"] > 0][col]; neg = d[d["surprise"] < 0][col]
            # quintile spread by surprise
            q = pd.qcut(d["surprise"].rank(method="first"), 5, labels=False)
            topq = d[col][q == 4]; botq = d[col][q == 0]
            spread = topq.mean() - botq.mean()
            print(f"    +{H:>2}d drift: beat(+) {pos.mean()*100:+.2f}% (n{len(pos)}, t{tstat(pos):+.1f}) | "
                  f"miss(-) {neg.mean()*100:+.2f}% (n{len(neg)}, t{tstat(neg):+.1f}) | "
                  f"Q5-Q1 spread {spread*100:+.2f}%  (Q5 {topq.mean()*100:+.2f} / Q1 {botq.mean()*100:+.2f})")


# ----------------------------- tradable overlay -----------------------------
def run_overlay(score, fwd, spy_fwd, dates):
    rebal_idx = range(252, len(dates) - REBAL, REBAL)
    prev_long, prev_short = set(), set()
    rows = []
    for p in rebal_idx:
        sc = score.iloc[p].dropna()
        fz = fwd.iloc[p]
        sc = sc.loc[sc.index[fz.reindex(sc.index).notna().values]]
        if len(sc) < 20:
            continue
        q = max(3, len(sc) // 5)
        ranked = sc.sort_values()
        shorts, longs = list(ranked.index[:q]), list(ranked.index[-q:])
        lr, sr = float(fz[longs].mean()), float(fz[shorts].mean())
        spy_f = float(spy_fwd.iloc[p]) if not np.isnan(spy_fwd.iloc[p]) else np.nan
        turn = (len(set(longs) ^ prev_long) + len(set(shorts) ^ prev_short)) / (2 * q) if prev_long else 1.0
        prev_long, prev_short = set(longs), set(shorts)
        rows.append(dict(date=dates[p], spread=lr - sr, long=lr, short=sr, spy=spy_f,
                         net=(lr - sr) - turn * COST_RT, turn=turn, n=len(sc)))
    return pd.DataFrame(rows)


def report_overlay(name, df):
    if len(df) < 5:
        print(f"  {name}: too few periods"); return
    ls, net = annualize(df["spread"]), annualize(df["net"])
    lo, sp = annualize(df["long"]), annualize(df["spy"])
    loex = annualize(df["long"].values - df["spy"].values)
    x, y = df["spy"].values, df["spread"].values
    m = ~(np.isnan(x) | np.isnan(y))
    beta = np.polyfit(x[m], y[m], 1)[0] if m.sum() > 5 else float("nan")
    print(f"\n  [{name}]  periods={ls['n']}  avg in-play names={df['n'].mean():.0f}  turnover={df['turn'].mean()*100:.0f}%")
    print(f"    Long-Short (alpha)  : ann {ls['ann']:+.1f}%  Sharpe {ls['sharpe']:.2f}  %+ {ls['pos']:.0f}  beta-to-SPY {beta:+.2f}")
    print(f"    Long-Short net costs: ann {net['ann']:+.1f}%  Sharpe {net['sharpe']:.2f}")
    print(f"    Long-only (topQ)    : ann {lo['ann']:+.1f}%  Sharpe {lo['sharpe']:.2f}  %+ {lo['pos']:.0f}")
    print(f"    SPY (same periods)  : ann {sp['ann']:+.1f}%  Sharpe {sp['sharpe']:.2f}")
    print(f"    Long-only ALPHA/SPY : ann {loex['ann']:+.1f}%  Sharpe {loex['sharpe']:.2f}  %+ {loex['pos']:.0f}")


def main():
    import argparse
    global N_CANDIDATES
    ap = argparse.ArgumentParser(description="Post-earnings-announcement drift study")
    ap.add_argument("--names", type=int, default=N_CANDIDATES, help="universe size to consider")
    ap.add_argument("--no-cache", action="store_true", help="ignore the earnings cache and refetch")
    args = ap.parse_args()
    N_CANDIDATES = args.names

    from stock_symbols_1243 import STOCK_SYMBOLS
    cands = [s for s in dict.fromkeys(STOCK_SYMBOLS) if s != BENCHMARK][:N_CANDIDATES]
    print(f"downloading {len(cands)} candidates + {BENCHMARK} (20y daily) ...", flush=True)
    data = {}
    for i in range(0, len(cands), 110):
        data.update(fetch_batch(cands[i:i + 110], "20y", "1d"))
    spy = fetch_batch([BENCHMARK], "20y", "1d").get(BENCHMARK)

    keep = {s: d for s, d in data.items()
            if len(d) >= MIN_BARS and float(d["Close"].iloc[-1]) > 5}
    closes = pd.DataFrame({s: d["Close"].astype(float) for s, d in keep.items()})
    spy_close = spy["Close"].astype(float).reindex(closes.index)
    print(f"universe after history/price filter: {closes.shape[1]} names, "
          f"{closes.index[0].date()} -> {closes.index[-1].date()}")

    earnings = load_earnings(list(closes.columns), use_cache=not args.no_cache)
    n_ev = sum(len(v) for v in earnings.values())
    have = sum(1 for v in earnings.values() if v)
    print(f"earnings history: {n_ev} surprise records across {have}/{closes.shape[1]} names")

    # ---- event study ----
    E = event_study(earnings, closes, spy_close)
    report_event_study(E)

    # ---- tradable overlay: surprise carried forward INPLAY_W trading days ----
    dates = closes.index
    dvals = dates.values
    surprise_mat = pd.DataFrame(np.nan, index=dates, columns=closes.columns)
    for sym, evs in earnings.items():
        if sym not in surprise_mat.columns:
            continue
        for dstr, sp in evs:
            ad = np.datetime64(pd.Timestamp(dstr).normalize())
            entry = int(np.searchsorted(dvals, ad, side="right"))
            if 0 <= entry < len(dates):
                surprise_mat.iat[entry, surprise_mat.columns.get_loc(sym)] = sp
    score = surprise_mat.ffill(limit=INPLAY_W)            # name "in play" for W days post-report
    fwd = closes.shift(-REBAL) / closes - 1
    spy_fwd = spy_close.shift(-REBAL) / spy_close - 1

    R = run_overlay(score, fwd, spy_fwd, dates)
    print("\n================ PEAD TRADABLE OVERLAY (weekly, in-play surprise quintile spread) ================")
    print(f"a name is 'in play' for {INPLAY_W} trading days after it reports; long top / short bottom surprise quintile")
    report_overlay("TRAIN < 2019", R[R["date"] < SPLIT])
    report_overlay("TEST 2019-present (incl. 2022)", R[R["date"] >= SPLIT])

    print("\n================ READS ================")
    print("Event study: PEAD is real here if Q5-Q1 spread is POSITIVE and the +63d > +21d (drift")
    print("accumulates), persisting into TEST. Overlay: beta~0 + positive net Sharpe in BOTH windows")
    print("= a deployable, orthogonal rank overlay worth STACKING onto the #12 ensemble (PEAD is")
    print("event/fundamental, ~uncorrelated to the price alphas). CAVEAT: survivorship + analyst-")
    print("surprise (not SUE) make this a conservative lower bound; trust monotonicity & OOS persistence.")


if __name__ == "__main__":
    main()
