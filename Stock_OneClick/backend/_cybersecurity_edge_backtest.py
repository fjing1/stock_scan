"""
Multi-year, benchmark-adjusted backtest of the daily-computable buy signals
(正式买入 = Gann_BUY_A, 回调买入点 = PULLBACK_BUY), run across every enabled
sector in the scan universe, with a chronological train/test split.

Why this exists: the live-tracked 买入历史记录 sheet only has ~16 closed
cybersecurity trades from a few overlapping weeks, all inside one bull-market
regime. That's too thin and too correlated to trust, and it was picked
*after* looking at ~20 sectors (multiple-comparisons bias). This script:
  1. replays the same daily signal logic over ~10y of history so "n=16" becomes
     however many independent, deduped signal events actually occurred,
  2. benchmark-adjusts every trade against SPY's return over the same window,
  3. splits chronologically (train: before 2023-01-01, test: on/after) and
     checks whether cybersecurity's apparent edge survives out of sample,
  4. is intentionally NOT scoped to only cybersecurity -- every enabled
     sector gets the same treatment so "best sector" isn't cherry-picked twice.

Known data-quality hazard (see 买入历史记录 DD/CRWD rows): this feed
occasionally has un-adjusted spinoff/split jumps. Any per-event forward
return with |ret| > GLITCH_ABS_RET is dropped as a probable data glitch,
and the drop count is reported rather than silently absorbed.
"""
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from xunlong import XunLongIndicator
from scan_stocks import add_pullback_entry, download_daily, load_input_and_meta, INPUT_FILE

HOLD_DAYS = 14          # matches the live tracker's max stage window (阶段收益率)
GLITCH_ABS_RET = 3.0    # |ret| beyond this = probable spinoff/split data glitch
TRAIN_TEST_CUTOFF = pd.Timestamp("2023-01-01")
HISTORY_PERIOD = "10y"
EXCLUDE_GROUPS = {"00 大环境", "01 市场环境"}   # macro/index tickers, not tradeable "sector" bets


def load_universe():
    _, df_meta = load_input_and_meta(INPUT_FILE)
    df_meta = df_meta[df_meta["enable"] == 1].copy()
    df_meta["group"] = df_meta["group"].astype(str)
    df_meta = df_meta[~df_meta["group"].isin(EXCLUDE_GROUPS)]
    return df_meta[["symbol", "name", "group"]].drop_duplicates(subset="symbol")


def compute_symbol_frame(sym):
    df_d = download_daily(sym, period=HISTORY_PERIOD)
    if df_d is None or len(df_d) < 260:
        return None
    if df_d.index[-1].date() == datetime.now().date():
        df_d = df_d.iloc[:-1]     # drop today's in-progress bar
    try:
        xl = XunLongIndicator()
        df_xl = xl.compute(df_d, None)
        df_pb = add_pullback_entry(df_xl)
    except Exception as e:
        print(f"  [{sym}] signal compute failed: {e}", flush=True)
        return None
    close = pd.to_numeric(df_xl["Close"], errors="coerce")
    fwd_ret = close.shift(-HOLD_DAYS) / close - 1.0
    return pd.DataFrame({
        "close": close,
        "fwd_ret": fwd_ret,
        "formal_buy": df_xl["Gann_BUY_A"].fillna(False).astype(bool),
        "pullback_buy": df_pb["PULLBACK_BUY"].fillna(False).astype(bool),
    })


def dedupe_events(dates):
    """Keep an event only if >= HOLD_DAYS trading days after the last kept one
    for this symbol -- a real trader isn't opening 5 overlapping positions in
    the same name in the same 2 weeks."""
    kept = []
    last_pos = -10**9
    for pos, d in dates:
        if pos - last_pos >= HOLD_DAYS:
            kept.append((pos, d))
            last_pos = pos
    return kept


def main():
    universe = load_universe()
    print(f"Universe: {len(universe)} symbols across {universe['group'].nunique()} sectors "
          f"(excluding {sorted(EXCLUDE_GROUPS)})", flush=True)

    print("Fetching SPY benchmark...", flush=True)
    spy_frame = compute_symbol_frame("SPY")
    if spy_frame is None:
        print("FATAL: could not load SPY benchmark"); return
    spy_fwd = spy_frame["fwd_ret"]
    spy_fwd_by_date = {d.normalize(): v for d, v in spy_fwd.items() if pd.notna(v)}

    events = []
    glitch_count = 0
    n_ok, n_skip = 0, 0
    for i, row in enumerate(universe.itertuples(index=False), 1):
        sym, name, group = row.symbol, row.name, row.group
        frame = compute_symbol_frame(sym)
        if frame is None:
            n_skip += 1
            continue
        n_ok += 1
        if i % 20 == 0 or i == len(universe):
            print(f"  [{i}/{len(universe)}] processed ({n_ok} ok, {n_skip} skipped)", flush=True)

        idx = frame.index
        pos_of = {d: p for p, d in enumerate(idx)}
        for sig_col, sig_name in (("formal_buy", "正式买入"), ("pullback_buy", "回调买入点")):
            sig_dates = [(pos_of[d], d) for d in idx[frame[sig_col].to_numpy()]]
            for pos, d in dedupe_events(sig_dates):
                fwd = frame["fwd_ret"].iloc[pos]
                if pd.isna(fwd):
                    continue
                if abs(fwd) > GLITCH_ABS_RET:
                    glitch_count += 1
                    continue
                bench = spy_fwd_by_date.get(pd.Timestamp(d).normalize())
                if bench is None:
                    continue
                events.append({
                    "symbol": sym, "group": group, "signal": sig_name,
                    "date": pd.Timestamp(d), "fwd_ret": fwd, "bench_ret": bench,
                    "excess_ret": fwd - bench,
                })

    ev = pd.DataFrame(events)
    print(f"\nTotal deduped signal events: {len(ev)}  (dropped {glitch_count} as data glitches "
          f"with |ret|>{GLITCH_ABS_RET:.0%})", flush=True)
    if ev.empty:
        print("No events -- aborting."); return

    ev["period"] = np.where(ev["date"] < TRAIN_TEST_CUTOFF, "train", "test")

    def summarize(df, label):
        n = len(df)
        wr = (df["excess_ret"] > 0).mean() if n else np.nan
        mean_raw = df["fwd_ret"].mean() if n else np.nan
        mean_exc = df["excess_ret"].mean() if n else np.nan
        return pd.Series({"n": n, "win_rate_vs_spy": wr, "mean_raw_ret": mean_raw,
                           "mean_excess_ret": mean_exc, "label": label})

    print("\n=== Overall (all sectors, deduped, benchmark-adjusted) ===")
    print(summarize(ev, "ALL"))
    print("\n  by period:")
    for p in ("train", "test"):
        print(f"  {p}: ", summarize(ev[ev["period"] == p], p).to_dict())

    print("\n=== By sector (train period, before 2023-01-01) ===")
    train_by_group = ev[ev["period"] == "train"].groupby("group").apply(
        lambda g: summarize(g, g.name)
    )
    train_by_group = train_by_group[train_by_group["n"] >= 5].sort_values("mean_excess_ret", ascending=False)
    print(train_by_group[["n", "win_rate_vs_spy", "mean_raw_ret", "mean_excess_ret"]].to_string())

    print("\n=== Same sectors, TEST period (on/after 2023-01-01) -- does the train ranking hold up? ===")
    test_by_group = ev[ev["period"] == "test"].groupby("group").apply(
        lambda g: summarize(g, g.name)
    )
    test_aligned = test_by_group.reindex(train_by_group.index)
    print(test_aligned[["n", "win_rate_vs_spy", "mean_raw_ret", "mean_excess_ret"]].to_string())

    print("\n=== Cybersecurity sector detail (train vs test) ===")
    cyber = ev[ev["group"] == "12 网络安全"]
    for p in ("train", "test"):
        sub = cyber[cyber["period"] == p]
        print(f"  {p}: n={len(sub)}", summarize(sub, p).to_dict())
        if len(sub):
            print(sub.groupby("symbol")["excess_ret"].agg(["count", "mean"]).to_string())

    out_path = BASE.parent / "history" / "cybersecurity_edge_backtest_events.csv"
    ev.to_csv(out_path, index=False)
    print(f"\nSaved {len(ev)} raw events to {out_path}")


if __name__ == "__main__":
    main()
