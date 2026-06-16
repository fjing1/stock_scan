"""Multi-timeframe cycle turning-point win-rate study.

Tests the hypothesis: when the "cycle" line (cycleh from cycle_patter_for_swing)
turns up on several timeframes at once, does the forward win rate improve versus
fewer timeframes aligned?

Two ladders are tested because Yahoo Finance only serves ~60 days of intraday
history (5m/15m/30m/60m) while daily/weekly/monthly go back 15-20+ years -- so
"5m + weekly together" is untestable (≈0 overlapping turns). Instead:

    intraday : entry 5m  + higher [15m, 30m, 60m]   (60-day window)
    swing    : entry 1d  + higher [1wk, 1mo]         (15-20y window)

Method (no lookahead):
  * Cycle turning point (bull) at bar t: cycle made a local trough at t-1 and
    ticked up at t -> confirmed at close of t. Optional oversold gate on the
    trough value (swing-bottom flavour).
  * Higher-TF "regime" = sign of its most recent confirmed pivot (+1 up-swing).
    Aligned onto the entry timeline by SHIFTING the higher TF one bar before
    forward-filling, so only fully-closed higher-TF bars are ever used.
  * Entry at the entry-bar close; forward return measured over H entry-bars.
  * Signals deduped with a cooldown so forward windows don't overlap.
  * Reported against the unconditional baseline win rate (edge = cond - base).

Usage:
    python cycle_mtf_winrate.py                # both ladders, default baskets
    python cycle_mtf_winrate.py --ladder swing
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cycle_patter_for_swing import compute_cycle_stoch  # noqa: E402

import yfinance as yf  # noqa: E402


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
INTRADAY_BASKET = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META",
                   "GOOGL", "TSLA", "AMD", "AVGO", "NFLX", "IWM", "XLK", "JPM"]
SWING_BASKET = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "JPM",
                "XOM", "JNJ", "KO", "WMT", "CAT", "IWM", "DIA"]

LADDERS = {
    "intraday": {
        "entry": ("5m", "60d"),
        "higher": [("15m", "60d"), ("30m", "60d"), ("60m", "60d")],
        "horizons": [6, 12, 24, 78],   # 5m bars (78 ≈ one trading day)
        "horizon_labels": ["30m", "1h", "2h", "1d"],
        "oversold": 40.0,
        "cooldown": 78,                # ~1 day between independent samples
        "basket": INTRADAY_BASKET,
    },
    "swing": {
        "entry": ("1d", "15y"),
        "higher": [("1wk", "15y"), ("1mo", "20y")],
        "horizons": [3, 5, 10, 21],    # trading days
        "horizon_labels": ["3d", "1w", "2w", "1mo"],
        "oversold": 40.0,
        "cooldown": 5,
        "basket": SWING_BASKET,
    },
}


# --------------------------------------------------------------------------- #
# Data / indicator
# --------------------------------------------------------------------------- #
def fetch_batch(symbols, interval, period):
    """Download a basket for one interval; return {symbol: OHLCV DataFrame}."""
    raw = yf.download(symbols, interval=interval, period=period,
                      auto_adjust=False, progress=False, group_by="ticker",
                      threads=True)
    out = {}
    for s in symbols:
        try:
            d = raw[s].dropna(how="all").copy()
        except Exception:
            continue
        if len(d) == 0:
            continue
        if getattr(d.index, "tz", None) is not None:
            d.index = d.index.tz_localize(None)
        out[s] = d
    return out


def cycle_of(df):
    """Cycle line for a df, or None if too short."""
    if df is None or len(df) < 20:
        return None
    try:
        return compute_cycle_stoch(df)["cycle"]
    except Exception:
        return None


def pivots(cycle):
    """Return (bull_turn bool, regime +/-1 ffilled, trough_value) for a cycle."""
    c = cycle.astype(float)
    trough = (c.shift(1) < c.shift(2)) & (c > c.shift(1))
    peak = (c.shift(1) > c.shift(2)) & (c < c.shift(1))
    pivot_dir = pd.Series(np.where(trough, 1.0, np.where(peak, -1.0, np.nan)),
                          index=c.index)
    regime = pivot_dir.ffill()
    return trough.fillna(False), regime, c.shift(1)


def align_regime(regime_h, entry_index):
    """Higher-TF regime on the entry timeline, using only closed higher bars.

    Shift the higher TF by one of its own bars before ffill so a still-forming
    higher-TF bar is never read (strict no-lookahead).
    """
    return regime_h.shift(1).reindex(entry_index, method="ffill")


# --------------------------------------------------------------------------- #
# Per-symbol signal extraction
# --------------------------------------------------------------------------- #
def signals_for_symbol(entry_df, higher_cycles, cfg):
    """Return (signals_df, baseline_fwd) for one symbol.

    signals_df : rows = oversold bull-turn bars, cols = n_up + fwd returns
    baseline_fwd : DataFrame of unconditional forward returns (all bars)
    """
    cyc = cycle_of(entry_df)
    if cyc is None:
        return None, None
    trough, _, trough_val = pivots(cyc)

    close = entry_df["Close"].astype(float)
    close.index = entry_df.index

    # forward returns per horizon (entry-bar close to close +H)
    fwd = pd.DataFrame(index=entry_df.index)
    for h, lab in zip(cfg["horizons"], cfg["horizon_labels"]):
        fwd[lab] = close.shift(-h) / close - 1.0

    # how many higher TFs are in an up-swing at each entry bar
    n_up = pd.Series(0, index=entry_df.index, dtype=int)
    for hc in higher_cycles:
        if hc is None:
            continue
        _, regime_h, _ = pivots(hc)
        aligned = align_regime(regime_h, entry_df.index)
        n_up = n_up.add((aligned == 1.0).astype(int), fill_value=0)

    entry_turn = trough & (trough_val < cfg["oversold"])
    sig = pd.DataFrame({"n_up": n_up.astype(int)}).join(fwd)
    sig = sig[entry_turn.reindex(sig.index, fill_value=False)]
    return sig, fwd


def dedup_positions(idx_positions, cooldown):
    """Greedy thinning so kept signals are >= cooldown bars apart."""
    kept, last = [], -(10 ** 12)
    for p in sorted(idx_positions):
        if p - last >= cooldown:
            kept.append(p)
            last = p
    return kept


# --------------------------------------------------------------------------- #
# Run a ladder
# --------------------------------------------------------------------------- #
def run_ladder(name, cfg):
    basket = cfg["basket"]
    entry_iv, entry_per = cfg["entry"]
    labels = cfg["horizon_labels"]

    print(f"\n{'='*78}\nLADDER: {name}   entry={entry_iv} ({entry_per})   "
          f"higher={[t for t, _ in cfg['higher']]}   oversold<{cfg['oversold']:.0f}")
    print(f"basket ({len(basket)}): {' '.join(basket)}")

    # download every interval once for the whole basket
    print("downloading ...", flush=True)
    entry_data = fetch_batch(basket, entry_iv, entry_per)
    higher_data = [fetch_batch(basket, iv, per) for iv, per in cfg["higher"]]

    all_signals = []          # per-symbol signals tagged with positional index
    base_win = {l: [] for l in labels}   # pooled unconditional returns
    base_mean = {l: [] for l in labels}
    n_symbols = 0

    for s in basket:
        edf = entry_data.get(s)
        if edf is None or len(edf) < 60:
            continue
        higher_cycles = [cycle_of(hd.get(s)) for hd in higher_data]
        sig, fwd = signals_for_symbol(edf, higher_cycles, cfg)
        if sig is None or fwd is None:
            continue
        n_symbols += 1

        # baseline: every bar's forward returns
        for l in labels:
            v = fwd[l].to_numpy()
            v = v[~np.isnan(v)]
            base_win[l].append(v)

        # tag signals with positional index for cooldown dedup
        pos = {ts: i for i, ts in enumerate(edf.index)}
        sig = sig.copy()
        sig["_pos"] = [pos[ts] for ts in sig.index]
        sig["_sym"] = s
        all_signals.append(sig)

    if not all_signals:
        print("no signals / data.")
        return

    sig_all = pd.concat(all_signals, ignore_index=True)
    max_up = len(cfg["higher"])

    # ---- baseline ----
    base_pool = {l: np.concatenate(base_win[l]) if base_win[l] else np.array([])
                 for l in labels}
    print(f"\nsymbols used: {n_symbols}   total oversold cycle-turns: {len(sig_all)}")
    print("\nBASELINE (unconditional, all bars):")
    hdr = "  " + "".join(f"{l:>12}" for l in labels)
    print(hdr)
    print("  win%" + "".join(f"{(p>0).mean()*100:>12.1f}" for p in
                             (base_pool[l] for l in labels)))
    print("  mean%" + "".join(f"{p.mean()*100:>11.2f}" for p in
                              (base_pool[l] for l in labels)))

    # ---- conditional by number of higher TFs aligned up ----
    print(f"\nWIN RATE by # higher TFs in up-swing (need >= k), "
          f"cooldown={cfg['cooldown']} bars:")
    print(f"  k = number of {max_up} higher TFs ({[t for t,_ in cfg['higher']]}) turned up\n")
    print("  cond" + "".join(f"{l:>12}" for l in labels) + f"{'N':>8}{'syms':>6}")

    for k in range(0, max_up + 1):
        sub = sig_all[sig_all["n_up"] >= k]
        if len(sub) == 0:
            continue
        # cooldown dedup per symbol
        keep_rows = []
        for s, g in sub.groupby("_sym"):
            kept = set(dedup_positions(g["_pos"].tolist(), cfg["cooldown"]))
            keep_rows.append(g[g["_pos"].isin(kept)])
        ded = pd.concat(keep_rows, ignore_index=True)

        tag = "entry only" if k == 0 else (f">= {k} up" if k < max_up else "ALL up")
        wins = []
        for l in labels:
            v = ded[l].to_numpy()
            v = v[~np.isnan(v)]
            wins.append((v > 0).mean() * 100 if len(v) else float("nan"))
        line = f"  {tag:<10}" + "".join(f"{w:>12.1f}" for w in wins)
        line += f"{len(ded):>8}{ded['_sym'].nunique():>6}"
        print(line)

    # ---- edge of ALL-aligned vs baseline ----
    print("\nEDGE (ALL-up win% minus baseline win%):")
    sub = sig_all[sig_all["n_up"] >= max_up]
    keep_rows = []
    for s, g in sub.groupby("_sym"):
        kept = set(dedup_positions(g["_pos"].tolist(), cfg["cooldown"]))
        keep_rows.append(g[g["_pos"].isin(kept)])
    ded = pd.concat(keep_rows, ignore_index=True) if keep_rows else sub
    print("  " + "".join(f"{l:>12}" for l in labels))
    edges = []
    for l in labels:
        v = ded[l].to_numpy(); v = v[~np.isnan(v)]
        cw = (v > 0).mean() * 100 if len(v) else float("nan")
        bw = (base_pool[l] > 0).mean() * 100 if len(base_pool[l]) else float("nan")
        edges.append(cw - bw)
    print("  " + "".join(f"{e:>+12.1f}" for e in edges))


def main(argv=None):
    p = argparse.ArgumentParser(description="Multi-TF cycle turning-point win rate")
    p.add_argument("--ladder", choices=["intraday", "swing", "both"],
                   default="both")
    args = p.parse_args(argv)

    names = ["intraday", "swing"] if args.ladder == "both" else [args.ladder]
    for n in names:
        run_ladder(n, LADDERS[n])


if __name__ == "__main__":
    main()
