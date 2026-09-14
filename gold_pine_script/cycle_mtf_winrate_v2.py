"""Multi-timeframe cycle turning-point study v2 — detrended, both sides,
deep thresholds, and literal "recent synchronized turn".

Extends cycle_mtf_winrate.py with the three follow-ups:

  1. DETRENDED baseline: forward EXCESS return vs SPY over the same window.
     long win  = symbol beats SPY (excess > 0);  short win = symbol lags SPY.
     This strips broad market drift so we measure timing skill, not beta.
  2. SHORT side + DEEP thresholds: long entries on oversold troughs
     (cycle trough < {40, 20}); short entries on overbought peaks
     (cycle peak > {60, 80}).
  3. RECENT synchronized turn: instead of "higher TF merely in up/down swing",
     require the higher TF to have actually TURNED within the last R of its own
     bars -- a genuine simultaneous turn. Compared against the regime version.

All alignment uses a one-bar shift before forward-fill, so only CLOSED higher
timeframe bars are ever read (no lookahead). Signals are cooldown-deduped per
symbol. The ALL-aligned row carries a z-score vs its own baseline.

Usage:
    python cycle_mtf_winrate_v2.py                 # both ladders, full matrix
    python cycle_mtf_winrate_v2.py --ladder swing
"""
from __future__ import annotations

import argparse
import math
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
BENCHMARK = "SPY"

LADDERS = {
    "intraday": {
        "entry": ("5m", "60d"),
        "higher": [("15m", "60d"), ("30m", "60d"), ("60m", "60d")],
        "horizons": [6, 12, 24, 78],
        "horizon_labels": ["30m", "1h", "2h", "1d"],
        "cooldown": 78,
        "recent_R": 4,           # higher-TF bars counted as a "recent" turn
        "basket": INTRADAY_BASKET,
    },
    "swing": {
        "entry": ("1d", "15y"),
        "higher": [("1wk", "15y"), ("1mo", "20y")],
        "horizons": [3, 5, 10, 21],
        "horizon_labels": ["3d", "1w", "2w", "1mo"],
        "cooldown": 5,
        "recent_R": 3,
        "basket": SWING_BASKET,
    },
}
LONG_THRESHOLDS = [40.0, 20.0]
SHORT_THRESHOLDS = [60.0, 80.0]


# --------------------------------------------------------------------------- #
# Data / indicator
# --------------------------------------------------------------------------- #
def fetch_batch(symbols, interval, period):
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
    if df is None or len(df) < 20:
        return None
    try:
        return compute_cycle_stoch(df)["cycle"]
    except Exception:
        return None


def pivots(cycle):
    """Return (trough bool, peak bool, regime +/-1, prev-cycle value)."""
    c = cycle.astype(float)
    trough = ((c.shift(1) < c.shift(2)) & (c > c.shift(1))).fillna(False)
    peak = ((c.shift(1) > c.shift(2)) & (c < c.shift(1))).fillna(False)
    pivot_dir = pd.Series(np.where(trough, 1.0, np.where(peak, -1.0, np.nan)),
                          index=c.index)
    regime = pivot_dir.ffill()
    return trough, peak, regime, c.shift(1)


def align(series, entry_index):
    """Numeric series -> entry timeline using only closed higher bars."""
    return series.astype(float).shift(1).reindex(entry_index, method="ffill")


def fwd_returns(close, horizons, labels):
    out = pd.DataFrame(index=close.index)
    for h, lab in zip(horizons, labels):
        out[lab] = close.shift(-h) / close - 1.0
    return out


# --------------------------------------------------------------------------- #
# Per-symbol bar table
# --------------------------------------------------------------------------- #
def build_bars(edf, higher_dfs, spy_fwd, cfg):
    """One row per entry bar with turns, higher-TF alignment counts, returns."""
    cyc = cycle_of(edf)
    if cyc is None:
        return None
    trough, peak, _, cprev = pivots(cyc)

    close = edf["Close"].astype(float)
    abs_fwd = fwd_returns(close, cfg["horizons"], cfg["horizon_labels"])
    exc_fwd = abs_fwd.subtract(
        spy_fwd.reindex(edf.index)) if spy_fwd is not None else None

    R = cfg["recent_R"]
    n_up_reg = pd.Series(0.0, index=edf.index)
    n_dn_reg = pd.Series(0.0, index=edf.index)
    n_up_rec = pd.Series(0.0, index=edf.index)
    n_dn_rec = pd.Series(0.0, index=edf.index)
    for hdf in higher_dfs:
        hc = cycle_of(hdf)
        if hc is None:
            continue
        t_h, p_h, reg_h, _ = pivots(hc)
        n_up_reg += (align(reg_h, edf.index) == 1.0).astype(float)
        n_dn_reg += (align(reg_h, edf.index) == -1.0).astype(float)
        up_rec_h = t_h.astype(float).rolling(R, min_periods=1).max()
        dn_rec_h = p_h.astype(float).rolling(R, min_periods=1).max()
        n_up_rec += (align(up_rec_h, edf.index) > 0.5).astype(float)
        n_dn_rec += (align(dn_rec_h, edf.index) > 0.5).astype(float)

    bars = pd.DataFrame({
        "trough": trough.values,
        "peak": peak.values,
        "cprev": cprev.values,
        "n_up_reg": n_up_reg.values.astype(int),
        "n_dn_reg": n_dn_reg.values.astype(int),
        "n_up_rec": n_up_rec.values.astype(int),
        "n_dn_rec": n_dn_rec.values.astype(int),
        "_pos": np.arange(len(edf)),
    }, index=edf.index)
    for lab in cfg["horizon_labels"]:
        bars["abs_" + lab] = abs_fwd[lab].values
        bars["exc_" + lab] = (exc_fwd[lab].values if exc_fwd is not None
                              else np.nan)
    return bars


def dedup(positions, cooldown):
    kept, last = [], -(10 ** 12)
    for p in sorted(positions):
        if p - last >= cooldown:
            kept.append(p)
            last = p
    return kept


def winrate(vals, positive=True):
    v = vals[~np.isnan(vals)]
    if len(v) == 0:
        return float("nan"), 0
    w = (v > 0).mean() if positive else (v < 0).mean()
    return w * 100.0, len(v)


def zscore(p_cond, p_base, n):
    if n == 0 or math.isnan(p_cond) or math.isnan(p_base):
        return float("nan")
    pc, pb = p_cond / 100.0, p_base / 100.0
    se = math.sqrt(pb * (1 - pb) / n)
    return (pc - pb) / se if se > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Run one (side, threshold, mode) block
# --------------------------------------------------------------------------- #
def run_block(all_bars, cfg, side, thr, mode):
    labels = cfg["horizon_labels"]
    max_k = len(cfg["higher"])
    positive = (side == "long")             # win direction for abs & excess
    metric = "exc_"                         # detrended primary
    # column selecting the alignment count
    cnt_col = {("long", "regime"): "n_up_reg", ("long", "recent"): "n_up_rec",
               ("short", "regime"): "n_dn_reg", ("short", "recent"): "n_dn_rec"}[
                   (side, mode)]

    # pooled per-symbol signal rows + baseline pools (excess, benchmark excluded)
    sig_rows, base = [], {l: [] for l in labels}
    for sym, bars in all_bars.items():
        if side == "long":
            sel = bars["trough"] & (bars["cprev"] < thr)
        else:
            sel = bars["peak"] & (bars["cprev"] > thr)
        if sym != BENCHMARK:
            for l in labels:
                base[l].append(bars[metric + l].to_numpy())
        sub = bars[sel].copy()
        if len(sub):
            sub["_sym"] = sym
            sig_rows.append(sub)

    base_pool = {l: np.concatenate(base[l]) if base[l] else np.array([])
                 for l in labels}
    base_win = {l: winrate(base_pool[l], positive)[0] for l in labels}

    print(f"\n--- {side:<5} | cycle {'<' if side=='long' else '>'}{thr:>2.0f} "
          f"| {mode:<6} | metric=excess-vs-{BENCHMARK} ---")
    bl = "  baseline  " + "".join(f"{base_win[l]:>9.1f}" for l in labels)
    print("  cond     " + "".join(f"{l:>9}" for l in labels) + f"{'N':>8}")
    print(bl)

    if not sig_rows:
        print("  (no signals)")
        return None
    sig = pd.concat(sig_rows, ignore_index=True)

    best = None
    for k in range(0, max_k + 1):
        ksub = sig[sig[cnt_col] >= k]
        if len(ksub) == 0:
            continue
        keep = []
        for sym, g in ksub.groupby("_sym"):
            if sym == BENCHMARK:
                continue
            kp = set(dedup(g["_pos"].tolist(), cfg["cooldown"]))
            keep.append(g[g["_pos"].isin(kp)])
        ded = pd.concat(keep, ignore_index=True) if keep else ksub.iloc[:0]
        tag = "entry" if k == 0 else (f">={k}" if k < max_k else "ALL")
        wins, ns = [], []
        for l in labels:
            w, n = winrate(ded[metric + l].to_numpy(), positive)
            wins.append(w); ns.append(n)
        line = f"  {tag:<8}" + "".join(f"{w:>9.1f}" for w in wins)
        line += f"{(max(ns) if ns else 0):>8}"
        if k == max_k:
            zs = [zscore(w, base_win[l], n) for w, l, n in zip(wins, labels, ns)]
            line += "  z:" + "".join(f"{z:>+6.1f}" for z in zs)
            best = (wins, base_win, zs, max(ns) if ns else 0)
        print(line)
    return best


def run_ladder(name, cfg):
    basket = cfg["basket"]
    entry_iv, entry_per = cfg["entry"]
    print(f"\n{'='*84}\nLADDER: {name}   entry={entry_iv} ({entry_per})   "
          f"higher={[t for t,_ in cfg['higher']]}   recent_R={cfg['recent_R']}")
    print(f"basket ({len(basket)}): {' '.join(basket)}  | benchmark={BENCHMARK}")
    print("downloading ...", flush=True)

    entry_data = fetch_batch(basket, entry_iv, entry_per)
    higher_data = [fetch_batch(basket, iv, per) for iv, per in cfg["higher"]]

    # benchmark forward returns on its own entry timeline
    bdf = entry_data.get(BENCHMARK)
    spy_fwd = (fwd_returns(bdf["Close"].astype(float), cfg["horizons"],
                           cfg["horizon_labels"]) if bdf is not None else None)

    all_bars = {}
    for s in basket:
        edf = entry_data.get(s)
        if edf is None or len(edf) < 60:
            continue
        hdfs = [hd.get(s) for hd in higher_data]
        bars = build_bars(edf, hdfs, spy_fwd, cfg)
        if bars is not None:
            all_bars[s] = bars
    print(f"symbols with data: {len(all_bars)}")

    summary = []
    for side, thresholds in (("long", LONG_THRESHOLDS), ("short", SHORT_THRESHOLDS)):
        for thr in thresholds:
            for mode in ("regime", "recent"):
                best = run_block(all_bars, cfg, side, thr, mode)
                if best:
                    wins, basew, zs, n = best
                    summary.append((side, thr, mode, wins, basew, zs, n))

    # headline: ALL-aligned excess edges with |z|>=2
    print(f"\n  SIGNIFICANT ALL-aligned excess edges (|z|>=2) for {name}:")
    labels = cfg["horizon_labels"]
    found = False
    for side, thr, mode, wins, basew, zs, n in summary:
        for l, w, z in zip(labels, wins, zs):
            if not math.isnan(z) and abs(z) >= 2.0:
                found = True
                print(f"    {side:<5} cyc{'<' if side=='long' else '>'}{thr:.0f} "
                      f"{mode:<6} {l:>4}: excess-win {w:.1f}% vs base "
                      f"{basew[l]:.1f}%  (edge {w-basew[l]:+.1f}, z={z:+.1f}, N={n})")
    if not found:
        print("    none — no multi-TF alignment beats its baseline at 2 sigma.")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ladder", choices=["intraday", "swing", "both"],
                   default="both")
    args = p.parse_args(argv)
    names = ["intraday", "swing"] if args.ladder == "both" else [args.ladder]
    for n in names:
        run_ladder(n, LADDERS[n])


if __name__ == "__main__":
    main()
