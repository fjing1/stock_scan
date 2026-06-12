"""Win-rate study for the 寻龙诀 Panel V1 signals — same process as the cycle study.

Tests the panel's actual triggers, detrended vs SPY (excess return), on both an
intraday and a swing ladder, with z-scores and multiple-comparison awareness:

  ENTRIES
    bbuy_long  : green-bar trigger (ema(typ,6) crosses above ema(.,5))  -> long
    red_short  : varr1 crosses under 82 (red bar)                       -> short
    panelbuy   : confluence score first crosses up to >= 5 (panel's rule)-> long

  MULTI-TF ALIGNMENT gradient (k = # higher TFs that agree), two definitions:
    trend  : higher-TF panel trend agrees (long: rising & >0 ; short: falling)
    signal : higher-TF fired the same trigger within recent R bars

  CONFLUENCE (single-TF): bbuy_long entries bucketed by the entry-TF 0-10 score.

Win (detrended): long = beats SPY over the window (excess>0); short = lags SPY.
Higher TFs aligned with a one-bar shift before ffill (only closed bars read).
Signals are cooldown-deduped; the ALL-aligned / score rows carry a z vs baseline.

    python xunlong_winrate.py                 # both ladders
    python xunlong_winrate.py --ladder swing
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xunlong_panel import compute_panel, panel_score  # noqa: E402

import yfinance as yf  # noqa: E402

INTRADAY_BASKET = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META",
                   "GOOGL", "TSLA", "AMD", "AVGO", "NFLX", "IWM", "XLK", "JPM"]
SWING_BASKET = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "JPM",
                "XOM", "JNJ", "KO", "WMT", "CAT", "IWM", "DIA"]
BENCHMARK = "SPY"

LADDERS = {
    "intraday": {
        "entry": ("5m", "60d"),
        "higher": [("15m", "60d"), ("30m", "60d"), ("60m", "60d")],
        "horizons": [6, 12, 24, 78], "horizon_labels": ["30m", "1h", "2h", "1d"],
        "cooldown": 78, "recent_R": 4, "basket": INTRADAY_BASKET,
    },
    "swing": {
        "entry": ("1d", "15y"),
        "higher": [("1wk", "15y"), ("1mo", "20y")],
        "horizons": [3, 5, 10, 21], "horizon_labels": ["3d", "1w", "2w", "1mo"],
        "cooldown": 5, "recent_R": 3, "basket": SWING_BASKET,
    },
}


# --------------------------------------------------------------------------- #
# Helpers (shared with the cycle study)
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


def panel_of(df):
    if df is None or len(df) < 70:
        return None
    try:
        return compute_panel(df)
    except Exception:
        return None


def align(series, entry_index):
    return series.astype(float).shift(1).reindex(entry_index, method="ffill")


def fwd_returns(close, horizons, labels):
    out = pd.DataFrame(index=close.index)
    for h, lab in zip(horizons, labels):
        out[lab] = close.shift(-h) / close - 1.0
    return out


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
# Per-symbol bar table
# --------------------------------------------------------------------------- #
def build_bars(edf, higher_dfs, spy_fwd, cfg):
    panel = panel_of(edf)
    if panel is None:
        return None
    score = panel_score(panel)
    t = panel["trend"]

    close = edf["Close"].astype(float)
    abs_fwd = fwd_returns(close, cfg["horizons"], cfg["horizon_labels"])
    exc_fwd = (abs_fwd.subtract(spy_fwd.reindex(edf.index))
               if spy_fwd is not None else None)

    R = cfg["recent_R"]
    n_tl = pd.Series(0.0, index=edf.index)   # trend agrees, long
    n_ts = pd.Series(0.0, index=edf.index)   # trend agrees, short
    n_sl = pd.Series(0.0, index=edf.index)   # same trigger recent, long
    n_ss = pd.Series(0.0, index=edf.index)   # same trigger recent, short
    for hdf in higher_dfs:
        ph = panel_of(hdf)
        if ph is None:
            continue
        th = ph["trend"]
        tl = ((th > th.shift(1)) & (th.shift(1) > th.shift(2)) & (th > 0)).astype(float)
        tshort = ((th < th.shift(1)) & (th.shift(1) < th.shift(2))).astype(float)
        sl = (ph["bbuy"].astype(float).rolling(R, min_periods=1).max() > 0).astype(float)
        ss = (ph["red"].astype(float).rolling(R, min_periods=1).max() > 0).astype(float)
        n_tl += (align(tl, edf.index) > 0.5).astype(float)
        n_ts += (align(tshort, edf.index) > 0.5).astype(float)
        n_sl += (align(sl, edf.index) > 0.5).astype(float)
        n_ss += (align(ss, edf.index) > 0.5).astype(float)

    score_ge5 = score >= 5
    panelbuy = score_ge5 & ~score_ge5.shift(1, fill_value=False)

    bars = pd.DataFrame({
        "bbuy": panel["bbuy"].to_numpy(),
        "red": panel["red"].to_numpy(),
        "panelbuy": panelbuy.to_numpy(),
        "score": score.to_numpy().astype(int),
        "n_tl": n_tl.to_numpy().astype(int),
        "n_ts": n_ts.to_numpy().astype(int),
        "n_sl": n_sl.to_numpy().astype(int),
        "n_ss": n_ss.to_numpy().astype(int),
        "_pos": np.arange(len(edf)),
    }, index=edf.index)
    for lab in cfg["horizon_labels"]:
        bars["abs_" + lab] = abs_fwd[lab].to_numpy()
        bars["exc_" + lab] = (exc_fwd[lab].to_numpy() if exc_fwd is not None
                              else np.nan)
    return bars


# --------------------------------------------------------------------------- #
# Reporting blocks
# --------------------------------------------------------------------------- #
def _baseline(all_bars, labels, positive):
    pool = {l: [] for l in labels}
    for sym, bars in all_bars.items():
        if sym == BENCHMARK:
            continue
        for l in labels:
            pool[l].append(bars["exc_" + l].to_numpy())
    base = {l: np.concatenate(pool[l]) if pool[l] else np.array([]) for l in labels}
    return {l: winrate(base[l], positive)[0] for l in labels}


def _ded_signals(all_bars, sel_fn, cooldown):
    rows = []
    for sym, bars in all_bars.items():
        if sym == BENCHMARK:
            continue
        sub = bars[sel_fn(bars)]
        if len(sub) == 0:
            continue
        keep = set(dedup(sub["_pos"].tolist(), cooldown))
        s = sub[sub["_pos"].isin(keep)].copy()
        s["_sym"] = sym
        rows.append(s)
    return pd.concat(rows, ignore_index=True) if rows else None


def run_entry(all_bars, cfg, entry_name, side, mode):
    """MTF-alignment gradient for one entry signal + alignment definition."""
    labels = cfg["horizon_labels"]
    positive = (side == "long")
    max_k = len(cfg["higher"])
    base_win = _baseline(all_bars, labels, positive)
    cnt_col = {("long", "trend"): "n_tl", ("long", "signal"): "n_sl",
               ("short", "trend"): "n_ts", ("short", "signal"): "n_ss"}[(side, mode)]

    def sel_base(bars):
        return bars[entry_name].astype(bool)

    print(f"\n--- {entry_name:<9} ({side}) | align={mode:<6} | excess-vs-{BENCHMARK} ---")
    print("  cond    " + "".join(f"{l:>9}" for l in labels) + f"{'N':>8}")
    print("  baseline" + "".join(f"{base_win[l]:>9.1f}" for l in labels))

    base_sig = _ded_signals(all_bars, sel_base, cfg["cooldown"])
    if base_sig is None:
        print("  (no signals)")
        return None

    # need per-row alignment count; re-pull from all_bars via merge on (_sym,_pos)
    out_best = None
    for k in range(0, max_k + 1):
        rows = []
        for sym, bars in all_bars.items():
            if sym == BENCHMARK:
                continue
            sub = bars[sel_base(bars) & (bars[cnt_col] >= k)]
            if len(sub) == 0:
                continue
            keep = set(dedup(sub["_pos"].tolist(), cfg["cooldown"]))
            rows.append(sub[sub["_pos"].isin(keep)])
        ded = pd.concat(rows, ignore_index=True) if rows else None
        if ded is None or len(ded) == 0:
            continue
        tag = "entry" if k == 0 else (f">={k}" if k < max_k else "ALL")
        wins, ns = [], []
        for l in labels:
            w, n = winrate(ded["exc_" + l].to_numpy(), positive)
            wins.append(w); ns.append(n)
        line = f"  {tag:<8}" + "".join(f"{w:>9.1f}" for w in wins) + f"{max(ns):>8}"
        if k == max_k:
            zs = [zscore(w, base_win[l], n) for w, l, n in zip(wins, labels, ns)]
            line += "  z:" + "".join(f"{z:>+6.1f}" for z in zs)
            out_best = (entry_name, side, mode, wins, base_win, zs, max(ns))
        print(line)
    return out_best


def run_confluence(all_bars, cfg):
    """bbuy long entries bucketed by entry-TF confluence score (single TF)."""
    labels = cfg["horizon_labels"]
    base_win = _baseline(all_bars, labels, True)
    print(f"\n--- confluence: bbuy_long by entry-TF score>=k | excess-vs-{BENCHMARK} ---")
    print("  cond    " + "".join(f"{l:>9}" for l in labels) + f"{'N':>8}")
    print("  baseline" + "".join(f"{base_win[l]:>9.1f}" for l in labels))
    results = []
    for k in [3, 5, 7, 9]:
        rows = []
        for sym, bars in all_bars.items():
            if sym == BENCHMARK:
                continue
            sub = bars[bars["bbuy"].astype(bool) & (bars["score"] >= k)]
            if len(sub) == 0:
                continue
            keep = set(dedup(sub["_pos"].tolist(), cfg["cooldown"]))
            rows.append(sub[sub["_pos"].isin(keep)])
        ded = pd.concat(rows, ignore_index=True) if rows else None
        if ded is None or len(ded) == 0:
            print(f"  score>={k:<2}  (no signals)")
            continue
        wins, ns = [], []
        for l in labels:
            w, n = winrate(ded["exc_" + l].to_numpy(), True)
            wins.append(w); ns.append(n)
        zs = [zscore(w, base_win[l], n) for w, l, n in zip(wins, labels, ns)]
        line = (f"  score>={k:<2}" + "".join(f"{w:>9.1f}" for w in wins)
                + f"{max(ns):>8}" + "  z:" + "".join(f"{z:>+6.1f}" for z in zs))
        print(line)
        results.append((f"score>={k}", "long", "conf", wins, base_win, zs, max(ns)))
    return results


def run_ladder(name, cfg):
    basket = cfg["basket"]
    entry_iv, entry_per = cfg["entry"]
    print(f"\n{'='*86}\nLADDER: {name}   entry={entry_iv} ({entry_per})   "
          f"higher={[t for t,_ in cfg['higher']]}   recent_R={cfg['recent_R']}")
    print(f"basket ({len(basket)}): {' '.join(basket)}  | benchmark={BENCHMARK}")
    print("downloading ...", flush=True)

    entry_data = fetch_batch(basket, entry_iv, entry_per)
    higher_data = [fetch_batch(basket, iv, per) for iv, per in cfg["higher"]]
    bdf = entry_data.get(BENCHMARK)
    spy_fwd = (fwd_returns(bdf["Close"].astype(float), cfg["horizons"],
                           cfg["horizon_labels"]) if bdf is not None else None)

    all_bars = {}
    for s in basket:
        edf = entry_data.get(s)
        if edf is None or len(edf) < 70:
            continue
        bars = build_bars(edf, [hd.get(s) for hd in higher_data], spy_fwd, cfg)
        if bars is not None:
            all_bars[s] = bars
    print(f"symbols with data: {len(all_bars)}")

    summary = []
    for entry_name, side in (("bbuy", "long"), ("red", "short"), ("panelbuy", "long")):
        for mode in ("trend", "signal"):
            b = run_entry(all_bars, cfg, entry_name, side, mode)
            if b:
                summary.append(b)
    summary += run_confluence(all_bars, cfg)

    print(f"\n  SIGNIFICANT detrended edges (|z|>=2) for {name}:")
    labels = cfg["horizon_labels"]
    found = False
    for name_, side, mode, wins, basew, zs, n in summary:
        for l, w, z in zip(labels, wins, zs):
            if not math.isnan(z) and abs(z) >= 2.0:
                found = True
                print(f"    {name_:<9} {side:<5} {mode:<6} {l:>4}: "
                      f"win {w:.1f}% vs base {basew[l]:.1f}%  "
                      f"(edge {w-basew[l]:+.1f}, z={z:+.1f}, N={n})")
    if not found:
        print("    none — nothing beats its detrended baseline at 2 sigma.")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ladder", choices=["intraday", "swing", "both"], default="both")
    args = p.parse_args(argv)
    names = ["intraday", "swing"] if args.ladder == "both" else [args.ladder]
    for n in names:
        run_ladder(n, LADDERS[n])


if __name__ == "__main__":
    main()
