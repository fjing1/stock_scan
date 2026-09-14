#!/usr/bin/env python3
"""Two-stage OBV-RSI divergence system: WATCH the divergence, ALERT on the price breakout.

Stage 1 -- WATCH (potential bull divergence)
    A name is armed when rsi_obv_slope_scan's divergence fires: RSI(OBV)-MA8 trending up
    while the price MA8 trends down (flow improving while price still slides). Being armed
    is NOT an entry -- price is still falling, and on its own this divergence has never
    been shown to pay (see the edge warning below).

Stage 2 -- ALERT (breakout confirmation)
    While armed, watch for price to actually turn up. The trigger is both of:
        a) close crosses UP through the MA8 (was at/below it on the prior bar), and
        b) today's high > the highest high of the previous 2 days.
    Armed names that trigger move to a separate ALERT list; the divergence is what puts a
    name on the radar, the breakout is what fires.

The arming window is --expire bars (default 20): a divergence older than that is stale and
stops arming. The divergence must fire on a bar STRICTLY BEFORE the breakout, so a name can
never be armed and triggered by the same bar.

Everything is recomputed from price history on each run, so this is stateless -- correct
whether you run it daily or once a month, and it backfills missed days via --within. The
only persisted file is trigger_log.csv, an append-only record of every alert so forward
returns can be studied later.

    ../../vcp_env/bin/python rsi_obv_div_watch.py --all
    ../../vcp_env/bin/python rsi_obv_div_watch.py --all --min-rsi-slope 2 --notify
    ../../vcp_env/bin/python rsi_obv_div_watch.py --all --within 3        # catch up 3 days
    ../../vcp_env/bin/python rsi_obv_div_watch.py --symbols GME DECK XPEV

Outputs (Stock_OneClick/tv_rsi_obv_div/):
    tv_rsi_obv_div_watch_<date>.txt   potential bull divergence, waiting for the breakout
    tv_rsi_obv_div_alert_<date>.txt   breakout confirmed -- the actual alert list
    trigger_log.csv                   append-only history of every alert

EDGE WARNING: the pivot form of this divergence had NO forward edge (rsi_obv_divergence.py)
and OBV/price RSI-MA crossovers all lost to buy-and-hold (rsi_obv_ma_compare.py). The
breakout filter is a real, independent confirmation, but the combination is UNBACKTESTED.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
from rsi_obv_slope_scan import (  # noqa: E402
    MIN_BARS, OUT_DIR, add_signal_args, is_illiquid, is_stale, iter_symbol_frames,
    print_filter_stats, reference_last_bar, resolve_partial_cutoff, resolve_universe,
    slope_divergence, trim_partial, write_tv_list,
)

TRIGGER_LOG = OUT_DIR / "trigger_log.csv"


def breakout_signal(df: pd.DataFrame, price_ma: int, allow_above: bool) -> tuple[pd.Series, pd.Series]:
    """Returns (breakout, ma_line).

    breakout = close breaks up through the price MA  AND  today's high exceeds the highest
    high of the previous 2 days. Default requires a true crossover (below/at the MA on the
    prior bar); --allow-above only requires close to sit above it, so a name already above
    the MA can still fire on a fresh 2-day-high thrust."""
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    ma_line = close.rolling(price_ma).mean() if price_ma > 1 else close
    above = close > ma_line
    up_through = above if allow_above else (above & ~above.shift(1).fillna(False))
    higher_high = high > high.shift(1).rolling(2).max()
    return (up_through & higher_high).fillna(False), ma_line


def evaluate(sym: str, df: pd.DataFrame, args, cutoff, ref_date=None) -> dict | None:
    """Classify one symbol on the latest closed bar as ALERT, WATCH, or nothing."""
    if df is None or len(df) < MIN_BARS:
        return None
    df = trim_partial(df.dropna(subset=["Close", "High", "Volume"]), cutoff)
    if len(df) < MIN_BARS or is_stale(df, ref_date, args.max_stale) or is_illiquid(df, args):
        return None

    div, rsi_line, px_line = slope_divergence(df, args.rsi_len, args.rsi_ma, args.price_ma,
                                              args.lookback, args.strict)
    # Magnitude filters applied per-bar, so a weak divergence never arms anything.
    L = args.lookback
    if args.min_rsi_slope > 0:
        div &= (rsi_line - rsi_line.shift(L)) >= args.min_rsi_slope
    if args.min_price_drop > 0:
        div &= (px_line.shift(L) / px_line - 1) >= args.min_price_drop
    if not div.any():
        return None

    brk, ma_line = breakout_signal(df, args.price_ma, args.allow_above)
    # Armed = a divergence fired within the last --expire bars, STRICTLY before this bar.
    armed = div.shift(1).rolling(args.expire, min_periods=1).max().fillna(0).astype(bool)
    trig = armed & brk

    last = len(df) - 1
    c = df["Close"].astype(float)
    h = df["High"].astype(float)
    ma200 = c.rolling(200).mean()
    trend = ("—" if not np.isfinite(ma200.iloc[last])
             else ("above200" if c.iloc[last] > ma200.iloc[last] else "below200"))

    div_pos = np.flatnonzero(div.values)
    trig_pos = np.flatnonzero(trig.values)

    # --- ALERT: breakout confirmed within the --within window
    if len(trig_pos) and last - trig_pos[-1] < args.within:
        t = trig_pos[-1]
        prior_div = div_pos[div_pos < t]
        if not len(prior_div):
            return None
        d = prior_div[-1]
        return {
            "state": "ALERT",
            "symbol": sym,
            "div_date": pd.Timestamp(df.index[d]).date().isoformat(),
            "trigger_date": pd.Timestamp(df.index[t]).date().isoformat(),
            "bars_ago": last - t,
            "lag": t - d,                                        # bars from divergence to breakout
            "close": round(float(c.iloc[t]), 2),
            "ma": round(float(ma_line.iloc[t]), 2),
            "gap_pct": round(float(c.iloc[t] / ma_line.iloc[t] - 1), 4),   # how far above the MA it closed
            "rsi_ma": round(float(rsi_line.iloc[t]), 1),
            "rsi_chg": round(float(rsi_line.iloc[d] - rsi_line.iloc[d - L]), 1),
            "trend": trend,
        }

    # --- WATCH: divergence still fresh, breakout hasn't happened since it fired
    fresh = div_pos[div_pos > last - args.expire]
    if not len(fresh):
        return None
    d = fresh[-1]
    if len(trig_pos) and trig_pos[-1] >= d:      # already fired on this divergence
        return None
    return {
        "state": "WATCH",
        "symbol": sym,
        "div_date": pd.Timestamp(df.index[d]).date().isoformat(),
        "trigger_date": "",
        "bars_ago": last - d,
        "lag": np.nan,
        "close": round(float(c.iloc[last]), 2),
        "ma": round(float(ma_line.iloc[last]), 2),
        # negative = still below the MA; this is the distance price must cover to trigger
        "gap_pct": round(float(c.iloc[last] / ma_line.iloc[last] - 1), 4),
        "need_high": round(float(h.iloc[last - 1:last + 1].max()), 2),   # 2-day high to clear
        "rsi_ma": round(float(rsi_line.iloc[last]), 1),
        "rsi_chg": round(float(rsi_line.iloc[d] - rsi_line.iloc[d - L]), 1),
        "trend": trend,
    }


def append_trigger_log(alerts: pd.DataFrame) -> int:
    """Append new alerts to the append-only log, deduped on (symbol, trigger_date), so the
    same breakout logged twice (e.g. --within 3 re-runs) doesn't duplicate."""
    if alerts.empty:
        return 0
    cols = ["symbol", "div_date", "trigger_date", "lag", "close", "ma", "gap_pct",
            "rsi_ma", "rsi_chg", "trend"]
    new = alerts[cols].copy()
    new["lag"] = new["lag"].astype(int)
    new["logged_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if TRIGGER_LOG.exists():
        old = pd.read_csv(TRIGGER_LOG)
        seen = set(zip(old["symbol"].astype(str), old["trigger_date"].astype(str)))
        new = new[[(s, d) not in seen for s, d in
                   zip(new["symbol"].astype(str), new["trigger_date"].astype(str))]]
        if new.empty:
            return 0
        pd.concat([old, new], ignore_index=True).to_csv(TRIGGER_LOG, index=False)
    else:
        new.to_csv(TRIGGER_LOG, index=False)
    return len(new)


def notify(alerts: pd.DataFrame) -> None:
    """Best-effort macOS notification. Never fails the run."""
    if alerts.empty:
        return
    syms = ", ".join(alerts["symbol"].head(8))
    more = f" +{len(alerts) - 8}" if len(alerts) > 8 else ""
    body = f"{len(alerts)} breakout(s): {syms}{more}"
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification {body!r} with title "OBV-RSI divergence breakout"'],
            check=False, capture_output=True, timeout=10)
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="*", help="explicit symbols (default: A-pool)")
    ap.add_argument("--all", action="store_true", help="scan the full ~1243-name market universe")
    ap.add_argument("--period", default="2y", help="history to download")
    ap.add_argument("--within", type=int, default=1, help="alert on breakouts within N latest bars (1=latest only)")
    ap.add_argument("--expire", type=int, default=20, help="a divergence arms the name for N bars (default 20)")
    ap.add_argument("--allow-above", action="store_true",
                    help="trigger on any close above the MA with a 2-day high, not just a fresh crossover")
    add_signal_args(ap)
    ap.add_argument("--notify", action="store_true", help="also raise a macOS notification on alerts")
    ap.add_argument("--no-write", action="store_true", help="don't write the TradingView lists or the log")
    args = ap.parse_args()

    cutoff = resolve_partial_cutoff(args.include_partial)
    ref_date = reference_last_bar(cutoff)
    symbols = resolve_universe(args)
    mode = "strict (every bar)" if args.strict else f"net over {args.lookback} bars"
    trig_desc = "close above MA" if args.allow_above else "close crosses up MA"
    print(f"Scanning {len(symbols)} names.")
    print(f"  arm    : RSI({args.rsi_len},OBV)-MA{args.rsi_ma} UP while price-MA{args.price_ma} DOWN "
          f"[{mode}], valid {args.expire} bars")
    print(f"  trigger: {trig_desc}{args.price_ma} AND high > highest high of prior 2 days "
          f"(within {args.within} bar[s])")

    rows = []
    for sym, df in iter_symbol_frames(symbols, args.period, verbose=len(symbols) > 60):
        r = evaluate(sym, df, args, cutoff, ref_date)
        if r:
            rows.append(r)
    print_filter_stats()

    if not rows:
        print("\nNothing armed and nothing triggered.")
        return 0
    res = pd.DataFrame(rows)
    alerts = res[res["state"] == "ALERT"].sort_values("rsi_chg", ascending=False).reset_index(drop=True)
    watch = res[res["state"] == "WATCH"].copy()
    # Nearest-to-triggering first: names just BELOW the MA can cross up on the next bar, so
    # they lead (gap closest to zero from below). Names already ABOVE the MA go last -- with
    # the default crossover rule they must dip back under before they can fire at all.
    watch["_above"] = watch["gap_pct"] > 0
    watch = watch.sort_values(["_above", "gap_pct"], ascending=[True, False]).drop(
        columns="_above").reset_index(drop=True)

    if alerts.empty:
        print("\n🔔 ALERT — breakout confirmed: none today.")
    else:
        print(f"\n🔔 ALERT — {len(alerts)} breakout(s) confirmed on an armed divergence:")
        print(f"  {'symbol':<10}{'trigger':>12}{'div':>12}{'lag':>5}{'ΔRSI@div':>10}"
              f"{'RSI-MA':>9}{'close':>10}{'vs MA':>8}{'trend':>11}")
        for _, r in alerts.iterrows():
            print(f"  {r['symbol']:<10}{r['trigger_date']:>12}{r['div_date']:>12}{int(r['lag']):>5}"
                  f"{r['rsi_chg']:>+10.1f}{r['rsi_ma']:>9.1f}{r['close']:>10.2f}"
                  f"{r['gap_pct']:>+8.1%}{r['trend']:>11}")

    print(f"\n👀 WATCH — {len(watch)} armed, waiting for the breakout "
          f"(nearest to triggering first; 'vs MA' > 0 = already above, needs a dip first):")
    if not watch.empty:
        print(f"  {'symbol':<10}{'div':>12}{'age':>5}{'ΔRSI@div':>10}{'RSI-MA':>9}"
              f"{'close':>10}{f'MA{args.price_ma}':>10}{'vs MA':>8}{'2d high':>10}{'trend':>11}")
        for _, r in watch.head(40).iterrows():
            print(f"  {r['symbol']:<10}{r['div_date']:>12}{r['bars_ago']:>5}{r['rsi_chg']:>+10.1f}"
                  f"{r['rsi_ma']:>9.1f}{r['close']:>10.2f}{r['ma']:>10.2f}{r['gap_pct']:>+8.1%}"
                  f"{r['need_high']:>10.2f}{r['trend']:>11}")
        if len(watch) > 40:
            print(f"  ... and {len(watch) - 40} more (full list in the TV file)")

    if not args.no_write:
        w = write_tv_list(watch["symbol"], "tv_rsi_obv_div_watch")
        a = write_tv_list(alerts["symbol"], "tv_rsi_obv_div_alert")
        n = append_trigger_log(alerts)
        print(f"\nwatch list -> {w}")
        print(f"alert list -> {a}")
        print(f"trigger log -> {TRIGGER_LOG} ({n} new row[s])")

    if args.notify:
        notify(alerts)

    print("\nUNBACKTESTED combination. The divergence alone had no edge (rsi_obv_divergence.py);")
    print("the breakout filter is untested on top of it. Watchlist tool, not an entry system.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
