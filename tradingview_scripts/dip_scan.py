"""Scan the full symbol universe for the dip-in-uptrend signal.

Applies the validated rule to the latest daily bar of every symbol in
stock_symbols_1243.py:

    Entry = Close > SMA200  AND  RSI(14) < 40  AND  Stoch %K(10,EMA4) < 20

Reports:
    FRESH BUY    -- signal true today and NOT true yesterday (actionable now)
    ACTIVE setup -- signal still true today (already triggered, may still be valid)
    WATCH        -- uptrend + getting oversold (RSI<45 & %K<30) but not triggered

Writes a dated CSV to results/<YYYYMMDD>/ and prints the fresh-buy list.

    python dip_scan.py                 # full universe (stocks + ETFs)
    python dip_scan.py --universe etfs
    python dip_scan.py --limit 100     # quick test
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cycle_patter_for_swing import compute_cycle_stoch  # noqa: E402
from pead_drift import load_earnings  # noqa: E402

import yfinance as yf  # noqa: E402

PEAD_WINDOW_DAYS = 95   # ~63 trading days a name stays "in play" after an earnings report
PEAD_MAX_TILT = 12      # max +/- DipRank points contributed by the earnings-surprise tilt


def load_universe(which):
    from stock_symbols_1243 import STOCK_SYMBOLS, ETF_SYMBOLS
    if which == "stocks":
        syms = list(STOCK_SYMBOLS)
    elif which == "etfs":
        syms = list(ETF_SYMBOLS)
    else:
        syms = list(STOCK_SYMBOLS) + list(ETF_SYMBOLS)
    # de-dup, keep order
    seen, out = set(), []
    for s in syms:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def fetch_chunk(symbols, period="2y", interval="1d"):
    raw = yf.download(symbols, period=period, interval=interval,
                      auto_adjust=False, progress=False, group_by="ticker",
                      threads=True)
    out = {}
    for s in symbols:
        try:
            d = raw[s].dropna(how="all").copy() if len(symbols) > 1 else raw.copy()
        except Exception:
            continue
        if len(d) == 0:
            continue
        if getattr(d.index, "tz", None) is not None:
            d.index = d.index.tz_localize(None)
        out[s] = d
    return out


def scan_symbol(sym, df, rsi_max, k_max):
    if df is None or len(df) < 260:           # need SMA200 + 12m momentum + warmup
        return None
    if not {"High", "Low", "Close"}.issubset(df.columns):
        return None
    try:
        cs = compute_cycle_stoch(df)
    except Exception:
        return None
    close = df["Close"].astype(float)
    sma200 = close.rolling(200).mean()
    ma50 = close.rolling(50).mean()
    if pd.isna(sma200.iloc[-1]) or pd.isna(cs["rsi"].iloc[-1]):
        return None

    trend = close > sma200
    oversold = (cs["rsi"] < rsi_max) & (cs["stoch_k"] < k_max)
    signal = (trend & oversold).fillna(False)

    sig_now = bool(signal.iloc[-1])
    sig_prev = bool(signal.iloc[-2])
    rsi0 = float(cs["rsi"].iloc[-1])
    k0 = float(cs["stoch_k"].iloc[-1])
    cyc0 = float(cs["cycle"].iloc[-1])
    c0 = float(close.iloc[-1])
    pct_ma = (c0 / float(sma200.iloc[-1]) - 1.0) * 100.0
    trend_now = bool(trend.iloc[-1])
    # ranking features (validated in rank_test.py)
    mom12m = (c0 / float(close.iloc[-253]) - 1.0) * 100.0
    vs_ma50 = (c0 / float(ma50.iloc[-1]) - 1.0) * 100.0 if not pd.isna(ma50.iloc[-1]) else 0.0

    if sig_now and not sig_prev:
        status = "FRESH_BUY"
    elif sig_now:
        status = "ACTIVE"
    elif trend_now and rsi0 < 45 and k0 < 30:
        status = "WATCH"
    else:
        return None
    return {
        "symbol": sym, "status": status, "date": df.index[-1].date().isoformat(),
        "close": round(c0, 2), "rsi": round(rsi0, 1), "stochK": round(k0, 1),
        "cycle": round(cyc0, 1), "pct_above_MA200": round(pct_ma, 1),
        "mom_12m": round(mom12m, 1), "vs_MA50": round(vs_ma50, 1),
    }


def confirm_15m(sym, window, body_mult, vol_mult, up_frac):
    """Intraday turn confirmation: is there a strong up 15m candle on above-avg
    volume within the last `window` bars?

    Strong up bar = close>open AND body >= body_mult x avg|body|(20) AND closes in
    top (1-up_frac) of its range AND volume >= vol_mult x avg vol(20).
    Returns dict with conf15 Y/N, the latest qualifying bar's up%, vol ratio, time.
    """
    try:
        d = yf.download(sym, period="5d", interval="15m",
                        auto_adjust=False, progress=False)
    except Exception:
        return None
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    if d.empty or len(d) < 25:
        return None
    o, h, l, c, v = (d["Open"], d["High"], d["Low"], d["Close"], d["Volume"])
    body = c - o
    avg_body = body.abs().rolling(20).mean().shift(1)
    avg_vol = v.rolling(20).mean().shift(1)
    rng = (h - l).replace(0, np.nan)
    close_pos = (c - l) / rng
    strong = ((c > o) & (body >= body_mult * avg_body)
              & (close_pos >= up_frac) & (v >= vol_mult * avg_vol)).fillna(False)
    recent = strong.iloc[-window:]
    if not recent.any():
        return {"conf15": "N", "up15": np.nan, "volx": np.nan, "bar15": ""}
    idx = recent[recent].index[-1]                       # latest qualifying bar
    up = float((c.loc[idx] - o.loc[idx]) / o.loc[idx] * 100)
    volx = float(v.loc[idx] / avg_vol.loc[idx]) if avg_vol.loc[idx] else np.nan
    return {"conf15": "Y", "up15": round(up, 2),
            "volx": round(volx, 1), "bar15": str(idx)[5:16]}


def apply_pead_tilt(df):
    """Long-only PEAD tilt (RESEARCH.md #19/#20): nudge DipRank by a name's most recent
    earnings surprise. A name is "in play" if it reported within PEAD_WINDOW_DAYS; its
    surprise percentile (among in-play hits) maps to +/-PEAD_MAX_TILT DipRank points, and
    out-of-play names are neutral (0). PEAD as a long-only tilt is the deployable use found
    in #20 (it did NOT reliably lift the market-neutral ensemble, but it is a real,
    orthogonal, OOS-persistent long-side effect). Earnings are fetched ONLY for the hits
    (fast) and reuse the cached surprises. Adds earn_surprise, earn_age_d, DipRank_PEAD."""
    earn = load_earnings(list(df["symbol"]), use_cache=True)
    surp, age = [], []
    for sym, dstr in zip(df["symbol"], df["date"]):
        today = pd.Timestamp(dstr)
        best = None
        for ds, sp in earn.get(sym, []):
            a = (today - pd.Timestamp(ds)).days
            if 0 <= a <= PEAD_WINDOW_DAYS and (best is None or a < best[0]):
                best = (a, sp)
        age.append(best[0] if best else np.nan)
        surp.append(round(best[1], 1) if best else np.nan)
    df["earn_surprise"] = surp
    df["earn_age_d"] = age
    inplay = df["earn_surprise"].notna()
    tilt = pd.Series(0.0, index=df.index)
    if inplay.sum() >= 3:
        pr = df.loc[inplay, "earn_surprise"].rank(pct=True)
        tilt.loc[inplay] = (pr - 0.5) * 2 * PEAD_MAX_TILT
    df["DipRank_PEAD"] = (df["DipRank"] + tilt).clip(0, 100).round().astype(int)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=["stocks", "etfs", "all"], default="all")
    ap.add_argument("--rsi-max", type=float, default=40.0)
    ap.add_argument("--k-max", type=float, default=20.0)
    ap.add_argument("--chunk", type=int, default=120)
    ap.add_argument("--limit", type=int, default=0, help="cap symbols (testing)")
    ap.add_argument("--no-confirm", action="store_true",
                    help="skip the 15-minute strong-up-bar confirmation")
    ap.add_argument("--require-confirm", action="store_true",
                    help="keep only hits with a 15m confirmation")
    ap.add_argument("--confirm-window", type=int, default=26,
                    help="how many recent 15m bars to scan (26 ~= 1 session)")
    ap.add_argument("--body-mult", type=float, default=1.5,
                    help="strong-bar body vs 20-bar avg body")
    ap.add_argument("--vol-mult", type=float, default=1.5,
                    help="bar volume vs 20-bar avg volume")
    ap.add_argument("--up-frac", type=float, default=0.6,
                    help="min close position in bar range (0.6 = upper 40%)")
    ap.add_argument("--no-pead", action="store_true",
                    help="skip the PEAD earnings-surprise tilt on DipRank")
    args = ap.parse_args(argv)

    syms = load_universe(args.universe)
    if args.limit:
        syms = syms[: args.limit]
    print(f"scanning {len(syms)} symbols (daily, dip-in-uptrend: "
          f"RSI<{args.rsi_max:.0f} & %K<{args.k_max:.0f} & >SMA200) ...", flush=True)

    rows, scanned, report_date = [], 0, None
    for i in range(0, len(syms), args.chunk):
        chunk = syms[i: i + args.chunk]
        data = fetch_chunk(chunk)
        for s in chunk:
            r = scan_symbol(s, data.get(s), args.rsi_max, args.k_max)
            scanned += 1
            if r:
                rows.append(r)
                report_date = report_date or r["date"]
        print(f"  ...{min(i + args.chunk, len(syms))}/{len(syms)} scanned, "
              f"{len(rows)} hits", flush=True)

    if not rows:
        print("no hits.")
        return

    df = pd.DataFrame(rows)
    # ---- DipRank: evidence-based composite (rank_test.py), cross-sectional ----
    # 12m momentum (higher=better) 0.45 + pullback depth below MA50 (deeper=better)
    # 0.30 + trend strength above MA200 (higher=better) 0.25.  Oversold DEPTH is
    # intentionally excluded — it showed no predictive power out-of-sample.
    p_mom = df["mom_12m"].rank(pct=True)
    p_pb = (-df["vs_MA50"]).rank(pct=True)
    p_tr = df["pct_above_MA200"].rank(pct=True)
    df["DipRank"] = (100 * (0.45 * p_mom + 0.30 * p_pb + 0.25 * p_tr)).round().astype(int)

    # ---- PEAD long-only tilt (#19/#20): nudge DipRank by recent earnings surprise ----
    if not args.no_pead:
        print(f"\napplying PEAD earnings-surprise tilt to {len(df)} hits ...", flush=True)
        df = apply_pead_tilt(df)
    rank_col = "DipRank_PEAD" if "DipRank_PEAD" in df.columns else "DipRank"

    # ---- 15-minute strong-up-bar confirmation (intraday turn) ----
    if not args.no_confirm:
        print(f"\nchecking 15m confirmation for {len(df)} hits ...", flush=True)
        recs = [confirm_15m(s, args.confirm_window, args.body_mult,
                            args.vol_mult, args.up_frac) for s in df["symbol"]]
        df["conf15"] = [r["conf15"] if r else "n/a" for r in recs]
        df["up15%"] = [r["up15"] if r else np.nan for r in recs]
        df["volx"] = [r["volx"] if r else np.nan for r in recs]
        df["bar15"] = [r["bar15"] if r else "" for r in recs]
        if args.require_confirm:
            df = df[df["conf15"] == "Y"]
            if df.empty:
                print("no hits with 15m confirmation.")
                return
        df["_c"] = (df["conf15"] != "Y").astype(int)   # confirmed first
    else:
        df["_c"] = 0

    order = {"FRESH_BUY": 0, "ACTIVE": 1, "WATCH": 2}
    df["_o"] = df["status"].map(order)
    df = df.sort_values(["_o", "_c", rank_col],
                        ascending=[True, True, False]).drop(columns=["_o", "_c"])

    # save dated CSV under results/
    rd = (report_date or "latest").replace("-", "")
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", rd)
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"dip_in_uptrend_{rd}.csv")
    df.to_csv(csv_path, index=False)

    counts = df["status"].value_counts().to_dict()
    print(f"\n==== Dip-in-Uptrend scan ({report_date}) — "
          f"FRESH_BUY={counts.get('FRESH_BUY',0)} "
          f"ACTIVE={counts.get('ACTIVE',0)} WATCH={counts.get('WATCH',0)} ====")
    print(f"saved: {csv_path}\n")

    fresh = df[df["status"] == "FRESH_BUY"]
    show = fresh if len(fresh) else df[df["status"] == "ACTIVE"]
    title = "FRESH BUYS today" if len(fresh) else "ACTIVE setups (no fresh buys today)"
    print(f"--- {title} (ranked by DipRank; 15m-confirmed first) ---")
    cols = ["symbol", "DipRank_PEAD", "DipRank", "earn_surprise", "earn_age_d",
            "conf15", "up15%", "volx", "bar15",
            "close", "mom_12m", "vs_MA50", "pct_above_MA200", "rsi", "stochK"]
    cols = [c for c in cols if c in show.columns]
    print(show[cols].head(40).to_string(index=False) if len(show) else "  (none)")


if __name__ == "__main__":
    main()
