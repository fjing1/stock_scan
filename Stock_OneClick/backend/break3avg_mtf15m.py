#!/usr/bin/env python3
"""EXPLORATORY: does a 15m intraday confirmation improve the daily break SMA 8/22 entry?

Prior work (RESEARCH.md #23, mtf-signal-alignment-no-edge memory): MTF *alignment*
has NO edge; a 15m "strong up bar on volume" *confirmation* of a daily DIP entry was
promising but unproven (z~1.9, ~67 samples, one 60-day regime). Nobody has tested 15m
confirmation on the break-8/22 MOMENTUM entry — this does, on the ~60-day window Yahoo
allows. Reuses the exact strong-bar definition from intraday_confirm_test.py.

On each break-8/22 signal day D, we check whether that day had a strong-up-volume 15m
bar (= real intraday buying, not a drift-up close). Compare forward HORIZON-day return
(abs + excess vs SPY) of confirmed vs unconfirmed vs all-signals baseline.

    ../../vcp_env/bin/python break3avg_mtf15m.py                # liquid ~120 names
    ../../vcp_env/bin/python break3avg_mtf15m.py --all          # full universe (max N)

*** HEAVY CAVEAT: Yahoo caps 15m history at ~60 days = ONE regime, tiny N. This is a
probe, never proof. A real test needs Alpaca intraday (see data-upgrade-plan). ***
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
import scan_stocks as scan  # noqa: E402
from rsi_ma_sweep import BASKET, INDEX  # noqa: E402
from break3avg_scan import break3avg_signal, load_all_market  # noqa: E402

HORIZON = 5
BODY_MULT, VOL_MULT, UP_FRAC = 1.5, 1.5, 0.6
LIQUID = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "ORCL", "CRM",
    "ADBE", "AMD", "INTC", "CSCO", "QCOM", "TXN", "MU", "AMAT", "NFLX", "JPM", "BAC",
    "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "UNH", "JNJ", "LLY", "ABBV", "MRK",
    "PFE", "TMO", "ABT", "DHR", "BMY", "AMGN", "WMT", "HD", "COST", "PG", "KO", "PEP",
    "MCD", "NKE", "SBUX", "LOW", "TGT", "DIS", "CAT", "BA", "HON", "GE", "UPS", "RTX",
    "XOM", "CVX", "COP", "T", "VZ", "CMCSA", "SPY", "QQQ", "IWM", "XLK", "XLF", "XLE",
    "XLV", "XLY", "XLP", "XLI", "SMH",
]


def batch_dl(symbols, period, interval):
    import yfinance as yf
    out = {}
    for i in range(0, len(symbols), 100):
        chunk = symbols[i:i + 100]
        try:
            raw = yf.download(chunk, period=period, interval=interval, auto_adjust=False,
                              progress=False, group_by="ticker", threads=True)
        except Exception:
            continue
        for s in chunk:
            try:
                d = raw[s].dropna(how="all").copy() if len(chunk) > 1 else raw.copy()
            except Exception:
                continue
            if len(d) == 0:
                continue
            if getattr(d.index, "tz", None) is not None:
                d.index = d.index.tz_localize(None)
            out[s] = d
    return out


def strong_dates(df15) -> set:
    o, h, l, c, v = (df15[k] for k in ("Open", "High", "Low", "Close", "Volume"))
    body = c - o
    avg_body = body.abs().rolling(20).mean().shift(1)
    avg_vol = v.rolling(20).mean().shift(1)
    rng = (h - l).replace(0, np.nan)
    close_pos = (c - l) / rng
    strong = ((c > o) & (body >= BODY_MULT * avg_body)
              & (close_pos >= UP_FRAC) & (v >= VOL_MULT * avg_vol)).fillna(False)
    return set(pd.Series(df15.index[strong.values]).dt.date)


def wr(a):
    a = np.array(a)
    return (len(a), (a > 0).mean() if len(a) else np.nan, a.mean() if len(a) else np.nan)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    universe = load_all_market() if args.all else list(dict.fromkeys(LIQUID + BASKET + INDEX))
    print(f"Universe: {len(universe)} names. Downloading 60d daily (need enough recent break-8/22 signals)...")
    daily = batch_dl([scan.to_yfinance_symbol(s) for s in universe], "60d", "1d")
    print(f"  daily frames: {len(daily)}")

    # SPY for excess returns
    spy = daily.get("SPY")
    if spy is None:
        import yfinance as yf
        spy = yf.download("SPY", period="60d", interval="1d", auto_adjust=False, progress=False)
    spy_c = spy["Close"].astype(float)
    spy_c.index = pd.to_datetime(spy_c.index).tz_localize(None) if getattr(spy_c.index, "tz", None) else pd.to_datetime(spy_c.index)

    # find break-8/22 signals with HORIZON forward days available
    signals = []   # (symbol, date, fwd_abs, fwd_excess)
    for sym, df in daily.items():
        if len(df) < 25:
            continue
        c = df["Close"].astype(float)
        e = break3avg_signal(df, 8, 22, "SMA", "SMA")
        idx = np.where(e.values)[0]
        for i in idx:
            if i + HORIZON >= len(c):
                continue
            d0 = c.index[i]
            fwd = c.iloc[i + HORIZON] / c.iloc[i] - 1
            try:
                sp0 = spy_c.asof(d0)
                spH = spy_c.asof(c.index[i + HORIZON])
                spy_fwd = spH / sp0 - 1
            except Exception:
                spy_fwd = np.nan
            signals.append((sym, pd.Timestamp(d0).date(), fwd, fwd - spy_fwd))
    print(f"  break-8/22 signals with {HORIZON}d forward in window: {len(signals)}")
    if len(signals) < 20:
        print("  Too few signals in the 60-day window for any read. Need --all or a longer intraday history (Alpaca).")

    sig_syms = sorted(set(s[0] for s in signals))
    print(f"Downloading 60d 15m for {len(sig_syms)} signal names...")
    intr = batch_dl(sig_syms, "60d", "15m")
    strong = {s: strong_dates(intr[s]) for s in intr}
    print(f"  15m frames: {len(intr)}\n")

    conf_abs, conf_exc, unconf_abs, unconf_exc, all_abs, all_exc = [], [], [], [], [], []
    n_no15m = 0
    for sym, d, fabs, fexc in signals:
        all_abs.append(fabs); all_exc.append(fexc)
        if sym not in strong:
            n_no15m += 1
            continue
        if d in strong[sym]:
            conf_abs.append(fabs); conf_exc.append(fexc)
        else:
            unconf_abs.append(fabs); unconf_exc.append(fexc)

    print(f"=== 15m confirmation of break-8/22 (forward {HORIZON}d, one ~60-day regime) ===")
    print(f"  {'cohort':<16}{'N':>6}{'win%':>8}{'mean abs':>10}{'mean excess vs SPY':>20}")
    for label, a, x in [("ALL signals", all_abs, all_exc),
                        ("CONFIRMED 15m", conf_abs, conf_exc),
                        ("unconfirmed", unconf_abs, unconf_exc)]:
        n, w, m = wr(a)
        _, _, mx = wr(x)
        print(f"  {label:<16}{n:>6}{w:>8.1%}{m:>+10.2%}{mx:>+20.2%}")

    nc = len(conf_abs)
    print(f"\n  confirmed N = {nc}  (signals with no 15m data: {n_no15m})")
    if nc >= 20 and len(unconf_abs) >= 20:
        _, wc, mc = wr(conf_abs); _, wu, mu = wr(unconf_abs)
        dwin = (wc - wu) * 100
        print(f"  confirmed vs unconfirmed: Δwin {dwin:+.1f}pp, Δmean {mc-mu:+.2%}")
        print("  -> " + ("CONFIRMATION HELPS (indicative)" if dwin >= 5
                         else "no meaningful confirmation edge") + " — still ONE regime, treat as a probe.")
    else:
        print("  Too few confirmed (need >=20 each side) — NO read. This is the 60-day Yahoo")
        print("  wall; a real MTF test needs Alpaca intraday history (data-upgrade-plan).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
