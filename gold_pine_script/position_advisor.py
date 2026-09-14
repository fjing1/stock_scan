"""Per-holding advisor: a HOLD / TRIM / SELL read for one ticker, grounded in this repo's
VALIDATED swing signals (RESEARCH.md) rather than generic TA.

How the validated edges map to a holder's decision:
  - >MA200 = uptrend intact -> the system HOLDS; MA200 is the risk line (tight stops HURT, #5).
    Below MA200 = the swing thesis is broken -> lean REDUCE / SELL.
  - Sell-into-strength exit (#5/#6): close>=SMA20 AND (%K>=70 OR RSI2>=70) -> TRIM into strength.
  - Dip-in-uptrend (#4): >MA200 & RSI(14)<40 & %K<20 -> oversold bounce zone -> HOLD / ADD.
  - 12-month relative strength (#7) = quality of the hold (high-RS names earn the benefit of doubt).
  - PEAD (#19): a recent earnings BEAT is a fundamental tailwind; a MISS is a caution flag.
  - Sector/peers trend = top-down confirmation (is the whole group healthy?).

NOT a valuation / price target and NOT personalized advice — a swing-system + earnings read.
Tailor with --cost (entry) and --horizon. Holds are ~1-3 weeks (short-horizon mean reversion).

    python position_advisor.py NTR [--cost 52.0] [--peers MOS,CF,MOO]
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

PEAD_WINDOW_DAYS = 95


def rsi(close, n):
    ch = close.diff()
    up = ch.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-ch).clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50)


def dl(sym, period="2y"):
    d = yf.download(sym, period=period, interval="1d", auto_adjust=False,
                    progress=False, group_by="ticker", threads=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(-1) if sym in d.columns.get_level_values(0) else d.columns.get_level_values(0)
        try:
            d = yf.download(sym, period=period, interval="1d", auto_adjust=False, progress=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
        except Exception:
            pass
    if getattr(d.index, "tz", None) is not None:
        d.index = d.index.tz_localize(None)
    return d.dropna(how="all")


def peer_health(peers):
    """Fraction of peers above their MA50 + median 3-month return = quick sector read."""
    above, r3m = [], []
    for p in peers:
        try:
            d = dl(p, "1y")
            c = d["Close"].astype(float)
            if len(c) < 70:
                continue
            above.append(float(c.iloc[-1] > c.rolling(50).mean().iloc[-1]))
            r3m.append(float(c.iloc[-1] / c.iloc[-63] - 1) * 100)
        except Exception:
            continue
    if not above:
        return None
    return dict(n=len(above), pct_above_ma50=100 * np.mean(above), med_3m=float(np.median(r3m)))


def analyze(sym, cost, peers):
    d = dl(sym)
    if d is None or len(d) < 260 or not {"High", "Low", "Close"}.issubset(d.columns):
        print(f"{sym}: insufficient data"); return
    cs = compute_cycle_stoch(d)
    c = d["Close"].astype(float)
    c0 = float(c.iloc[-1])
    sma20, sma50, sma200 = (c.rolling(n).mean() for n in (20, 50, 200))
    rsi14 = float(cs["rsi"].iloc[-1]); k = float(cs["stoch_k"].iloc[-1]); cyc = float(cs["cycle"].iloc[-1])
    rsi2 = float(rsi(c, 2).iloc[-1])
    s20, s50, s200 = float(sma20.iloc[-1]), float(sma50.iloc[-1]), float(sma200.iloc[-1])
    mom12 = (c0 / float(c.iloc[-253]) - 1) * 100
    above200 = c0 > s200

    # swing zone
    oversold = rsi14 < 40 and k < 20
    sell_strength = c0 >= s20 and (k >= 70 or rsi2 >= 70)

    # earnings (PEAD)
    earn = load_earnings([sym], use_cache=True).get(sym, [])
    today = d.index[-1]
    recent = None
    for ds, sp in earn:
        a = (today - pd.Timestamp(ds)).days
        if 0 <= a <= PEAD_WINDOW_DAYS and (recent is None or a < recent[0]):
            recent = (a, sp)
    last4 = sorted(earn, key=lambda x: x[0])[-4:]

    ph = peer_health(peers) if peers else None

    # ---- decision logic ----
    flags = []
    lean = 0
    if above200:
        flags.append(f"+ Uptrend intact: price {(c0/s200-1)*100:+.0f}% above MA200 ({s200:.2f}) — the system's hold line"); lean += 2
    else:
        flags.append(f"- Below MA200 ({s200:.2f}, {(c0/s200-1)*100:+.0f}%): swing uptrend BROKEN — primary sell trigger"); lean -= 3
    if sell_strength:
        flags.append(f"- Sell-into-strength zone: %K {k:.0f} / RSI2 {rsi2:.0f}, price>=SMA20 — validated TRIM trigger (#5/#6)"); lean -= 2
    elif oversold:
        flags.append(f"+ Oversold dip {'in uptrend' if above200 else '(but no uptrend)'}: RSI {rsi14:.0f}, %K {k:.0f} — bounce zone, HOLD/ADD if >MA200"); lean += (2 if above200 else -1)
    else:
        flags.append(f"~ Mid-swing: RSI {rsi14:.0f}, %K {k:.0f}, RSI2 {rsi2:.0f} — no exit/entry trigger firing")
    flags.append((f"+ Strong 12m relative strength {mom12:+.0f}%" if mom12 > 15 else
                  f"- Weak 12m momentum {mom12:+.0f}%" if mom12 < 0 else f"~ Modest 12m momentum {mom12:+.0f}%"))
    lean += 1 if mom12 > 15 else (-1 if mom12 < 0 else 0)
    if recent:
        beat = recent[1] > 0
        flags.append(f"{'+' if beat else '-'} Recent earnings {'BEAT' if beat else 'MISS'} {recent[1]:+.0f}% ({recent[0]}d ago) — PEAD {'tailwind' if beat else 'caution'}")
        lean += 1 if beat else -1
    else:
        flags.append("~ No earnings within ~3mo (no active PEAD drift)")
    if ph:
        healthy = ph["pct_above_ma50"] >= 50 and ph["med_3m"] > 0
        flags.append(f"{'+' if healthy else '-'} Sector: {ph['pct_above_ma50']:.0f}% of peers >MA50, median 3m {ph['med_3m']:+.0f}% — group {'healthy' if healthy else 'soft'}")
        lean += 1 if healthy else -1

    verdict = ("SELL / REDUCE" if lean <= -3 else "TRIM" if lean <= -1 else
               "HOLD (add on dips)" if lean >= 4 else "HOLD")

    print(f"\n================ POSITION ADVISOR: {sym} ================")
    print(f"price {c0:.2f}  |  MA20 {s20:.2f}  MA50 {s50:.2f}  MA200 {s200:.2f}  "
          f"|  RSI14 {rsi14:.0f}  %K {k:.0f}  RSI2 {rsi2:.0f}  cycle {cyc:.1f}")
    if cost:
        print(f"your cost {cost:.2f}  ->  unrealized {(c0/cost-1)*100:+.1f}%")
    if last4:
        print("earnings surprise history (oldest->newest): " +
              "  ".join(f"{d_[:7]} {sp:+.0f}%" for d_, sp in last4))
    print("\nsignal read:")
    for f in flags:
        print(f"  {f}")
    print(f"\n  >>> VERDICT: {verdict}   (signal lean {lean:+d})")
    print("\nkey levels for a holder:")
    print(f"  - risk line: MA200 = {s200:.2f}  (a weekly close below it = swing thesis broken, reduce)")
    print(f"  - trim trigger: a push to %K>=70 / RSI2>=70 while >SMA20 ({s20:.2f}) = sell into strength")
    print("  caveat: swing/earnings read (1-3wk horizon), not a valuation or price target; tailor to "
          "the client's cost basis, horizon, tax, and position size.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("--cost", type=float, default=0.0, help="client's entry price (optional)")
    ap.add_argument("--peers", default="", help="comma-separated peer/sector tickers")
    args = ap.parse_args()
    peers = [p.strip().upper() for p in args.peers.split(",") if p.strip()]
    analyze(args.ticker.upper(), args.cost or None, peers)


if __name__ == "__main__":
    main()
