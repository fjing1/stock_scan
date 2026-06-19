#!/usr/bin/env python3
"""End-of-day REGIME readout for the COMBO — mechanical, computable each close.
Tells you which regime SPY/QQQ are in and what the COMBO does TODAY.
"""
import numpy as np, pandas as pd, scan_stocks as scan
TD = 252
def rsi(x, n):
    d = x.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    return 100 - 100/(1 + up.ewm(alpha=1/n, adjust=False).mean()/dn.ewm(alpha=1/n, adjust=False).mean())

for tk in ("SPY", "QQQ"):
    df = scan.download_daily(tk, period="3y")
    C = df["Close"].astype(float)
    sma200 = C.rolling(200).mean(); sma50 = C.rolling(50).mean()
    last = C.iloc[-1]; s200 = sma200.iloc[-1]; s50 = sma50.iloc[-1]
    above = (C > sma200)
    confirm3 = bool(above.iloc[-3:].all())                       # 3 consecutive closes above 200SMA
    slope = (sma200.iloc[-1] / sma200.iloc[-22] - 1) * 100        # ~1-month 200SMA slope
    hi52 = C.iloc[-252:].max(); dd_from_hi = (last / hi52 - 1) * 100
    vol30 = C.pct_change().iloc[-30:].std() * np.sqrt(TD) * 100
    r2 = rsi(C, 2).iloc[-1]

    # regime label
    if last > s200 and slope > 0 and dd_from_hi > -5:
        regime = "STRONG / PURE BULL (hold dominates; MR thin)"
    elif last > s200 and slope > 0:
        regime = "UPTREND (pullback in bull)" if dd_from_hi < -5 else "UPTREND"
    elif last < s200 and slope < 0:
        regime = "BEAR / RISK-OFF (MR-bounce regime)"
    else:
        regime = "TRANSITION / CHOP (near 200SMA, flat slope)"

    # COMBO action today
    if confirm3 and last > s200:
        action = "HOLD (be long the index)"
    elif last < s200:
        action = f"MR-BOUNCE ARMED — BUY today if RSI2<10 (now {r2:.0f}); else CASH"
    else:
        action = "WAIT/CASH (above 200SMA but <3-day confirm)"

    print(f"\n===== {tk}  (as of {C.index[-1].date()}) =====")
    print(f"  Close {last:.2f} | 200SMA {s200:.2f} ({'ABOVE' if last>s200 else 'BELOW'}, 3-day confirm={confirm3})")
    print(f"  50SMA {s50:.2f} ({'>' if s50>s200 else '<'}200SMA) | 200SMA 1mo slope {slope:+.1f}%")
    print(f"  % from 52w high: {dd_from_hi:+.1f}% | 30d realized vol: {vol30:.0f}% | RSI(2): {r2:.0f}")
    print(f"  REGIME : {regime}")
    print(f"  COMBO  : {action}")
