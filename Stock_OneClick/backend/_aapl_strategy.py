#!/usr/bin/env python3
"""AAPL-only strategy study. Benchmark = buy & hold AAPL. Goal: beat it risk-adjusted
(Sharpe / MAR / drawdown) with simple, parameter-light, long-only daily rules. Full
history (many regimes) + out-of-sample split to guard against single-asset overfitting.
"""
import numpy as np
import pandas as pd
import scan_stocks as scan

COST_TURN = 0.0007    # ~7 bps per position change (AAPL is ultra-liquid)
TD = 252
OOS = pd.Timestamp("2018-01-01")

df = scan.download_daily("AAPL", period="max")
spy = scan.download_daily("SPY", period="max")
C = df["Close"].astype(float); H = df["High"].astype(float); L = df["Low"].astype(float)
spyC = spy["Close"].reindex(df.index).ffill().astype(float)
ret = C.pct_change().fillna(0.0)
print(f"AAPL history: {df.index[0].date()} → {df.index[-1].date()}  ({len(df)} days, {len(df)/TD:.0f}y)")

def rsi(x, n):
    d = x.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    rs = up.ewm(alpha=1/n, adjust=False).mean() / dn.ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1+rs)

sma = {n: C.rolling(n).mean() for n in (5, 50, 100, 200)}
ema100 = C.ewm(span=100, adjust=False).mean()
rsi2 = rsi(C, 2)
mom12 = C / C.shift(252) - 1
vol20 = ret.rolling(20).std() * np.sqrt(TD)
donch_hi = H.rolling(50).max().shift(1); donch_lo = L.rolling(20).min().shift(1)

def state_machine(entry, exit_):
    pos = np.zeros(len(C)); inpos = False
    en = entry.values; ex = exit_.values
    for i in range(len(C)):
        if inpos:
            if ex[i]: inpos = False
        else:
            if en[i]: inpos = True
        pos[i] = 1.0 if inpos else 0.0
    return pd.Series(pos, index=C.index)

positions = {
    "Buy & Hold AAPL": pd.Series(1.0, index=C.index),
    "Close > SMA200": (C > sma[200]).astype(float),
    "SMA50 > SMA200 (golden)": (sma[50] > sma[200]).astype(float),
    "Close > EMA100": (C > ema100).astype(float),
    "Abs momentum (12m>0)": (mom12 > 0).astype(float),
    "Dual regime (AAPL&SPY>200)": ((C > sma[200]) & (spyC > spyC.rolling(200).mean())).astype(float),
    "Vol-target 25% (cap 1.0)": (0.25 / vol20).clip(0, 1.0),
    "RSI2 dip-buy (in uptrend)": state_machine((rsi2 < 10) & (C > sma[200]), (C > sma[5])),
    "Donchian 50/20 breakout": state_machine(C > donch_hi, C < donch_lo),
}

def metrics(pos, lo=None, hi=None):
    p = pos.shift(1).fillna(0.0)                      # act next day (no look-ahead)
    turn = p.diff().abs().fillna(0.0)
    r = p * ret - turn * COST_TURN
    m = pd.Series(True, index=r.index)
    if lo is not None: m &= r.index >= lo
    if hi is not None: m &= r.index < hi
    r = r[m]
    if len(r) < 60: return None
    eq = (1 + r).cumprod(); n = len(r)
    cagr = eq.iloc[-1] ** (TD / n) - 1
    dd = (eq / eq.cummax() - 1).min()
    sh = r.mean() / r.std() * np.sqrt(TD) if r.std() > 0 else 0
    return dict(cagr=cagr, vol=r.std()*np.sqrt(TD), sharpe=sh, maxdd=dd,
                mar=cagr/abs(dd) if dd < 0 else np.inf, tim=(p[m] > 0).mean())

print("\n" + "="*94)
print("FULL-SAMPLE: single-name AAPL strategies vs buy & hold AAPL  (ranked by Sharpe)")
print("="*94)
rows = {nm: metrics(p) for nm, p in positions.items()}
res = pd.DataFrame(rows).T.sort_values("sharpe", ascending=False)
disp = res.copy()
for c in ["cagr","vol","maxdd","tim"]: disp[c] = (disp[c]*100).round(1)
disp["sharpe"] = disp["sharpe"].round(2); disp["mar"] = disp["mar"].round(2)
print(disp[["cagr","vol","sharpe","maxdd","mar","tim"]].to_string())

print("\n" + "="*94)
print(f"OUT-OF-SAMPLE: train < {OOS.date()}  vs  test ≥ {OOS.date()}  (CAGR / Sharpe / MaxDD)")
print("="*94)
print(f"{'strategy':<30}{'tr CAGR':>9}{'tr Shp':>8}{'tr DD':>8}{'te CAGR':>9}{'te Shp':>8}{'te DD':>8}")
for nm, p in positions.items():
    tr = metrics(p, hi=OOS); te = metrics(p, lo=OOS)
    if tr and te:
        print(f"{nm:<30}{tr['cagr']*100:8.1f}%{tr['sharpe']:8.2f}{tr['maxdd']*100:7.0f}%{te['cagr']*100:8.1f}%{te['sharpe']:8.2f}{te['maxdd']*100:7.0f}%")
