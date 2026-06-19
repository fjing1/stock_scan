#!/usr/bin/env python3
"""Refined AAPL strategy for a HIGH risk tolerance: leveraged volatility-targeting
(convert the vol-managed Sharpe edge into higher return), with a trend-scaled leverage
cap and an ATR-trail crash brake. Honest costs: turnover + financing on leverage.
Benchmark = buy & hold AAPL. Full 46y + out-of-sample (>=2018).
"""
import numpy as np
import pandas as pd
import scan_stocks as scan

COST_TURN, FIN = 0.0007, 0.05      # 7bps/turn ; 5%/yr financing on leverage>1
TD, OOS = 252, pd.Timestamp("2018-01-01")

df = scan.download_daily("AAPL", period="max")
C = df["Close"].astype(float); H = df["High"].astype(float); L = df["Low"].astype(float)
ret = C.pct_change().fillna(0.0)
vol30 = ret.rolling(30).std() * np.sqrt(TD)
sma50, sma100, sma200 = (C.rolling(n).mean() for n in (50, 100, 200))
prevc = C.shift(1)
tr = pd.concat([H - L, (H - prevc).abs(), (L - prevc).abs()], axis=1).max(axis=1)
atr22 = tr.ewm(alpha=1/22, adjust=False).mean()

def vt(target, cap, trend_brake=False):
    lev = pd.Series(cap, index=C.index)
    if trend_brake:                      # only lever in uptrend; de-risk hard below SMA100
        lev = pd.Series(np.where(C > sma100, cap, 0.5), index=C.index)
    return (target / vol30).clip(lower=0).clip(upper=lev)

def crash_brake(pos):
    """go flat after price closes 5xATR below its since-entry peak; re-engage when C>SMA50."""
    p = pos.values.copy(); c = C.values; a = atr22.values; s50 = sma50.values
    active = True; peak = c[0]
    for i in range(1, len(c)):
        if active:
            peak = max(peak, c[i])
            if np.isfinite(a[i]) and c[i] < peak - 5 * a[i]:
                active = False
        else:
            if np.isfinite(s50[i]) and c[i] > s50[i]:
                active = True; peak = c[i]
        if not active:
            p[i] = 0.0
    return pd.Series(p, index=C.index)

strategies = {
    "Buy & Hold AAPL": pd.Series(1.0, index=C.index),
    "Vol-target 25% (cap 1.0)": vt(0.25, 1.0),
    "Lev VT 35% (cap 2.0)": vt(0.35, 2.0),
    "Lev VT 45% (cap 3.0)": vt(0.45, 3.0),
    "Lev VT 35% + trend brake": vt(0.35, 2.0, trend_brake=True),
    "Lev VT 45% + trend brake": vt(0.45, 3.0, trend_brake=True),
    "Lev VT 35% + crash brake": crash_brake(vt(0.35, 2.0)),
    "Lev VT 45% + trend + crash": crash_brake(vt(0.45, 3.0, trend_brake=True)),
}

def metrics(pos, lo=None, hi=None):
    p = pos.shift(1).fillna(0.0)
    turn = p.diff().abs().fillna(0.0)
    fin = (p - 1).clip(lower=0) * (FIN / TD)
    r = p * ret - turn * COST_TURN - fin
    m = pd.Series(True, index=r.index)
    if lo is not None: m &= r.index >= lo
    if hi is not None: m &= r.index < hi
    r = r[m]
    if len(r) < 60: return None
    eq = (1 + r).cumprod(); n = len(r)
    cagr = eq.iloc[-1] ** (TD / n) - 1 if eq.iloc[-1] > 0 else -1
    dd = (eq / eq.cummax() - 1).min()
    sh = r.mean() / r.std() * np.sqrt(TD) if r.std() > 0 else 0
    return dict(cagr=cagr, vol=r.std()*np.sqrt(TD), sharpe=sh, maxdd=dd,
                mar=cagr/abs(dd) if dd < 0 else np.inf, avgexp=p[m].mean())

print(f"AAPL {df.index[0].date()}→{df.index[-1].date()} | costs: {COST_TURN*1e4:.0f}bps/turn + {FIN*100:.0f}%/yr financing")
for title, lo, hi in [("FULL SAMPLE (46y)", None, None), ("OUT-OF-SAMPLE (>=2018, ~8.5y)", OOS, None)]:
    print("\n" + "="*96)
    print(f"{title}  — ranked by CAGR (high-risk objective); vs buy & hold AAPL")
    print("="*96)
    rows = {nm: metrics(p, lo=lo, hi=hi) for nm, p in strategies.items()}
    r = pd.DataFrame({k: v for k, v in rows.items() if v}).T.sort_values("cagr", ascending=False)
    d = r.copy()
    for c in ["cagr","vol","maxdd","avgexp"]: d[c] = (d[c]*100).round(0)
    d["sharpe"] = d["sharpe"].round(2); d["mar"] = d["mar"].round(2)
    print(d[["cagr","vol","sharpe","maxdd","mar","avgexp"]].to_string())
