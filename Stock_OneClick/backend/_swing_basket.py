#!/usr/bin/env python3
"""COMBO regime-switch (hold>200MA + MR-bounce<200MA) on a basket of liquid large-cap
stocks, full cycle, vs equal-weight buy&hold basket and SPY. Long-only cash, 15bps/turn.
NOTE: survivorship bias — these are current liquid large-caps; absolute numbers flattered,
the COMBO-vs-hold COMPARISON (both on the same names) is the robust takeaway.
"""
import numpy as np, pandas as pd, scan_stocks as scan
from collections import defaultdict
COST, TD = 0.0015, 252
NAMES = ["AAPL","MSFT","AMZN","GOOGL","JPM","BAC","WFC","XOM","CVX","JNJ","PFE","MRK",
         "PG","KO","PEP","WMT","HD","MCD","DIS","NKE","INTC","CSCO","ORCL","IBM","QCOM",
         "TXN","CAT","BA","MMM","UNH","T","VZ","C","GS","COST","LOW","HON","AMGN","ADBE","CRM"]

def rsi(x, n):
    d = x.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    return 100 - 100/(1 + up.ewm(alpha=1/n, adjust=False).mean()/dn.ewm(alpha=1/n, adjust=False).mean())

def combo(df, confirm=3, mr_rsi=10):
    C = df["Close"].astype(float); c = C.values; s200 = C.rolling(200).mean().values
    r2 = rsi(C, 2).values
    above3 = ((C > C.rolling(200).mean()).rolling(confirm).sum() == confirm).values
    pos = np.zeros(len(c)); i = 200
    while i < len(c)-1:
        if above3[i]: pos[i] = 1.0; i += 1
        elif np.isfinite(s200[i]) and c[i] < s200[i] and r2[i] < mr_rsi:
            pos[i] = 1.0; j = i+1; held = 0
            while j < len(c):
                held += 1; pos[j] = 1.0
                if c[j] > c[j-1] or held >= 10: break
                j += 1
            i = j+1
        else: i += 1
    return pd.Series(pos, index=C.index)

combo_d = defaultdict(list); hold_d = defaultdict(list)
got = 0
for sym in NAMES:
    try:
        df = scan.download_daily(sym, period="max")
    except Exception:
        df = None
    if df is None or len(df) < 400: continue
    got += 1
    C = df["Close"].astype(float); ret = C.pct_change()
    pos = combo(df).shift(1).fillna(0.0); turn = pos.diff().abs().fillna(0.0)
    sret = pos*ret - turn*COST
    for dt, x in sret.items():
        if np.isfinite(x): combo_d[dt].append(x)
    for dt, x in ret.items():
        if np.isfinite(x): hold_d[dt].append(x)
print(f"basket names with data: {got}/{len(NAMES)}")

spy = scan.download_daily("SPY", period="max"); spyr = spy["Close"].pct_change()
cal = pd.DatetimeIndex(sorted(set(combo_d)|set(hold_d)))
def series(d): return pd.Series([np.mean(d[x]) if x in d else 0.0 for x in cal], index=cal)
combo_s, hold_s = series(combo_d), series(hold_d)
spy_s = spyr.reindex(cal).fillna(0.0)

def met(s, lo, hi):
    r = s[(s.index>=lo)&(s.index<hi)]; r = r[r!=0] if False else r
    if len(r) < 60: return None
    eq=(1+r).cumprod(); n=len(r); cagr=eq.iloc[-1]**(TD/n)-1 if eq.iloc[-1]>0 else -1
    dd=(eq/eq.cummax()-1).min(); sh=r.mean()/r.std()*np.sqrt(TD) if r.std()>0 else 0
    return cagr, sh, dd

eras=[("2000-2009 (bear/sideways)","2000-01-01","2010-01-01"),
      ("2010-2019 (bull)","2010-01-01","2020-01-01"),
      ("2020-2026","2020-01-01","2026-12-31"),
      ("FULL since 2005","2005-01-01","2026-12-31")]
print(f"\n{'era':<28}{'strategy':<26}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}")
for nm, lo, hi in eras:
    for sn, s in [("COMBO basket", combo_s), ("Buy&hold basket", hold_s), ("Buy&hold SPY", spy_s)]:
        m = met(s, pd.Timestamp(lo), pd.Timestamp(hi))
        if m: print(f"{nm:<28}{sn:<26}{m[0]*100:7.1f}%{m[1]:8.2f}{m[2]*100:7.0f}%")
    print()
