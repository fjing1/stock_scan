#!/usr/bin/env python3
"""Research-grounded swing strategies on SPY/QQQ, full history, by era, vs buy & hold.
Long-only, cash, no shorts/options/leverage. Costs 10bps round-trip (liquid ETFs).
Specs from research (Connors/Alvarez, Faber, Pagonidis):
  - MR (best practice): RSI(2)<5 AND IBS<0.2, ONLY above 200-SMA; exit first up-close OR 10d stop.
  - MR-IBS (alt A): IBS<0.2 above 200-SMA; exit IBS>0.5 OR first up-close OR 10d.
  - Faber trend-timing: hold when Close>200-SMA (3-day confirm), else cash. (documented full-cycle B&H beater)
  - Faber + MR: trend-hold above 200MA, cash below (single asset => equals Faber; basket version later).
"""
import numpy as np, pandas as pd, scan_stocks as scan
COST, TD = 0.0010, 252

def rsi(x, n):
    d = x.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    return 100 - 100/(1 + up.ewm(alpha=1/n, adjust=False).mean()/dn.ewm(alpha=1/n, adjust=False).mean())

def build(df):
    C = df["Close"].astype(float); H = df["High"].astype(float); L = df["Low"].astype(float)
    return dict(C=C, r2=rsi(C,2).values, ibs=((C-L)/(H-L).replace(0,np.nan)).fillna(0.5).values,
                s200=C.rolling(200).mean().values, s5=C.rolling(5).mean().values, c=C.values, idx=C.index)

def mr_pos(b, use_rsi=True, use_ibs=True, ibs_exit=False):
    c, r2, ibs, s200, s5 = b["c"], b["r2"], b["ibs"], b["s200"], b["s5"]
    pos = np.zeros(len(c)); i = 200
    while i < len(c)-1:
        entry = (np.isfinite(s200[i]) and c[i] > s200[i])
        if use_rsi: entry &= r2[i] < 5
        if use_ibs: entry &= ibs[i] < 0.2
        if entry:
            held = 0; j = i
            pos[i] = 1.0
            j = i+1
            while j < len(c):
                held += 1; pos[j] = 1.0
                up_close = c[j] > c[j-1]
                ex = up_close or held >= 10 or (ibs_exit and ibs[j] > 0.5)
                if ex: break
                j += 1
            i = j+1
        else:
            i += 1
    return pd.Series(pos, index=b["idx"])

def faber_pos(b, confirm=3):
    C = b["C"]; above = (C > C.rolling(200).mean())
    conf = above.rolling(confirm).sum() == confirm   # need `confirm` consecutive closes above
    return conf.astype(float)

def combo_pos(b, confirm=3, mr_rsi=10):
    """Hold the trend above the 200MA (Faber); below it, trade MR oversold bounces
    (RSI2<mr_rsi, exit first up-close or 10d) instead of sitting in cash."""
    c, r2, s200 = b["c"], b["r2"], b["s200"]
    C = b["C"]; above3 = ((C > C.rolling(200).mean()).rolling(confirm).sum() == confirm).values
    pos = np.zeros(len(c)); i = 200
    while i < len(c)-1:
        if above3[i]:
            pos[i] = 1.0; i += 1
        elif np.isfinite(s200[i]) and c[i] < s200[i] and r2[i] < mr_rsi:
            pos[i] = 1.0; j = i+1; held = 0
            while j < len(c):
                held += 1; pos[j] = 1.0
                if c[j] > c[j-1] or held >= 10: break
                j += 1
            i = j+1
        else:
            i += 1
    return pd.Series(pos, index=b["idx"])

def metrics(pos, ret, lo, hi):
    p = pos.shift(1).fillna(0.0); turn = p.diff().abs().fillna(0.0)
    r = (p*ret - turn*COST); r = r[(r.index>=lo)&(r.index<hi)]
    if len(r) < 60: return None
    eq=(1+r).cumprod(); n=len(r); cagr=eq.iloc[-1]**(TD/n)-1 if eq.iloc[-1]>0 else -1
    dd=(eq/eq.cummax()-1).min(); sh=r.mean()/r.std()*np.sqrt(TD) if r.std()>0 else 0
    return dict(cagr=cagr, sharpe=sh, maxdd=dd, tim=(p[(p.index>=lo)&(p.index<hi)]>0).mean())

eras = [("2000-2009 (bear/sideways)","2000-01-01","2010-01-01"),
        ("2010-2019 (bull)","2010-01-01","2020-01-01"),
        ("2020-2026 (bull+2022)","2020-01-01","2026-12-31"),
        ("FULL since 2000","2000-01-01","2026-12-31")]

for tk in ("SPY","QQQ"):
    df = scan.download_daily(tk, period="max"); b = build(df); ret = b["C"].pct_change().fillna(0.0)
    strat = {
      "Buy & Hold": pd.Series(1.0, index=b["idx"]),
      "Faber trend-time (3d confirm)": faber_pos(b),
      "MR RSI2<5 & IBS<.2 (>200MA)": mr_pos(b),
      "COMBO hold>200MA + MR-bounce<200MA": combo_pos(b),
    }
    print(f"\n===== {tk}  ({b['idx'][0].date()}→{b['idx'][-1].date()}) — research-grounded, vs buy&hold =====")
    for nm, lo, hi in eras:
        print(f"\n  [{nm}]")
        for sn, pos in strat.items():
            m = metrics(pos, ret, pd.Timestamp(lo), pd.Timestamp(hi))
            if m: print(f"    {sn:<32}CAGR {m['cagr']*100:6.1f}%  Sharpe {m['sharpe']:5.2f}  MaxDD {m['maxdd']*100:6.1f}%  TiM {m['tim']*100:3.0f}%")
