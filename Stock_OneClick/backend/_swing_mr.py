#!/usr/bin/env python3
"""Swing trading where it should work: FILTER to mean-reverting / choppy / liquid stocks
(by behavior, not returns), then run a long-only cash mean-reversion swing strategy and
test vs buy-&-hold those same names + SPY. Contrast with trenders (where hold should win).
Constraints: long only, cash (0-100%), no shorts/options/leverage. Cost 30bps round-trip.
"""
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
import scan_stocks as scan
import _exit_backtest as bt

panel = pickle.load(open(scan.BASE_DIR / "reports" / "exit_cache" / "panel_full.pkl", "rb"))
spy = scan.download_daily("SPY", period="8y")
COST, TD, OOS = 0.0030, 252, pd.Timestamp("2024-01-01")

def rsi(x, n):
    d = x.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    return 100 - 100/(1 + up.ewm(alpha=1/n, adjust=False).mean()/dn.ewm(alpha=1/n, adjust=False).mean())

# ---- per-stock behavioral features for the filter (no return/drift used) ----
feat = {}
for sym, df in panel.items():
    C = df["Close"].astype(float); V = df["Volume"].astype(float); r = C.pct_change().dropna()
    if len(r) < 300: continue
    er = ((C - C.shift(20)).abs() / C.diff().abs().rolling(20).sum()).replace([np.inf,-np.inf], np.nan).mean()
    feat[sym] = dict(autocorr1=np.corrcoef(r.values[:-1], r.values[1:])[0,1],
                     ER=er, vol=r.std()*np.sqrt(TD), px=float(C.iloc[-1]), dvol=float((C*V).median()))
f = pd.DataFrame(feat).T
liquid = (f.px > 5) & (f.dvol > 20e6) & (f.vol.between(0.30, 1.5))
swing_univ = f.index[liquid & (f.autocorr1 < f.autocorr1.quantile(.33)) & (f.ER < f.ER.median())]
trend_univ = f.index[liquid & (f.ER > f.ER.quantile(.66)) & (f.autocorr1 > 0)]
print(f"swing-friendly (mean-reverting/choppy/liquid): {len(swing_univ)} names | trenders (contrast): {len(trend_univ)}")
print("swing-univ sample:", list(swing_univ[:15]))

# ---- mean-reversion swing strategy (long-only, cash) ----
def mr_trades(df, entry=10, exit_rsi=70, maxd=10, stop=0.10, regime=True):
    C = df["Close"].astype(float).values; idx = df.index
    r2 = rsi(df["Close"].astype(float), 2).values
    sma5 = df["Close"].astype(float).rolling(5).mean().values
    sma200 = df["Close"].astype(float).rolling(200).mean().values
    daily = []  # (date, ret) while in position
    i, n = 200, len(C)
    while i < n - 1:
        ok = r2[i] < entry and (not regime or (np.isfinite(sma200[i]) and C[i] > sma200[i]))
        if ok:
            ent = C[i]; held = 0; j = i + 1
            while j < n:
                held += 1
                prev = C[j-1]
                ex = (r2[j] > exit_rsi) or (C[j] > sma5[j]) or (held >= maxd) or (C[j] < ent*(1-stop))
                px = C[j]
                dr = px/prev - 1 - (COST if ex else 0.0)
                daily.append((idx[j], max(dr, -0.99)))
                if ex: break
                j += 1
            i = j + 1
        else:
            i += 1
    return daily

def basket(univ, **kw):
    d = defaultdict(list)
    for sym in univ:
        for dt, rr in mr_trades(panel[sym], **kw):
            d[dt].append(rr)
    return d

def bh_basket(univ):
    d = defaultdict(list)
    for sym in univ:
        C = panel[sym]["Close"].astype(float); rr = C.pct_change()
        for dt, x in rr.items():
            if np.isfinite(x): d[dt].append(x)
    return d

cal = pd.DatetimeIndex(sorted(set().union(*[set(panel[s].index) for s in panel])))
def met(d, lo=None, hi=None):
    c = cal.copy()
    if lo is not None: c = c[c >= lo]
    if hi is not None: c = c[c < hi]
    port = np.array([np.mean(d[x]) if x in d else 0.0 for x in c]); act = np.array([x in d for x in c])
    eq = np.cumprod(1+port); n = len(c)
    cagr = eq[-1]**(TD/n)-1 if eq[-1]>0 else -1
    dd = ((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq)).min()
    sh = port.mean()/port.std()*np.sqrt(TD) if port.std()>0 else 0
    return dict(cagr=cagr, sharpe=sh, maxdd=dd, tim=act.mean())

def spym(lo=None, hi=None):
    s = spy.loc[(spy.index>=cal[0])]; r = s["Close"].pct_change().reindex(cal).fillna(0.0)
    c = cal.copy()
    if lo is not None: c=c[c>=lo]
    if hi is not None: c=c[c<hi]
    r=r[r.index.isin(c)].values; eq=np.cumprod(1+r)
    return dict(cagr=eq[-1]**(TD/len(r))-1, sharpe=r.mean()/r.std()*np.sqrt(TD),
                maxdd=((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq)).min(), tim=1.0)

# build once (full + split via metric windows)
swing_mr = basket(swing_univ, entry=10, regime=True)
swing_hold = bh_basket(swing_univ)
trend_mr = basket(trend_univ, entry=10, regime=True)
trend_hold = bh_basket(trend_univ)

def show(title, lo, hi):
    print("\n" + "="*94); print(title); print("="*94)
    print(f"{'strategy':<46}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>9}{'TiM':>6}")
    for nm, d in [("MR swing on SWING-friendly names", swing_mr),
                  ("  Buy&hold those swing names", swing_hold),
                  ("MR swing on TRENDERS (should lose)", trend_mr),
                  ("  Buy&hold those trenders", trend_hold)]:
        m = met(d, lo, hi)
        print(f"{nm:<46}{m['cagr']*100:7.1f}%{m['sharpe']:8.2f}{m['maxdd']*100:8.1f}%{m['tim']*100:5.0f}%")
    s = spym(lo, hi); print(f"{'Buy & hold SPY':<46}{s['cagr']*100:7.1f}%{s['sharpe']:8.2f}{s['maxdd']*100:8.1f}%{'100':>5}%")

show("FULL SAMPLE (2020-2026)", None, None)
show(f"IN-SAMPLE (< {OOS.date()})", None, OOS)
show(f"OUT-OF-SAMPLE (>= {OOS.date()})", OOS, None)
