#!/usr/bin/env python3
"""Is mean-reversion swing a REGIME tool? Test RSI2 MR vs buy-hold on SPY & QQQ over
the longest available history, split by era — expect MR to win in sideways/bear decades
(2000-2009) and lose in bull decades (2010s, 2020s). Long-only cash, 30bps round-trip.
"""
import numpy as np, pandas as pd, scan_stocks as scan
COST, TD = 0.0010, 252   # index ETFs: tight 10bps

def rsi(x, n):
    d = x.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    return 100 - 100/(1 + up.ewm(alpha=1/n, adjust=False).mean()/dn.ewm(alpha=1/n, adjust=False).mean())

def mr_daily(C, entry=10, exit_rsi=70, maxd=10, stop=0.10, regime=True):
    c = C.values; idx = C.index
    r2 = rsi(C, 2).values; s5 = C.rolling(5).mean().values; s200 = C.rolling(200).mean().values
    pos = np.zeros(len(c)); i = 200
    while i < len(c) - 1:
        if r2[i] < entry and (not regime or (np.isfinite(s200[i]) and c[i] > s200[i])):
            ent = c[i]; held = 0; j = i + 1
            while j < len(c):
                held += 1; pos[j] = 1.0
                if (r2[j] > exit_rsi) or (c[j] > s5[j]) or (held >= maxd) or (c[j] < ent*(1-stop)):
                    break
                j += 1
            i = j + 1
        else:
            i += 1
    return pd.Series(pos, index=idx)

def stats(ret, lo, hi):
    r = ret[(ret.index >= lo) & (ret.index < hi)]
    if len(r) < 60: return None
    eq = (1+r).cumprod(); n = len(r)
    cagr = eq.iloc[-1]**(TD/n)-1 if eq.iloc[-1] > 0 else -1
    dd = (eq/eq.cummax()-1).min()
    return cagr, (r.mean()/r.std()*np.sqrt(TD) if r.std() > 0 else 0), dd

eras = [("2000-2009 (dot-com bust + GFC, sideways/bear)", "2000-01-01", "2010-01-01"),
        ("2010-2019 (bull)", "2010-01-01", "2020-01-01"),
        ("2020-2026 (bull + 2022)", "2020-01-01", "2026-12-31"),
        ("FULL since 2000", "2000-01-01", "2026-12-31")]

for tk in ("SPY", "QQQ"):
    df = scan.download_daily(tk, period="max"); C = df["Close"].astype(float)
    bh = C.pct_change().fillna(0.0)
    pos = mr_daily(C).shift(1).fillna(0.0); turn = pos.diff().abs().fillna(0.0)
    mr = pos * bh - turn * COST
    print(f"\n===== {tk}  ({C.index[0].date()}→{C.index[-1].date()}) — MR swing vs buy&hold by era =====")
    print(f"{'era':<46}{'MR CAGR':>9}{'MR Shp':>8}{'  | ':>4}{'BH CAGR':>9}{'BH Shp':>8}{'  winner':>10}")
    for nm, lo, hi in eras:
        m = stats(mr, pd.Timestamp(lo), pd.Timestamp(hi)); b = stats(bh, pd.Timestamp(lo), pd.Timestamp(hi))
        if m and b:
            win = "MR" if m[0] > b[0] else "hold"
            print(f"{nm:<46}{m[0]*100:8.1f}%{m[1]:8.2f}{'  | ':>4}{b[0]*100:8.1f}%{b[1]:8.2f}{win:>10}")
