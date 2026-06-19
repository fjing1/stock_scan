#!/usr/bin/env python3
"""Cash-only (0-100%, long-only, no margin/shorts/options) AAPL strategy for HIGH risk.
Can't beat buy-hold's RETURN at 100% cap, so tune vol-targeting to stay near fully
invested (high target vol) and shed exposure only in genuine turbulence -> keep most
of the return, cut the worst drawdowns. Sweep target vol; full 46y + OOS (>=2018).
"""
import numpy as np
import pandas as pd
import scan_stocks as scan

COST_TURN, TD, OOS = 0.0007, 252, pd.Timestamp("2018-01-01")
df = scan.download_daily("AAPL", period="max")
C = df["Close"].astype(float); ret = C.pct_change().fillna(0.0)
vol30 = ret.rolling(30).std() * np.sqrt(TD)

strats = {"Buy & Hold AAPL (100%)": pd.Series(1.0, index=C.index)}
for tgt in (0.25, 0.30, 0.35, 0.40, 0.45):
    strats[f"Vol-target {int(tgt*100)}% (cap 100%)"] = (tgt / vol30).clip(0, 1.0)

def metrics(pos, lo=None, hi=None):
    p = pos.shift(1).fillna(0.0); turn = p.diff().abs().fillna(0.0)
    r = p * ret - turn * COST_TURN
    m = pd.Series(True, index=r.index)
    if lo is not None: m &= r.index >= lo
    if hi is not None: m &= r.index < hi
    r = r[m]
    if len(r) < 60: return None
    eq = (1+r).cumprod(); n = len(r)
    cagr = eq.iloc[-1]**(TD/n)-1
    dd = (eq/eq.cummax()-1).min()
    return dict(cagr=cagr, vol=r.std()*np.sqrt(TD), sharpe=r.mean()/r.std()*np.sqrt(TD),
                maxdd=dd, mar=cagr/abs(dd), avgexp=p[m].mean(), retcap=cagr)

for title, lo, hi in [("FULL SAMPLE (46y)", None, None), ("OUT-OF-SAMPLE (>=2018)", OOS, None)]:
    print("\n" + "="*92)
    print(f"{title} — cash-only AAPL (max 100%), ranked by CAGR")
    print("="*92)
    rows = {nm: metrics(p, lo, hi) for nm, p in strats.items()}
    r = pd.DataFrame({k:v for k,v in rows.items() if v}).T.sort_values("cagr", ascending=False)
    d = r.copy()
    for c in ["cagr","vol","maxdd","avgexp"]: d[c]=(d[c]*100).round(0)
    d["sharpe"]=d["sharpe"].round(2); d["mar"]=d["mar"].round(2)
    print(d[["cagr","vol","sharpe","maxdd","mar","avgexp"]].to_string())

# % of buy-hold return retained, OOS
bh = metrics(strats["Buy & Hold AAPL (100%)"], lo=OOS)
print("\nOOS return retained vs buy-hold, and drawdown saved:")
for tgt in (0.35, 0.40, 0.45):
    s = metrics(strats[f"Vol-target {int(tgt*100)}% (cap 100%)"], lo=OOS)
    print(f"  VT{int(tgt*100)}%: keeps {s['cagr']/bh['cagr']*100:.0f}% of buy-hold CAGR, "
          f"DD {s['maxdd']*100:.0f}% vs {bh['maxdd']*100:.0f}%, Sharpe {s['sharpe']:.2f} vs {bh['sharpe']:.2f}, avg exp {s['avgexp']*100:.0f}%")

# ---- current actionable exposure ----
cur_vol = vol30.iloc[-1]
print(f"\nCURRENT SIGNAL (as of {C.index[-1].date()}): AAPL 30d realized vol = {cur_vol*100:.0f}%")
for tgt in (0.35, 0.40, 0.45):
    print(f"  VT{int(tgt*100)}% -> target AAPL weight = {min(tgt/cur_vol,1.0)*100:.0f}% (rest in cash)")
