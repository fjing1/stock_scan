#!/usr/bin/env python3
"""(#1) Does 观海买点分 (buy-score) have CROSS-SECTIONAL selection skill?

For every formal-buy entry compute the buy-score, then:
 (A) Information coefficient — Spearman(score, forward return) where forward returns
     are de-meaned by the universe's same-day mean (market/drift removed). Pooled + per-day.
 (B) Expectancy by score quintile (does higher score -> higher per-trade return?).
 (C) Portfolio: top-decile-by-score entries + 5xATR trail, vs all-entries baseline vs SPY.
Also runs the same for Rank120 alone (relative-strength selector) as a comparison.
"""
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd

import scan_stocks as scan
import _exit_backtest as bt

PANEL = scan.BASE_DIR / "reports" / "exit_cache" / "panel_full.pkl"
panel = pickle.load(open(PANEL, "rb"))
spy = scan.download_daily("SPY", period="8y")
COST, TD = 0.0030, bt.TRADING_DAYS
FWD = 10  # forward horizon for IC (trading days)

# universe same-day mean forward return (market proxy to de-mean) ----
# build per-symbol fwd return aligned to a common calendar, average across names
cal = pd.DatetimeIndex(sorted(set().union(*[set(panel[s].index) for s in panel])))
fwd_wide = {}
for sym, df in panel.items():
    C = df["Close"]
    fwd_wide[sym] = (C.shift(-FWD) / C - 1).reindex(cal)
fwd_df = pd.DataFrame(fwd_wide)
univ_fwd = fwd_df.mean(axis=1, skipna=True)   # equal-weight universe forward return per date

# ---- per-entry: score + features + de-meaned forward return ----
rows = []
for sym, df in panel.items():
    C = df["Close"].values.astype(float)
    idx = df.index
    r120 = df["Rank120"].values; rsi = df["RSI"].values; l2 = df["L2_trend"].values
    fwd = np.r_[C[FWD:] / C[:-FWD] - 1, np.full(FWD, np.nan)]
    for e in np.flatnonzero(df["BUY_A"].values.astype(bool)):
        if e < 110 or not np.isfinite(fwd[e]):
            continue
        row = {"signal_side": "BUY", "signal_type": "正式买入", "model": "D1_BUY_A_0出",
               "rank120": r120[e], "RSI": rsi[e], "L2_trend": l2[e]}
        score = scan.score_buy_signal_row(pd.Series(row))
        d = idx[e]
        dem = fwd[e] - (univ_fwd.get(d, np.nan))
        rows.append({"symbol": sym, "date": d, "loc": int(e), "score": score,
                     "rank120": r120[e], "fwd": fwd[e], "dem_fwd": dem})
t = pd.DataFrame(rows).dropna(subset=["score", "dem_fwd"])
print(f"entries scored: {len(t)}  | score range {t.score.min():.0f}-{t.score.max():.0f}  mean {t.score.mean():.1f}")

# ---- (A) Information Coefficient (drift-removed) ----
def spearman(a, b):
    a = pd.Series(np.asarray(a, float)); b = pd.Series(np.asarray(b, float))
    m = a.notna() & b.notna()
    if m.sum() < 5:
        return np.nan
    ra, rb = a[m].rank(), b[m].rank()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])

def ic(col):
    pooled = spearman(t[col], t["dem_fwd"])
    per = [spearman(g[col], g["dem_fwd"]) for _, g in t.groupby("date") if len(g) > 4]
    per = np.array([x for x in per if np.isfinite(x)])
    return pooled, per.mean(), per.std()
for col in ["score", "rank120"]:
    p, dm, ds = ic(col)
    print(f"  IC[{col:7}] pooled Spearman={p:+.3f}   mean daily IC={dm:+.3f} (std {ds:.3f})")
print("  (rank120 is inverted: low rank = strong; expect NEGATIVE IC if it selects)")

# ---- (B) expectancy by score quintile (using actual 5xATR-trail per-trade returns) ----
def sim_ret(sym, e):
    df = panel[sym]
    a = {k: df[c].values.astype(float) for k, c in {"O":"Open","H":"High","L":"Low","C":"Close","ATR22":"ATR22"}.items()}
    a["SELL_1"] = df["SELL_1"].values.astype(bool); a["ATR22_prev"] = df["ATR22"].shift(1).values.astype(float)
    r = bt.simulate(a, e, "wide_atr_trail", {"m":5.0,"atr_col":"ATR22"})
    return r if r is None else (r[0], r[1], r[2])

t["q"] = pd.qcut(t["score"], 5, labels=["Q1(low)","Q2","Q3","Q4","Q5(high)"], duplicates="drop")
strat = {}
for _, r in t.iterrows():
    res = sim_ret(r["symbol"], r["loc"])
    if res: strat[(r["symbol"], r["loc"])] = res
t["strat_ret"] = t.apply(lambda r: strat.get((r["symbol"], r["loc"]), (np.nan,))[0], axis=1)
print("\n(B) per-trade expectancy by score quintile (5xATR trail):")
qb = t.groupby("q", observed=True).agg(n=("score","size"), score=("score","mean"),
       exp=("strat_ret","mean"), demfwd=("dem_fwd","mean"))
qb["exp"]=(qb["exp"]*100).round(2); qb["demfwd"]=(qb["demfwd"]*100).round(2); qb["score"]=qb["score"].round(1)
print(qb.to_string())

# ---- (C) top-decile-by-score portfolio vs baseline vs SPY ----
thr = t["score"].quantile(0.90)
def portfolio(mask_df):
    daily = defaultdict(list); n=0
    for _, r in mask_df.iterrows():
        res = strat.get((r["symbol"], r["loc"]))
        if not res: continue
        ret, hold, px = res; n+=1
        df=panel[r["symbol"]]; C=df["Close"].values.astype(float); idx=df.index; e=r["loc"]
        for k in range(1,hold+1):
            bar=e+k
            if bar>=len(C): break
            prev=C[bar-1]; daily[idx[bar]].append(max(px/prev-1-COST,-0.99) if k==hold else C[bar]/prev-1)
    return bt.portfolio_metrics(daily, cal), n

m_all, n_all = portfolio(t)
m_top, n_top = portfolio(t[t["score"]>=thr])
m_topr, n_topr = portfolio(t[t["rank120"]<=t["rank120"].quantile(0.10)])  # strongest RS decile
spw=spy.loc[(spy.index>=cal[0])&(spy.index<=cal[-1])]; sr=spw["Close"].pct_change().dropna().values; sq=np.cumprod(1+sr)
spy_m=dict(cagr=sq[-1]**(TD/len(sr))-1, sharpe=sr.mean()/sr.std(ddof=1)*np.sqrt(TD),
           maxdd=((sq-np.maximum.accumulate(sq))/np.maximum.accumulate(sq)).min())
print("\n(C) PORTFOLIO: select by score vs baseline vs SPY (5xATR trail, 30bps)")
print(f"{'variant':<34}{'entries':>9}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>9}")
for nm,(m,n) in [("All formal-buys (baseline)",(m_all,n_all)),
                 (f"Top-decile by score (>={thr:.0f})",(m_top,n_top)),
                 ("Top-decile by Rank120 (RS)",(m_topr,n_topr))]:
    print(f"{nm:<34}{n:>9}{m['cagr']*100:7.1f}%{m['sharpe']:8.2f}{m['maxdd']*100:8.1f}%")
print(f"{'Buy & hold SPY':<34}{'-':>9}{spy_m['cagr']*100:7.1f}%{spy_m['sharpe']:8.2f}{spy_m['maxdd']*100:8.1f}%")
