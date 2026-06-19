#!/usr/bin/env python3
"""Compare the chosen exit strategy (formal-buy entry + 5xATR(22) trailing stop,
no target) against buy-and-hold SPY over the same period.

Reuses the simulation primitives from _exit_backtest.py and the cached panel.
Shows: aggregate strategy portfolio vs SPY buy&hold (total return, CAGR, Sharpe,
MaxDD, MAR), the per-trade stats, and concrete example trades incl. NTR.
"""
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

import scan_stocks as scan
import _exit_backtest as bt   # simulate(), portfolio_metrics(), HORIZON, COST, TRADING_DAYS

STRAT, PARAMS, LABEL = "wide_atr_trail", {"m": 5.0, "atr_col": "ATR22"}, "Formal-buy + 5xATR(22) trailing stop"

with open(bt.PANEL, "rb") as f:
    panel = pickle.load(f)

# build arrays exactly like the backtest
arrays, entries = {}, []
for sym, df in panel.items():
    a = {k: df[c].values.astype(float) for k, c in
         {"O": "Open", "H": "High", "L": "Low", "C": "Close",
          "ATR14": "ATR14", "ATR22": "ATR22", "EMA20": "EMA20",
          "SMA50": "SMA50", "SMA100": "SMA100", "SAR_STD": "SAR_STD", "SAR_SLOW": "SAR_SLOW"}.items()}
    a["SELL_1"] = df["SELL_1"].values.astype(bool)
    a["ATR14_prev"] = df["ATR14"].shift(1).values.astype(float)
    a["ATR22_prev"] = df["ATR22"].shift(1).values.astype(float)
    a["DONCH10"] = df["DONCH10"].shift(1).values.astype(float)
    a["DONCH20"] = df["DONCH20"].shift(1).values.astype(float)
    arrays[sym] = a
    for e in np.flatnonzero(df["BUY_A"].values.astype(bool)):
        if e >= 110:
            entries.append((sym, int(e)))

# run the winning strategy: collect per-trade results + daily portfolio path
daily = defaultdict(list)
rets, holds = [], []
trades = []   # (sym, entry_date, entry_px, exit_date, exit_px, ret, hold, reason)
for sym, e in entries:
    res = bt.simulate(arrays[sym], e, STRAT, PARAMS)
    if res is None:
        continue
    ret, hold, px, reason = res
    rets.append(ret); holds.append(hold)
    C = arrays[sym]["C"]; sidx = panel[sym].index
    for k in range(1, hold + 1):
        bar = e + k
        if bar >= len(C):
            break
        prev = C[bar - 1]
        r = (px / prev - 1.0 - bt.COST) if k == hold else (C[bar] / prev - 1.0)
        daily[sidx[bar]].append(r)
    exit_bar = min(e + hold, len(C) - 1)
    trades.append((sym, sidx[e].date(), C[e], sidx[exit_bar].date(), px, ret, hold, reason))

cal = pd.DatetimeIndex(sorted(set().union(*[set(panel[s].index) for s in panel])))
first = min(d for d, _ in [(t[1], 0) for t in trades])
cal = cal[cal >= pd.Timestamp(first)]
strat_m = bt.portfolio_metrics(daily, cal)

# strategy equity span for SPY alignment
port_dates = sorted(daily.keys())
lo, hi = port_dates[0], port_dates[-1]

# ---- SPY buy & hold over the same span ----
spy = scan.download_daily("SPY", period="8y")
spy = spy.loc[(spy.index >= pd.Timestamp(lo)) & (spy.index <= pd.Timestamp(hi))]
spx_ret = spy["Close"].pct_change().dropna().values
eq = np.cumprod(1 + spx_ret)
yrs = max(len(spx_ret) / bt.TRADING_DAYS, 0.25)
spy_cagr = eq[-1] ** (1 / yrs) - 1
spy_peak = np.maximum.accumulate(eq)
spy_maxdd = ((eq - spy_peak) / spy_peak).min()
spy_sharpe = spx_ret.mean() / spx_ret.std(ddof=1) * np.sqrt(bt.TRADING_DAYS)
spy_total = eq[-1] - 1
spy_metrics = {"total_return": spy_total, "cagr": spy_cagr, "maxdd": spy_maxdd,
               "sharpe": spy_sharpe, "mar": spy_cagr / abs(spy_maxdd)}

rets = np.array(rets)
print("=" * 78)
print(f"STRATEGY vs BUY-AND-HOLD SPY    ({lo.date()} → {hi.date()}, {yrs:.1f}y, cost={bt.COST:.2%}/trade)")
print("=" * 78)
print(f"Strategy: {LABEL}")
print(f"Universe: {len(panel)} names (the enabled scan list) | entries: {len(rets)} formal buys\n")

def row(name, m):
    return (f"{name:<34} {m['total_return']*100:9.0f}%  {m['cagr']*100:7.1f}%  "
            f"{m['sharpe']:6.2f}  {m['maxdd']*100:7.1f}%  {m['mar']:5.2f}")

print(f"{'':34} {'TotalRet':>10} {'CAGR':>8} {'Sharpe':>7} {'MaxDD':>8} {'MAR':>6}")
print(row("Strategy (equal-wt basket)", strat_m))
print(row("Buy & hold SPY", spy_metrics))
print(f"\nExcess CAGR over SPY: {(strat_m['cagr']-spy_cagr)*100:+.1f} pts/yr   "
      f"Sharpe edge: {strat_m['sharpe']-spy_sharpe:+.2f}")

wins = rets[rets > 0]; losses = rets[rets < 0]
print("\n--- per-trade stats ---")
print(f"trades={len(rets)}  win%={100*(rets>0).mean():.1f}  avg_win={wins.mean()*100:.1f}%  "
      f"avg_loss={losses.mean()*100:.1f}%  payoff={abs(wins.mean()/losses.mean()):.2f}  "
      f"avg_hold={np.mean(holds):.0f}d  expectancy={rets.mean()*100:.2f}%/trade")

# ---- concrete example trades ----
def show(sym):
    ts = [t for t in trades if t[0] == sym]
    if not ts:
        print(f"\n{sym}: no trades"); return
    print(f"\n{sym} — last 4 trades (entry → exit):")
    for sym_, ed, ep, xd, xp, r, h, rs in ts[-4:]:
        print(f"  {ed} @ {ep:8.2f}  →  {xd} @ {xp:8.2f}  [{rs:11}] {h:3}d  {r*100:+7.1f}%")

show("NTR"); show("NVDA"); show("AAPL")
print("\n(NTR's last trade exit reason 'horizon' = still open at the 120d cap / latest bar — i.e. not yet stopped.)")
