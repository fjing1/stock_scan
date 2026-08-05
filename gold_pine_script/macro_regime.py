"""Macro-regime / risk-on-off research using Yahoo-reachable proxies.

(FRED's official series — ISM, NFCI, CPI — unlock after a Claude Code restart picks
up the new allowlist entry. Until then these tradeable proxies cover the same factors.)

Regime signals (all from Yahoo):
  yield curve : ^TNX - ^IRX        (10y - 13w; inversion = late-cycle/recession risk)
  credit      : HYG / LQD ratio    (high-yield vs IG; rising = risk appetite)
  cyc/def     : XLY / XLP ratio     (discretionary vs staples; rising = risk-on)
  trend       : SPY > 200-DMA

RISK-ON score (0-3) = [credit ratio > its 50d MA] + [XLY/XLP > its 50d MA] + [SPY>200DMA].

Tests:
  1. forward SPY return conditional on the risk-on score (does the regime predict?)
  2. timing: hold SPY only when risk-on (score>=2) vs buy & hold — CAGR / Sharpe / maxDD
  3. cyclicals vs defensives spread, conditional on regime

    python macro_regime.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yfinance as yf  # noqa: E402

SPLIT = pd.Timestamp("2015-01-01")
CYC = ["XLY", "XLK", "XLI", "XLF", "XLB"]
DEF = ["XLP", "XLU", "XLV"]


def fetch(symbols):
    raw = yf.download(symbols, period="max", interval="1d", auto_adjust=True,
                      progress=False, group_by="ticker", threads=True)
    cols = {}
    for s in symbols:
        try:
            d = raw[s]["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
        except Exception:
            continue
        d = d.dropna()
        if getattr(d.index, "tz", None) is not None:
            d.index = d.index.tz_localize(None)
        if len(d):
            cols[s] = d
    return pd.DataFrame(cols).sort_index()


def maxdd(equity):
    eq = np.asarray(equity, float)
    peak = np.maximum.accumulate(eq)
    return (eq / peak - 1).min() * 100


def main():
    syms = ["^TNX", "^IRX", "HYG", "LQD", "SPY"] + CYC + DEF
    print("downloading macro proxies + sectors ...", flush=True)
    c = fetch(list(dict.fromkeys(syms)))
    print(f"span {c.index[0].date()} -> {c.index[-1].date()}")

    yc = (c["^TNX"] - c["^IRX"])
    credit = (c["HYG"] / c["LQD"])
    cd = (c[[s for s in CYC if s in c][0]] / c["XLP"]) if "XLY" not in c else (c["XLY"] / c["XLP"])
    spy = c["SPY"]

    on = (((credit > credit.rolling(50).mean()).astype(int)
           + (cd > cd.rolling(50).mean()).astype(int)
           + (spy > spy.rolling(200).mean()).astype(int)))
    on = on.reindex(c.index).ffill()

    # restrict to the window where ALL proxies exist (HYG ~2007); else early decades
    # with missing credit/cyc-def data fake a score of 0 and contaminate everything.
    valid = (credit.rolling(50).mean().notna() & cd.rolling(50).mean().notna()
             & spy.rolling(200).mean().notna())
    idx0 = c.index[valid][0]
    c, on, spy, credit, cd, yc = (x.loc[idx0:] for x in (c, on, spy, credit, cd, yc))
    print(f"\nregime window (all proxies live): {idx0.date()} -> {c.index[-1].date()}")

    print(f"\ncurrent: 10y-13w curve {yc.iloc[-1]:+.2f}%  | HYG/LQD {'rising' if credit.iloc[-1]>credit.rolling(50).mean().iloc[-1] else 'falling'}"
          f"  | XLY/XLP {'rising' if cd.iloc[-1]>cd.rolling(50).mean().iloc[-1] else 'falling'}"
          f"  | SPY {'>' if spy.iloc[-1]>spy.rolling(200).mean().iloc[-1] else '<'}200DMA  -> RISK-ON score {int(on.iloc[-1])}/3")

    # ---- 1) forward SPY return by regime ----
    fwd21 = spy.shift(-21) / spy - 1
    fwd63 = spy.shift(-63) / spy - 1
    print("\n================ 1) forward SPY return by RISK-ON score ================")
    print(f"{'score':<7}{'days%':>7}{'fwd1m%':>8}{'1m +%':>7}{'fwd3m%':>9}{'3m +%':>7}")
    for s in [0, 1, 2, 3]:
        m = on == s
        n = m.sum()
        if n < 50:
            continue
        f1, f3 = fwd21[m].dropna(), fwd63[m].dropna()
        print(f"{s:<7}{n/len(on)*100:>7.0f}{f1.mean()*100:>8.2f}{(f1>0).mean()*100:>7.0f}"
              f"{f3.mean()*100:>9.2f}{(f3>0).mean()*100:>7.0f}")

    # ---- 2) timing: SPY only when risk-on (score>=2) ----
    ret = spy.pct_change()
    pos = (on.shift(1) >= 2).astype(float)        # lag signal -> no lookahead
    strat = ret * pos
    valid = strat.dropna().index
    print("\n================ 2) timing — long SPY when risk-on (score>=2) vs buy&hold ================")
    print(f"{'window':<14}{'strat CAGR%':>12}{'B&H CAGR%':>11}{'strat Sh':>9}{'B&H Sh':>8}{'strat maxDD%':>13}{'B&H maxDD%':>11}{'%invested':>10}")
    for label, mask in [("full", valid),
                        ("train<2015", valid[valid < SPLIT]),
                        ("test>=2015", valid[valid >= SPLIT])]:
        r_s, r_b = strat.loc[mask], ret.loc[mask]
        yrs = len(mask) / 252
        cagr_s = ((1 + r_s).prod()) ** (1 / yrs) - 1
        cagr_b = ((1 + r_b).prod()) ** (1 / yrs) - 1
        sh_s = r_s.mean() / r_s.std() * np.sqrt(252)
        sh_b = r_b.mean() / r_b.std() * np.sqrt(252)
        dd_s = maxdd((1 + r_s).cumprod())
        dd_b = maxdd((1 + r_b).cumprod())
        inv = pos.loc[mask].mean() * 100
        print(f"{label:<14}{cagr_s*100:>12.1f}{cagr_b*100:>11.1f}{sh_s:>9.2f}{sh_b:>8.2f}"
              f"{dd_s:>13.1f}{dd_b:>11.1f}{inv:>10.0f}")

    # ---- 3) cyclicals vs defensives by regime ----
    cyc_av = c[[s for s in CYC if s in c]].pct_change(21).mean(axis=1)
    def_av = c[[s for s in DEF if s in c]].pct_change(21).mean(axis=1)
    spread = (cyc_av - def_av)            # >0 means cyclicals leading over last month
    fwd_spread = spread.shift(-21)         # next month cyc-def
    print("\n================ 3) cyclicals-minus-defensives NEXT-month return by regime ================")
    print(f"{'score':<7}{'next cyc-def %':>15}{'cyc>def %':>11}")
    for s in [0, 1, 2, 3]:
        m = (on == s)
        v = fwd_spread[m].dropna()
        if len(v) < 50:
            continue
        print(f"{s:<7}{v.mean()*100:>15.2f}{(v>0).mean()*100:>11.0f}")
    print("\nReads: monotonic fwd return by score => regime predicts. Timing wins if it cuts")
    print("maxDD a lot while keeping CAGR/Sharpe. Cyc-def +ve in high score => risk-on tilt works.")


if __name__ == "__main__":
    main()
