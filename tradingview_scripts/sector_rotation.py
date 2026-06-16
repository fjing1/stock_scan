"""Sector rotation research — does "sector A up -> sector B next" exist?

Uses the 11 S&P sector SPDRs. Tests three flavors of the rotation idea:

  1. LEAD-LAG: corr(sector A return this week, sector B return NEXT week). If A
     reliably leads B, this is positive. (Skeptic's prior: mostly arbitraged away.)
  2. ROTATION MOMENTUM: each month rank sectors by trailing return, hold the top
     few, measure next-month return vs equal-weight sectors and SPY (train/test).
     This is the documented, tradeable form of rotation.
  3. LEADER TRANSITION: after sector A is the month's top performer, which sector
     leads next, and how often vs its base rate.

    python sector_rotation.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yfinance as yf  # noqa: E402

SECTORS = {
    "XLK": "Technology", "XLF": "Financials", "XLV": "HealthCare",
    "XLY": "Discretionary", "XLP": "Staples", "XLE": "Energy",
    "XLI": "Industrials", "XLB": "Materials", "XLRE": "RealEstate",
    "XLU": "Utilities", "XLC": "CommSvcs",
}
BENCH = "SPY"
SPLIT = pd.Timestamp("2015-01-01")


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
    return pd.DataFrame(cols)


def main():
    syms = list(SECTORS) + [BENCH]
    print(f"downloading {len(syms)} ETFs (max history) ...", flush=True)
    closes = fetch(syms)
    sect = [s for s in SECTORS if s in closes.columns]
    print(f"sectors: {len(sect)}   span {closes.index[0].date()} -> {closes.index[-1].date()}")

    # ---------- 1) LEAD-LAG (weekly) ----------
    wk = closes[sect].resample("W-FRI").last().pct_change().dropna(how="all")
    print("\n================ 1) LEAD-LAG: corr(A this week, B next week) ================")
    lead = pd.DataFrame(index=sect, columns=sect, dtype=float)
    for a in sect:
        for b in sect:
            x, y = wk[a].iloc[:-1].values, wk[b].iloc[1:].values
            m = ~(np.isnan(x) | np.isnan(y))
            lead.loc[a, b] = np.corrcoef(x[m], y[m])[0, 1] if m.sum() > 30 else np.nan
    pairs = [(a, b, lead.loc[a, b]) for a in sect for b in sect if a != b]
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    print("  strongest A->B (A's week predicts B's NEXT week):")
    for a, b, v in pairs[:8]:
        print(f"    {SECTORS[a]:<13}->  {SECTORS[b]:<13}  {v:+.3f}")
    diag = np.nanmean([lead.loc[s, s] for s in sect])
    offd = np.nanmean([lead.loc[a, b] for a in sect for b in sect if a != b])
    print(f"  mean self lag-1 (momentum/reversal): {diag:+.3f}   mean cross lag-1: {offd:+.3f}")
    print("  (|corr|<~0.1 = no exploitable lead-lag; 11x11 search -> some noise expected)")

    # ---------- 2) ROTATION MOMENTUM ----------
    mc = closes[sect].resample("ME").last()
    mret = mc.pct_change()
    spy_m = closes[BENCH].resample("ME").last().pct_change()
    print("\n================ 2) ROTATION MOMENTUM: hold top sectors by trailing return ================")
    print(f"{'lookback':<10}{'topK':>5}{'  ':<2}{'trAnn%':>8}{'teAnn%':>8}{'te>eqw%mo':>10}{'teSharpe':>9}")
    for lb in [3, 6, 12]:
        mom = mc.pct_change(lb)
        for K in [3]:
            rows = []
            idx = mret.index
            for i in range(lb + 1, len(idx) - 1):
                r = mom.iloc[i].dropna()
                if len(r) < 6:
                    continue
                top = list(r.sort_values(ascending=False).index[:K])
                nxt = mret.iloc[i + 1]
                port = float(nxt[top].mean())
                eqw = float(nxt[r.index].mean())
                rows.append((idx[i], port, eqw, float(spy_m.iloc[i + 1])))
            R = pd.DataFrame(rows, columns=["date", "port", "eqw", "spy"])
            tr, te = R[R["date"] < SPLIT], R[R["date"] >= SPLIT]

            def ann(x):
                x = np.asarray(x, float); x = x[~np.isnan(x)]
                return ((1 + x.mean()) ** 12 - 1) * 100 if len(x) else float("nan")

            def sh(x):
                x = np.asarray(x, float); x = x[~np.isnan(x)]
                return (x.mean() / x.std(ddof=1)) * np.sqrt(12) if len(x) > 5 and x.std(ddof=1) > 0 else float("nan")
            beat = (te["port"].values > te["eqw"].values).mean() * 100
            print(f"trail {lb:>2}m {K:>5}{'  ':<2}{ann(tr['port']):>8.1f}{ann(te['port']):>8.1f}{beat:>10.0f}{sh(te['port']):>9.2f}")
    print(f"  reference — equal-weight sectors: teAnn {ann(te['eqw']):.1f}%  Sharpe {sh(te['eqw']):.2f} | "
          f"SPY: teAnn {ann(te['spy']):.1f}%  Sharpe {sh(te['spy']):.2f}")

    # ---------- 3) LEADER TRANSITION ----------
    print("\n================ 3) LEADER TRANSITION: after A leads a month, who leads next ================")
    leader = mret[sect].idxmax(axis=1).dropna()
    base = leader.value_counts(normalize=True) * 100
    trans = {}
    lv = leader.values
    for i in range(len(lv) - 1):
        trans.setdefault(lv[i], []).append(lv[i + 1])
    print(f"  {'after leader':<14}{'most-likely next':<16}{'freq%':>7}{'(base%)':>9}{'n':>5}")
    for a in sect:
        nxts = trans.get(a, [])
        if len(nxts) < 12:
            continue
        vc = pd.Series(nxts).value_counts(normalize=True) * 100
        b = vc.index[0]
        print(f"  {SECTORS[a]:<14}{SECTORS[b]:<16}{vc.iloc[0]:>7.0f}{base.get(b,0):>9.0f}{len(nxts):>5}")
    print("\nVERDICT: cross lead-lag (|corr|<0.1) = no clean 'A then B'. The real, tradeable")
    print("rotation is MOMENTUM (top trailing sectors persist) — check te beats eqw/SPY above.")
    print("Leader-transition 'next' freq near its base% = persistence/noise, not a true A->B link.")


if __name__ == "__main__":
    main()
