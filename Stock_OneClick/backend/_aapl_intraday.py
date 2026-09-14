"""
_aapl_intraday.py — AAPL at hourly and 15-minute resolution, regular trading hours only.

READ THIS BEFORE TRUSTING ANY NUMBER BELOW.

yfinance intraday history is hard-capped, and the cap is the whole story:
    daily  5,933 bars = 24 INDEPENDENT years
    1h     5,073 bars = 2.9 independent years
    15m    1,559 bars = 0.24 independent years (88 calendar days)

A 1yr+ holding rule cannot be validated on that. The daily 200d banded rule flips ~1.6x/year, so
2.9 years of hourly data contains ~5 regime changes and 15m data contains zero. More BARS is not
more INFORMATION — the independent-observation count is set by the calendar span, not the
sampling rate. Anything this file reports about intraday TREND is therefore descriptive only.

What intraday data CAN answer honestly is a within-day question, because there each day is a fresh
observation: 88 days of 15m data gives 88 observations of "what happens at 10:15 vs 15:45". So the
centrepiece here is EXECUTION TIMING for the monthly tranche order, which is a real decision in
the system and is answerable with this sample.

Sections:
  1. RTH hygiene — confirm the bars really are 09:30-16:00 ET and count them per day.
  2. Intraday volatility / range shape (the U-curve) — when NOT to place a market order.
  3. EXECUTION TIMING: which 15m slot fills best vs the day's close and VWAP. The real deliverable.
  4. Hourly trend rules — reported with the power caveat, plus an explicit equivalence table so
     the sampling-rate illusion is visible.
  5. Does an hourly trend state add anything to the daily 200d state over the overlapping window.

Run: ../../vcp_env/bin/python _aapl_intraday.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TZ = "America/New_York"
RTH_OPEN, RTH_CLOSE = pd.Timestamp("09:30").time(), pd.Timestamp("16:00").time()
TD = 252


def fetch(interval: str, period: str) -> pd.DataFrame:
    import yfinance as yf
    h = yf.Ticker("AAPL").history(period=period, interval=interval, prepost=False)
    if h.empty:
        return h
    h.index = h.index.tz_convert(TZ)
    # RTH filter applies to INTRADAY bars only. Daily bars are stamped 00:00 ET, so applying a
    # 09:30-16:00 time window to them silently deletes the entire series.
    if interval.endswith(("m", "h")):
        h = h[(h.index.time >= RTH_OPEN) & (h.index.time < RTH_CLOSE)]
    return h[["Open", "High", "Low", "Close", "Volume"]]


def main():
    h1 = fetch("1h", "730d")
    m15 = fetch("15m", "60d")
    dly = fetch("1d", "max")

    print("=== 1. RTH HYGIENE ===")
    for lbl, df in (("1h", h1), ("15m", m15)):
        per_day = df.groupby(df.index.date).size()
        span = (df.index[-1] - df.index[0]).days
        print(f"  {lbl:>4}: {len(df):>6} RTH bars  {df.index[0].date()} -> {df.index[-1].date()}"
              f"  span {span}d = {span/365:.2f} independent yrs")
        print(f"        bars/day: median {int(per_day.median())}  "
              f"first slot {df.index.time.min()}  last slot {df.index.time.max()}")
        odd = per_day[per_day != per_day.median()]
        print(f"        days with a non-standard bar count (half-days etc): {len(odd)}")

    # ---------------- 2. intraday shape ---------------------------------- #
    print("\n=== 2. INTRADAY RANGE / VOLUME SHAPE (15m, RTH, 88 days) ===")
    m = m15.copy()
    m["slot"] = m.index.strftime("%H:%M")
    m["rng"] = (m["High"] - m["Low"]) / m["Close"] * 100.0
    m["ret"] = m["Close"].pct_change() * 100.0
    shape = m.groupby("slot").agg(range_pct=("rng", "mean"),
                                  abs_ret=("ret", lambda x: x.abs().mean()),
                                  volume=("Volume", "mean")).reset_index()
    shape["vol_share"] = shape.volume / shape.volume.sum() * 100
    print(f"  {'slot':>6}{'avg range%':>12}{'avg |ret|%':>12}{'vol share%':>12}")
    for _, x in shape.iterrows():
        bar = "#" * int(round(x.range_pct / shape.range_pct.max() * 22))
        print(f"  {x.slot:>6}{x.range_pct:>11.3f}%{x.abs_ret:>11.3f}%{x.vol_share:>11.1f}%  {bar}")
    worst = shape.loc[shape.range_pct.idxmax()]
    calm = shape.loc[shape.range_pct.idxmin()]
    print(f"\n  widest bars {worst.slot} ({worst.range_pct:.3f}%), "
          f"calmest {calm.slot} ({calm.range_pct:.3f}%) "
          f"-> {worst.range_pct/calm.range_pct:.1f}x difference in slippage risk")

    # ---------------- 3. EXECUTION TIMING (the real deliverable) --------- #
    print("\n=== 3. EXECUTION TIMING: which slot fills best? (15m, 88 days) ===")
    print("  For each day, compare each slot's close to that day's OFFICIAL CLOSE and to the")
    print("  day's VWAP. Negative 'vs close' = you bought cheaper than closing. N = one per day.")
    m["day"] = m.index.date
    day_close = m.groupby("day")["Close"].last()
    vwap = (m["Close"] * m["Volume"]).groupby(m["day"]).sum() / m.groupby("day")["Volume"].sum()
    rows = []
    for slot, grp in m.groupby("slot"):
        s = grp.set_index("day")["Close"]
        common = s.index.intersection(day_close.index)
        vs_close = (s.loc[common] / day_close.loc[common] - 1) * 100
        vs_vwap = (s.loc[common] / vwap.loc[common] - 1) * 100
        rows.append({"slot": slot, "n": len(common),
                     "vs_close_mean": vs_close.mean(), "vs_close_med": vs_close.median(),
                     "vs_close_se": vs_close.std(ddof=1) / np.sqrt(len(common)),
                     "win_vs_close": (vs_close < 0).mean() * 100,
                     "vs_vwap_mean": vs_vwap.mean()})
    ex = pd.DataFrame(rows)
    ex["t"] = ex.vs_close_mean / ex.vs_close_se
    print(f"  {'slot':>6}{'n':>5}{'vs close':>10}{'SE':>7}{'t':>7}{'cheaper%':>10}{'vs VWAP':>10}")
    for _, x in ex.iterrows():
        flag = "  *" if abs(x.t) > 2 else ""
        print(f"  {x.slot:>6}{int(x.n):>5}{x.vs_close_mean:>+9.3f}%{x.vs_close_se:>6.3f}"
              f"{x.t:>+7.2f}{x.win_vs_close:>9.0f}%{x.vs_vwap_mean:>+9.3f}%{flag}")
    best = ex.loc[ex.vs_close_mean.idxmin()]
    print(f"\n  cheapest slot on average: {best.slot} at {best.vs_close_mean:+.3f}% vs close "
          f"(t={best.t:+.2f}, {best.win_vs_close:.0f}% of days cheaper)")
    sig = ex[abs(ex.t) > 2]
    print(f"  slots significant at |t|>2: {len(sig)} of {len(ex)}"
          + (f" -> {', '.join(sig.slot)}" if len(sig) else " -> NONE; time of day is noise here"))
    print(f"  spread between best and worst slot: "
          f"{ex.vs_close_mean.max()-ex.vs_close_mean.min():.3f}% "
          f"(vs a typical 15m range of {shape.range_pct.mean():.3f}%)")

    # ---------------- 4. the sampling-rate illusion ---------------------- #
    print("\n=== 4. HOURLY TREND RULES — and why the bar count misleads ===")
    print("  Equivalence: the daily 200d MA spans 200 sessions = 1,400 hourly bars = 5,600 15m bars.")
    print(f"  Hourly history available: {len(h1)} bars = {len(h1)/7:.0f} sessions.")
    print(f"  So a 200-session MA uses {1400/len(h1)*100:.0f}% of ALL the hourly data you can get,")
    print(f"  leaving {(len(h1)-1400)/1400:.1f} non-overlapping windows to test it on. Not testable.")
    hr = h1["Close"]
    rr = hr.pct_change().dropna()
    rf_h = (1 + 0.0182) ** (1 / (TD * 7)) - 1
    print(f"\n  Descriptive only ({(h1.index[-1]-h1.index[0]).days/365:.1f} yrs, ~5 regime changes):")
    print(f"  {'hourly MA':>10}{'= sessions':>12}{'CAGR':>9}{'MaxDD':>9}{'Sharpe':>8}{'flips':>7}")
    bh = (1 + rr).cumprod()
    yrs = (h1.index[-1] - h1.index[0]).days / 365.25
    bc = bh.iloc[-1] ** (1 / yrs) - 1
    bv = rr.std(ddof=1) * np.sqrt(TD * 7)
    print(f"  {'buy-hold':>10}{'':>12}{bc*100:>8.2f}%{((bh/bh.cummax())-1).min()*100:>8.1f}%"
          f"{(bc-0.0182)/bv:>8.3f}{0:>7}")
    for n in (35, 70, 140, 350, 700, 1400):
        ma = hr.rolling(n).mean()
        st = pd.Series(np.nan, index=hr.index)
        st[hr > ma * 1.02] = 1.0
        st[hr < ma * 0.98] = 0.0
        st = st.ffill().fillna(1.0).astype(bool).shift(1).reindex(rr.index).fillna(True)
        c = (1 + rr.where(st, rf_h)).cumprod()
        cg = c.iloc[-1] ** (1 / yrs) - 1
        vv = c.pct_change().dropna().std(ddof=1) * np.sqrt(TD * 7)
        print(f"  {n:>10}{n/7:>12.0f}{cg*100:>8.2f}%{((c/c.cummax())-1).min()*100:>8.1f}%"
              f"{(cg-0.0182)/vv:>8.3f}{int((st.astype(int).diff().abs()==1).sum()):>7}")

    # ---------------- 5. does hourly add to the daily state? ------------- #
    print("\n=== 5. DOES AN HOURLY STATE ADD TO THE DAILY 200d? (overlapping window only) ===")
    d = dly["Close"].dropna()
    dma = d.rolling(200).mean()
    dst = pd.Series(np.nan, index=d.index)
    dst[d > dma * 1.02] = 1.0
    dst[d < dma * 0.98] = 0.0
    dst = dst.ffill().fillna(1.0).astype(bool)
    # carry the daily state onto the hourly grid; both sides must be tz-naive dates to align
    dst_naive = dst.copy()
    dst_naive.index = pd.DatetimeIndex(dst.index).tz_localize(None).normalize()
    h_days = pd.DatetimeIndex(h1.index.tz_localize(None)).normalize()
    dst_h = pd.Series(dst_naive.reindex(h_days, method="ffill").to_numpy(), index=h1.index)
    hma = hr.rolling(350).mean()          # ~50 sessions
    hst = pd.Series(np.nan, index=hr.index)
    hst[hr > hma * 1.02] = 1.0
    hst[hr < hma * 0.98] = 0.0
    hst = hst.ffill().fillna(1.0).astype(bool)
    for lbl, s in (("daily 200d only", dst_h),
                   ("hourly 350 only", hst),
                   ("daily AND hourly", dst_h & hst)):
        s2 = s.shift(1).reindex(rr.index).fillna(True)
        c = (1 + rr.where(s2, rf_h)).cumprod()
        cg = c.iloc[-1] ** (1 / yrs) - 1
        vv = c.pct_change().dropna().std(ddof=1) * np.sqrt(TD * 7)
        print(f"  {lbl:<20} CAGR {cg*100:>7.2f}%  MaxDD {((c/c.cummax())-1).min()*100:>7.1f}%  "
              f"Sharpe {(cg-0.0182)/vv:>6.3f}  inMkt {s2.mean()*100:>3.0f}%  "
              f"flips {int((s2.astype(int).diff().abs()==1).sum()):>4}")
    print(f"  agreement between the two states: {(dst_h == hst).mean()*100:.1f}% of hourly bars")
    print("  NOTE: 2.9 years and ~5 daily regime changes. Any ranking here is a coin flip.")

    print("\n=== CURRENT INTRADAY READING ===")
    print(f"  last 15m bar  {m15.index[-1]}  close {float(m15['Close'].iloc[-1]):,.2f}")
    print(f"  last 1h bar   {h1.index[-1]}  close {float(h1['Close'].iloc[-1]):,.2f}")
    print(f"  last daily    {dly.index[-1].date()}  close {float(dly['Close'].iloc[-1]):,.2f}")
    print(f"  hourly 350MA (~50 sessions) {float(hma.iloc[-1]):,.2f}  "
          f"state {'UPTREND' if bool(hst.iloc[-1]) else 'DOWNTREND'}")
    print(f"  daily 200MA {float(dma.iloc[-1]):,.2f}  "
          f"state {'UPTREND' if bool(dst.iloc[-1]) else 'DOWNTREND'}")


if __name__ == "__main__":
    main()
