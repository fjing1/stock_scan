"""
gold_system.py — long-horizon (1yr+) gold allocation research engine.

Purpose: answer four operational questions for a gold sleeve held 1yr+:
  IDENTIFY (trend state) / ALLOCATE (target weight) / BUY (staging) / REDUCE (trim rule)

Design notes:
- Monthly decision frequency. NO LOOKAHEAD: the signal is computed from month-t data and the
  resulting weight is applied to month t+1's return (``.shift(1)`` on every signal).
- Weights DRIFT between rebalances; rebalancing is an explicit, costed event. An earlier
  naive version computed ``book*(1-w) + gold*w`` on monthly returns, which silently rebalances
  every single month and flatters the result — don't do that.
- Taxes matter more than usual here: in a US taxable account, physical-gold ETFs (GLD/IAU/SGOL)
  are taxed as COLLECTIBLES at up to 28% on long-term gains, not the 20% LTCG rate. Turnover is
  therefore penalised harder than in an equity sleeve. Cost basis is tracked properly rather
  than assumed.
- The binding constraint is sample size: ~26 years of gold data is ~26 INDEPENDENT annual
  observations, spanning only two cycles (2000-2011 bull, 2011-2015 bear, 2015-2026 bull).
  Every reported number carries that caveat; see ``effective_n``.

Run:  ../../vcp_env/bin/python gold_system.py --refresh    # download + cache the panel
      ../../vcp_env/bin/python gold_system.py              # report from cache
"""
from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CACHE = Path(__file__).resolve().parent / "_gold_panel.pkl"

# Gold price history: GC=F (futures, 2000+) is the longest clean series. IAU/GLD are the
# tradeable instruments; IAU is preferred on cost (0.25% vs 0.40% expense ratio).
GOLD_SERIES = "GC=F"
EQUITY_PROXY = {"QQQ": 0.5, "SMH": 0.3, "SPY": 0.2}   # stand-in for a tech/AI-heavy book
TICKERS = ["GC=F", "GLD", "IAU", "GDX", "SPY", "QQQ", "SMH", "IEF", "TLT", "DX-Y.NYB", "^TNX",
           # volatility / stress gauges. Shorter histories than gold, so anything built on them
           # has less power: VIX 1990, MOVE 2002, VIX3M 2006, VVIX 2007, GVZ 2008, SKEW 1990.
           "^VIX", "^VVIX", "^VIX3M", "^GVZ", "^MOVE", "^SKEW"]

COLLECTIBLES_TAX = 0.28   # US max long-term rate on physical-gold ETFs
EQUITY_LTCG_TAX = 0.20
IAU_EXPENSE = 0.0025      # annual, drag applied monthly


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def refresh_cache() -> dict:
    """
    Download TOTAL-RETURN series (auto_adjust=True -> dividends and splits reinvested).

    Using the raw unadjusted Close understates every dividend-paying equity while leaving gold
    (which pays nothing) untouched, so it biases every gold-vs-equity comparison in gold's
    favour by ~1%/yr. Measured on this panel: SPY 6.60% price-only vs 8.52% total return,
    QQQ 8.25 vs 8.98, SMH 11.00 vs 11.77.
    """
    import yfinance as yf

    out = {}
    for t in TICKERS:
        h = yf.Ticker(t).history(period="max", interval="1d", auto_adjust=True)
        if h.empty:
            print(f"  WARN no data for {t}")
            continue
        if getattr(h.index, "tz", None) is not None:
            h.index = h.index.tz_localize(None)
        out[t] = h[["Open", "High", "Low", "Close", "Volume"]]
        print(f"  {t:<10} {h.index[0].date()} -> {h.index[-1].date()}  {len(h)} bars")
    with open(CACHE, "wb") as f:
        pickle.dump(out, f)
    print(f"cached -> {CACHE}")
    return out


def load_panel() -> dict:
    if not CACHE.exists():
        return refresh_cache()
    with open(CACHE, "rb") as f:
        return pickle.load(f)


def monthly(panel: dict, drop_partial: bool = True) -> pd.DataFrame:
    """
    Month-end closes on a common index.

    drop_partial: discard the final month when it is still in progress. resample("ME").last()
    happily builds an "August close" out of three days of August, which then poisons every
    moving average and momentum reading downstream. The same partial-bar mistake has bitten
    this repo's daily scanner too (see resolve_session_state in scan_stocks.py).
    """
    cols = {t: df["Close"].resample("ME").last() for t, df in panel.items()}
    m = pd.DataFrame(cols)
    if drop_partial and len(m):
        last_bar = max(df.index[-1] for df in panel.values())
        month_end = m.index[-1]
        if last_bar < month_end:          # month has not closed yet
            m = m.iloc[:-1]
    return m


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
RISK_FREE = 0.0182   # avg 13-week T-bill over the 2000-2026 window


def perf(equity: pd.Series, rf: float = RISK_FREE) -> dict:
    """CAGR / vol / MaxDD / Sharpe from a monthly equity curve starting at 1.0.

    Sharpe subtracts the risk-free rate. Without it, cagr/vol overstates every figure by
    roughly rf/vol (~0.08-0.12 here) and silently assumes a 0% hurdle.
    """
    eq = equity.dropna()
    if len(eq) < 24:
        return {}
    ret = eq.pct_change().dropna()
    yrs = len(ret) / 12.0
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = ret.std(ddof=1) * np.sqrt(12)
    mdd = ((eq / eq.cummax()) - 1).min()
    return {
        "cagr": cagr, "vol": vol, "maxdd": mdd,
        "sharpe": ((cagr - rf) / vol) if vol > 0 else np.nan,
        "years": yrs, "effective_n": int(yrs),   # independent annual observations
    }


def fmt(p: dict) -> str:
    if not p:
        return "n/a"
    return (f"CAGR {p['cagr']*100:6.2f}%  vol {p['vol']*100:5.1f}%  "
            f"MaxDD {p['maxdd']*100:7.1f}%  Sharpe {p['sharpe']:5.2f}")


# --------------------------------------------------------------------------- #
# signals  (all shifted: decided on month t, applied to month t+1)
# --------------------------------------------------------------------------- #
def trend_signals(gold: pd.Series) -> pd.DataFrame:
    """Long-horizon trend readings on the monthly gold series.

    RETAINED FOR REFERENCE ONLY — superseded by daily_trend_state(). The monthly reading has two
    defects, both measured in _gold_ma_research.py / _gold_ma_daily.py:
      1. The length was never justified. 18 months ranks 18th of 34 lengths by Sharpe, and a
         block bootstrap puts a 35.6% probability on the "best" length actually being worse than
         the median — i.e. the choice is noise.
      2. Month-end sampling is itself an unexamined parameter. Moving the sample day through the
         month swings the 18-month overlay's Sharpe from 0.481 to 0.630 (spread 0.149), which is
         the same order as the entire length-selection effect. Daily sampling removes it.
    """
    s = pd.DataFrame(index=gold.index)
    for n in (10, 12, 18, 24):
        s[f"ma{n}"] = gold.rolling(n).mean()
        s[f"above_ma{n}"] = (gold > s[f"ma{n}"]).shift(1)
    for n in (6, 12):
        s[f"mom{n}"] = (gold.pct_change(n) > 0).shift(1)
    s["ext18"] = (gold / gold.rolling(18).mean() - 1)          # extension vs long MA
    s["ext18_pct"] = s["ext18"].expanding(60).apply(lambda x: (x[:-1] < x[-1]).mean())
    return s


# The long-trend length, on DAILY bars. ~500 days is ~2 years. Evidence for it, all in
# _gold_ma_daily.py: it is an interior Sharpe peak (0.565, decaying to 0.297 by 2000d, so not a
# boundary artifact); annual walk-forward selection picked ~490d in 16 of 16 re-selections with
# zero drift; and it reduced the drawdown in 6 of 6 historical episodes, unlike the monthly 18m
# reading which helped in only 1 of 4. The response is flat from ~460-560d, so do not tune finer
# than "about 500" — 440d drops to 0.446, so the short edge of the plateau is real.
GOLD_TREND_DAYS = 500
GOLD_TREND_PLATEAU = (460, 560)
# Hysteresis band around the trend line. Measured in _gold_sma_vs_ema.py: a +/-2% band cuts
# crossovers from 65 to 12 over 26 years AND improves Sharpe (0.565 -> 0.586), because most raw
# crossings are 1-3 day flickers with a median state duration of 6 days. Better and cheaper.
GOLD_TREND_BAND = 0.02
# SMA, not EMA. At matched length the two families are statistically indistinguishable (EMA minus
# SMA: mean +0.002 across 31 lengths, EMA wins 14 of 31; bootstrap CI [-0.121, +0.047]). SMA is
# kept for two practical reasons: annual walk-forward selection picked SMA in 16 of 16
# re-selections, and under the +/-2% band the SMA flips 12 times vs the EMA's 20 (Sharpe 0.586 vs
# 0.499) — the EMA hugs recent price more closely, so a fixed band catches it more often.
GOLD_TREND_KIND = "SMA"
# Vote across this range instead of trusting one length; the ensemble needs no length choice.
ENSEMBLE_DAYS = list(range(40, 521, 10))


def daily_trend_state(gold_daily: pd.Series, band: float = GOLD_TREND_BAND) -> dict:
    """
    Read gold's long-term trend from DAILY closes: a 500-day SMA with a +/-2% hysteresis band,
    plus an ensemble vote across 40-520d (a single length is false precision — as of 2026-08,
    lengths >= 330d say uptrend and shorter ones say downtrend).

    The band matters. Without it there were 65 crossovers in 26 years with a MEDIAN state
    duration of 6 days; with it, 12 crossovers (~1 per 26 months). ``state`` therefore only
    changes when price clears the line by 2%, and ``in_band`` flags the indeterminate zone.

    This reading is DIAGNOSTIC, not a trade trigger. Tilting the sleeve weight on it loses to a
    flat weight at every tilt size (flat 20% Sharpe 0.476 vs 0.437 at 25/15, 0.387 at 30/10),
    because the 28% collectibles tax exceeds the signal's value. It answers "which regime am I
    in", and it answers that late: it read +43.5% ABOVE at the 2011 top and -13.5% BELOW at the
    2015 bottom. It never calls turns.
    """
    g = gold_daily.dropna()
    px = float(g.iloc[-1])
    ma = float(g.rolling(GOLD_TREND_DAYS).mean().iloc[-1])
    votes = {n: bool(px > float(g.rolling(n).mean().iloc[-1])) for n in ENSEMBLE_DAYS}
    up = [n for n in ENSEMBLE_DAYS if votes[n]]
    if px > ma * (1 + band):
        state = "UPTREND"
    elif px < ma * (1 - band):
        state = "DOWNTREND"
    else:
        state = "IN BAND (unchanged from prior state)"
    return {
        "asof": g.index[-1].date(),
        "price": px,
        "ma": ma,
        "state": state,
        "in_band": abs(px / ma - 1) <= band,
        "above_ma": px > ma,
        "extension": px / ma - 1,
        "upper": ma * (1 + band),
        "lower": ma * (1 - band),
        "vote_frac": len(up) / len(ENSEMBLE_DAYS),
        "crossover_days": min(up) if up else None,
        "votes_up": up,
    }


# --------------------------------------------------------------------------- #
# portfolio simulation with drift, rebalancing, costs and taxes
# --------------------------------------------------------------------------- #
def simulate(gold_ret: pd.Series,
             eq_ret: pd.Series,
             target: float | pd.Series,
             mode: str = "band",
             band: float = 0.25,
             tax: float = COLLECTIBLES_TAX,
             eq_tax: float = EQUITY_LTCG_TAX,
             expense: float = IAU_EXPENSE) -> dict:
    """
    Simulate a two-asset (gold sleeve + equity book) portfolio.

    target : constant weight, or a Series of month-by-month target weights (already shifted).
    mode   : 'none'   never rebalance (weights drift freely)
             'annual' rebalance every 12th month
             'band'   rebalance when |w - target| / target > band
    tax    : rate on the REALISED gain when gold is SOLD (28% collectibles in a US taxable acct).
    eq_tax : rate on the realised gain when EQUITY is sold to FUND a gold purchase. Charging only
             the gold side makes buying gold look free, and in a sample where gold outperformed
             that penalises exactly the direction that occurred — which is how a turnover finding
             gets mistaken for an allocation finding.

    Cost basis is tracked in DOLLARS for both sleeves. Tracking it as a fraction of the current
    portfolio value lets the implied basis compound along with the portfolio, understating
    realised gains (and therefore tax) by ~50% over 26 years.
    """
    idx = gold_ret.index
    tgt = pd.Series(target, index=idx) if np.isscalar(target) else target.reindex(idx)
    tgt = tgt.ffill().fillna(0.0).clip(0.0, 1.0)

    total = 1.0
    vg = float(tgt.iloc[0]) * total          # dollar value of the gold sleeve
    vb = total - vg                          # dollar value of the equity book
    basis_g = vg                             # dollar cost basis, gold
    basis_b = vb                             # dollar cost basis, equity
    trades = 0
    tax_paid = 0.0
    path = []

    for i, dt in enumerate(idx):
        rg = gold_ret.iloc[i]
        rb = eq_ret.iloc[i]
        if not (np.isfinite(rg) and np.isfinite(rb)):
            path.append(total)
            continue

        # grow both sleeves; the gold ETF pays its expense ratio monthly
        vg *= (1 + rg) * (1 - expense / 12.0)
        vb *= (1 + rb)
        total = vg + vb
        w = vg / total if total > 0 else 0.0

        t = float(tgt.iloc[i])
        do = False
        if mode == "annual":
            do = (i + 1) % 12 == 0
        elif mode == "band":
            do = t > 0 and abs(w - t) / t > band
        elif mode == "target_change":
            do = i > 0 and abs(t - float(tgt.iloc[i - 1])) > 1e-9

        if do and abs(w - t) > 1e-6:
            want_g = t * total
            if vg > want_g:                             # sell gold -> collectibles rate
                sell = vg - want_g
                frac = sell / vg
                gain = max(0.0, sell - basis_g * frac)
                cost = gain * tax
                basis_g -= basis_g * frac
                vg -= sell
                vb += sell - cost
                basis_b += sell - cost
            else:                                       # sell equity to buy gold -> LTCG rate
                buy = want_g - vg
                frac = buy / vb if vb > 0 else 0.0
                gain = max(0.0, buy - basis_b * frac)
                cost = gain * eq_tax
                basis_b -= basis_b * frac
                vb -= buy
                vg += buy - cost
                basis_g += buy - cost
            tax_paid += cost
            total = vg + vb
            trades += 1
        path.append(total)

    curve = pd.Series(path, index=idx)
    out = perf(curve)
    # report tax as a share of FINAL value; a running sum of nominal dollars measured against a
    # starting value of 1.0 is not comparable to anything else in the table.
    out.update({"trades": trades,
                "tax_pct_final": tax_paid / curve.iloc[-1] if curve.iloc[-1] else np.nan,
                "curve": curve,
                "final_weight": vg / total if total else 0.0})
    return out


def subperiods(curve: pd.Series) -> dict:
    """Per-regime performance — the honest way to see whether one cycle carries the result."""
    spans = {
        "2000-2011 bull": ("2000-01-01", "2011-08-31"),
        "2011-2015 bear": ("2011-09-01", "2015-12-31"),
        "2016-2026 bull": ("2016-01-01", "2026-12-31"),
    }
    out = {}
    for k, (a, b) in spans.items():
        seg = curve.loc[a:b]
        if len(seg) < 13:
            continue
        seg = seg / seg.iloc[0]
        out[k] = perf(seg)
    return out


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download the panel")
    args = ap.parse_args()

    panel = refresh_cache() if args.refresh else load_panel()
    m = monthly(panel)
    gold = m[GOLD_SERIES].dropna()
    gr = gold.pct_change()

    eq_px = m[list(EQUITY_PROXY)].dropna()
    eq_ret = sum(eq_px[t].pct_change() * w for t, w in EQUITY_PROXY.items())

    common = gr.dropna().index.intersection(eq_ret.dropna().index)
    gr, eq_ret = gr.loc[common], eq_ret.loc[common]
    print(f"window {common[0].date()} -> {common[-1].date()}  "
          f"({len(common)/12:.1f} yrs, effective N = {int(len(common)/12)} independent years)\n")

    print("=== 1. DOES TIMING GOLD BEAT HOLDING IT? (gold sleeve alone) ===")
    bh = perf((1 + gr).cumprod())
    print(f"  {'buy-and-hold gold':<30} {fmt(bh)}")
    sig = trend_signals(gold)
    for n in (10, 12, 18, 24):
        s = sig[f"above_ma{n}"].reindex(common).fillna(False)
        ov = perf((1 + gr.where(s, 0.0)).cumprod())
        print(f"  {f'{n}m MA overlay (long/cash)':<30} {fmt(ov)}  inMkt {s.mean()*100:3.0f}%")

    print("\n=== 2. STATIC GOLD SLEEVE IN A TECH-HEAVY BOOK (drift + 28% collectibles tax) ===")
    print(f"  {'weight':>7} {'method':<13}{'CAGR':>8}{'vol':>7}{'MaxDD':>9}{'Sharpe':>8}{'trades':>7}{'taxdrag':>9}")
    for wt in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25):
        for mode, band, lbl in (("none", None, "never"), ("annual", None, "annual"), ("band", 0.25, "25% band")):
            r = simulate(gr, eq_ret, wt, mode=mode, band=band)
            if not r:
                continue
            print(f"  {wt*100:>6.0f}% {lbl:<13}{r['cagr']*100:>7.2f}%{r['vol']*100:>6.1f}%"
                  f"{r['maxdd']*100:>8.1f}%{r['sharpe']:>8.2f}{r['trades']:>7}{r['tax_pct_final']*100:>8.1f}%")
            if wt == 0.0:
                break   # no gold -> method is irrelevant

    print("\n=== 3. START-DATE SENSITIVITY (20% sleeve, 25% band) ===")
    for start in ("2000-09-30", "2004-01-31", "2008-01-31", "2012-01-31", "2016-01-31"):
        if pd.Timestamp(start) < common[0]:
            continue
        sl = common[common >= start]
        r = simulate(gr.loc[sl], eq_ret.loc[sl], 0.20, mode="band", band=0.25)
        r0 = simulate(gr.loc[sl], eq_ret.loc[sl], 0.0, mode="none")
        if r and r0:
            print(f"  from {start}:  gold20 {fmt(r)}   |  no-gold Sharpe {r0['sharpe']:.2f} "
                  f"-> delta {r['sharpe']-r0['sharpe']:+.2f}")

    print("\n=== 4. DIVERSIFICATION STABILITY ===")
    c = gr.corr(eq_ret)
    roll = gr.rolling(36).corr(eq_ret)
    print(f"  full-sample corr(gold, book) {c:+.3f}")
    print(f"  rolling 36m: min {roll.min():+.2f} ({roll.idxmin().date()})  "
          f"max {roll.max():+.2f} ({roll.idxmax().date()})  now {roll.dropna().iloc[-1]:+.2f}")
    print(f"  months where corr > 0.4 (gold not diversifying): "
          f"{int((roll > 0.4).sum())} of {int(roll.notna().sum())}")

    print("\n=== 5. CURRENT READING ===")
    px = float(gold.iloc[-1])
    print(f"  gold (GC=F)  {px:,.2f}   as of {gold.index[-1].date()}")
    for n in (10, 12, 18, 24):
        ma = float(gold.rolling(n).mean().iloc[-1])
        print(f"  {n:>2}m MA      {ma:>9,.2f}   {(px/ma-1)*100:+6.1f}%  "
              f"{'ABOVE' if px > ma else 'BELOW'}")
    for n in (6, 12):
        print(f"  {n:>2}m momentum {(px/float(gold.iloc[-1-n])-1)*100:+6.1f}%")
    ext = float(sig['ext18'].iloc[-1])
    pct = float(sig['ext18_pct'].dropna().iloc[-1])
    print(f"  extension vs 18m MA {ext*100:+.1f}%  ({pct*100:.0f}th percentile of history)")
    hi = float(gold.tail(13).max())
    print(f"  vs 12m high  {hi:,.2f}   {(px/hi-1)*100:+.1f}%")


if __name__ == "__main__":
    main()
