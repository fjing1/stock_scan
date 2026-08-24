"""
btc_system.py — long-horizon BTC trend engine: buy-and-hold core + an optional low-frequency
(checked daily, flips ~monthly) trend overlay, for a "long-term hold, trade at most ~weekly"
investor.

Findings this is built on (full derivation in _btc_trend_research.py):
  - Unlike this repo's gold/stock trend-following studies, an SMA+band trend overlay on BTC has
    a REAL, bootstrap-confirmed edge over buy-and-hold gross of costs (P[worse than buy-hold] <
    2% for both a single-length pick and the ensemble). BTC's few, huge, multi-year cycles with
    -80%+ drawdowns give trend-following a mechanism that a diversified stock basket and gold's
    two-cycle history don't have.
  - Picking ONE (length, band) is fragile: PBO (probability of backtest overfitting) across a
    75-config grid is 53.8% — a coin flip. An ENSEMBLE VOTE across a band of lengths sidesteps
    that (same trick as gold's ENSEMBLE_DAYS) and tests BETTER, not just safer: vote>=70% beats
    every single-length pick (Sharpe 1.71 vs 1.59 pre-cost) with less than a third of the flips
    (83 trades / 11.9yr vs 265) and a much shallower max drawdown (-49% vs -83% for buy-hold).
  - The dominant real-world cost is NOT the exchange fee (0.2-0.5%/side barely dents it) — it's
    SHORT-TERM CAPITAL GAINS TAX. Every ensemble flip realizes a gain held <1yr, taxed at ordinary
    income rates instead of the 0/15/20% LTCG rate a never-sold position gets. At an illustrative
    32% blended short-term rate the ensemble's Sharpe drops from 1.71 to ~1.06 — still ahead of
    buy-hold (0.79 gross, 0.75 after an eventual one-time LTCG sale), but the gap narrows a lot,
    and at top-bracket rates (~41%) it can close further. This is gold's "sizing beats timing
    because tax eats the turnover benefit" story again, but for CAPITAL GAINS tax, not tax
    on a fixed sleeve — so it disappears entirely in a tax-advantaged account (IRA), where the
    overlay's raw edge applies with no drag at all.

Bottom line for a taxable account: buy-and-hold is still the accountable DEFAULT — it is what
"long-term hold" means, and it defers tax indefinitely. The ensemble state is reported as a risk
dashboard (current regime, how deep the last drawdown got) rather than a mandate to trade, UNLESS
you know your own bracket is low enough that the after-tax numbers above still favor trading, or
you're running this inside an IRA.

Run:  ../../vcp_env/bin/python btc_system.py --refresh    # download + cache the panel
      ../../vcp_env/bin/python btc_system.py               # report from cache
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import _btc_data as D

TRADING_DAYS = 365
RISK_FREE = 0.0182

ENSEMBLE_LENGTHS = [20, 25, 30, 40, 50, 65, 80, 100, 130, 160, 200]
ENSEMBLE_BAND = 0.03
VOTE_THRESHOLD = 0.70
COST_ONE_WAY = 0.0020
ST_TAX_ILLUSTRATIVE = 0.32     # blended short-term/ordinary-income rate, illustrative only
LT_TAX_ILLUSTRATIVE = 0.18     # blended long-term capital-gains rate, illustrative only


# --------------------------------------------------------------------------- #
# signal
# --------------------------------------------------------------------------- #
def trend_state(price: pd.Series, length: int, band: float = ENSEMBLE_BAND) -> pd.Series:
    """Single-length SMA+hysteresis-band long/cash state, decided on t-1 and applied to day t."""
    ma = price.rolling(length).mean()
    ext = price / ma - 1
    state = pd.Series(index=price.index, dtype=float)
    cur = 0.0
    for i, x in enumerate(ext.values):
        if np.isnan(x):
            state.iloc[i] = np.nan
            continue
        if x > band:
            cur = 1.0
        elif x < -band:
            cur = 0.0
        state.iloc[i] = cur
    return state.shift(1).fillna(0.0).astype(bool)


def ensemble_vote(price: pd.Series, lengths: list[int] = ENSEMBLE_LENGTHS,
                   band: float = ENSEMBLE_BAND) -> pd.DataFrame:
    """Per-length long/cash votes plus the fraction currently voting long."""
    votes = pd.DataFrame({n: trend_state(price, n, band) for n in lengths})
    votes["frac"] = votes[lengths].mean(axis=1)
    return votes


def ensemble_state(price: pd.Series, threshold: float = VOTE_THRESHOLD,
                    lengths: list[int] = ENSEMBLE_LENGTHS, band: float = ENSEMBLE_BAND) -> pd.Series:
    votes = ensemble_vote(price, lengths, band)
    return votes["frac"] >= threshold


# --------------------------------------------------------------------------- #
# performance / simulation
# --------------------------------------------------------------------------- #
def perf(curve: pd.Series, rf: float = RISK_FREE) -> dict:
    ret = curve.pct_change().dropna()
    if len(ret) < 30:
        return {}
    yrs = len(ret) / TRADING_DAYS
    cagr = curve.iloc[-1] ** (1 / yrs) - 1
    vol = ret.std(ddof=1) * np.sqrt(TRADING_DAYS)
    mdd = ((curve / curve.cummax()) - 1).min()
    return {"cagr": cagr, "vol": vol, "maxdd": mdd,
            "sharpe": (cagr - rf) / vol if vol > 0 else np.nan, "years": yrs}


def simulate(price: pd.Series, state: pd.Series, cost: float = 0.0, tax_rate: float = 0.0,
             cash_annual: float = RISK_FREE) -> dict:
    """
    Long/cash simulator with a per-flip fee and a per-SALE short-term-gain tax (realized vs the
    equity value at the last BUY; no cross-trade loss netting, matching gold_system.simulate()'s
    per-transaction simplification).
    """
    idx = state.index
    px = price.reindex(idx)
    ret = px.pct_change().fillna(0.0)
    cash_daily = (1 + cash_annual) ** (1 / TRADING_DAYS) - 1

    equity, in_pos, entry_equity = 1.0, False, 1.0
    trades, tax_paid = 0, 0.0
    path = []
    for i in range(len(idx)):
        want = bool(state.iloc[i])
        equity *= (1 + ret.iloc[i]) if in_pos else (1 + cash_daily)
        if want != in_pos:
            equity *= (1 - cost)
            trades += 1
            if in_pos and not want:                        # SELL: realize gain vs entry
                gain = equity - entry_equity
                if gain > 0 and tax_rate > 0:
                    tax = gain * tax_rate
                    tax_paid += tax
                    equity -= tax
            if not in_pos and want:                         # BUY: new cost basis
                entry_equity = equity
            in_pos = want
        path.append(equity)
    curve = pd.Series(path, index=idx)
    out = perf(curve)
    if not out:
        return out
    hold_days, run = [], 0
    for s in state:
        if s:
            run += 1
        elif run:
            hold_days.append(run); run = 0
    if run:
        hold_days.append(run)
    out.update({"trades": trades, "flips_per_yr": trades / (len(idx) / TRADING_DAYS),
                "median_hold_days": float(np.median(hold_days)) if hold_days else np.nan,
                "tax_pct_final": tax_paid / curve.iloc[-1] if curve.iloc[-1] else np.nan,
                "in_mkt_pct": state.mean() * 100, "curve": curve})
    return out


def fmt(p: dict) -> str:
    if not p:
        return "n/a"
    return (f"CAGR {p['cagr']*100:7.2f}%  vol {p['vol']*100:5.1f}%  MaxDD {p['maxdd']*100:7.1f}%  "
            f"Sharpe {p['sharpe']:5.2f}")


# --------------------------------------------------------------------------- #
# current reading — what the daily scan pipeline consumes
# --------------------------------------------------------------------------- #
def current_reading(price: pd.Series) -> dict:
    """Pure function of a price Series (index=dates, ascending) -> today's ensemble reading.
    Callers own the fetch: scan_stocks.py passes its own freshly-downloaded BTC-USD series so
    the reading stays live in the daily scan; the CLI below uses the on-disk cached panel."""
    votes = ensemble_vote(price)
    frac = float(votes["frac"].iloc[-1])
    state_now = frac >= VOTE_THRESHOLD
    # was the ensemble state different yesterday? (today's flip, if any)
    prev_frac = float(votes["frac"].iloc[-2]) if len(votes) > 1 else frac
    flipped_today = (frac >= VOTE_THRESHOLD) != (prev_frac >= VOTE_THRESHOLD)
    px = float(price.iloc[-1])
    hi_252 = float(price.tail(TRADING_DAYS).max())
    dd_now = px / float(price.cummax().iloc[-1]) - 1
    return {
        "asof": price.index[-1].date(),
        "price": px,
        "vote_frac": frac,
        "state": "LONG" if state_now else "CASH/RISK-OFF",
        "flipped_today": flipped_today,
        "votes_up": [n for n in ENSEMBLE_LENGTHS if bool(votes[n].iloc[-1])],
        "drawdown_from_ath": dd_now,
        "pct_from_52w_high": px / hi_252 - 1,
    }


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download the panel")
    args = ap.parse_args()

    panel = D.refresh_cache() if args.refresh else D.load_panel()
    price = D.production_series(panel)
    r = price.pct_change().dropna()
    print(f"BTC-USD  {price.index[0].date()} -> {price.index[-1].date()}  "
          f"({len(price)} bars, {len(price)/TRADING_DAYS:.1f} yrs)\n")

    print("=== 1. BUY-AND-HOLD (the default: never sold, tax deferred indefinitely) ===")
    bh = perf((1 + r).cumprod())
    print(f"  {fmt(bh)}")

    print(f"\n=== 2. ENSEMBLE TREND OVERLAY (vote>={VOTE_THRESHOLD*100:.0f}% of "
          f"{len(ENSEMBLE_LENGTHS)} SMA lengths, {ENSEMBLE_BAND*100:.0f}% band) ===")
    state = ensemble_state(price)
    scenarios = [
        ("gross (no fee/tax)", 0.0, 0.0),
        (f"fee {COST_ONE_WAY*100:.2f}%/side only", COST_ONE_WAY, 0.0),
        (f"+ {ST_TAX_ILLUSTRATIVE*100:.0f}% short-term tax on every sale", COST_ONE_WAY, ST_TAX_ILLUSTRATIVE),
    ]
    for lbl, cost, rate in scenarios:
        res = simulate(price, state, cost=cost, tax_rate=rate)
        print(f"  {lbl:<42} {fmt(res)}  flips/yr {res['flips_per_yr']:.1f}  trades {res['trades']}")
    print("  (in an IRA / tax-advantaged account: no capital-gains tax on flips at all -> the")
    print("   fee-only row applies regardless of bracket.)")

    print("\n=== 3. CURRENT READING ===")
    cr = current_reading(price)
    print(f"  BTC {cr['price']:,.2f}  as of {cr['asof']}")
    print(f"  ensemble vote: {cr['vote_frac']*100:.0f}% of lengths say LONG  -> state = {cr['state']}"
          f"{'  (FLIPPED TODAY)' if cr['flipped_today'] else ''}")
    print(f"  votes long: {cr['votes_up']}")
    print(f"  drawdown from all-time high: {cr['drawdown_from_ath']*100:.1f}%   "
          f"vs 52w high: {cr['pct_from_52w_high']*100:+.1f}%")


if __name__ == "__main__":
    main()
