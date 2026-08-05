"""Residual (idiosyncratic) momentum — the factor-neutral upgrade to plain momentum.

WHY THIS STUDY (provenance): surfaced by the `research-ideas` agentic workflow as the
#1 mature, replicated, NEW-to-this-repo edge. This repo's `rank_test.py` already found
plain 12-1 relative strength is a WEAK ranking feature (58% vs 52%), and
`market_neutral_ensemble.py` uses RAW 12-1 momentum as one leg. Residual momentum
(Blitz, Huij & Martens 2011, "Residual Momentum"; Gutierrez & Prinsky 2007;
Chaves 2016) is the canonical fix: regress each stock's returns on common factors,
rank momentum on the RESIDUAL. It strips out factor (beta/size/value) tilts, which
is what makes plain momentum crash on sharp factor reversals — residual momentum
has ~half the volatility and far smaller drawdowns, with higher risk-adjusted return.

WHAT IT DOES
  For each name, run a trailing-window OLS of daily excess returns on 3 daily factor
  proxies (all yfinance, keyless):
      mkt = SPY                      (market)
      smb = IWM - SPY                (small minus large)
      hml = IWD - IWF                (value minus growth)
  Take the residuals e_t (the part of the move NOT explained by factor exposure).
  Residual-momentum score = sum(e over t-12mo .. t-1mo) / std(e)   (information-ratio form,
  skipping the most recent month to avoid 1-month reversal contamination).
  Rank cross-sectionally, go LONG top quintile / SHORT bottom quintile (equal-weight,
  dollar-neutral), weekly rebalance. The RAW 12-1 momentum leg is run through the
  IDENTICAL engine as the apples-to-apples baseline.

HONEST EVAL (this repo's house style)
  OOS split at 2019-01-01 (train < 2019, test 2019->present incl. the 2022 factor-
  momentum crash). Reports gross + net-of-cost annualized return, Sharpe, % positive
  periods, beta-to-SPY (should be ~0 for the long-short), turnover, and the LONG-ONLY
  top-quintile ALPHA vs SPY (the realistic no-shorting retail read). The test is:
  does residual momentum beat RAW momentum out-of-sample, especially in 2022?

  CAVEAT: current-names universe (survivorship bias) inflates the long leg / deflates
  the short leg for BOTH methods equally — so the RESIDUAL-minus-RAW *difference* and
  the beta~0 / OOS-consistency are the trustworthy reads, not absolute magnitudes.

    python residual_momentum.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yfinance as yf  # noqa: E402

BENCHMARK = "SPY"
FACTOR_ETFS = ["SPY", "IWM", "IWD", "IWF"]   # market, small, value, growth
SPLIT = pd.Timestamp("2019-01-01")
REBAL = 5                 # trading days between rebalances (weekly)
PPY = 252 / REBAL         # rebalance periods per year (~50.4)
COST_RT = 0.0010          # round-trip cost (10 bps) on turned-over fraction
N_CANDIDATES = 220        # universe names to consider
MIN_BARS = 2500           # ~10y history required
LOOKBACK = 252            # trailing regression window (~12 months)
SKIP = 21                 # skip most recent ~1 month (avoid short-term reversal)


def fetch_batch(symbols, period, interval):
    raw = yf.download(symbols, period=period, interval=interval,
                      auto_adjust=False, progress=False, group_by="ticker", threads=True)
    out = {}
    for s in symbols:
        try:
            d = raw[s].dropna(how="all").copy()
        except Exception:
            continue
        if len(d):
            if getattr(d.index, "tz", None) is not None:
                d.index = d.index.tz_localize(None)
            out[s] = d
    return out


def annualize(series):
    s = np.asarray(series, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) < 5:
        return dict(n=len(s), ann=float("nan"), sharpe=float("nan"), pos=float("nan"),
                    cum=float("nan"))
    mean, std = s.mean(), s.std(ddof=1)
    ann = (1 + mean) ** PPY - 1
    sharpe = (mean / std) * np.sqrt(PPY) if std > 0 else float("nan")
    cum = np.prod(1 + s) - 1
    return dict(n=len(s), ann=ann * 100, sharpe=sharpe, pos=(s > 0).mean() * 100, cum=cum * 100)


def max_drawdown(period_returns):
    """Max drawdown of the compounded equity curve built from per-period returns."""
    s = np.asarray(period_returns, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) < 2:
        return float("nan")
    eq = np.cumprod(1 + s)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1).min() * 100)


def build_residual_scores(ret, factors, rebal_idx):
    """At each rebalance index p, OLS-residualize the trailing window across ALL names
    in ONE lstsq (shared X), then score = cum residual (skip last month) / residual std.
    Returns {p: Series(score, index=names)}."""
    F = factors.reindex(ret.index).fillna(0.0)
    scores = {}
    for p in rebal_idx:
        w0 = p - LOOKBACK
        if w0 < 1:
            continue
        Fw = F.iloc[w0:p]
        Rw = ret.iloc[w0:p]
        good = Rw.columns[Rw.notna().all().values]
        good = [g for g in good if Rw[g].abs().sum() > 0]
        if len(good) < 20:
            continue
        X = np.column_stack([np.ones(len(Fw)), Fw["mkt"].values, Fw["smb"].values, Fw["hml"].values])
        Y = Rw[good].values                              # LOOKBACK x G
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)     # 4 x G
        resid = Y - X @ beta                             # LOOKBACK x G
        sd = resid.std(axis=0, ddof=1)
        rm = resid[:-SKIP].sum(axis=0) / (sd + 1e-12)    # IR-form residual momentum
        scores[p] = pd.Series(rm, index=good)
    return scores


def run_engine(score_fn, dates, fwd, spy_fwd, rebal_idx):
    """Generic dollar-neutral top/bottom-quintile spread backtest.
    score_fn(p) -> Series(score, index=names) for rebalance index p (or None)."""
    prev_long, prev_short = set(), set()
    rows = []
    for p in rebal_idx:
        sc = score_fn(p)
        if sc is None:
            continue
        sc = sc.dropna()
        fz = fwd.iloc[p]
        valid = sc.index[fz.reindex(sc.index).notna().values]
        sc = sc.loc[valid]
        if len(sc) < 20:
            continue
        q = max(3, len(sc) // 5)
        ranked = sc.sort_values()
        shorts = list(ranked.index[:q])
        longs = list(ranked.index[-q:])
        lr = float(fz[longs].mean())
        sr = float(fz[shorts].mean())
        spread = lr - sr
        spy_f = float(spy_fwd.iloc[p]) if not np.isnan(spy_fwd.iloc[p]) else np.nan
        turn = (len(set(longs) ^ prev_long) + len(set(shorts) ^ prev_short)) / (2 * q) if prev_long else 1.0
        prev_long, prev_short = set(longs), set(shorts)
        rows.append(dict(date=dates[p], spread=spread, long=lr, short=sr, spy=spy_f,
                         net=spread - turn * COST_RT, turn=turn))
    return pd.DataFrame(rows)


def report(name, df):
    if len(df) < 5:
        print(f"  {name}: too few periods")
        return None
    ls = annualize(df["spread"]); net = annualize(df["net"])
    lo = annualize(df["long"]); sp = annualize(df["spy"])
    loex = annualize(df["long"].values - df["spy"].values)
    x, y = df["spy"].values, df["spread"].values
    m = ~(np.isnan(x) | np.isnan(y))
    beta = np.polyfit(x[m], y[m], 1)[0] if m.sum() > 5 else float("nan")
    mdd = max_drawdown(df["spread"].values)
    print(f"\n  [{name}]  periods={ls['n']}  avg turnover={df['turn'].mean()*100:.0f}%")
    print(f"    Long-Short (alpha)  : ann {ls['ann']:+.1f}%  Sharpe {ls['sharpe']:.2f}  "
          f"%+ {ls['pos']:.0f}  beta-to-SPY {beta:+.2f}  maxDD {mdd:.0f}%")
    print(f"    Long-Short net costs: ann {net['ann']:+.1f}%  Sharpe {net['sharpe']:.2f}")
    print(f"    Long-only (topQ)    : ann {lo['ann']:+.1f}%  Sharpe {lo['sharpe']:.2f}  %+ {lo['pos']:.0f}")
    print(f"    SPY (same periods)  : ann {sp['ann']:+.1f}%  Sharpe {sp['sharpe']:.2f}")
    print(f"    Long-only ALPHA/SPY : ann {loex['ann']:+.1f}%  Sharpe {loex['sharpe']:.2f}  %+ {loex['pos']:.0f}")
    return dict(ls_sharpe=ls["sharpe"], ls_ann=ls["ann"], net_sharpe=net["sharpe"],
                loex_ann=loex["ann"], loex_sharpe=loex["sharpe"], beta=beta, mdd=mdd)


def main():
    import argparse
    global REBAL, PPY, N_CANDIDATES
    ap = argparse.ArgumentParser(description="Residual vs raw momentum cross-sectional study")
    ap.add_argument("--rebal", type=int, default=REBAL,
                    help="trading days between rebalances (5=weekly, 21=monthly; momentum is conventionally monthly)")
    ap.add_argument("--names", type=int, default=N_CANDIDATES, help="universe size to consider")
    args = ap.parse_args()
    REBAL, N_CANDIDATES = args.rebal, args.names
    PPY = 252 / REBAL

    from stock_symbols_1243 import STOCK_SYMBOLS
    cands = [s for s in dict.fromkeys(STOCK_SYMBOLS) if s not in FACTOR_ETFS][:N_CANDIDATES]
    print(f"rebalance every {REBAL}d (~{'weekly' if REBAL == 5 else 'monthly' if REBAL == 21 else str(REBAL)+'d'}), universe target {N_CANDIDATES}")
    print(f"downloading {len(cands)} candidates + {len(FACTOR_ETFS)} factor ETFs (20y daily) ...", flush=True)
    data = {}
    for i in range(0, len(cands), 110):
        data.update(fetch_batch(cands[i:i + 110], "20y", "1d"))
    fdata = fetch_batch(FACTOR_ETFS, "20y", "1d")

    keep = {s: d for s, d in data.items()
            if len(d) >= MIN_BARS and float(d["Close"].iloc[-1]) > 5}
    closes = pd.DataFrame({s: d["Close"].astype(float) for s, d in keep.items()})
    print(f"universe after history/price filter: {closes.shape[1]} names, "
          f"{closes.index[0].date()} -> {closes.index[-1].date()}")

    # factor daily returns (aligned to the universe calendar)
    fc = {s: fdata[s]["Close"].astype(float).reindex(closes.index) for s in FACTOR_ETFS if s in fdata}
    spy_close = fc["SPY"]
    mkt = spy_close.pct_change(fill_method=None)
    smb = fc["IWM"].pct_change(fill_method=None) - mkt
    hml = fc["IWD"].pct_change(fill_method=None) - fc["IWF"].pct_change(fill_method=None)
    factors = pd.DataFrame({"mkt": mkt, "smb": smb, "hml": hml})

    ret = closes.pct_change(fill_method=None)
    fwd = closes.shift(-REBAL) / closes - 1
    spy_fwd = spy_close.shift(-REBAL) / spy_close - 1

    # RAW 12-1 momentum (the baseline this repo already uses), full DataFrame
    rawmom = closes.shift(SKIP) / closes.shift(LOOKBACK) - 1

    dates = closes.index
    rebal_idx = list(range(LOOKBACK, len(dates) - REBAL, REBAL))

    print("computing residual-momentum scores (trailing OLS per rebalance) ...", flush=True)
    resmom = build_residual_scores(ret, factors, rebal_idx)

    R_res = run_engine(lambda p: resmom.get(p), dates, fwd, spy_fwd, rebal_idx)
    R_raw = run_engine(lambda p: rawmom.iloc[p], dates, fwd, spy_fwd, rebal_idx)

    print("\n================ RESIDUAL vs RAW MOMENTUM (weekly, top/bottom quintile) ================")
    print("RAW = 12-1 total-return momentum (repo baseline)")
    print("RESIDUAL = momentum of OLS residuals on [SPY, IWM-SPY, IWD-IWF], IR-scaled, skip 1mo")

    print("\n---------------- RAW 12-1 MOMENTUM (baseline) ----------------")
    report("TRAIN < 2019", R_raw[R_raw["date"] < SPLIT])
    raw_te = report("TEST 2019-present (incl. 2022)", R_raw[R_raw["date"] >= SPLIT])

    print("\n---------------- RESIDUAL MOMENTUM ----------------")
    report("TRAIN < 2019", R_res[R_res["date"] < SPLIT])
    res_te = report("TEST 2019-present (incl. 2022)", R_res[R_res["date"] >= SPLIT])

    print("\n================ VERDICT ================")
    if raw_te and res_te:
        d_ls = res_te["ls_sharpe"] - raw_te["ls_sharpe"]
        d_lo = res_te["loex_ann"] - raw_te["loex_ann"]
        wins = sum(int(c) for c in (d_ls > 0, res_te["mdd"] > raw_te["mdd"], d_lo > 0))
        print(f"  OOS Long-Short Sharpe:  raw {raw_te['ls_sharpe']:.2f}  ->  residual {res_te['ls_sharpe']:.2f}  ({d_ls:+.2f})")
        print(f"  OOS Long-only alpha/yr: raw {raw_te['loex_ann']:+.1f}%  ->  residual {res_te['loex_ann']:+.1f}%  ({d_lo:+.1f}pp)")
        print(f"  OOS Long-Short maxDD:   raw {raw_te['mdd']:.0f}%  ->  residual {res_te['mdd']:.0f}%  (less negative = better)")
        print(f"  Residual beats raw on {wins}/3 OOS metrics (Sharpe / drawdown / long-only alpha).")
    print("\nReads: residual momentum should show HIGHER OOS Sharpe and a SHALLOWER drawdown")
    print("than raw momentum (it strips the factor tilt that drives momentum crashes). If the")
    print("long-only ALPHA/SPY is also positive OOS, it's a deployable cross-sectional RANK")
    print("overlay for the scanner. CAVEAT: survivorship bias inflates both legs equally.")


if __name__ == "__main__":
    main()
