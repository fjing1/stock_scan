"""
_btc_trend_research.py — does a low-frequency (daily-checked, weekly-ish-hold) trend-following
overlay beat buy-and-hold BTC, once fees AND short-term-gains tax are priced in?

Mirrors the gold sleeve study's rigor (_gold_ma_daily.py / _gold_signal_overlay.py): sweep a
grid, block-bootstrap the "best" against the field, re-select walk-forward with prior data only,
and deflate the Sharpe for how many configs were tried (Bailey & Lopez de Prado) before believing
any of it. The two things that make BTC different from gold:

  1. BTC has had six brutal drawdowns (2011 -93%, 2013-15 -83%, 2018 -84%, 2020 -65%, 2022 -77%,
     plus smaller ones) inside a handful of multi-year trend regimes — trend-following has a much
     more plausible mechanism here than it did for gold's two-cycle, low-turnover history.
  2. The user wants trades at most ~1/day to ~1/week — i.e. an UPPER bound on frequency, not a
     target. Any length/band combo that flips more than ~once a week violates that constraint and
     is reported but flagged, not recommended.
  3. Every flip this fast realizes a SHORT-TERM gain (<1yr holding period) taxed at ordinary
     income rates (up to ~37% federal + 3.8% NIIT), not the 0/15/20% long-term rate a true HODL
     position gets by simply never selling. This is gold's 28% collectibles-tax story again, but
     worse, because the alternative (never sell) pays ZERO tax, not a lower rate.

Run: ../../vcp_env/bin/python _btc_trend_research.py
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd

import _btc_data as D
import btc_system as B
from btc_system import TRADING_DAYS, fmt, perf, simulate, trend_state  # noqa: F401

RNG = np.random.default_rng(2024)

# --- dependency-free stats helpers (ported from gold_pine_script/walkforward.py; no scipy) ---
def _phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _phi_inv(p):
    if p <= 0:
        return -math.inf
    if p >= 1:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def deflated_sharpe(sel_ret, all_pp_sharpes, n_trials):
    """Bailey & Lopez de Prado (2014). Returns (DSR prob, SR_hat pp, SR0 haircut pp)."""
    r = np.asarray(sel_ret, float); r = r[~np.isnan(r)]
    T = len(r)
    sd = r.std(ddof=1)
    if T < 12 or sd == 0:
        return float("nan"), float("nan"), float("nan")
    sr = r.mean() / sd
    dev = r - r.mean()
    sd0 = math.sqrt((dev**2).mean())
    skew = (dev**3).mean() / sd0**3
    kurt = (dev**4).mean() / sd0**4
    trials = np.asarray([x for x in all_pp_sharpes if not np.isnan(x)], float)
    var_sr = trials.var(ddof=1) if len(trials) > 1 else 0.0
    gamma = 0.5772156649015329
    N = max(2, int(n_trials))
    sr0 = math.sqrt(var_sr) * ((1 - gamma) * _phi_inv(1 - 1.0/N) + gamma * _phi_inv(1 - 1.0/(N*math.e)))
    denom = math.sqrt(max(1e-12, 1 - skew*sr + (kurt - 1)/4.0 * sr*sr))
    dsr = _phi((sr - sr0) * math.sqrt(T - 1) / denom)
    return dsr, sr, sr0


def pbo_cscv(M, S=12):
    """Probability of Backtest Overfitting via CSCV. M = T x N per-period returns."""
    T, N = M.shape
    if N < 2 or T < 2 * S:
        return float("nan"), 0
    blocks = np.array_split(np.arange(T), S)
    lam = []
    for comb in combinations(range(S), S // 2):
        is_idx = np.concatenate([blocks[b] for b in comb])
        oos_idx = np.concatenate([blocks[b] for b in range(S) if b not in comb])
        IS, OOS = M[is_idx], M[oos_idx]
        is_sh = IS.mean(0) / (IS.std(0, ddof=1) + 1e-12)
        oos_sh = OOS.mean(0) / (OOS.std(0, ddof=1) + 1e-12)
        n_star = int(np.argmax(is_sh))
        rank = oos_sh.argsort().argsort()[n_star] + 1
        w = rank / (N + 1.0)
        lam.append(math.log(w / (1 - w)))
    lam = np.array(lam)
    return float((lam < 0).mean()), len(lam)


# --------------------------------------------------------------------------- #
LENGTHS = [10, 15, 20, 25, 30, 40, 50, 65, 80, 100, 130, 160, 200, 250, 300]
BANDS = [0.0, 0.03, 0.05, 0.08, 0.10]
COST_ONE_WAY = 0.0020        # 0.20%/side: mid-tier retail exchange taker fee + slippage
ST_TAX = 0.32                 # illustrative blended short-term (ordinary income) rate
LT_TAX = 0.18                 # illustrative blended long-term capital-gains rate


def main():
    panel = D.load_panel()
    price = D.production_series(panel)
    r_all = price.pct_change().dropna()
    print(f"BTC-USD (yfinance) {price.index[0].date()} -> {price.index[-1].date()}  "
          f"({len(price)} bars, {len(price)/TRADING_DAYS:.1f} yrs)\n")

    bh = perf((1 + r_all).cumprod())
    print(f"buy-and-hold, GROSS (never sold, no tax):  {fmt(bh)}")

    # ---- 1. grid sweep -------------------------------------------------------- #
    print(f"\n=== 1. SMA+band sweep, gross of cost/tax "
          f"({len(LENGTHS)} lengths x {len(BANDS)} bands = {len(LENGTHS)*len(BANDS)} trials) ===")
    states, rows = {}, []
    for n in LENGTHS:
        for b in BANDS:
            s = trend_state(price, n, b)
            states[(n, b)] = s
            res = simulate(price, s)
            if not res:
                continue
            rows.append({"len": n, "band": b, **res})
    sw = pd.DataFrame(rows).drop(columns=["curve"])
    top = sw.sort_values("sharpe", ascending=False).head(10)
    print(top[["len", "band", "cagr", "maxdd", "sharpe", "flips_per_yr", "median_hold_days",
               "in_mkt_pct"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    weekly_ok = sw[sw.flips_per_yr <= 52]
    print(f"\n  configs at or under ~1 flip/week ({len(weekly_ok)}/{len(sw)}): "
          f"best Sharpe {weekly_ok.sharpe.max():.3f} "
          f"(len={int(weekly_ok.loc[weekly_ok.sharpe.idxmax(),'len'])}, "
          f"band={weekly_ok.loc[weekly_ok.sharpe.idxmax(),'band']:.2f})")
    print(f"  median Sharpe across ALL {len(sw)} trials: {sw.sharpe.median():.3f}  "
          f"vs buy-hold {bh['sharpe']:.3f}  "
          f"({int((sw.sharpe > bh['sharpe']).sum())}/{len(sw)} beat buy-hold gross)")

    best_row = weekly_ok.loc[weekly_ok.sharpe.idxmax()]
    best_len, best_band = int(best_row["len"]), float(best_row["band"])
    best_state = states[(best_len, best_band)]

    # ---- 2. cost + tax sensitivity on the best weekly-cadence candidate ------ #
    print(f"\n=== 2. COST + SHORT-TERM-TAX DRAG on the best weekly-cadence candidate "
          f"(len={best_len}, band={best_band:.2f}) ===")
    print(f"  {'scenario':<38}{'CAGR':>8}{'Sharpe':>8}{'trades':>8}{'tax/final':>10}")
    for cost in (0.0, COST_ONE_WAY, 0.005):
        r = simulate(price, best_state, cost=cost, tax_rate=0.0)
        print(f"  fee {cost*100:>4.2f}%/side, no tax{'':<15}{r['cagr']*100:>7.2f}%{r['sharpe']:>8.2f}"
              f"{r['trades']:>8}{'':>10}")
    for tax in (LT_TAX, ST_TAX, 0.408):
        r = simulate(price, best_state, cost=COST_ONE_WAY, tax_rate=tax)
        print(f"  fee {COST_ONE_WAY*100:.2f}%/side + {tax*100:.1f}% tax on every sale"
              f"{'':<7}{r['cagr']*100:>7.2f}%{r['sharpe']:>8.2f}{r['trades']:>8}"
              f"{r['tax_pct_final']*100:>9.1f}%")
    bh_curve = (1 + r_all).cumprod()
    final_gain = bh_curve.iloc[-1] - 1
    ltcg_curve = bh_curve.copy()
    ltcg_curve.iloc[-1] = bh_curve.iloc[-1] - max(0.0, final_gain) * LT_TAX
    r_ltcg_once = perf(ltcg_curve)
    print(f"  buy-hold, sold ONCE at the end, LTCG {LT_TAX*100:.0f}%{'':<12}"
          f"{r_ltcg_once['cagr']*100:>7.2f}%{r_ltcg_once['sharpe']:>8.2f}{1:>8}"
          f"{max(0.0,final_gain)*LT_TAX/bh_curve.iloc[-1]*100:>9.1f}%  (one realized gain, illustrative)")

    # ---- 2b. ensemble vote across a band of lengths (sidesteps picking ONE length) ---- #
    print(f"\n=== 2b. ENSEMBLE VOTE across lengths {LENGTHS[2:-2]}, band 0.03 (no single length to pick) ===")
    vote_lengths = LENGTHS[2:-2]
    vote_states = pd.DataFrame({n: states[(n, 0.03)] for n in vote_lengths if (n, 0.03) in states})
    frac = vote_states.mean(axis=1)
    for thr in (0.3, 0.5, 0.7):
        ens_state = (frac >= thr)
        for tax in (0.0, ST_TAX):
            r = simulate(price, ens_state, cost=COST_ONE_WAY, tax_rate=tax)
            print(f"  vote >= {thr*100:3.0f}%,  tax {tax*100:4.1f}%:  {fmt(r)}  "
                  f"flips/yr {r['flips_per_yr']:.1f}  trades {r['trades']}")

    # ---- 2c. is the ensemble's edge real, bootstrapped on its own (only 3 thresholds tried) ---- #
    print("\n=== 2c. BLOCK BOOTSTRAP: ensemble vote>=70% (pre-tax) vs buy-and-hold ===")
    yrs_list = sorted({d.year for d in price.index})
    ens70 = (frac >= 0.7)
    ens_ret = simulate(price, ens70, cost=COST_ONE_WAY)["curve"].pct_change().dropna()
    common_e = r_all.index.intersection(ens_ret.index)
    ae, be = ens_ret.loc[common_e], r_all.loc[common_e]
    diffs_e = []
    for _ in range(1500):
        pick = RNG.choice(yrs_list, size=len(yrs_list), replace=True)
        sel = np.concatenate([np.where(common_e.year == y)[0] for y in pick])
        aa, bb = ae.iloc[sel], be.iloc[sel]
        sa, sb = aa.std(ddof=1), bb.std(ddof=1)
        if sa > 0 and sb > 0:
            diffs_e.append((aa.mean()/sa - bb.mean()/sb) * np.sqrt(TRADING_DAYS))
    diffs_e = np.array(diffs_e)
    print(f"  mean Sharpe diff {diffs_e.mean():+.3f}  SE {diffs_e.std(ddof=1):.3f}  "
          f"95% CI [{np.percentile(diffs_e,2.5):+.3f}, {np.percentile(diffs_e,97.5):+.3f}]")
    print(f"  P(ensemble actually worse than buy-hold) = {(diffs_e<0).mean()*100:.1f}%")

    # ---- 3. is the SINGLE-LENGTH "best" real, or the best of 75 coin flips? -- #
    print("\n=== 3. BLOCK BOOTSTRAP (annual blocks): best-weekly-cadence vs buy-and-hold ===")
    diffs = []
    bh_daily = r_all
    best_ret = simulate(price, best_state, cost=COST_ONE_WAY)["curve"].pct_change().dropna()
    common_idx = bh_daily.index.intersection(best_ret.index)
    a_full, b_full = best_ret.loc[common_idx], bh_daily.loc[common_idx]
    for _ in range(1500):
        pick = RNG.choice(yrs_list, size=len(yrs_list), replace=True)
        sel = np.concatenate([np.where(common_idx.year == y)[0] for y in pick])
        aa, bb = a_full.iloc[sel], b_full.iloc[sel]
        sa, sb = aa.std(ddof=1), bb.std(ddof=1)
        if sa > 0 and sb > 0:
            diffs.append((aa.mean()/sa - bb.mean()/sb) * np.sqrt(TRADING_DAYS))
    diffs = np.array(diffs)
    print(f"  mean Sharpe diff {diffs.mean():+.3f}  SE {diffs.std(ddof=1):.3f}  "
          f"95% CI [{np.percentile(diffs,2.5):+.3f}, {np.percentile(diffs,97.5):+.3f}]")
    print(f"  P(strategy actually worse than buy-hold) = {(diffs<0).mean()*100:.1f}%")

    # ---- 4. walk-forward: which (length, band) would you have picked? -------- #
    print("\n=== 4. WALK-FORWARD selection (annual re-pick, prior data only, cost-adjusted) ===")
    picks = []
    for i in range(TRADING_DAYS * 3, len(price), TRADING_DAYS):
        hist_idx = price.index[:i]
        sc = {}
        for (n, b), s in states.items():
            if n * 3 > len(hist_idx):     # need enough history to have a real signal
                continue
            sub = s.loc[hist_idx]
            res = simulate(price.loc[hist_idx], sub, cost=COST_ONE_WAY)
            if res and res.get("flips_per_yr", 999) <= 52:
                sc[(n, b)] = res["sharpe"]
        if sc:
            best = max(sc, key=sc.get)
            picks.append({"date": price.index[i].date(), "len": best[0], "band": best[1]})
    pk = pd.DataFrame(picks)
    print(f"  re-selections: {len(pk)}")
    print(pk.to_string(index=False))
    print(f"  distinct lengths picked: {pk.len.nunique()}  range {pk.len.min()}-{pk.len.max()}")

    # ---- 5. deflated Sharpe / PBO across the full grid ------------------------ #
    print(f"\n=== 5. DEFLATED SHARPE & PBO across all {len(sw)} trials (multiple-testing haircut) ===")
    daily_rets = {}
    for (n, b), s in states.items():
        res = simulate(price, s, cost=COST_ONE_WAY)
        if res:
            daily_rets[(n, b)] = res["curve"].pct_change().dropna()
    common = None
    for ser in daily_rets.values():
        common = ser.index if common is None else common.intersection(ser.index)
    M = np.column_stack([daily_rets[k].reindex(common).fillna(0.0).values for k in daily_rets])
    pp_sharpes = M.mean(0) / (M.std(0, ddof=1) + 1e-12)
    best_col = list(daily_rets.keys()).index((best_len, best_band))
    dsr, sr_hat, sr0 = deflated_sharpe(M[:, best_col], pp_sharpes, len(daily_rets))
    print(f"  best candidate (len={best_len}, band={best_band}): per-period SR {sr_hat:.4f}  "
          f"haircut SR0 {sr0:.4f}  DSR (P[true SR>0]) = {dsr*100:.1f}%")
    pbo, n_splits = pbo_cscv(M, S=12)
    print(f"  PBO (probability the in-sample winner is an out-of-sample loser): "
          f"{pbo*100:.1f}%  ({n_splits} CSCV splits)")

    # ---- 6. episode table: does it dodge the crashes buy-hold eats? ---------- #
    print("\n=== 6. EPISODE TABLE: strategy vs buy-and-hold drawdown through each BTC bear ===")
    episodes = {
        "2014-15 post-boom bear": ("2014-09-17", "2015-08-24"),
        "2017 top -> 2018 crash": ("2017-11-01", "2018-12-31"),
        "2020 COVID crash+recov": ("2020-02-01", "2020-07-01"),
        "2021 top -> 2022 crash": ("2021-10-01", "2022-12-31"),
        "2024-25 run": ("2024-01-01", price.index[-1].strftime("%Y-%m-%d")),
    }
    strat_curve = simulate(price, best_state, cost=COST_ONE_WAY)["curve"]
    for lbl, (a, b) in episodes.items():
        p_seg = price.loc[a:b]
        if len(p_seg) < 10:
            continue
        s_seg = strat_curve.loc[a:b]
        bh_ret = p_seg.iloc[-1] / p_seg.iloc[0] - 1
        strat_ret = s_seg.iloc[-1] / s_seg.iloc[0] - 1 if len(s_seg) else np.nan
        bh_dd = ((p_seg / p_seg.cummax()) - 1).min()
        strat_dd = ((s_seg / s_seg.cummax()) - 1).min() if len(s_seg) else np.nan
        print(f"  {lbl:<26} buy-hold {bh_ret*100:+7.1f}% (maxDD {bh_dd*100:6.1f}%)   "
              f"strategy {strat_ret*100:+7.1f}% (maxDD {strat_dd*100:6.1f}%)")

    # ---- 7. qualitative pre-2014 context (blockchain.info; not in the grid) -- #
    print("\n=== 7. PRE-2014 CONTEXT (blockchain.info; descriptive only, not backtested) ===")
    ext = D.extended_context_series(panel)
    for lbl, (a, b) in {"2011 bubble/crash": ("2011-05-01", "2011-12-31"),
                        "2013 boom -> 2015 bear": ("2013-11-01", "2015-01-15")}.items():
        seg = ext.loc[a:b]
        if len(seg) < 10:
            continue
        dd = ((seg / seg.cummax()) - 1).min()
        print(f"  {lbl:<26} {seg.index[0].date()} @ {seg.iloc[0]:,.2f} -> "
              f"{seg.index[-1].date()} @ {seg.iloc[-1]:,.2f}   maxDD {dd*100:.1f}%")

    # ---- 8. current reading --------------------------------------------------- #
    print(f"\n=== 8. CURRENT READING (best weekly-cadence candidate: len={best_len}, band={best_band}) ===")
    px_now = float(price.iloc[-1])
    ma = float(price.rolling(best_len).mean().iloc[-1])
    print(f"  BTC {px_now:,.2f}  as of {price.index[-1].date()}")
    print(f"  {best_len}d SMA {ma:,.2f}  extension {(px_now/ma-1)*100:+.1f}%  "
          f"band [{ma*(1-best_band):,.0f}, {ma*(1+best_band):,.0f}]  "
          f"state {'LONG' if best_state.iloc[-1] else 'CASH'}")


if __name__ == "__main__":
    main()
