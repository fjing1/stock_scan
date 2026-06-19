#!/usr/bin/env python3
"""Stage 2 of the exit-strategy backtest: simulate exit strategies on every
historical formal-buy (Gann_BUY_A) entry and rank by return, with a
walk-forward (out-of-sample) split and a Deflated-Sharpe haircut.

Entry convention: enter at the CLOSE of the signal bar (matches the system's
own D0_close "买入价"). Exits are evaluated from the next bar forward.

Each strategy returns, per trade: realized return (net of round-trip cost),
holding days, and exit reason. We aggregate per (strategy, param) combo:
  - mean / median per-trade return, win rate, profit factor, avg hold
  - a non-overlapping single-position-per-symbol equity curve -> CAGR, Sharpe,
    MaxDD, MAR (realistic portfolio path; avoids pyramiding the overlapping entries)
Then we pick each family's best param IN-SAMPLE (train) and report OUT-OF-SAMPLE
(test) to guard against overfitting, plus a Deflated Sharpe Ratio over all trials.
"""
import pickle
from pathlib import Path

import os
import numpy as np
import pandas as pd

import scan_stocks as scan

PANEL = Path(os.environ.get("EXIT_BT_PANEL", str(scan.BASE_DIR / "reports" / "exit_cache" / "panel.pkl")))
OUT_DIR = scan.BASE_DIR / "reports"
HORIZON = 120          # max trading days a trade can stay open
COST = float(os.environ.get("EXIT_BT_COST", "0.0010"))  # round-trip cost; env-overridable for sensitivity
WF_SPLIT = pd.Timestamp("2024-01-01")  # train: exit<split ; test: exit>=split
TRADING_DAYS = 252

# ----------------------------------------------------------------------------
# Per-trade exit simulation. A "trade" = entry at close[e]; we look at bars
# e+1 .. e+HORIZON. Each rule yields (offset, price) where offset is #days held
# and price is the fill. We combine rules by earliest offset (stop ties win).
# ----------------------------------------------------------------------------

def _first_true(mask):
    """Index of first True in a boolean np array, or -1."""
    nz = np.flatnonzero(mask)
    return int(nz[0]) if nz.size else -1


def simulate(panel_arrays, e, strat, p):
    """Return (ret, hold_days, reason) for one entry e under strategy strat/params p."""
    O, H, L, C = panel_arrays["O"], panel_arrays["H"], panel_arrays["L"], panel_arrays["C"]
    n = len(C)
    entry_px = C[e]
    j0 = e + 1
    j1 = min(e + 1 + HORIZON, n)
    if j1 <= j0:
        return None
    fo, fh, fl, fc = O[j0:j1], H[j0:j1], L[j0:j1], C[j0:j1]
    m = len(fc)
    sig = panel_arrays["SELL_1"][j0:j1]          # signal exit (close)
    atr_col = p.get("atr_col", "ATR22")
    atr_e = panel_arrays[atr_col][e]
    # prior-bar ATR for trailing stops (avoids intrabar look-ahead)
    atr_prev = panel_arrays[atr_col + "_prev"][j0:j1]
    atr_prev = np.where(np.isfinite(atr_prev), atr_prev, atr_e)

    cands = []  # (offset_index_into_forward, price, reason)

    def add_stop(level, reason):
        hit = _first_true(fl <= level)
        if hit >= 0:
            px = min(fo[hit], level)             # gap-down fills at open
            cands.append((hit, px, reason))

    def add_target(level, reason):
        hit = _first_true(fh >= level)
        if hit >= 0:
            px = max(fo[hit], level)             # gap-up fills at open
            cands.append((hit, px, reason))

    def add_close_rule(mask, price_arr, reason):
        hit = _first_true(mask)
        if hit >= 0:
            cands.append((hit, price_arr[hit], reason))

    def add_signal():
        hit = _first_true(sig)
        if hit >= 0:
            cands.append((hit, fc[hit], "signal"))

    kind = strat
    if kind == "baseline_signal":
        add_signal()

    elif kind == "hard_atr_stop":
        if np.isfinite(atr_e):
            add_stop(entry_px - p["m"] * atr_e, "atr_stop")
        add_signal()

    elif kind == "fixed_pct_stop":
        add_stop(entry_px * (1 - p["s"]), "pct_stop")
        add_signal()

    elif kind == "chandelier":
        if np.isfinite(atr_e):
            # peak high through PRIOR bar (incl. entry day high) -> stop known at bar open
            run_peak = np.maximum.accumulate(np.concatenate([[H[e]], fh]))[:-1]
            stop_series = run_peak - p["m"] * atr_prev
            hit = _first_true(fl <= stop_series)
            if hit >= 0:
                cands.append((hit, min(fo[hit], stop_series[hit]), "chandelier"))
        if p.get("with_signal", False):
            add_signal()

    elif kind == "ma_trail":
        ma = panel_arrays[p["ma_col"]][j0:j1]
        add_close_rule((fc < ma) & np.isfinite(ma), fc, "ma_trail")

    elif kind == "r_target":
        if np.isfinite(atr_e):
            stop = entry_px - p["base_m"] * atr_e
            R = entry_px - stop
            add_stop(stop, "atr_stop")
            add_target(entry_px + p["k"] * R, "r_target")
        add_signal()

    elif kind == "breakeven_trail":
        if np.isfinite(atr_e):
            init_stop = entry_px - p["base_m"] * atr_e
            R = entry_px - init_stop
            t_be = _first_true(fh >= entry_px + p["j"] * R)
            run_peak = np.maximum.accumulate(np.concatenate([[H[e]], fh]))[:-1]
            trail = run_peak - p["trail_m"] * atr_prev
            stop_series = np.full(m, init_stop, dtype=float)
            if t_be >= 0:
                # from t_be onward: max(breakeven, trailing)
                be_level = np.maximum(entry_px, trail)
                stop_series[t_be:] = np.maximum(init_stop, be_level[t_be:])
            hit = _first_true(fl <= stop_series)
            if hit >= 0:
                cands.append((hit, min(fo[hit], stop_series[hit]), "be_trail"))
        add_signal()

    elif kind == "donchian":
        if np.isfinite(atr_e):
            add_stop(entry_px - 2.0 * atr_e, "atr2_stop")
        donch = panel_arrays[p["donch_col"]][j0:j1]   # rolling-min already excludes today via shift below
        add_close_rule((fc <= donch) & np.isfinite(donch), fc, "donchian")

    elif kind == "wide_atr_trail":
        if np.isfinite(atr_e):
            run_peak = np.maximum.accumulate(np.concatenate([[H[e]], fh]))[:-1]
            stop_series = run_peak - p["m"] * atr_prev
            hit = _first_true(fl <= stop_series)
            if hit >= 0:
                cands.append((hit, min(fo[hit], stop_series[hit]), "wide_trail"))

    elif kind == "sar":
        sar = panel_arrays[p["sar_col"]][j0:j1]
        hit = _first_true(fl <= sar)
        if hit >= 0:
            cands.append((hit, min(fo[hit], sar[hit]), "sar"))

    elif kind == "time_cond_signal":
        # exit at signal; OR at day N if return-to-date < thresh (dead money)
        add_signal()
        N = p["N"]
        if N - 1 < m:
            ret_at_N = fc[N - 1] / entry_px - 1.0
            if ret_at_N < p["thresh"]:
                cands.append((N - 1, fc[N - 1], "time_cond"))

    else:
        raise ValueError(kind)

    if cands:
        cands.sort(key=lambda x: (x[0], 0 if "stop" in x[2] else 1))
        off, px, reason = cands[0]
        hold = off + 1
    else:
        off = m - 1
        px = fc[off]
        hold = m
        reason = "horizon"

    ret = px / entry_px - 1.0 - COST
    return ret, hold, px, reason


# ----------------------------------------------------------------------------
# Strategy grid
# ----------------------------------------------------------------------------
def build_grid():
    grid = []
    grid.append(("baseline_signal", {}, "Baseline: signal exit only (current behavior)"))
    for mm in [2.0, 2.5, 3.0, 4.0]:
        grid.append(("hard_atr_stop", {"m": mm, "atr_col": "ATR22"}, f"Hard ATR stop {mm}x + signal"))
    for ss in [0.07, 0.08, 0.10, 0.12, 0.15]:
        grid.append(("fixed_pct_stop", {"s": ss}, f"Fixed {int(ss*100)}% stop + signal"))
    for mm in [2.5, 3.0, 4.0]:
        grid.append(("chandelier", {"m": mm, "atr_col": "ATR22", "with_signal": False}, f"Chandelier {mm}xATR22 trail (standalone)"))
    for mm in [5.0, 6.0]:
        grid.append(("wide_atr_trail", {"m": mm, "atr_col": "ATR22"}, f"Wide ATR {mm}x trail, no target"))
    for col, lbl in [("EMA20", "EMA20"), ("SMA50", "SMA50"), ("SMA100", "SMA100")]:
        grid.append(("ma_trail", {"ma_col": col}, f"MA trail: close<{lbl}"))
    for kk in [2.0, 3.0, 4.0, 5.0]:
        grid.append(("r_target", {"k": kk, "base_m": 2.5, "atr_col": "ATR22"}, f"R-target {kk}R (2.5ATR stop) + signal"))
    for jj in [1.0, 1.5]:
        grid.append(("breakeven_trail", {"j": jj, "base_m": 2.5, "trail_m": 3.0, "atr_col": "ATR22"}, f"Breakeven@{jj}R then 3xATR trail + signal"))
    for col, lbl in [("DONCH10", "10d"), ("DONCH20", "20d")]:
        grid.append(("donchian", {"donch_col": col, "atr_col": "ATR22"}, f"Donchian {lbl}-low exit + 2ATR stop"))
    grid.append(("sar", {"sar_col": "SAR_STD"}, "Parabolic SAR (0.02/0.20)"))
    grid.append(("sar", {"sar_col": "SAR_SLOW"}, "Parabolic SAR slow (0.01/0.10)"))
    for N in [10, 20]:
        grid.append(("time_cond_signal", {"N": N, "thresh": 0.0}, f"Signal + dead-money time stop @{N}d if <0%"))
    return grid


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def trade_stats(rets):
    rets = np.asarray(rets, dtype=float)
    rets = rets[np.isfinite(rets)]
    if rets.size == 0:
        return {}
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf
    return {
        "n": rets.size,
        "mean": rets.mean(),
        "median": np.median(rets),
        "win%": 100.0 * (rets > 0).mean(),
        "avg_win": wins.mean() if wins.size else 0.0,
        "avg_loss": losses.mean() if losses.size else 0.0,
        "pf": pf,
        "expectancy": rets.mean(),
        "std": rets.std(ddof=1) if rets.size > 1 else 0.0,
    }


def portfolio_metrics(daily_ret_by_date, calendar, date_lo=None, date_hi=None):
    """Equal-weight-across-open-positions portfolio, always fully invested when
    >=1 position is open, cash (0%) on idle days -> honest CAGR with cash drag.
    daily_ret_by_date: dict[Timestamp] -> list of per-position daily returns.
    calendar: full sorted DatetimeIndex of market days to span (gives cash drag)."""
    cal = calendar
    if date_lo is not None:
        cal = cal[cal >= date_lo]
    if date_hi is not None:
        cal = cal[cal < date_hi]
    if len(cal) < 30:
        return {}
    port = np.array([np.mean(daily_ret_by_date[d]) if d in daily_ret_by_date else 0.0 for d in cal])
    active = np.array([d in daily_ret_by_date for d in cal])
    eq = np.cumprod(1 + port)
    yrs = max(len(cal) / TRADING_DAYS, 0.25)
    cagr = eq[-1] ** (1 / yrs) - 1
    peak = np.maximum.accumulate(eq)
    maxdd = ((eq - peak) / peak).min()
    vol = port.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = (port.mean() / port.std(ddof=1) * np.sqrt(TRADING_DAYS)) if port.std(ddof=1) > 0 else 0.0
    return {
        "cagr": cagr, "vol": vol, "sharpe": sharpe, "maxdd": maxdd,
        "mar": (cagr / abs(maxdd)) if maxdd < 0 else np.inf,
        "time_in_mkt": active.mean(), "total_return": eq[-1] - 1,
    }


def deflated_sharpe(best_sr, all_srs, n_obs):
    """Bailey & Lopez de Prado DSR haircut for selecting the best of N trials."""
    from math import sqrt
    from statistics import NormalDist
    srs = np.asarray([s for s in all_srs if np.isfinite(s)])
    N = len(srs)
    if N < 2 or n_obs < 10:
        return np.nan
    var_sr = srs.var(ddof=1)
    if var_sr <= 0:
        return np.nan
    nd = NormalDist()
    emc = 0.5772156649
    e_max = (1 - emc) * nd.inv_cdf(1 - 1.0 / N) + emc * nd.inv_cdf(1 - 1.0 / (N * np.e))
    sr0 = sqrt(var_sr) * e_max
    dsr = nd.cdf(((best_sr - sr0) * sqrt(n_obs - 1)))
    return dsr, sr0


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    with open(PANEL, "rb") as f:
        panel = pickle.load(f)
    print(f"Loaded panel: {len(panel)} symbols")

    # pre-extract numpy arrays per symbol
    arrays = {}
    entries = []   # (sym, entry_idx, entry_date)
    for sym, df in panel.items():
        a = {
            "O": df["Open"].values.astype(float),
            "H": df["High"].values.astype(float),
            "L": df["Low"].values.astype(float),
            "C": df["Close"].values.astype(float),
            "SELL_1": df["SELL_1"].values.astype(bool),
            "ATR14": df["ATR14"].values.astype(float),
            "ATR22": df["ATR22"].values.astype(float),
            "ATR14_prev": df["ATR14"].shift(1).values.astype(float),
            "ATR22_prev": df["ATR22"].shift(1).values.astype(float),
            "EMA20": df["EMA20"].values.astype(float),
            "SMA50": df["SMA50"].values.astype(float),
            "SMA100": df["SMA100"].values.astype(float),
        }
        # donchian "prior N low" = rolling-min shifted by 1 so today is excluded
        a["DONCH10"] = df["DONCH10"].shift(1).values.astype(float)
        a["DONCH20"] = df["DONCH20"].shift(1).values.astype(float)
        a["SAR_STD"] = df["SAR_STD"].values.astype(float)
        a["SAR_SLOW"] = df["SAR_SLOW"].values.astype(float)
        arrays[sym] = a
        idx = df.index
        buy_locs = np.flatnonzero(df["BUY_A"].values.astype(bool))
        for e in buy_locs:
            if e < 110:        # need indicator warmup (SMA100 etc.)
                continue
            entries.append((sym, int(e), idx[e]))

    print(f"Total entries (after warmup filter): {len(entries)}")
    dates = pd.to_datetime([d for _, _, d in entries])
    print(f"Entry date range: {dates.min().date()} .. {dates.max().date()}")

    # market calendar (union of all panel dates) -> gives honest cash drag
    cal = pd.DatetimeIndex(sorted(set().union(*[set(panel[s].index) for s in panel])))
    cal = cal[(cal >= dates.min())]
    idx_cache = {sym: panel[sym].index for sym in panel}

    grid = build_grid()
    print(f"Strategies in grid: {len(grid)}\n")

    from collections import defaultdict
    rows = []
    for strat, p, label in grid:
        rets_all, holds = [], []
        daily = defaultdict(list)          # date -> [per-position daily returns]
        for sym, e, edate in entries:
            res = simulate(arrays[sym], e, strat, p)
            if res is None:
                continue
            ret, hold, px, reason = res
            rets_all.append(ret)
            holds.append(hold)
            # reconstruct the daily P&L path for the portfolio curve
            C = arrays[sym]["C"]
            sidx = idx_cache[sym]
            for k in range(1, hold + 1):
                bar = e + k
                if bar >= len(C):
                    break
                prev = C[bar - 1]
                if k == hold:
                    r = px / prev - 1.0 - COST    # charge round-trip cost on exit -> turnover-aware
                else:
                    r = C[bar] / prev - 1.0
                daily[sidx[bar]].append(r)
        st = trade_stats(rets_all)
        full = portfolio_metrics(daily, cal)
        tr = portfolio_metrics(daily, cal, date_hi=WF_SPLIT)
        te = portfolio_metrics(daily, cal, date_lo=WF_SPLIT)
        rows.append({
            "strat": strat, "label": label, "params": p,
            **{k: st.get(k) for k in ["n", "mean", "median", "win%", "avg_win", "avg_loss", "pf"]},
            "avg_hold": float(np.mean(holds)) if holds else np.nan,
            "cagr": full.get("cagr"), "vol": full.get("vol"), "sharpe": full.get("sharpe"),
            "maxdd": full.get("maxdd"), "mar": full.get("mar"), "tim": full.get("time_in_mkt"),
            "train_cagr": tr.get("cagr"), "train_sharpe": tr.get("sharpe"),
            "test_cagr": te.get("cagr"), "test_sharpe": te.get("sharpe"),
            "test_maxdd": te.get("maxdd"), "test_mar": te.get("mar"),
        })

    res = pd.DataFrame(rows)
    srs = res["sharpe"].dropna().values
    base = res.loc[res["label"].str.startswith("Baseline")].iloc[0]

    def fmt(df, cols, sort, n=None):
        d = df.sort_values(sort, ascending=False).copy()
        if n:
            d = d.head(n)
        return d[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}")

    pd.set_option("display.width", 260)
    print("=" * 124)
    print("FULL-SAMPLE LEADERBOARD  (ranked by portfolio CAGR; equal-wt across open positions, cash on idle days)")
    print(f"cost={COST:.2%} round-trip | horizon={HORIZON}d | entry=signal-day close | {len(cal)} market days")
    print("=" * 124)
    cols = ["label", "n", "avg_hold", "win%", "pf", "mean", "cagr", "vol", "sharpe", "maxdd", "mar", "tim"]
    print(fmt(res, cols, "cagr"))

    print("\n" + "=" * 124)
    print("WALK-FORWARD (out-of-sample):  train exit<2024-01-01  ->  test exit>=2024-01-01   (ranked by TEST CAGR)")
    print("=" * 124)
    cols2 = ["label", "train_cagr", "train_sharpe", "test_cagr", "test_sharpe", "test_maxdd", "test_mar"]
    print(fmt(res, cols2, "test_cagr"))

    # family-best generalization: pick best param per family IN-SAMPLE, show OOS
    res["family"] = res["strat"]
    print("\n" + "=" * 124)
    print("PER-FAMILY: pick best param by TRAIN CAGR (in-sample), then show its OUT-OF-SAMPLE result")
    print("=" * 124)
    fam_rows = []
    for fam, g in res.groupby("family"):
        bi = g.loc[g["train_cagr"].idxmax()]
        fam_rows.append({"family": fam, "is_best_label": bi["label"],
                         "train_cagr": bi["train_cagr"], "test_cagr": bi["test_cagr"],
                         "test_sharpe": bi["test_sharpe"], "test_maxdd": bi["test_maxdd"]})
    fam_df = pd.DataFrame(fam_rows).sort_values("test_cagr", ascending=False)
    print(fam_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    dsr = deflated_sharpe(srs.max(), srs, TRADING_DAYS * 3)
    print("\n" + "=" * 124)
    print("OVERFIT HAIRCUT  (Deflated Sharpe over the whole grid)")
    print("=" * 124)
    print(f"Baseline (signal-only):  CAGR {base['cagr']:.2%}  Sharpe {base['sharpe']:.2f}  MaxDD {base['maxdd']:.2%}  MAR {base['mar']:.2f}")
    print(f"Best Sharpe in grid:     {srs.max():.2f}    Trials: {len(srs)}")
    if isinstance(dsr, tuple):
        print(f"Deflated Sharpe Ratio P[SR>SR0]: {dsr[0]:.3f}  (multiple-testing SR0 threshold = {dsr[1]:.2f})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.drop(columns=["params"]).to_csv(OUT_DIR / "exit_backtest_results.csv", index=False)
    print(f"\nSaved full results -> {OUT_DIR / 'exit_backtest_results.csv'}")


if __name__ == "__main__":
    main()
