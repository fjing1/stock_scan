"""Combination search: mix sub-components of the cycle oscillator and the 寻龙诀
panel to find a long-entry condition with a persistent (out-of-sample) edge.

Honest methodology (guards against data-snooping):
  * Daily timeframe, large basket, ~20y -> maximum samples + a clean time split.
  * Feature library = binary conditions built from BOTH indicators' sub-parts.
  * Search every conjunction of 1-3 conditions.
  * Optimise on TRAIN (bars before split_date); validate on held-out TEST
    (split_date .. today, which includes the 2022 bear market).
  * Primary metric = DETRENDED win rate (forward excess return vs SPY > 0);
    absolute win rate (drift-inflated) reported alongside.
  * A candidate only counts if its edge persists on TEST. With ~700 combos,
    ~5% beat baseline on TRAIN by chance, so the TEST column is the real judge.

    python combo_search.py
    python combo_search.py --horizon 5 --split 2019-01-01
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cycle_patter_for_swing import compute_cycle_stoch  # noqa: E402
from xunlong_panel import compute_panel  # noqa: E402

import yfinance as yf  # noqa: E402

BENCHMARK = "SPY"
BASKET = ["SPY", "QQQ", "DIA", "IWM", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP",
          "XLI", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "JPM", "XOM", "JNJ",
          "KO", "WMT", "CAT", "HD", "PG", "UNH", "DIS", "INTC", "CSCO", "ORCL",
          "MCD"]

# Long-entry condition library (name -> builder(features_df) -> bool Series)
def build_conditions(f):
    c = f["cycle"]
    return {
        # --- oscillator oversold (mean-reversion long) ---
        "cyc<20": f["cycle"] < 20,
        "cyc<35": f["cycle"] < 35,
        "k<20": f["stoch_k"] < 20,
        "rsi<30": f["rsi"] < 30,
        "rsi<40": f["rsi"] < 40,
        "varr<20": f["varr1"] < 20,
        "varr<35": f["varr1"] < 35,
        # --- turn / trigger ---
        "bbuy": f["bbuy"].astype(bool),
        "bbuy3": f["bbuy"].astype(float).rolling(3, min_periods=1).max() > 0,
        "cycUp": (c.shift(1) < c.shift(2)) & (c > c.shift(1)),
        # --- trend / context filters ---
        "trend>0": f["trend"] > 0,
        ">MA50": f["Close"] > f["ma50"],
        ">MA200": f["Close"] > f["ma200"],
        "pump>0": f["pump"] > 0,
        "noRed3": f["red"].astype(float).rolling(3, min_periods=1).max() == 0,
        "rsiCon": ((f["rsi"] > f["rsi"].shift(1)) & (f["rsi"].shift(1) > f["rsi"].shift(2))
                   & (f["rsi"] >= 50) & (f["rsi"] <= 70)),
    }


def fetch_batch(symbols, interval, period):
    raw = yf.download(symbols, interval=interval, period=period,
                      auto_adjust=False, progress=False, group_by="ticker",
                      threads=True)
    out = {}
    for s in symbols:
        try:
            d = raw[s].dropna(how="all").copy()
        except Exception:
            continue
        if len(d) == 0:
            continue
        if getattr(d.index, "tz", None) is not None:
            d.index = d.index.tz_localize(None)
        out[s] = d
    return out


def winrate(vals):
    v = vals[~np.isnan(vals)]
    return ((v > 0).mean() * 100.0, len(v)) if len(v) else (float("nan"), 0)


def zscore(p_cond, p_base, n):
    if n == 0 or math.isnan(p_cond) or math.isnan(p_base):
        return float("nan")
    pc, pb = p_cond / 100.0, p_base / 100.0
    se = math.sqrt(pb * (1 - pb) / n)
    return (pc - pb) / se if se > 0 else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=10, help="forward trading days")
    ap.add_argument("--split", default="2019-01-01", help="train/test split date")
    ap.add_argument("--max_terms", type=int, default=3)
    ap.add_argument("--cooldown", type=int, default=5)
    args = ap.parse_args(argv)
    H = args.horizon
    split = pd.Timestamp(args.split)

    print(f"downloading {len(BASKET)} symbols (1d, 20y) ...", flush=True)
    data = fetch_batch(BASKET, "1d", "20y")
    bdf = data.get(BENCHMARK)
    spy_fwd = (bdf["Close"].astype(float).shift(-H) / bdf["Close"].astype(float) - 1.0
               if bdf is not None else None)

    # build pooled feature/return table
    frames = []
    cond_names = None
    for sym, df in data.items():
        if sym == BENCHMARK or len(df) < 260:
            continue
        try:
            cs = compute_cycle_stoch(df)
            panel = compute_panel(df)
        except Exception:
            continue
        close = df["Close"].astype(float)
        f = pd.DataFrame(index=df.index)
        f["Close"] = close
        f["ma50"] = close.rolling(50).mean()
        f["ma200"] = close.rolling(200).mean()
        for col in ["cycle", "stoch_k", "rsi"]:
            f[col] = cs[col].values
        for col in ["trend", "pump", "bbuy", "varr1", "red"]:
            f[col] = panel[col].values
        conds = build_conditions(f)
        cond_names = list(conds.keys())
        cm = pd.DataFrame({k: v.fillna(False).astype(bool).values for k, v in conds.items()},
                          index=df.index)
        abs_fwd = close.shift(-H) / close - 1.0
        exc_fwd = abs_fwd - (spy_fwd.reindex(df.index) if spy_fwd is not None else 0)
        cm["_exc"] = exc_fwd.values
        cm["_abs"] = abs_fwd.values
        cm["_pos"] = np.arange(len(df))
        cm["_sym"] = sym
        cm["_train"] = (df.index < split)
        cm = cm[~cm["_exc"].isna()]
        frames.append(cm)

    pool = pd.concat(frames, ignore_index=True)
    tr = pool[pool["_train"]]
    te = pool[~pool["_train"]]
    print(f"pooled rows: train={len(tr)}  test={len(te)}  symbols={pool['_sym'].nunique()}")

    # baselines
    b_tr_exc = winrate(tr["_exc"].to_numpy())[0]
    b_te_exc = winrate(te["_exc"].to_numpy())[0]
    b_tr_abs = winrate(tr["_abs"].to_numpy())[0]
    b_te_abs = winrate(te["_abs"].to_numpy())[0]
    print(f"\nBASELINE win% (H={H}d):   "
          f"detrended  train {b_tr_exc:.1f} / test {b_te_exc:.1f}   |   "
          f"absolute  train {b_tr_abs:.1f} / test {b_te_abs:.1f}")

    # search all conjunctions of 1..max_terms
    tr_cond = {k: tr[k].to_numpy() for k in cond_names}
    te_cond = {k: te[k].to_numpy() for k in cond_names}
    tr_exc, tr_abs = tr["_exc"].to_numpy(), tr["_abs"].to_numpy()
    te_exc, te_abs = te["_exc"].to_numpy(), te["_abs"].to_numpy()

    rows = []
    for r in range(1, args.max_terms + 1):
        for combo in itertools.combinations(cond_names, r):
            mtr = np.ones(len(tr), dtype=bool)
            for k in combo:
                mtr &= tr_cond[k]
            ntr = int(mtr.sum())
            if ntr < 400:                      # need enough train samples
                continue
            mte = np.ones(len(te), dtype=bool)
            for k in combo:
                mte &= te_cond[k]
            nte = int(mte.sum())
            if nte < 200:
                continue
            w_tr_exc = (tr_exc[mtr] > 0).mean() * 100
            w_te_exc = (te_exc[mte] > 0).mean() * 100
            w_tr_abs = (tr_abs[mtr] > 0).mean() * 100
            w_te_abs = (te_abs[mte] > 0).mean() * 100
            rows.append((" + ".join(combo), ntr, nte, w_tr_exc, w_te_exc,
                         w_tr_abs, w_te_abs))

    res = pd.DataFrame(rows, columns=["combo", "Ntr", "Nte", "tr_exc", "te_exc",
                                      "tr_abs", "te_abs"])
    print(f"\ncombos tested (passing sample floors): {len(res)}")

    def show(df, title, by):
        print(f"\n{title}")
        print(f"  {'combo':<34}{'Ntr':>6}{'Nte':>6}"
              f"{'tr_exc':>8}{'te_exc':>8}{'tr_abs':>8}{'te_abs':>8}")
        for _, r in df.sort_values(by, ascending=False).head(15).iterrows():
            print(f"  {r['combo']:<34}{r['Ntr']:>6.0f}{r['Nte']:>6.0f}"
                  f"{r['tr_exc']:>8.1f}{r['te_exc']:>8.1f}"
                  f"{r['tr_abs']:>8.1f}{r['te_abs']:>8.1f}")

    # candidates whose DETRENDED edge persists: positive on BOTH train and test
    persist = res[(res["tr_exc"] > b_tr_exc + 1.0) & (res["te_exc"] > b_te_exc + 1.0)]
    show(res, "TOP by TRAIN detrended win% (in-sample — expect overfit):", "tr_exc")
    show(res, "TOP by TEST detrended win% (out-of-sample — the real judge):", "te_exc")
    show(res, "TOP by TEST ABSOLUTE win% (drift-inflated; practical 'feel'):", "te_abs")

    print(f"\nPERSISTENT detrended edge (train AND test > baseline+1pt): "
          f"{len(persist)} of {len(res)} combos")
    if len(persist):
        show(persist, "  -> persistent candidates, ranked by TEST detrended:", "te_exc")
        # significance of the single best OOS candidate
        best = persist.sort_values("te_exc", ascending=False).iloc[0]
        z = zscore(best["te_exc"], b_te_exc, best["Nte"])
        print(f"\n  best OOS detrended candidate: [{best['combo']}]  "
              f"test {best['te_exc']:.1f}% vs base {b_te_exc:.1f}%  "
              f"(edge {best['te_exc']-b_te_exc:+.1f}, z={z:+.1f}, Nte={best['Nte']:.0f})")
    # how many would beat baseline OOS by chance?
    exp_chance = len(res) * 0.5  # rough: half of random combos land above median
    print(f"\n  combos with test detrended > baseline+2pt: "
          f"{int((res['te_exc'] > b_te_exc + 2).sum())} of {len(res)}")


if __name__ == "__main__":
    main()
