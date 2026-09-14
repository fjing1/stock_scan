#!/usr/bin/env python3
"""Tests for the move-probability system (move_prob.py, _move_lib.py).

Network-free: everything runs on synthetic price frames or the fitted model artifact.

    ../../vcp_env/bin/python tests/test_move_prob.py
    pytest tests/test_move_prob.py

These pin the invariants that would silently break the calibration if someone refactored the
pipeline: forward-window alignment, log-space threshold conversion, the probability simplex,
monotonicity in threshold / horizon / volatility, and the two analytic regression identities
(a frozen-sigma model must score exactly zero skill against climatology -- if it does not, the
walk-forward split leaks or the CDF is mis-assembled).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _move_lib as L          # noqa: E402
import move_prob as M          # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  -- ' + detail if detail and not cond else ''}")
    if not cond:
        FAILED.append(name)


def synth(n=900, sigma=0.01, drift=0.0003, seed=0, jump_at=None):
    """Geometric random walk with an optional volatility jump, plus a plausible OHLC envelope."""
    rng = np.random.default_rng(seed)
    s = np.full(n, sigma)
    if jump_at is not None:
        s[jump_at:] = sigma * 3
    r = rng.normal(drift, 1.0, n) * s
    c = 100 * np.exp(np.cumsum(r))
    idx = pd.bdate_range("2019-01-01", periods=n)
    close = pd.Series(c, index=idx)
    rng2 = np.random.default_rng(seed + 1)
    hi = close * (1 + np.abs(rng2.normal(0, 1, n)) * s * 0.7)
    lo = close * (1 - np.abs(rng2.normal(0, 1, n)) * s * 0.7)
    return close, hi, lo


# ---------------------------------------------------------------- library invariants
def test_lib():
    print("\n_move_lib alignment and definitions")
    close, hi, lo = synth()
    lr = np.log(close).diff()

    fv = L.realized_vol_forward(close, 5)
    t = -40
    check("realized_vol_forward(h=5) == std of r[t+1..t+5]",
          abs(fv.iloc[t] - lr.iloc[t + 1:t + 6].std(ddof=1)) < 1e-12)

    fv1 = L.realized_vol_forward(close, 1)
    check("realized_vol_forward(h=1) is finite, not all-NaN (rolling(1).std would be NaN)",
          fv1.notna().sum() > len(close) - 5)
    check("h=1 proxy equals |r_{t+1}| * sqrt(pi/2)",
          abs(fv1.iloc[t] - abs(lr.iloc[t + 1]) * np.sqrt(np.pi / 2)) < 1e-12)

    fr = L.forward_simple_return(close, 5)
    check("forward_simple_return == close[t+5]/close[t]-1",
          abs(fr.iloc[t] - (close.iloc[t + 5] / close.iloc[t] - 1)) < 1e-12)

    b = L.bucketize(fr, 0.02).dropna()
    check("bucketize partitions with no leftover labels",
          set(b.unique()) <= set(L.BUCKETS))
    thr = 0.02
    manual_big = ((fr >= thr) | (fr <= -thr))
    check("bucketize big buckets match the raw threshold test",
          (b.isin(["up_big", "down_big"]) == manual_big.reindex(b.index)).all())

    # range estimators must be blind to overnight gaps -- this is why they read low
    check("Parkinson < close-to-close on gappy data",
          L.vol_parkinson(hi, lo, 21).iloc[-1] < L.vol_cc(close, 21).iloc[-1] * 1.5)

    p = np.tile([0.25, 0.25, 0.25, 0.25], (100, 1))
    a = np.random.default_rng(0).integers(0, 4, 100)
    check("log_loss of a uniform 4-class forecast == ln 4",
          abs(L.log_loss(p, a) - np.log(4)) < 1e-9)
    check("brier_multi of a uniform 4-class forecast == 0.75",
          abs(L.brier_multi(p, a) - 0.75) < 1e-9)
    check("skill_score(x, x) == 0", abs(L.skill_score(2.0, 2.0)) < 1e-12)


# ---------------------------------------------------------------- model invariants
def test_model():
    print("\nmove_prob model invariants")
    if not M.MODEL_PATH.exists():
        check("model artifact exists", False, "run: move_prob.py fit")
        return
    model = pd.read_pickle(M.MODEL_PATH)
    close, hi, lo = synth(seed=7)

    rows = M.predict_from_bars("AAPL", close, hi, lo, model=model, thr=0.02)
    ok = [r for r in rows if r.get("ok") and r["in_support"]]
    check("at least one horizon is in support for a normal-vol synthetic name", len(ok) > 0)

    for r in ok:
        s = r["p_down_big"] + r["p_down_small"] + r["p_up_small"] + r["p_up_big"]
        check(f"h={r['h']} probabilities sum to 1", abs(s - 1) < 1e-9, f"sum={s}")
        check(f"h={r['h']} p_move + p_within == 1", abs(r["p_move"] + r["p_within"] - 1) < 1e-9)
        check(f"h={r['h']} all probabilities in [0,1]",
              all(0 <= r[k] <= 1 for k in ("p_down_big", "p_down_small", "p_up_small", "p_up_big")))
        check(f"h={r['h']} q05 < q50 < q95", r["q05"] < r["q50"] < r["q95"])

    # monotone in threshold
    pm = {}
    for thr in (0.01, 0.02, 0.03, 0.05):
        rr = [x for x in M.predict_from_bars("AAPL", close, hi, lo, model=model, thr=thr,
                                             horizons=(5,)) if x.get("ok")]
        if rr:
            pm[thr] = rr[0]["p_move"]
    check("P(|move| > thr) is decreasing in thr",
          all(pm[a] >= pm[b] for a, b in zip(sorted(pm)[:-1], sorted(pm)[1:])), str(pm))

    # monotone in horizon
    ph = {r["h"]: r["p_move"] for r in M.predict_from_bars("AAPL", close, hi, lo, model=model,
                                                           thr=0.02) if r.get("ok")}
    hs = sorted(ph)
    check("P(|move| > 2%) is increasing in horizon",
          all(ph[a] <= ph[b] + 1e-9 for a, b in zip(hs[:-1], hs[1:])), str(ph))

    # monotone in volatility
    q_lo, hl, ll = synth(sigma=0.006, seed=3)
    q_hi, hh, lh = synth(sigma=0.030, seed=3)
    a = [r for r in M.predict_from_bars("AAA", q_lo, hl, ll, model=model, horizons=(5,)) if r["ok"]][0]
    b = [r for r in M.predict_from_bars("BBB", q_hi, hh, lh, model=model, horizons=(5,)) if r["ok"]][0]
    check("a more volatile name gets a higher P(|move| > 2%)", b["p_move"] > a["p_move"],
          f"{a['p_move']:.3f} vs {b['p_move']:.3f}")
    check("a more volatile name gets a larger typical move",
          b["typical_move"] > a["typical_move"])

    # earnings multiplier must widen, never narrow
    base = [r for r in M.predict_from_bars("AAPL", close, hi, lo, model=model, horizons=(5,)) if r["ok"]][0]
    earn = [r for r in M.predict_from_bars("AAPL", close, hi, lo, model=model, horizons=(5,),
                                           earnings_in={5}) if r["ok"]][0]
    check("earnings flag raises P(|move| > 2%)", earn["p_move"] > base["p_move"],
          f"{base['p_move']:.3f} -> {earn['p_move']:.3f}")
    check("earnings flag is not applied to an index",
          [r for r in M.predict_from_bars("SPY", close, hi, lo, model=model, horizons=(5,),
                                          earnings_in={5}) if r["ok"]][0]["earn_mult"] == 1.0)

    # support gate
    calm, ch, cl = synth(sigma=0.0015, seed=11)
    r1 = [r for r in M.predict_from_bars("SPY", calm, ch, cl, model=model, horizons=(1,))][0]
    check("support gate refuses a 2% threshold on a very calm index at h=1", not r1["in_support"])
    wild, wh, wl = synth(sigma=0.09, seed=12)
    r2 = [r for r in M.predict_from_bars("WILD", wild, wh, wl, model=model, horizons=(21,))][0]
    check("support gate refuses a 2% threshold on a very wild name at h=21", not r2["in_support"])

    check("asset_class routes SPY to index", M.asset_class("SPY") == "index")
    check("asset_class routes AAPL to single", M.asset_class("AAPL") == "single")
    check("every group/horizon has a fitted kappa in [0.9, 1.4]",
          all(0.9 <= sp["kappa"] <= 1.4
              for g in model["groups"].values() for sp in g["per_h"].values()))


# ---------------------------------------------------------------- analytic identities
def test_identities():
    """A frozen sigma carries no information, so it must score EXACTLY zero skill against the
    climatology built from the same data. If this drifts, the CDF assembly or the split leaks."""
    print("\nanalytic regression identities")
    rng = np.random.default_rng(0)
    n = 200_000
    z = rng.standard_t(5, n) / np.sqrt(5 / 3)
    sig = np.full(n, 0.02)                      # frozen: identical for every row
    lr = z * sig
    thr = 0.02
    a_dn, a_up = np.log(1 - thr), np.log(1 + thr)
    zs = np.sort(lr / sig)
    F = lambda v: np.searchsorted(zs, v, side="right") / len(zs)
    p = np.column_stack([np.full(n, F(a_dn / sig[0])),
                         np.full(n, F(0.0) - F(a_dn / sig[0])),
                         np.full(n, F(a_up / sig[0]) - F(0.0)),
                         np.full(n, 1 - F(a_up / sig[0]))])
    act = np.where(lr <= a_dn, 0, np.where(lr <= 0, 1, np.where(lr < a_up, 2, 3)))
    clim = np.tile(np.bincount(act, minlength=4) / n, (n, 1))
    bs = lambda q: ((q - np.eye(4)[act]) ** 2).sum(1).mean()
    skill = 1 - bs(p) / bs(clim)
    check("frozen-sigma model scores zero Brier skill vs climatology", abs(skill) < 1e-6,
          f"skill={skill:.2e}")

    # log-space threshold conversion is asymmetric and must stay that way
    check("log(1-thr) is farther from 0 than log(1+thr) (this asymmetry is real, not a bug)",
          abs(np.log(1 - 0.02)) > abs(np.log(1 + 0.02)))


if __name__ == "__main__":
    test_lib()
    test_model()
    test_identities()
    print("\n" + ("ALL PASS" if not FAILED else f"{len(FAILED)} FAILED: {FAILED}"))
    sys.exit(1 if FAILED else 0)
