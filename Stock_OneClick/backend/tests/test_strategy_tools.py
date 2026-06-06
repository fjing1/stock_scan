#!/usr/bin/env python3
"""Tests for the strategy tooling: gate_calc, benchmark_ma, score_calibration.

Network-free, deterministic. Run with the project venv:
    ../../vcp_env/bin/python tests/test_strategy_tools.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import gate_calc
import benchmark_ma
import score_calibration as sc


# --------------------------- gate mapping ---------------------------
def test_gate_riskoff_states_force_cash():
    for st in ["看跌/避险", "风险升高"]:
        g, _, _ = gate_calc.gate(st, 0.10)
        assert g == 0, f"{st} should be 0% long"


def test_gate_high_sell_share_forces_cash():
    # 2026-06-05 real case: strong-ish label but SELL_share 0.93 -> risk-off override
    g, _, _ = gate_calc.gate("看跌/避险", 0.933)
    assert g == 0
    g2, _, _ = gate_calc.gate("强势/中性看涨", 0.70)   # share >= 0.60 overrides the label
    assert g2 == 0


def test_gate_caution_blocks_new_longs():
    g, _, action = gate_calc.gate("谨慎看涨", 0.10)
    assert g == 0 and "CAUTION" in action


def test_gate_strong_lowbreadth_is_full():
    g, _, _ = gate_calc.gate("强势/中性看涨", 0.20)
    assert g == 40


def test_gate_strong_softbreadth_is_half():
    g, _, _ = gate_calc.gate("强势/中性看涨", 0.35)   # 0.30 <= share < 0.40
    assert g == 20


def test_gate_blank_state_is_cash():
    g, _, _ = gate_calc.gate("", 0.10)
    assert g == 0


# --------------------------- MA de-risk benchmark ---------------------------
def _series(vals):
    idx = pd.bdate_range("2025-01-01", periods=len(vals))
    return pd.Series(vals, index=idx, dtype=float)


def test_ma_derisk_stays_in_during_uptrend():
    s = _series(list(np.linspace(100, 130, 30)))
    out = benchmark_ma.ma_derisk(s, window=3)
    assert out["pct_in_market"] > 0.7          # mostly invested in a clean uptrend
    assert out["strat_maxdd"] >= -0.02          # negligible drawdown


def test_ma_derisk_exits_and_reduces_drawdown_in_crash():
    up = list(np.linspace(100, 120, 15))
    crash = [116, 108, 100, 92, 86, 82]         # sharp decline below the MA
    out = benchmark_ma.ma_derisk(_series(up + crash), window=3)
    # buy&hold suffers the full crash; the MA rule exits next-day and bleeds less
    assert out["strat_maxdd"] > out["bench_maxdd"], "MA de-risk should cut drawdown vs hold"
    assert out["pct_in_market"] < 1.0
    assert out["switches"] >= 1


def test_ma_derisk_no_lookahead_first_bar_flat():
    out = benchmark_ma.ma_derisk(_series([100, 110, 90, 120]), window=2)
    # position is shift(1) of the in-market flag, so the first bar is always out (no peeking)
    assert out["position"].iloc[0] == 0.0


# --------------------------- IC helper ---------------------------
def test_ic_perfect_and_inverse():
    a = pd.Series([1, 2, 3, 4, 5], dtype=float)
    n, p, sp = sc.ic(a, a * 2 + 1)
    assert sp == 1.0 and p == 1.0
    n, p, sp = sc.ic(a, -a)
    assert sp == -1.0


def test_ic_constant_is_nan():
    a = pd.Series([1, 2, 3, 4], dtype=float)
    n, p, sp = sc.ic(a, pd.Series([5, 5, 5, 5], dtype=float))
    assert pd.isna(p) and pd.isna(sp)


def test_day_demean_zeroes_group_means():
    df = pd.DataFrame({"date": ["a", "a", "b", "b"], "x": [1.0, 3.0, 10.0, 20.0]})
    dm = sc.day_demean(df, "x")
    assert abs(dm.iloc[0] + 1.0) < 1e-9 and abs(dm.iloc[1] - 1.0) < 1e-9   # a-mean=2
    assert abs(dm.iloc[2] + 5.0) < 1e-9 and abs(dm.iloc[3] - 5.0) < 1e-9   # b-mean=15


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t(); passed += 1; print(f"  PASS  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"  FAIL  {t.__name__}: {exc}"); traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
