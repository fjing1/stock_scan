#!/usr/bin/env python3
"""Unit / regression tests for the Stock OneClick engine and indicator.

Network-free: everything runs on synthetic price frames or the injectable
data-fetcher seam added to ``scan_one_symbol``. Run with the project venv:

    ../../vcp_env/bin/python tests/test_engine.py     # plain runner
    pytest tests/test_engine.py                       # if pytest is installed

These pin the *behavioral invariants* of the Pine->Python port (the riskiest
code) and the scoring formula, so a future refactor that silently changes them
fails loudly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the backend package importable regardless of CWD.
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import xunlong as xl_mod  # noqa: E402
import scan_stocks as scan  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _ohlc_from_close(close: np.ndarray) -> pd.DataFrame:
    """Build a tidy OHLCV daily frame from a close path (tight H/L bands)."""
    close = np.asarray(close, dtype="float64")
    idx = pd.bdate_range("2025-01-01", periods=len(close))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = close * 1.004
    low = close * 0.996
    vol = np.full(len(close), 1_000_000.0)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


# --------------------------------------------------------------------------- #
# pure helper functions
# --------------------------------------------------------------------------- #
def test_safe_div_handles_zero_and_nan():
    out = xl_mod.safe_div([1.0, 2.0, 3.0], [0.0, np.nan, 2.0])
    assert out[0] == 0.0 and out[1] == 0.0 and out[2] == 1.5


def test_rma_matches_ewm_alpha():
    s = pd.Series(np.arange(1, 21, dtype="float64"))
    got = xl_mod.rma(s, 5)
    exp = s.ewm(alpha=1.0 / 5, adjust=False).mean()
    assert np.allclose(got.values, exp.values)


def test_xsa_is_finite_and_tracks_level():
    s = pd.Series(np.r_[np.full(10, 10.0), np.full(10, 20.0)])
    got = xl_mod.xsa(s, 5, 1)
    assert got.notna().sum() > 0
    # After a sustained step up, the smoother should rise toward the new level.
    assert got.iloc[-1] > got.iloc[9]


# --------------------------------------------------------------------------- #
# scoring formula (exact values, hand-computed)
# --------------------------------------------------------------------------- #
def test_score_non_buy_is_nan():
    row = pd.Series({"signal_side": "SELL", "signal_type": "正式卖出"})
    assert pd.isna(scan.score_buy_signal_row(row))


def test_score_formal_buy_clamps_to_100():
    # 50 +20(正式买入) +8(0出/BUY_A in model) +10(rank<=0.25) +7(RSI 42-65)
    #    +5(L2<=35) +4(H4_FJ<=55) +3(H4_RSI>=45) = 107 -> clamp 100
    row = pd.Series({
        "signal_side": "BUY", "signal_type": "正式买入", "model": "D1_BUY_A_0出",
        "rank120": 0.20, "RSI": 50.0, "L2_trend": 20.0, "H4_FJ": 40.0, "H4_RSI": 50.0,
    })
    assert scan.score_buy_signal_row(row) == 100.0


def test_score_first_buy_midrange_exact():
    # 50 +10(第一买入点) +0(model no BUY_A/0出) +3(rank 0.45-0.65) +3(RSI 35-42) = 66
    row = pd.Series({
        "signal_side": "BUY", "signal_type": "第一买入点", "model": "LOW_START_FIRST_GREEN",
        "rank120": 0.50, "RSI": 38.0, "L2_trend": 100.0,
    })
    assert scan.score_buy_signal_row(row) == 66.0


def test_score_overbought_penalty():
    # 50 +14(预警买入) +8(0出) -5(rank>=0.85) -6(RSI>75) = 61
    row = pd.Series({
        "signal_side": "BUY", "signal_type": "预警买入", "model": "H4_BUY_A_0出",
        "rank120": 0.90, "RSI": 80.0,
    })
    assert scan.score_buy_signal_row(row) == 61.0


def test_realtime_dashboard_score_delegates_to_engine():
    try:
        import realtime_dashboard as rt
    except ImportError as exc:
        # realtime_dashboard imports tkinter at module load; headless CI venvs
        # often lack _tkinter. The delegation is a one-liner to the engine, so
        # skip rather than fail when the GUI stack is unavailable.
        print(f"  SKIP (no GUI stack: {exc})", end="")
        return
    row = pd.Series({
        "signal_side": "BUY", "signal_type": "正式买入", "model": "D1_BUY_A_0出",
        "rank120": 0.20, "RSI": 50.0, "L2_trend": 20.0, "H4_FJ": 40.0, "H4_RSI": 50.0,
    })
    assert rt.score_signal_row(row) == scan.score_buy_signal_row(row) == 100.0


# --------------------------------------------------------------------------- #
# Gann Box structural invariants (the core of the Pine port)
# --------------------------------------------------------------------------- #
def _valid_leg_close_path():
    # build a low base, a strong >8% rise over many up-bars, then a decline.
    base = np.linspace(110, 100, 30)          # slow decline -> establishes the 0 low
    rise = np.linspace(100, 118, 25)          # +18% leg, many EMA-up bars
    fall = np.linspace(118, 108, 18)          # roll over -> EMA turns down -> 1出
    return np.r_[base, rise, fall]


def test_gann_one_birth_only_after_zero_birth_on_valid_leg():
    df = _ohlc_from_close(_valid_leg_close_path())
    xi = xl_mod.XunLongIndicator()
    marks = xi._calc_gann_turn_marks(df)
    zero = marks["Gann_0_birth"].to_numpy()
    one = marks["Gann_1_birth"].to_numpy()
    assert zero.sum() >= 1, "a valid up-leg must produce at least one 0出"
    assert one.sum() >= 1, "a >8% leg that rolls over must produce a 1出"
    # the first 1出 must come strictly after the first 0出
    assert np.argmax(one) > np.argmax(zero)


def test_gann_no_one_birth_when_gain_below_threshold():
    # ~3% rise: below the 8% daily min_gain -> no confirmed 1出
    close = np.r_[np.linspace(103, 100, 20), np.linspace(100, 103, 8), np.linspace(103, 99, 12)]
    df = _ohlc_from_close(close)
    xi = xl_mod.XunLongIndicator()
    marks = xi._calc_gann_turn_marks(df)
    assert marks["Gann_1_birth"].to_numpy().sum() == 0


def test_gannbox_sell_requires_prior_buy_in_segment():
    df = _ohlc_from_close(_valid_leg_close_path())
    xi = xl_mod.XunLongIndicator()
    helpers = xi._calc_ema_rsi_helpers(df)
    rsi = xi._calc_manual_rsi(df["Close"])
    g = xi._calc_gannbox_buy_sell(df, helpers=helpers, rsi_val=rsi)
    sell = g["Gann_SELL_1_confirmed"].to_numpy()
    buy = g["Gann_BUY_A"].to_numpy()
    if sell.sum() > 0:
        # every confirmed sell must have a BUY A earlier in the same series
        assert buy.sum() > 0 and np.argmax(buy) < np.argmax(sell)


# --------------------------------------------------------------------------- #
# compute() smoke + emit-source columns
# --------------------------------------------------------------------------- #
def test_compute_produces_expected_columns_and_dtypes():
    rng = np.random.default_rng(7)
    n = 220
    drift = np.cumsum(rng.normal(0.1, 1.0, n))
    close = 100 + drift + 8 * np.sin(np.linspace(0, 6 * np.pi, n))
    close = np.clip(close, 5, None)
    df = _ohlc_from_close(close)
    xi = xl_mod.XunLongIndicator()
    out = xi.compute(df, None)

    for col in ["L2_trend", "L2_pump", "RSI", "FJ_value", "Rank120",
                "Gann_BUY_A", "Gann_SELL_1_confirmed", "Gann_0", "Gann_1"]:
        assert col in out.columns, f"missing column {col}"
    # the daily emit-source booleans must be real booleans
    for col in ["Gann_BUY_A", "Gann_SELL_1_confirmed"]:
        assert out[col].dropna().map(lambda v: isinstance(v, (bool, np.bool_))).all()
    # RSI within [0,100] where defined
    rsi = out["RSI"].dropna()
    assert (rsi >= -1e-9).all() and (rsi <= 100 + 1e-9).all()


# --------------------------------------------------------------------------- #
# injectable-fetcher seam + in-run bar cache
# --------------------------------------------------------------------------- #
def test_scan_one_symbol_runs_offline_via_injected_fetcher():
    rng = np.random.default_rng(11)
    close = 100 + np.cumsum(rng.normal(0.05, 1.0, 220))
    close = np.clip(close, 5, None)
    df = _ohlc_from_close(close)

    calls = {"daily": 0, "h4": 0}

    def fake_daily(symbol, period="1y"):
        calls["daily"] += 1
        return df.copy()

    def fake_4h(symbol, period="90d"):
        calls["h4"] += 1
        return None

    xi = xl_mod.XunLongIndicator()
    res = scan.scan_one_symbol("FAKE", "Fake Co", xi,
                               daily_fetcher=fake_daily, h4_fetcher=fake_4h)
    assert isinstance(res, pd.DataFrame)          # may be empty; must not raise
    assert calls["daily"] == 1                     # used the injected fetcher, no network


def test_in_run_bar_cache_dedups_fetches():
    scan.clear_bar_cache()
    n = {"daily": 0}
    real = scan._fetch_daily_raw
    try:
        def counting(symbol, period="1y"):
            n["daily"] += 1
            return pd.DataFrame(
                {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1.0]},
                index=pd.bdate_range("2025-01-01", periods=1),
            )
        scan._fetch_daily_raw = counting
        a = scan.download_daily("ZZZ", period="1y")
        b = scan.download_daily("ZZZ", period="1y")     # served from cache
        assert n["daily"] == 1, "second identical fetch should hit the cache"
        # cache returns a copy, not the same object (mutation safety)
        assert a is not b
    finally:
        scan._fetch_daily_raw = real
        scan.clear_bar_cache()


# --------------------------------------------------------------------------- #
# tiny runner (so it works without pytest)
# --------------------------------------------------------------------------- #
def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  PASS  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"  FAIL  {t.__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
