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
from datetime import datetime
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


def test_score_raw_is_uncapped_for_tiebreak():
    # Same 107-point row as the clamp test: capped score is 100 but the raw
    # tiebreaker keeps the true 107 so two 100-capped names still sort apart.
    row = pd.Series({
        "signal_side": "BUY", "signal_type": "正式买入", "model": "D1_BUY_A_0出",
        "rank120": 0.20, "RSI": 50.0, "L2_trend": 20.0, "H4_FJ": 40.0, "H4_RSI": 50.0,
    })
    assert scan.score_buy_signal_row(row) == 100.0
    assert scan.score_buy_signal_row_raw(row) == 107.0
    # non-BUY rows stay NaN on the raw path too
    assert pd.isna(scan.score_buy_signal_row_raw(pd.Series({"signal_side": "SELL"})))


def test_score_first_observation_weight():
    # 50 +10(第一观察点) +0(model no BUY_A/0出) +3(rank 0.45-0.65) +3(RSI 35-42) = 66
    # (used to be a nameless tail of test_score_raw_is_uncapped_for_tiebreak, so a
    #  regression in the 第一观察点 weight could only surface as that test failing.)
    row = pd.Series({
        "signal_side": "BUY", "signal_type": "第一观察点", "model": "LOW_START_FIRST_GREEN",
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


def test_sell_score_non_sell_is_nan():
    assert pd.isna(scan.score_sell_signal_row(pd.Series({"signal_side": "BUY", "signal_type": "正式买入"})))


def test_sell_score_formal_clamps_to_100():
    # 50 +20(正式卖出) +8(1出) +10(rank>=0.85) +7(RSI 55-70) +5(L2>=60)
    #    +4(H4_FJ>=60) +3(H4_RSI<=55) = 107 -> clamp 100
    row = pd.Series({
        "signal_side": "SELL", "signal_type": "正式卖出", "model": "D1_SELL_1出",
        "rank120": 0.90, "RSI": 62.0, "L2_trend": 70.0, "H4_FJ": 70.0, "H4_RSI": 50.0,
    })
    assert scan.score_sell_signal_row(row) == 100.0


def test_sell_score_warning_midrange_exact():
    # 50 +14(预警卖出) +8(1出 in model) +3(rank 0.45-0.65) -6(RSI<35) = 69
    row = pd.Series({
        "signal_side": "SELL", "signal_type": "预警卖出", "model": "H4_SELL_1出",
        "rank120": 0.50, "RSI": 30.0,
    })
    assert scan.score_sell_signal_row(row) == 69.0


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


def test_intraday_store_merge_dedups_and_revises_partial():
    import intraday_store as st
    idx = pd.date_range("2026-07-08 15:00", periods=3, freq="15min", tz="America/New_York")
    old = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0,
                        "Close": [10.0, 11.0, 12.0], "Volume": 5.0}, index=idx)
    # revise the last stored bar (partial finalizing) + append one new bar
    new = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": [11.5, 99.0], "Volume": 7.0},
                       index=[idx[2], idx[2] + pd.Timedelta(minutes=15)])
    m, note = st.merge_bars(old, new)
    assert len(m) == 4 and int(m.index.duplicated().sum()) == 0
    assert m.index.is_monotonic_increasing
    assert m["Close"].iloc[2] == 11.5   # fresh fetch overrides the stored partial bar
    assert m["Close"].iloc[-1] == 99.0  # genuinely-new bar appended


def test_intraday_store_split_vs_real_move():
    import intraday_store as st
    idx = pd.date_range("2026-07-07 10:00", periods=12, freq="15min", tz="America/New_York")
    base = np.linspace(100.0, 110.0, 12)
    old = pd.DataFrame({"Open": base, "High": base, "Low": base, "Close": base, "Volume": 100.0}, index=idx)
    # uniform 0.5x rescale over the overlap == a 2:1 split yfinance back-adjusted -> rescale stored
    m_split, note_split = st.merge_bars(old, old * 0.5)
    assert "split" in note_split and abs(m_split["Close"].iloc[0] - 50.0) < 1e-6
    # a single non-uniform bar is a real move, NOT a split -> no rescale
    new_move = old.copy()
    new_move.iloc[-1, new_move.columns.get_loc("Close")] *= 1.3
    m_move, note_move = st.merge_bars(old, new_move)
    assert note_move == "appended" and abs(m_move["Close"].iloc[0] - 100.0) < 1e-6


# --------------------------------------------------------------------------- #
# session anchoring: signals are dated by the last real bar, not the wall clock
# --------------------------------------------------------------------------- #
def _empty_history_dir() -> Path:
    """A Path with no scan_result_*.xlsx in it (glob on a missing dir is empty)."""
    return BACKEND_DIR / "tests" / "_no_such_history_dir"


def test_catchup_anchors_on_session_date_not_wall_clock():
    # Ran Friday post-close; now it's Saturday. Signals are dated Friday.
    # Anchoring on the wall clock (Saturday) filtered every signal away and
    # overwrote Summary / dashboard / tv lists with "no signals".
    friday = pd.Timestamp("2026-07-24").date()
    saturday = datetime(2026, 7, 25, 9, 0)
    got = scan._get_catchup_signal_dates(_empty_history_dir(), saturday, anchor_date=friday)
    assert got == [friday], got
    # without an anchor it still degrades to the wall-clock date (fallback path)
    assert scan._get_catchup_signal_dates(_empty_history_dir(), saturday) == [saturday.date()]


def test_catchup_after_non_business_day_run_keeps_every_missed_session(tmp=None):
    # Last run was itself on a Saturday (weekend catch-up run), so bdate_range's
    # first element is Monday, not last_run_date -> the old `[1:]` slice silently
    # threw Monday's signals away.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        hist = Path(d)
        (hist / "scan_result_20260725_120000.xlsx").write_bytes(b"")   # Sat 7/25
        anchor = pd.Timestamp("2026-07-31").date()                     # Fri 7/31
        got = scan._get_catchup_signal_dates(hist, datetime(2026, 7, 31, 17, 0),
                                             max_bdays=10, anchor_date=anchor)
    assert got[0] == pd.Timestamp("2026-07-27").date(), f"lost Monday: {got}"
    assert got[-1] == anchor
    assert len(got) == 5


def test_resolve_session_state_flags_partial_and_non_trading_day():
    real = scan._fetch_daily_raw
    scan.clear_bar_cache()
    try:
        idx = pd.bdate_range("2026-07-01", "2026-07-30")   # last bar = Thu 2026-07-30
        frame = pd.DataFrame(
            {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0}, index=idx
        )
        scan._fetch_daily_raw = lambda symbol, period="1y": frame.copy()

        # mid-session on the session date itself -> partial
        mid = scan.resolve_session_state(datetime(2026, 7, 30, 11, 2))
        assert mid["session_date"] == pd.Timestamp("2026-07-30").date()
        assert mid["is_partial"] is True and mid["is_trading_day"] is True

        # after the close on the same day -> complete
        scan.clear_bar_cache()
        post = scan.resolve_session_state(datetime(2026, 7, 30, 17, 30))
        assert post["is_partial"] is False

        # weekend: wall clock is Saturday, session date is the Thursday bar
        scan.clear_bar_cache()
        wknd = scan.resolve_session_state(datetime(2026, 8, 1, 10, 0))
        assert wknd["session_date"] == pd.Timestamp("2026-07-30").date()
        assert wknd["is_trading_day"] is False and wknd["is_partial"] is False
    finally:
        scan._fetch_daily_raw = real
        scan.clear_bar_cache()


def test_resolve_session_state_falls_back_when_reference_missing():
    real = scan._fetch_daily_raw
    scan.clear_bar_cache()
    try:
        scan._fetch_daily_raw = lambda symbol, period="1y": None
        st = scan.resolve_session_state(datetime(2026, 7, 30, 11, 2))
        assert st["session_date"] is None and st["source"] == "fallback"
    finally:
        scan._fetch_daily_raw = real
        scan.clear_bar_cache()


# --------------------------------------------------------------------------- #
# trailing stop invariants
# --------------------------------------------------------------------------- #
def test_trailing_stop_invalid_when_atr_wider_than_price():
    # A violently volatile low-priced name: 5*ATR22 exceeds the share price, so
    # `peak - 5*ATR` is <= 0. A negative "stop" can never be breached, and the
    # old code reported it as a real level with 止损状态=持有 forever.
    rng = np.random.default_rng(3)
    n = 120
    close = 40 + np.cumsum(rng.normal(0, 6.0, n))     # huge daily ranges
    close = np.clip(close, 5, None)
    idx = pd.bdate_range("2026-01-01", periods=n)
    df = pd.DataFrame({
        "Open": close, "High": close * 1.35, "Low": close * 0.65,
        "Close": close, "Volume": 1e6,
    }, index=idx)
    ts = scan.compute_trailing_stop(df, base_loc=n - 20, latest_loc=n - 1)
    assert ts is not None
    raw = ts["peak"] - scan.EXIT_TRAIL_ATR_MULT * ts["atr"]
    assert raw <= 0, f"fixture no longer exercises the underflow (raw={raw:.2f})"
    assert ts["stop"] is None and ts["stop_invalid"] is True


def test_trailing_stop_valid_case_reports_positive_level():
    close = np.linspace(100, 130, 120)
    df = _ohlc_from_close(close)
    ts = scan.compute_trailing_stop(df, base_loc=100, latest_loc=119)
    assert ts is not None and ts["stop_invalid"] is False
    assert ts["stop"] > 0 and ts["stop"] < ts["peak"]


def test_trailing_stop_entry_bar_cannot_breach():
    # A gap-down entry bar must not stop itself out on day 0.
    close = np.r_[np.linspace(100, 101, 40), [70.0], np.linspace(70, 72, 10)]
    df = _ohlc_from_close(close)
    ts = scan.compute_trailing_stop(df, base_loc=40, latest_loc=40)
    assert ts is not None and ts["breached"] is False


# --------------------------------------------------------------------------- #
# lifecycle: an unclosed (partial) bar must not become the recorded entry price
# --------------------------------------------------------------------------- #
def _sig_row(**kw):
    row = {
        "symbol": "AAA", "name": "Alpha", "板块": "01 测试",
        "signal_date": pd.Timestamp("2026-06-01").date(),
        "signal_type": "正式买入", "signal_side": "BUY",
        "close": 100.0, "low": 99.0, "model": "D1_BUY_A_0出",
        "extra_info": "", "Gann_1_date": pd.NaT, "Gann_1_price": np.nan,
        "bar_partial": False,
    }
    row.update(kw)
    return row


def test_lifecycle_prefers_complete_bar_over_partial_for_same_signal():
    # Same (symbol, date, type) seen twice: once mid-session (provisional price)
    # and once after the close. The closed-bar price must win regardless of which
    # run wrote last, or a mid-session snapshot is frozen in as 买入价 forever.
    partial = _sig_row(close=548.16, bar_partial=True)
    complete = _sig_row(close=552.05, bar_partial=False)
    for order in ([partial, complete], [complete, partial]):
        out = scan._collect_lifecycle_signal_rows(_empty_history_dir(), pd.DataFrame(order))
        assert len(out) == 1
        assert out.iloc[0]["close"] == 552.05, f"partial price won for order {order is not None}"


def test_lifecycle_unknown_provenance_loses_to_known_complete():
    legacy = _sig_row(close=10.0, bar_partial=np.nan)    # old file, no flag
    complete = _sig_row(close=11.0, bar_partial=False)
    out = scan._collect_lifecycle_signal_rows(_empty_history_dir(),
                                              pd.DataFrame([complete, legacy]))
    assert len(out) == 1 and out.iloc[0]["close"] == 11.0
    # ...but a known-partial row still loses to the legacy unknown one
    partial = _sig_row(close=9.0, bar_partial=True)
    out2 = scan._collect_lifecycle_signal_rows(_empty_history_dir(),
                                               pd.DataFrame([partial, legacy]))
    assert len(out2) == 1 and out2.iloc[0]["close"] == 10.0


# --------------------------------------------------------------------------- #
# 第一观察点 tracker: the window rules must be decided inside the window
# --------------------------------------------------------------------------- #
def _tracker_df_run(symbol="AAA"):
    return pd.DataFrame([{"symbol": symbol, "name": "Alpha", "group": "01 测试"}])


def _run_tracker(rows, price_close_by_date=None, run_date=datetime(2026, 7, 30, 17, 0),
                 min_days=14, entry_low=99.0):
    """Drive the tracker offline with a stubbed price series."""
    real = scan._fetch_price_series_for_tracker
    try:
        if price_close_by_date is None:
            scan._fetch_price_series_for_tracker = lambda symbol: None
        else:
            idx = pd.DatetimeIndex([pd.Timestamp(d) for d in price_close_by_date])
            closes = np.array(list(price_close_by_date.values()), dtype=float)
            frame = pd.DataFrame(
                {"Open": closes, "High": closes, "Low": closes,
                 "Close": closes, "Volume": 1e6}, index=idx
            )
            scan._fetch_price_series_for_tracker = lambda symbol: frame.copy()
        return scan._build_first_observation_tracker(
            _empty_history_dir(), pd.DataFrame(rows), run_date,
            _tracker_df_run(), min_days=min_days,
        )
    finally:
        scan._fetch_price_series_for_tracker = real


def test_tracker_confirm_outside_window_is_a_timeout_not_a_confirmation():
    # The bug this pins: CEG's 2026-05-22 observation broke down within days, and
    # a 二进宫 that landed 49 business days later was credited as "已确认".
    obs_date = pd.Timestamp("2026-05-22").date()
    rows = [
        _sig_row(signal_type="第一观察点", signal_date=obs_date, close=100.0, low=99.0),
        _sig_row(signal_type="二进宫买入点", signal_date=pd.Timestamp("2026-07-24").date()),
    ]
    # price never breaks the 99.0 entry low, so only the window rule decides
    prices = {d: 105.0 for d in pd.bdate_range("2026-05-23", "2026-07-30")}
    out = _run_tracker(rows, prices)
    assert len(out) == 1
    assert out.iloc[0]["状态"].startswith("移除（超过"), out.iloc[0]["状态"]
    assert pd.isna(out.iloc[0]["二进宫确认日期"])


def test_tracker_confirm_inside_window_is_confirmed():
    obs_date = pd.Timestamp("2026-07-10").date()
    rows = [
        _sig_row(signal_type="第一观察点", signal_date=obs_date, close=100.0, low=99.0),
        _sig_row(signal_type="二进宫买入点", signal_date=pd.Timestamp("2026-07-16").date()),
    ]
    prices = {d: 105.0 for d in pd.bdate_range("2026-07-11", "2026-07-30")}
    out = _run_tracker(rows, prices)
    assert out.iloc[0]["状态"] == "已确认（二进宫买入点）"
    assert out.iloc[0]["二进宫确认日期"] == pd.Timestamp("2026-07-16").date()


def test_tracker_break_below_starting_low_fires_and_beats_later_confirm():
    obs_date = pd.Timestamp("2026-07-10").date()
    rows = [
        _sig_row(signal_type="第一观察点", signal_date=obs_date, close=100.0, low=99.0),
        _sig_row(signal_type="二进宫买入点", signal_date=pd.Timestamp("2026-07-24").date()),
    ]
    prices = {d: 105.0 for d in pd.bdate_range("2026-07-11", "2026-07-30")}
    prices[pd.Timestamp("2026-07-15")] = 90.0          # closes below the 99.0 low
    out = _run_tracker(rows, prices)
    assert out.iloc[0]["状态"] == "移除（跌破启动日低点）"
    assert out.iloc[0]["跌破日期"] == pd.Timestamp("2026-07-15").date()


def test_tracker_backfills_missing_low_from_prices():
    # Rows written before the `low` column existed have entry_low = NaN. The old
    # code skipped the break check entirely AND claimed "也未跌破启动日低点".
    obs_date = pd.Timestamp("2026-07-10").date()
    rows = [_sig_row(signal_type="第一观察点", signal_date=obs_date, close=100.0, low=np.nan)]
    prices = {d: 105.0 for d in pd.bdate_range("2026-07-10", "2026-07-30")}
    prices[pd.Timestamp("2026-07-10")] = 98.0          # the entry bar -> low backfills to 98.0
    prices[pd.Timestamp("2026-07-15")] = 90.0          # below it
    out = _run_tracker(rows, prices)
    assert out.iloc[0]["低点来源"] == "补算"
    assert out.iloc[0]["启动日低点"] == 98.0
    assert out.iloc[0]["状态"] == "移除（跌破启动日低点）"


def test_tracker_never_claims_no_break_when_low_is_unknown():
    obs_date = pd.Timestamp("2026-05-22").date()
    rows = [_sig_row(signal_type="第一观察点", signal_date=obs_date, close=100.0, low=np.nan)]
    out = _run_tracker(rows, price_close_by_date=None)   # no price series at all
    row = out.iloc[0]
    assert row["低点来源"] == "缺失"
    assert "无法判定" in row["移除原因"]
    assert "未跌破" not in row["移除原因"]


def test_tracker_still_watching_inside_window():
    obs_date = pd.Timestamp("2026-07-28").date()
    rows = [_sig_row(signal_type="第一观察点", signal_date=obs_date, close=100.0, low=99.0)]
    prices = {d: 105.0 for d in pd.bdate_range("2026-07-28", "2026-07-30")}
    out = _run_tracker(rows, prices)
    assert out.iloc[0]["状态"] == "观察中"


def test_tracker_column_set_is_stable_across_branches():
    # The column set used to depend on which branch fired first (二进宫确认日期 /
    # 跌破日期 are branch-only keys), so the sheet's shape moved run to run.
    obs = pd.Timestamp("2026-07-10").date()
    watching = [_sig_row(signal_type="第一观察点", signal_date=pd.Timestamp("2026-07-29").date())]
    confirmed = [
        _sig_row(signal_type="第一观察点", signal_date=obs),
        _sig_row(signal_type="二进宫买入点", signal_date=pd.Timestamp("2026-07-16").date()),
    ]
    prices = {d: 105.0 for d in pd.bdate_range("2026-07-10", "2026-07-30")}
    a = _run_tracker(watching, prices)
    b = _run_tracker(confirmed, prices)
    assert list(a.columns) == list(b.columns) == scan.FIRST_OBS_TRACKER_COLS
    empty = _run_tracker([_sig_row(signal_type="正式买入")], prices)
    assert list(empty.columns) == scan.FIRST_OBS_TRACKER_COLS and empty.empty


def test_persisted_signal_schema_includes_tracker_inputs():
    # RawSignals in history/ is the system's long-term memory; the lifecycle and
    # 第一观察点 trackers read these back out of old workbooks. A column that is
    # computed but not listed here is silently dropped on the way to disk.
    for col in ["low", "bar_partial", "Gann_0", "Gann_gain_pct", "buy_score_raw",
                "close", "signal_date", "signal_type"]:
        assert col in scan.SIGNAL_COL_ORDER, f"{col} would not be persisted"
    assert len(set(scan.SIGNAL_COL_ORDER)) == len(scan.SIGNAL_COL_ORDER), "duplicate column"


# --------------------------------------------------------------------------- #
# 回调买入点 / pullback entry — the one rule that survived the Elliott study
# --------------------------------------------------------------------------- #
def _uptrend_with_pullback(retrace: float, n_base: int = 240) -> pd.DataFrame:
    """
    Build the exact geometry the rule looks for: prev swing LOW -> swing HIGH -> pullback LOW,
    with a final bounce so that last low gets CONFIRMED.

    The initial dip matters: ZigZag only records a swing low once price has risen off it, so a
    smooth monotonic base yields no low pivot at all and the rule can never see three pivots.
    """
    base = np.linspace(100.0, 150.0, n_base)      # establishes the 200d MA below price
    dip = np.linspace(150.0, 140.0, 10)           # -6.7%: records the H(150), sets up L(140)
    up = np.linspace(140.0, 200.0, 30)            # the up-leg 140 -> 200, confirms L(140)
    low = 200.0 - retrace * (200.0 - 140.0)       # pullback retracing `retrace` of that leg
    down = np.linspace(200.0, low, 20)
    bounce = np.linspace(low, low * 1.10, 15)     # +10% confirms the pullback low
    return _ohlc_from_close(np.r_[base, dip, up, down, bounce])


def test_pullback_fires_inside_the_fib_zone():
    df = _uptrend_with_pullback(0.60)             # 60% retrace: inside 38.2-78.6%
    out = scan.add_pullback_entry(df)
    assert out["PULLBACK_BUY"].any(), "a 60% retracement in an uptrend should fire"
    hit = out.index[out["PULLBACK_BUY"]][0]
    r = float(out.loc[hit, "PULLBACK_RETRACE"])
    assert 0.382 <= r <= 0.786, f"recorded retracement {r} outside the zone"


def test_pullback_silent_outside_the_fib_zone():
    # 20% is too shallow, 95% breaks the structure — neither is the target geometry.
    # The 60% case is asserted to fire elsewhere, so these are not vacuous passes.
    for retrace in (0.20, 0.95):
        out = scan.add_pullback_entry(_uptrend_with_pullback(retrace))
        assert not out["PULLBACK_BUY"].any(), f"{retrace:.0%} retracement should not fire"


def test_pullback_requires_an_uptrend():
    # identical geometry, but riding a long decline so price sits below the 200d MA
    close = np.r_[np.linspace(300.0, 100.0, 240), np.linspace(100.0, 93.0, 10),
                  np.linspace(93.0, 133.0, 30), np.linspace(133.0, 109.0, 20),
                  np.linspace(109.0, 120.0, 15)]
    df = _ohlc_from_close(close)
    # sanity: the fixture must be below the MA, else the test passes for the wrong reason
    ma = df["Close"].rolling(scan.PULLBACK_MA_LEN).mean()
    assert (df["Close"].iloc[-40:] < ma.iloc[-40:]).all(), "fixture is not below the MA"
    out = scan.add_pullback_entry(df)
    assert not out["PULLBACK_BUY"].any(), "must not fire below the trend MA"


def test_pullback_fires_at_confirmation_not_at_the_pivot():
    """THE lookahead test. A swing low is only knowable after price rises off it by the
    threshold; stamping the signal at the pivot bar overstates forward returns 1.7x-9.8x."""
    df = _uptrend_with_pullback(0.60)
    out = scan.add_pullback_entry(df)
    hit = out.index[out["PULLBACK_BUY"]][0]
    hit_pos = out.index.get_loc(hit)
    lo_pos = int(np.argmin(out["Close"].to_numpy()[:hit_pos + 1][-40:])) + max(0, hit_pos - 39)
    assert hit_pos > lo_pos, "signal must be dated AFTER the swing low, not on it"
    # and price must already have risen at least the threshold off that low
    rise = out["Close"].iloc[hit_pos] / out["Close"].iloc[lo_pos] - 1
    assert rise >= scan.PULLBACK_ZIGZAG_THR - 1e-9, (
        f"only rose {rise:.3%} off the low; confirmation needs "
        f"{scan.PULLBACK_ZIGZAG_THR:.1%}")


def test_pullback_signal_is_causal_on_a_growing_prefix():
    """Re-running on data[:t] must reproduce a prefix of the full-history signals. If a
    later bar can change an earlier signal, the detector repaints and the backtest is void."""
    df = _uptrend_with_pullback(0.60)
    full = scan.add_pullback_entry(df)["PULLBACK_BUY"]
    for cut in (len(df) - 1, len(df) - 5, len(df) - 12):
        part = scan.add_pullback_entry(df.iloc[:cut])["PULLBACK_BUY"]
        assert (part.to_numpy() == full.iloc[:cut].to_numpy()).all(), (
            f"signals changed when data was truncated at {cut} — the detector repaints")


def test_pullback_scores_below_every_real_buy_type():
    base = {"signal_side": "BUY", "model": "PULLBACK_FIB_CONFIRM",
            "rank120": 0.50, "RSI": 50.0}
    pb = scan.score_buy_signal_row(pd.Series({**base, "signal_type": "回调买入点"}))
    for stronger in ("第一观察点", "预警买入", "二进宫买入点", "正式买入"):
        s = scan.score_buy_signal_row(pd.Series({**base, "signal_type": stronger}))
        assert pb < s, f"回调买入点 ({pb}) must score below {stronger} ({s})"


def test_pullback_alone_does_not_create_a_tracked_position():
    """It's entry timing, not a buy. On its own it must not open a D0 batch or reach the
    TV buy list; alongside a real buy on the same symbol+date it may enrich the rule text."""
    d = pd.Timestamp("2026-07-01").date()
    only_pb = pd.DataFrame([_sig_row(signal_type="回调买入点", signal_date=d, close=50.0)])
    out = scan._extract_anchor_signals(_empty_history_dir(), only_pb, signal_side="BUY")
    assert out.empty, "a lone 回调买入点 must not become a tracked anchor"

    with_real = pd.DataFrame([
        _sig_row(signal_type="回调买入点", signal_date=d, close=50.0),
        _sig_row(signal_type="正式买入", signal_date=d, close=50.0),
    ])
    out2 = scan._extract_anchor_signals(_empty_history_dir(), with_real, signal_side="BUY")
    assert len(out2) == 1, "a real buy on the same date should still anchor"
    assert "回调买入点" in str(out2.iloc[0]["d0_rule"])


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
