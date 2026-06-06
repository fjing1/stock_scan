#!/usr/bin/env python3
"""Stock OneClick test runner: previous close vs 12:00 PT intraday snapshot."""

from __future__ import annotations

import os
import subprocess
from contextlib import contextmanager
from datetime import datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import scan_stocks as scan


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = BASE_DIR / "exports"
HISTORY_DIR = BASE_DIR / "history"
NOON_PT = time(12, 0)
COL_ORDER = [
    "run_date", "run_time", "数据状态", "symbol", "name", "板块",
    "signal_date", "signal_type", "signal_side", "model",
    "close", "volume", "vol_ma20", "L2_trend", "L2_pump", "RSI",
    "rank120", "H4_RSI", "H4_FJ", "H4_0_birth", "H4_1_birth",
    "Gann_1_date", "Gann_1_price", "buy_score", "extra_info",
]


def _normalize_index_to_pt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = pd.to_datetime(out.index)
    if idx.tz is None:
        idx = idx.tz_localize("America/New_York")
    out.index = idx.tz_convert("America/Los_Angeles")
    return out


def _download_daily_raw(symbol: str) -> pd.DataFrame | None:
    df = yf.download(scan.to_yfinance_symbol(symbol), period="1y", interval="1d", auto_adjust=False, progress=False)
    df = scan.normalize_yf_df(df)
    if df.empty:
        return None
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def _download_intraday_raw(symbol: str, interval: str) -> pd.DataFrame | None:
    df = yf.download(scan.to_yfinance_symbol(symbol), period="5d", interval=interval, auto_adjust=False, progress=False, prepost=False)
    df = scan.normalize_yf_df(df)
    if df.empty:
        return None
    return _normalize_index_to_pt(df[["Open", "High", "Low", "Close", "Volume"]].dropna())


def _previous_business_day(run_dt: datetime):
    return pd.bdate_range(end=run_dt.date(), periods=2).date[0]


def _close_daily_through(df_daily: pd.DataFrame, test_date) -> pd.DataFrame:
    out = df_daily.copy()
    out.index = pd.to_datetime(out.index)
    return out[out.index.date <= test_date]


def _previous_close_4h(df_4h: pd.DataFrame | None, close_date) -> pd.DataFrame | None:
    if df_4h is None or df_4h.empty:
        return df_4h
    out = _normalize_index_to_pt(df_4h)
    return out[out.index.date <= close_date]


def _noon_daily(df_daily: pd.DataFrame, df_5m: pd.DataFrame | None, test_date) -> pd.DataFrame:
    base = _close_daily_through(df_daily, test_date)
    base = base[pd.to_datetime(base.index).date < test_date]
    if df_5m is None or df_5m.empty:
        return base
    today_rows = df_5m[(df_5m.index.date == test_date) & (df_5m.index.time <= NOON_PT)]
    if today_rows.empty:
        return base
    noon_bar = pd.DataFrame(
        {
            "Open": [float(today_rows["Open"].iloc[0])],
            "High": [float(today_rows["High"].max())],
            "Low": [float(today_rows["Low"].min())],
            "Close": [float(today_rows["Close"].iloc[-1])],
            "Volume": [float(today_rows["Volume"].sum())],
        },
        index=[pd.Timestamp(test_date)],
    )
    return pd.concat([base, noon_bar]).sort_index()


def _noon_4h(df_4h: pd.DataFrame | None, test_date) -> pd.DataFrame | None:
    if df_4h is None or df_4h.empty:
        return df_4h
    out = _normalize_index_to_pt(df_4h)
    cutoff = pd.Timestamp.combine(test_date, NOON_PT).tz_localize("America/Los_Angeles")
    return out[out.index <= cutoff]


@contextmanager
def _patched_downloads(daily_map: dict[str, pd.DataFrame], h4_map: dict[str, pd.DataFrame | None]):
    old_daily = scan.download_daily
    old_4h = scan.download_4h

    def patched_daily(symbol, period="1y"):
        df = daily_map.get(str(symbol).strip().upper())
        return None if df is None else df.copy()

    def patched_4h(symbol, period="90d"):
        df = h4_map.get(str(symbol).strip().upper())
        return None if df is None else df.copy()

    scan.download_daily = patched_daily
    scan.download_4h = patched_4h
    try:
        yield
    finally:
        scan.download_daily = old_daily
        scan.download_4h = old_4h


def _scan_snapshot(df_run: pd.DataFrame, daily_map: dict[str, pd.DataFrame], h4_map: dict[str, pd.DataFrame | None], run_dt: datetime, test_date, label: str) -> pd.DataFrame:
    rows = []
    xl = scan.XunLongIndicator()
    total = len(df_run)
    with _patched_downloads(daily_map, h4_map):
        for i, (_, r) in enumerate(df_run.iterrows(), start=1):
            sym = str(r["symbol"]).strip().upper()
            print(f"[{label} {i}/{total}] 扫描 {sym} ({r.get('name', '')})", flush=True)
            try:
                df_sig = scan.scan_one_symbol(sym, r.get("name", ""), xl)
            except Exception as exc:
                print(f"[{label} {sym}] 扫描出错：{exc}", flush=True)
                continue
            if not df_sig.empty:
                rows.append(df_sig)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=COL_ORDER)
    if not out.empty:
        profile_map = df_run[["symbol", "group"]].drop_duplicates(subset=["symbol"], keep="first").copy()
        profile_map["板块"] = profile_map["group"].apply(scan._normalize_sector_with_code)
        out = out.merge(profile_map[["symbol", "板块"]], how="left", on="symbol")
    out["signal_date"] = pd.to_datetime(out.get("signal_date"), errors="coerce").dt.date
    out = out[(out["signal_date"] >= scan.LIFECYCLE_START_DATE) & (out["signal_date"] == test_date)].reset_index(drop=True)
    out["run_date"] = run_dt.date()
    out["run_time"] = run_dt.strftime("%H:%M:%S")
    out["数据状态"] = label
    if "板块" in out.columns:
        out["板块"] = out["板块"].apply(scan._normalize_sector_with_code)
    out["buy_score"] = out.apply(scan.score_buy_signal_row, axis=1) if not out.empty else np.nan
    for col in COL_ORDER:
        if col not in out.columns:
            out[col] = np.nan
    return out[COL_ORDER]


def _anchors_from_signals(df_signals: pd.DataFrame, signal_side: str) -> pd.DataFrame:
    side = str(signal_side or "").upper()
    if df_signals is None or df_signals.empty:
        return pd.DataFrame(columns=["symbol", "signal_date", "d0_close", "d0_rule"])
    out = df_signals.copy()
    out = out[out.get("signal_side", "").astype(str).str.upper() == side]
    if side == "SELL":
        out = out[out.get("signal_type", "").astype(str).eq("正式卖出")]
    if out.empty:
        return pd.DataFrame(columns=["symbol", "signal_date", "d0_close", "d0_rule"])
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce").dt.date
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["signal_type"] = out.get("signal_type", "").fillna("").astype(str)
    out = out.dropna(subset=["symbol", "signal_date", "close"])
    if out.empty:
        return pd.DataFrame(columns=["symbol", "signal_date", "d0_close", "d0_rule"])
    if "buy_score" not in out.columns:
        out["buy_score"] = np.nan
    out["buy_score"] = pd.to_numeric(out["buy_score"], errors="coerce")
    return (
        out.sort_values(["signal_date", "symbol", "signal_type"])
        .groupby(["symbol", "signal_date"], as_index=False)
        .agg(
            d0_close=("close", "first"),
            d0_rule=("signal_type", lambda s: " | ".join([x for x in pd.unique(s) if x])),
            buy_score=("buy_score", "max"),
        )
        .reset_index(drop=True)
    )


def _combine_followup_dict(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for key in sorted(sheets.keys()):
        df = sheets[key]
        if df is not None and not df.empty:
            frames.append(df.copy())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _build_followup_pair(
    df_signals: pd.DataFrame,
    daily_map: dict[str, pd.DataFrame],
    sector_map: dict[str, str],
    run_dt: datetime,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    buy_anchors = _anchors_from_signals(df_signals, "BUY")
    sell_anchors = _anchors_from_signals(df_signals, "SELL")
    with _patched_downloads(daily_map, {}):
        buy_sheets, buy_completed = scan._build_followup_sheets(
            buy_anchors,
            run_dt,
            max_days=scan.TRACK_MAX_DAYS,
            sector_map=sector_map,
        )
        sell_sheets, sell_completed = scan._build_followup_sheets(
            sell_anchors,
            run_dt,
            max_days=scan.TRACK_MAX_DAYS,
            sector_map=sector_map,
            sheet_prefix="SELL_",
        )
    buy_df = _combine_followup_dict({**buy_sheets, **buy_completed})
    sell_df = _combine_followup_dict({**sell_sheets, **sell_completed})
    return buy_df, sell_df


def _load_historical_buy_signal_dates() -> dict[str, set]:
    out: dict[str, set] = {}
    for path in sorted(HISTORY_DIR.glob("scan_result_*.xlsx")):
        try:
            rows = scan._read_signal_rows_from_result(path, "BUY")
        except Exception:
            continue
        if rows is None or rows.empty or not {"symbol", "signal_date"}.issubset(rows.columns):
            continue
        rows = rows.copy()
        rows["symbol"] = rows["symbol"].astype(str).str.strip().str.upper()
        rows["signal_date"] = pd.to_datetime(rows["signal_date"], errors="coerce").dt.date
        rows = rows.dropna(subset=["symbol", "signal_date"])
        for sym, dates in rows.groupby("symbol")["signal_date"]:
            out.setdefault(sym, set()).update(dates.tolist())
    return out


def _mark_first_seen_14d_for_stats(df: pd.DataFrame, historical_dates: dict[str, set]) -> pd.DataFrame:
    if df is None or df.empty or "symbol" not in df.columns or "D0_date" not in df.columns:
        return df
    out = df.copy()
    flags = []
    for _, row in out.iterrows():
        sym = str(row.get("symbol", "") or "").strip().upper()
        d0 = pd.to_datetime(row.get("D0_date"), errors="coerce")
        if not sym or pd.isna(d0):
            flags.append(False)
            continue
        d0_date = d0.date()
        has_prior = False
        for prior_date in historical_dates.get(sym, set()):
            if prior_date >= d0_date:
                continue
            days = scan._business_days_between(prior_date, d0_date)
            if 0 < days <= scan.TRACK_MAX_DAYS:
                has_prior = True
                break
        flags.append(not has_prior)
    out["_first_seen_14d"] = flags
    return out


def main():
    run_dt = datetime.now()
    test_date = _previous_business_day(run_dt)
    _, df_meta = scan.load_input_and_meta(scan.INPUT_FILE)
    df_enabled = df_meta[df_meta["enable"] == 1].copy()
    df_run, df_excluded = scan.filter_scannable_universe(df_enabled)
    if not df_excluded.empty:
        skipped_groups = ", ".join(
            df_excluded["group"].fillna("").astype(str).replace("", "未分组").drop_duplicates().tolist()
        )
        print(f"跳过不扫描分组：{skipped_groups}（{len(df_excluded)} 只）", flush=True)
    if df_run.empty:
        print("过滤市场环境/顶部系统分组后，没有可测试标的。")
        return

    daily_close_map, h4_close_map = {}, {}
    daily_noon_map, h4_noon_map = {}, {}
    print(f"测试规则：只测试前一个交易日 {test_date} 的收盘信号 + {test_date} 12:00 PT 盘中信号", flush=True)
    for i, (_, r) in enumerate(df_run.iterrows(), start=1):
        sym = str(r["symbol"]).strip().upper()
        print(f"[DATA {i}/{len(df_run)}] 下载 {sym}", flush=True)
        daily_raw = _download_daily_raw(sym)
        if daily_raw is None or daily_raw.empty:
            continue
        h4_raw = _download_intraday_raw(sym, "4h")
        m5_raw = _download_intraday_raw(sym, "5m")
        close_daily = _close_daily_through(daily_raw, test_date)
        if close_daily.empty:
            continue
        close_date = close_daily.index[-1].date()
        if close_date != test_date:
            continue
        daily_close_map[sym] = close_daily
        h4_close_map[sym] = _previous_close_4h(h4_raw, close_date)
        daily_noon_map[sym] = _noon_daily(daily_raw, m5_raw, test_date)
        h4_noon_map[sym] = _noon_4h(h4_raw, test_date)

    close_signals = _scan_snapshot(df_run, daily_close_map, h4_close_map, run_dt, test_date, f"{test_date} 收盘价")
    noon_signals = _scan_snapshot(df_run, daily_noon_map, h4_noon_map, run_dt, test_date, f"{test_date} 12点盘中")
    sector_map = (
        df_run[["symbol", "group"]]
        .drop_duplicates(subset=["symbol"], keep="first")
        .assign(板块=lambda x: x["group"].apply(scan._normalize_sector_with_code))
        .set_index("symbol")["板块"]
        .to_dict()
    )
    close_buy_df, close_sell_df = _build_followup_pair(close_signals, daily_close_map, sector_map, run_dt)
    noon_buy_df, noon_sell_df = _build_followup_pair(noon_signals, daily_noon_map, sector_map, run_dt)
    historical_buy_dates = _load_historical_buy_signal_dates()
    close_buy_df = _mark_first_seen_14d_for_stats(close_buy_df, historical_buy_dates)
    noon_buy_df = _mark_first_seen_14d_for_stats(noon_buy_df, historical_buy_dates)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    ts = run_dt.strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / "stock_oneclick_test_latest.xlsx"
    hist_path = HISTORY_DIR / f"stock_oneclick_test_{ts}.xlsx"
    with pd.ExcelWriter(hist_path, engine="openpyxl") as writer:
        close_title = f"{test_date} 收盘价"
        noon_title = f"{test_date} 12点盘中"
        scan._write_combined_followup_sheet(writer, close_title, close_buy_df, close_sell_df)
        scan._write_combined_followup_sheet(writer, noon_title, noon_buy_df, noon_sell_df)
    out_path.write_bytes(hist_path.read_bytes())
    print(f"✅ 测试结果已生成：{out_path}")
    print(f"✅ 测试历史已保存：{hist_path}")
    if os.environ.get("STOCK_ONECLICK_NO_OPEN", "").strip() != "1":
        subprocess.run(["open", str(out_path)], check=False)


if __name__ == "__main__":
    main()
