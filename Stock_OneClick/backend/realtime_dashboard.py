#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
import pandas as pd

import scan_stocks as scan
from dashboard_data import DashboardDataProvider
from xunlong import XunLongIndicator


REFRESH_SECONDS = int(os.getenv("STOCK_DASHBOARD_REFRESH_SECONDS", "300"))
DEFAULT_LIMIT = int(os.getenv("STOCK_DASHBOARD_LIMIT", "0"))
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "stock_input_template.xlsx"


def score_signal_row(row: pd.Series) -> float:
    # Delegates to the engine's canonical scorer. This used to be a
    # byte-for-byte copy of scan.score_buy_signal_row; keeping one source of
    # truth avoids the two drifting apart when the scoring weights change.
    return scan.score_buy_signal_row(row)


def load_universe(limit: int = 0) -> pd.DataFrame:
    df_input, df_meta = scan.load_input_and_meta(INPUT_FILE)
    if df_meta.empty:
        raise RuntimeError("Sheet2_Classified 为空，请先同步/导入 TradingView 清单。")
    if "enable" not in df_meta.columns:
        df_meta["enable"] = 1
    df_enabled = df_meta[pd.to_numeric(df_meta["enable"], errors="coerce").fillna(1).astype(int) == 1].copy()
    df_run, _ = scan.filter_scannable_universe(df_enabled)
    df_run = df_run.drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)
    if limit and limit > 0:
        df_run = df_run.head(limit).copy()
    return df_run


def format_number(value, digits: int = 2) -> str:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return ""
    return f"{float(num):.{digits}f}"


def empty_result_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "观海买点分", "symbol", "板块", "signal_date", "signal_type",
            "signal_side", "close", "RSI", "H4_RSI", "H4_FJ", "extra_info", "updated_at",
        ]
    )


def scan_universe(
    limit: int = 0,
    progress=None,
    today_only: bool = True,
    provider: DashboardDataProvider | None = None,
) -> pd.DataFrame:
    df_run = load_universe(limit)
    xl = XunLongIndicator()
    provider = provider or DashboardDataProvider()
    rows = []
    total = len(df_run)
    original_download_daily = scan.download_daily
    original_download_4h = scan.download_4h
    scan.download_daily = provider.download_daily
    scan.download_4h = provider.download_4h
    try:
        for idx, (_, meta) in enumerate(df_run.iterrows(), start=1):
            symbol = str(meta.get("symbol", "")).strip().upper()
            name = str(meta.get("name", "") or "")
            group = scan._normalize_sector_with_code(meta.get("group", ""))
            if progress:
                progress(idx, total, symbol)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    sig = scan.scan_one_symbol(symbol, name, xl)
            except Exception as exc:
                rows.append(
                    {
                        "symbol": symbol,
                        "板块": group,
                        "signal_type": "ERROR",
                        "signal_side": "",
                        "观海买点分": np.nan,
                        "close": np.nan,
                        "RSI": np.nan,
                        "H4_RSI": np.nan,
                        "extra_info": str(exc),
                        "updated_at": datetime.now().strftime("%H:%M:%S"),
                    }
                )
                continue
            if sig.empty:
                continue
            sig["板块"] = group
            sig["观海买点分"] = sig.apply(score_signal_row, axis=1)
            sig["updated_at"] = datetime.now().strftime("%H:%M:%S")
            rows.extend(sig.to_dict("records"))
    finally:
        scan.download_daily = original_download_daily
        scan.download_4h = original_download_4h

    if not rows:
        return empty_result_frame()

    out = pd.DataFrame(rows)
    if today_only and "signal_date" in out.columns:
        today = datetime.now().date()
        signal_dates = pd.to_datetime(out["signal_date"], errors="coerce").dt.date
        error_mask = out.get("signal_type", pd.Series("", index=out.index)).astype(str).eq("ERROR")
        out = out[(signal_dates == today) | error_mask].copy()
        if out.empty:
            return empty_result_frame()
    out["signal_side"] = out.get("signal_side", "").fillna("").astype(str).str.upper()
    out = out[(out["signal_side"] == "BUY") | out.get("signal_type", pd.Series("", index=out.index)).astype(str).eq("ERROR")].copy()
    if out.empty:
        return empty_result_frame()
    for col in ["观海买点分", "close", "RSI", "H4_RSI", "H4_FJ"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["_side_sort"] = out["signal_side"].map({"BUY": 0, "SELL": 1}).fillna(2)
    out["_score_sort"] = out["观海买点分"].fillna(-1)
    out = out.sort_values(["_side_sort", "_score_sort", "symbol"], ascending=[True, False, True])
    return out.drop(columns=["_side_sort", "_score_sort"], errors="ignore").reset_index(drop=True)


class RealtimeDashboard:
    def __init__(self, root: tk.Tk, limit: int = 0, today_only: bool = True):
        self.root = root
        self.limit = limit
        self.today_only = today_only
        self.provider = DashboardDataProvider()
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.running = False
        self.last_started = None
        self.next_after_id = None

        self.root.title("Stock OneClick 5分钟买点 Dashboard")
        self.root.geometry("1160x720")
        self.root.minsize(980, 600)
        self._build_ui()
        self.root.after(200, self._poll_events)
        self.refresh()

    def _build_ui(self):
        self.root.configure(bg="#f5f7fb")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Helvetica", 17, "bold"), foreground="#1f2a44", background="#f5f7fb")
        style.configure("Sub.TLabel", font=("Helvetica", 10), foreground="#4b5563", background="#f5f7fb")
        style.configure("Main.TButton", font=("Helvetica", 10, "bold"), padding=7)
        style.configure("Treeview", rowheight=25, font=("Helvetica", 10))
        style.configure("Treeview.Heading", font=("Helvetica", 10, "bold"))

        top = tk.Frame(self.root, bg="#f5f7fb")
        top.pack(fill="x", padx=16, pady=(14, 8))
        ttk.Label(top, text="Stock OneClick 5分钟买点 Dashboard", style="Title.TLabel").pack(side="left")
        self.status_var = tk.StringVar(value="准备中")
        ttk.Label(top, textvariable=self.status_var, style="Sub.TLabel").pack(side="right")

        controls = tk.Frame(self.root, bg="#f5f7fb")
        controls.pack(fill="x", padx=16, pady=(0, 8))
        self.refresh_btn = ttk.Button(controls, text="立即刷新", style="Main.TButton", command=self.refresh)
        self.refresh_btn.pack(side="left")
        ttk.Button(controls, text="打开输入表", command=lambda: os.system(f"open '{INPUT_FILE}'")).pack(side="left", padx=(8, 0))
        ttk.Button(controls, text="打开项目文件夹", command=lambda: os.system(f"open '{BASE_DIR}'")).pack(side="left", padx=(8, 0))
        scope = "今天触发" if self.today_only else "最近窗口"
        ttk.Label(
            controls,
            text=f"只看BUY | 数据源：{self.provider.label} | 自动刷新：{REFRESH_SECONDS // 60}分钟 | 范围：{scope} | 正式买入仍以收盘确认为准",
            style="Sub.TLabel",
        ).pack(side="left", padx=(16, 0))

        cols = ["score", "symbol", "side", "rule", "group", "date", "close", "rsi", "h4rsi", "h4fj", "updated", "info"]
        self.tree = ttk.Treeview(self.root, columns=cols, show="headings")
        headings = {
            "score": "观海分",
            "symbol": "Symbol",
            "side": "方向",
            "rule": "规则",
            "group": "板块",
            "date": "触发日",
            "close": "价格",
            "rsi": "RSI",
            "h4rsi": "4H RSI",
            "h4fj": "4H分金",
            "updated": "刷新",
            "info": "备注",
        }
        widths = {
            "score": 70,
            "symbol": 80,
            "side": 60,
            "rule": 100,
            "group": 150,
            "date": 90,
            "close": 80,
            "rsi": 70,
            "h4rsi": 75,
            "h4fj": 75,
            "updated": 70,
            "info": 320,
        }
        for col in cols:
            self.tree.heading(col, text=headings[col])
            self.tree.column(col, width=widths[col], minwidth=50, stretch=(col == "info"))

        ybar = ttk.Scrollbar(self.root, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=ybar.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(16, 0), pady=(0, 14))
        ybar.pack(side="right", fill="y", padx=(0, 16), pady=(0, 14))
        self.tree.tag_configure("buy", background="#e7f5ec")
        self.tree.tag_configure("sell", background="#fdecec")
        self.tree.tag_configure("error", background="#fff4ce")

    def refresh(self):
        if self.running:
            return
        self.running = True
        self.last_started = datetime.now()
        self.refresh_btn.configure(state="disabled")
        self.status_var.set("正在刷新...")
        if self.next_after_id is not None:
            self.root.after_cancel(self.next_after_id)
            self.next_after_id = None
        thread = threading.Thread(target=self._worker, daemon=True)
        thread.start()

    def _worker(self):
        def progress(current, total, symbol):
            self.events.put(("progress", (current, total, symbol)))

        try:
            df = scan_universe(self.limit, progress=progress, today_only=self.today_only, provider=self.provider)
            self.events.put(("result", df))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def _poll_events(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "progress":
                    current, total, symbol = payload
                    self.status_var.set(f"扫描中 {current}/{total}: {symbol}")
                elif kind == "result":
                    self._show_result(payload)
                elif kind == "error":
                    self._finish()
                    messagebox.showerror("刷新失败", str(payload))
        except queue.Empty:
            pass
        self.root.after(200, self._poll_events)

    def _show_result(self, df: pd.DataFrame):
        for item in self.tree.get_children():
            self.tree.delete(item)

        for _, r in df.iterrows():
            side = str(r.get("signal_side", "")).upper()
            tag = "buy" if side == "BUY" else "sell" if side == "SELL" else "error"
            self.tree.insert(
                "",
                "end",
                values=[
                    format_number(r.get("观海买点分"), 1),
                    str(r.get("symbol", "")),
                    side,
                    str(r.get("signal_type", "")),
                    str(r.get("板块", "")),
                    str(r.get("signal_date", "")),
                    format_number(r.get("close"), 2),
                    format_number(r.get("RSI"), 1),
                    format_number(r.get("H4_RSI"), 1),
                    format_number(r.get("H4_FJ"), 1),
                    str(r.get("updated_at", "")),
                    str(r.get("extra_info", ""))[:240],
                ],
                tags=(tag,),
            )

        buy_count = int((df.get("signal_side", pd.Series(dtype=str)).astype(str).str.upper() == "BUY").sum()) if not df.empty else 0
        elapsed = time.time() - self.last_started.timestamp() if self.last_started else 0
        self._finish()
        self.status_var.set(
            f"完成 {datetime.now().strftime('%H:%M:%S')} | BUY {buy_count} | 用时 {elapsed:.0f}s"
        )

    def _finish(self):
        self.running = False
        self.refresh_btn.configure(state="normal")
        self.next_after_id = self.root.after(REFRESH_SECONDS * 1000, self.refresh)


def main():
    parser = argparse.ArgumentParser(description="Stock OneClick 5分钟实时dashboard原型")
    parser.add_argument("--once", action="store_true", help="只跑一次并在终端输出结果")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="只扫描前 N 只，0 表示全量")
    parser.add_argument("--all-recent", action="store_true", help="显示策略最近窗口里的所有信号，而不只显示今天")
    args = parser.parse_args()
    today_only = not args.all_recent

    if args.once:
        started = time.time()
        provider = DashboardDataProvider()
        df = scan_universe(
            args.limit,
            progress=lambda i, total, sym: print(f"[{i}/{total}] {sym}", flush=True),
            today_only=today_only,
            provider=provider,
        )
        cols = ["观海买点分", "symbol", "板块", "signal_date", "signal_type", "signal_side", "close", "RSI", "H4_RSI", "extra_info"]
        print(df[cols].to_string(index=False) if not df.empty else "没有触发信号")
        print(f"elapsed_seconds={time.time() - started:.1f}")
        return

    root = tk.Tk()
    RealtimeDashboard(root, limit=args.limit, today_only=today_only)
    root.mainloop()


if __name__ == "__main__":
    main()
