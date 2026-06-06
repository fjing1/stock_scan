#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import os
import threading
import time
import tkinter as tk
from datetime import date, datetime
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk

import numpy as np
import pandas as pd

os.environ.setdefault("STOCK_DASHBOARD_DATA_PROVIDER", "alpaca")

import scan_stocks as scan
from dashboard_data import DashboardDataProvider
from watchlist_importer import sync_watchlist_url
from xunlong import XunLongIndicator


REFRESH_SECONDS = int(os.getenv("STOCK_DASHBOARD_REFRESH_SECONDS", "300"))


def _symbol_ok(symbol: str) -> bool:
    text = str(symbol or "").strip().upper()
    return bool(text) and len(text) <= 16 and all(ch.isalnum() or ch in ".!:-" for ch in text)


def _load_profile(symbol: str) -> dict:
    symbol = str(symbol or "").strip().upper()
    profile = {"symbol": symbol, "name": "", "sector": "", "excluded": False}
    try:
        _, df_meta = scan.load_input_and_meta(scan.INPUT_FILE)
    except Exception:
        return profile
    if df_meta.empty or "symbol" not in df_meta.columns:
        return profile
    row = df_meta[df_meta["symbol"].astype(str).str.strip().str.upper().eq(symbol)]
    if row.empty:
        return profile
    r = row.iloc[0]
    group = str(r.get("group", "") or "")
    profile.update(
        {
            "name": str(r.get("name", "") or ""),
            "sector": scan._normalize_sector_with_code(group),
            "excluded": scan.is_excluded_from_scan_group(group),
        }
    )
    return profile


def _num(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _scan_symbol_latest_daily_4h(symbol: str, name: str, provider: DashboardDataProvider) -> pd.DataFrame:
    return _scan_symbol_from_frames(
        symbol,
        name,
        provider.download_daily(symbol, period="1y"),
        provider.download_4h(symbol, period="90d"),
    )


def _scan_symbol_from_frames(
    symbol: str,
    name: str,
    daily_df: pd.DataFrame | None,
    h4_df: pd.DataFrame | None,
) -> pd.DataFrame:
    old_daily = scan.download_daily
    old_4h = scan.download_4h

    def patched_daily(_symbol, period="1y"):
        return None if daily_df is None else daily_df.copy()

    def patched_4h(_symbol, period="90d"):
        return None if h4_df is None else h4_df.copy()

    scan.download_daily = patched_daily
    scan.download_4h = patched_4h
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return scan.scan_one_symbol(symbol, name, XunLongIndicator())
    finally:
        scan.download_daily = old_daily
        scan.download_4h = old_4h


def scan_single_symbol(symbol: str, provider: DashboardDataProvider) -> dict:
    symbol = str(symbol or "").strip().upper()
    if not _symbol_ok(symbol):
        raise ValueError("请输入有效股票代码。")

    started = time.time()
    profile = _load_profile(symbol)
    df = _scan_symbol_latest_daily_4h(symbol, profile.get("name", ""), provider)

    signals = []
    if df is not None and not df.empty:
        df = df.copy()
        df["signal_date"] = pd.to_datetime(df["signal_date"], errors="coerce").dt.date
        df["score"] = df.apply(scan.score_buy_signal_row, axis=1)
        df = df.sort_values(["signal_date", "signal_side", "signal_type"], ascending=[False, True, True])
        for _, r in df.iterrows():
            side = str(r.get("signal_side", "") or "").upper()
            signals.append(
                {
                    "date": str(r.get("signal_date", "") or ""),
                    "side": side,
                    "type": str(r.get("signal_type", "") or ""),
                    "model": str(r.get("model", "") or ""),
                    "score": _num(r.get("score")) if side == "BUY" else None,
                    "close": _num(r.get("close")),
                    "rsi": _num(r.get("RSI")),
                    "h4_rsi": _num(r.get("H4_RSI")),
                    "h4_fj": _num(r.get("H4_FJ")),
                    "extra": str(r.get("extra_info", "") or ""),
                }
            )

    latest_buy = next((x for x in signals if x["side"] == "BUY"), None)
    latest_sell = next((x for x in signals if x["side"] == "SELL"), None)
    return {
        "profile": profile,
        "signals": signals,
        "latest_buy": latest_buy,
        "latest_sell": latest_sell,
        "provider": f"{provider.label} / 日线+4H",
        "elapsed": round(time.time() - started, 1),
        "updated": datetime.now().strftime("%H:%M:%S"),
    }


def _load_universe() -> pd.DataFrame:
    _, df_meta = scan.load_input_and_meta(scan.INPUT_FILE)
    if df_meta.empty:
        return pd.DataFrame()
    if "enable" not in df_meta.columns:
        df_meta["enable"] = 1
    enabled = df_meta[pd.to_numeric(df_meta["enable"], errors="coerce").fillna(1).astype(int) == 1].copy()
    df_run, _ = scan.filter_scannable_universe(enabled)
    return df_run.drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)


def _historical_buy_dates() -> dict[str, set]:
    rows = []
    latest = scan.BASE_DIR / "scan_result_latest.xlsx"
    history_files = sorted((scan.BASE_DIR / "history").glob("scan_result_*.xlsx"))[-20:]
    candidates = []
    if latest.exists():
        candidates.append(latest)
    candidates.extend(path for path in history_files if path != latest)
    for path in candidates:
        try:
            df = scan._read_signal_rows_from_result(Path(path), "BUY")
        except Exception:
            continue
        if df is not None and not df.empty:
            rows.append(df)
    if not rows:
        return {}
    hist = pd.concat(rows, ignore_index=True)
    hist["symbol"] = hist["symbol"].astype(str).str.strip().str.upper()
    hist["signal_date"] = pd.to_datetime(hist["signal_date"], errors="coerce").dt.date
    hist = hist.dropna(subset=["symbol", "signal_date"])
    out: dict[str, set] = {}
    for sym, dates in hist.groupby("symbol")["signal_date"]:
        out.setdefault(sym, set()).update(dates.tolist())
    return out


def _latest_signal_date(rows: list[dict]) -> date | None:
    dates = [row.get("signal_date") for row in rows if row.get("signal_date")]
    return max(dates) if dates else None


def _frame_for_symbol(frame_map: dict[str, pd.DataFrame | None], symbol: str) -> pd.DataFrame | None:
    symbol = str(symbol or "").strip().upper()
    if symbol in frame_map:
        return frame_map.get(symbol)
    alt = symbol.replace("-", ".")
    if alt in frame_map:
        return frame_map.get(alt)
    return None


def scan_buy_sector_stats(provider: DashboardDataProvider, progress=None) -> dict:
    started = time.time()
    df_run = _load_universe()
    if df_run.empty:
        raise RuntimeError("没有可扫描标的。")

    historical = _historical_buy_dates()
    rows = []
    errors: list[tuple[str, str]] = []
    total = len(df_run)
    symbols = [str(symbol).strip().upper() for symbol in df_run["symbol"].tolist()]
    try:
        progress(0, total, "预取日线") if progress else None
        daily_map = provider.download_many_daily(symbols, period="1y")
        progress(0, total, "预取4H") if progress else None
        h4_map = provider.download_many_4h(symbols, period="90d")
    except Exception as exc:
        raise RuntimeError(f"批量读取行情失败：{exc}") from exc
    for idx, (_, meta) in enumerate(df_run.iterrows(), start=1):
        symbol = str(meta.get("symbol", "")).strip().upper()
        if progress:
            progress(idx, total, symbol)
        try:
            sig = _scan_symbol_from_frames(
                symbol,
                str(meta.get("name", "") or ""),
                _frame_for_symbol(daily_map, symbol),
                _frame_for_symbol(h4_map, symbol),
            )
        except Exception as exc:
            errors.append((symbol, str(exc)))
            continue
        if sig is None or sig.empty:
            continue
        sig = sig.copy()
        sig["signal_date"] = pd.to_datetime(sig["signal_date"], errors="coerce").dt.date
        sig = sig[sig["signal_side"].astype(str).str.upper().eq("BUY")].copy()
        if sig.empty:
            continue
        sector = scan._normalize_sector_with_code(meta.get("group", ""))
        for signal_date, day_sig in sig.groupby("signal_date"):
            if pd.isna(signal_date):
                continue
            score = day_sig.apply(scan.score_buy_signal_row, axis=1).max()
            prior_dates = historical.get(symbol, set())
            first_seen = True
            for prior_date in prior_dates:
                if prior_date >= signal_date:
                    continue
                days = scan._business_days_between(prior_date, signal_date)
                if 0 < days <= scan.TRACK_MAX_DAYS:
                    first_seen = False
                    break
            rows.append(
                {
                    "symbol": symbol,
                    "signal_date": signal_date,
                    "sector": sector,
                    "score": float(score) if pd.notna(score) else np.nan,
                    "rules": " | ".join([x for x in pd.unique(day_sig["signal_type"].astype(str)) if x]),
                    "first_seen": first_seen,
                }
            )

    df = pd.DataFrame(rows)
    latest_signal_date = _latest_signal_date(rows)
    if not rows and errors:
        sample = "；".join([f"{symbol}: {message}" for symbol, message in errors[:3]])
        raise RuntimeError(f"数据源请求全部失败，无法生成买入统计。示例错误：{sample}")
    if latest_signal_date is not None and not df.empty:
        df = df[df["signal_date"].eq(latest_signal_date)].copy()
    display_rows = [] if df.empty else df.sort_values(["score", "symbol"], ascending=[False, True]).to_dict("records")
    if df.empty:
        all_stats = pd.DataFrame(columns=["sector", "count"])
        first_stats = pd.DataFrame(columns=["sector", "count"])
    else:
        unique = df.drop_duplicates(subset=["symbol"], keep="first")
        all_stats = (
            unique["sector"].astype(str).str.strip().replace("", "99 未分组").value_counts().head(5)
            .rename_axis("sector").reset_index(name="count")
        )
        first_stats = (
            unique[unique["first_seen"]]["sector"].astype(str).str.strip().replace("", "99 未分组").value_counts().head(5)
            .rename_axis("sector").reset_index(name="count")
        )
    return {
        "rows": display_rows,
        "all_stats": all_stats.to_dict("records"),
        "first_stats": first_stats.to_dict("records"),
        "updated": datetime.now().strftime("%H:%M:%S"),
        "elapsed": round(time.time() - started, 1),
        "total": int(len(df_run)),
        "signal_date": str(latest_signal_date or ""),
        "error_count": len(errors),
    }


class IntradayDashboardApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.provider = DashboardDataProvider(os.getenv("STOCK_DASHBOARD_DATA_PROVIDER", "alpaca"))
        self.running = False
        self.stats_running = False
        self.after_id = None
        self.stats_after_id = None
        self.root.title("Stock OneClick 日内Dashboard")
        self.root.geometry("1080x760")
        self.root.minsize(940, 640)
        self._build_ui()

    def _build_ui(self):
        self.root.configure(bg="#f5f7fb")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Helvetica", 18, "bold"), foreground="#1f2a44", background="#f5f7fb")
        style.configure("Sub.TLabel", font=("Helvetica", 10), foreground="#5b677a", background="#f5f7fb")
        style.configure("Main.TButton", font=("Helvetica", 11, "bold"), padding=7)
        style.configure("Card.TFrame", background="#ffffff", relief="flat")
        style.configure("Treeview", rowheight=28, font=("Helvetica", 11))
        style.configure("Treeview.Heading", font=("Helvetica", 11, "bold"))

        top = tk.Frame(self.root, bg="#f5f7fb")
        top.pack(fill="x", padx=18, pady=(16, 8))
        ttk.Label(top, text="日内Dashboard", style="Title.TLabel").pack(side="left")
        self.status_var = tk.StringVar(value=f"数据源：{self.provider.label} | 不生成Excel")
        ttk.Label(top, textvariable=self.status_var, style="Sub.TLabel").pack(side="right")

        controls = tk.Frame(self.root, bg="#f5f7fb")
        controls.pack(fill="x", padx=18, pady=(0, 10))
        ttk.Label(controls, text="股票代码", style="Sub.TLabel").pack(side="left")
        self.symbol_var = tk.StringVar()
        entry = ttk.Entry(controls, textvariable=self.symbol_var, width=16, font=("Helvetica", 14))
        entry.pack(side="left", padx=(8, 8))
        entry.bind("<Return>", lambda _event: self.refresh())
        ttk.Button(controls, text="查询", style="Main.TButton", command=self.refresh).pack(side="left")
        ttk.Button(controls, text="TradingView", command=self.open_tradingview).pack(side="left", padx=(8, 0))
        ttk.Button(controls, text="刷新买入板块统计", command=self.refresh_sector_stats).pack(side="left", padx=(14, 0))
        ttk.Button(controls, text="同步TV清单", command=self.sync_tv_watchlist).pack(side="left", padx=(8, 0))

        summary = tk.Frame(self.root, bg="#f5f7fb")
        summary.pack(fill="x", padx=18, pady=(0, 10))
        self.name_var = tk.StringVar(value="-")
        self.score_var = tk.StringVar(value="-")
        self.buy_var = tk.StringVar(value="-")
        self.sell_var = tk.StringVar(value="-")
        for label, var in [
            ("名称/板块", self.name_var),
            ("最新买点分", self.score_var),
            ("最近BUY", self.buy_var),
            ("最近SELL", self.sell_var),
        ]:
            card = tk.Frame(summary, bg="#ffffff", highlightbackground="#d8e0ee", highlightthickness=1)
            card.pack(side="left", fill="x", expand=True, padx=(0, 8))
            ttk.Label(card, text=label, background="#ffffff", foreground="#6b7280", font=("Helvetica", 10)).pack(anchor="w", padx=10, pady=(8, 0))
            ttk.Label(card, textvariable=var, background="#ffffff", foreground="#111827", font=("Helvetica", 15, "bold")).pack(anchor="w", padx=10, pady=(3, 8))

        stats_frame = tk.Frame(self.root, bg="#f5f7fb")
        stats_frame.pack(fill="x", padx=18, pady=(0, 10))
        self.stats_status_var = tk.StringVar(value="买入板块统计：未刷新")
        ttk.Label(stats_frame, textvariable=self.stats_status_var, style="Sub.TLabel").pack(anchor="w", pady=(0, 5))
        stats_tables = tk.Frame(stats_frame, bg="#f5f7fb")
        stats_tables.pack(fill="x")
        self.all_stats_tree = self._make_stats_tree(stats_tables, "买入板块Top5统计（按出现只数）")
        self.first_stats_tree = self._make_stats_tree(stats_tables, "买入板块Top5统计（按出现只数）首次出现")

        table_frame = tk.Frame(self.root, bg="#ffffff", highlightbackground="#d8e0ee", highlightthickness=1)
        table_frame.pack(fill="both", expand=True, padx=18, pady=(0, 14))
        cols = ("date", "side", "rule", "score", "close", "rsi", "h4", "note")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings")
        headings = {
            "date": "日期",
            "side": "方向",
            "rule": "规则",
            "score": "分数",
            "close": "价格",
            "rsi": "RSI",
            "h4": "4H",
            "note": "说明",
        }
        widths = {"date": 105, "side": 70, "rule": 130, "score": 60, "close": 80, "rsi": 70, "h4": 100, "note": 360}
        for col in cols:
            self.tree.heading(col, text=headings[col])
            self.tree.column(col, width=widths[col], anchor="w")
        self.tree.pack(fill="both", expand=True)

    def _make_stats_tree(self, parent, title: str):
        frame = tk.Frame(parent, bg="#ffffff", highlightbackground="#d8e0ee", highlightthickness=1)
        frame.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Label(frame, text=title, background="#ffffff", foreground="#374151", font=("Helvetica", 11, "bold")).pack(anchor="w", padx=10, pady=(7, 2))
        tree = ttk.Treeview(frame, columns=("rank", "sector", "count"), show="headings", height=5)
        tree.heading("rank", text="排名")
        tree.heading("sector", text="板块")
        tree.heading("count", text="股票只数")
        tree.column("rank", width=50, anchor="center")
        tree.column("sector", width=260, anchor="w")
        tree.column("count", width=80, anchor="center")
        tree.pack(fill="x", padx=8, pady=(0, 8))
        return tree

    def refresh(self):
        if self.running:
            return
        symbol = self.symbol_var.get().strip().upper()
        if not symbol:
            messagebox.showinfo("提示", "请输入股票代码。")
            return
        self.symbol_var.set(symbol)
        self.running = True
        self.status_var.set(f"正在查询 {symbol}...")
        threading.Thread(target=self._refresh_worker, args=(symbol,), daemon=True).start()

    def _refresh_worker(self, symbol: str):
        try:
            payload = scan_single_symbol(symbol, self.provider)
        except Exception as exc:
            self.root.after(0, self._show_error, str(exc))
            return
        self.root.after(0, self._render, payload)

    def refresh_sector_stats(self):
        if self.stats_running:
            return
        self.stats_running = True
        self.stats_status_var.set("买入板块统计：正在用日线+4H扫描...")
        threading.Thread(target=self._sector_stats_worker, daemon=True).start()

    def _sector_stats_worker(self):
        def progress(idx, total, symbol):
            if idx == 1 or idx % 10 == 0 or idx == total:
                self.root.after(0, self.stats_status_var.set, f"买入板块统计：{idx}/{total} {symbol}（日线+4H）")

        try:
            payload = scan_buy_sector_stats(self.provider, progress=progress)
        except Exception as exc:
            self.root.after(0, self._show_stats_error, str(exc))
            return
        self.root.after(0, self._render_sector_stats, payload)

    def _show_error(self, message: str):
        self.running = False
        self.status_var.set(f"错误：{message}")
        messagebox.showerror("日内Dashboard", message)

    def _show_stats_error(self, message: str):
        self.stats_running = False
        self.stats_status_var.set(f"买入板块统计错误：{message}")
        messagebox.showerror("买入板块统计", message)

    def _render(self, payload: dict):
        self.running = False
        profile = payload["profile"]
        name = profile.get("name") or profile.get("symbol")
        sector = profile.get("sector") or "未分组"
        self.name_var.set(f"{name} / {sector}")
        latest_buy = payload.get("latest_buy")
        latest_sell = payload.get("latest_sell")
        self.score_var.set("-" if not latest_buy or latest_buy.get("score") is None else f"{latest_buy['score']:.0f}")
        self.buy_var.set("-" if not latest_buy else f"{latest_buy['date']} {latest_buy['type']}")
        self.sell_var.set("-" if not latest_sell else f"{latest_sell['date']} {latest_sell['type']}")
        self.status_var.set(f"数据源：{payload['provider']} | 更新：{payload['updated']} | 用时：{payload['elapsed']}s | 不生成Excel")

        for item in self.tree.get_children():
            self.tree.delete(item)
        for sig in payload.get("signals", []):
            score = "" if sig.get("score") is None else f"{sig['score']:.0f}"
            close = "" if sig.get("close") is None else f"{sig['close']:.2f}"
            rsi = "" if sig.get("rsi") is None else f"{sig['rsi']:.1f}"
            h4_parts = []
            if sig.get("h4_rsi") is not None:
                h4_parts.append(f"RSI {sig['h4_rsi']:.1f}")
            if sig.get("h4_fj") is not None:
                h4_parts.append(f"FJ {sig['h4_fj']:.1f}")
            self.tree.insert(
                "",
                "end",
                values=(sig["date"], sig["side"], sig["type"], score, close, rsi, " / ".join(h4_parts), sig["extra"]),
            )
        self._schedule_next()

    def _render_sector_stats(self, payload: dict):
        self.stats_running = False
        self._fill_stats_tree(self.all_stats_tree, payload.get("all_stats", []))
        self._fill_stats_tree(self.first_stats_tree, payload.get("first_stats", []))
        signal_count = len(payload.get("rows", []))
        signal_date = payload.get("signal_date") or "无"
        error_text = "" if not payload.get("error_count") else f" | 失败 {payload['error_count']} 只"
        self.stats_status_var.set(
            f"买入板块统计：{payload['updated']} 更新 | 信号日 {signal_date} | 触发 {signal_count} 只 | 扫描 {payload['total']} 只{error_text} | 用时 {payload['elapsed']}s | 日线+4H | 5分钟刷新"
        )
        self._schedule_stats_next()

    def _fill_stats_tree(self, tree, rows):
        for item in tree.get_children():
            tree.delete(item)
        for idx, row in enumerate(rows or [], start=1):
            tree.insert("", "end", values=(idx, row.get("sector", ""), row.get("count", "")))

    def _schedule_next(self):
        if self.after_id:
            self.root.after_cancel(self.after_id)
        self.after_id = self.root.after(REFRESH_SECONDS * 1000, self.refresh)

    def _schedule_stats_next(self):
        if self.stats_after_id:
            self.root.after_cancel(self.stats_after_id)
        self.stats_after_id = self.root.after(REFRESH_SECONDS * 1000, self.refresh_sector_stats)

    def open_tradingview(self):
        symbol = self.symbol_var.get().strip().upper()
        if not symbol:
            return
        import subprocess

        subprocess.run(["open", f"https://www.tradingview.com/chart/?symbol={symbol}"], check=False)

    def sync_tv_watchlist(self):
        if self.running or self.stats_running:
            messagebox.showinfo("提示", "查询/统计正在运行，结束后再同步清单。")
            return
        url = simpledialog.askstring(
            "同步 TradingView 清单",
            "粘贴 TradingView watchlist 链接：",
            initialvalue="https://www.tradingview.com/watchlists/323650703/",
            parent=self.root,
        )
        if not url:
            return
        self.status_var.set("正在同步 TradingView 清单...")
        threading.Thread(target=self._sync_tv_worker, args=(url.strip(),), daemon=True).start()

    def _sync_tv_worker(self, url: str):
        try:
            result = sync_watchlist_url(
                url,
                scan.INPUT_FILE,
                scan.HISTORY_DIR,
                scan.EXPORT_DIR,
            )
        except Exception as exc:
            self.root.after(0, self._show_error, f"同步TV清单失败：{exc}")
            return
        self.root.after(0, self._show_sync_result, result)

    def _show_sync_result(self, result: dict):
        self.status_var.set(
            f"TV清单同步完成：{result.get('parsed_count', 0)} 个 | 新增 {result.get('added_count', 0)} 个"
        )
        messagebox.showinfo(
            "同步完成",
            "TradingView 清单已同步到 Stock OneClick 模板。\n"
            f"清单：{result.get('watchlist_name', '')}\n"
            f"标的：{result.get('parsed_count', 0)} 个\n"
            f"新增：{result.get('added_count', 0)} 个",
        )


def main():
    root = tk.Tk()
    app = IntradayDashboardApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
