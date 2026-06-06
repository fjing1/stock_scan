#!/usr/bin/env python3
from __future__ import annotations

import re
import os
import subprocess
import threading
import tkinter as tk
from pathlib import Path
import sys
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText

from watchlist_importer import import_watchlist_file, sync_watchlist_url
import scan_stocks


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
PYTHON_BIN = sys.executable or "/usr/local/bin/python3"
SCAN_SCRIPT = SCRIPT_DIR / "scan_stocks.py"
TEST_SCRIPT = SCRIPT_DIR / "stock_oneclick_test.py"
INTRADAY_DASHBOARD_SCRIPT = SCRIPT_DIR / "intraday_dashboard_app.py"
RESULT_LATEST = BASE_DIR / "scan_result_latest.xlsx"
TEST_RESULT_LATEST = BASE_DIR / "exports" / "stock_oneclick_test_latest.xlsx"
TV_A_POOL = BASE_DIR / "exports" / "tv_A_pool.txt"
TV_BUY_SIGNALS_DIR = BASE_DIR / "tv_buy_signals"
INPUT_FILE = BASE_DIR / "stock_input_template.xlsx"
HISTORY_DIR = BASE_DIR / "history"


class ScanGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Stock OneClick")
        self.root.geometry("860x560")
        self.root.minsize(760, 500)

        self.process: subprocess.Popen[str] | None = None
        self.running = False

        self.total = 0
        self.current = 0

        self._build_ui()

    def _build_ui(self):
        self.root.configure(bg="#f5f7fb")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Helvetica", 18, "bold"), foreground="#1f2a44", background="#f5f7fb")
        style.configure("Sub.TLabel", font=("Helvetica", 11), foreground="#4b5563", background="#f5f7fb")
        style.configure("Main.TButton", font=("Helvetica", 11, "bold"), padding=8)
        style.configure("Aux.TButton", font=("Helvetica", 10), padding=6)
        style.configure("TProgressbar", troughcolor="#dfe6f1", background="#2f6feb", thickness=14)

        top = tk.Frame(self.root, bg="#f5f7fb")
        top.pack(fill="x", padx=18, pady=(16, 8))

        ttk.Label(top, text="Stock OneClick", style="Title.TLabel").pack(anchor="w")
        ttk.Label(top, text="一键扫描 + 实时进度 + TradingView 清单", style="Sub.TLabel").pack(anchor="w", pady=(2, 0))

        controls = tk.Frame(self.root, bg="#f5f7fb")
        controls.pack(fill="x", padx=18, pady=(6, 4))
        controls2 = tk.Frame(self.root, bg="#f5f7fb")
        controls2.pack(fill="x", padx=18, pady=(0, 8))

        self.btn_start = ttk.Button(controls, text="开始扫描", style="Main.TButton", command=self.start_scan)
        self.btn_start.pack(side="left")
        self.btn_test = ttk.Button(controls, text="测试扫描", style="Main.TButton", command=self.start_test_scan)
        self.btn_test.pack(side="left", padx=(8, 0))

        ttk.Button(controls, text="打开结果文件夹", style="Aux.TButton", command=self.open_folder).pack(side="left", padx=8)
        ttk.Button(controls, text="打开最新结果", style="Aux.TButton", command=self.open_latest_result).pack(side="left")
        ttk.Button(controls, text="打开测试结果", style="Aux.TButton", command=self.open_test_result).pack(side="left", padx=8)
        ttk.Button(controls, text="打开A池文件", style="Aux.TButton", command=self.open_a_pool).pack(side="left", padx=8)
        ttk.Button(controls, text="打开买入TXT", style="Aux.TButton", command=self.open_buy_signals_folder).pack(side="left")
        ttk.Button(controls2, text="导入TV清单", style="Aux.TButton", command=self.import_tv_watchlist).pack(side="left")
        ttk.Button(controls2, text="同步TV链接", style="Aux.TButton", command=self.sync_tv_watchlist_url).pack(side="left", padx=(8, 0))
        ttk.Button(controls2, text="日内Dashboard", style="Aux.TButton", command=self.open_intraday_dashboard).pack(side="left", padx=(8, 0))
        ttk.Button(controls2, text="市场判断说明", style="Aux.TButton", command=self.show_market_context_help).pack(side="left", padx=(8, 0))

        prog_frame = tk.Frame(self.root, bg="#f5f7fb")
        prog_frame.pack(fill="x", padx=18, pady=(0, 10))

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(prog_frame, textvariable=self.status_var, style="Sub.TLabel").pack(anchor="w")
        self.progress = ttk.Progressbar(prog_frame, mode="determinate", maximum=100, value=0)
        self.progress.pack(fill="x", pady=(6, 0))

        log_card = tk.Frame(self.root, bg="#ffffff", highlightbackground="#d8e0ee", highlightthickness=1)
        log_card.pack(fill="both", expand=True, padx=18, pady=(0, 16))

        self.log = ScrolledText(
            log_card,
            wrap="word",
            font=("Menlo", 11),
            bg="#ffffff",
            fg="#1f2937",
            insertbackground="#1f2937",
            relief="flat",
            padx=10,
            pady=10,
        )
        self.log.pack(fill="both", expand=True)
        self.log.insert("end", "点击“开始扫描”后，这里会显示实时日志。\n")
        self.log.configure(state="disabled")

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def append_log(self, text: str):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def set_status(self, text: str):
        self.status_var.set(text)

    def update_progress(self, current: int, total: int):
        if str(self.progress.cget("mode")) != "determinate":
            self.progress.stop()
            self.progress.configure(mode="determinate")
        self.current = current
        self.total = total
        if total <= 0:
            self.progress.configure(value=0, maximum=100)
            return
        self.progress.configure(maximum=total, value=current)
        pct = int(current / total * 100)
        self.set_status(f"扫描进度：{current}/{total} ({pct}%)")

    def open_folder(self):
        subprocess.run(["open", str(BASE_DIR)], check=False)

    def open_latest_result(self):
        if RESULT_LATEST.exists():
            subprocess.run(["open", str(RESULT_LATEST)], check=False)
        else:
            messagebox.showinfo("提示", "还没有最新结果文件。请先运行一次扫描。")

    def open_test_result(self):
        if TEST_RESULT_LATEST.exists():
            subprocess.run(["open", str(TEST_RESULT_LATEST)], check=False)
        else:
            messagebox.showinfo("提示", "还没有测试结果文件。请先运行一次测试扫描。")

    def open_a_pool(self):
        if TV_A_POOL.exists():
            subprocess.run(["open", str(TV_A_POOL)], check=False)
        else:
            messagebox.showinfo("提示", "未找到 tv_A_pool.txt。请先运行一次扫描。")

    def open_buy_signals_folder(self):
        TV_BUY_SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(["open", str(TV_BUY_SIGNALS_DIR)], check=False)

    def open_intraday_dashboard(self):
        if not INTRADAY_DASHBOARD_SCRIPT.exists():
            messagebox.showerror("错误", f"找不到脚本：{INTRADAY_DASHBOARD_SCRIPT}")
            return
        subprocess.Popen(
            [PYTHON_BIN, str(INTRADAY_DASHBOARD_SCRIPT)],
            cwd=str(SCRIPT_DIR),
            env={**os.environ, "STOCK_DASHBOARD_DATA_PROVIDER": os.environ.get("STOCK_DASHBOARD_DATA_PROVIDER", "alpaca")},
        )

    def show_market_context_help(self):
        win = tk.Toplevel(self.root)
        win.title("市场方向判断说明")
        win.geometry("760x620")
        win.minsize(680, 520)
        frame = tk.Frame(win, bg="#f5f7fb")
        frame.pack(fill="both", expand=True, padx=14, pady=14)
        text = ScrolledText(
            frame,
            wrap="word",
            font=("Helvetica", 12),
            bg="#ffffff",
            fg="#1f2937",
            relief="flat",
            padx=12,
            pady=12,
        )
        text.pack(fill="both", expand=True)
        text.insert("end", scan_stocks.MARKET_CONTEXT_HELP_TEXT)
        text.configure(state="disabled")
        ttk.Button(frame, text="关闭", command=win.destroy).pack(anchor="e", pady=(10, 0))

    def import_tv_watchlist(self):
        if self.running:
            messagebox.showinfo("提示", "扫描正在运行，结束后再导入清单。")
            return
        path = filedialog.askopenfilename(
            title="选择 TradingView 导出的 watchlist .txt",
            initialdir=str(BASE_DIR),
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            result = import_watchlist_file(Path(path), INPUT_FILE, HISTORY_DIR)
        except Exception as exc:
            messagebox.showerror("导入失败", str(exc))
            self.append_log(f"\n[导入失败] {exc}\n")
            return

        added = result["added_symbols"]
        preview = ", ".join(added[:30])
        if len(added) > 30:
            preview += f" ... +{len(added) - 30}"
        self.append_log(
            "\n=== TradingView 清单导入完成 ===\n"
            f"文件：{path}\n"
            f"解析：{result['parsed_count']} 个 symbol\n"
            f"新增：{result['added_count']} 个\n"
            f"新增列表：{preview if preview else '无，Excel 已经包含这些 symbol'}\n"
            f"备份：{result['backup_path']}\n"
        )
        self.set_status(f"TV清单导入完成：新增 {result['added_count']} 个")
        messagebox.showinfo("导入完成", f"解析 {result['parsed_count']} 个，新增 {result['added_count']} 个。")

    def sync_tv_watchlist_url(self):
        if self.running:
            messagebox.showinfo("提示", "扫描正在运行，结束后再同步清单。")
            return
        url = simpledialog.askstring(
            "同步 TradingView 链接",
            "粘贴 TradingView watchlist 链接：",
            initialvalue="https://www.tradingview.com/watchlists/323650703/",
            parent=self.root,
        )
        if not url:
            return
        try:
            result = sync_watchlist_url(url.strip(), INPUT_FILE, HISTORY_DIR, BASE_DIR / "exports")
        except Exception as exc:
            messagebox.showerror("同步失败", str(exc))
            self.append_log(f"\n[同步失败] {exc}\n")
            return

        added = result["added_symbols"]
        preview = ", ".join(added[:30])
        if len(added) > 30:
            preview += f" ... +{len(added) - 30}"
        self.append_log(
            "\n=== TradingView 链接同步完成 ===\n"
            f"链接：{url}\n"
            f"清单：{result['watchlist_name']} / modified={result['modified']}\n"
            f"同步：{result['parsed_count']} 个唯一 symbol\n"
            f"新增：{result['added_count']} 个\n"
            f"新增列表：{preview if preview else '无'}\n"
            f"备份：{result['backup_path']}\n"
            f"快照：{result['snapshot_path']}\n"
        )
        self.set_status(f"TV链接同步完成：{result['parsed_count']} 个")
        messagebox.showinfo("同步完成", f"同步 {result['parsed_count']} 个唯一 symbol，新增 {result['added_count']} 个。")

    def start_scan(self):
        self.start_task("正式扫描", SCAN_SCRIPT, "=== 开始扫描 ===")

    def start_test_scan(self):
        self.start_task("测试扫描", TEST_SCRIPT, "=== 开始测试扫描：收盘价 + 12点盘中 ===")

    def start_task(self, task_name: str, script_path: Path, log_title: str):
        if self.running:
            return
        if not script_path.exists():
            messagebox.showerror("错误", f"找不到脚本：{script_path}")
            return

        self.running = True
        self.btn_start.configure(state="disabled")
        self.btn_test.configure(state="disabled")
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        self.set_status(f"{task_name}启动中...")
        self.append_log(f"\n{log_title}\nPython: {PYTHON_BIN}\nScript: {script_path}\n")
        self.root.update_idletasks()

        t = threading.Thread(target=self._run_scan_thread, args=(task_name, script_path), daemon=True)
        t.start()

    def _run_scan_thread(self, task_name: str, script_path: Path):
        cmd = [PYTHON_BIN, "-u", str(script_path)]
        progress_re = re.compile(r"\[[^\]]*?(\d+)/(\d+)(?:\s*\||\])")
        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=str(SCRIPT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.root.after(0, self.append_log, line)
                m = progress_re.search(line)
                if m:
                    cur = int(m.group(1))
                    total = int(m.group(2))
                    self.root.after(0, self.update_progress, cur, total)

            code = self.process.wait()
            self.root.after(0, self._on_scan_done, code, task_name)
        except Exception as e:
            self.root.after(0, self.append_log, f"\n[GUI错误] 启动失败: {e}\n")
            self.root.after(0, self._on_scan_done, 1, task_name)

    def _on_scan_done(self, code: int, task_name: str = "扫描"):
        self.running = False
        self.btn_start.configure(state="normal")
        self.btn_test.configure(state="normal")
        self.progress.stop()
        if code == 0:
            if self.total > 0 and self.current > 0:
                self.set_status(f"完成：{self.current}/{self.total}")
                self.progress.configure(value=self.total)
            else:
                self.set_status("完成")
                self.progress.configure(mode="determinate", maximum=100, value=100)
            self.append_log(f"=== {task_name}完成 ===\n")
            self.root.bell()
        else:
            self.set_status("运行失败，请看日志")
            self.progress.configure(mode="determinate", maximum=100, value=0)
            self.append_log(f"=== {task_name}失败 ===\n")
            self.root.bell()

    def on_close(self):
        if self.running and self.process and self.process.poll() is None:
            if not messagebox.askyesno("确认退出", "扫描仍在运行，确定退出吗？"):
                return
            try:
                self.process.terminate()
            except Exception:
                pass
        self.root.destroy()


def main():
    root = tk.Tk()
    app = ScanGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
