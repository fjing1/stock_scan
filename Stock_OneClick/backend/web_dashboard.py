#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

# 盘中 dashboard 使用独立数据源，避免影响每日收盘 Stock OneClick 扫描。
# 如需临时切换，可在启动前显式设置 STOCK_DASHBOARD_DATA_PROVIDER。
os.environ.setdefault("STOCK_DASHBOARD_DATA_PROVIDER", "alpaca")

from dashboard_data import DashboardDataProvider
from realtime_dashboard import REFRESH_SECONDS, scan_universe
import scan_stocks as scan
from xunlong import XunLongIndicator


HOST = os.getenv("STOCK_WEB_DASHBOARD_HOST", "127.0.0.1")
PORT = int(os.getenv("STOCK_WEB_DASHBOARD_PORT", "8765"))
DEFAULT_LIMIT = int(os.getenv("STOCK_DASHBOARD_LIMIT", "0"))
BASE_DIR = Path(__file__).resolve().parent.parent


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stock OneClick 买点雷达</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f4;
      --ink: #17201b;
      --muted: #66736d;
      --line: #dfe5dc;
      --panel: #ffffff;
      --green: #0b8f69;
      --green-soft: #dff3ea;
      --gold: #be8a1d;
      --gold-soft: #fbefcf;
      --red: #b34242;
      --blue: #2b5f9f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      letter-spacing: 0;
    }
    .shell { max-width: 1220px; margin: 0 auto; padding: 22px 22px 34px; }
    header {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 18px;
      align-items: end;
      border-bottom: 1px solid var(--line);
      padding-bottom: 18px;
    }
    h1 { margin: 0; font-size: 28px; line-height: 1.1; font-weight: 760; }
    .sub { margin-top: 7px; color: var(--muted); font-size: 14px; }
    .toolbar { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
    button, select, input {
      height: 34px;
      border: 1px solid #cfd8d1;
      background: #fff;
      color: var(--ink);
      border-radius: 7px;
      padding: 0 11px;
      font: inherit;
      font-size: 13px;
    }
    input { min-width: 150px; text-transform: uppercase; }
    button.primary { background: var(--ink); color: #fff; border-color: var(--ink); }
    .single {
      margin: 18px 0 0;
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 8px;
      padding: 13px;
    }
    .single-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .single-title { font-size: 13px; color: var(--muted); font-weight: 680; margin: 0 0 10px; }
    .single-result { margin-top: 12px; display: grid; gap: 10px; }
    .single-head {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: start;
      border-top: 1px solid var(--line);
      padding-top: 12px;
    }
    .single-meta { color: var(--muted); font-size: 12px; margin-top: 4px; }
    .signal-table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .signal-table th, .signal-table td { text-align: left; border-top: 1px solid var(--line); padding: 8px 6px; vertical-align: top; }
    .signal-table th { color: var(--muted); font-weight: 680; }
    .side-buy { color: var(--green); font-weight: 730; }
    .side-sell { color: var(--red); font-weight: 730; }
    .status-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin: 18px 0;
    }
    .stat {
      background: transparent;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 13px;
      min-height: 72px;
    }
    .stat .label { color: var(--muted); font-size: 12px; }
    .stat .value { margin-top: 6px; font-size: 22px; font-weight: 740; }
    .layout { display: grid; grid-template-columns: 260px minmax(0, 1fr); gap: 18px; align-items: start; }
    .side {
      border-right: 1px solid var(--line);
      padding-right: 18px;
      position: sticky;
      top: 14px;
    }
    .side h2, .main h2 { font-size: 13px; color: var(--muted); font-weight: 680; margin: 0 0 10px; }
    .sector-list { display: grid; gap: 7px; }
    .sector {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: center;
      padding: 9px 10px;
      border-radius: 7px;
      border: 1px solid transparent;
      background: transparent;
      cursor: pointer;
    }
    .sector.active { background: #fff; border-color: var(--line); }
    .sector-name { font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .sector-count {
      min-width: 26px;
      height: 22px;
      border-radius: 999px;
      background: var(--green-soft);
      color: var(--green);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 12px;
      font-weight: 720;
    }
    .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(282px, 1fr)); gap: 12px; }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 13px;
      min-height: 184px;
      display: grid;
      gap: 10px;
      align-content: start;
    }
    .card-top { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }
    .symbol { font-size: 23px; font-weight: 800; line-height: 1; }
    .sector-chip { color: var(--muted); font-size: 12px; margin-top: 4px; max-width: 190px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .score { text-align: right; }
    .score-num { font-size: 25px; font-weight: 820; color: var(--green); line-height: 1; }
    .score-label { color: var(--muted); font-size: 11px; margin-top: 3px; }
    .bar { height: 7px; background: #edf1eb; border-radius: 999px; overflow: hidden; }
    .bar span { display: block; height: 100%; background: linear-gradient(90deg, var(--green), #83bd45); width: 0%; }
    .tags { display: flex; flex-wrap: wrap; gap: 6px; }
    .tag { font-size: 12px; padding: 5px 7px; border-radius: 999px; background: var(--green-soft); color: var(--green); font-weight: 650; }
    .tag.formal { background: var(--gold-soft); color: var(--gold); }
    .meta { display: grid; grid-template-columns: repeat(3, 1fr); gap: 7px; }
    .metric { border-top: 1px solid var(--line); padding-top: 7px; }
    .metric .k { color: var(--muted); font-size: 11px; }
    .metric .v { margin-top: 2px; font-size: 14px; font-weight: 690; }
    .why { color: #35423b; font-size: 12px; line-height: 1.45; }
    .empty {
      border: 1px dashed #cbd6ce;
      border-radius: 8px;
      padding: 40px 18px;
      text-align: center;
      color: var(--muted);
      background: rgba(255,255,255,.5);
    }
    .error { color: var(--red); white-space: pre-wrap; font-size: 13px; }
    .loading {
      position: fixed;
      inset: auto 18px 18px auto;
      background: var(--ink);
      color: #fff;
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 13px;
      box-shadow: 0 10px 24px rgba(0,0,0,.16);
      display: none;
    }
    .loading.show { display: block; }
    @media (max-width: 820px) {
      header { grid-template-columns: 1fr; align-items: start; }
      .toolbar { justify-content: flex-start; }
      .status-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .layout { grid-template-columns: 1fr; }
      .side { position: static; border-right: 0; border-bottom: 1px solid var(--line); padding: 0 0 14px; }
      .cards { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div>
        <h1>买点雷达</h1>
        <div class="sub" id="subtitle">输入单只股票代码查询；全市场买点雷达需手动刷新。</div>
      </div>
      <div class="toolbar">
        <select id="scope">
          <option value="today">今天触发</option>
          <option value="recent">最近窗口</option>
        </select>
        <button id="refresh" class="primary">刷新全市场</button>
      </div>
    </header>

    <section class="single">
      <div class="single-title">单票查询</div>
      <div class="single-row">
        <input id="singleSymbol" placeholder="输入代码，如 CLS" />
        <button id="singleSearch" class="primary">查询</button>
      </div>
      <div class="single-result" id="singleResult"></div>
    </section>

    <section class="status-grid">
      <div class="stat"><div class="label">买点数量</div><div class="value" id="buyCount">-</div></div>
      <div class="stat"><div class="label">最高观海分</div><div class="value" id="topScore">-</div></div>
      <div class="stat"><div class="label">数据源</div><div class="value" id="provider">-</div></div>
      <div class="stat"><div class="label">刷新时间</div><div class="value" id="updated">-</div></div>
    </section>

    <div class="layout">
      <aside class="side">
        <h2>板块</h2>
        <div class="sector-list" id="sectors"></div>
      </aside>
      <main class="main">
        <h2 id="sectionTitle">信号</h2>
        <div class="cards" id="cards"></div>
      </main>
    </div>
  </div>
  <div class="loading" id="loading">正在扫描...</div>

  <script>
    const state = { rows: [], selected: "全部" };
    const refreshMs = Number("{{REFRESH_MS}}");

    function fmt(v, digits = 1) {
      if (v === null || v === undefined || v === "" || Number.isNaN(Number(v))) return "";
      return Number(v).toFixed(digits);
    }
    function cleanRule(rule) {
      return String(rule || "").replaceAll("|", "+");
    }
    function scoreClass(row) {
      const rule = String(row.signal_type || "");
      if (rule.includes("正式")) return "formal";
      return "";
    }
    function setLoading(on) {
      document.getElementById("loading").classList.toggle("show", on);
      document.getElementById("refresh").disabled = on;
    }
    function render() {
      const rows = state.rows;
      const filtered = state.selected === "全部" ? rows : rows.filter(r => r.sector === state.selected);
      document.getElementById("buyCount").textContent = String(rows.length);
      const top = rows.reduce((m, r) => Math.max(m, Number(r.score || 0)), 0);
      document.getElementById("topScore").textContent = rows.length ? String(Math.round(top)) : "-";
      document.getElementById("sectionTitle").textContent = state.selected === "全部" ? "全部买点" : state.selected;

      const counts = new Map();
      rows.forEach(r => counts.set(r.sector, (counts.get(r.sector) || 0) + 1));
      const sectors = [["全部", rows.length], ...Array.from(counts.entries()).sort((a,b) => b[1] - a[1] || a[0].localeCompare(b[0]))];
      document.getElementById("sectors").innerHTML = sectors.map(([name, count]) => `
        <div class="sector ${state.selected === name ? "active" : ""}" data-sector="${name}">
          <div class="sector-name">${name}</div>
          <div class="sector-count">${count}</div>
        </div>
      `).join("");
      document.querySelectorAll(".sector").forEach(el => {
        el.addEventListener("click", () => { state.selected = el.dataset.sector; render(); });
      });

      const cards = document.getElementById("cards");
      if (!filtered.length) {
        cards.innerHTML = `<div class="empty">当前范围没有买点。</div>`;
        return;
      }
      cards.innerHTML = filtered.map(row => {
        const score = Math.max(0, Math.min(100, Number(row.score || 0)));
        return `
          <article class="card">
            <div class="card-top">
              <div>
                <div class="symbol">${row.symbol}</div>
                <div class="sector-chip">${row.sector || ""}</div>
              </div>
              <div class="score">
                <div class="score-num">${Math.round(score)}</div>
                <div class="score-label">观海分</div>
              </div>
            </div>
            <div class="bar"><span style="width:${score}%"></span></div>
            <div class="tags">
              <span class="tag ${scoreClass(row)}">${cleanRule(row.signal_type)}</span>
              <span class="tag">${row.signal_date || ""}</span>
            </div>
            <div class="meta">
              <div class="metric"><div class="k">价格</div><div class="v">${fmt(row.close, 2)}</div></div>
              <div class="metric"><div class="k">RSI</div><div class="v">${fmt(row.rsi, 1)}</div></div>
              <div class="metric"><div class="k">4H RSI</div><div class="v">${fmt(row.h4_rsi, 1)}</div></div>
            </div>
            <div class="why">${row.extra_info || ""}</div>
          </article>
        `;
      }).join("");
    }
    async function loadData() {
      setLoading(true);
      try {
        const scope = document.getElementById("scope").value;
        const response = await fetch(`/api/signals?scope=${scope}`);
        const payload = await response.json();
        if (!response.ok || payload.error) throw new Error(payload.error || "请求失败");
        state.rows = payload.rows || [];
        document.getElementById("provider").textContent = payload.provider || "-";
        document.getElementById("updated").textContent = payload.updated_at || "-";
        document.getElementById("subtitle").textContent = `${payload.scope_label}，只看 BUY，${Math.round(payload.elapsed_seconds || 0)} 秒完成。`;
        render();
      } catch (err) {
        document.getElementById("cards").innerHTML = `<div class="empty error">${err.message}</div>`;
      } finally {
        setLoading(false);
      }
    }
    function sideClass(side) {
      return String(side || "").toUpperCase() === "SELL" ? "side-sell" : "side-buy";
    }
    async function loadSingle() {
      const input = document.getElementById("singleSymbol");
      const symbol = input.value.trim().toUpperCase();
      if (!symbol) return;
      input.value = symbol;
      const target = document.getElementById("singleResult");
      target.innerHTML = `<div class="empty">正在查询 ${symbol}...</div>`;
      try {
        const response = await fetch(`/api/single?symbol=${encodeURIComponent(symbol)}`);
        const payload = await response.json();
        if (!response.ok || payload.error) throw new Error(payload.error || "请求失败");
        const rows = payload.signals || [];
        const latestBuy = payload.latest_buy;
        target.innerHTML = `
          <div class="single-head">
            <div>
              <div class="symbol">${payload.symbol}</div>
              <div class="single-meta">${payload.name || ""}${payload.sector ? " · " + payload.sector : ""} · ${payload.provider || ""} · ${payload.elapsed_seconds || 0}s</div>
            </div>
            <div class="score">
              <div class="score-num">${latestBuy && latestBuy.score !== null ? Math.round(Number(latestBuy.score)) : "-"}</div>
              <div class="score-label">${latestBuy ? latestBuy.signal_date + " 买点分" : "无近期买点"}</div>
            </div>
          </div>
          ${rows.length ? `
            <table class="signal-table">
              <thead><tr><th>日期</th><th>方向</th><th>规则</th><th>分数</th><th>价格</th><th>说明</th></tr></thead>
              <tbody>${rows.map(r => `
                <tr>
                  <td>${r.signal_date || ""}</td>
                  <td class="${sideClass(r.signal_side)}">${r.signal_side || ""}</td>
                  <td>${cleanRule(r.signal_type || "")}</td>
                  <td>${r.score !== null && r.score !== undefined ? Math.round(Number(r.score)) : ""}</td>
                  <td>${fmt(r.close, 2)}</td>
                  <td>${r.extra_info || ""}</td>
                </tr>
              `).join("")}</tbody>
            </table>
          ` : `<div class="empty">最近窗口没有 BUY/SELL 信号。</div>`}
        `;
      } catch (err) {
        target.innerHTML = `<div class="empty error">${err.message}</div>`;
      }
    }
    document.getElementById("refresh").addEventListener("click", loadData);
    document.getElementById("scope").addEventListener("change", loadData);
    document.getElementById("singleSearch").addEventListener("click", loadSingle);
    document.getElementById("singleSymbol").addEventListener("keydown", event => {
      if (event.key === "Enter") loadSingle();
    });
    document.getElementById("cards").innerHTML = `<div class="empty">需要全市场买点雷达时，点击“刷新全市场”。</div>`;
    setInterval(() => {
      if (document.getElementById("singleSymbol").value.trim()) loadSingle();
    }, refreshMs);
  </script>
</body>
</html>
"""


class SignalCache:
    def __init__(self, limit: int = 0):
        self.limit = limit
        self.provider = DashboardDataProvider()
        self.lock = threading.Lock()
        self.cache: dict[bool, dict] = {}

    def get(self, today_only: bool, force: bool = False) -> dict:
        now = time.time()
        with self.lock:
            cached = self.cache.get(today_only)
            if cached and not force and now - cached["ts"] < REFRESH_SECONDS:
                return cached["payload"]

        started = time.time()
        df = scan_universe(limit=self.limit, today_only=today_only, provider=self.provider)
        payload = {
            "rows": dataframe_to_rows(df),
            "updated_at": datetime.now().strftime("%H:%M:%S"),
            "elapsed_seconds": round(time.time() - started, 1),
            "provider": self.provider.label,
            "scope_label": "今天触发" if today_only else "最近窗口",
            "refresh_seconds": REFRESH_SECONDS,
        }
        with self.lock:
            self.cache[today_only] = {"ts": time.time(), "payload": payload}
        return payload


def load_symbol_profile(symbol: str) -> dict:
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


def single_signal_payload(symbol: str, provider: DashboardDataProvider) -> dict:
    symbol = str(symbol or "").strip().upper()
    if not symbol:
        raise ValueError("请输入股票代码。")
    if not re_symbol_ok(symbol):
        raise ValueError("股票代码格式不支持。")

    profile = load_symbol_profile(symbol)
    started = time.time()
    xl = XunLongIndicator()
    original_download_daily = scan.download_daily
    original_download_4h = scan.download_4h
    scan.download_daily = provider.download_daily
    scan.download_4h = provider.download_4h
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            df = scan.scan_one_symbol(symbol, profile.get("name", ""), xl)
    finally:
        scan.download_daily = original_download_daily
        scan.download_4h = original_download_4h

    signals = []
    if df is not None and not df.empty:
        df = df.copy()
        df["signal_date"] = pd.to_datetime(df["signal_date"], errors="coerce").dt.date
        df["score"] = df.apply(scan.score_buy_signal_row, axis=1)
        df = df.sort_values(["signal_date", "signal_side", "signal_type"], ascending=[False, True, True])
        for _, r in df.iterrows():
            side = str(r.get("signal_side", "") or "").upper()
            score = number_or_none(r.get("score")) if side == "BUY" else None
            signals.append(
                {
                    "signal_date": str(r.get("signal_date", "") or ""),
                    "signal_side": side,
                    "signal_type": str(r.get("signal_type", "") or ""),
                    "model": str(r.get("model", "") or ""),
                    "score": score,
                    "close": number_or_none(r.get("close")),
                    "rsi": number_or_none(r.get("RSI")),
                    "h4_rsi": number_or_none(r.get("H4_RSI")),
                    "h4_fj": number_or_none(r.get("H4_FJ")),
                    "extra_info": str(r.get("extra_info", "") or ""),
                }
            )

    latest_buy = next((row for row in signals if row.get("signal_side") == "BUY"), None)
    latest_sell = next((row for row in signals if row.get("signal_side") == "SELL"), None)
    return {
        "symbol": symbol,
        "name": profile.get("name", ""),
        "sector": profile.get("sector", ""),
        "excluded": profile.get("excluded", False),
        "signals": signals,
        "latest_buy": latest_buy,
        "latest_sell": latest_sell,
        "provider": provider.label,
        "updated_at": datetime.now().strftime("%H:%M:%S"),
        "elapsed_seconds": round(time.time() - started, 1),
    }


def re_symbol_ok(symbol: str) -> bool:
    text = str(symbol or "").strip().upper()
    return bool(text) and len(text) <= 16 and all(ch.isalnum() or ch in ".!:-" for ch in text)


def dataframe_to_rows(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "score": number_or_none(r.get("观海买点分")),
                "symbol": str(r.get("symbol", "")),
                "sector": str(r.get("板块", "")),
                "signal_date": str(r.get("signal_date", "")),
                "signal_type": str(r.get("signal_type", "")),
                "close": number_or_none(r.get("close")),
                "rsi": number_or_none(r.get("RSI")),
                "h4_rsi": number_or_none(r.get("H4_RSI")),
                "h4_fj": number_or_none(r.get("H4_FJ")),
                "extra_info": str(r.get("extra_info", "")),
                "updated_at": str(r.get("updated_at", "")),
            }
        )
    return rows


def number_or_none(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def json_response(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def html_response(handler: BaseHTTPRequestHandler):
    body = HTML.replace("{{REFRESH_MS}}", str(REFRESH_SECONDS * 1000)).encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(cache: SignalCache):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                html_response(self)
                return
            if parsed.path == "/api/signals":
                params = parse_qs(parsed.query)
                scope = (params.get("scope") or ["today"])[0]
                today_only = scope != "recent"
                force = (params.get("force") or ["0"])[0] == "1"
                try:
                    json_response(self, cache.get(today_only=today_only, force=force))
                except Exception as exc:
                    json_response(self, {"error": str(exc)}, status=500)
                return
            if parsed.path == "/api/single":
                params = parse_qs(parsed.query)
                symbol = (params.get("symbol") or [""])[0]
                try:
                    json_response(self, single_signal_payload(symbol, cache.provider))
                except Exception as exc:
                    json_response(self, {"error": str(exc)}, status=500)
                return
            json_response(self, {"error": "not found"}, status=404)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Stock OneClick web dashboard")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    args = parser.parse_args()

    cache = SignalCache(limit=args.limit)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(cache))
    url = f"http://{args.host}:{args.port}/"
    print(f"Stock OneClick web dashboard: {url}", flush=True)
    print(f"Project: {BASE_DIR}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
