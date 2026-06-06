# Components: Dashboards, GUI & Tools

Everything that is *not* the scan engine or the indicator. These programs wrap,
launch, feed, or visualize `scan_stocks.py`. Line references are within each
file's own module. For the engine itself see [ARCHITECTURE.md](ARCHITECTURE.md);
for file formats see [DATA_AND_OUTPUTS.md](DATA_AND_OUTPUTS.md).

---

## 1. Dashboards

Three independent front-ends share one idea: **reuse `scan.scan_one_symbol` but
feed it live intraday bars** instead of letting it pull end-of-day yfinance data.
They do this by monkey-patching the engine's module-level downloaders (see
[ARCHITECTURE.md §4.2](ARCHITECTURE.md#42-dashboards-inject-data-by-monkey-patching-the-engine)):

```python
scan.download_daily = provider.download_daily   # (or a closure over prefetched frames)
scan.download_4h    = provider.download_4h
try:
    df_sig = scan.scan_one_symbol(sym, name, xl)
finally:
    scan.download_daily, scan.download_4h = old_daily, old_4h
```

| File | UI tech | View | Launch | Default provider |
|------|---------|------|--------|------------------|
| `web_dashboard.py` | stdlib `http.server` + hand-written HTML/JS/CSS (no framework, no chart lib) | Browser "买点雷达": full-market score cards **+** single-symbol table | `run_web_dashboard.command`, port **8765** | Alpaca (forced at import) |
| `realtime_dashboard.py` | Tkinter `ttk.Treeview` | Full-market BUY table, auto-refresh | `run_dashboard.command` | **yfinance** (launcher sets nothing) |
| `intraday_dashboard_app.py` | Tkinter (cards + trees) | Single-symbol **+** buy-sector Top-5 stats + TV sync | `run_intraday_dashboard.command` | Alpaca (forced at import) |

### 1.1 `dashboard_data.py` — shared data-provider layer
`DashboardDataProvider` abstracts three back-ends behind `download_daily` /
`download_4h` (and unused `download_1m`/`download_5m`):
- **yfinance** (`_yf_download`, 150) — duplicates the engine's own download/normalize.
- **Polygon** (`_polygon_aggs`, 158) — `api.polygon.io/v2/aggs`, needs `POLYGON_API_KEY`.
- **Alpaca** (`_alpaca_bars`/`_alpaca_bars_many`, 196/239) — IEX feed, needs key+secret; supports a **multi-symbol batch** path used by the intraday app.

Cross-cutting behavior: in-memory cache keyed `(symbol, interval, period)` with a
`STOCK_DASHBOARD_CACHE_TTL_SECONDS=240` freshness window, a
`STOCK_DASHBOARD_REQUEST_DELAY_SECONDS=0.15` sleep after each uncached fetch, and
vendor-specific ticker normalization (`BRK.B`↔`BRK-B`). At import it loads
`backend/dashboard_env.local` (gitignored; template `dashboard_env.example`)
without overriding already-set vars. Note: `STOCK_DASHBOARD_CACHE_DIR` is created
but never read/written — the cache is memory-only.

### 1.2 `realtime_dashboard.py`
The heart is `scan_universe()` (109): load the enabled universe from the input
workbook, patch the downloaders, loop `scan_one_symbol`, score with a **local
copy** of the buy-score function (`score_signal_row`, 30 — a duplicate of
`scan.score_buy_signal_row`), keep today's BUY rows, sort. The Tk app
(`RealtimeDashboard`, 185) runs the scan on a background thread, marshals results
through a `queue.Queue` polled by `root.after`, color-codes buy/sell/error rows,
and re-arms a `STOCK_DASHBOARD_REFRESH_SECONDS=300` timer. `--once` runs it
head-less in the terminal. It reads no scan-result Excel — purely live.

### 1.3 `web_dashboard.py`
A `ThreadingHTTPServer` with three JSON routes and one HTML page (a single
hard-coded string, lines 35–417; scores rendered as CSS bars):
- `GET /` → the page (injects the refresh interval).
- `GET /api/signals?scope=today|recent[&force=1]` → `SignalCache.get()` →
  **`realtime_dashboard.scan_universe()`** (the web app reuses realtime's scan
  wholesale) → JSON cards. Cached per-scope for 300 s.
- `GET /api/single?symbol=XYZ` → `single_signal_payload`: live daily+4H for one
  symbol, scored with `scan.score_buy_signal_row`, returned as the full recent
  signal list + latest buy/sell.

The full-market view refreshes only on button click; only the single-symbol view
auto-refreshes client-side.

### 1.4 `intraday_dashboard_app.py`
Two independent paths:
- **Single symbol** (`scan_single_symbol`, 100): prefetch one symbol's daily+4H
  once, patch the engine with closures over those frames, scan, score, show
  cards + a full signal table; "打开TV图" opens the TradingView chart.
- **Sector stats** (`scan_buy_sector_stats`, 198): the *only* dashboard path that
  reads the nightly Excel — it loads `scan_result_latest.xlsx` + the last 20
  `history/scan_result_*.xlsx` to flag "first-seen-in-14-days" buys, batch-fetches
  the whole universe via the Alpaca multi endpoint, and renders buy-sector Top-5
  (all + first-seen).

It can also **write**: "同步TV清单" calls `watchlist_importer.sync_watchlist_url`
to refresh the input workbook from a TradingView shared list.

> Duplication to be aware of (see [ARCHITECTURE.md §6](ARCHITECTURE.md#6-known-issues--tech-debt)):
> the symbol-profile loader, the symbol validator, and the single-symbol scan are
> each implemented twice (web vs intraday); the buy-score function exists in three
> places.

---

## 2. `run_scan_gui.py` — the operator GUI

A Tkinter launcher (`ScanGUI`, 32) — the primary human surface; `run_scan.command`
is the no-GUI equivalent. Buttons:

| Button | Action |
|--------|--------|
| 开始扫描 | `Popen(["python3","-u","scan_stocks.py"])`, cwd=`backend/`, streams stdout to the log pane |
| 测试扫描 | same for `stock_oneclick_test.py` (close + 12:00 PT snapshots) |
| 打开结果文件夹 / 打开最新结果 / 打开测试结果 / 打开A池文件 / 打开买入TXT | `open` the relevant artifact |
| 导入TV清单 | file picker → `watchlist_importer.import_watchlist_file` |
| 同步TV链接 | URL prompt (prefilled) → `watchlist_importer.sync_watchlist_url` |
| 日内Dashboard | `Popen` the intraday app (sets `STOCK_DASHBOARD_DATA_PROVIDER` default Alpaca) |
| 市场判断说明 | shows `scan.MARKET_CONTEXT_HELP_TEXT` |

Progress is driven by a **regex** (`\[…(\d+)/(\d+)…\]`, line 286) over the scan's
stdout, matched against the engine's `[{i}/{total} | …]` and `[META …]` print
formats — reformatting those breaks the bar silently. Notes:
- The scan subprocess is launched with the **GUI's unmodified environment** — the
  GUI does **not** set `STOCK_ONECLICK_RESCAN_FROM` or `STOCK_ONECLICK_NO_OPEN`;
  to force a rescan you must `export` it in the shell before launching.
- `import scan_stocks` is done at startup just to read one help string, which
  drags the full pandas/numpy/yfinance/openpyxl stack into GUI launch.
- The GUI uses `sys.executable`; `run_scan.command` hard-codes
  `/usr/local/bin/python3` — they can resolve to different interpreters.

---

## 3. `watchlist_importer.py` — TradingView → input workbook

A pure library (no CLI), called only from the GUI / intraday app. Two paths, with
**different merge semantics**:

| Function | Input | Merge behavior | Default `group` |
|----------|-------|----------------|-----------------|
| `import_watchlist_file(path,…)` (190) | a TradingView-exported `.txt`/`.csv` | **Additive** — appends only new symbols; backfills blank exchanges; never removes | `98 TradingView导入` |
| `sync_watchlist_url(url,…)` (287) | a shared-watchlist URL | **Replace-and-merge** — rebuilds `Sheet1_Input` to exactly the watchlist (can *remove* symbols), preserving prior per-symbol metadata by join | `00 市场环境` |

Parsing: lines/tokens split on `[\s,;]+`, header words skipped, `EXCHANGE:SYMBOL`
split and the exchange normalized via `EXCHANGE_ALIASES`. **Sector grouping is
encoded by `###`-prefixed marker lines** (`_parse_group_marker`, 57): e.g.
`### 01 市场环境` flips the `current_group` applied to following symbols. The
values are free text passed through verbatim to the workbook's `group` column —
the importer does **not** special-case `00`/`01`; the *scanner* is what later
excludes those groups from the scan universe.

URL fetch (`fetch_tradingview_watchlist`, 257) scrapes TradingView's embedded
`application/prs.init-data+json` blocks (with an unverified-SSL retry) and reads
`sharedWatchlist.list` — brittle to site markup changes. Every write is preceded
by a backup to `history/stock_input_template_before_<label>_<ts>.xlsx`; URL sync
also drops `exports/tv_watchlist_from_url_latest.txt`.

> The `tv_custom/` and `tv_A_pool.txt` lists are **not** produced here — they're
> written by `scan_stocks.py` from the hardcoded `CUSTOM_WATCHLISTS_CN` /
> `A_POOL_SYMBOLS`. This importer only writes the workbook + the URL snapshot.

---

## 4. Standalone tools (not in the runtime path)

### 4.1 `dual_mode_scan_v1.py` — shelved prototype
A self-contained module implementing two detectors — **低位启动** (low-base start)
and **二进宫反弹** (Fibonacci-retrace rebound, with ATR/EMA "impulse-high lock")
— with its own re-implemented indicators (`xsa`, `compute_bbuy`, `compute_l2_t_p`)
and Markdown report renderers. **No other file imports it**, and its `__main__`
only prints "wire up your daily/4h data and call …". It is an earlier, parallel
experiment of the dual-buy idea that the production engine superseded (the `v1`
suffix and the duplicated math reinforce this). Documented for completeness; safe
to ignore for understanding the live system.

### 4.2 `stock_list_10B.py` — universe builder CLI
An offline tool (argparse) that *builds* a large-cap universe rather than storing
one: it downloads the NASDAQ/NYSE/AMEX symbol directories from
`ftp.nasdaqtrader.com`, fans out `yfinance` `Ticker.get_info()` across a thread
pool, drops ETFs, and keeps `marketCap ≥ --min-mcap` (default **$10 B** → the
"10B" name). Output is a 4-column `symbol,name,exchange,market_cap` `.xlsx`/`.csv`
sorted by market cap. **Not imported by anything** — its output is meant to be
used manually to seed/expand `stock_input_template.xlsx`.

```bash
python3 stock_list_10B.py --out stock_list_10B.xlsx --min-mcap 10000000000 --workers 16
```

---

## 5. Launchers (`.command` files, macOS)

| File | Runs | Notes |
|------|------|-------|
| `run_scan.command` | `scan_stocks.py` | `#!/bin/zsh`; hard-codes `/usr/local/bin/python3`; pauses on a `read` so the Terminal stays open |
| `run_scan_gui.py` | (the GUI itself) | not a `.command`; run with `python3` |
| `run_dashboard.command` | `realtime_dashboard.py` | sets **no** provider → yfinance |
| `run_web_dashboard.command` | `web_dashboard.py` | exports provider (default Alpaca) |
| `run_intraday_dashboard.command` | `intraday_dashboard_app.py` | `source`s `dashboard_env.local` for keys, exports provider (default Alpaca) |

All `cd "$(dirname "$0")"` so the working directory is `backend/`, which the
engine's relative paths (`BASE_DIR = parent of backend/`) depend on.
