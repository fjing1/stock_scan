# Architecture & Design

This document describes how Stock OneClick is structured, how data flows through
it, and the design decisions (and accidental complexity) that shape it. For the
analytics themselves see **[SIGNAL_LOGIC.md](SIGNAL_LOGIC.md)**; for the
front-ends and tools see **[COMPONENTS.md](COMPONENTS.md)**.

---

## 1. Layered view

```
                       ┌─────────────────────────────────────────────┐
   INPUT               │  stock_input_template.xlsx                  │
   (curated universe)  │   • Sheet1_Input  : symbol                  │
                       │   • Sheet2_Classified : symbol,name,group,  │
                       │                          exchange,enable…   │
                       └───────────────┬─────────────────────────────┘
                                       │ load_input_and_meta()
                                       │ filter_scannable_universe()  (drop 00/01/市场环境)
                                       ▼
   INDICATOR     ┌───────────────────────────────────────────────────────────┐
   xunlong.py    │  XunLongIndicator.compute(df_daily, df_4h)                 │
                 │  → ~40 columns: L2_trend/pump, FJ_value, RSI, Gann_BUY_A,  │
                 │    Gann_SELL_1_confirmed, Gann_0..1, H4_Gann_0/1_birth,    │
                 │    plus many V2_*/BUY_*/SELL_* columns (mostly unemitted)  │
                 └───────────────┬───────────────────────────────────────────┘
                                 │ scan_one_symbol() picks 6 columns
                                 ▼
   ENGINE        ┌───────────────────────────────────────────────────────────┐
   scan_stocks.py│  per-symbol signals → score_buy_signal_row (观海买点分)    │
                 │  → catch-up/forced-rescan date filter                      │
                 │  → _build_followup_sheets (D0..D14 lifecycle, cycles)      │
                 │  → _build_lifecycle_tables (buy/sell observation & history)│
                 │  → build_market_context (index risk model)                 │
                 └───────────────┬───────────────────────────────────────────┘
                                 ▼
   OUTPUTS       ┌──────────────────────┬──────────────────────┬─────────────┐
                 │ scan_result_latest   │ TradingView .txt     │ HTML        │
                 │ .xlsx (6 sheets +    │ (daily, buy, pools,  │ dashboard   │
                 │ per-date sheets)     │ sector groups)       │             │
                 └──────────────────────┴──────────────────────┴─────────────┘
                                 ▲
                                 │ history/  (timestamped archive of everything)
                                 │
   LIVE LAYER    ┌───────────────────────────────────────────────────────────┐
   (separate)    │  web_dashboard / realtime_dashboard / intraday_dashboard   │
                 │  reuse scan_one_symbol but monkey-patch download_daily/4h  │
                 │  to a DashboardDataProvider (yfinance / Polygon / Alpaca)  │
                 └───────────────────────────────────────────────────────────┘
```

The system has a clean conceptual spine — **input → indicator → engine →
outputs** — with the live dashboards bolted on as an alternate front-end that
reuses the engine's per-symbol scan.

---

## 2. Module responsibilities

| Module | Lines | Role |
|--------|------:|------|
| `backend/xunlong.py` | ~860 | The indicator. Pure functions + one stateful class `XunLongIndicator`; no I/O, no network. Computes every signal column. |
| `backend/scan_stocks.py` | ~3,080 | The engine + entry point. Universe loading, data download (`download_daily`/`download_4h`), per-symbol scan (`scan_one_symbol`), scoring, lifecycle tracking, market context, and *all* Excel/TradingView/HTML output. |
| `backend/stock_oneclick_test.py` | ~345 | A research harness: re-runs the engine for the prior business day at both the **close** and a **12:00 PT intraday snapshot**, by monkey-patching the engine's downloaders with date-sliced frames. Writes `exports/stock_oneclick_test_latest.xlsx`. |
| `backend/dashboard_data.py` | ~336 | Shared live-bar provider (`DashboardDataProvider`) abstracting yfinance / Polygon / Alpaca, with in-memory caching + throttling. |
| `backend/realtime_dashboard.py` | ~381 | Tkinter full-market dashboard **and** the `scan_universe()` library reused by the web app. |
| `backend/web_dashboard.py` | ~645 | Browser dashboard (stdlib `http.server`, hand-written HTML/JS). |
| `backend/intraday_dashboard_app.py` | ~561 | Tkinter single-symbol + sector-stats dashboard; can sync a TradingView watchlist. |
| `backend/run_scan_gui.py` | ~349 | Tkinter launcher for the whole pipeline (start scan, stream log, open results, import watchlist). |
| `backend/watchlist_importer.py` | ~345 | Library: TradingView export-file / shared-list-URL → the input workbook. |
| `backend/dual_mode_scan_v1.py` | ~520 | **Shelved prototype** of the dual buy idea; not imported anywhere. |
| `backend/stock_list_10B.py` | ~134 | Standalone CLI that builds a ≥$10B-market-cap US-equity universe from the NASDAQ/NYSE directories. Output consumed manually, not at scan time. |
| `*.command` | — | macOS double-click launchers (see [COMPONENTS.md](COMPONENTS.md)). |

`scan_stocks.py` is the hub: every other runtime module imports it (`import
scan_stocks as scan`) and reuses its functions and module-level constants.

---

## 3. The daily-scan pipeline (`scan_stocks.main()`)

End to end, one run does the following (line numbers are in `scan_stocks.py`):

1. **Load** `stock_input_template.xlsx` → `Sheet1_Input` + `Sheet2_Classified` (`load_input_and_meta`, 135).
2. **Enrich metadata** for every input symbol via `yfinance` `Ticker.info` — name, exchange, sector, market cap — without overwriting the user's `group`/`note`/`enable` (`enrich_meta_with_yfinance`, 2318). Writes the workbook back.
3. **Export static TradingView lists** first (so they exist even if the scan aborts): the fixed A pool and the 17 hardcoded Chinese sector watchlists (`export_tv_a_pool`, `export_custom_watchlists_cn`).
4. **Build the scan universe**: keep `enable == 1`, then drop the "system" groups `00 …` / `01 市场环境` / `核心指数映射` (`filter_scannable_universe`, 2153). Export the full pool + dynamic per-group lists.
5. **Scan each symbol** (`scan_one_symbol`, 2598): download daily (1y, needs ≥150 rows) + 4H (90d), run `XunLongIndicator.compute`, then emit any of the six signal types that fired within the recent lookback window.
6. **Date-filter** the signals: normally a **catch-up** window (fill in any trading days missed since the last run, capped at 10 business days); or, if `STOCK_ONECLICK_RESCAN_FROM` is set, a **forced rescan** from that date (`_get_catchup_signal_dates` / `_get_forced_rescan_signal_dates`).
7. **Score** every buy row → `观海买点分` (`score_buy_signal_row`, 1862).
8. **Build the lifecycle / follow-up sheets** (`_build_followup_sheets`, 512) — one sheet per signal date with D0…D14 close & %-vs-D0 columns, cycle-locking, and retrigger tracking — for buys and sells, merging in prior `history/` results.
9. **Build the four lifecycle tables** (`_build_lifecycle_tables`, 1414): buy observation / buy history / sell observation / sell history, pairing formal buys with subsequent formal sells.
10. **Compute the market context** per date (`build_market_context`, 852): an index-basket risk score that prints a colored banner atop each date sheet.
11. **Archive** any signal batch that has completed its full 14-day track into `history/completed_14d/`.
12. **Write the workbook** `scan_result_latest.xlsx` (and a timestamped copy in `history/`): 6 fixed sheets + per-date sheets in reverse-chronological order.
13. **Export** the HTML dashboard and the daily/buy TradingView `.txt` lists.
14. **Open** the result (macOS `open`, unconditional).

Phases are logged to stdout as `阶段 N/…` and `[i/total | pct%]` lines — which
the GUI's progress bar parses by regex (a fragile but intentional contract).

---

## 4. Key design decisions & seams

### 4.1 The indicator is a "kitchen sink"; the engine is selective
`XunLongIndicator.compute()` returns roughly **40 columns**, including many fully
formed buy/sell booleans: `V1_Buy`, `DailyStrong`, `BUY_low_reset_confirmed`,
`V2_early_turn`, `V2_strong_buy`, `V2_base_breakout`, `BUY_C_confirmed`,
`BUY_B_D1FJ_low_H4_0birth`, `SHORT_A_D1FJ_weak_H4_1birth`, `SELL_profit_protect`,
`SELL_trend_break`, `A_ok`, `C_ok`, `C_rev_ok`, … But `scan_one_symbol` emits
**only six** of them (the table in [README.md](README.md#signal-types-at-a-glance)).

The rest are **computed every run but never turned into signals** by the nightly
scanner. They are the fossil record of earlier strategy iterations plus the
dashboard-oriented `BUY_B`/`SHORT_A` strategies described in the author's
`寻龙诀_GannBox_买卖点说明.md`. This is the single most important thing to
understand before modifying the system: *the production signal set is small and
lives entirely in `scan_one_symbol`'s emit block (lines 2655–2709)*, while the
indicator carries a large, mostly-dormant payload. See
[SIGNAL_LOGIC.md §6](SIGNAL_LOGIC.md#6-computed-but-not-emitted) for the full list.

### 4.2 Dashboards inject data by monkey-patching the engine
`scan_one_symbol` fetches bars by calling the *module-level* functions
`download_daily` / `download_4h`. Every dashboard exploits this: it temporarily
reassigns `scan.download_daily` and `scan.download_4h` to a
`DashboardDataProvider` method (or to closures returning pre-fetched frames),
calls `scan.scan_one_symbol`, then restores the originals in a `finally`.

This is what lets the same scan logic run against live intraday bars from a
different vendor without duplicating it — elegant, but **fragile**: any refactor
that inlines, renames, or parameterizes those two functions silently breaks all
three dashboards. It also means the dashboards reach into a number of
underscore-prefixed engine internals. See [COMPONENTS.md §1](COMPONENTS.md#1-dashboards).

### 4.3 History is an append-only audit trail
Every run writes immutable, timestamped copies of *everything* into `history/`
(`scan_result_*.xlsx`, the TradingView lists, whole snapshot directories
`tv_groups_<ts>/`, `tv_custom_<ts>/`, etc.), while the repo keeps one mutable
`_latest` copy of each for convenience. The engine reads prior `history/`
results back in to reconstruct lifecycle/follow-up state, so **`history/` is
load-bearing, not just an archive** — the 14-day tracker and the "first-seen in
14 days" statistics depend on it. This is why the directory holds thousands of
files (see [DATA_AND_OUTPUTS.md](DATA_AND_OUTPUTS.md)).

### 4.4 Two data layers, deliberately decoupled
The nightly scanner pulls its own daily/4H bars from yfinance
(`download_daily`/`download_4h`). The dashboards use a *separate*
`DashboardDataProvider`, defaulting to a different vendor (Alpaca/Polygon). Per
`dashboard_data_source.md`, the intent is to avoid the intraday dashboards
hammering the same yfinance endpoints the nightly scan relies on. The cost is
that the two layers duplicate the yfinance download/normalize logic.

### 4.5 Lifecycle "cycle locking"
A symbol that keeps re-triggering should not spawn a new tracking row every day.
`_build_followup_sheets` (512) treats the first trigger as `D0` and folds any
re-trigger within 14 trading days into the same cycle (recording it under
`retrigger_dates` / `prior_14d_signal_dates`); a trigger after the window opens a
new cycle. `_build_lifecycle_tables` pairs formal buys with the next formal sell
to produce closed "历史记录" rows and leaves unmatched ones in "观察列表".

---

## 5. Dependencies & runtime

- **Engine + test runner:** `yfinance`, `pandas`, `numpy`, `openpyxl`. Pure
  Python otherwise. Network egress to Yahoo Finance.
- **Dashboards:** standard library only (`http.server`, `tkinter`, `urllib`,
  `ssl`, `threading`, `queue`) plus `pandas`/`numpy`; `certifi` if present.
  Polygon/Alpaca paths need API keys (see [DATA_AND_OUTPUTS.md](DATA_AND_OUTPUTS.md)).
- **`stock_list_10B.py`:** also `requests` + `concurrent.futures`.
- **Platform:** macOS-centric — results auto-open with `open`, launchers are
  `.command` files, and `dashboard_data_source.md` references `/Users/ben/…`.
- There is **no `requirements.txt` inside `Stock_OneClick/`**; the repo-root one
  (`yfinance, pandas, numpy, ta, openpyxl`) is for the unrelated VCP project but
  covers the engine's needs (`ta` is not used here).

---

## 6. Known issues & tech debt

A register of concrete problems found while reading the code. None are blockers;
all are worth knowing before changing anything.

| # | Issue | Evidence | Impact |
|---|-------|----------|--------|
| 1 | **`STOCK_ONECLICK_NO_OPEN` doesn't work for the main scanner.** It's honored only by the test runner (`stock_oneclick_test.py:339`); `scan_stocks.py:3073` opens its result unconditionally. The release notes show it with `scan_stocks.py`, which is misleading. | grep | A "headless" nightly run still pops a window. No env flag suppresses it in the engine. |
| 2 | **`_archive_completed_cycles` (scan_stocks.py:706) is dead code** — never called. Live archiving is inlined in `main()` (2965–2982). | grep (only the `def`) | Two divergent archive code paths; the unused one writes a different filename (see #3). |
| 3 | **Triple naming mismatch for completed batches:** the directory is `completed_14d/`, the files written are `…共计20天数据.xlsx` (2970), and the dead helper would write `…共计14天数据.xlsx` (721). The tracking horizon is actually `TRACK_MAX_DAYS = 14`. | source | Confusing artifacts ("20天" files in a "14d" folder). Cosmetic but misleading. |
| 4 | **Provider default is inconsistent across dashboards.** `web_dashboard.py` and `intraday_dashboard_app.py` `setdefault` Alpaca at import; `realtime_dashboard.py` launched via `run_dashboard.command` sets nothing → falls back to yfinance. The same `scan_universe` can thus use different vendors depending on entry point. | dashboard_data.py:41; web:21; intraday:17 | Surprising data-source drift between front-ends. |
| 5 | **`dashboard_data_source.md` is stale.** It predates the Alpaca integration: documents a Polygon/yfinance default and a hard-coded `/Users/ben/Desktop/…` path. | doc vs code | Following it sets up the wrong provider. |
| 6 | **Scoring logic is duplicated.** `realtime_dashboard.score_signal_row` (30–76) is a byte-for-byte copy of `scan.score_buy_signal_row` (1862–1908); the other two dashboards call the engine version. | source | Score changes must be made twice or they drift. |
| 7 | **Engine ↔ GUI progress contract is a regex.** `run_scan_gui.py:286` matches the exact `[{i}/{total} | …]` / `[META …]` print formats; reformatting those `print`s silently breaks the progress bar (no error). | source | Brittle coupling. |
| 8 | **Dead/unused data-provider methods:** `DashboardDataProvider.download_1m`/`download_5m` are never called; `STOCK_DASHBOARD_CACHE_DIR` is created but never read/written (cache is memory-only). | source | Misleading surface area. |
| 9 | **Two divergent watchlist merge semantics.** "导入TV清单" (file import) is purely additive; "同步TV链接" (URL sync) is replace-and-merge and can *remove* symbols dropped from the watchlist. They also seed different default `group` labels (`98 TradingView导入` vs `00 市场环境`). | watchlist_importer.py | A URL sync can silently shrink the universe. |
| 10 | **Backup `.py` files live next to live code** (`scan_stocks_backup_*.py`, `xunlong_backup_*.py`). They're not imported but clutter the package and confuse grep. | listing | Hygiene. |
| 11 | **`stock_input_template.xlsx` is rewritten in place each run** (step 2). A `PermissionError` (e.g. the file open in Excel) is caught and the run continues from in-memory metadata, but the workbook won't reflect enrichment that run. | scan_stocks.py:2736–2741 | Mostly benign; worth knowing. |

See [SIGNAL_LOGIC.md](SIGNAL_LOGIC.md) and [COMPONENTS.md](COMPONENTS.md) for
finer-grained notes within each subsystem.
