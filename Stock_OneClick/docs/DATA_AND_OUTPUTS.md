# Data, Outputs & Configuration

Where everything lives, the schemas of the files the system reads and writes, the
naming/archival conventions, and the complete environment-variable reference. For
how these files are produced see [ARCHITECTURE.md](ARCHITECTURE.md); for column
meanings see [SIGNAL_LOGIC.md](SIGNAL_LOGIC.md).

---

## 1. Directory layout

```
Stock_OneClick/
├── stock_input_template.xlsx     # INPUT watchlist (the universe the scanner reads)
├── scan_result_latest.xlsx       # MAIN output (6 fixed sheets + per-date sheets)
├── RELEASE_2026_06_V1.md         # author release notes
├── docs/                         # ← this documentation set
├── backend/                      # all code (engine, indicator, dashboards, tools)
│   ├── stock_output_template.xlsx   # stale sample skeleton, not used at runtime
│   └── dashboard_env.example        # template → copy to dashboard_env.local (gitignored)
├── exports/                      # current "_latest" copies + static TV pools
│   ├── tv_custom/                # 17 static sector pools + _index.txt
│   ├── tv_groups/                # dynamic per-scan sector pools (drift over time)
│   ├── full_scan_pool.txt, tv_A_pool.txt, tv_watchlists_merged_cn.txt
│   ├── xl_signal_dashboard_latest.html
│   └── 寻龙诀_GannBox_买卖点说明.{md,txt} (+ indicator-source .zip)
├── tv_daily_signals/             # per-day BUY+SELL trigger symbols (LIVE dir)
├── tv_buy_signals/               # per-day BUY-only triggers (pure + notes)
└── history/                      # append-only timestamped archive of EVERYTHING
    └── completed_14d/            # graduated 14-day batches (…共计20天数据.xlsx)
```

Path constants (`scan_stocks.py:25–29`): `BASE_DIR` = the `Stock_OneClick/` root
(parent of `backend/`), `INPUT_FILE` = `stock_input_template.xlsx`, `EXPORT_DIR` =
`exports/`, `HISTORY_DIR` = `history/`, `COMPLETED_20D_DIR` =
`history/completed_14d/`.

> The repo-root `tv_daily_signals/` is the **live** one the engine writes
> (`history_dir.parent / "tv_daily_signals"`); the same-named folder under
> `exports/` is **stale** (old March files) — don't confuse them.

> Scale: `history/` holds **thousands** of files (~28 MB) because every run
> archives full copies + whole snapshot directories. It is load-bearing (the
> lifecycle tracker reads it back), not disposable. See
> [the push note below](#5-version-control-note).

---

## 2. Input workbook — `stock_input_template.xlsx`

Two sheets (schema authoritative from `scan_stocks.py:135–144` and
`watchlist_importer.py`).

**`Sheet1_Input`** — one column:

| Column | Meaning |
|--------|---------|
| `symbol` | Ticker, upper-cased/stripped, no exchange prefix (e.g. `NVDA`). The raw scan list. |

**`Sheet2_Classified`** — the enriched metadata, columns in this exact order:

| Column | Meaning |
|--------|---------|
| `symbol` | Ticker (join key). |
| `name` | Company name (yfinance; may be blank for TV imports). |
| `exchange` | Exchange code → TradingView prefix (`NASDAQ`/`NYSE`/`AMEX`/…). |
| `sector` / `industry` / `market_cap` | yfinance enrichment (optional). |
| `group` | **User sector bucket**, numeric-prefixed for ordering, e.g. `08 核能`, `11 AI软件 / SaaS`. Drives the `板块` column and per-group exports. Groups `00 …` / `01 市场环境` / `核心指数映射` are **excluded from scanning** (kept for context only). |
| `note` | Free text. |
| `enable` | **0/1**; the scanner runs only `enable == 1` rows. |

The scanner rewrites this workbook each run to refresh `name/sector/market_cap`,
**preserving** the user's `group`/`note`/`enable`.

---

## 3. Output workbook — `scan_result_latest.xlsx`

Written to `history/scan_result_<YYYYMMDD_HHMMSS>.xlsx`, then copied to the root
`scan_result_latest.xlsx`. **Six fixed sheets, then N per-date sheets** (newest
first):

| # | Sheet | Contents |
|---|-------|----------|
| 1 | `Summary` | Two stacked sections (买入跟踪 / 卖出跟踪, 14-day) + a 板块 Top-5 block; Top-5 sector rows highlighted. |
| 2 | `RawSignals` | Flat per-signal table — the most data-dense sheet. Columns: `run_date, run_time, symbol, name, 板块, signal_date, signal_type, signal_side, model, close, volume, vol_ma20, L2_trend, L2_pump, RSI, rank120, H4_RSI, H4_FJ, H4_0_birth, H4_1_birth, Gann_1_date, Gann_1_price, buy_score, extra_info`. |
| 3–4 | `买入观察列表` / `买入历史记录` | Open / closed BUY lifecycle ([SIGNAL_LOGIC §8.2](SIGNAL_LOGIC.md#82-lifecycle-tables-_build_lifecycle_tables-scan_stockspy1414)). |
| 5–6 | `卖出观察列表` / `卖出历史记录` | Open / closed SELL lifecycle. |
| 7+ | `YYYY-MM-DD` (one per signal date) | A market-context banner + buy/sell follow-up tracking (or a snapshot for dates without follow-up data). |

**Per-date follow-up columns:** `symbol, 观海买点分, 板块, D0_date, D0_rule,
D0_close, prior_14d_signal_dates`, then for each tracked day `D1…D14` a close
column **renamed to embed the date** (`Dᵢ_YYYY-MM-DD`) plus `Dᵢ_pct_vs_D0`
(formatted `0.00%`, green if >0 / red if <0), and finally `retrigger_dates`.

**Market-context banner** (top of each date sheet): six labeled rows — `市场环境 /
日线判断 / 4H提示 / 轮动判断 / 指数快照 / 策略提示` — cell-filled by risk state
(red 看跌 / amber 风险 / light 谨慎 / green 强势). See
[SIGNAL_LOGIC §10](SIGNAL_LOGIC.md#10-market-context-risk-model-build_market_context-scan_stockspy852).

**Completed-batch archive:** when a date's batch fills all 14 follow-up days it's
written once to `history/completed_14d/<YYYYMMDD>共计20天数据.xlsx` and never
overwritten. (Naming quirk: dir says `14d`, files say `20天`, the horizon is
actually 14 — see [ARCHITECTURE.md §6 #3](ARCHITECTURE.md#6-known-issues--tech-debt).)

**Test-runner output** (`exports/stock_oneclick_test_latest.xlsx`): exactly two
sheets — `{prev_business_day} 收盘价` and `{…} 12点盘中` — same follow-up layout,
comparing prior-close vs a 12:00-PT intraday snapshot.

---

## 4. TradingView & HTML exports

All `.txt` lists use one encoder, `build_tv_symbol(symbol, exchange)`: known
exchanges → `EXCHANGE:SYMBOL`; unknown → bare ticker (so some lines are
prefix-less). The reverse map `to_yfinance_symbol` handles index/futures specials
(`SPX→^GSPC`, `VIX→^VIX`, `DXY→DX-Y.NYB`, `CL1!→CL=F`, `HG1!→HG=F`).

| Artifact | Producer | Format / purpose |
|----------|----------|------------------|
| `tv_daily_signals/tv_today_<date>.txt` (+ `_latest`) | `export_tradingview_lists` | One `EXCHANGE:SYMBOL` per line — the day's D0 (first-trigger) symbols, BUY+SELL. Empty day → `# No signals today`. |
| `tv_buy_signals/tv_buy_today_<date>.txt` (+ `_latest`) | `export_tv_buy_signal_notes` | **Pure import** version — BUY-only symbols, sorted by 观海买点分 desc. |
| `tv_buy_signals/tv_buy_today_notes_<date>.txt` (+ `_latest`) | same | **Notes** version — same symbols + a header and `code \| 日期 \| 观海买点分 \| 规则 \| 板块 \| D0_close` per line (the rule's `\|` becomes `+`). TradingView ignores `#` lines, imports the rest. |
| `exports/full_scan_pool.txt` | `export_full_scan_pool` | The entire enabled scan universe. |
| `exports/tv_A_pool.txt` | `export_tv_a_pool` | The fixed 23-symbol "A pool" leaders. |
| `exports/tv_custom/NN_*.txt` + `_index.txt` | `export_custom_watchlists_cn` | 17 **static** sector pools from `CUSTOM_WATCHLISTS_CN`. |
| `exports/tv_groups/NN_*.txt` + `_index.txt` | `export_tv_group_lists` | **Dynamic** sector pools rebuilt from the live `group` column each run. |
| `exports/tv_watchlists_merged_cn.txt` | `export_custom_watchlists_cn` | The 17 pools merged into one paste-section file. |
| `exports/xl_signal_dashboard_latest.html` | `export_signal_dashboard` | A standalone styled HTML table of current BUY/SELL signals. |

On content change, the pool exporters back up the previous version to
`history/<name>_prev_<ts>.txt` before overwriting, and always drop a
`history/<name>_<ts>.txt` snapshot.

### Naming & archival conventions
- **`_latest`** — the single mutable "current" copy you open. For the xlsx it's a
  byte copy of the newest timestamped archive.
- **`_YYYYMMDD_HHMMSS`** (run timestamp) — an immutable per-run archive in
  `history/`, including whole snapshot directories `tv_groups_<ts>/`,
  `tv_custom_<ts>/`.
- **`_YYYY-MM-DD`** (calendar date) — a per-trading-day file in the live signal
  dirs; re-running the same day overwrites that day's file.

Net: `history/` is a complete append-only audit trail; the root / `exports/`
always hold one friendly `_latest` (and per-day) copy.

---

## 5. Configuration reference

### 5.1 Engine constants (`scan_stocks.py`, top of file)
| Constant | Value | Meaning |
|----------|-------|---------|
| `LIFECYCLE_START_DATE` | `2026-05-22` | Signals before this date are ignored everywhere. The system's "epoch." |
| `TRACK_MAX_DAYS` | `14` | Follow-up / lifecycle horizon in trading days. |
| `V1_LOOKBACK_DAYS` / `V2_LOOKBACK_DAYS` | `5` / `5` | Recent-window components. |
| `GANN_LOOKBACK_DAYS` | `10` | Recent emit window (the max of the three) + catch-up cap. |
| `V2_MAX_RANK120` | `0.4` | Low-position filter for the (dormant) V2 strong model. |
| `A_POOL_SYMBOLS` | 23 tickers | The fixed "A pool" leaders list. |
| `CUSTOM_WATCHLISTS_CN` | 17 groups | Static sector→symbols map for `tv_custom/`. |
| `MARKET_CONTEXT_SYMBOLS` | 8 tickers | SPX/QQQ/IWM/RSP/VIX/HYG/SMH/XLU risk basket. |

Indicator parameters are constructor args of `XunLongIndicator`
(`xunlong.py:93`) — notably `gann_ema_len=10`, `gann_min_ema_up_bars=3`,
`gann_min_gain_pct=0.08`, `gann_buy_a_rsi_min=35`, `rsi_len=14`, `K=9`, `D=3`.

### 5.2 Environment variables
| Variable | Read by | Default | Effect |
|----------|---------|---------|--------|
| `STOCK_ONECLICK_RESCAN_FROM` | `scan_stocks.py:501` | — | `YYYY-MM-DD`: force-rebuild signal history from this date (clamped to `LIFECYCLE_START_DATE`) to the run date, instead of the normal catch-up. |
| `STOCK_ONECLICK_NO_OPEN` | `scan_stocks.py` `main()` + `stock_oneclick_test.py:339` | — | `=1` suppresses the macOS auto-open. As of 2026-06-06 honored by **both** the scanner and the test runner (previously test-runner only). See [IMPROVEMENTS.md](IMPROVEMENTS.md). |
| `STOCK_ONECLICK_REFRESH_META` | `scan_stocks.py` `enrich_meta_with_yfinance` | — | `=1` forces a full yfinance metadata refresh. Default: only new/incomplete symbols are fetched (avoids re-hitting the whole universe every run). |
| `STOCK_ONECLICK_META_WORKERS` | `scan_stocks.py` `enrich_meta_with_yfinance` | `8` | Thread-pool size for parallel metadata fetches. |
| `STOCK_ONECLICK_DOWNLOAD_WORKERS` | `scan_stocks.py` `prefetch_bars` | `8` | Thread-pool size for the parallel pre-scan bar prefetch. |
| `STOCK_DASHBOARD_DATA_PROVIDER` | dashboards | `yfinance` (lib) / `alpaca` (web & intraday force) | `yfinance` \| `polygon` \| `alpaca`. |
| `STOCK_DASHBOARD_REFRESH_SECONDS` | dashboards | `300` | Auto-refresh interval / web cache TTL. |
| `STOCK_DASHBOARD_LIMIT` | web, realtime | `0` (all) | Cap tickers scanned (not honored by the intraday app). |
| `STOCK_DASHBOARD_CACHE_TTL_SECONDS` | `dashboard_data.py` | `240` | Live-bar memory-cache freshness. |
| `STOCK_DASHBOARD_REQUEST_DELAY_SECONDS` | `dashboard_data.py` | `0.15` | Sleep after each uncached fetch. |
| `STOCK_DASHBOARD_CACHE_DIR` | `dashboard_data.py` | `<repo>/.dashboard_cache` | Directory created but **never used** (cache is memory-only). |
| `STOCK_WEB_DASHBOARD_HOST` / `_PORT` | `web_dashboard.py` | `127.0.0.1` / `8765` | HTTP bind. |
| `POLYGON_API_KEY` (or `STOCK_DASHBOARD_POLYGON_API_KEY`) | `dashboard_data.py` | — | Polygon auth. |
| `ALPACA_API_KEY`/`APCA_API_KEY_ID` + `ALPACA_SECRET_KEY`/`APCA_API_SECRET_KEY` | `dashboard_data.py` | — | Alpaca auth (also `STOCK_DASHBOARD_ALPACA_*` aliases). |

Dashboard secrets are conventionally placed in `backend/dashboard_env.local`
(copied from `dashboard_env.example`), which `dashboard_data.py` loads at import.

---

## 6. Version-control note

`Stock_OneClick/` is currently **untracked**. Its own `.gitignore` only excludes
`backend/dashboard_env.local` and `.dashboard_cache/`, so a naïve `git add
Stock_OneClick/` would also commit the entire `history/` archive (thousands of
files, ~28 MB of generated `.xlsx`/`.txt`) plus the `exports/` outputs and the
backup `.py` files. Before committing, decide whether to track **source + docs
only** (recommended — add `history/`, `exports/` outputs, `*_backup_*.py`, and the
result workbooks to `.gitignore`) or the directory as-is. See
[ARCHITECTURE.md §6](ARCHITECTURE.md#6-known-issues--tech-debt) for the related
hygiene items.
