# Stock OneClick — System Documentation

> Reverse-engineered design documentation for the `Stock_OneClick` sub-project.
> Authored from a full read of the source (`backend/*.py`) plus the existing
> Chinese release notes. Where behavior contradicts the older notes, this set of
> docs reflects **what the code actually does** and flags the discrepancy.

Stock OneClick is a self-contained, single-operator **US-equity swing-trading
scanner** built around a Python port of a TradingView "寻龙诀 / Gann Box" Pine
indicator. Once a day (typically after the US close) it:

1. reads a hand-curated watchlist from an Excel workbook,
2. downloads daily + 4-hour bars from yfinance for every enabled symbol,
3. runs the 寻龙诀 indicator and emits **six** buy/sell signal types,
4. scores each buy with a 0–100 "观海买点分" (buy score),
5. tracks every signal's forward performance for 14 trading days (lifecycle),
6. and writes a multi-sheet Excel workbook + TradingView-importable `.txt`
   lists + a lightweight HTML dashboard.

A separate family of **intraday dashboards** (one browser app, two Tkinter apps)
reuses the same scan engine but feeds it *live* intraday bars from
yfinance / Polygon / Alpaca, for watching buy points form during the session.

---

## Documentation map

| Doc | What it covers |
|-----|----------------|
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | The layered design, end-to-end data flow, module responsibilities, the key coupling seams (monkey-patching, the "kitchen-sink indicator"), and a register of dead code / tech debt. Start here for the big picture. |
| **[SIGNAL_LOGIC.md](SIGNAL_LOGIC.md)** | The analytical core: every sub-indicator inside `XunLongIndicator` (L2 swing, 分金/FJ, volume oscillator, RSI, A/C/C_rev, the Gann Box engine), the six production signal types and how they map to indicator columns, the 观海买点分 scoring formula, the 14-day lifecycle tracker, and the market-context risk model. |
| **[COMPONENTS.md](COMPONENTS.md)** | The non-engine programs: the three dashboards + shared data-provider layer, the Tkinter scan GUI, the TradingView watchlist importer, and the two standalone tools (`dual_mode_scan_v1.py`, `stock_list_10B.py`). |
| **[DATA_AND_OUTPUTS.md](DATA_AND_OUTPUTS.md)** | File layout, the input-workbook schema, the output workbook's 6 fixed sheets + per-date sheets, all TradingView `.txt` exports, the `_latest` / timestamp / date naming conventions, and the full environment-variable + configuration reference. |
| **[IMPROVEMENTS.md](IMPROVEMENTS.md)** | Changelog of the 2026-06-06 reliability / performance / architecture changes, the new test suite + backtest tool, and the (deferred) monolith-split plan. |

The original author docs are kept alongside the code:
`../RELEASE_2026_06_V1.md` (release notes) and
`../exports/寻龙诀_GannBox_买卖点说明.md` (the indicator's own buy/sell spec).

---

## Quick start

```bash
# 1. Dependencies (no requirements.txt ships inside Stock_OneClick; the engine needs):
python3 -m pip install yfinance pandas numpy openpyxl
#    Dashboards additionally use only the Python standard library
#    (http.server, tkinter, urllib). Polygon/Alpaca providers use urllib + certifi.

cd Stock_OneClick/backend

# 2. Run a full daily scan (writes ../scan_result_latest.xlsx and opens it on macOS)
python3 scan_stocks.py
#    or double-click run_scan.command

# 3. Or drive it from the GUI (start scan, watch progress, import watchlists, open results)
python3 run_scan_gui.py

# 4. Intraday dashboards (live bars, not the nightly Excel):
python3 web_dashboard.py            # browser app at http://127.0.0.1:8765
python3 realtime_dashboard.py       # full-market Tkinter table
python3 intraday_dashboard_app.py   # single-symbol + sector-stats Tkinter app

# 5. Dev tools (use a venv with pandas/numpy/openpyxl, e.g. the repo's vcp_env):
../../vcp_env/bin/python tests/test_engine.py            # indicator + scoring tests
../../vcp_env/bin/python backtest_score.py --source both # 观海买点分 vs. forward-return calibration
../../vcp_env/bin/python make_report.py                  # write dated Markdown reports to ../reports/
```

Useful environment toggles (see [DATA_AND_OUTPUTS.md](DATA_AND_OUTPUTS.md) for the full table):

```bash
# Rebuild signal history from a start date (forced rescan)
STOCK_ONECLICK_RESCAN_FROM=2026-05-22 python3 scan_stocks.py

# Choose the dashboards' live data provider
export STOCK_DASHBOARD_DATA_PROVIDER=alpaca   # yfinance | polygon | alpaca
```

> `STOCK_ONECLICK_NO_OPEN=1` suppresses the macOS auto-open. As of 2026-06-06 it
> is honored by **both** `scan_stocks.py` and the test runner (it was previously
> test-runner only). See [IMPROVEMENTS.md](IMPROVEMENTS.md).

---

## Glossary

These terms recur throughout the code, the Excel output, and these docs.

| Term | Meaning |
|------|---------|
| **寻龙诀 (XunLongJue)** | The name of the source TradingView indicator; ported to Python as `XunLongIndicator` in `xunlong.py`. |
| **Gann Box** | The indicator's structural framing of one up-leg: a low (`0`), a high (`1`), and Fibonacci levels 0.382 / 0.5 / 0.618 in between. |
| **0出 / 1出 ("zero birth" / "one birth")** | The *day the system confirms* a Gann `0` (leg start, EMA10 turns up) or `1` (leg top, EMA10 turns down after a valid rise). Distinct from where the `0`/`1` price is drawn. |
| **分金 / FJ (`FJ_value`)** | A smoothed RSI-like momentum oscillator (0–100). Low = washed-out/repairing, high = overheated. Drives sell-side and several gating conditions. |
| **L2_trend / L2_pump** | A KDJ-style swing oscillator pair; `L2_trend==0` marks a low/quiet state used as a buy precondition. |
| **rank120 / Rank120** | Where today's close sits in its trailing 120-day range, `0`=range low … `1`=range high. The primary "is this low in its base?" gauge. |
| **观海买点分 (buy_score)** | 0–100 composite score for buy signals (`score_buy_signal_row`). Higher = more attractive. |
| **D0 … D14** | A signal's anchor day (`D0`) and the 1–14 trading days of forward tracking after it. |
| **板块 (sector / group)** | The user-assigned sector bucket (`group` column), normalized to a numbered label like `08 核能`. |
| **lifecycle** | The pairing of formal buys with later formal sells (and vice versa) into open ("观察") and closed ("历史") records. |
| **A pool** | A small fixed list of 23 high-liquidity leaders (`A_POOL_SYMBOLS`) exported as a convenience TradingView list. |

## Signal types at a glance

| 信号 (signal_type) | Side | Trigger (indicator column) | Meaning |
|--------------------|------|----------------------------|---------|
| 第一买入点 | BUY | `LOW_START_FIRST_BUY` | First green bar from a low base — *preliminary watch only*. |
| 二进宫买入点 | BUY | `LOW_START_SECOND_BUY` | Pullback held structure, second green bar confirms. |
| 预警买入 (warning buy) | BUY | `H4_Gann_0_birth_daily` | 4-hour Gann `0出` — early rebound structure. |
| 正式买入 (formal buy) | BUY | `Gann_BUY_A` | Daily Gann BUY A / `0出` — the primary buy. |
| 预警卖出 (warning sell) | SELL | `H4_Gann_1_birth_daily` | 4-hour Gann `1出` — short-cycle top. |
| 正式卖出 (formal sell) | SELL | `Gann_SELL_1_confirmed` | Daily Gann `1出` confirmed — the primary sell. |

Full definitions, formulas, and the (large) set of *computed-but-unemitted*
columns are in **[SIGNAL_LOGIC.md](SIGNAL_LOGIC.md)**.
