# Improvements — 2026-06-06

Changes made on top of the reverse-engineered baseline, in response to the
review in [ARCHITECTURE.md §6](ARCHITECTURE.md#6-known-issues--tech-debt). Every
change is backward-compatible (no public function signature or output-schema
break) and was verified with `vcp_env/bin/python` — the indicator/scoring math
is now pinned by `backend/tests/test_engine.py` (**14 tests, all passing**, one
auto-skipped where the headless venv lacks `tkinter`).

> Scope note: a physical split of the 3,080-line `scan_stocks.py` into packages
> was **deliberately not done** — see [§3](#3-deliberately-deferred-the-monolith-split).

---

## 1. Correctness & reliability

| Change | File | Detail |
|--------|------|--------|
| **`STOCK_ONECLICK_NO_OPEN` now works in the scanner** | `scan_stocks.py` `main()` | The final macOS `open` is now gated on the env var (previously unconditional — only the test runner honored it). Enables true headless/cron runs. |
| **Failed scans are surfaced, not swallowed** | `scan_stocks.py` `main()` | Per-symbol exceptions are collected and printed as an end-of-run summary (`⚠️ N 只标的扫描失败 …`), so a data outage no longer looks identical to "no signal." |
| **Dead-code logic bug cleaned** | `xunlong.py` | `d1_fj_weak` was `(A) | (A) & (B)` which by precedence reduces to `A` — the second clause never affected the result. Rewritten to the explicit effective condition with a comment + the intended stricter form available. (Behavior unchanged; the column feeds the dormant `SHORT_A`.) |
| **More robust `enable` coercion** | `scan_stocks.py` `enrich_meta_with_yfinance` | `pd.to_numeric(...).fillna(1).astype(int)` instead of `.fillna(1).astype(int)`, which threw on non-numeric cells. |

## 2. Performance & network hygiene

| Change | File | Detail |
|--------|------|--------|
| **Metadata enrichment no longer re-hits yfinance for the whole universe every run** | `scan_stocks.py` `enrich_meta_with_yfinance` | Now only fetches **new or incomplete** symbols (missing `name`/`exchange`) by default; `STOCK_ONECLICK_REFRESH_META=1` forces a full refresh. This was the biggest rate-limit/ban risk (`.info` is yfinance's flakiest endpoint). |
| **Metadata fetches run in parallel** | same | `ThreadPoolExecutor` (`STOCK_ONECLICK_META_WORKERS`, default 8) instead of a serial per-symbol loop. |
| **In-run bar cache** | `scan_stocks.py` | `download_daily`/`download_4h` memoize within a run (keyed `kind+symbol+period`), returning a defensive copy. The scan pass, the follow-up tracker, and the market-context basket no longer refetch the same symbol. Dashboards are unaffected (they replace these functions wholesale). |
| **Parallel universe prefetch** | `scan_stocks.py` `prefetch_bars` + `main()` | The whole scan universe's daily+4H bars are fetched concurrently (`STOCK_ONECLICK_DOWNLOAD_WORKERS`, default 8) to warm the cache before the serial scan loop — the dominant wall-clock cost previously. |

## 3. Architecture & maintainability

| Change | File | Detail |
|--------|------|--------|
| **Injectable data fetchers (DI seam)** | `scan_stocks.py` `scan_one_symbol(..., daily_fetcher=None, h4_fetcher=None)` | A clean way to supply data without the global monkey-patch. Defaults resolve to the module downloaders **at call time**, so existing dashboard/test monkey-patching still works unchanged. Tests use this to run the full scan offline. |
| **Scoring de-duplicated** | `realtime_dashboard.py` | `score_signal_row` was a byte-for-byte copy of `scan.score_buy_signal_row`; it now delegates, removing the drift risk (the score lived in 3 places; the dashboards now all use the engine's). |

### Deliberately deferred: the monolith split
Splitting `scan_stocks.py` into `io/ signals/ output/ lifecycle.py market.py` is the
right end state, but it is **high-risk and not integration-testable here** (no live
network / Excel input / macOS), and it would threaten the deep `import scan_stocks`
coupling that all four dashboards rely on (many reach into `_`-prefixed internals).
Recommended safe sequence when you do tackle it:
1. First migrate the dashboards off monkey-patching onto the new `daily_fetcher`/
   `h4_fetcher` parameters (already available) — removes the global-mutation coupling.
2. Extract the pure Excel/TradingView/HTML writers (no engine state) into
   `output/` — lowest-risk, ~700 lines.
3. Extract `build_market_context` + helpers into `market.py`.
4. Extract the lifecycle/follow-up builders into `lifecycle.py`.
5. Keep `scan_stocks.py` as a thin façade re-exporting the public names so existing
   imports keep working; delete the façade only after all callers migrate.
Each step is independently shippable and testable.

## 4. New tooling

| Tool | What it does |
|------|--------------|
| **`backend/tests/test_engine.py`** | Network-free unit/regression suite: helper math (`safe_div`/`rma`/`xsa`), the exact 观海买点分 formula, Gann `0出/1出` structural invariants (a ≥8% leg yields `0` then `1`; a <8% wiggle yields none; a confirmed sell requires a prior buy in the segment), a `compute()` smoke test, the injectable-fetcher seam, and the in-run cache. Run: `../../vcp_env/bin/python tests/test_engine.py`. |
| **`backend/backtest_score.py`** | Closes the analytics loop: mines the D0→D14 forward returns already archived in the per-date sheets / `completed_14d/` and measures whether 观海买点分 predicts forward return (Information Coefficient + per-bucket hit-rate). Read-only. Run: `../../vcp_env/bin/python backtest_score.py --source both`. |

### First backtest result (and how to read it)
On the current data (221 scored signals, **median 4 forward days** — the 2026-05-22
lifecycle epoch is recent), buys averaged **−4.4%** with a **25%** hit-rate and the
score showed **~zero ranking edge** (Spearman ≈ −0.04; the 90–100 bucket did not beat
70–80). This is **regime-specific, not a verdict on the score**: the window is a sharp
drawdown that the market-context model itself flags as 看跌/避险 (risk 13, VIX +40%).
The valuable part is that the loop now exists — rerun it as more batches accrue forward
days and across different regimes, and use it to recalibrate the hand-tuned weights.

---

## Verification

```
$ ../../vcp_env/bin/python -m py_compile xunlong.py scan_stocks.py realtime_dashboard.py \
      backtest_score.py tests/test_engine.py        # COMPILE OK
$ ../../vcp_env/bin/python tests/test_engine.py      # 14 passed, 0 failed
$ ../../vcp_env/bin/python backtest_score.py --source both   # runs; 221 obs
```

Not done here (needs a live run): an end-to-end nightly scan, which requires
network + the input workbook + macOS. The changes preserve the existing output
schema, so `scan_result_latest.xlsx` and all exports are unaffected in shape.

---

## Update — sell-conviction score (卖出分) + dated reports

- **`score_sell_signal_row` (scan_stocks.py)** — answers "why do buys have a
  score but sells don't?": the original scorer returned `NaN` for SELL rows. Added
  a symmetric 0–100 sell score (formal vs. warning sell, `1出` confirmation, high
  rank120 = room to fall, overbought-rolling-over RSI, topping L2/4H). Written to
  a new `RawSignals.sell_score` column (future runs) and unit-tested. See
  [SIGNAL_LOGIC.md §7.1](SIGNAL_LOGIC.md#71-卖出分--the-sell-conviction-score-score_sell_signal_row-added-2026-06).
- **`backend/make_report.py`** — generates the dated reports in
  [`../reports/`](../reports/). Now attaches 观海买点分 to buys and 卖出分 to
  sells (computed from RawSignals across recent runs via the engine's scorers),
  drops the row cap (full lists, sorted by score), and adds a "Top sell" column to
  the index. A trailing-`NaN` parse bug that inflated each block's count by one
  was fixed.
- Tests: **17 passing** (added 3 sell-score cases).
