# Exit-Strategy Research & Backtest — formal-buy (正式买入 / Gann_BUY_A) entries

**Date:** 2026-06-19
**Question:** The system fires `正式买入` entries but has no risk-managed exit (only the
independent `正式卖出` / EMA10-turn-down signal). Which exit strategy yields the best return?
**Method:** literature research → reconstruct all historical formal-buy entries over 6y →
simulate a grid of exit rules path-dependently on OHLC → rank by a time-aware portfolio CAGR
with walk-forward (out-of-sample) split + cost sensitivity + Deflated-Sharpe haircut.

---

## Data & method

- **Universe:** 135 of the 140 enabled scan names (5 dropped for insufficient yfinance history).
- **Entries:** every bar where `Gann_BUY_A` is True over ~6y daily data (2020-06 → 2026-06).
  **16,290 formal-buy entries** after a 110-bar indicator warmup. Entry price = signal-day close
  (matches the system's own `D0_close` / 买入价 convention). Exits evaluated from the next bar.
- **Exit indicators** (ATR, MAs, Donchian lows, Parabolic SAR) computed directly from OHLC, so
  the backtest is decoupled from engine internals. The `正式卖出` (`Gann_SELL_1_confirmed`) flag is
  the existing "signal exit" and is the **baseline**.
- **Fill model:** stops fill intraday at the stop (gap-down → at the open); targets at the target
  (gap-up → at open); MA/Donchian/signal/time exits at the close. Trailing stops use prior-bar
  ATR and prior-bar running peak (no intrabar look-ahead). 120-trading-day max horizon.
- **Return metric ("best return"):** a time-aware **portfolio** — equal weight across all open
  positions, fully invested when ≥1 position is open, cash (0%) on idle days (so holding-period
  and time-in-market are charged honestly; faster exits recycle capital but also pay more cost).
  Round-trip cost charged on exit; tested at **10 bps and 30 bps**.
- **Validation:** walk-forward — train = trades exiting before 2024-01-01 (incl. the 2022 bear),
  test = 2024-01-01 onward. Per-family best param chosen IN-SAMPLE, reported OUT-OF-SAMPLE.
  Deflated Sharpe over all 30 trials.

## Result — out-of-sample (test ≥ 2024), ranked by CAGR

| Exit strategy | test CAGR | test Sharpe | test MaxDD | test MAR | hold (d) | cost-robust? |
|---|---:|---:|---:|---:|---:|:--:|
| **Wide ATR 5× trail, no target** | **0.70** | 1.94 | -0.31 | **2.25** | ~46 | ✅ (0.695→0.675) |
| Wide ATR 6× trail, no target | 0.67 | 1.89 | -0.32 | 2.11 | ~62 | ✅ |
| Donchian 20-day-low exit + 2ATR stop | 0.67 | 1.86 | -0.36 | 1.85 | ~29 | ✅ |
| Chandelier 4×ATR22 trail (standalone) | 0.62 | 1.82 | -0.33 | 1.87 | ~31 | ✅ (very stable) |
| MA trail: close < SMA50 | 0.69 | 1.77 | -0.43 | 1.61 | ~14 | ⚠ degrades |
| MA trail: close < SMA100 | 0.63 | 1.68 | -0.41 | 1.56 | ~25 | ⚠ |
| MA trail: close < EMA20 | 0.60 | 1.71 | -0.38 | 1.58 | ~8 | ❌ turnover-fragile |
| **Baseline: signal exit only (current)** | **0.44** | 1.68 | -0.26 | 1.70 | ~24 | — |
| Hard ATR stop (any m) + signal | 0.40–0.44 | ~1.5 | -0.29 | ~1.4 | ~12–18 | below baseline |
| Fixed % stop + signal | 0.30–0.33 | ~1.5 | -0.27 | ~1.2 | ~15–20 | below baseline |
| R-target (2–5R) + signal | 0.33–0.36 | ~1.4 | -0.32 | ~1.1 | ~13 | below baseline |
| Breakeven@1R/1.5R then trail | 0.33–0.34 | ~1.3 | -0.33 | ~1.0 | ~13 | below baseline |

(10 bps shown; ranking identical at 30 bps. Full grid in `exit_backtest_results.csv`.)

## Findings

1. **The single most impactful exit lever = let winners run with a WIDE, volatility-scaled
   trailing stop and NO profit target.** Wide ATR (≈5×ATR22) trailing the running peak is the
   best on both return and risk-adjusted return (MAR ≈2.2 vs baseline ≈1.7), and is cost-robust
   (moderate holding period → low turnover). Donchian-20-low and Chandelier-4×ATR are close
   equivalents and even more cost-stable.
2. **Tight risk management HURTS this entry.** Every hard stop, fixed-% stop, R-multiple profit
   target, and breakeven hybrid ranks **at or below the current signal-only baseline** OOS.
   Targets cap the fat right tail; tight stops whipsaw. This confirms the trend-following thesis
   for this momentum entry and **does not replicate Alvarez's "hard stop beats trailing" finding**
   (his was a 52-week-high breakout; the Gann EMA10 entry behaves like classic trend).
3. **Beats the current behavior by a wide margin and generalizes.** The trailing winners beat the
   signal-only exit by ~50% higher OOS CAGR with better MAR, and the per-family in-sample→out-of-
   sample check holds (train 0.17–0.22 → test 0.62–0.70). The ranking is stable across 2020–2023
   (incl. 2022 bear) and 2024–2026.
4. **Short-hold / high-turnover exits (EMA20, SAR, Chandelier 2.5×) are partly turnover artifacts** —
   they degrade most as cost rises and fall below the cost-robust winners at 30 bps.

## Recommendation

Replace the implicit "wait for `正式卖出`" exit on formal buys with a **wide ATR trailing stop, no
profit target**: `stop = (highest close/high since entry) − 5×ATR(22)`, exit when price closes/
trades below it; keep `正式卖出` as a secondary exit (whichever fires first). Do **not** add a
profit target or a tight stop — both reduced return in every test. Use the Donchian-20 or
Chandelier-4×ATR variant if a rule that references a visible chart level is preferred.

## Caveats (absolute numbers are optimistic; the *ranking* is the robust takeaway)

- **Survivorship:** yfinance current-names only; delisted names excluded → inflates absolute CAGR,
  most of all for let-winners-run. 
- **Regime:** the 2024–2026 test window is a strong AI-led bull → absolute CAGRs are bull-inflated.
  The relative ranking (trailing > baseline > stops/targets) also holds in the bear-containing train set.
- **Portfolio model:** equal-weight across all open positions (often dozens concurrently) → the
  Sharpe ~1.9 is a *diversified-basket* Sharpe, not a concentrated-portfolio Sharpe.
- Entry assumed at signal-day close; next-open entry would shift absolute results slightly.

## Repro

```
cd Stock_OneClick/backend
STOCK_ONECLICK_NO_OPEN=1 ../../vcp_env/bin/python _exit_build.py        # build/cache 6y panel (network)
EXIT_BT_COST=0.0010 ../../vcp_env/bin/python _exit_backtest.py          # run grid (fast); set 0.0030 for 30bps
```
Outputs: `reports/exit_cache/panel.pkl`, `reports/exit_backtest_results.csv`.

---

## ADDENDUM — expanded universe (1,040 names), 2026-06-19

The original run used the 140-name curated AI/semis/nuclear watchlist. To test the
survivorship/curation caveat, I rebuilt the panel on the repo's broad list
(`stock_symbols_1243.py`): **1,040 symbols** with usable history (1,283 attempted,
243 delisted/dropped), **146k formal-buy entries**. Findings:

**1. The system no longer beats SPY.** On the broad universe the 5×ATR-trail strategy
returns **CAGR 9.1%, Sharpe 0.58, MaxDD −30%** vs **buy-hold SPY 13.9% / 0.86 / −25%** —
i.e. **−4.9 pts/yr and −0.28 Sharpe (negative alpha)**. Per-trade expectancy fell from
+8.3%/trade (curated) to **+0.66%/trade** (broad). The 5× outperformance on the watchlist
was almost entirely **survivorship + universe curation + the AI-era regime**, not the exit.

**2. The exit ranking flips.** On the broad universe the **existing signal exit (baseline)
is the best out-of-sample exit** (test CAGR 0.190, Sharpe 1.25), fractionally ahead of the
trailing-stop family (Wide-ATR / Donchian-20 / Chandelier-4× ≈ 0.15). The wide-trailing
advantage seen on the watchlist **did NOT replicate**. No exit variant beats SPY on the broad set.

**Revised takeaway:** the 5×ATR(22) trailing stop is worth keeping as a **risk-management
overlay** — it gives every formal buy a concrete, volatility-scaled exit level where the
system previously had none — but it is **not a demonstrated return enhancer** on a broad
universe. On the actual traded watchlist it helped; broadly it is ~neutral-to-slightly-negative
vs simply using the signal exit. The system's apparent edge lives in the *universe selection*,
not the exit rule.
