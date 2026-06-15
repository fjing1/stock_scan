# Swing-Signal Research Project

A systematic study: take two TradingView indicators, port them to Python, and
test — rigorously, out-of-sample — whether their signals (alone, combined, or
across timeframes) produce a tradeable edge. Where an edge is found, build the
full system (entry → rank → exit), expose it as a TradingView script, and scan
the market with it.

All code is in this directory and runs against Yahoo Finance via the project
venv: `vcp_env/bin/python tradingview_scripts/<script>.py`.

---

## 1. Objective

Find a **high-win-rate, evidence-backed** swing signal — not a curve-fit. Every
claim is judged on a held-out test period, against a drift-adjusted baseline, and
against the trading literature.

## 2. Data & universe

- **Source:** Yahoo Finance (`yfinance`). Daily history 15–20y; intraday capped by
  Yahoo at ~60 days (5/15/30m) and ~7 days (1m) — a hard limit that shapes every
  intraday test below.
- **Universe:** `stock_symbols_1243.py` (1,233 stocks + 50 ETFs) for scanning;
  a 30-name liquid basket for backtests; SPY as benchmark.
- **Survivorship:** baskets are *current* names → absolute win rates are inflated;
  detrended (vs-SPY) metrics are the trustworthy ones.

## 3. Methodology (the rules we held ourselves to)

- **Out-of-sample split:** train < 2019-01-01, test 2019-01-01 → present (the test
  window includes the 2022 bear — a real stress test for long systems).
- **Detrended metric:** forward return *minus SPY* over the same window. Strips
  market drift, which otherwise makes any long signal look ~55–60% just by being long.
- **No lookahead:** higher-timeframe series are shifted one bar before forward-fill;
  signals fire on completed bars; entries at the signal bar's close.
- **Exit metric:** profit factor + expectancy (win rate alone is misleading — a
  tight exit can give 90% wins yet make no money).
- **Honesty about multiplicity:** searching N combos guarantees in-sample winners;
  only OOS persistence + economic coherence counts. Sample sizes and z-scores reported.

## 4. Experiments & findings

| # | Question | Script | Verdict |
|---|----------|--------|---------|
| 1 | Port the two indicators | `cycle_patter_for_swing.py`, `xunlong_panel.py` | ✅ faithful, live-verified |
| 2 | Cycle turning-points aligned across timeframes? | `cycle_mtf_winrate.py`, `..._v2.py` | ❌ **no edge** (detrended ≈ coin flip) |
| 3 | 寻龙诀 panel signals across timeframes? | `xunlong_winrate.py` | ❌ **no edge** |
| 4 | Combine sub-components (1–3 conditions)? | `combo_search.py` | ✅ **found it:** oversold + >MA200 |
| 5 | Best exit rule? | `exit_search.py` | ✅ sell-into-strength (%K≥70); **stops hurt** |
| 6 | Literature exits (Connors/Chandelier/SAR/BB)? | `exit_strategies_lit.py` | ✅ **BBmid/SMA20 & RSI(2)≥70** edge out %K≥70 |
| 7 | How to rank signals? | `rank_test.py` | ✅ **12-month relative strength** (not oversold depth) |
| 8 | Which intraday TF confirms best? | `intraday_confirm_test.py` | ~ **15m** sweet spot (weak: z≈1.9, 60-day data) |
| 9 | 15m sell bar as exit? | `exit_15m_test.py` | ❌ **harmful** (sells into weakness) |
| 10 | Broaden: which entry indicators have edge? | `entry_sweep.py` | ✅ oversold+>MA200 best; momentum needs a trend-exit |
| 11 | Momentum entries × trend-following exits? | `momentum_exit_sweep.py` | ⚠️ high absolute PF but it's **beta, not alpha** (detrended <50%) |
| 12 | Cross-sectional market-neutral ensemble? | `market_neutral_ensemble.py` | ✅ **real alpha** (beta≈0, OOS-persistent) — but turnover/survivorship-limited |
| 13 | Cut ensemble turnover for net edge? | `mn_turnover.py` | ✅ **rebalance buffer** halves turnover, ~2× train net Sharpe; smoothing/low-freq hurt |
| 14 | Stack more alphas + sector-neutralize? | `alpha_stack.py` | ❌ naive stacking **hurts** (negative/decayed anomalies); balanced 4-alpha stays best |
| 15 | Analyst-revision momentum (free, non-price)? | `revision_alpha.py` | ❌ orthogonal (corr≈0) but **negative** OOS — ratings lag price; adding it hurt |

### Key results

**MTF alignment is a dead end (2,3).** Across both indicators, both sides, every
alignment definition, ~120 win-rate cells each — *zero* beat baseline at 2σ once
detrended. Raw 55–62% win rates were entirely market drift.

**The edge is "buy the dip in a long-term uptrend" (4).** `Close > SMA200` **and**
`RSI(14) < 40` **and** `Stoch %K(10,EMA4) < 20`. Out-of-sample: ~+3pp detrended,
~60% absolute 10-day win rate, persistent at 5d & 10d, decays by 21d (short-horizon
mean reversion). The **>MA200 trend filter is the active ingredient**; oversold
alone has no edge; a single timeframe beats any multi-timeframe stack. This matches
the documented Connors "buy pullbacks above the 200-day MA" anomaly.

**Exit: sell into strength, never into weakness (5,6,9).** Best exits are
reversion targets: **close ≥ SMA20 (Bollinger middle, PF 2.18 / 74% win)**,
**RSI(2) ≥ 70 (PF 2.22, 5-day hold)**, and **Stoch %K ≥ 70 (PF 1.99, 71% win)**.
Trailing/volatility stops (Chandelier 39–44% win, ATR, SAR) and any tight stop
*reduce* profit factor — the deep dips are the ones that bounce. A 15m strong-sell
bar as exit is the worst (PF collapses 16.7→1.7 in-window): it bails ~2 days in,
before the bounce. (PSAR shows a high PF but 22% win rate — a trend-following
shape, not the high-win-rate goal.)

**Ranking: relative strength, not oversold depth (7).** Among dip signals, the
12-month return tercile sorts forward outcomes strongly (high-RS dips ~58% OOS
excess-win vs ~52%); pullback depth and trend strength add a little; **how oversold
a name is carries no ranking information** (counterintuitive, verified). Encoded as
`DipRank` in the scanner.

**Intraday confirmation: 15m is the sweet spot, but unproven (8).** Of 1/2/5/15/30m,
only 15m both fires selectively and shows a lift (71% confirmed vs 48% unconfirmed),
but z≈1.9 on ~67 samples in one regime — a sensible "wait for buyers" overlay, not
a proven booster. Finer TFs confirm on ~every bar (no information).

**Broad entry sweep — oversold+trend dominates; momentum needs its own exit (10).**
A 32-variant entry library (mean-reversion / breakout / volume / candlestick), each
backtested with the standardized SMA20 exit, ranks `StochK<20 & >MA200` first (test
PF 1.93 / 60% detrended win — the only "robust"-flagged signal), with `close<lower-BB
& >MA200` and `3-down-days & >MA200` close behind, and `RSI2<10 & >MA200` the
highest win rate (76%). The >MA200 filter lifts nearly every mean-reversion entry.
Momentum/breakout entries (52w-high, Donchian, MACD-cross) score poorly *here* —
but that is an exit mismatch (a mean-revert exit sells a breakout instantly), so they
must be re-tested with a trend-following exit before any verdict.

**Momentum × trend-exit — high return, but beta not alpha (11).** Paired with
trend-following exits, momentum entries DO make money: MACD-cross + 63-day hold returns
+4.85%/trade, PF 2.71, stable OOS; a simple time-based hold beats trailing stops
(Chandelier/PSAR cut win rate to 25–48%). BUT the detrended (vs-SPY) win rate is <50%
for every momentum combo — these capture **market beta over a multi-month hold, not a
timing edge**. Only the mean-reversion dip-in-uptrend shows true alpha (detrended
~56–60%). Lesson: absolute PF can hide beta; judge on excess return.

## 4b. Principles from elite quant practice (Citadel / RenTec / Two Sigma)

Adoptable (process, not secrets) and mapped to this project:
- **Alpha, not beta** — neutralize market/sector/size; chase excess return. (Our
  detrended metric; experiment 11 shows why it matters.)
- **Many small uncorrelated bets, market-neutral** — rank the universe and trade the
  spread (long top / short-or-underweight bottom) rather than discrete directional bets.
- **Ensemble weak signals** — blend the validated entries (StochK<20, lower-BB,
  3-down-days, RS) into one composite alpha, not a single trigger.
- **Costs & capacity first-class**; **risk/sizing drives returns** (vol-target, ATR
  sizing, per-name/sector caps); **overfitting paranoia** (walk-forward, deflated
  Sharpe, multiple-testing); **edges decay** (re-validate; Connors RSI2 has decayed).

Not replicable retail: HFT speed/co-location, alternative data at scale, cheap
leverage/financing, market-making (earn vs pay the spread), large research teams.
Highest-value next move: a **cross-sectional, market-neutral ensemble** — see §8.

**Market-neutral ensemble — real alpha, but turnover/survivorship-limited (12).**
Weekly, score every name by `z(-5d ret) + z(12-1 mom) + z(close/MA200) + z(-RSI2)`;
long top quintile / short bottom quintile, dollar-neutral. The long-short spread is
**market-neutral (beta-to-SPY −0.11 train / +0.02 test)** and shows **OOS-persistent
alpha**: gross Sharpe 0.60 → 0.84, 55–56% of weeks positive, through the 2022 bear.
This is the project's one clean *alpha* (vs the momentum system's beta). Honest
limits: (a) **turnover ~120%/week** → after a 10bps cost the Sharpe falls to
0.26 (train) / 0.58 (test) — deployability is cost-sensitive; (b) **survivorship
bias** (current-names universe) inflates the long leg — long-only top-quintile
"alpha vs SPY" (~+23%/yr) is overstated. Trustworthy reads: beta≈0 and OOS
consistency. Improvements (Citadel playbook): slower signals / rebalance threshold
to cut turnover, more uncorrelated alphas to lift Sharpe, sector/size neutralization,
and a point-in-time universe to remove survivorship.

**Turnover reduction — the buffer is the deployable win (13).** Of the standard
fixes, a **rebalance buffer/hysteresis** (enter in the top/bottom 20%, only drop a
name when it leaves the 40% band) cut turnover 118%→66% and nearly doubled the weak-
period net Sharpe (train 0.26→0.46, test 0.58→0.53), beta still ≈0 — it removes churn,
not signal (gross Sharpe even rose). Score smoothing and lower rebalance frequency
both HURT: they wash out the fast short-term-reversal alpha and (monthly) let beta
creep to 0.14–0.20. Net Sharpe ~0.5 is still modest — the next lever is stacking more
uncorrelated alphas, not more turnover tuning.

**Multi-alpha stack — more alphas made it WORSE (14).** Added low-vol, MAX/lottery,
52-week-high, and 1-month reversal to the 4-alpha blend. Average pairwise correlation
was low (0.12 — genuinely diversifying), BUT half the added anomalies are
negative-Sharpe in this survivor universe / recent regime (low-vol −0.85 with β−0.96,
MAX −0.82, 52w-high ≈0) — so equal-weighting dragged the ensemble from +0.53 to −0.08
net and added a −0.46 beta. Weighting by train Sharpe overfit to the reversal/oversold
cluster, which had **decayed and flipped negative in test (+0.74 → −0.30)** — a live
edge-decay demonstration. Sector-neutralization didn't rescue bad components. The
**balanced 4-alpha blend + buffer remains best** (test 0.53, β 0.03) precisely because
it mixes a decaying reversal with holding-up momentum/trend. Lesson: diversification
needs individually-positive, non-decayed components; with OHLCV alone the easy
cross-sectional alphas are exhausted — further lift needs new orthogonal data
(fundamentals / revisions / alt-data) and a survivorship-free universe.

**Analyst-revision momentum — orthogonal but negative (15).** The one free, reachable,
backtestable NON-price signal: trailing net analyst up/down-grades from yfinance
`.upgrades_downgrades` (history to 2012, 125 names). It IS orthogonal (corr ≈0 to the
price alphas, β≈0) — the diversifier we wanted — but standalone Sharpe is NEGATIVE
(−0.44 train / −0.68 test): broker rating changes LAG price, so buying upgrades loses.
Adding it dragged the ensemble (test 0.48→0.42). The inverse (+0.44/+0.68) is likely
survivorship-contaminated (downgraded survivors recovered). Note yfinance gives rating
CHANGES, not the stronger EPS-estimate revisions. This settles the frontier: the free
orthogonal option is exhausted — real further alpha needs EPS-estimate revisions,
point-in-time fundamentals, and a survivorship-free universe (not available via Yahoo).

## 5. The resulting system

```
ENTRY  : Close > SMA200  AND  RSI(14) < 40  AND  Stoch %K(10,EMA4) < 20   (daily)
RANK   : DipRank = 0.45·(12m return) + 0.30·(pullback below MA50) + 0.25·(% above MA200)
CONFIRM: optional — a strong up 15m candle on >=1.5x avg volume that session
EXIT   : close >= SMA20  (or Stoch %K >= 70, or RSI(2) >= 70)   — sell into strength
STOP   : none tight (they hurt); >MA200 is the risk control. Optional wide disaster stop.
HOLD   : ~1–3 weeks
```

- TradingView: `dip_in_uptrend.pine` (indicator, BUY/SELL markers + status table),
  `dip_in_uptrend_strategy.pine` (Strategy Tester: win rate / profit factor / equity).
- Scanner: `dip_scan.py` → ranked dated CSV in `results/<YYYYMMDD>/`.
- Sector overlay: `sector_confirm.py` → given a stock, scores its industry's trend
  (sector SPDR + thematic ETF + peer breadth) so a dip-buy is only high-conviction
  when the whole industry is healthy. Verdict tiers: STRONG / CONSTRUCTIVE (uptrend
  intact but pulling back) / WEAK. Reveals when a "stock" dip is really a sector move.

## 6. File index

**Ports (faithful Python of the Pine indicators)**
- `cycle_patter_for_swing.pine` / `.py` — "Cycle and Stoch" oscillator
- `xunlong_panel.pine` / `.py` — 寻龙诀 Panel V1 (trend/pump/bbuy/varr1/RSI + 0–10 score)

**Studies**
- `cycle_mtf_winrate.py`, `cycle_mtf_winrate_v2.py` — MTF cycle alignment (null)
- `xunlong_winrate.py` — MTF panel alignment (null)
- `combo_search.py` — sub-component grid search (found the entry)
- `entry_sweep.py` — broad entry-indicator library (32 variants, OOS)
- `momentum_exit_sweep.py` — momentum entries × trend-following exits (beta check)
- `market_neutral_ensemble.py` — cross-sectional long-short alpha ensemble
- `mn_turnover.py` — turnover-reduction pass (buffer/smoothing/frequency)
- `alpha_stack.py` — multi-alpha stacking + sector-neutralization (8 alphas)
- `revision_alpha.py` — analyst-revision-momentum (non-price) alpha test
- `exit_search.py` — exit-rule search (found %K≥70; stops hurt)
- `exit_strategies_lit.py` — literature exits (BBmid/RSI2 upgrade)
- `rank_test.py` — which feature ranks signals (12m RS)
- `intraday_confirm_test.py` — which intraday TF confirms (15m)
- `exit_15m_test.py` — 15m sell-bar exit (harmful)

**Deliverables**
- `dip_in_uptrend.pine`, `dip_in_uptrend_strategy.pine` — TradingView
- `dip_scan.py` — universe scanner with DipRank + 15m confirmation
- `sector_confirm.py` — top-down sector/industry trend confirmation overlay

## 7. Limitations

- **Survivorship bias** inflates absolute win rates (current-name baskets).
- **Intraday tests are ~60-day, single-regime** (Yahoo limit) — indicative only.
- **Multiple comparisons** — the entry/exit edges are small; trust the OOS
  persistence and economic logic, not any single z-score.
- **Entry/exit on close, modest cost modeling** — the strategy file adds 0.03%
  commission + 2-tick slippage; portfolio-level sizing not modeled.
- **No delisted tickers** — would lower absolute (long) results.

## 8. Next steps (open)

- Swap in a survivorship-free universe (point-in-time S&P 1500) and re-confirm.
- **Pair momentum/breakout entries (52w-high, Donchian, MACD) with trend-following
  exits** (Chandelier / MA-trail) — they were unfairly judged under the mean-revert
  exit; this is the open "buying × exiting" cross-study.
- A/B the **SMA20 (BBmid) exit** into the live Pine + scanner (small upgrade over %K≥70).
- Add `--min-price` / `--min-dollar-volume` liquidity filter to the scanner.
- Regime-conditioning: does the edge concentrate in high-vol / specific sectors?
- Expectancy & equity-curve / drawdown at the portfolio level (not just per-trade).
- **Cross-sectional, market-neutral ensemble** (Citadel-style, §4b): rank the
  universe daily by a blend of validated entries, trade the top-vs-bottom spread,
  size by volatility — targets alpha directly and is the highest-value upgrade.
  *Status: built (#12), turnover-tuned (#13), alpha-stacking explored (#14).
  Recommended config = balanced 4-alpha + rebalance buffer (~0.5 net Sharpe, β≈0).*
- **New orthogonal data** is now the binding constraint: OHLCV cross-sectional alphas
  are exhausted (#14). Real Sharpe lift needs fundamentals / earnings & analyst
  revisions / alt-data, plus a **point-in-time, survivorship-free universe** (CRSP /
  Sharadar / Norgate) — neither available via Yahoo.

## 9. Reproduce

```bash
# entry edge search
vcp_env/bin/python tradingview_scripts/combo_search.py --horizon 10
# exit comparison (literature)
vcp_env/bin/python tradingview_scripts/exit_strategies_lit.py
# ranking feature test
vcp_env/bin/python tradingview_scripts/rank_test.py
# scan the universe today
vcp_env/bin/python tradingview_scripts/dip_scan.py
```

*Sources for §6 exits (standard references, not live-fetched — sandbox is
Yahoo-only): Connors & Alvarez, "Short Term Trading Strategies That Work";
LeBeau, Chandelier Exit; Wilder, "New Concepts in Technical Trading Systems"
(ATR/SAR/RSI); Bollinger, "Bollinger on Bollinger Bands".*
