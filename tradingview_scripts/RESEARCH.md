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
- **Walk-forward verdict (adopted #21):** the single 2019 split is a useful first screen but
  fragile, so cross-sectional results are now also judged by rolling-origin walk-forward (refit
  on a trailing window with a 1-week embargo, stitch all test windows into one OOS curve), a
  **recent-3y** readout (regimes change — 2007–18's market/players/tech ≠ now), and overfit
  haircuts: **Deflated Sharpe** (discount for the N configs tried) + **PBO/CSCV** (is the
  in-sample-best config overfit). A result must clear the haircut, not just one cut's OOS.

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
| 16 | Sector rotation — "sector A up → B next"? | `sector_rotation.py` | ❌ no clean lead-lag (weak & negative); momentum rotation ≈ market; real rotation is macro-regime-driven |
| 17 | Macro risk-on/off regime timing (Yahoo proxies)? | `macro_regime.py` | ❌ no edge — regime doesn't predict returns; risk-off timing cuts maxDD but lowers Sharpe; cyc/def inverted (mean-reversion). Official FRED test pending restart |
| 18 | Residual (factor-neutral) momentum — beats raw 12-1? | `residual_momentum.py` | ⚠️ in-sample crash-protection **replicates** (LS maxDD −65%→−22%, β≈0) but **no net OOS edge** — IR-form churns 78–163%; raw long-only wins OOS |
| 19 | Post-earnings drift (PEAD) — orthogonal event alpha? | `pead_drift.py` | ✅ **real & OOS-persistent** — Q5−Q1 surprise drift +2.25%→**+3.05%** (63d, train→test); low-turnover (14%) overlay net Sharpe **0.27** OOS, β≈0 → the orthogonal leg to STACK on #12 |
| 20 | Stack PEAD onto the #12 ensemble — does orthogonality pay? | `pead_stack.py` | ❌ **no reliable lift** — score-stack −0.05 test; 50/50 sleeve +0.15 test but −0.10 train; PEAD is period-concentrated (weak pre-2019). Keep 4-alpha+buffer |
| 21 | Is the static 2019 split right? Walk-forward + overfit haircuts | `walkforward.py` | ⚠️ rolling-origin OOS + DSR/PBO. Equal-weight robust (OOS ~0.37, recent-3y 0.73); adaptive weighting needs a ≥4–6mo window and **fails the haircut** (DSR 0.78, PBO 0.65) → config selection is overfit |
| 22 | Is SMA200 the best trend filter? (1–200 sweep + fib/EMA) | `ma_filter_sweep.py`, `ma_length_curve.py` | ✅ **No — 200 is too long.** Edge peaks at the ~85d MA (OOS ~0.9%/trade vs 0.55% at 200) and the **50>200 regime-cross** (t 4.7, DSR 1.00); fib & EMA show no magic; dead-zone n≈10–55. Scanner filter → SMA50>SMA200 (+ Close>SMA85 hi-conv tier) |
| 23 | Does a 15/30/60m intraday confirmation improve the daily entry? | `mtf_intraday_test.py` | ~ **weak/indicative** — 60m (2yr, n=1219): confirmed +0.29% vs unconfirmed −1.11% (Δ+1.4pp, t 2.2); 15m directional (70% vs 52% win, t 1.2, n 46); 30m null. A timing overlay (`dip_scan --confirm-tf`), not a proven edge |
| 24 | Regime-switch (hold uptrend + MR-bounce downtrend) vs buy&hold over a FULL cycle (2000-26)? | `_swing_research.py`, `_swing_basket.py` (Stock_OneClick/backend); report `Stock_OneClick/reports/swing_strategy_2026-06-19.md` | ✅ **on the INDEX** — COMBO beats B&H: SPY 6.4%/Sharpe 0.50/−29%DD vs 6.4%/0.42/−56%; QQQ 9.8%/0.56/−56% vs 8.2%/0.43/−83%. ❌ on a survivor large-cap stock basket (B&H wins 11.7% vs 5.2% — survivorship = no bear pain). MR best-practice (Connors/Alvarez/Faber/Pagonidis 2013): RSI2<5 & IBS<0.2 entry, **first-up-close exit**, MR only in regime; reversal decayed post-2010 |

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

**Regime-switch beats buy-&-hold — but only on the index (24).** The dip-in-uptrend edge
(4–11) is cash in bears; this asks whether a full strategy can beat *buy-and-hold* over a
cycle that includes 2000–02 and 2008. The **COMBO** — hold the index when above its 200-day
SMA (Faber, 3-day confirm), and trade short MR bounces (RSI2<10, exit first up-close / 10d)
when below, instead of cash — does, on **SPY/QQQ**: SPY 2000–26 CAGR 6.4% / Sharpe 0.50 /
MaxDD −29% vs B&H 6.4% / 0.42 / −56%; QQQ 9.8% / 0.56 / −56% vs 8.2% / 0.43 / −83%. The edge is
crash-protection + bear-bounce harvesting (+2.4% SPY in 2000–09 while B&H lost −2.7%); it lags
in the uninterrupted 2010s bull (the insurance premium). **It does NOT beat B&H on a survivor
large-cap stock basket** (40 names: B&H 11.7% vs COMBO 5.2%) — survivorship means the basket
never had the bear pain to protect against, and the trend filter's cash-drag guts bull returns.
So swing/MR + trend-timing belongs on the **broad index**, not hand-picked stocks. MR rules are
research best-practice (Connors/Alvarez: RSI2 oversold + IBS<0.2, first-up-close exit, stops hurt
MR; Faber trend-timing for the drawdown win; Pagonidis 2013 / Pandey&Joshi 2023 for the IBS edge).
Honest caveat: short-term reversal has decayed post-2010, so the bear sleeve is weaker going
forward. Encoded as the **COMBO mode** in `dip_in_uptrend_strategy.pine`.

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

**Residual (factor-neutral) momentum — crash-protection replicates, net edge doesn't (18).**
*Surfaced by the `research-ideas` agentic workflow as the #1 mature, replicated, NEW-to-repo
edge.* Plain 12-1 momentum (the ensemble's #12 leg) carries factor/beta tilt that drives the
notorious momentum crashes; residual momentum (Blitz–Huij–Martens 2011) regresses each name's
daily returns on 3 factor proxies (market = SPY, size = IWM−SPY, value = IWD−IWF), ranks the
**IR-scaled cumulative residual** (skip 1mo), and trades the top/bottom-quintile spread vs the
RAW 12-1 baseline through an identical engine. **In-sample, the literature's headline benefit
replicates cleanly:** the factor-neutral long-short halves drawdown (train maxDD raw −65% →
residual −22%) and flips raw's ~zero train alpha positive (LS Sharpe −0.13 → +0.37), β≈0.
**But out-of-sample (2019→, incl. 2022) the edge does not survive costs.** The IR-scaling
re-ranks violently — turnover 78%/wk and a runaway 163%/mo — so net-of-cost LS Sharpe is
0.15 (weekly) / −0.11 (monthly), and the long-only top-quintile alpha vs SPY is *lower* than
raw momentum's OOS (+12.7% vs +20.8% weekly). Raw long-only momentum remains the stronger
deployable signal on this survivorship-biased OHLCV universe. This **confirms #14's verdict**:
the easy OHLCV cross-sectional alphas are exhausted — residual momentum's edge is real in-
sample but gets decayed/churned away net. Open levers before shelving: a non-IR (plain
cumulative-residual) score to cut churn, the #13 rebalance buffer, and a point-in-time
universe. `python residual_momentum.py [--rebal 5|21] [--names N]`.

**Post-earnings-announcement drift (PEAD) — the orthogonal alpha that holds up (19).**
*Chosen as the follow-up to #18 because #14/#15 said the binding constraint is NEW data
orthogonal to price.* PEAD is driven by the EARNINGS SURPRISE (a fundamental/event signal,
not OHLCV). Realized analyst surprises ARE reachable via yfinance's earnings-dates endpoint
with enough depth for OOS (AAPL to 2005; **10,856 surprise records across 135/136 names**;
needs `lxml` + the `guce.yahoo.com`/`consent.yahoo.com` allowlist — see the apple sandbox CSV).
**Event study (9,232 tradable events, entry the session AFTER the announce so it's tradable,
drift detrended vs SPY):** the top-minus-bottom surprise-quintile drift is **+0.67% → +2.25%**
over +21d → +63d in TRAIN and **+0.95% → +3.05%** in TEST — i.e. it ACCUMULATES over the
quarter (as Bernard–Thomas predict) and STRENGTHENS out-of-sample; in test the miss-quintile
(Q1) turns negative, the clean PEAD signature. **Tradable overlay** (a name is "in play" for
63 trading days post-report, weekly long-top/short-bottom-surprise-quintile): long-short net
Sharpe **0.05 train → 0.27 test**, β −0.01/+0.15, and crucially **turnover is only 14–16%**
(quarterly events rotate slowly) — so costs barely dent it, the opposite of #18's churn. As a
standalone long-short it's modest, but it is the **low-turnover, orthogonal, OOS-persistent**
leg #14 said was missing. **Recommended next build: stack PEAD as a new alpha in the #12
ensemble** (event-driven ⇒ ~uncorrelated to the price alphas) and/or tilt the dip-scanner
toward names with a recent positive surprise. CAVEAT: survivorship inflates the long leg and
analyst-surprise is weaker than SUE/estimate-revisions — treat magnitudes as a lower bound;
trust the quintile monotonicity, OOS persistence, and β≈0. `python pead_drift.py [--names N]`.

**Stacking PEAD onto the ensemble — orthogonality didn't reliably pay (20).** Added the #19
PEAD leg to the validated 4-alpha blend under the #13 buffer config, two ways. The base
reproduces ~0.5 net Sharpe (train 0.45 / test 0.53, β 0.03). PEAD's spread is only **+0.19**
correlated to the 4-alpha spread — the diversifier #14 asked for — yet **neither combination
lifts net Sharpe in BOTH windows**: (a) *score-stacking* (z-sum into one ranking) gives
train +0.03 / **test −0.05** — a sparse signal just perturbs the ranking at the margin
(echoes #14); (b) the correct *sleeve* combination (hold PEAD as its own book, blend net
returns 50/50) gives **test 0.53→0.69 (+0.15)** but **train 0.45→0.35 (−0.10)**. The split is
the lesson: PEAD's *tradable* net Sharpe is strong in test (0.53) but weak pre-2019 (0.05) —
**period-concentrated**, so it can't raise the combined Sharpe across both windows even though
the drift itself persists (#19). By the house rule (persist in train AND test), the **4-alpha
+ buffer remains the production config**; PEAD is better used as a **long-only tilt** in the
dip-scanner (its test long-only alpha is +12%/yr at low turnover) than as an ensemble leg. The
binding constraint is now clearly the survivorship universe + new-alpha period-concentration,
not a shortage of orthogonal signals. `python pead_stack.py [--names N]`.

**Walk-forward beats the static split — and the haircuts kill the "adaptive" edge (21).**
The single 2019 split is fragile (arbitrary cut; lumps very different regimes into one "test")
and 2007–18 may not represent how the market trades now. Replaced it with rolling-origin
walk-forward: each step refit sleeve weights on a trailing window (1-week embargo so the 5-day
forward label can't leak), hold over the next window, and STITCH every test window into one
full-power OOS curve. Two things fell out. (a) **Window length matters and short is bad:** a
4-week train window *breaks* adaptive weighting (OOS Sharpe −0.08 vs +0.37 equal — refitting on
~4 returns just chases last month's winner); adaptive (trailing-Sharpe / max-Sharpe) only beats
equal once the window is ≥17–26 weeks (4–6mo). **Equal-weight is robust everywhere** (0.29–0.41)
because it estimates nothing. (b) **The recent regime really is better:** recent-3y Sharpe is
0.73 (equal), up to ~1.0–1.4 (adaptive at the right window) vs the full-history 0.37 — a single
2019 split buries this. BUT the rigor haircuts settle it: over the **N=18 configs tried**, the
best config's **Deflated Sharpe is 0.78** (< 0.95 → NOT significant once you account for the
search) and **PBO (CSCV, 924 combos) is 0.65** (HIGH — the in-sample-best config lands below the
OOS median two-thirds of the time). **Verdict:** only *static equal-weight* (no fitting, no
selection) survives the haircut; adaptive weighting and window-picking are overfit on this
survivorship universe — which reinforces #14. The recent strength is real but **regime-driven,
not from clever weighting**. Adopt walk-forward + a recent-window readout + DSR/PBO as the
verdict mechanism going forward (see §3). `python walkforward.py [--names N]`.

**Is SMA200 the best trend filter? No — it's too long; the edge is a regime, not a number (22).**
#4 used Close>SMA200; this stress-tested it by holding the oversold entry + 10d exit fixed and
varying ONLY the trend filter. Three passes: (a) a discrete sweep of lengths/types/stacks; (b)
Fibonacci lengths + EMA; (c) an exhaustive 1..200 length curve (SMA & EMA). Findings, all
detrended-vs-SPY, OOS 2019+: **(1) the regime-CROSS shape wins** — `SMA50>SMA200` (golden cross)
is the most robust filter (OOS +0.67%/trade, t **4.7**, 2,893 signals, **DSR 1.00**, barely
degrades train→test) and beats `Close>SMA200` (+0.55%, t 3.0, *halves* train→test). **(2)
Fibonacci is not special** — the fib golden cross `SMA89>SMA233` (+0.62) ties the round
`SMA50>SMA200`, and fib price-above filters land on the same curve as nearby round numbers.
**(3) EMA is a wash** — EMA variants score within noise of SMA. **(4) The length curve is a
broad hump:** a *dead zone* at n≈10–55 (you can't be deeply oversold AND above a short MA — ~0
signals), a peak at **n≈75–100** (Close>SMA~85 gives the highest per-trade edge, ~+0.9%/trade vs
+0.55% at 200, and is *more* stable train→test), then a gentle fade to 200. **(5) Rigor:** the
single best length has **DSR 0.89** (not uniquely significant) — read it as a *band* (~80–100),
not a magic number; the "winner" C>SMA55 (+1.5% OOS) was a 119-trade fluke (negative in train)
that the persistence+DSR guard correctly discarded. **Deployed (dip_scan.py):** the gate is now
the robust `SMA50>SMA200` regime cross (catches good dips that briefly pierce the 200-day; the
ranking de-prioritizes weak ones), with `Close>SMA85` surfaced as a `hi_conv` tier (the per-trade
sweet spot). `--legacy-trend` restores `Close>SMA200`. `python ma_filter_sweep.py` /
`python ma_length_curve.py`.

**Multi-timeframe intraday confirmation — weak-but-real on 60m, data-walled on 15/30m (23).**
Re-opened the question "use 15/30m for better entries" with the current daily signal. Among
daily dip signals, split by whether the NEXT session shows an intraday strong-up bar (the
dip_scan conf rule) and compared the daily +10d detrended forward return of confirmed vs
unconfirmed. Result depends entirely on the timeframe's available history: **60m has ~2 yrs
(n=1,219) and shows a mild, marginally-significant lift** — confirmed +0.29% vs unconfirmed
**−1.11%** (Δ +1.4pp, t **2.2**, win 50% vs 44%); the value is mostly in AVOIDING the
unconfirmed dips ("nobody bought it intraday → it keeps bleeding"). **15m is only 60 days
(n=46)**: directionally strong (70% vs 52% win, +5.1% vs +1.5%) but t **1.2** — i.e. exactly
#8's weak/unproven situation. **30m is null** (t −0.1). Caveats: 3 timeframes tested (multiple-
testing would haircut the 60m t≈2.2 toward marginal), single ~2yr regime, survivorship. EXIT
is unchanged from #9 (a 15m sell-bar exit was harmful; a proper intraday-exit test is blocked
by the 60-day data wall for the ~2-week hold). **Verdict:** intraday confirmation is a sensible
LIVE timing overlay, not a proven backtested edge — deployed in `dip_scan.py` as
`--confirm-tf 15m|30m|60m` (60m is the best-powered choice; confirmed hits sort first).
`python mtf_intraday_test.py [--names N]`.

## 5. The resulting system

```
ENTRY  : SMA50 > SMA200 (up-regime, #22)  AND  RSI(14) < 40  AND  Stoch %K(10,EMA4) < 20  (daily)
         hi_conv tier = also Close > SMA85 (the ~85-day per-trade-edge peak); --legacy-trend = Close>SMA200
RANK   : DipRank = 0.45·(12m return) + 0.30·(pullback below MA50) + 0.25·(% above MA200)
TILT   : DipRank_PEAD = DipRank ± up to 12 pts by recent earnings-surprise percentile
         (#19/#20 long-only PEAD tilt; in-play = reported within ~63 trading days)
CONFIRM: optional — a strong up intraday candle on >=1.5x avg volume next session
         (#23, --confirm-tf 15m/30m/60m; 60m best-powered, weak/indicative edge)
EXIT   : close >= SMA20  (or Stoch %K >= 70, or RSI(2) >= 70)   — sell into strength
STOP   : none tight (they hurt); >MA200 is the risk control. Optional wide disaster stop.
HOLD   : ~1–3 weeks
```

- TradingView: `dip_in_uptrend.pine` (indicator — #22 regime filter SMA50>SMA200 with the
  Close>SMA85 hi-conv tier, BUY/BUY★/SELL markers, PEAD earnings-surprise label + status table),
  `dip_in_uptrend_strategy.pine` (Strategy Tester: win rate / profit factor / equity, with
  optional hi-conv and earnings-beat entry filters). `--legacy` Close>SMA200 selectable in both.
- Scanner: `dip_scan.py` → ranked dated CSV in `results/<YYYYMMDD>/`. Ranks by `DipRank_PEAD`
  (DipRank with the #19/#20 earnings-surprise tilt; `--no-pead` to disable), surfaces a `hi_conv`
  tier (#22) and an optional intraday confirmation (`--confirm-tf 15m/30m/60m`, #23), and reports
  each hit's recent `earn_surprise` / `earn_age_d`.
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
- `residual_momentum.py` — residual (factor-neutral) vs raw 12-1 momentum (workflow-surfaced; in-sample only)
- `pead_drift.py` — post-earnings-announcement-drift event study + tradable overlay (orthogonal event alpha; OOS-persistent)
- `pead_stack.py` — stack PEAD onto the #12 ensemble (score-stack + 50/50 sleeve); no reliable both-window lift
- `walkforward.py` — rolling-origin walk-forward + Deflated-Sharpe & PBO/CSCV overfit haircuts (the #21 verdict harness)
- `ma_filter_sweep.py` — trend-filter comparison (lengths, types, stacks, fib, EMA) for #22
- `ma_length_curve.py` — exhaustive 1..200 MA-length edge curve (SMA & EMA) for #22
- `mtf_intraday_test.py` — multi-timeframe (15/30/60m) intraday entry-confirmation test for #23
- `mn_turnover.py` — turnover-reduction pass (buffer/smoothing/frequency)
- `alpha_stack.py` — multi-alpha stacking + sector-neutralization (8 alphas)
- `revision_alpha.py` — analyst-revision-momentum (non-price) alpha test
- `sector_rotation.py` — sector lead-lag / rotation-momentum market research
- `macro_regime.py` — macro risk-on/off regime + market-timing test (Yahoo proxies)
- `exit_search.py` — exit-rule search (found %K≥70; stops hurt)
- `exit_strategies_lit.py` — literature exits (BBmid/RSI2 upgrade)
- `rank_test.py` — which feature ranks signals (12m RS)
- `intraday_confirm_test.py` — which intraday TF confirms (15m)
- `exit_15m_test.py` — 15m sell-bar exit (harmful)

**Deliverables**
- `dip_in_uptrend.pine`, `dip_in_uptrend_strategy.pine` — TradingView
- `dip_scan.py` — universe scanner: #22 regime filter (SMA50>SMA200, +SMA85 hi-conv) + DipRank + PEAD tilt + 15m confirm
- `position_advisor.py` — per-holding HOLD/TRIM/SELL read for one ticker (trend + swing + PEAD + sector)
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
- **New orthogonal data** is the binding constraint for OHLCV alphas (#14). **Update (#19/#20):
  realized earnings surprises ARE reachable via Yahoo with OOS depth, and PEAD is a real,
  OOS-persistent, low-turnover, orthogonal alpha — but stacking it onto the #12 ensemble did
  NOT reliably lift net Sharpe in both windows (#20: PEAD is period-concentrated, strong post-
  2019, weak before).** So the production config stays 4-alpha + buffer; PEAD is best deployed
  as a long-only TILT in the dip-scanner (recent positive surprise), not as an ensemble leg.
  Open: SUE/standardized surprise, and a point-in-time universe to retest #20 without the
  period/survivorship confounds. Estimate *revisions* and PIT fundamentals remain unavailable
  via Yahoo (#15).

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
