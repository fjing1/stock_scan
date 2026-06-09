# Stock_OneClick: How to (Carefully) Try to Profit With It

> **Provenance & status.** Synthesized by a 15-agent analysis workflow grounded in
> the project's own labeled signal data (`reports/strategy_dataset.csv`, built by
> `backend/build_dataset.py`). **Decision support, not investment advice, and not a
> guarantee.** Every "data-driven" figure was recomputed from the local dataset; every
> "principle" item is prudent practice the single-episode data cannot confirm. The
> supporting tools (`build_dataset.py`, `score_calibration.py`, `gate_calc.py`,
> `benchmark_ma.py`) live in `backend/`.

---

## 1. Bottom line up front

**The single biggest takeaway: this system has shown NO proven, tradeable
security-selection edge.** The only component correlated with outcomes is the
market-regime label, and that correlation is an in-sample artifact of one
bull-to-crash episode (2026-05-22 → 06-05), not a demonstrated forecast. The honest
play is to use the scanner as a **risk dashboard / exposure dial**, prove it forward
on paper, and recalibrate the score before risking real capital.

What the data **does** show (n=637; 410 BUY / 227 SELL; ~20 effectively independent
date-clusters; **one** regime transition):

- The buy score **观海买点分 has zero ranking edge.** IC (Pearson/Spearman): fwd_d1
  −0.071/−0.027, fwd_d3 −0.032/+0.065, fwd_d5 −0.145/−0.059, fwd_last −0.056/−0.034 —
  all inside the noise band (±~0.07 at n=218). Buckets are non-monotonic (fwd_last:
  70-80 −3.7%, 80-90 −5.2%, 90-100 −4.2%). **Do not rank or size by it.**
- **Regime separates near-term outcomes at a fixed horizon.** Current-gen BUYs,
  fwd_d3: 强势/中性看涨 **+3.87%, 61% hit (n=104)** vs 谨慎看涨 **−5.29%, 25% hit
  (n=61)**. Regime explains ~4× the outcome variance of the score (eta² 0.217 vs 0.054).
- The **SELL/short side "worked" but was ~93% market beta**, not selectivity: long-short
  spread only **+0.75pp**; Spearman(卖出分, short-pnl) = 0.012.
- **Heavy concentration:** 245 names but ~64% sit in one AI-semi-datacenter-power
  complex; ~60% same-day co-movement → effectively **1-2 macro bets.**

What the data **does NOT** show — fatal to any "proven edge" claim:

- **Regime is 1:1 collinear with calendar-date blocks** (强势 = 05-22…05-29; 谨慎 =
  06-01/02/04; 风险 = 06-03; 看跌 = 06-05). "Regime works" is statistically
  indistinguishable from "those 5 days went up and the next few went down." The model
  **relabels** the episode; it is not shown to **lead** it.
- The headline 强势 fwd_d3 **+3.87% collapses to +1.74% ex-top-10**, and those winners
  are late-May **earnings gappers** (OKTA +43%, NOW +33%, ORCL +30%, TEAM +30%, MDB
  +22%) — symmetric event risk, not a swing edge.
- The 谨慎 "−5.29% trap" is essentially **one day** (06-02, whose 3-day window catches
  the 06-05 crash).
- The most extreme **看跌/避险 label (06-05) has ZERO forward data** — the most
  important risk-off rule is unvalidated.
- Returns are **close-to-close**, ignoring spreads, slippage, borrow, and taxes.

**Verdict: treat today as the start of a forward test, not a deployment.**

---

## 2. Recommended setup — "Regime-gated defensive overlay + small long sleeve"

A blend of the two top-scored proposals (capital-preservation core + hedge-overlay
breadth trigger), borrowing only the regime-downgrade exit from the long-only swing.
**Rejected:** single-name shorting, net-short bets, and all score-based sizing.

### Regime gate (master switch)
Read two same-day, look-ahead-free inputs at the close: (1) the market-context label,
(2) `SELL_share = production SELLs / (production BUYs + SELLs)` that day.

| State / breadth | Long sleeve gross | Hedge / cash |
|---|---|---|
| 强势/中性看涨 **and** SELL_share < 0.30 | up to **40%** of NAV, equal-weight | rest cash |
| 谨慎看涨 **or** SELL_share ≥ 0.40 | **0% new**, hold ≤15% | start index hedge |
| 风险升高 **or** SELL_share ≥ 0.60 | **0%** | cash; optional small SPY/QQQ put |
| 看跌/避险 | **0%, 100% cash** | (principle — unvalidated) |
| BLANK (legacy) | **0% — no signal** | cash |

Gross capped at **40%** because the edge is unproven and the book is one factor. The
gate can only *cut* exposure as risk rises; its worst case is foregone upside.
`SELL_share` printed 0.03–0.41 in 强势, 0.40 on 06-03, **0.93** on the 06-05 crash day —
a fast, independent, look-ahead-free confirmer.

### What to trade / NOT trade
- **Long sleeve:** current-production BUY types only (第一买入点, 二进宫买入点,
  预警买入, 正式买入, 买入跟踪), price **≥ $10**, equal-weight, **max 6 names**.
- **Do NOT trade:** legacy/blank types (C_ok, Gann_*, BUY_B_D1FJ_*, V1/V2). **No
  single-name shorts. No options except SPY/QQQ index hedges. No leverage. No net-short.**
- **Hedge:** SPY/QQQ shares or 1-3mo, 5-10% OTM puts (~60% QQQ / 40% SPY to match the
  AI-heavy book) — index only.

### Entries (fill at the next OPEN, never the signal close)
- Only in 强势/中性看涨 **and** SELL_share < 0.30; enter next session's open.
- **Earnings exclusion:** no new entry within 3 trading days of a report (~55% of the
  in-sample edge was earnings gaps).
- **No score floor** (requiring ≥80 *cut* in-sample returns; within-regime score
  Spearman −0.07).
- **Factor cap:** AI-semi-datacenter-power-uranium complex ≤ 30% of NAV.

### Exits
1. **Regime/breadth downgrade (primary, dominates all others):** state leaves
   {强势/中性看涨} **or** SELL_share ≥ 0.40 → flatten the sleeve at the next open.
2. A 正式卖出 / 预警卖出 on a held name.
3. **5-day time stop** (no scored row reached d10).
4. **−7% hard stop** (close basis).
5. Optional trim half at +6%.

### Sizing & cadence
Equal-weight, 10%/name cap, **≥60% cash even in the best regime**, portfolio at-risk
≤5% NAV. Daily EOD: scan → set gate → **act next morning at the open**, no intraday
discretion.

---

## 3. Why this, not the others (per the adversarial critiques)

| Proposal | Robustness | Overfit | Why kept/dropped |
|---|:--:|:--:|---|
| Regime-gated long-only swing | 28 | 5/5 | Edge is earnings-gap + close-fill leak. Kept only its downgrade exit. |
| Long/short regime rotation | 34 | 4/5 | Self-defeating: data-supported shorts are unborrowable small-caps; short selectivity ~0. Dropped single-name shorts. |
| **Capital-preservation defensive rotation** | **38** | 4/5 | **Core.** Asymmetric, safe-if-wrong de-risk to cash. |
| **Hedge-overlay / exposure-timing** | **38** | 4/5 | **Contributes** the `SELL_share` breadth confirmer + index-only hedging. |

Nothing scored above 38 because every setup is gated on a regime signal collinear with
date in one episode — which is exactly why §8 (forward validation) is mandatory.

---

## 4. Account & instruments
**Cash account** (or margin used as cash) — no shorting/leverage → no borrow/recall/
margin/PDT risk; majority parked in **T-bills/MMF** (a real position earning the bill
rate). Long sleeve = **cash equity shares** (not options — avoids theta, lets stops
work). Hedge = **SPY/QQQ shares or index puts** (index-only: one short removes most of
the single-factor risk; single-name SELLs showed ~0 selectivity + real borrow problems).
Broker with an API (IBKR/Alpaca) for **alerting/logging**, but **execution stays manual**
while the edge is unproven.

## 5. Automation (once network is allowed)
1. **EOD cron** ~30-60 min after the 16:00 ET close:
   `vcp_env/bin/python Stock_OneClick/backend/scan_stocks.py` (writes
   `scan_result_latest.xlsx` + the per-date sheet).
2. `make_report.py` for the dated report; `gate_calc.py` to append `gate_log.csv`
   (date, state, sell_share, target_gross, action).
3. **Alert** on any state downgrade or SELL_share crossing 0.40 / 0.60.
4. **Act at the open**, manually; log fills at the open price.
5. Weekly `score_calibration.py` to track whether the score is gaining IC.
6. **Paper first — non-negotiable.**

## 6. Hard risk limits
Max gross long **40% NAV** (强势 only), 0% otherwise; never leveraged/net-short. **Max 6
names**, 10%/name. **−7% stop**, **5-day stop**, **downgrade overrides both**. Portfolio
at-risk ≤5% NAV. **Single-theme cap 30%** (AI-core / datacenter-power /
uranium-nuclear-grid-storage = one factor). **Daily circuit breaker:** −4% NAV in a
session → flatten. Put budget ≤1%/mo, buy only when an independent signal (VIX spike,
SPX<MA, credit) confirms. ≥$10 price; verify ADV manually. No averaging down.

## 7. Fix the score first (highest-leverage engineering)
观海买点分 IC ≈ 0; until it earns positive monotonic IC it is dead weight. Use
`score_calibration.py` + the dataset: (1) snapshot the baseline; (2) **de-confound** —
day-demean returns, compute **per-date IC** distribution, winsorize buckets so single
gappers don't dominate (within-date IC was still −0.04 → score adds nothing); (3)
**decompose the score** into sub-features (rank120, RSI, L2_trend, H4_RSI, H4_FJ) and
regress each vs day-demeaned returns; keep only stable-sign features; (4) refit on dates
≤T, test on >T — never evaluate on the fit window; (5) 卖出分 is the more promising
candidate (Spearman −0.24, correct sign) — same protocol; (6) **go/no-go to size by
score:** OOS Spearman ≥ +0.10, monotonic, across ≥3 dates. Until then: **equal-weight,
score unused.**

## 8. Forward-validation (prove on paper BEFORE capital)
Paper/blotter, full §2/§6 rules, **open fills**, realistic costs (≥0.1-0.3% long
slippage; index hedge 2-5 bps; puts 3-8% of premium). **Duration:** until ≥3-4 distinct
episodes incl. ≥2 risk-off transitions **and ≥1 V-recovery** (the data has zero recovery
sample) → realistically **3-6 months**. **Primary test: does the gate LEAD?** Benchmark
it against a **dumb price rule** (`benchmark_ma.py`: SPX < 20-day MA → cash). **GO** only
if: it flips risk-off *before/with* drawdowns in ≥2 episodes; the paper curve beats both
40%-long-and-hold **and** the dumb rule net of costs; whipsaw is tolerable; (and OOS
Spearman ≥0.10 if you'll size by score). **NO-GO:** if it only confirms *after* drops or
can't beat the price rule → **use the cheaper MA rule and shelve the regime model for
selection.** Ramp from ≤10% of intended capital even on GO.

## 9. Honest expectations & failure modes
Best-case description: *"a small high-beta AI/semi long sleeve that de-risks to cash on a
regime/breadth signal"* — a **drawdown-reduction tool, not an alpha engine**; honest EV
after costs is **flat-to-modestly-positive in trends with smaller drawdowns**, with a
real chance of **no edge at all.** Ranked risks: (1) **the gate lags instead of leads**
(EOD, 1:1 with date; the 06-05 看跌 label coincided with the crash) → buy-low/sell-high
whipsaw; (2) **whipsaw** in choppy tapes erodes a ~+1.7% ex-outlier edge; (3) the "edge"
was **earnings gaps**; (4) **single-factor gap** risk; (5) cost/insurance drag; (6)
labels overfit to this episode; (7) the unresolved 06-05 cohort could flip conclusions.
**If forward validation is NO-GO:** keep the scanner as a watchlist/dashboard, use a
transparent price-based de-risk rule for exposure, and do not deploy a selection strategy.

## 10. Monday checklist
1. **Baseline (offline):** `score_calibration.py` → confirm IC ≈ 0; snapshot the bar to beat.
2. **Gate calculator:** `gate_calc.py` → `gate_log.csv` (state + SELL_share + target_gross).
3. **Paper blotter** with the exact §2/§6 rules + open-fill + cost model.
4. **Benchmark:** `benchmark_ma.py` (the dumb MA de-risk rule to beat).
5. **When network is allowed:** register the EOD cron + a downgrade / SELL_share alert.
6. **GO/NO-GO scorecard** (§8 criteria) committed to the repo — pre-commit the decision.
7. **Do NOT trade real capital this week.** Paper only until the window + recalibration clear.
