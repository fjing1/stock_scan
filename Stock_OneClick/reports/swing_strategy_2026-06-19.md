# Swing-vs-Buy&Hold — research-grounded findings & the strategy that beats buy-hold

**Date:** 2026-06-19 | long-only, cash (no shorts/options/leverage), realistic costs

## The question
Across the whole session, every technical timing/swing idea lost to buy-and-hold *in 2020–26*
(a historic bull). This study asks the question properly: **over a full market cycle that
includes bear markets, is there a swing strategy that beats buy-and-hold — and on what?**

## MR best practice used (from research: Connors/Alvarez, Faber, Pagonidis 2013, Pandey&Joshi 2023)
- **Entry:** RSI(2) < 5 **and** IBS < 0.2 (oversold + closed near the low), only relevant context.
- **Exit:** first up-close (best per Alvarez's exit study), 10-day time-stop backstop.
- **Trend filter:** MR longs only **above** the 200-day SMA; below it is lower-Sharpe / falling-knife.
- **Full-cycle buy-hold beater = Faber trend-timing** (hold above 200MA w/ 3-day confirm, cash below):
  matches return, ~halves drawdown. MR works on **indices/ETFs**, not single small/illiquid names.
  Short-term reversal edge has **decayed** post-2010 (honest caveat).

## The strategy: COMBO regime-switch
**Hold the index when it's above its 200-day SMA (3-day confirm); when below, trade short MR
bounces (RSI2<10, exit first up-close / 10d) instead of sitting in cash.** Rides bulls, harvests
bear bounces, avoids the deep-crash hold.

## Results — full cycle (2000–2026), vs buy & hold

| Market | Strategy | CAGR | Sharpe | MaxDD |
|---|---|--:|--:|--:|
| **SPY** | Buy & hold | 6.4% | 0.42 | −56% |
| **SPY** | **COMBO** | **6.4%** | **0.50** | **−29%** |
| **SPY** | Faber trend-time | 4.1% | 0.42 | −32% |
| **QQQ** | Buy & hold | 8.2% | 0.43 | −83% |
| **QQQ** | **COMBO** | **9.8%** | **0.56** | **−56%** |
| **QQQ** | Faber trend-time | 7.6% | 0.53 | −56% |

By era (SPY): COMBO **+2.4%** in 2000–09 (buy-hold **−2.7%**); lags in the 2010s bull (6.4% vs 11.2%);
nearly matches 2020–26 (12.9% vs 14.0%) at half the drawdown.

**On a survivor large-cap STOCK basket (40 names, since 2005): buy-hold WINS** (11.7% / 0.68 / −50%
vs COMBO 5.2% / 0.49 / −21%) — survivorship means the basket never had the bear pain to protect
against, and the 200MA filter's cash-drag gutted bull returns. The COMBO only cut drawdown.

## Conclusion (the honest answer)
1. **Yes — a swing strategy beats buy-and-hold over a full cycle, but at the INDEX level (SPY/QQQ),
   not on a survivor stock basket.** The COMBO beats SPY/QQQ buy-hold on Sharpe and drawdown (and on
   QQQ, on return too), *because the index actually crashes in bears* and the regime-switch sidesteps it.
2. **The edge is crash-protection + bear-bounce harvesting**, realized only when bears occur. In an
   uninterrupted bull (2010s) buy-hold wins; over a full cycle (which includes bears) the COMBO wins,
   with far lower drawdown.
3. **Don't swing-trade a survivor stock basket** — buy-hold of winners that survived is near-unbeatable
   (peak survivorship). Trade the COMBO on the **broad index**, where MR/trend research says it belongs.

## Recommended deployable strategy
**COMBO on SPY (and/or QQQ):** hold 100% when SPY > 200-day SMA (3 consecutive closes); when below,
buy RSI(2)<10 dips and exit on the first up-close (10-day max hold), else cash. Best risk-adjusted
full-cycle profile; SPY variant is the cleanest (−29% DD). QQQ variant has higher return but the
bear-MR sleeve carries more tail risk (−56% DD).

## Caveats
- Index reconstitution has *some* survivorship but far less than a hand-picked stock basket.
- "Beats on return" depends on bear markets recurring; pure-bull futures favor buy-hold.
- Short-term reversal has decayed since ~2010 — the bear-bounce sleeve may be weaker going forward.
- Costs modeled at 10bps (ETF) / 15bps (stocks); MR turnover is the main cost risk.

## Repro
`backend/_swing_research.py` (SPY/QQQ COMBO by era), `backend/_swing_basket.py` (stock basket).
