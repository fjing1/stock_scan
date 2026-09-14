# Signal horizon & escalation study — 2026-07-08

Two questions, one labeled dataset (`build_dataset.py`: 1,165 BUY signals,
2026-03-06 → 2026-07-08, close-to-close forward returns from the D0 signal-day
close, no costs). Tools: `horizon_sweep.py`, `escalation_backtest.py`. Read-only,
no network, reproducible.

> Close-to-close, no slippage/commission, recent lifecycle epoch (from 2026-05-22
> most signals have <14 forward days). Relative ranking checks, **not** a P&L
> statement. Treat single-window "edges" as provisional until they hold across folds.

## Q1 — Best holding horizon (1–5 days): highest win rate?

Win rate = share of signals with forward return > 0.

| Cohort | D1 | D2 | D3 | D4 | D5 | best |
|---|---|---|---|---|---|---|
| ALL BUY (n≈1140) | **50.5%** | 48.6% | 47.6% | 45.5% | 46.7% | **D1** |
| 预警买入 pre-alert (651) | 49.0% | 45.4% | 47.6% | 45.6% | 45.6% | D1 |
| 正式买入 formal (402) | 48.5% | 46.2% | 47.4% | 44.3% | 46.4% | D1 |
| 第一观察点 1st-obs (60) | **58.3%** | 51.7% | 50.8% | 44.6% | 44.0% | **D1** |
| 二进宫 2nd-entry (3) | 100% | 100% | 100% | 33% | 50% | D1 (n=3, ignore) |
| score≥90 (356) | 46.8% | 45.6% | 49.4% | 44.5% | 46.5% | D3 |

**Finding: the win rate (and mean return) decays monotonically from D1 → D5 for
essentially every cohort.** The best horizon is the shortest one tested (D1) — but
even D1 is a coin flip (~50%) with a near-zero mean (ALL BUY +0.18%; formal −0.06%).
Holding toward D5 systematically bleeds (ALL BUY mean −0.93% at D5, t −3.0).

Only genuinely >50% cohort: **第一观察点 (first-observation) at D1 = 58.3% win,
mean +0.20%, n=60** — worth watching, but small sample and not fold-verified.

Walk-forward (ALL BUY, D1): 4/5 months >50%, cross-fold mean 49.1% — but 2026-07
drops to 38.4%. Formal-buy at D1: only 1/3 folds >50%. **No robust cycle edge; the
only consistent signal is "shorter is better, exit fast."**

## Q2 — Does "pre-alert → formal buy" escalation add a tradeable edge?

`--escal-window 5` trading days, D10 forward returns.

| Cohort | n | mean | hit>0 | t |
|---|---|---|---|---|
| PA_ESCALATED (pre-alert that later escalated) | 165 | **+1.50%** | 58.8% | +2.0 |
| PA_FIZZLED (pre-alert that didn't) | 178 | −5.13% | 29.8% | −6.1 |
| FORMAL_WARM (formal buy after a pre-alert) — *tradeable* | 241 | **−2.43%** | 42.3% | −3.5 |
| FORMAL_COLD (formal buy, no prior pre-alert) | 161 | −1.15% | 46.6% | −1.3 |

- **PA_ESCALATED looks great (+1.5%, survives haircut) but is LOOKAHEAD** — you only
  know it escalated after the fact. Not tradeable.
- **The tradeable read, FORMAL_WARM, is NEGATIVE and negative in all 3 monthly folds**
  (−2.2% cross-fold). A pre-alert warm-up does **not** improve a formal buy
  (FORMAL_WARM − FORMAL_COLD: CI spans 0, no edge).

**Finding: escalation has no tradeable edge** — consistent with the prior
multi-timeframe-alignment result. Do **not** wire pre-alert→buy promotion into live
signals. Kept as an info-only pre-market watch line in the notes file.

## Recommendation

1. **Holding period: treat these as ~1-day signals.** Edge decays with holding; D1
   is best but marginal. If anything, the mean-reversion exit (short hold) beats a
   multi-day swing hold on this universe.
2. **第一观察点 @ D1** is the single most promising cohort — worth a dedicated,
   larger-sample follow-up before trusting.
3. **Escalation: no-go for live.** Info-only watch line stays.
4. Re-run both tools as more batches accrue D1–D14 (the 2026-07 fold is thin).

## Q3 — Does combining D1 with a later day beat a fixed hold? (`horizon_combo.py`)

Path-dependent rules for partner day N∈{2..5}: FIRSTGREEN (exit first green day in
1..N else DN), CUTLOSER (ride if D1>0 else cut at D1), TAKEPOP (take D1 pop else
ride to DN), BESTOF (max(D1,DN), lookahead ceiling).

**Headline: no tradeable combo materially beats just exiting at D1.** Every
tradeable rule clusters near zero mean. FIRSTGREEN raises the *win rate* nicely
(ALL BUY 50.5%→61.8% at D2, up to 72.9% at D5) **but its mean goes negative as N
grows** — a high hit rate hiding a fat losing tail (never-green names held to a
deeper DN loss). Best tradeable by mean: FIRSTGREEN@D2 (+0.21%, 61.8% win) — barely
above EXIT_D1 (+0.18%, 50.5%), and it collapses to 46.6% in the 2026-07 fold. CUTLOSER
(let winners run) is the *worst* (mean negative, 30–37% win) — riding winners gives
the pop back (mean reversion). 4/5 folds >50% for FIRSTGREEN@D2 but fragile.

**The real find — D1 sign is a strong short-term momentum predictor:**

| DN | P(DN>0 \| D1>0) | mean DN \| D1>0 | P(DN>0 \| D1≤0) | mean DN \| D1≤0 |
|---|---|---|---|---|
| D2 | 73.4% | +3.07% | 23.5% | −2.90% |
| D3 | 64.9% | +2.26% | 30.2% | −2.93% |
| D5 | 60.2% | +1.91% | 33.2% | −3.77% |

A signal green at D1 stays up ~73% of the time at D2 (+3.1% mean); a signal red at
D1 keeps bleeding (24% up, −2.9%). This continuation is the most promising structure
found — but it's a **D1-confirmation filter** (act after the D1 close), not a fixed
exit, and converting it to positive net expectancy still needs cost modeling + tail
management, and it weakens in 2026-07. Worth a dedicated study, not a live wire-in yet.

## Q4 — Entry timing: enter at D0, or wait for confirmation? (`entry_timing.py`)

Returns priced from cumulative pct_vs_D0: ret(a→b) = (1+c_b)/(1+c_a)−1, c_0=0.

**Finding 1 — entering at the signal close (D0) is the best entry; waiting hurts.**
Every delayed-entry cell is worse than D0: enter-D0/hold-1d = 50.5% win / +0.18%;
enter-D1/hold-1d = 49.9% / −0.06%; enter-D2 and D3 progressively more negative
(t −3 to −4). The D0→D1 move is the most valuable leg — give it up and you lose.

**Finding 2 — the "green-D1 momentum" is NOT harvestable, and it retracts the Q3
lead.** The +3.1% / 73% number was the D0→D2 return of names that *ended up* green at
D1 — it INCLUDES the D0→D1 pop. Actually entering at the D1 close (after confirmation)
gives −0.18% / 47.8% at D2, −0.97% / 42.7% at D3: the move is already spent by the
time you confirm. Walk-forward enter-D1-green→D3: 2/5 folds >50%, mean win 44.5%,
negative in 2026-03 and 2026-06. **A confirmation entry has no tradeable edge** — the
earlier "promising filter" read was a lookahead artifact. Dip entry (enter D1 if red)
is ~flat (+0.06% at D2), also nothing.

## Overall verdict (four studies)

Enter at **D0**, exit at **D1**. No timing device tested — escalation promotion,
combination exits (FIRSTGREEN/CUTLOSER/TAKEPOP), or confirmation entry — adds a
tradeable edge over that. The signals are near coin-flips whose only reliable
property is that holding longer bleeds. Consistent with the repo's prior
no-edge / survivorship findings.

## Q5 — Executable entry: buy next-morning OPEN (`entry_open.py`, real OHLC)

The +0.18% "D0 entry" assumes a fill at the D0 close, but the daily signal only
completes at that close and the scan runs after hours — so the earliest executable
fill is the next session's OPEN (D1 open). Splitting the D0close→D1close move with
real opens (n=1144):

| entry → exit | n | win | mean | t |
|---|---|---|---|---|
| D0 close → D1 close (optimistic) | 1144 | 50.3% | +0.17% | +1.3 |
| &nbsp;&nbsp;of which overnight gap D0c→D1o (**missed**) | 1144 | 49.2% | +0.11% | +1.6 |
| **EXECUTABLE D1 open → D1 close** | 1144 | 49.8% | **+0.05%** | +0.4 |
| **EXECUTABLE D1 open → D2 close** | 1125 | 48.4% | **−0.01%** | −0.1 |
| EXECUTABLE D1 open → D3 close | 1099 | 49.0% | −0.23% | −1.1 |

**~2/3 of the already-tiny +0.18% edge is the overnight gap you MISS by entering at
the open.** What's left (D1 open→D1 close) is +0.05% — statistically zero (t 0.4) —
and turns negative if you hold beyond the day. Formal buys are negative at every
executable exit. Walk-forward D1open→D2close: 2/5 folds >50%, negative in 2026-03 and
2026-06. **The executable entry has no edge before costs.**

**Conclusion: the near-close-entry idea is moot.** The optimistic edge lives in an
overnight gap you can only capture by buying at/before the D0 close — i.e. by
anticipating an unconfirmed signal with intraday data the scan doesn't have. Not
worth building the intraday-confirmation (#23) path to chase +0.05% below costs.
</content>
