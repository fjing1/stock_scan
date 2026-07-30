# Wide-universe research scan — bias-reduced signal read (2026-07-08)

Isolated, READ-ONLY scan of the expanded universe run through the live engine
(`research_scan.py`, 4 parallel shards) — **never touched the live book, lifecycle,
or history**. Then a 5-lens analysis + 3 adversarial verifications (each recomputed
independently from the CSVs). Artifacts: `reports/research_scan_all.csv`,
`reports/research_per_symbol.csv`, `reports/research_scan_{1..4}.csv` (+ `_fail.txt`).

## Coverage (clean, not throttle-biased)
- 1,045 scannable → **1,035 scanned OK, 10 yfinance failures** (a failure ≠ "no signal").
- **912 symbols emitted ≥1 BUY**; 4,023 signal rows total (2,751 BUY).

## Headline findings (adversarially verified — numbers re-derived, not trusted)

**1. The score does NOT discriminate on a broad universe. [CONFIRMED]**
552 / 912 signal-emitters (60.5%) — and **53.3% of the entire scanned universe** —
reach `buy_score ≥ 90`. Median buy_score = 92. When more than half of *all US
large-caps* score ≥90, a "score ≥90 buy" carries little cross-sectional information.

**2. But that breadth is ~1.9× inflated by MAX-over-window aggregation. [CONFIRMED]**
`research_per_symbol.csv` takes the MAX score across a symbol's recent signals
(mean ~4 signals/symbol). Restricting to the **latest signal_date only**:
- ≥90 count: **552 → 294** (−47%, 1.88× inflation)
- Perfect 100s: **176 → ~49** (only ~14% still perfect on the latest bar)
- 191 symbols fall from ≥90 into 80–89; 67 fall below 80.
So the true latest-bar breadth is far smaller than the leaderboard implies — much of
the "everything is a buy" impression is stale signals kept alive by the MAX operator.

**3. Curation added theme tilt + only MODEST signal selectivity. [PARTIAL]**
Live vs research on identical metrics: live ≥90 rate 67.5% vs research 59.5%
(**+8pp — real but modest**); score distributions otherwise near-identical
(medians both 92, Cohen's d ≈ 0.08 = negligible). The big difference is **theme**:
live is 49% AI/tech vs research 1.4% (+48pp). Curation bought concentration, not a
sharper signal — consistent with the survivorship finding, now generalized.

## What this means
Widening the universe **confirms and extends the prior conclusion**: the signals
have no cross-sectional discrimination and (from earlier studies) no tradeable
forward edge on the curated set. Nothing about going 162→1,045 changed that; it just
removed the curation veil and showed the score lights up ~half the market in a bull
regime. The score is best used as a **relative tiebreaker + filter**, never an
absolute cutoff.

## The only defensible use: a tight structural SCREEN (not an edge)
Requiring `正式买入 (formal) + raw ≥ 100 (multi-signal confluence) + RSI < 70 +
rank120 > 0.10` shrinks 912 → **137 symbols (15%)**; RSI mean 47.7 (not overbought),
rank120 mean 0.25 (constructive). Top raw=107: CHTR, IDXX, NYT. **Explicitly a
cross-sectional quality screen — no forward-return validation exists** (research
names were never in the lifecycle).

## Recommended next step
The research pool has **no forward data** — that's the real gap. To learn whether
the signals carry *any* edge on a bias-reduced set, **begin point-in-time
forward-tracking** the research names (mark today's signals, accrue D1–D14 like the
live lifecycle) so in a few months they become backtestable with `horizon_sweep.py`
/ `entry_open.py`. Until then, treat the wide scan as a screen, not a buy list.

## Method caveats
Point-in-time (current membership) → still not fully survivorship-free. Strong-bull
regime inflates breadth; re-measure across regimes. Close-to-close, no costs.
</content>
