# Signal Logic

This is the analytical core of Stock OneClick: the `XunLongIndicator` and the way
`scan_stocks.py` turns its output into scored, tracked buy/sell signals. Line
references are `xunlong.py` unless noted as `scan_stocks.py`.

> The indicator is a Python port of a TradingView **Pine** script (the helpers
> `xsa`, `hist`, `nz`, `rma` mirror Pine built-ins). Its own buy/sell spec, in
> the author's words, is `../exports/寻龙诀_GannBox_买卖点说明.md` — this doc is
> the implementation-level companion to it.

---

## 1. `XunLongIndicator.compute()` — the column factory

`compute(df_daily, df_4h)` (647) takes OHLCV daily bars (and optional 4H bars)
and returns the daily frame **augmented with ~40 columns**. It is pure and
stateless across symbols (one instance is reused for the whole run). The
pipeline:

1. `_calc_L2_trend_pump` → `L2_trend`, `L2_pump`
2. `_calc_ema_rsi_helpers` → `d2/d3`, `bbuy`, `varr1` (=分金), plus the sell helpers
3. `_calc_vol_osc` → `VolOsc`, `VolOscPool`, `VolOscZeroLine`
4. `_calc_manual_rsi` → `RSI`; also stamps `FJ_value`, `SELL_high_weakening`
5. `_calc_ABC_daily` → `ema8/ma13/ma21`, OBV, `A_ok`, `C_ok`, `C_rev_ok`
6. `_calc_gannbox_buy_sell` → the **Gann Box engine** (the production buy/sell)
7. 4H merge → `H4_*_daily`, `H4_RSI_last`, `H4_FJ_last`
8. A "strategy layer": `BUY_B_D1FJ_low_H4_0birth`, `SHORT_A_D1FJ_weak_H4_1birth`
9. A "low-strength / breakout" layer: `Rank120`, `VolMA20`, `V2_early_turn`, `V2_strong_buy`, `V2_base_breakout`
10. A "trade-confirmation" layer: `BUY_C_confirmed`, `SELL_profit_protect`, `SELL_trend_break`

Of all that, the nightly engine consumes only the handful in §5–§6. The rest is
covered in §9.

---

## 2. Core oscillators

### L2 swing — `L2_trend`, `L2_pump` (`_calc_L2_trend_pump`, 145)
A KDJ/Stochastic-style swing oscillator over a `K=9`/`D=3` window, smoothed with
the Pine `xsa` function. Two outputs:
- **`L2_trend`** = the swing value, but **zeroed whenever it is ≤ 45** (`var6b.where(var6b > 45, 0.0)`). So `L2_trend == 0` is the meaningful state: "not in an up-swing / quiet-or-low." It's used as a buy precondition (`A_ok` needs trend and pump both zero) and as a low-context proxy (`L2_trend <= 30`/`<= 35`).
- **`L2_pump`** = a volume/low-driven "pump" gauge capped at 100, high when price prints new 30-bar lows with energy.

### 分金 / FJ — `FJ_value` (`varr1`, in `_calc_ema_rsi_helpers`, 184)
The signature oscillator. `varr1 = xsa(up, 6, 1) / xsa(|Δclose|, 6, 1) * 100`,
i.e. a 6-period smoothed RSI on a 0–100 scale. Interpretation (matches the柱体
description in the author spec): **low (≤ ~40) = washed out / repairing**, **high
(≥ ~85) = overheated**. It drives the sell-side state machine (§2.1) and gates
several buy conditions.

`bbuy` (192): a "green-bar" event — the EMA cascade `d1=(C+L+H)/3 → d2=ema(d1,6)
→ d3=ema(d2,5)` crossing up (`d2 > d3` while previously `≤`). This "首绿柱 / first
green bar" notion recurs in the low-start logic.

#### 2.1 High-position sell state machine (`_calc_sell_signal`, 208)
A small armed/triggered FSM on `varr1`: arm when `varr1 ≥ 85` (recently); fire a
**primary** sell when it "turns purple" (`varr1` rolls over) on a red/weak bar;
allow **follow-through** sells while the weak cycle persists; disarm once
`varr1 < 60`. Outputs `SELL_high_weakening` + `SELL_weak_trend_followthrough`.
*(These feed dashboards/analysis, not the six emitted signals.)*

### Volume oscillator (`_calc_vol_osc`, 275)
`100 · (EMA₅(vol) − EMA₁₀(vol)) / EMA₁₀(vol)`, with a stateful "oscpool"
accumulator that only grows on positive momentum (or small give-backs). Surfaced
as `VolOscZeroLine = osc + 40`, used to gate the daily `C_ok` signal
(`volosc_zero_line > C_vol_threshold=40`).

### RSI — `RSI` (`_calc_manual_rsi`, 329)
Standard Wilder RSI (`rma`-smoothed, `rsi_len=14`). Distinct from `FJ_value`.
Used heavily in scoring and as a buy gate.

### A / C / C_rev (`_calc_ABC_daily`, 340)
Three classic entry conditions, **all computed but none emitted** by the nightly
scanner:
- **`A_ok`**: `L2_trend==0 & L2_pump==0` (a low/quiet base) **and** (`bbuy` or
  `RSI>50` or RSI crossing 50 up). The "early base turn."
- **`C_ok`**: two closes back above `ema8` with bullish MA stack (`ema8<ma13<ma21`,
  i.e. price reclaiming a still-falling stack) **or** an OBV cross-up, gated by
  volume.
- **`C_rev_ok`**: a mean-reversion-to-midrange condition.

---

## 3. The Gann Box engine — the production buy/sell

Two methods implement the Gann structure. Both walk the bars once, maintaining a
single "active up-leg" segment.

### 3.1 Segment / 0出 / 1出 detection (`_calc_gann_turn_marks`, 442)
Definitions (matching the author spec's distinction between *where the level is
drawn* and *when the system confirms it*):

- **EMA10** = `ema(close, 10)`. `ema_turn_up` = EMA10 turns from flat/down to up;
  `ema_turn_down` = turns from flat/up to down.
- While **no** segment is active, track `temp_zero` = the running minimum low (the
  candidate `0`).
- **`0出` (Gann_0_birth):** on the bar where `ema_turn_up` fires (with a
  candidate low in hand) → open a segment, set `seg_low = temp_zero`, mark
  `Gann_0_birth = True` *that day*. The drawn `0` sits on the earlier low; the
  birth flag marks the confirmation day.
- While active, count consecutive `ema_up` bars and track `seg_high`.
- **`1出` (Gann_1_birth):** on `ema_turn_down`, if the leg is "valid" — at least
  `gann_min_ema_up_bars = 3` up bars **and** gain `seg_high/seg_low − 1 ≥
  min_gain` — mark `Gann_1_birth = True` and close the segment. `min_gain` is
  **8 %** on the daily, **3 %** on the 4H (`_calc_4h_B_C` calls it with `0.03`).

This method only produces the birth booleans (used for the 4H warning signals).

### 3.2 Full daily engine (`_calc_gannbox_buy_sell`, 511)
Same segment walk, but it additionally computes the **Gann levels** on every
active bar — `Gann_0 = seg_low`, `Gann_1 = seg_high`, and
`Gann_382/50/618` as `seg_low + (seg_high−seg_low)·{0.382,0.5,0.618}`, plus
`Gann_1_date` (date of the segment high) and `Gann_gain_pct` — and derives two
tradeable booleans:

- **`Gann_BUY_A`** (→ 正式买入). Fires once per segment when **all** hold:
  - segment active and no BUY A yet this segment;
  - a **trigger**: `ema_turn_up` *or* `rsi_low_turn` (RSI between
    `gann_buy_a_rsi_min=35` and `gann_buy_a_rsi_low_ceiling=50`, turning up);
  - a **launch confirm**: a recent green bar (`bbuy` within
    `gann_buy_green_lookback=8`) *or* `fj_repair` (FJ rising & RSI>45) *or*
    `close > EMA10`;
  - `RSI > 35`; `close ≥ seg_low`; and a constructive bar (`close>open` or
    `close>prev close`).
- **`Gann_SELL_1_confirmed`** (→ 正式卖出). Fires on `ema_turn_down` when the leg
  is valid (≥3 up bars, ≥8 % gain) **and** the segment already produced a BUY A
  (`segment_has_buy`). The crucial design point: **a formal sell only fires for a
  leg that previously gave a formal buy** — sells are tied to the buy that
  preceded them.

So the daily "formal" pair is a closed loop: `0出 → BUY A → … → 1出 confirmed`.

---

## 4. 4H signals and the daily merge (`_calc_4h_B_C`, 401 + `compute` 696)

On the 4H frame the indicator computes `B_firstGreen` (a 12-bar momentum cross
up), `C_ok_4h`, `H4_RSI`, `H4_FJ_value`, and the 4H `Gann_0_birth`/`Gann_1_birth`
(via §3.1 with the 3 % gain floor). These are then **collapsed to one row per
calendar day** (`groupby(date).agg`: `max` for the booleans, `last` for the RSI/FJ
readings) and merged onto the daily frame as:
`H4_Gann_0_birth_daily`, `H4_Gann_1_birth_daily`, `H4_RSI_last`, `H4_FJ_last`
(also `B_firstGreen_daily`, `C_ok_4h_daily`). These `_daily` columns are what the
scanner reads for the "warning" signals. If no 4H data is available the columns
default to `False`.

---

## 5. The low-start buy points (`add_low_start_buy_points`, scan_stocks.py:2538)

Implements the operator's preferred low-base entries:
- `green_sig`: the same `d1→ema6→ema5` cascade cross-up as `bbuy`.
- `low_context`: `Rank120 ≤ 0.45` **or** `L2_trend ≤ 30` **or** `FJ ≤ 45`.
- `valid_green` = `green_sig & low_context & stabilized` (`close≥open` or up day).
- **`LOW_START_FIRST_BUY` (第一买入点):** the first `valid_green` of a base —
  *preliminary observation only*.
- **`LOW_START_SECOND_BUY` (二进宫买入点):** a later `valid_green` whose gap from
  the prior green is 2–15 bars, with a real pullback (depth ≥ 2.5 %) that held
  structure (`pullback_low ≥ prev_low · 0.97`). The "second entry confirms."

---

## 6. The six emitted signals (`scan_one_symbol`, scan_stocks.py:2598)

For each symbol: download daily (1y, **requires ≥ 150 rows** or it's skipped) +
4H (90d), `compute`, join the V1/V2/low-start helper columns, then emit signals
that fired within a **recent lookback window** —
`max(V1_LOOKBACK=5, V2_LOOKBACK=5, GANN_LOOKBACK=10) = 10` trailing trading days
(`daily_recent_mask`). Each emitted row carries the fields recorded by
`append_signal` (2629): `close, volume, vol_ma20, L2_trend, L2_pump, RSI,
rank120, H4_RSI, H4_FJ, H4_0_birth, H4_1_birth, Gann_1_date, Gann_1_price,
extra_info`.

| 信号 (signal_type) | Side | `model` | Trigger column | Source |
|--------------------|------|---------|----------------|--------|
| 第一买入点 | BUY | `LOW_START_FIRST_GREEN` | `LOW_START_FIRST_BUY` | §5 |
| 二进宫买入点 | BUY | `LOW_START_SECOND_GREEN` | `LOW_START_SECOND_BUY` | §5 |
| 预警买入 | BUY | `H4_BUY_A_0出` | `H4_Gann_0_birth_daily` | §4 |
| 正式买入 | BUY | `D1_BUY_A_0出` | `Gann_BUY_A` | §3.2 |
| 预警卖出 | SELL | `H4_SELL_1出` | `H4_Gann_1_birth_daily` | §4 |
| 正式卖出 | SELL | `D1_SELL_1出` | `Gann_SELL_1_confirmed` | §3.2 |

Conceptual ladder: **第一买入点** (earliest, riskiest) → **预警买入** (4H rebound
structure) → **正式买入** (daily confirmed) on the buy side; **预警卖出** (4H top)
→ **正式卖出** (daily confirmed top) on the sell side. The author spec stresses
these are *observation aids*, to be combined with market environment, position,
volume, and the Gann levels — not standalone orders.

---

## 7. 观海买点分 — the buy score (`score_buy_signal_row`, scan_stocks.py:1862)

A 0–100 composite, computed for **BUY rows only** (`NaN` for sells). Starts at
**50** and adds/subtracts:

| Factor | Adjustment |
|--------|-----------|
| signal_type | 正式买入 **+20**, 二进宫买入点 **+16**, 预警买入 **+14**, 第一买入点 **+10** |
| model contains `BUY_A` or `0出` | **+8** |
| `rank120` (position in base) | ≤0.25 **+10**, ≤0.45 **+7**, ≤0.65 **+3**, ≥0.85 **−5** |
| `RSI` | 42–65 **+7**, 35–42 **+3**, >75 **−6** |
| `L2_trend ≤ 35` | **+5** |
| `H4_FJ ≤ 55` | **+4** |
| `H4_RSI ≥ 45` | **+3** |

Clamped to `[0, 100]`, rounded to 1 decimal. Net: the score rewards a *confirmed
buy*, *low in its base*, with *constructive-but-not-overbought* momentum. It is
the sort key for the buy follow-up sheets and the TradingView buy lists. The
release notes call it "观海买点分."

---

## 8. The 14-day lifecycle & follow-up tracker

Two complementary structures, both keyed off `LIFECYCLE_START_DATE = 2026-05-22`
and `TRACK_MAX_DAYS = 14` trading days.

### 8.1 Per-date follow-up sheets (`_build_followup_sheets`, scan_stocks.py:512)
For each signal date it produces a sheet with, per symbol: `观海买点分`, `板块`,
`D0_date`, `D0_rule`, `D0_close`, then `D1…D14` columns of forward close +
`Dᵢ_pct_vs_D0` (green/red %), plus `prior_14d_signal_dates` and
`retrigger_dates`. **Cycle locking:** the first trigger is `D0`; any re-trigger
within 14 trading days is folded into the same cycle (recorded, not restarted); a
trigger after the window opens a fresh cycle. A batch whose 14 days are all
filled is split off as "completed" and archived.

### 8.2 Lifecycle tables (`_build_lifecycle_tables`, scan_stocks.py:1414)
Pairs **formal buys** with the next **formal sell** (and formal sells with the
next formal buy) to produce four tables — the workbook sheets
`买入观察列表 / 买入历史记录 / 卖出观察列表 / 卖出历史记录`:
- **历史记录 (history):** a buy that has been closed by a later formal sell (with
  stage return `卖出价/买入价 − 1` and stage length in trading days).
- **观察列表 (observation):** a buy still open after ≥14 trading days with no
  formal sell yet (and the symmetric sell→buy case).

---

## 9. Computed but **not** emitted

These columns are produced by `compute()` every run yet never become signals in
`scan_one_symbol`. They matter because they look like live strategies but aren't
wired into the nightly output:

- **Earlier-iteration buys/sells (dormant):** `V1_Buy` (SATS low-vol breakout,
  scan_stocks.py:2412), `DailyStrong` (UBER daily-strong, 2443),
  `BUY_low_reset_4h_green`/`BUY_low_reset_confirmed` (2475/2507), `A_ok`,
  `C_ok`, `C_rev_ok`, `V2_early_turn`, `V2_strong_buy`, `V2_base_breakout`,
  `BUY_C_confirmed`, `SELL_profit_protect`, `SELL_trend_break`,
  `SELL_high_weakening`.
- **Dashboard strategies (from the author spec, used by the live layer, not the
  nightly scan):**
  - **`BUY_B_D1FJ_low_H4_0birth`** (compute, 752): daily FJ in a low context
    (`FJ ≤ 40` or recent-5-day min ≤ 40) that isn't deteriorating, **plus** a 4H
    `0出`. The "日线分金低位 + 4H 0出" rebound buy.
  - **`SHORT_A_D1FJ_weak_H4_1birth`** (753): a 4H `1出` **plus** a weakening daily
    backdrop (FJ falling, or price weak, or FJ was >60 in the last 8 days). The
    "4H 顶 + 日线转弱" short/trim watch.

If you want any of these in the nightly output, you must add an emit block to
`scan_one_symbol` (and likely a `signal_type`/`model` label + scoring weight).
See [ARCHITECTURE.md §4.1](ARCHITECTURE.md#41-the-indicator-is-a-kitchen-sink-the-engine-is-selective).

---

## 10. Market-context risk model (`build_market_context`, scan_stocks.py:852)

Independent of the per-symbol scan, this rates the *overall market* from a
hard-wired index basket — SPX, QQQ, IWM (core), RSP (breadth), VIX & HYG (risk),
SMH (offense), XLU (defense) — by reusing `scan_one_symbol` on each and tallying
a `risk` score from conditions like: a core index printing 正式卖出 or losing
MA20, a sharp single-day SPX/QQQ drop, VIX spiking or reclaiming MA20, RSP lagging
SPX (breadth), HYG weak (credit), and XLU outperforming SMH (defensive rotation).

The score maps to a state + advice banner placed atop each per-date sheet:

| risk | state | guidance |
|-----:|-------|----------|
| ≥ 6 | 看跌/避险 | de-weight all buys; pause chasing; favor cash-flow/defensive |
| ≥ 4 | 风险升高 | buys de-weighted; only high-score, clean structures |
| ≥ 2 | 谨慎看涨 | trend intact but tighten size / avoid chasing |
| else | 强势/中性看涨 | normal buy observation; high-score offense preferred |

Per the in-code note (`MARKET_CONTEXT_HELP_TEXT`), this is **display-only** — it
does **not** currently adjust individual 观海买点分 scores.
