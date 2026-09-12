"""
_btc_mtf_research.py — deep research task: does adding the 4H chart to the validated daily
ensemble (btc_system.py) improve on it, or replicate this repo's "MTF alignment has no edge"
finding from the equity/xunlong study ([[mtf-signal-alignment-no-edge]])?

Data honesty note: yfinance's 4H BTC history is capped at 730 days (confirmed empirically). The
best free/keyless alternative, Binance.US, has a 586-DAY GAP (2023-07-14 -> 2025-02-19, missing
essentially all of 2024 including the spot-ETF-launch rally) — Binance.US suspended/thinned USD
pairs during its 2023 SEC-lawsuit period. Kraken's free OHLC endpoint doesn't support historical
backfill (returns only the latest ~720 bars regardless of the `since` param — checked empirically).
Rather than splice across that hole, this script treats the data as TWO SEPARATE CONTINUOUS
SEGMENTS and never computes a return across the gap:
  Segment A: 2019-09-18 -> 2023-07-13  (Binance.US, ~3.8yr: 2020 COVID crash, 2021 boom, 2022 bust)
  Segment B: 2024-08-27 -> present     (yfinance,   ~2.0yr: 2024-25 continuation/chop)
That's ~5.8yr of 4H-capable history against an 11.9yr daily series -- weaker statistical power,
say so plainly in the verdict, don't paper over it.

Method: build a 4H ensemble analogous to btc_system's daily one (same 11-length/3%-band/70%-vote
design, lengths scaled x6 for 4H-bar-per-day), collapsed to one reading per day (last completed 4H
bar of that day -> no lookahead). Compare:
  (1) daily-only ensemble (btc_system.ensemble_state), restricted to the SAME dates the 4H data
      covers -- NOT the full 11.9yr number, or this isn't a fair comparison.
  (2) STRICT alignment: long only when daily AND 4H both vote long.
  (3) alignment-conditioned forward returns: on days the daily ensemble says LONG, does the
      subset where 4H AGREES outperform the subset where 4H DISAGREES? This is the direct BTC
      analogue of the equity MTF-turn-alignment test that found no edge on stocks.

Run: ../../vcp_env/bin/python _btc_mtf_research.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _btc_data as D
import _btc_data_4h as D4
import btc_system as B

GAP_THRESHOLD = pd.Timedelta(hours=8)
RNG = np.random.default_rng(77)
COST_ONE_WAY = B.COST_ONE_WAY
ST_TAX = B.ST_TAX_ILLUSTRATIVE


def find_segments(idx: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    diffs = idx.to_series().diff()
    gap_mask = (diffs > GAP_THRESHOLD).values
    seg_id = np.cumsum(gap_mask)
    segs = []
    for sid in np.unique(seg_id):
        sub = idx[seg_id == sid]
        segs.append((sub[0], sub[-1]))
    return segs


def h4_ensemble_daily(h4_close: pd.Series, lengths_days=B.ENSEMBLE_LENGTHS,
                       band: float = B.ENSEMBLE_BAND, threshold: float = B.VOTE_THRESHOLD) -> pd.Series:
    """4H ensemble vote (lengths scaled x6 bars/day), collapsed to one reading/day (last bar)."""
    votes = pd.DataFrame({n: B.trend_state(h4_close, n * 6, band) for n in lengths_days})
    frac = votes.mean(axis=1)
    daily_frac = frac.resample("1D").last().dropna()
    return daily_frac >= threshold


def bootstrap_vs(a_ret: pd.Series, b_ret: pd.Series, label: str, block="M"):
    common = a_ret.index.intersection(b_ret.index)
    a, b = a_ret.loc[common], b_ret.loc[common]
    blocks = a.index.to_series().dt.to_period(block).astype(str).values
    uniq = np.unique(blocks)
    diffs = []
    for _ in range(1500):
        pick = RNG.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([np.where(blocks == u)[0] for u in pick])
        aa, bb = a.iloc[sel], b.iloc[sel]
        sa, sb = aa.std(ddof=1), bb.std(ddof=1)
        if sa > 0 and sb > 0:
            diffs.append((aa.mean() / sa - bb.mean() / sb) * np.sqrt(B.TRADING_DAYS))
    diffs = np.array(diffs)
    print(f"  {label}: mean Sharpe diff {diffs.mean():+.3f}  SE {diffs.std(ddof=1):.3f}  "
          f"95% CI [{np.percentile(diffs, 2.5):+.3f}, {np.percentile(diffs, 97.5):+.3f}]  "
          f"P(worse)={  (diffs < 0).mean()*100:.1f}%")


def main():
    panel4 = D4.load_4h()
    h4_close = panel4["Close"]
    segments = find_segments(h4_close.index)
    print("4H segments found (gap-free continuous runs):")
    for a, b in segments:
        print(f"  {a} -> {b}  ({(b - a).days} days)")

    daily_price = D.production_series()
    daily_state_full = B.ensemble_state(daily_price)   # full 11.9yr, we'll slice per segment

    # trim 1 day off each segment edge to avoid the boundary/thin-liquidity artifact measured
    # in the cross-validation step (2023-07-13 showed a 17.6% divergence vs the daily series).
    trimmed = [(a + pd.Timedelta(days=1), b - pd.Timedelta(days=1)) for a, b in segments]
    trimmed = [(a, b) for a, b in trimmed if b > a + pd.Timedelta(days=250)]  # need ensemble warmup
    print(f"\nusable segments (>250d, trimmed): {len(trimmed)}")

    all_daily_ret, all_v1_ret, all_agree_fwd, all_disagree_fwd = [], [], [], []
    for seg_a, seg_b in trimmed:
        seg_h4 = h4_close.loc[seg_a:seg_b]
        h4_daily_state = h4_ensemble_daily(seg_h4)
        d_state = daily_state_full.reindex(h4_daily_state.index).fillna(False)
        price_seg = daily_price.reindex(h4_daily_state.index).ffill()

        v1_state = d_state & h4_daily_state   # STRICT alignment

        print(f"\n=== Segment {seg_a.date()} -> {seg_b.date()} ({len(h4_daily_state)} days) ===")
        for label, state in (("daily-only", d_state), ("daily AND 4H (strict)", v1_state)):
            for tax in (0.0, ST_TAX):
                r = B.simulate(price_seg, state, cost=COST_ONE_WAY, tax_rate=tax)
                if r:
                    print(f"  {label:<24} tax={tax*100:4.1f}%: {B.fmt(r)}  "
                          f"flips/yr {r['flips_per_yr']:.1f}")

        r_daily = B.simulate(price_seg, d_state, cost=COST_ONE_WAY)
        r_v1 = B.simulate(price_seg, v1_state, cost=COST_ONE_WAY)
        if r_daily and r_v1:
            all_daily_ret.append(r_daily["curve"].pct_change().dropna())
            all_v1_ret.append(r_v1["curve"].pct_change().dropna())
            bootstrap_vs(r_v1["curve"].pct_change().dropna(),
                         r_daily["curve"].pct_change().dropna(),
                         "strict-alignment vs daily-only", block="W")

        # --- alignment-conditioned forward returns (the direct MTF-alignment-edge test) ---
        fwd5 = price_seg.pct_change(5).shift(-5)
        on_long_days = d_state
        agree = on_long_days & h4_daily_state
        disagree = on_long_days & ~h4_daily_state
        af = fwd5[agree].dropna()
        df_ = fwd5[disagree].dropna()
        if len(af) > 10 and len(df_) > 10:
            print(f"  daily=LONG & 4H AGREES   (n={len(af):>4}): fwd5d mean {af.mean()*100:+.2f}%  "
                  f"win% {(af>0).mean()*100:.0f}%")
            print(f"  daily=LONG & 4H DISAGREES(n={len(df_):>4}): fwd5d mean {df_.mean()*100:+.2f}%  "
                  f"win% {(df_>0).mean()*100:.0f}%")
            all_agree_fwd.append(af)
            all_disagree_fwd.append(df_)

    if all_agree_fwd:
        af_all = pd.concat(all_agree_fwd)
        df_all = pd.concat(all_disagree_fwd)
        print(f"\n=== POOLED alignment-conditioned forward 5d return (both segments) ===")
        print(f"  4H AGREES    n={len(af_all):>4}  mean {af_all.mean()*100:+.2f}%  win% {(af_all>0).mean()*100:.0f}%")
        print(f"  4H DISAGREES n={len(df_all):>4}  mean {df_all.mean()*100:+.2f}%  win% {(df_all>0).mean()*100:.0f}%")
        # bootstrap the mean difference (independent resample of each pool, monthly blocks not
        # meaningful here since this is a cross-sectional day-level pool, not a time series -
        # use plain iid bootstrap on the pooled day-level observations)
        diffs = []
        for _ in range(2000):
            a_s = RNG.choice(af_all.values, size=len(af_all), replace=True)
            b_s = RNG.choice(df_all.values, size=len(df_all), replace=True)
            diffs.append(a_s.mean() - b_s.mean())
        diffs = np.array(diffs)
        print(f"  mean(agree)-mean(disagree): {diffs.mean()*100:+.2f}%  "
              f"95% CI [{np.percentile(diffs,2.5)*100:+.2f}%, {np.percentile(diffs,97.5)*100:+.2f}%]  "
              f"P(agree<=disagree)={(diffs<=0).mean()*100:.1f}%")

    if all_daily_ret:
        combined_daily = pd.concat(all_daily_ret)
        combined_v1 = pd.concat(all_v1_ret)
        sd, sv = combined_daily.std(ddof=1), combined_v1.std(ddof=1)
        print(f"\n=== POOLED (both segments concatenated, NOT bridged) ===")
        print(f"  daily-only pooled Sharpe:  {(combined_daily.mean()/sd)*np.sqrt(B.TRADING_DAYS):.3f}")
        print(f"  strict-align pooled Sharpe:{(combined_v1.mean()/sv)*np.sqrt(B.TRADING_DAYS):.3f}")
        print(f"  ({len(combined_daily)} trading days across {len(trimmed)} segments, "
              f"~{len(combined_daily)/365.25:.1f}yr of usable 4H-informed history vs "
              f"{len(daily_price)/365.25:.1f}yr for the daily-only system)")


if __name__ == "__main__":
    main()
