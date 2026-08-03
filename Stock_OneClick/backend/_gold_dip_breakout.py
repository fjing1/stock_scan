"""
_gold_dip_breakout.py — test two user-requested signals on gold DAILY bars:

  A) 85-day MA "dip in uptrend"  — buy pullbacks to the 85d MA while the long trend is up.
  B) MA8 crossover breakout      — the repo's break3avg family (8 vs ~22), applied to gold.

Both are SHORT-horizon signals being evaluated for a 1yr+ allocation system, so they are tested
for two different jobs, which have different bars to clear:

  Job 1 — ENTRY TIMING for the staging plan. Bar: beat buying on an arbitrary date. This is a
          low bar and a legitimate use even for a weak signal, because you have to buy *somewhere*
          and the tax cost is zero (you are buying either way).
  Job 2 — a TIMING SYSTEM that switches exposure. Bar: beat buy-and-hold after 28% collectibles
          tax. This is a high bar; the repo's history says signals almost never clear it.

Event studies are detrended against SPY where relevant and report N, because with 26 years of
gold there are not many independent pullbacks.

Run: ../../vcp_env/bin/python _gold_dip_breakout.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import gold_system as G

TD = 252
RNG = np.random.default_rng(7)


def ann(curve: pd.Series):
    ret = curve.pct_change().dropna()
    yrs = len(ret) / TD
    cagr = curve.iloc[-1] ** (1 / yrs) - 1
    vol = ret.std(ddof=1) * np.sqrt(TD)
    return cagr, vol, ((curve / curve.cummax()) - 1).min(), (cagr - G.RISK_FREE) / vol


def fwd_table(g: pd.Series, events: pd.DatetimeIndex, label: str,
              horizons=(21, 63, 126, 252), baseline: dict | None = None):
    """Forward total returns from the close of each event day."""
    out = {}
    print(f"  {label}  (N={len(events)})")
    if len(events) == 0:
        return out
    for h in horizons:
        vals = []
        for d in events:
            i = g.index.get_loc(d)
            if i + h < len(g):
                vals.append(g.iloc[i + h] / g.iloc[i] - 1)
        if not vals:
            continue
        v = np.array(vals)
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
        base = f"   vs base {baseline[h]*100:+6.2f}%  edge {(v.mean()-baseline[h])*100:+6.2f}pp" if baseline and h in baseline else ""
        print(f"    +{h:>3}d  mean {v.mean()*100:+6.2f}%  median {np.median(v)*100:+6.2f}%  "
              f"win {(v>0).mean()*100:3.0f}%  SE {se*100:4.2f}%{base}")
        out[h] = v.mean()
    return out


def main():
    panel = G.load_panel()
    g = panel["GC=F"]["Close"].dropna()
    r = g.pct_change()
    idx = r.dropna().index
    r = r.loc[idx]
    rf_d = (1 + G.RISK_FREE) ** (1 / TD) - 1
    print(f"gold daily {idx[0].date()} -> {idx[-1].date()}  ({len(idx)} bars, "
          f"{len(idx)/TD:.0f} independent years)\n")

    bh = ann((1 + r).cumprod())
    print(f"buy-and-hold gold:  CAGR {bh[0]*100:.2f}%  vol {bh[1]*100:.1f}%  "
          f"MaxDD {bh[2]*100:.1f}%  Sharpe {bh[3]:.3f}")

    ma85 = g.rolling(85).mean()
    ma500 = g.rolling(500).mean()
    uptrend = g > ma500                      # the long-trend filter established earlier

    # unconditional baseline: forward return from ANY day in an uptrend
    base = {}
    up_days = idx[uptrend.reindex(idx).fillna(False)]
    for h in (21, 63, 126, 252):
        v = [g.iloc[i + h] / g.iloc[i] - 1 for i in
             (g.index.get_loc(d) for d in up_days) if i + h < len(g)]
        base[h] = float(np.mean(v))
    print(f"\nBASELINE: forward return from a RANDOM day while above the 500d MA "
          f"(N={len(up_days)})")
    for h, v in base.items():
        print(f"    +{h:>3}d  mean {v*100:+6.2f}%")

    # ----------------------------------------------------------------- A) 85d dip
    print("\n" + "=" * 78)
    print("A) 85-DAY MA 'DIP IN UPTREND'")
    print("=" * 78)

    below85 = g < ma85
    # event = first day price closes below the 85d MA, while above the 500d MA
    dip_event = below85 & (~below85.shift(1).fillna(False)) & uptrend
    ev = idx[dip_event.reindex(idx).fillna(False)]
    print("\n  Job 1 — ENTRY TIMING: forward return after a fresh dip below the 85d MA")
    fwd_table(g, ev, "dip below 85d MA while above 500d MA", baseline=base)

    # reclaim event = closes back above the 85d MA after being below, still in uptrend
    rec = (~below85) & below85.shift(1).fillna(False) & uptrend
    ev2 = idx[rec.reindex(idx).fillna(False)]
    print()
    fwd_table(g, ev2, "RECLAIM of the 85d MA (dip confirmed over)", baseline=base)

    print("\n  Job 2 — TIMING SYSTEM variants (cash earns T-bill):")
    variants = {
        "long only when above 500d AND above 85d": uptrend & (~below85),
        "long only when above 500d AND BELOW 85d (buy-dip)": uptrend & below85,
        "long when above 500d (any 85d state)": uptrend,
        "long when above 85d only": ~below85,
    }
    for lbl, s in variants.items():
        s = s.shift(1).reindex(idx).fillna(False)
        c, v, d, sh = ann((1 + r.where(s, rf_d)).cumprod())
        f = int((s.astype(int).diff().abs() == 1).sum())
        print(f"    {lbl:<50} CAGR {c*100:5.2f}%  MaxDD {d*100:6.1f}%  "
              f"Sharpe {sh:.3f}  inMkt {s.mean()*100:3.0f}%  flips/yr {f/(len(idx)/TD):4.1f}")

    # ----------------------------------------------------------- B) MA8 crossover
    print("\n" + "=" * 78)
    print("B) MA8 CROSSOVER BREAKOUT (the repo's break3avg family, on gold)")
    print("=" * 78)
    print("\n  break3avg definition used in this repo: close > MA8 AND close > MAlong AND MA8 > MAlong")
    print("\n  Job 2 — as a long/cash timing system, sweeping the long leg:")
    print(f"    {'long leg':>9}  {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'inMkt':>6} {'flips/yr':>9}")
    best = None
    for nl in (15, 21, 22, 23, 30, 50, 100, 200):
        m8 = g.rolling(8).mean()
        ml = g.rolling(nl).mean()
        s = ((g > m8) & (g > ml) & (m8 > ml)).shift(1).reindex(idx).fillna(False)
        c, v, d, sh = ann((1 + r.where(s, rf_d)).cumprod())
        f = int((s.astype(int).diff().abs() == 1).sum())
        print(f"    8 / {nl:>3}d   {c*100:6.2f}% {d*100:7.1f}% {sh:7.3f} {s.mean()*100:5.0f}% "
              f"{f/(len(idx)/TD):9.1f}")
        if best is None or sh > best[1]:
            best = (nl, sh)
    print(f"\n    best long leg: 8/{best[0]}d at Sharpe {best[1]:.3f}  "
          f"vs buy-and-hold {bh[3]:.3f}")

    print("\n  Job 1 — ENTRY TIMING: forward return after a fresh 8/22 breakout")
    m8, m22 = g.rolling(8).mean(), g.rolling(22).mean()
    bk = (g > m8) & (g > m22) & (m8 > m22)
    fresh = bk & (~bk.shift(1).fillna(False))
    fwd_table(g, idx[fresh.reindex(idx).fillna(False)], "fresh 8/22 breakout (all regimes)",
              baseline=base)
    fresh_up = fresh & uptrend
    print()
    fwd_table(g, idx[fresh_up.reindex(idx).fillna(False)],
              "fresh 8/22 breakout WHILE above the 500d MA", baseline=base)

    # ------------------------------------------------- combined: dip + breakout
    print("\n" + "=" * 78)
    print("C) COMBINED — dip to 85d in an uptrend, then an 8/22 breakout to confirm")
    print("=" * 78)
    dipped = below85.rolling(63).max().astype(bool)      # dipped at some point in the last ~3mo
    combo = fresh & uptrend & dipped
    print()
    fwd_table(g, idx[combo.reindex(idx).fillna(False)],
              "8/22 breakout, above 500d, after a recent 85d dip", baseline=base)

    print("\n  As a staging trigger: of the 3 tranches in the system, how long would you wait?")
    ev_c = idx[combo.reindex(idx).fillna(False)]
    if len(ev_c) > 1:
        gaps = np.diff([g.index.get_loc(d) for d in ev_c])
        print(f"    signals: {len(ev_c)} in {len(idx)/TD:.0f} yrs = {len(ev_c)/(len(idx)/TD):.1f}/yr")
        print(f"    median gap between signals: {np.median(gaps):.0f} trading days "
              f"({np.median(gaps)/21:.1f} months); max gap {gaps.max()} days "
              f"({gaps.max()/21:.0f} months)")
    print(f"\n  CURRENT STATE ({g.index[-1].date()}):")
    px = float(g.iloc[-1])
    print(f"    price {px:,.2f} | 85d MA {float(ma85.iloc[-1]):,.2f} "
          f"({(px/float(ma85.iloc[-1])-1)*100:+.1f}%) | 500d MA {float(ma500.iloc[-1]):,.2f} "
          f"({(px/float(ma500.iloc[-1])-1)*100:+.1f}%)")
    print(f"    above 500d (uptrend): {bool(uptrend.iloc[-1])} | below 85d (dipped): "
          f"{bool(below85.iloc[-1])} | 8/22 breakout on: {bool(bk.iloc[-1])}")
    print(f"    MA8 {float(m8.iloc[-1]):,.2f}  MA22 {float(m22.iloc[-1]):,.2f}  "
          f"MA8>MA22: {bool(m8.iloc[-1] > m22.iloc[-1])}")


if __name__ == "__main__":
    main()
