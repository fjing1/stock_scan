"""
_gold_signal_overlay.py — does ANY existing repo signal add value on top of SMA(500) for gold?

Base case: SMA(500) with a +/-2% hysteresis band. Sharpe 0.586, in-market 83%, 12 flips/26yr.

A signal only earns its place if it adds INCREMENTAL value — reproducing what the 500d trend
already knows is worthless. Each candidate is therefore tested as a modifier on the base, not
standalone:

  AND   hold gold only when the 500d trend is up AND the signal is bullish  (tightening)
  VETO  hold per the 500d trend, but step aside while the signal is bearish (risk-off overlay)

Signals that are discrete EVENTS rather than states (Gann_BUY_A, break3avg crossings) cannot be
AND-ed without destroying time-in-market, so those are measured as entry timing instead.

Multiple testing is the main hazard here: ~20 candidates against 26 independent years will throw
up two or three "winners" by luck alone. The prior from this repo's history is that nothing adds
value, so the bar is a Deflated-Sharpe-style haircut across the whole candidate set, not a raw
improvement.

Run: ../../vcp_env/bin/python _gold_signal_overlay.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import gold_system as G
from xunlong import XunLongIndicator

TD = 252
RNG = np.random.default_rng(20260803)


def banded_state(g: pd.Series, n: int = 500, band: float = 0.02) -> pd.Series:
    ma = g.rolling(n).mean()
    st = pd.Series(np.nan, index=g.index)
    st[g > ma * (1 + band)] = 1.0
    st[g < ma * (1 - band)] = 0.0
    return st.ffill().fillna(1.0).astype(bool)


def perf(r: pd.Series, sig: pd.Series, rf_d: float) -> dict:
    s = sig.astype(bool)
    curve = (1 + r.where(s, rf_d)).cumprod()
    ret = curve.pct_change().dropna()
    yrs = len(ret) / TD
    cagr = curve.iloc[-1] ** (1 / yrs) - 1
    vol = ret.std(ddof=1) * np.sqrt(TD)
    return {"cagr": cagr * 100, "maxdd": ((curve / curve.cummax()) - 1).min() * 100,
            "sharpe": (cagr - G.RISK_FREE) / vol, "inmkt": s.mean() * 100,
            "flips": int((s.astype(int).diff().abs() == 1).sum())}


def main():
    panel = G.load_panel()
    gd = panel["GC=F"][["Open", "High", "Low", "Close", "Volume"]].dropna()
    g = gd["Close"]
    r = g.pct_change()
    idx = r.dropna().index
    r = r.loc[idx]
    rf_d = (1 + G.RISK_FREE) ** (1 / TD) - 1
    print(f"gold daily {idx[0].date()} -> {idx[-1].date()} ({len(idx)/TD:.0f} independent years)")

    base_sig = banded_state(g).shift(1).reindex(idx).fillna(True)
    base = perf(r, base_sig, rf_d)
    bh = perf(r, pd.Series(True, index=idx), rf_d)
    print(f"\nbuy-and-hold      CAGR {bh['cagr']:5.2f}%  MaxDD {bh['maxdd']:6.1f}%  "
          f"Sharpe {bh['sharpe']:.3f}")
    print(f"BASE SMA500+/-2%  CAGR {base['cagr']:5.2f}%  MaxDD {base['maxdd']:6.1f}%  "
          f"Sharpe {base['sharpe']:.3f}  inMkt {base['inmkt']:.0f}%  flips {base['flips']}\n")

    # ---------------- build candidate STATE signals ------------------------- #
    print("computing xunlong engine on gold ...", flush=True)
    xl = XunLongIndicator().compute(gd, None)
    states: dict[str, pd.Series] = {}

    # -- the repo's own engine
    states["xl: L2_trend>0 (trend active)"] = xl["L2_trend"] > 0
    states["xl: L2_pump>0"] = xl["L2_pump"] > 0
    states["xl: RSI>50"] = xl["RSI"] > 50
    states["xl: RSI>40"] = xl["RSI"] > 40
    states["xl: FJ_value>50"] = xl["FJ_value"] > 50
    states["xl: Rank120>0.5"] = xl["Rank120"] > 0.5
    states["xl: Rank120>0.25"] = xl["Rank120"] > 0.25
    states["xl: above Gann_0"] = g > xl["Gann_0"]
    states["xl: no SELL_trend_break"] = ~xl["SELL_trend_break"].fillna(False).astype(bool)
    states["xl: no SELL_profit_protect"] = ~xl["SELL_profit_protect"].fillna(False).astype(bool)
    states["xl: ema8>ma21"] = xl["ema8"] > xl["ma21"]
    states["xl: obv rising"] = xl["obvS"] > xl["obvS"].rolling(50).mean()

    # -- repo strategies ported to gold
    m8, m22 = g.rolling(8).mean(), g.rolling(22).mean()
    states["break3avg 8/22"] = (g > m8) & (g > m22) & (m8 > m22)
    states["COMBO: above 200d"] = g > g.rolling(200).mean()
    states["85d MA (dip filter)"] = g > g.rolling(85).mean()

    # -- 5xATR trailing stop, applied to the gold sleeve itself
    import scan_stocks as scan          # reuse the repo's Wilder ATR, don't reimplement it
    atr = scan._atr_wilder(gd["High"], gd["Low"], gd["Close"], 22)
    peak = g.cummax()
    states["5xATR22 trail intact"] = g > (peak - 5 * atr)

    # -- macro / cross-asset (the strongest priors for gold specifically)
    dxy = panel["DX-Y.NYB"]["Close"].reindex(g.index).ffill()
    tnx = panel["^TNX"]["Close"].reindex(g.index).ffill()
    spy = panel["SPY"]["Close"].reindex(g.index).ffill()
    gdx = panel["GDX"]["Close"].reindex(g.index).ffill()
    states["DXY below its 200d (weak $)"] = dxy < dxy.rolling(200).mean()
    states["DXY 60d change < 0"] = dxy.pct_change(60) < 0
    states["10y yield below its 200d"] = tnx < tnx.rolling(200).mean()
    states["gold/SPY ratio above 200d"] = (g / spy) > (g / spy).rolling(200).mean()
    states["GDX above its 200d (miners)"] = gdx > gdx.rolling(200).mean()
    states["gold vol < median"] = (r.rolling(60).std() < r.rolling(60).std().rolling(500).median())

    # ---------------- test each as AND / VETO ------------------------------- #
    print("\n=== AND FILTER: hold only when SMA500 up AND signal bullish ===")
    print(f"  {'signal':<34}{'CAGR':>7}{'MaxDD':>8}{'Sharpe':>8}{'dSh':>7}{'inMkt':>7}{'flips':>7}")
    results = []
    for name, s in states.items():
        sig = (base_sig & s.shift(1).reindex(idx).fillna(False))
        p = perf(r, sig, rf_d)
        d = p["sharpe"] - base["sharpe"]
        results.append({"name": name, "mode": "AND", **p, "d": d})
        print(f"  {name:<34}{p['cagr']:>6.2f}%{p['maxdd']:>7.1f}%{p['sharpe']:>8.3f}"
              f"{d:>+7.3f}{p['inmkt']:>6.0f}%{p['flips']:>7}")

    print("\n=== VETO: follow SMA500, but stand aside while the signal is bearish ===")
    print("  (identical to AND for pure state signals; differs only where the signal is")
    print("   undefined early in the sample — shown for the macro overlays that matter)")
    macro = ["DXY below its 200d (weak $)", "DXY 60d change < 0",
             "10y yield below its 200d", "GDX above its 200d (miners)",
             "gold/SPY ratio above 200d"]
    for name in macro:
        s = states[name].shift(1).reindex(idx)
        sig = base_sig & s.fillna(True).astype(bool)     # unknown -> do not veto
        p = perf(r, sig, rf_d)
        d = p["sharpe"] - base["sharpe"]
        results.append({"name": name + " [veto]", "mode": "VETO", **p, "d": d})
        print(f"  {name:<34}{p['cagr']:>6.2f}%{p['maxdd']:>7.1f}%{p['sharpe']:>8.3f}"
              f"{d:>+7.3f}{p['inmkt']:>6.0f}%{p['flips']:>7}")

    # ---------------- ranking + multiple-testing haircut -------------------- #
    rf = pd.DataFrame(results).sort_values("d", ascending=False)
    print(f"\n=== RANKING (of {len(rf)} candidate overlays) ===")
    print(rf[["name", "mode", "sharpe", "d", "maxdd", "inmkt", "flips"]]
          .head(8).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n  candidates BEATING the base: {int((rf.d > 0).sum())} of {len(rf)}")
    print(f"  candidates beating base by >0.05 Sharpe: {int((rf.d > 0.05).sum())}")

    print("\n=== MULTIPLE-TESTING HAIRCUT ===")
    srs = np.array([x / np.sqrt(TD) for x in rf.sharpe.dropna()])   # per-observation
    dsr = deflated_sharpe(srs.max(), srs, len(idx))
    if isinstance(dsr, tuple):
        print(f"  trials {len(srs)}   n_obs {len(idx)}")
        print(f"  best annualised Sharpe {srs.max()*np.sqrt(TD):.3f}   "
              f"SR0 threshold {dsr[1]*np.sqrt(TD):.3f}")
        print(f"  Deflated Sharpe P[SR>SR0] = {dsr[0]:.3f}  "
              f"-> {'clears' if dsr[0] > 0.95 else 'DOES NOT clear'} 95%")
    print(f"  base alone (no candidate search) Sharpe {base['sharpe']:.3f}; the best overlay")
    print(f"  must beat that by more than selection noise across {len(rf)} tries to matter.")

    # ---------------- bootstrap the top candidate --------------------------- #
    top = rf.iloc[0]
    print(f"\n=== BOOTSTRAP the top candidate: {top['name']} ===")
    s = states[top["name"].replace(" [veto]", "")].shift(1).reindex(idx)
    sig = base_sig & (s.fillna(True) if "[veto]" in top["name"] else s.fillna(False)).astype(bool)
    years = sorted({d.year for d in idx})
    diffs = []
    for _ in range(1500):
        pick = RNG.choice(years, size=len(years), replace=True)
        sel = np.concatenate([np.where(idx.year == y)[0] for y in pick])
        a = r.iloc[sel].where(base_sig.iloc[sel], rf_d)
        b = r.iloc[sel].where(sig.iloc[sel], rf_d)
        sa, sb = a.std(ddof=1), b.std(ddof=1)
        if sa > 0 and sb > 0:
            diffs.append((((b.mean() - rf_d) / sb) - ((a.mean() - rf_d) / sa)) * np.sqrt(TD))
    d = np.array(diffs)
    print(f"  advantage over base: mean {d.mean():+.3f}  SE {d.std(ddof=1):.3f}  "
          f"95% CI [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}]")
    print(f"  P(the overlay is actually WORSE than the base) = {(d<0).mean()*100:.0f}%")

    # ---------------- current readings -------------------------------------- #
    print("\n=== CURRENT STATE OF EVERY CANDIDATE (2026-08-03) ===")
    for name, s in states.items():
        v = s.dropna()
        print(f"  {name:<34} {'BULLISH' if bool(v.iloc[-1]) else 'bearish'}")


def deflated_sharpe(best_sr, all_srs, n_obs):
    from math import sqrt
    from statistics import NormalDist
    srs = np.asarray([s for s in all_srs if np.isfinite(s)])
    N = len(srs)
    if N < 2 or n_obs < 10:
        return np.nan
    v = srs.var(ddof=1)
    if v <= 0:
        return np.nan
    nd = NormalDist()
    emc = 0.5772156649
    e_max = (1 - emc) * nd.inv_cdf(1 - 1.0 / N) + emc * nd.inv_cdf(1 - 1.0 / (N * np.e))
    sr0 = sqrt(v) * e_max
    return nd.cdf((best_sr - sr0) * sqrt(n_obs - 1)), sr0


if __name__ == "__main__":
    main()
