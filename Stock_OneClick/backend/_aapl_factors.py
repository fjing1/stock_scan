"""
_aapl_factors.py — do any of the factors we've discussed add value on top of an AAPL trend base?

Base: 200-day SMA with a +/-2% hysteresis band (the in-sample best from aapl_system.py).

This is the same overlay harness used for gold, but the prior is DIFFERENT and more favourable:
the repo's signals (xunlong/观海, break3avg, 5xATR) were designed for equities, and break3avg has
a documented real forward edge on this repo's stock universe. Gold had no earnings, no sector, no
equity correlation, so those signals had no mechanism there. AAPL does.

Also tested here and not applicable to gold:
  • PEAD — the beat-and-selloff drift. Measured earlier on AAPL: 11 unique events, D20 excess vs
    SPY was NEGATIVE in 11 of 11, mean -5.11%, t=-3.46. That is the single strongest AAPL-specific
    statistical finding in this whole project, so it gets tested as a live overlay.
  • Relative strength vs SPY/QQQ, and the market's own regime (AAPL is ~1.2 beta, so an AAPL trend
    rule is partly a market-timing rule in disguise).

Any candidate that beats the base is then re-run UNCHANGED on the peer basket. AAPL alone proves
nothing — the generalisation test is what separates a factor from a fitted price path.

Run: ../../vcp_env/bin/python _aapl_factors.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import aapl_system as A
import scan_stocks as scan
from xunlong import XunLongIndicator

TD = 252
RNG = np.random.default_rng(4711)
START = "2003-01-01"
BASE_N = 200


def main():
    panel = A.load_panel()
    ohlc = panel["AAPL"][["Open", "High", "Low", "Close", "Volume"]].dropna()
    ohlc = ohlc[ohlc.index >= START]
    px = ohlc["Close"]
    r = px.pct_change().dropna()
    idx = r.index
    rf_d = (1 + A.RISK_FREE) ** (1 / TD) - 1

    base_sig = A.banded_state(px, BASE_N).shift(1).reindex(idx).fillna(True)
    base = A.perf(r, base_sig)
    bh = A.perf(r)
    print(f"AAPL {idx[0].date()} -> {idx[-1].date()} ({len(idx)/TD:.0f} independent years)")
    print(f"buy-and-hold      CAGR {bh['cagr']:6.2f}%  MaxDD {bh['maxdd']:6.1f}%  "
          f"Sharpe {bh['sharpe']:.3f}")
    print(f"BASE 200d +/-2%   CAGR {base['cagr']:6.2f}%  MaxDD {base['maxdd']:6.1f}%  "
          f"Sharpe {base['sharpe']:.3f}  inMkt {base['inmkt']:.0f}%\n")

    def ser(t):
        return panel[t]["Close"].reindex(px.index).ffill() if t in panel else None

    spy, qqq = ser("SPY"), ser("QQQ")
    vix, vvix, move, skew = ser("^VIX"), ser("^VVIX"), ser("^MOVE"), ser("^SKEW")
    dxy, tnx = ser("DX-Y.NYB"), ser("^TNX")

    print("computing xunlong engine on AAPL ...", flush=True)
    xl = XunLongIndicator().compute(ohlc, None)

    C: dict[str, pd.Series] = {}
    # ---- the repo's own engine (designed for equities — the real test) ----
    C["xl: L2_trend>0"] = xl["L2_trend"] > 0
    C["xl: RSI>50"] = xl["RSI"] > 50
    C["xl: RSI>40"] = xl["RSI"] > 40
    C["xl: FJ_value>50"] = xl["FJ_value"] > 50
    C["xl: Rank120>0.5"] = xl["Rank120"] > 0.5
    C["xl: Rank120>0.25"] = xl["Rank120"] > 0.25
    C["xl: above Gann_0"] = px > xl["Gann_0"]
    C["xl: no SELL_trend_break"] = ~xl["SELL_trend_break"].fillna(False).astype(bool)
    C["xl: no SELL_profit_protect"] = ~xl["SELL_profit_protect"].fillna(False).astype(bool)
    C["xl: ema8>ma21"] = xl["ema8"] > xl["ma21"]
    C["xl: obv rising"] = xl["obvS"] > xl["obvS"].rolling(50).mean()

    # ---- break3avg: the one signal with a documented stock edge ----
    m8, m22 = px.rolling(8).mean(), px.rolling(22).mean()
    C["break3avg 8/22"] = (px > m8) & (px > m22) & (m8 > m22)
    m23 = px.rolling(23).mean()
    C["break3avg 8/23"] = (px > m8) & (px > m23) & (m8 > m23)

    # ---- their 5xATR trailing exit ----
    atr = scan._atr_wilder(ohlc["High"], ohlc["Low"], ohlc["Close"], 22)
    C["5xATR22 trail intact"] = px > (px.cummax() - 5 * atr)
    C["3xATR22 trail intact"] = px > (px.cummax() - 3 * atr)

    # ---- dip-in-uptrend ----
    C["above 85d MA"] = px > px.rolling(85).mean()
    C["below 85d MA (buy dip)"] = px < px.rolling(85).mean()

    # ---- relative strength / market regime ----
    if spy is not None:
        rs = px / spy
        C["RS vs SPY above 200d"] = rs > rs.rolling(200).mean()
        C["SPY above its 200d (mkt regime)"] = spy > spy.rolling(200).mean()
    if qqq is not None:
        rq = px / qqq
        C["RS vs QQQ above 200d"] = rq > rq.rolling(200).mean()

    # ---- stress gauges (destroyed value on gold — do they here?) ----
    if vix is not None:
        C["VIX below its 200d (calm)"] = vix < vix.rolling(200).mean()
        C["VIX < 20"] = vix < 20
        C["VIX above its 200d (stress)"] = vix > vix.rolling(200).mean()
    if vvix is not None:
        C["VVIX below its 200d"] = vvix < vvix.rolling(200).mean()
    if move is not None:
        C["MOVE below its 200d"] = move < move.rolling(200).mean()
    if skew is not None:
        C["SKEW above its 200d"] = skew > skew.rolling(200).mean()
    if dxy is not None:
        C["DXY below its 200d"] = dxy < dxy.rolling(200).mean()
    if tnx is not None:
        C["10y yield below its 200d"] = tnx < tnx.rolling(200).mean()

    # ---- PEAD: sit out the 20 trading days after a beat-and-selloff ----
    pead = pead_exclusion(px, idx)
    if pead is not None:
        C["not in PEAD drift window (20d)"] = pead

    # ---- run the overlays ----
    print("\n=== AND FILTER: hold only when 200d trend up AND factor bullish ===")
    print(f"  {'factor':<36}{'CAGR':>8}{'MaxDD':>8}{'Sharpe':>8}{'dSh':>7}{'inMkt':>7}{'flips':>7}")
    rows = []
    for name, s in C.items():
        sv = s.shift(1).reindex(idx)
        valid = sv.notna()
        sig = base_sig & sv.fillna(False).astype(bool)
        m = A.perf(r.loc[valid], sig.loc[valid])
        b2 = A.perf(r.loc[valid], base_sig.loc[valid])
        if not m or not b2:
            continue
        d = m["sharpe"] - b2["sharpe"]
        rows.append({"name": name, **m, "d": d})
        print(f"  {name:<36}{m['cagr']:>7.2f}%{m['maxdd']:>7.1f}%{m['sharpe']:>8.3f}"
              f"{d:>+7.3f}{m['inmkt']:>6.0f}%{int(m['flips']):>7}")

    rf_ = pd.DataFrame(rows).sort_values("d", ascending=False)
    print(f"\n=== RANKING (of {len(rf_)} factors) ===")
    print(rf_[["name", "cagr", "maxdd", "sharpe", "d", "inmkt"]].head(8)
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n  beating the base: {int((rf_.d > 0).sum())} of {len(rf_)}   "
          f"by >0.05: {int((rf_.d > 0.05).sum())}")
    print(f"  beating buy-and-hold CAGR ({bh['cagr']:.2f}%): "
          f"{int((rf_.cagr > bh['cagr']).sum())} of {len(rf_)}")

    # ---- bootstrap + generalisation for the winners ----
    winners = rf_[rf_.d > 0.05].head(4)
    for _, w in winners.iterrows():
        name = w["name"]
        print(f"\n=== VALIDATE: {name} ===")
        sv = C[name].shift(1).reindex(idx)
        valid = sv.notna()
        sig = (base_sig & sv.fillna(False).astype(bool)).loc[valid]
        d = A.block_bootstrap_diff(r.loc[valid], base_sig.loc[valid], sig)
        print(f"  bootstrap vs base: mean {d.mean():+.3f}  SE {d.std(ddof=1):.3f}  "
              f"95% CI [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}]  "
              f"P(worse) {(d<0).mean()*100:.0f}%")
        # generalisation: same factor + same base on the peers
        gen = []
        for t in A.PEERS:
            if t not in panel:
                continue
            o2 = panel[t][["Open", "High", "Low", "Close", "Volume"]].dropna()
            o2 = o2[o2.index >= START]
            if len(o2) < TD * 8:
                continue
            p2 = o2["Close"]
            r2 = p2.pct_change().dropna()
            b2s = A.banded_state(p2, BASE_N).shift(1).reindex(r2.index).fillna(True)
            f2 = rebuild_factor(name, o2, p2, panel, xl_cache={})
            if f2 is None:
                continue
            f2 = f2.shift(1).reindex(r2.index).fillna(False).astype(bool)
            mb, mf = A.perf(r2, b2s), A.perf(r2, b2s & f2)
            if mb and mf:
                gen.append(mf["sharpe"] - mb["sharpe"])
        if gen:
            gen = np.array(gen)
            print(f"  generalisation across {len(gen)} peers: mean {gen.mean():+.3f}  "
                  f"median {np.median(gen):+.3f}  improved {int((gen>0).sum())}/{len(gen)}")
            print(f"  -> AAPL {w['d']:+.3f} vs peer mean {gen.mean():+.3f}: "
                  f"{'GENERALISES' if gen.mean() > 0.02 else 'AAPL-SPECIFIC (likely fitted)'}")

    print("\n=== CURRENT STATE OF EVERY FACTOR ===")
    for name, s in C.items():
        v = s.dropna()
        if len(v):
            print(f"  {name:<36} {'BULLISH' if bool(v.iloc[-1]) else 'bearish'}")


def pead_exclusion(px: pd.Series, idx: pd.DatetimeIndex):
    """False for the 20 trading days after a beat-that-sold-off; True otherwise."""
    try:
        import yfinance as yf
        e = yf.Ticker("AAPL").get_earnings_dates(limit=90)
    except Exception:
        return None
    if e is None or e.empty:
        return None
    e = e[e["Surprise(%)"].notna()].copy()
    e.index = pd.DatetimeIndex(e.index).tz_localize(None)
    out = pd.Series(True, index=px.index)
    r = px.pct_change()
    n = 0
    for dt, row in e.iterrows():
        nxt = px.index[px.index > dt.normalize()]
        if len(nxt) == 0:
            continue
        d = nxt[0]
        i = px.index.get_loc(d)
        if row["Surprise(%)"] > 0 and r.iloc[i] <= -0.05:
            out.iloc[i:min(i + 21, len(out))] = False
            n += 1
    print(f"  (PEAD: {n} beat-and-selloff events flagged, 20 trading days excluded each)")
    return out


def rebuild_factor(name, ohlc, px, panel, xl_cache):
    """Rebuild a factor for a peer ticker so the generalisation test uses the SAME rule."""
    def s(t):
        return panel[t]["Close"].reindex(px.index).ffill() if t in panel else None
    if name.startswith("break3avg"):
        long_n = 23 if "8/23" in name else 22
        m8, ml = px.rolling(8).mean(), px.rolling(long_n).mean()
        return (px > m8) & (px > ml) & (m8 > ml)
    if "ATR22 trail" in name:
        mult = 3 if name.startswith("3x") else 5
        atr = scan._atr_wilder(ohlc["High"], ohlc["Low"], ohlc["Close"], 22)
        return px > (px.cummax() - mult * atr)
    if name == "above 85d MA":
        return px > px.rolling(85).mean()
    if name == "below 85d MA (buy dip)":
        return px < px.rolling(85).mean()
    if name.startswith("RS vs "):
        b = s("SPY" if "SPY" in name else "QQQ")
        if b is None:
            return None
        rs = px / b
        return rs > rs.rolling(200).mean()
    for key, t, op in (("VIX below", "^VIX", "lt"), ("VIX above", "^VIX", "gt"),
                       ("VVIX below", "^VVIX", "lt"), ("MOVE below", "^MOVE", "lt"),
                       ("SKEW above", "^SKEW", "gt"), ("DXY below", "DX-Y.NYB", "lt"),
                       ("10y yield below", "^TNX", "lt"),
                       ("SPY above", "SPY", "gt")):
        if name.startswith(key):
            x = s(t)
            if x is None:
                return None
            m = x.rolling(200).mean()
            return (x < m) if op == "lt" else (x > m)
    if name == "VIX < 20":
        x = s("^VIX")
        return None if x is None else x < 20
    if name.startswith("xl:") or name.startswith("not in PEAD"):
        return None      # engine/earnings factors are per-ticker; skip in generalisation
    return None


if __name__ == "__main__":
    main()
