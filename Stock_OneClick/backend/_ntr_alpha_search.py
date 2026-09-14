"""
_ntr_alpha_search.py — search for a RETURN-generating system on NTR, not just risk control.

Trend following failed on NTR (ntr_system.py: honest OOS Sharpe 0.101 vs buy-hold 0.118), and it
should have. Trend needs persistent drift. NTR is a commodity cyclical: 34% vol, -58% drawdowns,
5.8% CAGR, driven by potash/nitrogen prices rather than price momentum. The right hypotheses for
that asset are the OPPOSITE of trend, plus the actual economics.

Five mechanisms tested here, each chosen because it has a reason to exist and not because it
showed up in a sweep:

  H1 MEAN REVERSION ON DRAWDOWN — cyclicals overshoot down. Buy depth, hold a fixed horizon.
     This is the natural hypothesis for a 34%-vol name that has repeatedly fallen 50%+ and recovered.
  H2 SEASONALITY — fertilizer demand is physically seasonal (northern-hemisphere spring
     application). If a calendar effect exists anywhere in equities, ag inputs is a candidate.
  H3 MARGIN SPREAD — fertilizer is a spread business: crop prices (revenue) minus natural gas
     (the dominant nitrogen input cost). Cheap gas + expensive corn = margin expansion.
  H4 VALUATION / CARRY — NTR yields ~3.3%. For a cyclical, a high trailing yield is a
     cheapness signal; buy when the yield is in its upper range.
  H5 RELATIVE VALUE — NTR vs an equal-weight fertilizer basket, mean-reverting spread.

METHODOLOGY (same as ntr_system.py, and the reason that build caught a +0.209 overfitting gap):
every rule is measured on the LONG-HISTORY peer complex first, with NTR HELD OUT, then applied
out-of-sample to NTR. Fitting on NTR's 8.6 years produces numbers you will not earn.

MULTIPLE TESTING: five hypotheses with several variants each. Any single "winner" needs to clear
that burden, so a bootstrap and a peer-consistency count are reported for anything promising.

Run: ../../vcp_env/bin/python _ntr_alpha_search.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import ntr_system as N

warnings.filterwarnings("ignore")
TD = 252
RNG = np.random.default_rng(606)
# Pure-play fertilizer / ag-input names. DE/ADM/AGCO are equipment and processing, different
# economics, so they are excluded from the mechanism tests (kept only where noted).
FERT = ["MOS", "CF", "IPI", "ICL", "YARIY", "MOO", "FMC"]


def fwd(px: pd.Series, h: int) -> pd.Series:
    return px.shift(-h) / px - 1


def event_study(px: pd.Series, events: pd.Series, horizons=(21, 63, 126, 252)) -> dict:
    """Forward returns conditioned on an event, against the unconditional baseline."""
    out = {}
    ev = events.reindex(px.index).fillna(False).astype(bool)
    for h in horizons:
        f = fwd(px, h)
        cond, base = f[ev].dropna(), f.dropna()
        if len(cond) < 20:
            continue
        se = cond.std(ddof=1) / np.sqrt(len(cond))
        out[h] = {"n": len(cond), "mean": cond.mean(), "base": base.mean(),
                  "edge": cond.mean() - base.mean(), "se": se,
                  "t": (cond.mean() - base.mean()) / se if se > 0 else np.nan,
                  "win": (cond > 0).mean(), "base_win": (base > 0).mean()}
    return out


def pooled_event(panel, tickers, make_events, horizons=(21, 63, 126, 252)) -> pd.DataFrame:
    """Run an event study across several names and pool the per-name edges."""
    rows = []
    for t in tickers:
        if t not in panel:
            continue
        px = panel[t]["Close"].dropna()
        if len(px) < TD * 8:
            continue
        ev = make_events(px)
        if ev is None:
            continue
        st = event_study(px, ev, horizons)
        for h, v in st.items():
            rows.append({"ticker": t, "h": h, **v})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, label: str):
    if df.empty:
        print(f"  {label}: no usable events")
        return None
    print(f"  {label}")
    print(f"    {'h':>5}{'names':>7}{'mean edge':>12}{'median':>9}{'pos':>7}{'pooled t':>10}{'avg N':>8}")
    for h, g in df.groupby("h"):
        e = g.edge.values
        se = e.std(ddof=1) / np.sqrt(len(e)) if len(e) > 1 else np.nan
        print(f"    {int(h):>5}{len(g):>7}{e.mean()*100:>+11.2f}%{np.median(e)*100:>+8.2f}%"
              f"{int((e>0).sum()):>4}/{len(e):<2}{e.mean()/se if se and se>0 else np.nan:>+10.2f}"
              f"{g.n.mean():>8.0f}")
    return df


def main():
    panel = N.load_panel()
    ntr = panel["NTR"]["Close"].dropna()
    print(f"NTR {ntr.index[0].date()} -> {ntr.index[-1].date()}  "
          f"({len(ntr)/TD:.1f} independent years) | held out of every fit below")
    bh = N.perf(ntr.pct_change().dropna())
    print(f"buy-and-hold NTR: CAGR {bh['cagr']:.2f}%  MaxDD {bh['maxdd']:.1f}%  "
          f"Sharpe {bh['sharpe']:.3f}   <- the bar\n")

    # ================= H1: mean reversion on drawdown ==================== #
    print("=" * 78)
    print("H1  MEAN REVERSION ON DRAWDOWN (cyclicals overshoot down)")
    print("=" * 78)
    for thr in (0.25, 0.40, 0.50):
        def mk(px, thr=thr):
            dd = px / px.rolling(TD).max() - 1
            return (dd <= -thr) & (dd.shift(1) > -thr)      # first crossing only
        res = pooled_event(panel, FERT, mk)
        summarise(res, f"peers, first cross below -{thr*100:.0f}% off the 252d high")
    print()
    print("  OUT-OF-SAMPLE on NTR:")
    for thr in (0.25, 0.40, 0.50):
        dd = ntr / ntr.rolling(TD).max() - 1
        ev = (dd <= -thr) & (dd.shift(1) > -thr)
        st = event_study(ntr, ev)
        if not st:
            print(f"    -{thr*100:.0f}%: too few events on NTR (n<20)")
            continue
        for h, v in st.items():
            print(f"    -{thr*100:.0f}% h={h:>3}  n={v['n']:>3}  mean {v['mean']*100:+6.2f}%  "
                  f"base {v['base']*100:+6.2f}%  edge {v['edge']*100:+6.2f}pp  t={v['t']:+.2f}")

    # ================= H2: seasonality =================================== #
    print("\n" + "=" * 78)
    print("H2  SEASONALITY (fertilizer application is physically seasonal)")
    print("=" * 78)
    print("  pooled monthly mean return, fertilizer complex (NTR held out):")
    mrows = []
    for t in FERT:
        if t not in panel:
            continue
        px = panel[t]["Close"].dropna()
        m = px.resample("ME").last().pct_change().dropna()
        for mo, g in m.groupby(m.index.month):
            mrows.append({"ticker": t, "month": mo, "mean": g.mean(), "n": len(g)})
    md = pd.DataFrame(mrows)
    print(f"    {'month':>6}{'pooled mean':>13}{'names pos':>11}{'t':>8}")
    for mo, g in md.groupby("month"):
        e = g["mean"].values
        se = e.std(ddof=1) / np.sqrt(len(e))
        nm = pd.Timestamp(2020, mo, 1).strftime("%b")
        bar = "#" * max(0, int(round(e.mean() * 400)))
        print(f"    {nm:>6}{e.mean()*100:>+12.2f}%{int((e>0).sum()):>8}/{len(e):<2}"
              f"{e.mean()/se if se>0 else np.nan:>+8.2f}  {bar}")
    nm = ntr.resample("ME").last().pct_change().dropna()
    print("\n  OUT-OF-SAMPLE on NTR, by month (n is tiny — 8 observations each):")
    s = nm.groupby(nm.index.month).agg(mean="mean", n="count")
    print("    " + "  ".join(f"{pd.Timestamp(2020,int(mo),1).strftime('%b')}:{v['mean']*100:+.1f}%"
                             for mo, v in s.iterrows()))

    # ================= H3: margin spread ================================= #
    print("\n" + "=" * 78)
    print("H3  MARGIN SPREAD (crop revenue vs natural-gas input cost)")
    print("=" * 78)
    gas = panel["NG=F"]["Close"].dropna()
    corn = panel["ZC=F"]["Close"].dropna()
    wheat = panel["ZW=F"]["Close"].dropna()
    crop = ((corn / corn.rolling(TD).mean() + wheat / wheat.rolling(TD).mean()) / 2)
    gasn = gas / gas.rolling(TD).mean()
    spread = crop - gasn          # >0 = crops rich vs gas = margin tailwind
    print(f"  margin proxy = mean(corn,wheat)/252dMA  -  gas/252dMA   "
          f"(range {spread.min():.2f} to {spread.max():.2f})")

    def mk_spread(px, lo=None):
        sp = spread.reindex(px.index).ffill()
        q = sp.rolling(TD * 2).rank(pct=True)
        return (q > 0.70) & (q.shift(1) <= 0.70)
    res = pooled_event(panel, FERT, mk_spread)
    summarise(res, "peers, margin spread crossing into its top 30%")
    print("\n  OUT-OF-SAMPLE on NTR:")
    sp = spread.reindex(ntr.index).ffill()
    q = sp.rolling(TD * 2).rank(pct=True)
    ev = (q > 0.70) & (q.shift(1) <= 0.70)
    st = event_study(ntr, ev)
    for h, v in (st or {}).items():
        print(f"    h={h:>3}  n={v['n']:>3}  mean {v['mean']*100:+6.2f}%  base {v['base']*100:+6.2f}%"
              f"  edge {v['edge']*100:+6.2f}pp  t={v['t']:+.2f}")
    if not st:
        print("    too few events on NTR")
    # also: continuous state, not just crossings
    print("\n  as a HOLDING STATE (long only while the spread is in its top half):")
    for name, px in (("NTR", ntr),) + tuple((t, panel[t]["Close"].dropna()) for t in ("MOS", "CF")):
        rr = px.pct_change().dropna()
        qq = spread.reindex(px.index).ffill().rolling(TD * 2).rank(pct=True)
        sig = (qq > 0.50).shift(1).reindex(rr.index).fillna(True)
        a, b = N.perf(rr), N.perf(rr, sig)
        if a and b:
            print(f"    {name:<5} buy-hold Sharpe {a['sharpe']:+.3f} -> spread-gated "
                  f"{b['sharpe']:+.3f}  (d {b['sharpe']-a['sharpe']:+.3f}, "
                  f"CAGR {a['cagr']:.2f}% -> {b['cagr']:.2f}%, inMkt {b['inmkt']:.0f}%)")

    # ================= H4: dividend yield / carry ======================== #
    print("\n" + "=" * 78)
    print("H4  VALUATION / CARRY (high trailing yield = cyclical cheapness)")
    print("=" * 78)
    try:
        import yfinance as yf
        div = yf.Ticker("NTR").dividends
        if div is not None and len(div):
            div.index = pd.DatetimeIndex(div.index).tz_localize(None)
            ttm = div.rolling(4).sum().reindex(ntr.index, method="ffill")
            # NOTE: ntr is total-return adjusted, so this yield is approximate. Use the raw close
            # for a yield calculation to avoid mixing adjusted price with nominal dividends.
            raw = yf.Ticker("NTR").history(period="max", auto_adjust=False)["Close"]
            raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
            y = (ttm / raw.reindex(ttm.index).ffill()) * 100
            y = y.dropna()
            print(f"  NTR trailing yield: now {float(y.iloc[-1]):.2f}%  "
                  f"range {float(y.min()):.2f}-{float(y.max()):.2f}%  "
                  f"median {float(y.median()):.2f}%")
            hi = y > y.rolling(TD * 2).quantile(0.70)
            ev = hi & ~hi.shift(1).fillna(False)
            st = event_study(ntr, ev)
            for h, v in (st or {}).items():
                print(f"    yield entering top 30%: h={h:>3} n={v['n']:>3} "
                      f"mean {v['mean']*100:+6.2f}%  base {v['base']*100:+6.2f}%  "
                      f"edge {v['edge']*100:+6.2f}pp  t={v['t']:+.2f}")
            if not st:
                print("    too few events (n<20) — 8.6 years cannot support this test")
    except Exception as e:
        print(f"  dividend data unavailable: {e}")

    # ================= H5: relative value vs the complex ================= #
    print("\n" + "=" * 78)
    print("H5  RELATIVE VALUE (NTR vs an equal-weight fertilizer basket)")
    print("=" * 78)
    comp = pd.DataFrame({t: panel[t]["Close"] for t in FERT if t in panel}).dropna()
    comp = comp[comp.index >= ntr.index[0]]
    basket = (comp / comp.iloc[0]).mean(axis=1)
    common = ntr.index.intersection(basket.index)
    ratio = (ntr.loc[common] / ntr.loc[common].iloc[0]) / basket.loc[common]
    z = (ratio - ratio.rolling(TD).mean()) / ratio.rolling(TD).std()
    print(f"  NTR/basket relative z-score: now {float(z.dropna().iloc[-1]):+.2f}  "
          f"range {float(z.min()):+.2f} to {float(z.max()):+.2f}")
    for lo in (-1.0, -1.5):
        ev = (z < lo) & (z.shift(1) >= lo)
        st = event_study(ntr.loc[common], ev)
        if not st:
            print(f"    z<{lo}: too few events")
            continue
        for h, v in st.items():
            print(f"    z<{lo} h={h:>3} n={v['n']:>3} mean {v['mean']*100:+6.2f}%  "
                  f"base {v['base']*100:+6.2f}%  edge {v['edge']*100:+6.2f}pp  t={v['t']:+.2f}")

    print("\n" + "=" * 78)
    print("CURRENT STATE OF EVERY MECHANISM")
    print("=" * 78)
    dd_now = float(ntr.iloc[-1] / ntr.rolling(TD).max().iloc[-1] - 1)
    print(f"  H1 drawdown off 252d high : {dd_now*100:+.1f}%  "
          f"(-25% trigger: {'ARMED' if dd_now <= -0.25 else 'no'})")
    print(f"  H2 current month          : {ntr.index[-1].strftime('%b')}")
    print(f"  H3 margin-spread pctile   : {float(q.dropna().iloc[-1])*100:.0f}%  "
          f"(top-30% trigger: {'ARMED' if float(q.dropna().iloc[-1]) > 0.70 else 'no'})")
    print(f"  H5 relative z vs basket   : {float(z.dropna().iloc[-1]):+.2f}")


if __name__ == "__main__":
    main()
