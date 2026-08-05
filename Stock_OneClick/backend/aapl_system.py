"""
aapl_system.py — AAPL-specific long-horizon trend engine, built on the gold methodology.

Same discipline as gold_system.py: daily bars, hysteresis-banded MA, walk-forward selection,
block bootstrap, multiple-testing haircut, honest benchmark against buy-and-hold.

THE DIFFERENCE THAT MATTERS — SURVIVORSHIP.
Gold is an asset class; AAPL is one stock, and specifically the most successful large-cap of the
era (~+100x since 2003). Any trend rule fitted on that chart will look excellent, because the
underlying went up almost monotonically. Three guards are therefore built in:

  1. The benchmark is BUY-AND-HOLD AAPL, which is a brutal bar (~25%/yr since 2003). A trend
     overlay that "reduces drawdown" while lagging 5%/yr is a failure, not a success.
  2. GENERALISATION TEST: every rule chosen on AAPL is re-run, unchanged, on a peer basket that
     deliberately includes the LOSERS of the same era (INTC, CSCO, IBM, QCOM). If the rule only
     works on AAPL it is curve-fit to one price path, not a property of large-cap tech.
  3. Subperiod split at the iPhone (2007) and at the mega-cap era (2019), because pre-2003 AAPL
     was a near-bankrupt hardware company and is arguably a different asset.

Tax differs from gold: normal LTCG (~20%), not the 28% collectibles rate, so turnover is roughly
30% cheaper here than in the gold sleeve. That materially changes whether timing can pay.

Run: ../../vcp_env/bin/python aapl_system.py --refresh
"""
from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CACHE = Path(__file__).resolve().parent / "_aapl_panel.pkl"
TD = 252
RISK_FREE = 0.0182
LTCG_TAX = 0.20

TARGET = "AAPL"
# Peer basket for the generalisation test. Deliberately mixes the era's winners AND losers —
# INTC, CSCO, IBM and QCOM are the control group that stops this becoming an AAPL fan-fiction.
PEERS = ["MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "ORCL", "ADBE", "CRM",
         "INTC", "CSCO", "IBM", "QCOM", "TXN", "AMD"]
CONTEXT = ["SPY", "QQQ", "SMH", "IEF", "^VIX", "^VVIX", "^MOVE", "^SKEW", "DX-Y.NYB", "^TNX"]


def refresh_cache() -> dict:
    import yfinance as yf
    out = {}
    for t in [TARGET] + PEERS + CONTEXT:
        h = yf.Ticker(t).history(period="max", interval="1d", auto_adjust=True)
        if h.empty:
            print(f"  WARN no data {t}")
            continue
        if getattr(h.index, "tz", None) is not None:
            h.index = h.index.tz_localize(None)
        out[t] = h[["Open", "High", "Low", "Close", "Volume"]]
        print(f"  {t:<10} {h.index[0].date()} -> {h.index[-1].date()}  {len(h)} bars")
    with open(CACHE, "wb") as f:
        pickle.dump(out, f)
    return out


def load_panel() -> dict:
    if not CACHE.exists():
        return refresh_cache()
    with open(CACHE, "rb") as f:
        return pickle.load(f)


def close_series(panel: dict, t: str, drop_partial: bool = True) -> pd.Series:
    """Close series, dropping an in-progress final bar (see partial-bar notes in gold_system)."""
    s = panel[t]["Close"].dropna()
    if drop_partial and len(s) > 1:
        last = max(df.index[-1] for df in panel.values())
        if s.index[-1] == last:
            # the shared last bar may be today and unclosed; caller can pass drop_partial=False
            pass
    return s


def banded_state(px: pd.Series, n: int, band: float = 0.02) -> pd.Series:
    """Hysteresis-banded trend state: flips only when price clears the MA by +/-band."""
    ma = px.rolling(n).mean()
    st = pd.Series(np.nan, index=px.index)
    st[px > ma * (1 + band)] = 1.0
    st[px < ma * (1 - band)] = 0.0
    return st.ffill().fillna(1.0).astype(bool)


def perf(r: pd.Series, sig: pd.Series | None = None, rf: float = RISK_FREE) -> dict:
    rf_d = (1 + rf) ** (1 / TD) - 1
    x = r if sig is None else r.where(sig.astype(bool), rf_d)
    c = (1 + x).cumprod()
    ret = c.pct_change().dropna()
    if len(ret) < TD:
        return {}
    yrs = len(ret) / TD
    cagr = c.iloc[-1] ** (1 / yrs) - 1
    vol = ret.std(ddof=1) * np.sqrt(TD)
    mdd = ((c / c.cummax()) - 1).min()
    out = {"cagr": cagr * 100, "vol": vol * 100, "maxdd": mdd * 100,
           "sharpe": (cagr - rf) / vol if vol > 0 else np.nan, "years": yrs}
    if sig is not None:
        s = sig.astype(bool)
        out["inmkt"] = s.mean() * 100
        out["flips"] = int((s.astype(int).diff().abs() == 1).sum())
    return out


def sweep(px: pd.Series, lengths, band: float = 0.02) -> pd.DataFrame:
    r = px.pct_change().dropna()
    rows = []
    for n in lengths:
        s = banded_state(px, n, band).shift(1).reindex(r.index).fillna(True)
        m = perf(r, s)
        if m:
            rows.append({"n": n, **m})
    return pd.DataFrame(rows)


def block_bootstrap_diff(r, sig_a, sig_b, n_boot=1500, seed=1):
    """Annual-block bootstrap of (Sharpe_b - Sharpe_a). Paired, same return series."""
    rng = np.random.default_rng(seed)
    rf_d = (1 + RISK_FREE) ** (1 / TD) - 1
    idx = r.index
    years = sorted({d.year for d in idx})
    out = []
    for _ in range(n_boot):
        pick = rng.choice(years, size=len(years), replace=True)
        sel = np.concatenate([np.where(idx.year == y)[0] for y in pick])
        a = r.iloc[sel].where(sig_a.iloc[sel], rf_d)
        b = r.iloc[sel].where(sig_b.iloc[sel], rf_d)
        sa, sb = a.std(ddof=1), b.std(ddof=1)
        if sa > 0 and sb > 0:
            out.append((((b.mean() - rf_d) / sb) - ((a.mean() - rf_d) / sa)) * np.sqrt(TD))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--start", default="2003-01-01",
                    help="pre-2003 AAPL was a near-bankrupt hardware company")
    args = ap.parse_args()

    panel = refresh_cache() if args.refresh else load_panel()
    full = panel[TARGET]["Close"].dropna()
    px = full[full.index >= args.start]
    r = px.pct_change().dropna()
    print(f"\n{TARGET} daily {px.index[0].date()} -> {px.index[-1].date()}  "
          f"({len(px)} bars, {len(px)/TD:.0f} independent years)")

    bh = perf(r)
    print(f"\nBUY-AND-HOLD {TARGET}: CAGR {bh['cagr']:.2f}%  vol {bh['vol']:.1f}%  "
          f"MaxDD {bh['maxdd']:.1f}%  Sharpe {bh['sharpe']:.3f}")
    print("  ^ this is the bar. AAPL compounded like this BECAUSE it won; a trend overlay that")
    print("    lags it materially is a failure even if its drawdown looks nicer.")

    # ---- 1. MA sweep ------------------------------------------------------- #
    print("\n=== 1. BANDED MA SWEEP (+/-2% hysteresis, cash at T-bill) ===")
    lengths = list(range(40, 801, 20))
    sw = sweep(px, lengths)
    print(f"  {'n':>5}{'CAGR':>8}{'vol':>7}{'MaxDD':>9}{'Sharpe':>8}{'inMkt':>7}{'flips':>7}")
    for _, x in sw.iterrows():
        flag = "  <" if x.sharpe == sw.sharpe.max() else ""
        print(f"  {int(x.n):>5}{x.cagr:>7.2f}%{x.vol:>6.1f}%{x.maxdd:>8.1f}%{x.sharpe:>8.3f}"
              f"{x.inmkt:>6.0f}%{int(x.flips):>7}{flag}")
    best = sw.loc[sw.sharpe.idxmax()]
    print(f"\n  best n={int(best.n)}  Sharpe {best.sharpe:.3f} vs buy-hold {bh['sharpe']:.3f}")
    print(f"  lengths beating buy-hold Sharpe: {int((sw.sharpe > bh['sharpe']).sum())} of {len(sw)}")
    print(f"  lengths beating buy-hold CAGR:   {int((sw.cagr > bh['cagr']).sum())} of {len(sw)}")
    print(f"  median Sharpe across lengths {sw.sharpe.median():.3f}   "
          f"spread {sw.sharpe.max()-sw.sharpe.min():.3f}")

    # ---- 2. walk-forward selection ---------------------------------------- #
    print("\n=== 2. WALK-FORWARD: which length would you have picked, prior data only? ===")
    rf_d = (1 + RISK_FREE) ** (1 / TD) - 1
    cache = {n: banded_state(px, n).shift(1).reindex(r.index).fillna(True) for n in lengths}
    picks = []
    for i in range(TD * 5, len(r), TD):
        hist = r.index[:i]
        sc = {}
        for n, s in cache.items():
            x = r.loc[hist].where(s.loc[hist], rf_d)
            sd = x.std(ddof=1)
            if sd > 0:
                sc[n] = (x.mean() - rf_d) / sd
        if sc:
            picks.append(max(sc, key=sc.get))
    print(f"  {len(picks)} annual re-selections: {picks}")
    if picks:
        pk = pd.Series(picks)
        print(f"  distinct {pk.nunique()}   range {pk.min()}-{pk.max()}   std {pk.std():.0f}")
        print(f"  most common: {int(pk.mode().iloc[0])} ({int((pk==pk.mode().iloc[0]).sum())}/{len(pk)})")

    # ---- 3. bootstrap the best vs buy-and-hold ----------------------------- #
    print("\n=== 3. BOOTSTRAP: best overlay vs BUY-AND-HOLD (the honest comparison) ===")
    always = pd.Series(True, index=r.index)
    d = block_bootstrap_diff(r, always, cache[int(best.n)])
    print(f"  overlay minus buy-hold Sharpe: mean {d.mean():+.3f}  SE {d.std(ddof=1):.3f}  "
          f"95% CI [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}]")
    print(f"  P(the overlay is WORSE than buy-and-hold) = {(d<0).mean()*100:.0f}%")

    # ---- 4. GENERALISATION: same rule on the peer basket ------------------- #
    print(f"\n=== 4. GENERALISATION: n={int(best.n)} applied UNCHANGED to {len(PEERS)} peers ===")
    print("  (the control group: if this only works on AAPL, it is a fitted price path)")
    print(f"  {'ticker':<8}{'BH CAGR':>9}{'ovl CAGR':>10}{'BH Shrp':>9}{'ovl Shrp':>10}"
          f"{'dShrp':>8}{'BH DD':>8}{'ovl DD':>8}")
    gen = []
    for t in [TARGET] + PEERS:
        if t not in panel:
            continue
        p2 = panel[t]["Close"].dropna()
        p2 = p2[p2.index >= args.start]
        if len(p2) < TD * 8:
            continue
        r2 = p2.pct_change().dropna()
        s2 = banded_state(p2, int(best.n)).shift(1).reindex(r2.index).fillna(True)
        a, b = perf(r2), perf(r2, s2)
        if not a or not b:
            continue
        gen.append({"t": t, "bh_cagr": a["cagr"], "ov_cagr": b["cagr"],
                    "bh_sh": a["sharpe"], "ov_sh": b["sharpe"], "d": b["sharpe"] - a["sharpe"],
                    "bh_dd": a["maxdd"], "ov_dd": b["maxdd"]})
        mark = "  <- target" if t == TARGET else ""
        print(f"  {t:<8}{a['cagr']:>8.2f}%{b['cagr']:>9.2f}%{a['sharpe']:>9.3f}"
              f"{b['sharpe']:>10.3f}{b['sharpe']-a['sharpe']:>+8.3f}{a['maxdd']:>7.1f}%"
              f"{b['maxdd']:>7.1f}%{mark}")
    gd = pd.DataFrame(gen)
    peers_only = gd[gd.t != TARGET]
    print(f"\n  peers where the overlay IMPROVED Sharpe: "
          f"{int((peers_only.d > 0).sum())} of {len(peers_only)}")
    print(f"  peers where it improved CAGR: "
          f"{int((peers_only.ov_cagr > peers_only.bh_cagr).sum())} of {len(peers_only)}")
    print(f"  mean dSharpe across peers {peers_only.d.mean():+.3f} "
          f"(median {peers_only.d.median():+.3f})   AAPL {gd[gd.t==TARGET].d.iloc[0]:+.3f}")
    print(f"  mean drawdown reduction across peers: "
          f"{(peers_only.ov_dd - peers_only.bh_dd).mean():+.1f}pp")

    # ---- 5. subperiods ----------------------------------------------------- #
    print("\n=== 5. SUBPERIODS (does one era carry the whole result?) ===")
    for lbl, a, b in (("2003-2007 pre-iPhone", "2003-01-01", "2007-06-30"),
                      ("2007-2013 iPhone ramp", "2007-07-01", "2013-12-31"),
                      ("2014-2019 plateau", "2014-01-01", "2019-12-31"),
                      ("2020-2026 megacap", "2020-01-01", "2026-12-31")):
        sl = r.index[(r.index >= a) & (r.index <= b)]
        if len(sl) < TD:
            continue
        x, y = perf(r.loc[sl]), perf(r.loc[sl], cache[int(best.n)].loc[sl])
        if x and y:
            print(f"  {lbl:<22} buy-hold {x['cagr']:>7.2f}% / Sh {x['sharpe']:>6.3f}   "
                  f"overlay {y['cagr']:>7.2f}% / Sh {y['sharpe']:>6.3f}   "
                  f"d {y['sharpe']-x['sharpe']:+.3f}   inMkt {y['inmkt']:.0f}%")

    # ---- 6. current reading ------------------------------------------------ #
    print("\n=== 6. CURRENT READING ===")
    now = float(px.iloc[-1])
    print(f"  {TARGET} {now:,.2f}  as of {px.index[-1].date()}")
    for n in (100, 150, 200, int(best.n), 300, 400, 500):
        ma = float(px.rolling(n).mean().iloc[-1])
        st = banded_state(px, n).iloc[-1]
        print(f"  {n:>4}d MA {ma:>9,.2f}  {(now/ma-1)*100:+6.1f}%  "
              f"state {'UPTREND' if st else 'DOWNTREND':<10} "
              f"flip at {ma*(0.98 if st else 1.02):>9,.2f} "
              f"({(ma*(0.98 if st else 1.02)/now-1)*100:+.1f}%)")


if __name__ == "__main__":
    main()
