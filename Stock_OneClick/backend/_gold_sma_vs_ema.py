"""
_gold_sma_vs_ema.py — SMA(500) vs EMA(500) as gold's long-term trend filter.

Methodological care: 500 was chosen by optimising an SMA sweep, so a head-to-head at exactly
length 500 is biased in the SMA's favour. Both families are therefore swept across their own
range, and compared three ways:
  1. at matched length (the naive comparison the question implies)
  2. at each family's OWN optimum (the fair comparison)
  3. across the whole sweep — is one family better on average, i.e. is the choice of smoother
     more robust than the choice of length?

Note on parameterisation: an EMA with span=N has the same centre of mass as an SMA(N),
(N-1)/2, so span is the like-for-like knob. The EMA differs by having a long tail — it reacts
faster to recent moves but never fully forgets old data. A median filter is included as a
robustness check because it is the one smoother that ignores outliers entirely.

Run: ../../vcp_env/bin/python _gold_sma_vs_ema.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import gold_system as G

TD = 252
RNG = np.random.default_rng(4242)
LENGTHS = list(range(200, 801, 20))


def smoother(g: pd.Series, n: int, kind: str) -> pd.Series:
    if kind == "SMA":
        return g.rolling(n).mean()
    if kind == "EMA":
        return g.ewm(span=n, adjust=False).mean()
    if kind == "WMA":
        w = np.arange(1, n + 1, dtype=float)
        return g.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)
    if kind == "median":
        return g.rolling(n).median()
    raise ValueError(kind)


def ann(curve: pd.Series):
    ret = curve.pct_change().dropna()
    yrs = len(ret) / TD
    cagr = curve.iloc[-1] ** (1 / yrs) - 1
    vol = ret.std(ddof=1) * np.sqrt(TD)
    return cagr, vol, ((curve / curve.cummax()) - 1).min(), (cagr - G.RISK_FREE) / vol


def build(g, r, idx, rf_d, n, kind, buffer=0.0):
    """Long/cash overlay. buffer creates a hysteresis band around the line."""
    ma = smoother(g, n, kind)
    if buffer <= 0:
        s = (g > ma)
    else:
        st = pd.Series(np.nan, index=g.index)
        st[g > ma * (1 + buffer)] = 1.0
        st[g < ma * (1 - buffer)] = 0.0
        s = st.ffill().fillna(1.0).astype(bool)
    s = s.shift(1).reindex(idx).fillna(False)
    curve = (1 + r.where(s, rf_d)).cumprod()
    flips = int((s.astype(int).diff().abs() == 1).sum())
    return s, curve, flips


def main():
    g = G.load_panel()["GC=F"]["Close"].dropna()
    r = g.pct_change()
    idx = r.dropna().index
    r = r.loc[idx]
    rf_d = (1 + G.RISK_FREE) ** (1 / TD) - 1
    yrs_total = len(idx) / TD
    print(f"gold daily {idx[0].date()} -> {idx[-1].date()}  "
          f"({len(idx)} bars, {yrs_total:.0f} independent years)")
    bh = ann((1 + r).cumprod())
    print(f"buy-and-hold: CAGR {bh[0]*100:.2f}%  vol {bh[1]*100:.1f}%  "
          f"MaxDD {bh[2]*100:.1f}%  Sharpe {bh[3]:.3f}\n")

    # ---- 1. head-to-head at 500 ------------------------------------------- #
    print("=== 1. NAIVE HEAD-TO-HEAD AT LENGTH 500 (biased toward SMA: 500 was fitted on SMA) ===")
    print(f"  {'smoother':<10}{'CAGR':>8}{'vol':>7}{'MaxDD':>9}{'Sharpe':>8}{'inMkt':>7}{'flips/yr':>10}")
    for kind in ("SMA", "EMA", "WMA", "median"):
        s, c, f = build(g, r, idx, rf_d, 500, kind)
        a = ann(c)
        print(f"  {kind:<10}{a[0]*100:>7.2f}%{a[1]*100:>6.1f}%{a[2]*100:>8.1f}%{a[3]:>8.3f}"
              f"{s.mean()*100:>6.0f}%{f/yrs_total:>10.1f}")

    # ---- 2. full sweep, both families ------------------------------------- #
    print("\n=== 2. FULL SWEEP: each family's OWN optimum (the fair comparison) ===")
    res = {}
    for kind in ("SMA", "EMA"):
        rows = []
        for n in LENGTHS:
            s, c, f = build(g, r, idx, rf_d, n, kind)
            a = ann(c)
            rows.append({"n": n, "cagr": a[0] * 100, "maxdd": a[2] * 100,
                         "sharpe": a[3], "inmkt": s.mean() * 100, "flips_yr": f / yrs_total})
        res[kind] = pd.DataFrame(rows)
    comp = pd.DataFrame({"days": res["SMA"].n,
                         "SMA": res["SMA"].sharpe.values,
                         "EMA": res["EMA"].sharpe.values})
    comp["EMA-SMA"] = comp.EMA - comp.SMA
    print(comp.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    for kind in ("SMA", "EMA"):
        d = res[kind]
        b = d.loc[d.sharpe.idxmax()]
        print(f"\n  {kind}: best n={int(b.n)}  Sharpe {b.sharpe:.3f}  CAGR {b.cagr:.2f}%  "
              f"MaxDD {b.maxdd:.1f}%  flips/yr {b.flips_yr:.1f}")
        print(f"       median Sharpe across lengths {d.sharpe.median():.3f}   "
              f"worst {d.sharpe.min():.3f}   spread {d.sharpe.max()-d.sharpe.min():.3f}")
        print(f"       lengths beating buy-and-hold: {int((d.sharpe>bh[3]).sum())} of {len(d)}")
    print(f"\n  EMA minus SMA at matched length: mean {comp['EMA-SMA'].mean():+.3f}  "
          f"median {comp['EMA-SMA'].median():+.3f}  "
          f"EMA wins {int((comp['EMA-SMA']>0).sum())} of {len(comp)} lengths")

    # ---- 3. bootstrap the family difference ------------------------------- #
    print("\n=== 3. BLOCK BOOTSTRAP: is the family difference real? ===")
    sma_n = int(res["SMA"].loc[res["SMA"].sharpe.idxmax(), "n"])
    ema_n = int(res["EMA"].loc[res["EMA"].sharpe.idxmax(), "n"])
    pairs = [(500, 500, "SMA500 vs EMA500 (matched)"),
             (sma_n, ema_n, f"SMA{sma_n} vs EMA{ema_n} (each best)")]
    years = sorted({d.year for d in idx})
    for a_n, b_n, lbl in pairs:
        sa, _, _ = build(g, r, idx, rf_d, a_n, "SMA")
        sb, _, _ = build(g, r, idx, rf_d, b_n, "EMA")
        diffs = []
        for _ in range(1500):
            pick = RNG.choice(years, size=len(years), replace=True)
            sel = np.concatenate([np.where(idx.year == y)[0] for y in pick])
            x = r.iloc[sel].where(sa.iloc[sel], rf_d)
            y = r.iloc[sel].where(sb.iloc[sel], rf_d)
            xs, ys = x.std(ddof=1), y.std(ddof=1)
            if xs > 0 and ys > 0:
                diffs.append((((y.mean() - rf_d) / ys) - ((x.mean() - rf_d) / xs)) * np.sqrt(TD))
        d = np.array(diffs)
        print(f"  {lbl}")
        print(f"     EMA advantage: mean {d.mean():+.3f}  SE {d.std(ddof=1):.3f}  "
              f"95% CI [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}]  "
              f"P(EMA worse) = {(d<0).mean()*100:.0f}%")

    # ---- 4. walk-forward: would you have picked EMA or SMA? --------------- #
    print("\n=== 4. WALK-FORWARD: family + length chosen on prior data only ===")
    cache = {(k, n): build(g, r, idx, rf_d, n, k)[0] for k in ("SMA", "EMA") for n in LENGTHS}
    picks = []
    for i in range(TD * 10, len(idx), TD):
        hist = idx[:i]
        best, sc = None, -9
        for (k, n), s in cache.items():
            x = r.loc[hist].where(s.loc[hist], rf_d)
            sd = x.std(ddof=1)
            if sd > 0:
                v = (x.mean() - rf_d) / sd
                if v > sc:
                    sc, best = v, (k, n)
        picks.append(best)
    pk = pd.DataFrame(picks, columns=["kind", "n"])
    print(f"  re-selections: {len(pk)}   SMA chosen {int((pk.kind=='SMA').sum())}x, "
          f"EMA chosen {int((pk.kind=='EMA').sum())}x")
    print(f"  choices: {[f'{k}{n}' for k, n in picks]}")

    # ---- 5. whipsaw + hysteresis ----------------------------------------- #
    print("\n=== 5. WHIPSAW: crossovers over 26 years, with and without a band ===")
    print(f"  {'':<12}{'raw':>10}{'+/-2%':>10}{'+/-5%':>10}")
    for kind in ("SMA", "EMA"):
        out = []
        for buf in (0.0, 0.02, 0.05):
            _, _, f = build(g, r, idx, rf_d, 500, kind, buffer=buf)
            out.append(f)
        print(f"  {kind+'(500)':<12}{out[0]:>10}{out[1]:>10}{out[2]:>10}")
    print("\n  Sharpe with a +/-2% hysteresis band (the practically tradeable version):")
    for kind in ("SMA", "EMA"):
        _, c, f = build(g, r, idx, rf_d, 500, kind, buffer=0.02)
        a = ann(c)
        print(f"    {kind}(500) +/-2%:  CAGR {a[0]*100:5.2f}%  MaxDD {a[2]*100:6.1f}%  "
              f"Sharpe {a[3]:.3f}  flips {f} ({f/yrs_total:.1f}/yr)")

    # ---- 6. episode attribution ------------------------------------------ #
    print("\n=== 6. DRAWDOWN EPISODES: does EMA protect differently? ===")
    bhc = (1 + r).cumprod()
    bdd = (bhc / bhc.cummax()) - 1
    curves = {k: build(g, r, idx, rf_d, 500, k)[1] for k in ("SMA", "EMA")}
    inep, st = False, None
    for dt, v in bdd.items():
        if v < -0.15 and not inep:
            inep, st = True, dt
        if v > -0.02 and inep:
            inep = False
            seg = f"  {st.date()} -> {dt.date()}  buy-hold {bdd[st:dt].min()*100:6.1f}%"
            for k, c in curves.items():
                sub = c[st:dt]
                seg += f"   {k} {((sub/sub.cummax())-1).min()*100:6.1f}%"
            print(seg)

    # ---- 7. current reading ---------------------------------------------- #
    print("\n=== 7. CURRENT READING ===")
    px = float(g.iloc[-1])
    print(f"  gold {px:,.2f}  as of {g.index[-1].date()}")
    for kind in ("SMA", "EMA", "WMA", "median"):
        ma = float(smoother(g, 500, kind).iloc[-1])
        print(f"  {kind:<8}500 {ma:>9,.2f}   {(px/ma-1)*100:+6.1f}%   "
              f"{'ABOVE' if px > ma else 'BELOW'}")


if __name__ == "__main__":
    main()
