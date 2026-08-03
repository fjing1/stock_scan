"""
_gold_ma_research.py — is there a defensible moving-average length for gold's long-term trend?

Motivation: the 18-month MA in gold_system.py was never justified. It was inherited from a
first-pass sweep where lengths 16-24 all reported an identical MaxDD, which looked like a
"plateau" but actually meant every one of them caught the same single exit (N=1).

The length is not cosmetic: as of 2026-07 gold sits BELOW its 10m/12m MA and ABOVE its 18m/24m,
so the choice alone decides whether the trend reads "intact" or "broken".

Tests, in order of how much they should move the conclusion:
  1. Full sweep 3-36 months on several metrics -> is any length actually best, or is it flat?
  2. Block-bootstrap CI on (best length - median length) -> is the spread even real?
  3. Walk-forward selection -> if you picked the best length using only prior data, what would
     you have held at each date, and does that choice drift? Drift = the parameter is noise.
  4. Decision sensitivity -> at each historical month, what FRACTION of lengths said "uptrend"?
     A robust reading is one where lengths agree. Where they disagree, no single length is a fact.
  5. Ensemble (fraction-of-lengths vote) vs any single length.
  6. Smoother family: SMA vs EMA vs median-filter.
  7. Multiple-testing haircut over the whole sweep.

Run: ../../vcp_env/bin/python _gold_ma_research.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import gold_system as G

RNG = np.random.default_rng(12345)
LENGTHS = list(range(3, 37))


def load():
    panel = G.load_panel()
    m = G.monthly(panel)
    gold = m["GC=F"].dropna()
    px = m[list(G.EQUITY_PROXY)].dropna()
    eq = sum(px[t].pct_change() * w for t, w in G.EQUITY_PROXY.items())
    return gold, gold.pct_change(), eq


def overlay(gold_ret, sig, rf_monthly):
    """Long/cash overlay; cash earns the risk-free rate (0% cash unfairly penalises timing)."""
    r = gold_ret.where(sig, rf_monthly)
    return (1 + r).cumprod()


def ann(curve):
    ret = curve.pct_change().dropna()
    yrs = len(ret) / 12
    cagr = curve.iloc[-1] ** (1 / yrs) - 1
    vol = ret.std(ddof=1) * np.sqrt(12)
    mdd = ((curve / curve.cummax()) - 1).min()
    return cagr, vol, mdd, (cagr - G.RISK_FREE) / vol if vol > 0 else np.nan


def main():
    gold, gr, eq = load()
    rf_m = (1 + G.RISK_FREE) ** (1 / 12) - 1
    idx = gr.dropna().index
    gr = gr.loc[idx]
    print(f"gold monthly {idx[0].date()} -> {idx[-1].date()}  "
          f"({len(idx)} months, {len(idx)/12:.0f} independent years)\n")

    bh = (1 + gr).cumprod()
    bc, bv, bd, bs = ann(bh)
    print(f"buy-and-hold gold: CAGR {bc*100:.2f}%  vol {bv*100:.1f}%  "
          f"MaxDD {bd*100:.1f}%  Sharpe {bs:.3f}\n")

    # ---- 1. full sweep ----------------------------------------------------- #
    print("=== 1. FULL SWEEP (SMA, long/cash with cash at T-bill) ===")
    rows = []
    sigs = {}
    for n in LENGTHS:
        s = (gold > gold.rolling(n).mean()).shift(1).reindex(idx).fillna(False)
        sigs[n] = s
        c, v, d, sh = ann(overlay(gr, s, rf_m))
        flips = int((s.astype(int).diff() != 0).sum())
        rows.append({"n": n, "cagr": c * 100, "vol": v * 100, "maxdd": d * 100,
                     "sharpe": sh, "inmkt": s.mean() * 100, "flips": flips})
    sw = pd.DataFrame(rows)
    print(sw.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    best = sw.loc[sw.sharpe.idxmax()]
    print(f"\n  best Sharpe: n={int(best.n)} at {best.sharpe:.3f}   "
          f"median across lengths {sw.sharpe.median():.3f}   "
          f"worst {sw.sharpe.min():.3f} (n={int(sw.loc[sw.sharpe.idxmin(),'n'])})")
    print(f"  lengths beating buy-and-hold Sharpe ({bs:.3f}): "
          f"{int((sw.sharpe > bs).sum())} of {len(sw)}")
    print(f"  n=18 rank: {int((sw.sharpe > sw.loc[sw.n==18,'sharpe'].iloc[0]).sum())+1} "
          f"of {len(sw)} (Sharpe {sw.loc[sw.n==18,'sharpe'].iloc[0]:.3f})")

    # ---- 2. is the spread real? ------------------------------------------- #
    print("\n=== 2. BLOCK BOOTSTRAP: is (best - median) distinguishable from zero? ===")
    med_n = int(sw.n.median())
    yrs = sorted({d.year for d in idx})
    B = 2000
    diffs = []
    for _ in range(B):
        pick = RNG.choice(yrs, size=len(yrs), replace=True)
        sel = np.concatenate([np.where(idx.year == y)[0] for y in pick])
        g2 = gr.iloc[sel]
        a = gr.iloc[sel].where(sigs[int(best.n)].iloc[sel], rf_m)
        b = gr.iloc[sel].where(sigs[med_n].iloc[sel], rf_m)
        sa = (a.mean() - rf_m) / a.std(ddof=1) if a.std(ddof=1) > 0 else np.nan
        sb = (b.mean() - rf_m) / b.std(ddof=1) if b.std(ddof=1) > 0 else np.nan
        if np.isfinite(sa) and np.isfinite(sb):
            diffs.append((sa - sb) * np.sqrt(12))
    diffs = np.array(diffs)
    print(f"  best n={int(best.n)} vs median n={med_n}:  mean diff {diffs.mean():+.3f}  "
          f"SE {diffs.std(ddof=1):.3f}  95% CI [{np.percentile(diffs,2.5):+.3f}, "
          f"{np.percentile(diffs,97.5):+.3f}]")
    print(f"  P(best is actually worse than median) = {(diffs<0).mean()*100:.1f}%")

    # ---- 3. walk-forward selection ---------------------------------------- #
    print("\n=== 3. WALK-FORWARD: which length would you have PICKED, using only prior data? ===")
    picks = []
    for i in range(120, len(idx), 12):          # re-select annually after a 10y burn-in
        hist = idx[:i]
        scores = {}
        for n in LENGTHS:
            s = sigs[n].loc[hist]
            r = gr.loc[hist].where(s, rf_m)
            sd = r.std(ddof=1)
            if sd > 0:
                scores[n] = (r.mean() - rf_m) / sd
        if scores:
            picks.append({"date": idx[i], "best_n": max(scores, key=scores.get)})
    pk = pd.DataFrame(picks)
    print(f"  annual re-selections: {len(pk)}")
    print(f"  chosen length over time: {pk.best_n.tolist()}")
    print(f"  distinct choices: {pk.best_n.nunique()}   range {pk.best_n.min()}-{pk.best_n.max()}"
          f"   std {pk.best_n.std():.1f}")
    print("  -> a stable parameter would keep picking the same length; drift means it is noise.")

    # ---- 4. decision sensitivity ------------------------------------------ #
    print("\n=== 4. DO THE LENGTHS EVEN AGREE? (fraction voting 'uptrend' each month) ===")
    votes = pd.DataFrame({n: sigs[n] for n in LENGTHS}).astype(float)
    frac = votes.mean(axis=1)
    print(f"  months where ALL lengths agree:        {int(((frac==0)|(frac==1)).sum())} "
          f"of {len(frac)} ({((frac==0)|(frac==1)).mean()*100:.0f}%)")
    print(f"  months where lengths are split 40-60%: {int(((frac>0.4)&(frac<0.6)).sum())} "
          f"({((frac>0.4)&(frac<0.6)).mean()*100:.0f}%)")
    print(f"  CURRENT vote: {frac.iloc[-1]*100:.0f}% of lengths say uptrend")
    up = [n for n in LENGTHS if bool(sigs[n].iloc[-1])]
    print(f"  lengths currently saying UPTREND: {up}")
    print(f"  lengths currently saying DOWNTREND: {[n for n in LENGTHS if n not in up]}")

    # ---- 5. ensemble ------------------------------------------------------ #
    print("\n=== 5. ENSEMBLE (hold gold when >=X% of lengths agree) vs single lengths ===")
    for thr in (0.3, 0.5, 0.7):
        s = (frac >= thr)
        c, v, d, sh = ann(overlay(gr, s, rf_m))
        flips = int((s.astype(int).diff() != 0).sum())
        print(f"  vote >= {thr*100:3.0f}%:  CAGR {c*100:5.2f}%  vol {v*100:4.1f}%  "
              f"MaxDD {d*100:6.1f}%  Sharpe {sh:.3f}  flips {flips}")
    print(f"  for reference, single-length Sharpe range: "
          f"{sw.sharpe.min():.3f} to {sw.sharpe.max():.3f}")

    # ---- 6. smoother family ----------------------------------------------- #
    print("\n=== 6. SMOOTHER FAMILY at a few lengths (does the TYPE matter?) ===")
    for n in (10, 12, 18, 24):
        out = []
        for lbl, ma in (("SMA", gold.rolling(n).mean()),
                        ("EMA", gold.ewm(span=n, adjust=False).mean()),
                        ("median", gold.rolling(n).median())):
            s = (gold > ma).shift(1).reindex(idx).fillna(False)
            _, _, _, sh = ann(overlay(gr, s, rf_m))
            out.append(f"{lbl} {sh:.3f}")
        print(f"  n={n:>2}:  " + "   ".join(out))

    # ---- 7. multiple-testing haircut -------------------------------------- #
    print("\n=== 7. MULTIPLE-TESTING HAIRCUT over the sweep ===")
    srs = sw.sharpe.dropna().values / np.sqrt(12)      # per-observation Sharpes
    n_obs = len(idx)
    dsr = deflated_sharpe(srs.max(), srs, n_obs)
    if isinstance(dsr, tuple):
        print(f"  trials {len(srs)}  n_obs {n_obs}")
        print(f"  best annualised Sharpe {srs.max()*np.sqrt(12):.3f}  "
              f"SR0 threshold {dsr[1]*np.sqrt(12):.3f} annualised")
        print(f"  Deflated Sharpe P[SR>SR0] = {dsr[0]:.3f}  "
              f"-> {'clears' if dsr[0] > 0.95 else 'DOES NOT CLEAR'} the 95% bar")


def deflated_sharpe(best_sr, all_srs, n_obs):
    """Bailey & Lopez de Prado DSR. Inputs must be PER-OBSERVATION Sharpes."""
    from math import sqrt
    from statistics import NormalDist
    srs = np.asarray([s for s in all_srs if np.isfinite(s)])
    N = len(srs)
    if N < 2 or n_obs < 10:
        return np.nan
    var_sr = srs.var(ddof=1)
    if var_sr <= 0:
        return np.nan
    nd = NormalDist()
    emc = 0.5772156649
    e_max = (1 - emc) * nd.inv_cdf(1 - 1.0 / N) + emc * nd.inv_cdf(1 - 1.0 / (N * np.e))
    sr0 = sqrt(var_sr) * e_max
    return nd.cdf((best_sr - sr0) * sqrt(n_obs - 1)), sr0


if __name__ == "__main__":
    main()
