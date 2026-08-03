"""Verify the three core claims of the gold sleeve system on the corrected engine."""
import numpy as np
import pandas as pd

import gold_system as G

panel = G.load_panel()
m = G.monthly(panel)
gold = m["GC=F"].dropna()
gr = gold.pct_change()
px = m[list(G.EQUITY_PROXY)].dropna()
er = sum(px[t].pct_change() * w for t, w in G.EQUITY_PROXY.items())
idx = gr.dropna().index.intersection(er.dropna().index)
gr, er = gr.loc[idx], er.loc[idx]
sig = G.trend_signals(gold)
print(f"window {idx[0].date()} -> {idx[-1].date()}  N={int(len(idx)/12)} independent years\n")

print("=== A. DOES TILTING THE WEIGHT ON TREND HELP? (50% band) ===")
flat = G.simulate(gr, er, 0.20, mode="band", band=0.50)
print(f"  {'flat 20% (no tilt)':<22} Sharpe {flat['sharpe']:.3f}  CAGR {flat['cagr']*100:5.2f}%  "
      f"vol {flat['vol']*100:4.1f}%  MaxDD {flat['maxdd']*100:6.1f}%  trades {flat['trades']:>2}  "
      f"tax/final {flat['tax_pct_final']*100:4.1f}%")
for hi, lo, key, lbl in [(0.25, 0.15, "above_ma18", "25/15 on 18m MA"),
                         (0.25, 0.15, "mom12", "25/15 on 12m mom"),
                         (0.30, 0.10, "above_ma18", "30/10 on 18m MA"),
                         (0.20, 0.00, "above_ma18", "20/0 switch 18m MA")]:
    s = sig[key].reindex(idx).fillna(False).astype(bool)
    tgt = pd.Series(np.where(s, hi, lo), index=idx)
    r = G.simulate(gr, er, tgt, mode="band", band=0.50)
    print(f"  {lbl:<22} Sharpe {r['sharpe']:.3f}  CAGR {r['cagr']*100:5.2f}%  "
          f"vol {r['vol']*100:4.1f}%  MaxDD {r['maxdd']*100:6.1f}%  trades {r['trades']:>2}  "
          f"tax/final {r['tax_pct_final']*100:4.1f}%")

print("\n=== B. BAND WIDTH at a fixed 20% target ===")
for mode, band, lbl in [("annual", None, "annual"), ("band", 0.25, "25% band"),
                        ("band", 0.50, "50% band"), ("band", 0.75, "75% band"),
                        ("none", None, "never")]:
    r = G.simulate(gr, er, 0.20, mode=mode, band=band)
    print(f"  {lbl:<10} Sharpe {r['sharpe']:.3f}  MaxDD {r['maxdd']*100:6.1f}%  "
          f"trades {r['trades']:>2}  tax/final {r['tax_pct_final']*100:4.1f}%  "
          f"final w {r['final_weight']*100:4.1f}%")

print("\n=== C. HOW MUCH OF THE BENEFIT IS GOLD'S RETURN vs ITS CORRELATION? ===")
print("  (gold's monthly deviations and therefore its correlation are held fixed;")
print("   only its average return is re-centered)")
z = G.simulate(gr, er, 0.0, mode="none")
realized = (1 + gr).prod() ** (12 / len(gr)) - 1
for target_cagr, lbl in [(None, "gold as realized"),
                         (G.RISK_FREE, "gold re-centered to rf 1.82%"),
                         (0.0, "gold re-centered to 0%")]:
    g2 = gr if target_cagr is None else gr - ((1 + realized) ** (1 / 12) - (1 + target_cagr) ** (1 / 12))
    r = G.simulate(g2, er, 0.20, mode="band", band=0.50)
    print(f"  {lbl:<30} CAGR {r['cagr']*100:5.2f}%  vol {r['vol']*100:4.1f}%  "
          f"MaxDD {r['maxdd']*100:6.1f}%  Sharpe {r['sharpe']:.3f}  "
          f"delta {r['sharpe']-z['sharpe']:+.3f}")
print(f"  {'no-gold book':<30} CAGR {z['cagr']*100:5.2f}%  vol {z['vol']*100:4.1f}%  "
      f"MaxDD {z['maxdd']*100:6.1f}%  Sharpe {z['sharpe']:.3f}")
print(f"\n  gold realized CAGR over window: {realized*100:.2f}%")
