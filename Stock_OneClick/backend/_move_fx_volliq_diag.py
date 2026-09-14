"""_move_fx_volliq_diag.py — panel-level diagnostics for the volume/liquidity study.
Zero-range-bar concentration by liquidity, feature correlations, dividend-adjustment caveat."""
from __future__ import annotations

import numpy as np
import pandas as pd

import _move_data as D
import _move_lib as L
import move_prob as M
from _move_fx_volliq import volume_features, VOL_FEATS

p = D.load()
C, H, Lo, V = p["Close"], p["High"], p["Low"], p["Volume"]
usable = [s for s in C.columns if s != "^VIX" and C[s].notna().sum() >= 500]
sing = [s for s in usable if s not in M.INDEX_LIKE]

rows = []
for s in sing:
    c = C[s].dropna()
    if len(c) < 400:
        continue
    hi, lo = H[s].reindex(c.index), Lo[s].reindex(c.index)
    fv = volume_features(c, V[s])
    zr = (hi == lo).astype(float)                       # zero-range bar
    pk = L.vol_parkinson(hi, lo, 1)
    rows.append(pd.DataFrame({
        "sym": s, "ldv22": fv["ldv22"].values, "amih": fv["amih"].values,
        "vs_d": fv["vs_d"].values, "vt_q": fv["vt_q"].values, "vvol": fv["vvol"].values,
        "zr": zr.values, "pk": pk.values, "rv_d": M._clip_log(pk).values,
        "v0": (V[s].reindex(c.index).values == 0).astype(float),
        "year": c.index.year.values}))
df = pd.concat(rows, ignore_index=True).dropna(subset=["ldv22"])
print(f"rows {len(df):,}  symbols {df.sym.nunique()}")

print("\n--- zero-range / zero-volume bars by dollar-volume decile (IN-SAMPLE descriptive) ---")
df["dec"] = pd.qcut(df.ldv22, 10, labels=False)
g = df.groupby("dec").agg(n=("zr", "size"), ldv22=("ldv22", "mean"),
                          zero_range=("zr", "mean"), zero_vol=("v0", "mean"),
                          clipped_lo=("pk", lambda x: float((x <= 1e-3).mean())),
                          mean_rv=("rv_d", "mean"))
g["dollar_vol_$M"] = np.exp(g.ldv22) / 1e6
print(g.to_string(float_format=lambda v: f"{v:,.5f}"))

print("\n--- feature correlations (IN-SAMPLE, single names) ---")
cc = df[["ldv22", "amih", "vs_d", "vt_q", "vvol", "rv_d"]].corr()
print(cc.round(3).to_string())

print("\n--- dividend-adjustment caveat: panel dollar volume vs true dollar volume ---")
raw = pd.read_pickle("_move_fx_volliq_rawcheck.pkl")
for s in ["XOM", "AAPL", "NVDA"]:
    r = raw[s]
    f = (r["Close"] / r["Adj Close"]).dropna()
    print(f"  {s}: raw_close/adj_close  2001={f.iloc[0]:.3f}  2010={f.loc['2010':].iloc[0]:.3f}  "
          f"2026={f.iloc[-1]:.3f}  -> panel dollar volume understates 2001 level by this factor")
