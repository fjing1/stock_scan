"""_vix_wf_finalverdict3.py — the decisive test for the integration spec:
does BB WIDTH (the squeeze) add anything over the VIX LEVEL, which the scanner
already prints? Outcome = forward |SPX 10d move| (g10 abs) and P(|g10|>4%).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _vix_data

d = _vix_data.add_features(_vix_data.load())
d = d[d.spx.notna()].copy()
d["width_pct2y"] = d.bb10_width.rolling(504).rank(pct=True)
d["sqz"] = (d.width_pct2y <= 0.20)
d["absg10"] = d.g10.abs()
d["big10"] = (d.g10.abs() > 0.04).astype(float).where(d.g10.notna())
d["rv20"] = d.spx.pct_change(fill_method=None).rolling(20).std(ddof=0) * np.sqrt(252) * 100


def rot_exact(mask, y, min_n=20):
    mask = np.asarray(mask, dtype=float); y = np.asarray(y, dtype=float)
    valid = (~np.isnan(y)).astype(float); ys = np.where(np.isnan(y), 0.0, y)
    n = len(y)
    F = np.fft.rfft(mask)
    S = np.fft.irfft(np.conj(F) * np.fft.rfft(ys), n)
    C = np.fft.irfft(np.conj(F) * np.fft.rfft(valid), n)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = S / C
    means[C < min_n] = np.nan
    obs, null = means[0], means[1:]
    null = null[~np.isnan(null)]
    base = null.mean()
    return obs, float((np.abs(null - base) >= abs(obs - base)).mean()), int(round(C[0]))


def fe_coef(dummy, y, groups, label):
    """Frisch-Waugh dummy coefficient under one-hot group FE, with exact rotation p
    on the residualised dummy (rotate the dummy, re-residualise is not needed because
    the FE projection is applied to y once; we rotate the RAW dummy then residualise
    it against the same FE, which is the conservative version)."""
    ok = pd.notna(y) & pd.notna(groups) & pd.notna(dummy)
    yy = pd.Series(y)[ok].astype(float)
    gg = pd.Series(groups)[ok]
    xx = pd.Series(dummy)[ok].astype(float)
    ymu = yy.groupby(gg).transform("mean")
    xmu = xx.groupby(gg).transform("mean")
    yr, xr = (yy - ymu).values, (xx - xmu).values
    coef = float(np.dot(xr, yr) / np.dot(xr, xr))
    # rotation null on the residualised pair
    n = len(xr)
    F = np.fft.rfft(xr)
    num = np.fft.irfft(np.conj(F) * np.fft.rfft(yr), n)
    den = np.dot(xr, xr)
    null = num[1:] / den
    p = float((np.abs(null - null.mean()) >= abs(coef - null.mean())).mean())
    print(f"{label:<58} coef={coef*100:+.3f}  p={p:.3f}  n_treat={int(xx.sum())}")
    return coef


print("=== [L] Does the SQUEEZE survive the VIX LEVEL control? ==================")
base_abs = np.nanmean(d.absg10)
obs, p, n = rot_exact(d.sqz.fillna(False).values, d.absg10.values)
print(f"raw squeeze |SPX 10d|: {obs*100:.3f} vs base {base_abs*100:.3f} -> exc {(obs-base_abs)*100:+.3f}pp  p={p:.3f}  n={n}")

lvl_dec = pd.qcut(d.vix, 10, labels=False, duplicates="drop")
p1y_dec = pd.qcut(d.vix_pct1y, 10, labels=False, duplicates="drop")
rv_dec = pd.qcut(d.rv20, 10, labels=False, duplicates="drop")
yr = pd.Series(d.index.year, index=d.index)

sq = d.sqz.fillna(False)
fe_coef(sq, d.absg10, lvl_dec, "|SPX 10d| ~ squeeze | RAW VIX decile FE")
fe_coef(sq, d.absg10, pd.qcut(d.vix, 20, labels=False, duplicates="drop"), "|SPX 10d| ~ squeeze | RAW VIX vigintile FE")
fe_coef(sq, d.absg10, p1y_dec, "|SPX 10d| ~ squeeze | VIX 1y-pctile decile FE")
fe_coef(sq, d.absg10, rv_dec, "|SPX 10d| ~ squeeze | SPX rv20 decile FE")
fe_coef(sq, d.absg10, lvl_dec.astype(str) + "_" + rv_dec.astype(str), "|SPX 10d| ~ squeeze | VIXdec x rv20dec (100 cells)")
fe_coef(sq, d.absg10, yr, "|SPX 10d| ~ squeeze | YEAR FE")
fe_coef(sq, d.big10, lvl_dec, "P(|SPX10d|>4%) ~ squeeze | RAW VIX decile FE")
fe_coef(sq, d.big10, lvl_dec.astype(str) + "_" + rv_dec.astype(str), "P(|SPX10d|>4%) ~ squeeze | VIXdec x rv20dec")

print("\n-- head to head at matched frequency (bottom ~19% of each variable) ------")
for nm, s, lo in (("bb10_width 2y-pctile (squeeze)", d.width_pct2y, True),
                  ("raw VIX level", d.vix, True),
                  ("VIX 1y percentile", d.vix_pct1y, True),
                  ("SPX realized vol 20d", d.rv20, True),
                  ("bb10_width RAW (not pctile)", d.bb10_width, True)):
    ss = s.where(d.absg10.notna())
    thr = ss.quantile(0.192)
    m = ((ss <= thr) if lo else (ss >= thr)).fillna(False)
    o, pp, nn = rot_exact(m.values, d.absg10.values)
    o2, pp2, _ = rot_exact(m.values, d.big10.values)
    print(f"{nm:<34} n={nn:5d}  |SPX10d| {o*100:.3f} (exc {(o-base_abs)*100:+.3f}, p={pp:.3f})   "
          f"P(>4%) {o2*100:.2f} (exc {(o2-np.nanmean(d.big10))*100:+.2f}, p={pp2:.3f})")

print("\n-- post-2015 only --------------------------------------------------------")
late = d.index >= "2015-01-01"
dl = d[late]
b1, b2 = np.nanmean(dl.absg10), np.nanmean(dl.big10)
o, pp, nn = rot_exact(sq[late].values, dl.absg10.values)
o2, pp2, _ = rot_exact(sq[late].values, dl.big10.values)
print(f"squeeze 2015+: n={nn}  |SPX10d| {o*100:.3f} (base {b1*100:.3f}, exc {(o-b1)*100:+.3f}, p={pp:.3f})  "
      f"P(>4%) {o2*100:.2f} (base {b2*100:.2f}, exc {(o2-b2)*100:+.2f}, p={pp2:.3f})")
fe_coef(sq[late], dl.absg10, pd.qcut(dl.vix, 10, labels=False, duplicates="drop"),
        "  2015+ |SPX 10d| ~ squeeze | VIX decile FE")

print("\n=== [M] What the WIDE band (opposite state) says =========================")
wide = (d.width_pct2y >= 0.80).fillna(False)
o, pp, nn = rot_exact(wide.values, d.absg10.values)
o2, pp2, _ = rot_exact(wide.values, d.big10.values)
print(f"width>=p80: n={nn}  |SPX10d| {o*100:.3f} (exc {(o-base_abs)*100:+.3f}, p={pp:.3f})  "
      f"P(>4%) {o2*100:.2f} (exc {(o2-np.nanmean(d.big10))*100:+.2f}, p={pp2:.3f})")
fe_coef(wide, d.absg10, lvl_dec, "  |SPX 10d| ~ wide | RAW VIX decile FE")

print("\n=== [N] live width reading ==============================================")
last = d.iloc[-1]
print(f"{d.index[-1].date()}  bb10_width={last.bb10_width:.3f}  2y pctile={last.width_pct2y*100:.0f}%  "
      f"squeeze={bool(last.sqz)}  VIX={last.vix:.2f}  rv20={last.rv20:.1f}")
print(d[["vix", "bb10_width", "width_pct2y"]].tail(6).to_string())
