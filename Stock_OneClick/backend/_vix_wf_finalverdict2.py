"""_vix_wf_finalverdict2.py — supplementary numbers for the final verdict:
cell-A/B era checks, squeeze robustness + tail framing, and power (how many more
years of observation would be needed to overturn the verdict). Self-contained.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _vix_data

d = _vix_data.add_features(_vix_data.load())
d = d[d.spx.notna()].copy()
d["spx_ma20"] = d.spx.rolling(20).mean()
d["below_ma20"] = d.spx < d.spx_ma20
d["width_pct2y"] = d.bb10_width.rolling(504).rank(pct=True)
d["abs_g10"] = d.g10.abs()
d["bad5"] = (d.g5 < -0.02).astype(float).where(d.g5.notna())
d["big10"] = (d.g10.abs() > 0.04).astype(float).where(d.g10.notna())


def rot_exact(mask, y, min_n=20):
    mask = np.asarray(mask, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = (~np.isnan(y)).astype(float)
    ys = np.where(np.isnan(y), 0.0, y)
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
    return obs, float((np.abs(null - base) >= abs(obs - base)).mean()), int(round(C[0])), float(null.std())


def cell(mask, y, label, base):
    obs, p, n, sd = rot_exact(np.asarray(mask), np.asarray(y))
    if n < 25:
        print(f"{label:<52} n={n:5d}   n<25 INCONCLUSIVE")
        return
    print(f"{label:<52} n={n:5d}  val={obs*100:+.2f}  exc={(obs-base)*100:+.2f}  p={p:.3f}  nullsd={sd*100:.2f}")


print("=== [G] CELL A: BB(10,2) below-lower WHILE SPX>MA20, by era ==============")
up = d[(~d.below_ma20).fillna(False) & d.spx_ma20.notna()]
for tag, sub in (("full", up.index >= "1900"), ("2010+", up.index >= "2010-01-01"),
                 ("2015+", up.index >= "2015-01-01")):
    s = up[sub]
    b_bad, b_g5 = np.nanmean(s.bad5), s.g5.mean()
    print(f"-- SPX>MA20 {tag}: n={len(s)}  P(g5<-2%)={b_bad*100:.2f}%  meang5={b_g5*100:+.3f}%")
    cell(s["bb10_2.0_below"], s.bad5, f"   below&up P(g5<-2%) {tag}", b_bad)
    cell(s["bb10_2.0_below"], s.g5, f"   below&up mean g5 {tag}", b_g5)
    cell(s["bb10_1.5_below"], s.bad5, f"   below1.5&up P(g5<-2%) {tag}", b_bad)
    cell(s["bb10_1.5_below"], s.g5, f"   below1.5&up mean g5 {tag}", b_g5)

print("\n=== [H] SQUEEZE robustness: width percentile cut & horizons ==============")
for cut in (0.10, 0.20, 0.30):
    m = (d.width_pct2y <= cut).fillna(False)
    print(f"-- width<=p{int(cut*100)} of trailing 2y   ({m.sum()} days, {m.mean()*100:.1f}% of sessions)")
    for col, nm, base in ((d.abs_g10, "|SPX 10d|", np.nanmean(d.abs_g10)),
                          (d.big10, "P(|SPX 10d|>4%)", np.nanmean(d.big10)),
                          (d.g10, "SPX 10d ret", np.nanmean(d.g10))):
        cell(m, col, f"   {nm}", base)
    m5 = m
    cell(m5, d.g5.abs(), "   |SPX 5d|", np.nanmean(d.g5.abs()))
    cell(m5, d.bad5, "   P(g5<-2%)", np.nanmean(d.bad5))

print("\n-- squeeze |SPX 10d| by era (width<=p20) ---------------------------------")
m = (d.width_pct2y <= 0.20).fillna(False)
for tag, sub in (("1990s", (d.index.year < 2000)), ("2000s", (d.index.year // 10 == 200)),
                 ("2010s", (d.index.year // 10 == 201)), ("2020s", (d.index.year >= 2020)),
                 ("2021+", (d.index >= "2021-01-01"))):
    s = d[sub]
    cell(m[sub], s.abs_g10, f"   |SPX 10d| {tag}", np.nanmean(s.abs_g10))

print("\n=== [I] POWER: how much data to detect the historical edges ==============")
print("rotation-null sd of the conditional mean scales ~ 1/sqrt(n_signal_days)")
for name, mask, hist in (("stretch>=+10%", d.stretch >= 0.10, 0.00305),
                         ("stretch>=+20%", d.stretch >= 0.20, 0.00649),
                         ("BB(10,2) above", d["bb10_2.0_above"], 0.00511),
                         ("BB(10,2) below", d["bb10_2.0_below"], -0.00677),
                         ("BB(10,1.5) below", d["bb10_1.5_below"], -0.00474)):
    m = np.asarray(mask.fillna(False))
    late = m & (d.index >= "2015-01-01")
    _, _, n_late, sd_late = rot_exact(late, d.g5.values)
    per_year = late.sum() / (len(d[d.index >= "2015-01-01"]) / 252)
    need_n = (2 * sd_late * np.sqrt(n_late) / abs(hist)) ** 2 if hist else np.nan
    print(f"{name:<18} late n={n_late:4d} ({per_year:4.1f}/yr)  nullsd={sd_late*100:.3f}pp  "
          f"2sig MDE={2*sd_late*100:.3f}pp  hist edge={hist*100:+.3f}pp  "
          f"need n~{need_n:.0f} = {need_n/per_year:.0f} more years")

print("\n=== [J] CURRENT-STATE FREQUENCY TABLE for the playbook ===================")
states = {
    "z10 >= +2 (above upper band)": d["bb10_2.0_above"],
    "z10 >= +1.5": d["bb10_1.5_above"],
    "z10 <= -1.5 (lower-band zone)": d["bb10_1.5_below"],
    "z10 <= -2": d["bb10_2.0_below"],
    "inside +/-1.5": (~d["bb10_1.5_above"].fillna(False)) & (~d["bb10_1.5_below"].fillna(False)) & d.bb10_pctb.notna(),
}
tot = d.bb10_pctb.notna().sum()
for k, m in states.items():
    m = m.fillna(False)
    print(f"{k:<34} {m.sum():5d} days  {m.sum()/tot*100:5.1f}% of sessions  "
          f"{m[d.index>='2016-01-01'].sum()/(len(d[d.index>='2016-01-01'])/252):5.1f}/yr since 2016")

print("\n=== [K] z10 dose-response (full sample, g5 excess) =======================")
z = d.bb10_pctb * 4 - 2
bins = [(-9, -2), (-2, -1.5), (-1.5, -1), (-1, 0), (0, 1), (1, 1.5), (1.5, 2), (2, 9)]
b0 = d.g5.mean()
for lo, hi in bins:
    m = ((z >= lo) & (z < hi)).fillna(False)
    cell(m, d.g5, f"   z10 in [{lo:+.1f},{hi:+.1f})", b0)
