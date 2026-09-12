"""_vix_wf_finalverdict4.py — final addendum: is ANY of the family alive post-2015 at
matched frequency, and what does the honest 'best available fear gauge' look like now?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _vix_data

d = _vix_data.add_features(_vix_data.load())
d = d[d.spx.notna()].copy()
d["spx_ma20"] = d.spx.rolling(20).mean()
d["spx_r5"] = d.spx.pct_change(5, fill_method=None)
d["z10"] = d.bb10_pctb * 4 - 2


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
    return obs, float((np.abs(null - base) >= abs(obs - base)).mean()), int(round(C[0])), float(null.std())


for tag, sub in (("FULL 1990-2026", d.index >= "1900"),
                 ("1990-2009", (d.index >= "1900") & (d.index < "2010-01-01")),
                 ("2010-2026", d.index >= "2010-01-01"),
                 ("2015-2026", d.index >= "2015-01-01")):
    s = d[sub]
    base = s.g5.mean()
    print(f"\n=== matched top-12% horse race, {tag}   baseline g5 {base*100:+.3f}%  ({len(s)} sessions)")
    cands = {
        "VIX stretch vs MA10":   s.stretch,
        "VIX z10 (=%B)":         s.z10,
        "VIX level raw":         s.vix,
        "VIX 1y percentile":     s.vix_pct1y,
        "SPX 5d return (LOW)":  -s.spx_r5,
        "SPX vs MA20 (LOW)":    -(s.spx / s.spx_ma20 - 1.0),
    }
    for nm, ser in cands.items():
        ss = ser.where(s.g5.notna())
        m = (ss >= ss.quantile(0.88)).fillna(False)
        o, p, n, sd = rot_exact(m.values, s.g5.values)
        print(f"   {nm:<22} n={n:5d}  exc={(o-base)*100:+.3f}pp  p={p:.3f}  (2sig MDE {2*sd*100:.3f}pp)")

print("\n=== BOTTOM 12% (the 'complacency' side), same eras ======================")
for tag, sub in (("FULL 1990-2026", d.index >= "1900"),
                 ("2010-2026", d.index >= "2010-01-01"),
                 ("2015-2026", d.index >= "2015-01-01")):
    s = d[sub]
    base = s.g5.mean()
    print(f"-- {tag}  baseline {base*100:+.3f}%")
    for nm, ser in (("VIX z10 LOW 12%", s.z10), ("VIX stretch LOW 12%", s.stretch),
                    ("VIX level LOW 12%", s.vix), ("VIX 1y pctile LOW 12%", s.vix_pct1y)):
        ss = ser.where(s.g5.notna())
        m = (ss <= ss.quantile(0.12)).fillna(False)
        o, p, n, sd = rot_exact(m.values, s.g5.values)
        print(f"   {nm:<24} n={n:5d}  exc={(o-base)*100:+.3f}pp  p={p:.3f}")

print("\n=== YEAR-BY-YEAR sign of the two headline signals (g5 excess) ============")
for nm, m in (("stretch>=+10%", d.stretch >= 0.10), ("BB(10,2) above", d["bb10_2.0_above"]),
              ("BB(10,1.5) below", d["bb10_1.5_below"])):
    rows = []
    for y in range(2015, 2027):
        sub = d[d.index.year == y]
        mm = m[d.index.year == y]
        if mm.sum() < 5 or sub.g5.notna().sum() < 50:
            rows.append(f"{y}:n/a")
            continue
        exc = sub.g5[mm].mean() - sub.g5.mean()
        rows.append(f"{y}:{exc*100:+.2f}({int(mm.sum())})")
    print(f"{nm:<18} " + "  ".join(rows))
