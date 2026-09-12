"""_vix_wf_finalverdict.py — consolidation run for the final VIX-MA10 / Bollinger verdict.

Produces exactly the numbers quoted in the deliverable: the 6-rule verdict table, the
tail-risk cells, state frequencies, the matched-frequency horse race vs plain SPX/level
variables, and today's live reading. Exact circular-rotation p-values (all offsets, FFT).

Read-only w.r.t. the shared panel; writes nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _vix_data

pd.set_option("display.width", 200)

d = _vix_data.add_features(_vix_data.load())
print(f"panel {len(d):,} rows  {d.index[0].date()} -> {d.index[-1].date()}")

# ---- data-hygiene fix flagged by the conditioning study: US cash holidays where ^VIX
# printed but ^GSPC did not. Left in, any spx.rolling() silently NaNs the trailing window.
bad = d.index[d.spx.isna()]
print("rows with VIX but no SPX close:", [str(x.date()) for x in bad])
d = d[d.spx.notna()].copy()

d["spx_ma20"] = d.spx.rolling(20).mean()
d["spx_below_ma20"] = d.spx < d.spx_ma20
d["spx_r5"] = d.spx.pct_change(5, fill_method=None)
d["rv20"] = d.spx.pct_change(fill_method=None).rolling(20).std(ddof=0) * np.sqrt(252) * 100
d["width_pct2y"] = d.bb10_width.rolling(504).rank(pct=True)
d["sqz"] = d.width_pct2y <= 0.20
d["vix_fwd10"] = d.vix.shift(-10) / d.vix - 1.0
d["abs_g10"] = d.g10.abs()


# ---------------------------------------------------------------- exact rotation null
def rot_exact(mask: np.ndarray, y: np.ndarray, min_n: int = 20):
    """Mean of y over mask, plus two-sided p from the EXACT circular-rotation null
    (all offsets, computed at once by FFT cross-correlation). NaNs in y are dropped
    per-offset, exactly as rotation_pvalue() in _vix_ma10_bb_research.py does."""
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
    obs = means[0]
    null = means[1:]
    null = null[~np.isnan(null)]
    base = null.mean()
    p = float((np.abs(null - base) >= abs(obs - base)).mean())
    return obs, p, int(round(C[0])), float(null.std())


def cell(mask, y, label, base=None):
    obs, p, n, sd = rot_exact(mask.values, y.values)
    yv = y.values[mask.values & ~np.isnan(y.values)]
    b = base if base is not None else np.nanmean(y.values)
    exc = obs - b
    flag = "  n<25 INCONCL" if n < 25 else ""
    print(f"{label:<44} n={n:5d}  mean={obs*100:+.3f}%  exc={exc*100:+.3f}%  p={p:.3f} "
          f"nullsd={sd*100:.3f}{flag}")
    return dict(label=label, n=n, mean=obs, exc=exc, p=p, nullsd=sd)


SIGS = {
    "stretch>=+10% (VIX above MA10)":  d.stretch >= 0.10,
    "stretch>=+20%":                   d.stretch >= 0.20,
    "stretch<=-10% (VIX below MA10)":  d.stretch <= -0.10,
    "BB(10,2) above upper":            d["bb10_2.0_above"],
    "BB(10,2) re-entry":               d["bb10_2.0_reentry"],
    "BB(10,2) BELOW lower":            d["bb10_2.0_below"],
    "BB(10,1.5) BELOW lower":          d["bb10_1.5_below"],
    "squeeze (width bot-20% of 2y)":   d["sqz"].fillna(False),
}

print("\n=== [A] FULL SAMPLE, D5 (g5) ==============================================")
print(f"baseline g5 = {d.g5.mean()*100:+.3f}%   win {(d.g5>0).mean()*100:.1f}%   n_valid={d.g5.notna().sum()}")
full = {k: cell(v, d.g5, k) for k, v in SIGS.items()}

for tag, sub in (("2010-2026", d.index >= "2010-01-01"),
                 ("2015-2026", d.index >= "2015-01-01"),
                 ("2021-2026", d.index >= "2021-01-01")):
    dd = d[sub]
    print(f"\n=== [A] {tag}  baseline g5 = {dd.g5.mean()*100:+.3f}%  ({len(dd)} sessions) ===")
    for k, v in SIGS.items():
        cell(v[sub], dd.g5, k)

print("\n=== [B] SQUEEZE: what it actually predicts (10d horizon) ==================")
sq = d["sqz"].fillna(False)
for col, nm in ((d.vix_fwd10, "fwd 10d VIX % change"), (d.abs_g10, "fwd |SPX 10d move|"), (d.g10, "fwd SPX 10d ret")):
    print(f"-- {nm}: baseline {np.nanmean(col)*100:+.3f}%")
    cell(sq, col, f"   squeeze full")
    for tag, sub in (("2010-2026", d.index >= "2010-01-01"), ("2015-2026", d.index >= "2015-01-01")):
        b = np.nanmean(col[sub])
        cell(sq[sub], col[sub], f"   squeeze {tag} (base {b*100:+.3f}%)", base=b)

print("\n=== [C] TAIL CELLS: band x SPX-vs-MA20, P(g5 < -2%) =======================")
d["bad5"] = (d.g5 < -0.02).astype(float).where(d.g5.notna())
for regime_name, reg in (("SPX<MA20", d.spx_below_ma20.fillna(False)),
                         ("SPX>MA20", (~d.spx_below_ma20).where(d.spx_ma20.notna()).fillna(False).astype(bool))):
    sub = d[reg]
    print(f"\n-- regime {regime_name}: n={len(sub)}  P(g5<-2%)={np.nanmean(sub.bad5)*100:.2f}%  "
          f"mean g5={sub.g5.mean()*100:+.3f}%")
    for k in ("bb10_2.0_above", "bb10_2.0_below", "bb10_1.5_above", "bb20_2.0_above"):
        m = sub[k]
        cell(m, sub.bad5, f"   {k} P(g5<-2%)", base=np.nanmean(sub.bad5))
        cell(m, sub.g5, f"   {k} mean g5", base=sub.g5.mean())
    for tag, s2 in (("2010+", sub.index >= "2010-01-01"), ("2015+", sub.index >= "2015-01-01")):
        s3 = sub[s2]
        cell(s3["bb10_2.0_above"], s3.bad5, f"   bb10_2.0_above P(g5<-2%) {tag}", base=np.nanmean(s3.bad5))

print("\n=== [D] MATCHED-FREQUENCY HORSE RACE (top 12% of days, g5) ================")
n_pick = int(round(0.12 * d.g5.notna().sum()))
cands = {
    "VIX stretch vs MA10 (top 12%)":      d.stretch,
    "VIX level, raw (top 12%)":           d.vix,
    "VIX 1y percentile (top 12%)":        d.vix_pct1y,
    "SPX trailing 5d return (LOW 12%)":   -d.spx_r5,
    "SPX vs MA20 gap (LOW 12%)":          -(d.spx / d.spx_ma20 - 1.0),
    "SPX realized vol 20d (top 12%)":     d.rv20,
}
for nm, series in cands.items():
    s = series.where(d.g5.notna())
    thr = s.quantile(1 - 0.12)
    m = (s >= thr).fillna(False)
    cell(m, d.g5, nm)

print("\n=== [E] STATE FREQUENCY (days per year, last 10 calendar years) ===========")
recent = d[d.index >= "2016-01-01"]
for k, v in SIGS.items():
    vv = v[d.index >= "2016-01-01"]
    print(f"{k:<44} {vv.sum():4d} days / {len(recent)/252:.1f}y = {vv.sum()/(len(recent)/252):5.1f} per year")
byyear = pd.DataFrame({k: v.groupby(d.index.year).sum() for k, v in SIGS.items()})
print(byyear.tail(12).to_string())

print("\n=== [F] LIVE READING ======================================================")
last = d.iloc[-1]
print(f"as of {d.index[-1].date()}")
print(f"  VIX {last.vix:.2f}  MA10 {last.ma10:.2f}  stretch {last.stretch*100:+.1f}%")
print(f"  BB(10,2.0) {last['bb10_2.0_lo']:.2f} .. {last['bb10_2.0_up']:.2f}   pctB {last.bb10_pctb:.2f}"
      f"   z10 {(last.bb10_pctb*4-2):+.2f}")
print(f"  BB(10,1.5) {last['bb10_1.5_lo']:.2f} .. {last['bb10_1.5_up']:.2f}")
print(f"  vix_pct1y {last.vix_pct1y*100:.0f}%   bb10_width {last.bb10_width:.3f} "
      f"(2y pctile {last.width_pct2y*100:.0f}%)  squeeze={bool(last['sqz'])}")
print(f"  SPX {last.spx:.2f} vs MA20 {last.spx_ma20:.2f} -> {'BELOW (risk-off)' if last.spx_below_ma20 else 'above'}")
print(f"  SPX trailing 5d {last.spx_r5*100:+.2f}%")
print(f"  vix3m last valid: {d.vix3m.last_valid_index().date()}  ({(d.index[-1]-d.vix3m.last_valid_index()).days}d ago)")
print(f"  vvix last valid: {d.vvix.last_valid_index().date()}  value {d.vvix.loc[d.vvix.last_valid_index()]:.2f}")
print(f"  vix3m missing in last 504 sessions: {d.vix3m.tail(504).isna().sum()}")
print(f"  vvix  missing in last 504 sessions: {d.vvix.tail(504).isna().sum()}")
