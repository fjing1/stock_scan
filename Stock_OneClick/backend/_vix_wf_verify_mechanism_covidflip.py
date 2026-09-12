"""
_vix_wf_verify_mechanism_covidflip.py

ADVERSARIAL verification, MECHANISM/CONFOUNDING lens, of the claim:

  "The 2020s negative sign on every bullish VIX variant is 24 COVID days, not decay.
   Excluding 2020-02-01..2020-06-30 flips stretch>=+10% from -0.172pp to +0.210pp and
   stretch>=+20% from -0.365pp to +0.800pp (p=.029); all four bullish signals flip positive."

The claimed mechanism is "one idiosyncratic crisis window contaminates the 2020s block".
Simpler competing mechanisms tested here:
  (C1) ARITHMETIC / SELECTION: excluding the *worst contiguous block* always flips a sign.
       Placebo: slide the same-length exclusion window across 2020-2026.
  (C2) ASYMMETRIC SURGERY: earlier decades keep THEIR crises. Apply an identical, ex-ante
       crisis excision (window containing each era's VIX peak) to every era and re-test decay.
  (C3) VIX LEVEL: the signal simply fails at extreme VIX. Bucket by level; reweight the
       2020s signal days to the full-sample level mix; excise by LEVEL not by DATE.
  (C4) SPX MOMENTUM / drawdown depth: signal days are "SPX just fell"; excise/reweight on
       trailing SPX return instead of on the calendar.
  (C5) TERM STRUCTURE (VRP): backwardation (vix/vix3m>1) days. 2006+, honest n reported.
  (C6) BASELINE-SIDE vs SIGNAL-SIDE decomposition of the flip.
  (C7) MEDIANS: does the "median is positive anyway" defence survive an era comparison?

All excesses = conditional mean MINUS same-sample unconditional mean.
p-values = circular rotation test (>=5000 rolls) computed WITHIN the analysis frame.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import _vix_data

N_ROT = 5000
RNG = np.random.default_rng(20260910)
H = 5
FWD = f"g{H}"

COVID0, COVID1 = pd.Timestamp("2020-02-01"), pd.Timestamp("2020-06-30")


# ----------------------------------------------------------------- helpers
def rotation_pvalue(mask: np.ndarray, fwd: np.ndarray, observed: float,
                    n_rot: int = N_ROT) -> float:
    """Circular rotation test, same construction as _vix_ma10_bb_research.rotation_pvalue,
    but the frame passed in is already restricted to the sub-period under test."""
    n = len(mask)
    if n < 10 or mask.sum() == 0:
        return float("nan")
    valid = ~np.isnan(fwd)
    offs = RNG.integers(1, n, size=n_rot)
    null = np.empty(n_rot)
    for i, off in enumerate(offs):
        m = np.roll(mask, off) & valid
        null[i] = fwd[m].mean() if m.sum() else np.nan
    null = null[~np.isnan(null)]
    if not len(null):
        return float("nan")
    base = np.nanmean(null)
    return float((np.abs(null - base) >= abs(observed - base)).mean())


def stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "   "
    return "***" if p < .01 else ("** " if p < .05 else ("*  " if p < .10 else "   "))


def excess(frame: pd.DataFrame, mask: pd.Series, fwd_col: str = FWD):
    """excess, n, cond, base over the SAME sample (rows of `frame` with fwd non-NaN)."""
    f = frame[fwd_col]
    ok = f.notna()
    m = mask.reindex(frame.index).fillna(False).astype(bool) & ok
    if m.sum() == 0:
        return np.nan, 0, np.nan, np.nan
    base = f[ok].mean()
    cond = f[m].mean()
    return (cond - base) * 100, int(m.sum()), cond * 100, base * 100


def excess_med(frame: pd.DataFrame, mask: pd.Series, fwd_col: str = FWD):
    f = frame[fwd_col]
    ok = f.notna()
    m = mask.reindex(frame.index).fillna(False).astype(bool) & ok
    if m.sum() == 0:
        return np.nan, 0
    return (f[m].median() - f[ok].median()) * 100, int(m.sum())


def excess_p(frame: pd.DataFrame, mask: pd.Series, fwd_col: str = FWD, n_rot=N_ROT):
    exc, n, cond, base = excess(frame, mask, fwd_col)
    if n == 0:
        return exc, n, float("nan")
    m = mask.reindex(frame.index).fillna(False).astype(bool).values
    p = rotation_pvalue(m, frame[fwd_col].values.astype(float), cond / 100, n_rot=n_rot)
    return exc, n, p


# ----------------------------------------------------------------- data
d = _vix_data.add_features(_vix_data.load())
d["tr5"] = d.spx / d.spx.shift(5) - 1.0            # trailing 5d SPX return (known at t)
d["tr10"] = d.spx / d.spx.shift(10) - 1.0
d["tr21"] = d.spx / d.spx.shift(21) - 1.0
d["dd252"] = d.spx / d.spx.rolling(252).max() - 1.0

SIGNALS = {
    "stretch>=+10%": d.stretch >= 0.10,
    "stretch>=+20%": d.stretch >= 0.20,
    "BB(10,2) above": d["bb10_2.0_above"].astype(bool),
    "BB(10,2) re-entry": d["bb10_2.0_reentry"].astype(bool),
}

ERAS = [("1990s", 1990, 1999), ("2000s", 2000, 2009), ("2010s", 2010, 2019), ("2020s", 2020, 2026)]

covid_mask = (d.index >= COVID0) & (d.index <= COVID1)
d20 = d[d.index.year >= 2020]
d20_ex = d20[~((d20.index >= COVID0) & (d20.index <= COVID1))]

print("=" * 100)
print(f"PANEL {len(d):,} rows {d.index[0].date()} -> {d.index[-1].date()}   horizon {FWD} (next-close entry)")
print(f"2020-2026 block: {len(d20)} rows; ex-COVID({COVID0.date()}..{COVID1.date()}): {len(d20_ex)} rows "
      f"(removed {len(d20)-len(d20_ex)} rows = {(len(d20)-len(d20_ex))/len(d20)*100:.1f}%)")
print("=" * 100)

# ============================================================ 0. reproduce
print("\n### 0. REPRODUCTION of the claim's headline numbers (2020-2026, g5 excess)")
print(f"{'signal':<20}{'n_all':>7}{'exc_all':>10}{'n_exCV':>8}{'exc_exCV':>10}{'p_exCV':>9}  {'delta':>8}")
repro = {}
for name, sig in SIGNALS.items():
    a_exc, a_n, _, a_base = excess(d20, sig)
    b_exc, b_n, b_p = excess_p(d20_ex, sig)
    repro[name] = (a_exc, a_n, b_exc, b_n, b_p)
    print(f"{name:<20}{a_n:>7}{a_exc:>9.3f}p{b_n:>8}{b_exc:>9.3f}p{b_p:>9.3f}{stars(b_p)}{b_exc-a_exc:>+8.3f}")
print(f"  2020-2026 unconditional g5 baseline: all={d20[FWD].mean()*100:+.3f}%  "
      f"exCOVID={d20_ex[FWD].mean()*100:+.3f}%  COVID-window-only={d[covid_mask][FWD].mean()*100:+.3f}%")

# ============================================================ C6. decomposition
print("\n### C6. Is the flip a SIGNAL-side removal or a BASELINE-side removal?")
print("     exc = mean(signal) - mean(all). Dropping COVID changes BOTH terms. Hold one fixed.")
print(f"{'signal':<20}{'exc_all':>10}{'sig-only drop':>15}{'base-only drop':>16}{'both (claim)':>14}")
for name, sig in SIGNALS.items():
    f = d20[FWD]; ok = f.notna()
    m = sig.reindex(d20.index).fillna(False).astype(bool) & ok
    base_all = f[ok].mean()
    base_ex = d20_ex[FWD].mean()
    m_ex = m & ~((d20.index >= COVID0) & (d20.index <= COVID1))
    cond_all, cond_ex = f[m].mean(), f[m_ex].mean()
    print(f"{name:<20}{(cond_all-base_all)*100:>9.3f}p{(cond_ex-base_all)*100:>14.3f}p"
          f"{(cond_all-base_ex)*100:>15.3f}p{(cond_ex-base_ex)*100:>13.3f}p")

# ============================================================ C1. placebo windows
print("\n### C1. PLACEBO: slide the SAME-LENGTH exclusion window across 2020-2026.")
WLEN = int(covid_mask.sum())          # trading days in the COVID window
print(f"     window length = {WLEN} trading days. Excluding ANY block changes the excess;")
print(f"     question is whether the COVID block is special or merely the most negative block.")
idx20 = d20.index
for name, sig in SIGNALS.items():
    vals, labels = [], []
    for s in range(0, len(idx20) - WLEN + 1, 5):
        keep = np.ones(len(idx20), bool); keep[s:s + WLEN] = False
        sub = d20[keep]
        e, n, _, _ = excess(sub, sig)
        if n >= 25:
            vals.append(e); labels.append(idx20[s].date())
    vals = np.array(vals)
    e_cov = repro[name][2]
    pct = (vals <= e_cov).mean() * 100
    frac_pos = (vals > 0).mean() * 100
    order = np.argsort(vals)
    print(f"  {name:<20} placebo excess: min {vals.min():+.3f}p (start {labels[order[0]]}) "
          f"med {np.median(vals):+.3f}p max {vals.max():+.3f}p (start {labels[order[-1]]})")
    print(f"  {'':<20} COVID-excision gives {e_cov:+.3f}p = pctile {100-pct:.0f} of all excisions; "
          f"{frac_pos:.0f}% of ALL windows give a positive excess")

# ============================================================ C2. symmetric surgery
print("\n### C2. SYMMETRIC SURGERY: excise each era's own crisis by an identical ex-ante rule")
print("     rule = drop the calendar window [peak_VIX_date - 1mo, peak_VIX_date + 4mo] of that era")
print("     (COVID peak 2020-03-16 -> 2020-02-16..2020-07-16, i.e. the claim's own window).")


def era_peak_window(y0, y1):
    sub = d[(d.index.year >= y0) & (d.index.year <= y1)]
    pk = sub.vix.idxmax()
    return pk, pk - pd.DateOffset(months=1), pk + pd.DateOffset(months=4)


print(f"{'signal':<20}" + "".join(f"{e[0]+' raw':>14}" for e in ERAS))
print(f"{'':<20}" + "".join(f"{e[0]+' cut':>14}" for e in ERAS))
for name, sig in SIGNALS.items():
    raws, cuts, ns = [], [], []
    for _, y0, y1 in ERAS:
        sub = d[(d.index.year >= y0) & (d.index.year <= y1)]
        e_raw, n_raw, _, _ = excess(sub, sig)
        pk, w0, w1 = era_peak_window(y0, y1)
        sub_cut = sub[~((sub.index >= w0) & (sub.index <= w1))]
        e_cut, n_cut, _, _ = excess(sub_cut, sig)
        raws.append((e_raw, n_raw)); cuts.append((e_cut, n_cut)); ns.append((pk.date(), w0.date(), w1.date()))
    print(f"{name:<20}" + "".join(f"{r[0]:>9.3f}p n{r[1]:<3}" for r in raws))
    print(f"{'  -> crisis excised':<20}" + "".join(f"{c[0]:>9.3f}p n{c[1]:<3}" for c in cuts))
for nm, y0, y1 in ERAS:
    pk, w0, w1 = era_peak_window(y0, y1)
    print(f"     {nm}: VIX peak {pk.date()} ({d.vix[pk]:.1f}) -> excised {w0.date()}..{w1.date()}")

# ============================================================ C3. VIX level
print("\n### C3. CONFOUND = VIX LEVEL. Full-sample g5 excess of stretch>=+10% by VIX level bucket")
BUCKETS = [(0, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, 200)]
for name in ("stretch>=+10%", "stretch>=+20%"):
    sig = SIGNALS[name]
    print(f"  {name}:")
    for lo, hi in BUCKETS:
        inb = (d.vix >= lo) & (d.vix < hi)
        sub = d[inb]
        e, n, c, b = excess(sub, sig)
        tag = "  (n<25 INCONCLUSIVE)" if n < 25 else ""
        print(f"    VIX {lo:>3}-{hi:<3}  frame n={len(sub):>5}  signal n={n:>4}  "
              f"cond {c:>+7.3f}%  base {b:>+7.3f}%  excess {e:>+7.3f}p{tag}")

print("\n     Level MIX of signal days, by era (share of signal days with VIX>=30):")
for nm, y0, y1 in ERAS:
    sub = d[(d.index.year >= y0) & (d.index.year <= y1)]
    for name in ("stretch>=+10%", "stretch>=+20%"):
        m = SIGNALS[name].reindex(sub.index).fillna(False) & sub[FWD].notna()
        if m.sum() >= 25:
            print(f"       {nm} {name:<16} n={int(m.sum()):>4}  "
                  f"medianVIX {sub.vix[m].median():>5.1f}  share VIX>=30 {(sub.vix[m]>=30).mean()*100:>5.1f}%  "
                  f"share VIX>=40 {(sub.vix[m]>=40).mean()*100:>5.1f}%")

print("\n     LEVEL-BASED excision instead of DATE-BASED, applied identically to every era")
print("     (drop all rows with VIX>=40 -- catches COVID but also 2008, 2018 spike, 2025):")
d_lvl = d[d.vix < 40]
for name in ("stretch>=+10%", "stretch>=+20%", "BB(10,2) above", "BB(10,2) re-entry"):
    sig = SIGNALS[name]
    row = []
    for nm, y0, y1 in ERAS:
        sub = d_lvl[(d_lvl.index.year >= y0) & (d_lvl.index.year <= y1)]
        e, n, _, _ = excess(sub, sig)
        row.append(f"{nm} {e:>+7.3f}p n={n:<4}")
    print(f"     {name:<20} " + " | ".join(row))

print("\n     REWEIGHT the 2020s signal days to the FULL-SAMPLE signal-day VIX-level mix")
print("     (if level mix explains the 2020s sign, the reweighted excess should match history):")
LB = [0, 15, 20, 25, 30, 40, 1e9]
for name in ("stretch>=+10%", "stretch>=+20%"):
    sig = SIGNALS[name]
    ok_all = d[FWD].notna()
    m_all = sig & ok_all
    w_target = pd.cut(d.vix[m_all], LB, right=False).value_counts(normalize=True, sort=False)
    ok20 = d20[FWD].notna()
    m20 = sig.reindex(d20.index).fillna(False) & ok20
    b20 = pd.cut(d20.vix[m20], LB, right=False)
    grp = d20[FWD][m20].groupby(b20, observed=False)
    means, counts = grp.mean(), grp.count()
    use = counts[counts > 0].index
    w = w_target[use] / w_target[use].sum()
    rw_cond = float((means[use] * w).sum())
    base20 = d20[FWD][ok20].mean()
    raw_exc = repro[name][0]
    print(f"     {name:<16} raw 2020s excess {raw_exc:+.3f}p -> level-reweighted "
          f"{(rw_cond-base20)*100:+.3f}p   (buckets used {list(map(str,use))})")
    print(f"     {'':<16} bucket means (2020s): " +
          " ".join(f"{str(k)}:{v*100:+.2f}%/n{int(counts[k])}" for k, v in means[use].items()))

# ============================================================ C4. SPX momentum
print("\n### C4. CONFOUND = trailing SPX move (signal days are 'the index just fell').")
print("     Full-sample g5 excess of stretch>=+10%, bucketed by trailing 5d SPX return:")
TR = [(-1, -.10), (-.10, -.05), (-.05, -.02), (-.02, 0), (0, 1)]
for name in ("stretch>=+10%", "stretch>=+20%"):
    sig = SIGNALS[name]
    print(f"  {name}:")
    for lo, hi in TR:
        inb = (d.tr5 >= lo) & (d.tr5 < hi)
        sub = d[inb]
        e, n, c, b = excess(sub, sig)
        tag = "  (n<25 INCONCLUSIVE)" if n < 25 else ""
        print(f"    tr5 [{lo:>6.0%},{hi:>6.0%})  frame n={len(sub):>5}  signal n={n:>4}  "
              f"cond {c:>+7.3f}%  base {b:>+7.3f}%  excess {e:>+7.3f}p{tag}")

print("\n     MOMENTUM-BASED excision (drop rows with trailing-5d SPX <= -8%), all eras:")
d_mom = d[~(d.tr5 <= -0.08)]
for name in ("stretch>=+10%", "stretch>=+20%"):
    sig = SIGNALS[name]
    row = []
    for nm, y0, y1 in ERAS:
        sub = d_mom[(d_mom.index.year >= y0) & (d_mom.index.year <= y1)]
        e, n, _, _ = excess(sub, sig)
        row.append(f"{nm} {e:>+7.3f}p n={n:<4}")
    print(f"     {name:<16} " + " | ".join(row))

# ============================================================ C5. term structure
print("\n### C5. CONFOUND = term structure / VRP (vix/vix3m). vix3m 2006+ with real holes.")
dt = d[d.term.notna()]
print(f"     rows with term available: {len(dt)}  {dt.index[0].date()}..{dt.index[-1].date()}")
for name in ("stretch>=+10%", "stretch>=+20%"):
    sig = SIGNALS[name]
    for lab, cond_m in (("backwardation term>1", dt.term > 1.0), ("contango term<=1", dt.term <= 1.0)):
        sub = dt[cond_m]
        e, n, c, b = excess(sub, sig)
        tag = "  (n<25 INCONCLUSIVE)" if n < 25 else ""
        print(f"     {name:<16} {lab:<22} frame n={len(sub):>5} signal n={n:>4} excess {e:>+7.3f}p{tag}")
print("     2020s only, term available:")
dt20 = dt[dt.index.year >= 2020]
for name in ("stretch>=+10%", "stretch>=+20%"):
    sig = SIGNALS[name]
    for lab, cond_m in (("backwardation", dt20.term > 1.0), ("contango", dt20.term <= 1.0)):
        sub = dt20[cond_m]
        e, n, _, _ = excess(sub, sig)
        tag = "  (n<25 INCONCLUSIVE)" if n < 25 else ""
        print(f"     {name:<16} {lab:<22} frame n={len(sub):>5} signal n={n:>4} excess {e:>+7.3f}p{tag}")

# ============================================================ C7. medians
print("\n### C7. The 'median excess is positive anyway' defence -- compare eras on MEDIANS")
print(f"{'signal':<20}" + "".join(f"{e[0]:>18}" for e in ERAS))
for name, sig in SIGNALS.items():
    cells = []
    for _, y0, y1 in ERAS:
        sub = d[(d.index.year >= y0) & (d.index.year <= y1)]
        em, n = excess_med(sub, sig)
        cells.append(f"{em:>+8.3f}p n{n:<5}" if n >= 25 else f"{'n<25':>13} n{n:<3}")
    print(f"{name:<20}" + "".join(f"{c:>18}" for c in cells))
print("     (if medians ALSO fall decade over decade, a positive 2020s median is not a defence)")

# ============================================================ robustness of ex-COVID s20
print("\n### R. Robustness of the surviving headline: ex-COVID 2020s stretch>=+20% (+0.800p, p=.029)")
sig = SIGNALS["stretch>=+20%"]
ok = d20_ex[FWD].notna()
m = sig.reindex(d20_ex.index).fillna(False) & ok
sd = d20_ex[FWD][m].sort_values()
base = d20_ex[FWD][ok].mean()
print(f"     n={int(m.sum())}  cond {d20_ex[FWD][m].mean()*100:+.3f}%  base {base*100:+.3f}%  "
      f"median cond {d20_ex[FWD][m].median()*100:+.3f}%  median base {d20_ex[FWD][ok].median()*100:+.3f}%")
print(f"     top 5 contributors: " + ", ".join(f"{i.date()} {v*100:+.2f}%" for i, v in sd.tail(5).items()))
print(f"     bottom 5:           " + ", ".join(f"{i.date()} {v*100:+.2f}%" for i, v in sd.head(5).items()))
tot = (d20_ex[FWD][m] - base).sum()
top5 = (sd.tail(5) - base).sum()
print(f"     share of the total excess coming from the best 5 of {int(m.sum())} days: {top5/tot*100:.0f}%")
print("     drop-one-quarter jackknife WITHIN the ex-COVID 2020s block:")
q = pd.PeriodIndex(d20_ex.index, freq="Q")
rows = []
for qq in sorted(set(q)):
    keep = q != qq
    sub = d20_ex[keep]
    e, n, _, _ = excess(sub, sig)
    nq = int((sig.reindex(d20_ex.index).fillna(False) & ok & (q == qq)).sum())
    rows.append((str(qq), e, n, nq))
rows.sort(key=lambda r: r[1])
print("       most influential quarters (lowest ex-quarter excess = quarter was HELPING):")
for r in rows[:4]:
    print(f"         drop {r[0]}: excess {r[1]:+.3f}p (n={r[2]}, that quarter had {r[3]} signal days)")
print("       (highest ex-quarter excess = quarter was HURTING):")
for r in rows[-3:]:
    print(f"         drop {r[0]}: excess {r[1]:+.3f}p (n={r[2]}, that quarter had {r[3]} signal days)")

print("\n### R2. How much of the claim's p=.029 is a SEARCHED window? Placebo p-value distribution.")
print("     For each sliding exclusion window, compute the rotation p of the resulting excess.")
sig = SIGNALS["stretch>=+20%"]
ps, es = [], []
for s in range(0, len(idx20) - WLEN + 1, 15):
    keep = np.ones(len(idx20), bool); keep[s:s + WLEN] = False
    sub = d20[keep]
    e, n, p = excess_p(sub, sig, n_rot=2000)
    if n >= 25:
        ps.append(p); es.append(e)
ps = np.array(ps); es = np.array(es)
print(f"     {len(ps)} placebo excisions: fraction with p<.05 = {(ps<.05).mean()*100:.0f}%, "
      f"p<.10 = {(ps<.10).mean()*100:.0f}%; median excess {np.median(es):+.3f}p")
print(f"     best (most significant) placebo p = {ps.min():.4f} at excess {es[ps.argmin()]:+.3f}p")

print("\nDONE")
