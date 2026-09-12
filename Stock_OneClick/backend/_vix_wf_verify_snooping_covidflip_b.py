"""
_vix_wf_verify_snooping_covidflip_b.py — part 2 of the adversarial check.

  T8  why is my rotation p 0.070 when the claim says 0.029? enumerate the estimator variants.
  T9  STUDENTIZED family-wise max-statistic (the naive max-stat in part 1 was dominated by the
      small-n configs, so it read ~1.00 for everything).
  T10 the FAIR version of the claim's operation: instead of deleting the one window that hurts,
      apply a pre-specified symmetric outlier treatment (winsorize / trim g5 at full-sample
      percentiles) to EVERY era equally. If the 2020s only turns positive when you delete
      COVID but not when you winsorize, the flip is the deletion, not the tail.
  T11 is the "2020s went negative" fact even real? test excess_2020s - excess_pre2020.
  T12 the median-excess support: is a positive median excess in the 2020s informative, or is a
      positive median excess simply true in every era and every rotation?

Run: ../../vcp_env/bin/python _vix_wf_verify_snooping_covidflip_b.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _vix_data

RNG = np.random.default_rng(20260910)
N_ROT = 5000
MIN_N = 25
HORIZONS = [1, 3, 5, 10, 21]


def hr(t):
    print("\n" + "=" * 108)
    print(t)
    print("=" * 108)


d = _vix_data.add_features(_vix_data.load())
idx = d.index
N = len(d)
g5 = d.g5.values
valid5 = ~np.isnan(g5)

SIGNALS = {
    "stretch>=+10%":       (d.stretch >= 0.10).values,
    "stretch>=+20%":       (d.stretch >= 0.20).values,
    "stretch>=+25%":       (d.stretch >= 0.25).values,
    "BB(10,2.0) above":    d["bb10_2.0_above"].values,
    "BB(10,2.0) re-entry": d["bb10_2.0_reentry"].values,
}
BULL4 = ["stretch>=+10%", "stretch>=+20%", "BB(10,2.0) above", "BB(10,2.0) re-entry"]
is20s = np.asarray(idx.year >= 2020)
COVID = np.asarray((idx >= "2020-02-01") & (idx <= "2020-06-30"))
keep20_ex = is20s & ~COVID
ERAS = [("1990s", np.asarray(idx.year <= 1999)),
        ("2000s", np.asarray((idx.year >= 2000) & (idx.year <= 2009))),
        ("2010s", np.asarray((idx.year >= 2010) & (idx.year <= 2019))),
        ("2020s", is20s),
        ("2020s-exCOVID", keep20_ex)]


def circ_corr(a, b):
    n = len(a)
    return np.fft.irfft(np.conj(np.fft.rfft(a)) * np.fft.rfft(b), n)


def null_means_full(mask, fwd, keep, min_n=MIN_N):
    """rotate the mask over the WHOLE series, then intersect with `keep` (variant A)."""
    v = ~np.isnan(fwd)
    w = (keep & v).astype(float)
    y = np.where(v, np.nan_to_num(fwd), 0.0) * w
    num, den = circ_corr(mask.astype(float), y), circ_corr(mask.astype(float), w)
    dr = np.rint(den)
    return np.where(dr >= min_n, num / np.where(dr > 0, den, np.nan), np.nan)[1:]


def null_means_sub(mask, fwd, keep, min_n=MIN_N):
    """subset to `keep` FIRST, then rotate inside that subsample (variant B)."""
    v = ~np.isnan(fwd)
    k = keep & v
    m, y = mask[k].astype(float), fwd[k]
    n = len(y)
    num, den = circ_corr(m, y), circ_corr(m, np.ones(n))
    dr = np.rint(den)
    return np.where(dr >= min_n, num / np.where(dr > 0, den, np.nan), np.nan)[1:]


def p_from_null(null, observed, two_sided=True):
    null = null[~np.isnan(null)]
    if len(null) < 100:
        return float("nan"), len(null)
    c = null.mean()
    if two_sided:
        return float((np.abs(null - c) >= abs(observed - c)).mean()), len(null)
    return float((null >= observed).mean()), len(null)


# ------------------------------------------------------------------ T8
hr("T8 — WHICH ROTATION ESTIMATOR GIVES p=0.029? (stretch>=+20%, D5, 2020s ex-COVID, n=72)")
print(f"{'variant':<52}{'p':>9}{'null draws':>12}")
print("-" * 108)
for sname in ["stretch>=+20%", "stretch>=+10%", "BB(10,2.0) above", "BB(10,2.0) re-entry"]:
    m = SIGNALS[sname]
    k = keep20_ex & valid5
    cond = g5[m & k].mean()
    print(f"  [{sname}] observed cond mean {cond*100:+.3f}%  "
          f"excess {(cond - g5[k].mean())*100:+.3f}pp  n={int((m & k).sum())}")
    for lbl, nl, ts in [
        ("A rotate whole series, intersect era  (2-sided)", null_means_full(m, g5, keep20_ex), True),
        ("A rotate whole series, intersect era  (1-sided)", null_means_full(m, g5, keep20_ex), False),
        ("B rotate INSIDE the era subsample     (2-sided)", null_means_sub(m, g5, keep20_ex), True),
        ("B rotate INSIDE the era subsample     (1-sided)", null_means_sub(m, g5, keep20_ex), False),
    ]:
        p, nd = p_from_null(nl, cond, ts)
        print(f"    {lbl:<50}{p:>9.3f}{nd:>12,}")
    # 5000 random draws (what the claim used), several seeds, variant B
    ps = []
    for seed in range(6):
        r = np.random.default_rng(1000 + seed)
        nl = null_means_sub(m, g5, keep20_ex)
        nl = nl[~np.isnan(nl)]
        draw = r.choice(nl, size=min(N_ROT, len(nl)), replace=True)
        p, _ = p_from_null(draw, cond, True)
        ps.append(p)
    print(f"    {'B with 5000 random draws, 6 seeds':<50}"
          f"{np.mean(ps):>9.3f}   range [{min(ps):.3f},{max(ps):.3f}]")

# ------------------------------------------------------------------ T9 studentized FWE
hr("T9 — STUDENTIZED family-wise max-statistic (Romano-Wolf style), shared rotations")
family = [(s, h, en, ek) for s in SIGNALS for h in HORIZONS for en, ek in ERAS]
zs, znull, lab = [], [], []
for s, h, en, ek in family:
    m = SIGNALS[s]
    fwd = d[f"g{h}"].values
    v = ~np.isnan(fwd)
    k = ek & v
    n = int((m & k).sum())
    if n < MIN_N:
        continue
    cond, base = fwd[m & k].mean(), fwd[k].mean()
    nl = null_means_full(m, fwd, ek)
    ok = ~np.isnan(nl)
    if ok.sum() < 800:
        continue
    c, sd = np.nanmean(nl), np.nanstd(nl)
    if not np.isfinite(sd) or sd <= 0:
        continue
    zs.append(abs(cond - c) / sd)
    z = np.abs(nl - c) / sd
    znull.append(z)
    lab.append((s, h, en, cond - base, n))

L = max(len(z) for z in znull)
Z = np.full((len(znull), L), np.nan)
for i, z in enumerate(znull):
    Z[i, :len(z)] = z
maxnull = np.nanmax(Z, axis=0)
cover = (~np.isnan(Z)).sum(axis=0)
good = cover >= int(0.9 * len(znull))
maxnull = maxnull[good]
print(f"  family: {len(lab)} configs (5 signals x 5 horizons x 5 samples, n>=25); "
      f"{good.sum():,} usable shared rotations")
print(f"\n{'signal':<22}{'H':>3}  {'sample':<16}{'n':>6}{'excess':>10}{'z':>7}{'raw p':>8}{'FWE p':>8}{'BH q':>8}")
print("-" * 108)
raws = np.array([float((znull[i] >= zs[i])[~np.isnan(znull[i])].mean()) for i in range(len(zs))])
fwes = np.array([float((maxnull >= zs[i]).mean()) for i in range(len(zs))])
o = np.argsort(raws)
mm = len(raws)
bh = np.empty(mm)
prev = 1.0
for r in range(mm - 1, -1, -1):
    i = o[r]
    prev = min(prev, raws[i] * mm / (r + 1))
    bh[i] = prev
for i in o[:14]:
    s, h, en, exc, n = lab[i]
    print(f"{s:<22}{h:>3}  {en:<16}{n:>6}{exc*100:>9.3f}pp{zs[i]:>7.2f}{raws[i]:>8.3f}{fwes[i]:>8.3f}{bh[i]:>8.3f}")
print("\n  the claim's headline + the three companions it says 'all flip positive':")
for i in range(len(lab)):
    s, h, en, exc, n = lab[i]
    if h == 5 and en == "2020s-exCOVID" and s in BULL4:
        print(f"    {s:<22} n={n:<5} exc {exc*100:+7.3f}pp  z={zs[i]:.2f}  raw p={raws[i]:.3f}  "
              f"FWE p={fwes[i]:.3f}  BH q={bh[i]:.3f}")
print(f"\n  configs with BH q<0.10: {int((bh < 0.10).sum())} / {mm}")
print(f"  configs with FWE p<0.05: {int((fwes < 0.05).sum())} / {mm}")

# ------------------------------------------------------------------ T10 fair robustness
hr("T10 — THE FAIR VERSION: pre-specified symmetric outlier treatment applied to EVERY era")
print("  (winsorize/trim g5 at FULL-SAMPLE percentiles -- no cherry-picked date window)\n")
for s in BULL4:
    m = SIGNALS[s]
    print(f"  {s}")
    print(f"    {'treatment':<26}" + "".join(f"{e[0]:>16}" for e in ERAS))
    rows = [("raw (no treatment)", None, "wins")]
    for q in (0.005, 0.01, 0.025, 0.05):
        rows.append((f"winsorize {q*100:.1f}/{100-q*100:.1f}", q, "wins"))
    for q in (0.01, 0.025, 0.05):
        rows.append((f"trim {q*100:.1f}/{100-q*100:.1f}", q, "trim"))
    rows.append(("median (not mean)", None, "med"))
    for lbl, q, kind in rows:
        cells = []
        for en, ek in ERAS:
            k = ek & valid5
            y = g5.copy()
            keeprow = np.ones(N, bool)
            if kind == "wins" and q is not None:
                lo, hi = np.nanpercentile(g5[valid5], [q * 100, 100 - q * 100])
                y = np.clip(y, lo, hi)
            elif kind == "trim":
                lo, hi = np.nanpercentile(g5[valid5], [q * 100, 100 - q * 100])
                keeprow = (g5 >= lo) & (g5 <= hi)
            kk = k & keeprow
            sel = m & kk
            if sel.sum() < MIN_N:
                cells.append(f"{'n<25':>16}")
                continue
            if kind == "med":
                exc = np.median(g5[sel]) - np.median(g5[kk])
            else:
                exc = y[sel].mean() - y[kk].mean()
            cells.append(f"{exc*100:+8.3f}(n{int(sel.sum())})".rjust(16))
        print(f"    {lbl:<26}" + "".join(cells))
    print()

# ------------------------------------------------------------------ T11 era difference
hr("T11 — WAS THERE EVER A '2020s NEGATIVE' TO EXPLAIN? test excess(2020s) - excess(pre-2020)")
print("  Null: the era label carries no information. Rotate the SIGNAL mask over the whole")
print("  series and recompute the same era-difference. n reported for both halves.\n")
pre = np.asarray(idx.year <= 2019)
print(f"{'signal':<22}{'n pre':>7}{'exc pre':>10}{'n 20s':>7}{'exc 20s':>10}{'diff':>10}{'rot p(diff)':>13}")
print("-" * 108)
for s in BULL4:
    m = SIGNALS[s]
    kp, kt = pre & valid5, is20s & valid5
    ep = g5[m & kp].mean() - g5[kp].mean()
    et = g5[m & kt].mean() - g5[kt].mean()
    nlp = null_means_full(m, g5, pre)
    nlt = null_means_full(m, g5, is20s)
    dif = (nlt - g5[kt].mean()) - (nlp - g5[kp].mean())
    dif = dif[~np.isnan(dif)]
    obs = et - ep
    c = dif.mean()
    p = float((np.abs(dif - c) >= abs(obs - c)).mean())
    print(f"{s:<22}{int((m&kp).sum()):>7}{ep*100:>9.3f}pp{int((m&kt).sum()):>7}{et*100:>9.3f}pp"
          f"{obs*100:>9.3f}pp{p:>13.3f}")
# and the same but with the 2020s ex-COVID
print()
for s in BULL4:
    m = SIGNALS[s]
    kp, kt = pre & valid5, keep20_ex & valid5
    ep = g5[m & kp].mean() - g5[kp].mean()
    et = g5[m & kt].mean() - g5[kt].mean()
    print(f"  {s:<22} pre-2020 {ep*100:+.3f}pp (n={int((m&kp).sum())})  vs  "
          f"2020s-exCOVID {et*100:+.3f}pp (n={int((m&kt).sum())})  diff {(et-ep)*100:+.3f}pp")

# ------------------------------------------------------------------ T12 median support
hr("T12 — IS A POSITIVE MEDIAN EXCESS INFORMATIVE? (the claim's corroborating stat)")
print("  median(signal g5) - median(same-sample g5), by era, WITH covid included.\n")
print(f"{'signal':<22}" + "".join(f"{e[0]:>17}" for e in ERAS))
print("-" * 108)
for s in BULL4:
    m = SIGNALS[s]
    cells = []
    for en, ek in ERAS:
        k = ek & valid5
        sel = m & k
        if sel.sum() < MIN_N:
            cells.append(f"{'n<25':>17}")
            continue
        cells.append(f"{(np.median(g5[sel]) - np.median(g5[k]))*100:+9.3f}pp".rjust(17))
    print(f"{s:<22}" + "".join(cells))
print("\n  rotation p for the 2020s (WITH covid) median excess, and the share of rotated masks")
print("  that also produce a positive median excess in the 2020s:")
for s in BULL4:
    m = SIGNALS[s]
    k = is20s & valid5
    obs = np.median(g5[m & k]) - np.median(g5[k])
    ii = np.flatnonzero(k)
    yv = g5[ii]
    mm_ = m[ii]
    nsub = len(ii)
    meds = np.empty(nsub - 1)
    for j, off in enumerate(range(1, nsub)):
        r = np.roll(mm_, off)
        meds[j] = np.median(yv[r]) if r.sum() >= MIN_N else np.nan
    meds = meds[~np.isnan(meds)]
    base = np.median(yv)
    exc_null = meds - base
    c = exc_null.mean()
    p = float((np.abs(exc_null - c) >= abs(obs - c)).mean())
    print(f"    {s:<22} median excess {obs*100:+.3f}pp   rot p={p:.3f}   "
          f"{(exc_null > 0).mean()*100:.0f}% of ROTATED (random-timing) masks also give a positive "
          f"median excess")

print("\ndone.")
