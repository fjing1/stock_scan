"""
_vix_wf_verify_mechanism_covidflip3.py -- part 3, the decisive controls.

Part 1: the claim's arithmetic is exact; COVID really is the most influential 5-month block.
Part 2: but symmetric outlier treatment / mean-reversion controls leave the 2020s last.

Part 3 closes the loop:
  Q1  Does the CLAIMED positive (ex-COVID 2020s +0.210 / +0.800) survive the trailing-SPX
      mean-reversion control? If not, the "clean" number is index mean reversion, not VIX.
  Q2  Exactly which days does the calendar window delete that a VIX>=40 rule does NOT, and how
      much of the flip do those extra days buy?
  Q3  Combined: winsorise AND residualise, every era, no calendar surgery.
  Q4  Formal decay test: is (pre-2020 excess - 2020s excess) significantly > 0 under an
      outlier-robust estimator? Block bootstrap on the difference.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import _vix_data

RNG = np.random.default_rng(4242)
FWD = "g5"
COVID0, COVID1 = pd.Timestamp("2020-02-01"), pd.Timestamp("2020-06-30")
ERAS = [("1990s", 1990, 1999), ("2000s", 2000, 2009), ("2010s", 2010, 2019), ("2020s", 2020, 2026)]

d = _vix_data.add_features(_vix_data.load())
d["tr5"] = d.spx / d.spx.shift(5) - 1.0
d["tr10"] = d.spx / d.spx.shift(10) - 1.0
d["dd63"] = d.spx / d.spx.rolling(63).max() - 1.0
CTRL = ["tr5", "tr10", "dd63", "vix"]

SIGNALS = {
    "stretch>=+10%": d.stretch >= 0.10,
    "stretch>=+20%": d.stretch >= 0.20,
    "BB(10,2) above": d["bb10_2.0_above"].astype(bool),
    "BB(10,2) re-entry": d["bb10_2.0_reentry"].astype(bool),
}


def resid_frame(sub, cols=CTRL):
    f = sub[FWD]
    X = sub[cols]
    ok = f.notna() & X.notna().all(axis=1)
    A = np.column_stack([np.ones(int(ok.sum()))] + [X[c][ok].values for c in cols])
    y = f[ok].values
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return pd.Series(y - A @ beta, index=f[ok].index)


def exc_series(r, sig):
    m = sig.reindex(r.index).fillna(False).astype(bool)
    if m.sum() == 0:
        return np.nan, 0
    return (r[m].mean() - r.mean()) * 100, int(m.sum())


d20 = d[d.index.year >= 2020]
covid_in20 = (d20.index >= COVID0) & (d20.index <= COVID1)
d20ex = d20[~covid_in20]

print("=" * 104)
print("### Q1. Does the CLAIM'S OWN positive number survive the mean-reversion control?")
print("     Frame = 2020-2026 EXCLUDING the COVID window (exactly the claim's frame).")
print("     Then residualise g5 on trailing SPX move + VIX level, fit inside that frame.")
print(f"     {'signal':<20}{'raw exc':>12}{'resid exc':>12}{'n':>6}   {'% of edge that was mean-reversion':>10}")
for name, sig in SIGNALS.items():
    f = d20ex[FWD]; ok = f.notna()
    m = sig.reindex(d20ex.index).fillna(False).astype(bool) & ok
    raw = (f[m].mean() - f[ok].mean()) * 100
    r = resid_frame(d20ex)
    res, n = exc_series(r, sig)
    share = (1 - res / raw) * 100 if raw != 0 else np.nan
    print(f"     {name:<20}{raw:>+11.3f}p{res:>+11.3f}p{n:>6}   {share:>10.0f}%")
print("     Same, for the 2010s ex-nothing (reference for what a 'real' era looks like):")
d10 = d[(d.index.year >= 2010) & (d.index.year <= 2019)]
for name, sig in SIGNALS.items():
    f = d10[FWD]; ok = f.notna()
    m = sig.reindex(d10.index).fillna(False).astype(bool) & ok
    raw = (f[m].mean() - f[ok].mean()) * 100
    res, n = exc_series(resid_frame(d10), sig)
    print(f"     {'  2010s ' + name:<20}{raw:>+11.3f}p{res:>+11.3f}p{n:>6}")

print("\n" + "=" * 104)
print("### Q2. WHICH days does the calendar window delete that a VIX>=40 rule does not?")
for name in ("stretch>=+10%", "stretch>=+20%"):
    sig = SIGNALS[name]
    m20 = sig.reindex(d20.index).fillna(False).astype(bool) & d20[FWD].notna()
    del_cal = d20[m20 & covid_in20]
    extra = del_cal[del_cal.vix < 40]
    print(f"\n  {name}: calendar window deletes {len(del_cal)} signal days "
          f"(mean g5 {del_cal[FWD].mean()*100:+.2f}%)")
    print(f"     of which VIX>=40: {len(del_cal)-len(extra)} days, mean g5 "
          f"{del_cal[del_cal.vix>=40][FWD].mean()*100:+.2f}%")
    print(f"     of which VIX<40 (the EXTRA days a level rule would keep): {len(extra)} days, "
          f"mean g5 {extra[FWD].mean()*100:+.2f}%")
    if len(extra):
        print("       " + ", ".join(f"{i.date()} VIX{r.vix:.0f} g5{r[FWD]*100:+.1f}%"
                                    for i, r in extra.iterrows()))
    print(f"     2020s baseline for reference: {d20[FWD].mean()*100:+.3f}%")

print("\n" + "=" * 104)
print("### Q3. COMBINED outlier-robust + mean-reversion control, EVERY era, NO calendar surgery.")
print("     (winsorise g5 at each era's 2.5%/97.5%, then residualise on tr5+tr10+dd63+VIX)")
print(f"     {'signal':<20}" + "".join(f"{e[0]:>17}" for e in ERAS))
for name, sig in SIGNALS.items():
    cells = []
    for _, y0, y1 in ERAS:
        sub = d[(d.index.year >= y0) & (d.index.year <= y1)].copy()
        f = sub[FWD]; ok = f.notna()
        lo, hi = f[ok].quantile(.025), f[ok].quantile(.975)
        sub[FWD] = f.clip(lo, hi)
        r = resid_frame(sub)
        e, n = exc_series(r, sig)
        cells.append(f"{e:>+8.3f}p n{n:<4}" if n >= 25 else f"{'n<25':>9} n{n:<4}")
    print(f"     {name:<20}" + "".join(f"{c:>17}" for c in cells))

print("\n" + "=" * 104)
print("### Q4. FORMAL DECAY TEST under the outlier-robust estimator (no calendar surgery).")
print("     H0: 2020s excess == pre-2020 excess. Stationary block bootstrap (block=21d, 5000 reps)")
print("     on the DIFFERENCE (pre2020 excess - 2020s excess), using 2.5% winsorised g5.")


def wins_excess(sub, sig, q=0.025):
    f = sub[FWD]; ok = f.notna()
    lo, hi = f[ok].quantile(q), f[ok].quantile(1 - q)
    fw = f.clip(lo, hi)
    m = sig.reindex(sub.index).fillna(False).astype(bool) & ok
    if m.sum() == 0:
        return np.nan
    return (fw[m].mean() - fw[ok].mean()) * 100


def block_boot_diff(sig, reps=5000, block=21):
    pre = d[d.index.year < 2020]
    post = d[d.index.year >= 2020]
    obs = wins_excess(pre, sig) - wins_excess(post, sig)
    out = np.empty(reps)
    arrs = {}
    for lab, fr in (("pre", pre), ("post", post)):
        f = fr[FWD]; ok = f.notna()
        lo, hi = f[ok].quantile(.025), f[ok].quantile(.975)
        arrs[lab] = (f.clip(lo, hi).values, sig.reindex(fr.index).fillna(False).astype(bool).values,
                     ok.values)
    for i in range(reps):
        vals = []
        for lab in ("pre", "post"):
            y, s, ok = arrs[lab]
            n = len(y)
            nb = int(np.ceil(n / block))
            starts = RNG.integers(0, n, size=nb)
            idx = np.concatenate([(np.arange(st, st + block) % n) for st in starts])[:n]
            yy, ss, oo = y[idx], s[idx], ok[idx]
            mm = ss & oo
            vals.append(((yy[mm].mean() - yy[oo].mean()) * 100) if mm.sum() else np.nan)
        out[i] = vals[0] - vals[1]
    out = out[~np.isnan(out)]
    p = float((out <= 0).mean())      # one-sided: is pre truly > post?
    return obs, p, out


for name, sig in SIGNALS.items():
    obs, p, dist = block_boot_diff(sig)
    print(f"     {name:<20} pre2020 - 2020s = {obs:>+7.3f}p   "
          f"bootstrap P(diff<=0) = {p:.3f}   [boot mean {dist.mean():+.3f}p, sd {dist.std():.3f}]")

print("\n" + "=" * 104)
print("### Q5. Sanity: the same era table at other horizons (g3, g10), 2.5% winsorised, raw excess")
for h in (3, 10):
    col = f"g{h}"
    print(f"\n  horizon {col}:")
    print(f"     {'signal':<20}" + "".join(f"{e[0]:>17}" for e in ERAS))
    for name, sig in SIGNALS.items():
        cells = []
        for _, y0, y1 in ERAS:
            sub = d[(d.index.year >= y0) & (d.index.year <= y1)]
            f = sub[col]; ok = f.notna()
            lo, hi = f[ok].quantile(.025), f[ok].quantile(.975)
            fw = f.clip(lo, hi)
            m = sig.reindex(sub.index).fillna(False).astype(bool) & ok
            e = (fw[m].mean() - fw[ok].mean()) * 100 if m.sum() else np.nan
            cells.append(f"{e:>+8.3f}p n{int(m.sum()):<4}" if m.sum() >= 25 else f"{'n<25':>9} n{int(m.sum()):<4}")
        print(f"     {name:<20}" + "".join(f"{c:>17}" for c in cells))

print("\nDONE")
