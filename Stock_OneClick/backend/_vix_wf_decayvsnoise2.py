"""
_vix_wf_decayvsnoise2.py — robustness companion to _vix_wf_decayvsnoise.py.

Part 1 found: (a) every bullish signal's late-period excess is ~0, (b) only BB(10,2) above-upper
shows a statistically significant early-vs-late DIFFERENCE, (c) removing Feb-Jun 2020 flips the
2020s back positive. That last result is suspicious — deleting the largest vol event of the
sample is exactly the deletion that flatters a signal designed to fire in vol events. This script
stress-tests it.

  6. influence / jackknife: drop one calendar year at a time; drop one quarter at a time
  7. robust location: median and winsorised-mean excess (is the sign driven by a few tails?)
  8. SYMMETRIC crisis exclusion — if you delete COVID from the late window you must delete the
     GFC from the early window
  9. non-overlapping ~6y blocks (the rolling-10y table overlaps 90%, so it cannot show a trend)
 10. family-wise multiple testing on the 5 early-vs-late difference tests (max-|z| rotation)
 11. the BEARISH family at wider specs, where the late-window sample is not starved

Run: ../../vcp_env/bin/python _vix_wf_decayvsnoise2.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _vix_data

RNG = np.random.default_rng(20260910)
N_ROT = 5000
H = 5
MIN_N = 25

SIGNALS = [
    ("stretch>=+10%", lambda d: (d.stretch >= 0.10).values),
    ("stretch>=+20%", lambda d: (d.stretch >= 0.20).values),
    ("BB(10,2) above upper", lambda d: d["bb10_2.0_above"].values.astype(bool)),
    ("BB(10,2) re-entry", lambda d: d["bb10_2.0_reentry"].values.astype(bool)),
    ("BB(10,2) below lower", lambda d: d["bb10_2.0_below"].values.astype(bool)),
]


def rot_null(mask, fwd, offs):
    n = len(mask)
    idx0 = np.flatnonzero(mask)
    if idx0.size == 0:
        return np.full(len(offs), np.nan)
    valid = ~np.isnan(fwd)
    fv = np.where(valid, fwd, 0.0)
    vv = valid.astype(np.float64)
    j = (idx0[:, None] + offs[None, :]) % n
    cnt = vv[j].sum(axis=0)
    s = fv[j].sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


def rot_p(mask, fwd, observed, n_rot=N_ROT):
    offs = RNG.integers(1, len(mask), size=n_rot)
    null = rot_null(mask, fwd, offs)
    null = null[~np.isnan(null)]
    if not len(null):
        return float("nan")
    b = null.mean()
    return float((np.abs(null - b) >= abs(observed - b)).mean())


def exc(mask, fwd):
    valid = ~np.isnan(fwd)
    sel = mask & valid
    if sel.sum() == 0:
        return 0, np.nan, np.nan, np.nan
    return int(sel.sum()), fwd[sel].mean(), fwd[valid].mean(), fwd[sel].mean() - fwd[valid].mean()


def main():
    d = _vix_data.add_features(_vix_data.load())
    fwd = d[f"g{H}"].values
    yrs = d.index.year.values
    masks = {nm: fn(d) for nm, fn in SIGNALS}
    early, late = yrs <= 2014, yrs >= 2015
    covid = np.asarray((d.index >= "2020-02-01") & (d.index <= "2020-06-30"))
    gfc = np.asarray((d.index >= "2008-09-01") & (d.index <= "2009-06-30"))

    print(f"Panel {d.index[0].date()} -> {d.index[-1].date()}  {len(d):,} sessions; "
          f"outcome g{H}, {int((~np.isnan(fwd)).sum()):,} valid")

    # ------------------------------------------------------- 6. influence
    print(f"\n{'='*116}\n6. INFLUENCE — drop-one-calendar-year jackknife of the 2015-2026 excess "
          f"(how few observations carry the result?)\n{'='*116}")
    print(f"{'signal':<22}{'full 15-26':>12}{'jack min':>11}{'(year)':>8}{'jack max':>11}{'(year)':>8}"
          f"{'range':>9}{'#yrs flip sign':>16}")
    print("-" * 116)
    for nm, _ in SIGNALS:
        m = masks[nm]
        n0, _, _, e0 = exc(m & late, np.where(late, fwd, np.nan))
        if n0 < MIN_N:
            print(f"{nm:<22}  n={n0} INCONCLUSIVE")
            continue
        rows = []
        for Y in range(2015, 2027):
            keep = late & (yrs != Y)
            n1, _, _, e1 = exc(m & keep, np.where(keep, fwd, np.nan))
            rows.append((Y, e1, n1))
        es = np.array([r[1] for r in rows])
        lo, hi = rows[int(es.argmin())], rows[int(es.argmax())]
        flips = int(((es > 0) != (e0 > 0)).sum())
        print(f"{nm:<22}{e0*100:>+11.3f}%{lo[1]*100:>+10.3f}%{lo[0]:>8}{hi[1]*100:>+10.3f}%"
              f"{hi[0]:>8}{(hi[1]-lo[1])*100:>8.3f}{flips:>16}")

    print("\n  drop-one-QUARTER jackknife, 2015-2026 (the single most influential 3-month block):")
    q = pd.PeriodIndex(d.index, freq="Q")
    for nm, _ in SIGNALS:
        m = masks[nm]
        n0, _, _, e0 = exc(m & late, np.where(late, fwd, np.nan))
        if n0 < MIN_N:
            continue
        best = None
        for qq in q[late].unique():
            keep = late & np.asarray(q != qq)
            n1, _, _, e1 = exc(m & keep, np.where(keep, fwd, np.nan))
            nq = int((m & late & np.asarray(q == qq)).sum())
            if nq == 0:
                continue
            if best is None or abs(e1 - e0) > abs(best[1] - e0):
                best = (qq, e1, nq)
        print(f"    {nm:<22} full {e0*100:+.3f}%  ->  dropping {best[0]} (only {best[2]} signal "
              f"days) gives {best[1]*100:+.3f}%   shift {(best[1]-e0)*100:+.3f}pp")

    # ------------------------------------------------------- 7. robust location
    print(f"\n{'='*116}\n7. ROBUST LOCATION — is the sign a tail artefact? median excess and "
          f"5%-winsorised mean excess\n{'='*116}")
    def wins(x, p=0.05):
        lo, hi = np.quantile(x, p), np.quantile(x, 1 - p)
        return np.clip(x, lo, hi)
    periods = [("1990-2014", early), ("2015-2026", late), ("2020-2026", yrs >= 2020),
               ("2020-2026 exCOVID", (yrs >= 2020) & ~covid)]
    print(f"{'period':<20}{'signal':<22}{'n':>5}{'mean exc%':>11}{'median exc%':>13}"
          f"{'wins5 exc%':>12}{'win%-base':>11}")
    print("-" * 116)
    for pn, sub in periods:
        for nm, _ in SIGNALS:
            m = masks[nm] & sub
            v = ~np.isnan(fwd) & sub
            sel = m & v
            n = int(sel.sum())
            if n < MIN_N:
                print(f"{pn:<20}{nm:<22}{n:>5}   INCONCLUSIVE (n<25)")
                continue
            a, b = fwd[sel], fwd[v]
            print(f"{pn:<20}{nm:<22}{n:>5}{(a.mean()-b.mean())*100:>+11.3f}"
                  f"{(np.median(a)-np.median(b))*100:>+13.3f}"
                  f"{(wins(a).mean()-wins(b).mean())*100:>+12.3f}"
                  f"{((a>0).mean()-(b>0).mean())*100:>+11.1f}")
        print()

    # ------------------------------------------------------- 8. symmetric crisis exclusion
    print(f"{'='*116}\n8. SYMMETRIC CRISIS EXCLUSION — deleting COVID from the late window is only "
          f"fair if the GFC goes too\n{'='*116}")
    print(f"  COVID block 2020-02-01..2020-06-30 = {int(covid.sum())} sessions; "
          f"GFC block 2008-09-01..2009-06-30 = {int(gfc.sum())} sessions")
    print(f"\n{'signal':<22}{'early full':>12}{'early exGFC':>13}{'late full':>12}"
          f"{'late exCOVID':>14}{'diff full':>11}{'diff symm':>11}")
    print("-" * 116)
    for nm, _ in SIGNALS:
        m = masks[nm]
        out = []
        for sub in (early, early & ~gfc, late, late & ~covid):
            n, _, _, e = exc(m & sub, np.where(sub, fwd, np.nan))
            out.append((n, e))
        cells = []
        for n, e in out:
            cells.append(f"{'n<25':>12}" if n < MIN_N else f"{e*100:>+11.3f}%")
        dfull = out[2][1] - out[0][1]
        dsym = out[3][1] - out[1][1]
        ok = all(n >= MIN_N for n, _ in out)
        print(f"{nm:<22}{cells[0]}{cells[1]:>13}{cells[2]:>12}{cells[3]:>14}"
              f"{dfull*100:>+10.3f}{dsym*100:>+10.3f}" if ok else
              f"{nm:<22}{cells[0]}{cells[1]:>13}{cells[2]:>12}{cells[3]:>14}   (some n<25)")
    print("\n  ns per cell:")
    for nm, _ in SIGNALS:
        m = masks[nm]
        ns = [exc(m & sub, np.where(sub, fwd, np.nan))[0]
              for sub in (early, early & ~gfc, late, late & ~covid)]
        print(f"    {nm:<22} early {ns[0]:>4} / exGFC {ns[1]:>4} / late {ns[2]:>4} / exCOVID {ns[3]:>4}")

    # what the deleted blocks actually contained
    print("\n  what the deleted blocks contain (mean g5 of the signal days inside them):")
    for tag, blk in (("COVID 2020H1", covid), ("GFC 2008H2-09H1", gfc)):
        for nm, _ in SIGNALS:
            sel = masks[nm] & blk & ~np.isnan(fwd)
            if sel.sum() == 0:
                continue
            print(f"    {tag:<18}{nm:<22} n={int(sel.sum()):>3}  mean g5 {fwd[sel].mean()*100:+7.2f}%"
                  f"   (block baseline {np.nanmean(np.where(blk, fwd, np.nan))*100:+.2f}%)")

    # ------------------------------------------------------- 9. non-overlapping blocks
    print(f"\n{'='*116}\n9. NON-OVERLAPPING ~6-YEAR BLOCKS (the rolling-10y table overlaps 90%; "
          f"these do not)\n{'='*116}")
    blocks = [(1990, 1995), (1996, 2001), (2002, 2007), (2008, 2013), (2014, 2019), (2020, 2026)]
    print(f"{'block':<12}{'sessions':>9}{'base%':>8}   " + "".join(f"{s[:17]:>19}" for s, _ in SIGNALS))
    print(f"{'':<12}{'':>9}{'':>8}   " + "".join(f"{'exc% (n)':>19}" for _ in SIGNALS))
    print("-" * 116)
    for a, b in blocks:
        sub = (yrs >= a) & (yrs <= b)
        fs = np.where(sub, fwd, np.nan)
        line = f"{f'{a}-{b}':<12}{int((~np.isnan(fs)).sum()):>9}{np.nanmean(fs)*100:>7.3f}   "
        for nm, _ in SIGNALS:
            n, _, _, e = exc(masks[nm] & sub, fs)
            line += f"{'n<25 (' + str(n) + ')':>19}" if n < MIN_N else f"{e*100:>+12.3f} ({n:>3})"
        print(line)

    # ------------------------------------------------------- 10. family-wise
    print(f"\n{'='*116}\n10. FAMILY-WISE MULTIPLE TESTING on the 5 early-vs-late difference tests"
          f"\n{'='*116}")
    offs = RNG.integers(1, len(d), size=N_ROT)
    be, bl = np.nanmean(np.where(early, fwd, np.nan)), np.nanmean(np.where(late, fwd, np.nan))
    zs, obs_d, per_p = {}, {}, {}
    nulls = {}
    for nm, _ in SIGNALS:
        m = masks[nm]
        ne_ = rot_null(m & np.ones(len(m), bool), np.where(early, fwd, np.nan), offs) - be
        nl_ = rot_null(m, np.where(late, fwd, np.nan), offs) - bl
        nd = nl_ - ne_
        _, _, _, e_e = exc(m & early, np.where(early, fwd, np.nan))
        _, _, _, e_l = exc(m & late, np.where(late, fwd, np.nan))
        dobs = e_l - e_e
        ok = ~np.isnan(nd)
        nulls[nm] = nd
        mu, sd = np.nanmean(nd), np.nanstd(nd)
        zs[nm] = abs(dobs - mu) / sd
        obs_d[nm] = dobs
        per_p[nm] = float((np.abs(nd[ok] - mu) >= abs(dobs - mu)).mean())
    Z = np.vstack([(np.abs(nulls[nm] - np.nanmean(nulls[nm])) / np.nanstd(nulls[nm]))
                   for nm, _ in SIGNALS])
    maxz = np.nanmax(Z, axis=0)
    maxz = maxz[~np.isnan(maxz)]
    print(f"{'signal':<22}{'diff pp':>10}{'|z|':>8}{'p (own)':>10}{'p family-wise':>15}"
          f"{'Bonferroni x5':>15}")
    print("-" * 116)
    for nm, _ in SIGNALS:
        pfw = float((maxz >= zs[nm]).mean())
        print(f"{nm:<22}{obs_d[nm]*100:>+10.3f}{zs[nm]:>8.2f}{per_p[nm]:>10.3f}{pfw:>15.3f}"
              f"{min(per_p[nm]*5, 1.0):>15.3f}")
    print("\n  p family-wise = fraction of rotations whose LARGEST |z| across all 5 signals is at")
    print("  least as big as this signal's observed |z| (step-down max-statistic; controls FWER).")

    # ------------------------------------------------------- 11. bearish family
    print(f"\n{'='*116}\n11. THE BEARISH FAMILY at wider specs — BB(10,2) below fires only "
          f"{int((masks['BB(10,2) below lower'] & late).sum())}x since 2015, so widen it\n{'='*116}")
    bear = [
        ("BB(10,2.0) below", d["bb10_2.0_below"].values.astype(bool)),
        ("BB(10,1.5) below", d["bb10_1.5_below"].values.astype(bool)),
        ("BB(20,2.0) below", d["bb20_2.0_below"].values.astype(bool)),
        ("BB(20,1.5) below", d["bb20_1.5_below"].values.astype(bool)),
        ("stretch<=-5%", (d.stretch <= -0.05).values),
        ("stretch<=-10%", (d.stretch <= -0.10).values),
        ("VIX 1y pctile <=10%", (d.vix_pct1y <= 0.10).values & d.vix_pct1y.notna().values),
    ]
    print(f"{'signal':<22}{'n all':>7}{'exc all%':>10}{'p':>7}   {'n_e':>5}{'exc_e%':>9}{'p_e':>7}"
          f"   {'n_l':>5}{'exc_l%':>9}{'p_l':>7}   {'n 20s':>6}{'exc 20s%':>10}{'p':>7}")
    print("-" * 116)
    for nm, m in bear:
        cells = []
        for sub in (np.ones(len(d), bool), early, late, yrs >= 2020):
            fs = np.where(sub, fwd, np.nan)
            n, cond, _, e = exc(m & sub, fs)
            if n < MIN_N:
                cells.append((n, np.nan, np.nan))
            else:
                cells.append((n, e, rot_p(m[sub], fwd[sub], cond, 3000)))
        s = f"{nm:<22}"
        for i, (n, e, p) in enumerate(cells):
            if np.isnan(e):
                s += f"{n:>7}{'  n<25':>10}{'':>7}" if i == 0 else f"   {n:>5}{'  n<25':>9}{'':>7}"
            else:
                s += (f"{n:>7}{e*100:>+10.3f}{p:>7.3f}" if i == 0
                      else f"   {n:>5}{e*100:>+9.3f}{p:>7.3f}")
        print(s)

    print("\n  bearish family, 2015-2026 rolling check — excess by non-overlapping 4y block:")
    for nm, m in bear:
        parts = []
        for a, b in [(2015, 2018), (2019, 2022), (2023, 2026)]:
            sub = (yrs >= a) & (yrs <= b)
            n, _, _, e = exc(m & sub, np.where(sub, fwd, np.nan))
            parts.append(f"{a}-{b}: " + (f"n={n} n<25" if n < MIN_N else f"{e*100:+.2f}% (n={n})"))
        print(f"    {nm:<22} " + " | ".join(parts))


if __name__ == "__main__":
    main()
