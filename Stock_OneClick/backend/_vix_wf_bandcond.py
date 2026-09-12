"""
_vix_wf_bandcond.py — does the VIX BAND STATE add anything ON TOP of what the live scanner
already shows (term structure, and "index below MA20")?

Four questions, all measured as EXCESS over the SAME-SAMPLE unconditional baseline, all
significance from the circular rotation test (same approach as _vix_ma10_bb_research.py:
roll the boolean mask by a random offset, recompute, p = fraction of rolls at least as extreme).

  1. band state x term structure (term = vix/vix3m), 2006+, vix3m NaNs DROPPED (never ffilled).
  2. band state x VVIX tercile, 2007+ — the "fragile calm" hypothesis.
  3. band state x SPX above/below its 20d MA — does the vol state add to the risk-off flag?
  4. downside asymmetry per band state: 5th pctile of g5, max adverse excursion over next 10d,
     P(g5 < -2%).

Outcome column is always g* (entry at the NEXT close; VIX settles 16:15 ET).
Run: ../../vcp_env/bin/python _vix_wf_bandcond.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

import _vix_data  # noqa: E402

RNG = np.random.default_rng(20260910)
N_ROT = 4000
MIN_N = 25


# ------------------------------------------------------------------ rotation test (generic)
def rot_p(mask: np.ndarray, vals: np.ndarray, statfn=np.mean, n_rot: int = N_ROT,
          min_cell: int = 10) -> tuple[float, float]:
    """Circular-rotation p-value for an arbitrary statistic of the masked sample.

    Null: the signal's shape (count + burstiness) says nothing about WHEN it fires.
    Rolling the mask preserves both the autocorrelation of `vals` (overlapping forward
    windows) and the clustering of the signal, which a t-test does not.
    Returns (observed_stat, two_sided_p).
    """
    mask = np.asarray(mask, dtype=bool)
    vals = np.asarray(vals, dtype=float)
    valid = ~np.isnan(vals)
    sel = mask & valid
    if sel.sum() < min_cell:
        return (float("nan"), float("nan"))
    obs = float(statfn(vals[sel]))
    n = len(mask)
    offs = RNG.integers(1, n, size=n_rot)
    null = np.empty(n_rot)
    null[:] = np.nan
    for i, off in enumerate(offs):
        m = np.roll(mask, off) & valid
        if m.sum() >= min_cell:
            null[i] = statfn(vals[m])
    null = null[~np.isnan(null)]
    if len(null) < 100:
        return (obs, float("nan"))
    base = float(np.mean(null))
    p = float((np.abs(null - base) >= abs(obs - base)).mean())
    return (obs, p)


def stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "   "
    return "***" if p < .01 else ("** " if p < .05 else ("*  " if p < .10 else "   "))


# ------------------------------------------------------------------ stats used in part 4
def q05(x):
    return float(np.percentile(x, 5))


def p_lt2(x):
    return float((x < -0.02).mean())


# ------------------------------------------------------------------ data
def build():
    d = _vix_data.add_features(_vix_data.load())
    # 2026-05-25 and 2026-09-07 are US equity holidays where ^VIX printed but ^GSPC/^VVIX did not.
    # Left in place these two rows null out every 20d/252d rolling window that spans them
    # (rolling().rank/mean propagate NaN), silently deleting ~250 later sessions. Drop the rows
    # and recompute forward returns on the clean calendar rather than forward-filling anything.
    n0 = len(d)
    d = d[d.spx.notna()].copy()
    if len(d) != n0:
        print(f"  dropped {n0-len(d)} rows with no SPX print (VIX-only holiday bars)")
    for h in (1, 3, 5, 10, 21):
        d[f"g{h}"] = d.spx.shift(-(h + 1)) / d.spx.shift(-1) - 1.0

    # band state: the established BB(10,2.0) definition
    d["band"] = np.where(d["bb10_2.0_above"], "above",
                         np.where(d["bb10_2.0_below"], "below", "inside"))
    d.loc[d["bb10_2.0_up"].isna(), "band"] = None
    # SPX 20d MA regime (the scanner's risk-off flag)
    d["spx_ma20"] = d.spx.rolling(20).mean()
    d["above_ma20"] = (d.spx > d.spx_ma20) & d.spx_ma20.notna()   # always filter on spx_ma20.notna()
    # VVIX point-in-time 1y percentile. Computed on the dropna'd series then reindexed, so a
    # missing VVIX print costs one row instead of the following 252.
    vv = d.vvix.dropna()
    d["vvix_pct1y"] = vv.rolling(252).rank(pct=True).reindex(d.index)
    # max adverse / favourable excursion over the 10 sessions after a t+1 close entry
    ent = d.spx.shift(-1)
    paths = pd.concat([d.spx.shift(-(1 + k)) / ent - 1.0 for k in range(1, 11)], axis=1)
    d["mae10"] = paths.min(axis=1)
    d["mfe10"] = paths.max(axis=1)
    return d


def sect(t):
    print(f"\n{'='*112}\n{t}\n{'='*112}")


def cell_row(label, mask, vals, base, n_rot=N_ROT, statfn=np.mean, pct=True, width=112):
    n = int((mask & ~np.isnan(vals)).sum())
    if n < MIN_N:
        return f"  {label:<44}{n:>6}   (n<{MIN_N}: INCONCLUSIVE, not interpreted)"
    obs, p = rot_p(mask, vals, statfn, n_rot)
    scale = 100 if pct else 1
    return (f"  {label:<44}{n:>6}{obs*scale:>10.2f}{'%' if pct else ' '}"
            f"{(obs-base)*scale:>+10.2f}{'%' if pct else ' '}  p={p:>5.3f} {stars(p)}")


def head(colname="mean D5"):
    return f"  {'cell':<44}{'n':>6}{colname:>11}{'excess':>11}  {'rot p':>7}"


# ==================================================================== main
def main():
    d = build()
    print(f"panel {len(d):,} rows  {d.index[0].date()} -> {d.index[-1].date()}")
    print(f"band state counts (full history): "
          f"{d.band.value_counts().to_dict()}  (NaN warmup {int(d.band.isna().sum())})")

    # ---------------------------------------------------------------- PART 1: term structure
    sect("PART 1 — band state x VIX term structure (term = vix/vix3m).  2006-07-17+, vix3m NaNs DROPPED")
    t = d[d.term.notna() & d.band.notna()].copy()
    print(f"  surviving sample n={len(t):,}  {t.index[0].date()} -> {t.index[-1].date()}"
          f"   (dropped {int(d.index.year.isin(range(2006, 2027)).sum() - len(t))} rows in the 2006+ window "
          f"for missing vix3m/band, incl. the 2026-07-17..2026-09-09 gap)")
    t["termreg"] = np.where(t.term >= 1.0, "backwardation", "contango")
    print(f"  term regime counts: {t.termreg.value_counts().to_dict()}")

    for h in (5, 10):
        g = t[f"g{h}"].values
        v = ~np.isnan(g)
        base = float(g[v].mean())
        print(f"\n  --- D{h}.  same-sample unconditional baseline = {base*100:+.3f}%  (n={int(v.sum())})")
        print(head(f"mean D{h}"))
        # marginals first
        for b in ("below", "inside", "above"):
            print(cell_row(f"[marginal] band={b}", (t.band == b).values, g, base))
        for r in ("contango", "backwardation"):
            print(cell_row(f"[marginal] term {r}", (t.termreg == r).values, g, base))
        print("  " + "-" * 108)
        for r in ("contango", "backwardation"):
            for b in ("below", "inside", "above"):
                print(cell_row(f"band={b:<6} & {r}", ((t.band == b) & (t.termreg == r)).values, g, base))

    # controlled effects: does band add over term, and vice versa?
    sect("PART 1b — WHICH DOMINATES?  effect of one factor holding the other's group mean out")
    g = t["g5"].values
    v = ~np.isnan(g)

    def controlled(mask, groups):
        """Weighted mean of (cell mean - that group's own mean). Zero => the factor adds
        nothing once the other factor's group mean is removed."""
        mask = np.asarray(mask, dtype=bool) & v
        tot, wsum = 0.0, 0
        for gname in np.unique(groups):
            gi = (groups == gname) & v
            mi = mask & gi
            if mi.sum() < 10:
                continue
            tot += mi.sum() * (g[mi].mean() - g[gi].mean())
            wsum += mi.sum()
        return tot / wsum if wsum else np.nan

    termg = t.termreg.values
    bandg = t.band.values
    print(f"  {'effect':<58}{'n':>6}{'ctrl excess D5':>16}  {'rot p':>7}")
    for b in ("below", "inside", "above"):
        m = (t.band == b).values
        n = int((m & v).sum())
        if n < MIN_N:
            print(f"  {'band=' + b + ' | controlling for term regime':<58}{n:>6}   n<{MIN_N} INCONCLUSIVE")
            continue
        # rotation on the controlled statistic
        mm = np.asarray(m, dtype=bool)
        o = controlled(mm, termg)
        offs = RNG.integers(1, len(mm), size=N_ROT)
        null = np.array([controlled(np.roll(mm, off), termg) for off in offs])
        null = null[~np.isnan(null)]
        pv = float((np.abs(null - null.mean()) >= abs(o - null.mean())).mean())
        print(f"  {'band=' + b + ' | controlling for term regime':<58}{n:>6}{o*100:>+15.3f}%  p={pv:>5.3f} {stars(pv)}")
    for r in ("contango", "backwardation"):
        m = (t.termreg == r).values
        n = int((m & v).sum())
        mm = np.asarray(m, dtype=bool)
        o = controlled(mm, bandg)
        offs = RNG.integers(1, len(mm), size=N_ROT)
        null = np.array([controlled(np.roll(mm, off), bandg) for off in offs])
        null = null[~np.isnan(null)]
        pv = float((np.abs(null - null.mean()) >= abs(o - null.mean())).mean())
        print(f"  {'term=' + r + ' | controlling for band state':<58}{n:>6}{o*100:>+15.3f}%  p={pv:>5.3f} {stars(pv)}")

    print("\n  DISAGREEMENT cells (one factor bullish, the other not):")
    print(head("mean D5"))
    base = float(g[v].mean())
    for lbl, m in [("BULL band (above upper) but CALM term (contango)",
                    ((t.band == "above") & (t.termreg == "contango")).values),
                   ("CALM band (inside/below) but STRESS term (backwardation)",
                    ((t.band != "above") & (t.termreg == "backwardation")).values),
                   ("BOTH stressed (above upper & backwardation)",
                    ((t.band == "above") & (t.termreg == "backwardation")).values),
                   ("BOTH calm (below lower & contango)",
                    ((t.band == "below") & (t.termreg == "contango")).values)]:
        print(cell_row(lbl, m, g, base))

    # continuous term buckets, to check the term>=1 cut isn't the whole story
    print("\n  term structure as quintiles (is the >=1 cut arbitrary?):  D5")
    tq = pd.qcut(t.term, 5, labels=False)
    for q in range(5):
        m = (tq == q).values
        lo, hi = t.term[tq == q].min(), t.term[tq == q].max()
        print(cell_row(f"term Q{q+1} ({lo:.3f}-{hi:.3f})", m, g, base))

    print("\n  WITHIN-TERM-REGIME band effect (the head-to-head the scanner question needs):  D5")
    for reg in ("contango", "backwardation"):
        rm = (t.termreg == reg).values
        sub = rm & v
        subbase = float(g[sub].mean())
        print(f"    term={reg}: within-regime baseline {subbase*100:+.3f}% (n={int(sub.sum())})")
        gsub = np.where(rm, g, np.nan)
        for b in ("below", "inside", "above"):
            m = (t.band == b).values
            n = int((m & ~np.isnan(gsub)).sum())
            if n < MIN_N:
                print(f"      band={b:<6} n={n} INCONCLUSIVE")
                continue
            obs, p = rot_p(m, gsub, np.mean)
            print(f"      band={b:<6} n={n:<5} mean {obs*100:+.3f}%  "
                  f"excess-vs-regime {(obs-subbase)*100:+.3f}%  rot p={p:.3f} {stars(p)}")

    print("\n  WITHIN-BAND term effect (the mirror image):  D5")
    for b in ("inside", "above"):
        bm = (t.band == b).values
        sub = bm & v
        subbase = float(g[sub].mean())
        print(f"    band={b}: within-band baseline {subbase*100:+.3f}% (n={int(sub.sum())})")
        gsub = np.where(bm, g, np.nan)
        for reg in ("contango", "backwardation"):
            m = (t.termreg == reg).values
            n = int((m & ~np.isnan(gsub)).sum())
            if n < MIN_N:
                print(f"      term={reg:<14} n={n} INCONCLUSIVE")
                continue
            obs, p = rot_p(m, gsub, np.mean)
            print(f"      term={reg:<14} n={n:<5} mean {obs*100:+.3f}%  "
                  f"excess-vs-band {(obs-subbase)*100:+.3f}%  rot p={p:.3f} {stars(p)}")

    print("\n  SANITY: the same band marginals on the FULL 1990+ sample vs this 2006+ subsample")
    gf = d["g5"].values
    vf = ~np.isnan(gf)
    bf = float(gf[vf].mean())
    for b in ("below", "inside", "above"):
        mf = (d.band == b).values
        nf = int((mf & vf).sum())
        of_ = gf[mf & vf].mean()
        ms = (t.band == b).values
        ns = int((ms & v).sum())
        os_ = g[ms & v].mean() if ns else np.nan
        print(f"    band={b:<7} 1990+: n={nf:<5} excess {(of_-bf)*100:+.2f}%   "
              f"2006+(term sample): n={ns:<5} excess "
              f"{((os_-base)*100 if ns >= MIN_N else float('nan')):+.2f}%"
              f"{'  <- n<25, inconclusive' if ns < MIN_N else ''}")

    # ---------------------------------------------------------------- PART 2: VVIX
    sect("PART 2 — band state x VVIX tercile (point-in-time 1y pctile rank).  2007+ (rank needs 252d warmup)")
    w = d[d.vvix_pct1y.notna() & d.band.notna()].copy()
    print(f"  surviving sample n={len(w):,}  {w.index[0].date()} -> {w.index[-1].date()}")
    w["vt"] = pd.cut(w.vvix_pct1y, [-.001, 1/3, 2/3, 1.001], labels=["vvixLO", "vvixMID", "vvixHI"])
    print(f"  vvix tercile counts: {w.vt.value_counts().to_dict()}")
    for h in (5, 10):
        gg = w[f"g{h}"].values
        vv = ~np.isnan(gg)
        base = float(gg[vv].mean())
        print(f"\n  --- D{h}.  baseline {base*100:+.3f}%  (n={int(vv.sum())})")
        print(head(f"mean D{h}"))
        for b in ("below", "inside", "above"):
            print(cell_row(f"[marginal] band={b}", (w.band == b).values, gg, base))
        for vt in ("vvixLO", "vvixMID", "vvixHI"):
            print(cell_row(f"[marginal] {vt}", (w.vt == vt).values, gg, base))
        print("  " + "-" * 108)
        for b in ("below", "inside", "above"):
            for vt in ("vvixLO", "vvixMID", "vvixHI"):
                print(cell_row(f"band={b:<6} & {vt}", ((w.band == b) & (w.vt == vt)).values, gg, base))

    sect("PART 2b — 'FRAGILE CALM': VIX low but VVIX high.  VIX tercile (1y pctile) x VVIX tercile")
    w2 = w[w.vix_pct1y.notna()].copy()
    w2["xt"] = pd.cut(w2.vix_pct1y, [-.001, 1/3, 2/3, 1.001], labels=["vixLO", "vixMID", "vixHI"])
    for h in (5, 10):
        gg = w2[f"g{h}"].values
        vv = ~np.isnan(gg)
        base = float(gg[vv].mean())
        print(f"\n  --- D{h}.  baseline {base*100:+.3f}%  (n={int(vv.sum())})")
        print(head(f"mean D{h}"))
        for xt in ("vixLO", "vixMID", "vixHI"):
            for vt in ("vvixLO", "vvixMID", "vvixHI"):
                print(cell_row(f"{xt} & {vt}", ((w2.xt == xt) & (w2.vt == vt)).values, gg, base))
        a = ((w2.xt == "vixLO") & (w2.vt == "vvixHI")).values
        b_ = ((w2.xt == "vixLO") & (w2.vt == "vvixLO")).values
        na, nb = int((a & vv).sum()), int((b_ & vv).sum())
        if na >= MIN_N and nb >= MIN_N:
            diff = gg[a & vv].mean() - gg[b_ & vv].mean()
            # rotation on the DIFFERENCE: roll both masks by the same offset
            offs = RNG.integers(1, len(a), size=N_ROT)
            nulls = []
            for off in offs:
                ra, rb = np.roll(a, off) & vv, np.roll(b_, off) & vv
                if ra.sum() >= 10 and rb.sum() >= 10:
                    nulls.append(gg[ra].mean() - gg[rb].mean())
            nulls = np.array(nulls)
            pv = float((np.abs(nulls - nulls.mean()) >= abs(diff - nulls.mean())).mean())
            print(f"\n  FRAGILE-CALM TEST D{h}: (vixLO & vvixHI, n={na}) minus (vixLO & vvixLO, n={nb})"
                  f" = {diff*100:+.3f}%   rot p={pv:.3f} {stars(pv)}")

    # ---------------------------------------------------------------- PART 3: SPX MA20 regime
    sect("PART 3 — band state x SPX vs its 20d MA (the scanner's risk-off flag).  full history 1990+")
    r = d[d.band.notna() & d.spx_ma20.notna()].copy()
    print(f"  surviving sample n={len(r):,}  {r.index[0].date()} -> {r.index[-1].date()}")
    print(f"  regime counts: above MA20 {int(r.above_ma20.sum())} / below MA20 {int((~r.above_ma20).sum())}")
    for h in (5, 10):
        gg = r[f"g{h}"].values
        vv = ~np.isnan(gg)
        base = float(gg[vv].mean())
        print(f"\n  --- D{h}.  baseline {base*100:+.3f}%  (n={int(vv.sum())})")
        print(head(f"mean D{h}"))
        for b in ("below", "inside", "above"):
            print(cell_row(f"[marginal] band={b}", (r.band == b).values, gg, base))
        for nm, m in [("[marginal] SPX above MA20", r.above_ma20.values),
                      ("[marginal] SPX below MA20", (~r.above_ma20).values)]:
            print(cell_row(nm, m, gg, base))
        print("  " + "-" * 108)
        for reg, rm in [("SPX>MA20", r.above_ma20.values), ("SPX<MA20", (~r.above_ma20).values)]:
            for b in ("below", "inside", "above"):
                print(cell_row(f"band={b:<6} & {reg}", (r.band == b).values & rm, gg, base))

    print("\n  WITHIN-REGIME band effect (does the vol state add on top of the MA20 flag?):  D5")
    gg = r["g5"].values
    vv = ~np.isnan(gg)
    for reg, rm in [("SPX>MA20", r.above_ma20.values), ("SPX<MA20", (~r.above_ma20).values)]:
        sub = rm & vv
        subbase = float(gg[sub].mean())
        print(f"    regime {reg}: within-regime baseline {subbase*100:+.3f}% (n={int(sub.sum())})")
        gsub = np.where(rm, gg, np.nan)          # restrict outcome to the regime
        for b in ("below", "inside", "above"):
            m = (r.band == b).values
            n = int((m & ~np.isnan(gsub)).sum())
            if n < MIN_N:
                print(f"      band={b:<6} n={n} INCONCLUSIVE")
                continue
            obs, p = rot_p(m, gsub, np.mean)
            print(f"      band={b:<6} n={n:<5} mean {obs*100:+.3f}%  "
                  f"excess-vs-regime {(obs-subbase)*100:+.3f}%  rot p={p:.3f} {stars(p)}")

    # overlap check — does the band state merely duplicate the MA20 flag?
    ct = pd.crosstab(r.band, r.above_ma20, normalize="index") * 100
    print("\n  overlap: P(SPX above MA20 | band state)  — if the band merely restates the flag "
          "these would be 0/100")
    for b in ("below", "inside", "above"):
        print(f"    band={b:<7} above MA20 {ct.loc[b, True]:5.1f}%   below MA20 {ct.loc[b, False]:5.1f}%")

    # ---------------------------------------------------------------- PART 4: downside
    sect("PART 4 — DOWNSIDE ASYMMETRY per band state (full history, entry t+1 close)")
    a = d[d.band.notna()].copy()
    specs = [("mean g5", "g5", np.mean, True),
             ("5th pctile g5", "g5", q05, True),
             ("P(g5 < -2%)", "g5", p_lt2, True),
             ("mean MAE over next 10d", "mae10", np.mean, True),
             ("5th pctile MAE next 10d", "mae10", q05, True),
             ("mean MFE over next 10d", "mfe10", np.mean, True)]
    for nm, col, fn, pct in specs:
        vals = a[col].values
        vv = ~np.isnan(vals)
        base = float(fn(vals[vv]))
        print(f"\n  {nm}   baseline {base*100:+.3f}%  (n={int(vv.sum())})")
        print(head(nm))
        for b in ("below", "inside", "above"):
            print(cell_row(f"band={b}", (a.band == b).values, vals, base, statfn=fn))
        print(cell_row("BB(20,2.0) below lower (established)", a["bb20_2.0_below"].values, vals, base, statfn=fn))
        print(cell_row("BB(10,2.0) re-entry (established)", a["bb10_2.0_reentry"].values, vals, base, statfn=fn))

    # downside conditioned on regime too, since that's the practical question
    print("\n  --- downside, band x SPX-MA20 regime (P(g5<-2%) and 5th pctile g5)")
    rr = d[d.band.notna() & d.spx_ma20.notna()].copy()
    vals = rr["g5"].values
    vv = ~np.isnan(vals)
    for fn, nm in ((p_lt2, "P(g5<-2%)"), (q05, "5th pct g5")):
        base = float(fn(vals[vv]))
        print(f"\n    {nm}  baseline {base*100:+.2f}%")
        for reg, rm in [("SPX>MA20", rr.above_ma20.values), ("SPX<MA20", (~rr.above_ma20).values)]:
            for b in ("below", "inside", "above"):
                m = (rr.band == b).values & rm
                n = int((m & vv).sum())
                if n < MIN_N:
                    print(f"      {b:<6} & {reg}: n={n} INCONCLUSIVE")
                    continue
                obs, p = rot_p(m, vals, fn)
                print(f"      {b:<6} & {reg}: n={n:<5} {nm} {obs*100:+.2f}%  "
                      f"excess {(obs-base)*100:+.2f}%  rot p={p:.3f} {stars(p)}")

    # ---------------------------------------------------------------- PART 6: incremental info
    sect("PART 6 — INCREMENTAL INFORMATION: subset rotation + explicit interaction tests")
    print("  The full-index rotation used above widens the null when the outcome is regime-masked,")
    print("  so the crux numbers are re-tested by rotating the band mask WITHIN the regime subset")
    print("  (exact n preserved). Both p-values are shown; the honest answer is the weaker one.")

    def subset_rot(sub_df, mask_col, val_col, statfn=np.mean):
        """Rotate the band mask inside a regime subset (rows reindexed 0..k-1)."""
        vals = sub_df[val_col].values
        mask = sub_df[mask_col].values.astype(bool)
        valid = ~np.isnan(vals)
        n = int((mask & valid).sum())
        if n < MIN_N:
            return n, np.nan, np.nan, np.nan
        obs = float(statfn(vals[mask & valid]))
        sb = float(statfn(vals[valid]))
        offs = RNG.integers(1, len(mask), size=N_ROT)
        null = []
        for off in offs:
            m = np.roll(mask, off) & valid
            if m.sum() >= 10:
                null.append(statfn(vals[m]))
        null = np.array(null)
        p = float((np.abs(null - null.mean()) >= abs(obs - null.mean())).mean())
        return n, obs, obs - sb, p

    r = d[d.band.notna() & d.spx_ma20.notna()].copy()
    for bcol in ("is_above", "is_below", "is_inside"):
        r[bcol] = (r.band == bcol.split("_")[1]).values
    print(f"\n  {'regime / band':<34}{'n':>6}{'stat':>10}{'excess vs regime':>19}{'subset p':>11}")
    for stat_nm, col, fn, unit in (("mean g5", "g5", np.mean, "%"),
                                   ("P(g5<-2%)", "g5", p_lt2, "%"),
                                   ("5th pct g5", "g5", q05, "%"),
                                   ("mean MAE10", "mae10", np.mean, "%"),
                                   ("mean MFE10", "mfe10", np.mean, "%")):
        print(f"\n  --- {stat_nm}")
        for reg, sub in (("SPX>MA20", r[r.above_ma20]), ("SPX<MA20", r[~r.above_ma20])):
            vv = ~np.isnan(sub[col].values)
            print(f"  {reg + ' regime baseline':<34}{int(vv.sum()):>6}"
                  f"{fn(sub[col].values[vv])*100:>9.2f}{unit}")
            for b in ("below", "above"):
                n, obs, exc, p = subset_rot(sub, f"is_{b}", col, fn)
                if n < MIN_N:
                    print(f"    {'band=' + b:<32}{n:>6}   n<{MIN_N} INCONCLUSIVE")
                    continue
                print(f"    {'band=' + b:<32}{n:>6}{obs*100:>9.2f}{unit}"
                      f"{exc*100:>+18.2f}{unit}  p={p:>5.3f} {stars(p)}")

    print("\n  INTERACTION TESTS (difference of two cells; both masks rolled by the same offset):")
    gg = r["g5"].values
    vv = ~np.isnan(gg)

    def diff_test(m1, m2, vals, valid):
        n1, n2 = int((m1 & valid).sum()), int((m2 & valid).sum())
        if n1 < MIN_N or n2 < MIN_N:
            return n1, n2, np.nan, np.nan
        obs = vals[m1 & valid].mean() - vals[m2 & valid].mean()
        offs = RNG.integers(1, len(m1), size=N_ROT)
        null = []
        for off in offs:
            a, b = np.roll(m1, off) & valid, np.roll(m2, off) & valid
            if a.sum() >= 10 and b.sum() >= 10:
                null.append(vals[a].mean() - vals[b].mean())
        null = np.array(null)
        return n1, n2, obs, float((np.abs(null - null.mean()) >= abs(obs - null.mean())).mean())

    pairs = [("band=above & SPX<MA20  MINUS  band=above & SPX>MA20",
              ((r.band == "above") & ~r.above_ma20).values, ((r.band == "above") & r.above_ma20).values),
             ("band=above & SPX<MA20  MINUS  band=inside & SPX<MA20",
              ((r.band == "above") & ~r.above_ma20).values, ((r.band == "inside") & ~r.above_ma20).values),
             ("band=below & SPX>MA20  MINUS  band=inside & SPX>MA20",
              ((r.band == "below") & r.above_ma20).values, ((r.band == "inside") & r.above_ma20).values),
             ("SPX<MA20 (any band)    MINUS  SPX>MA20 (any band)",
              (~r.above_ma20).values, r.above_ma20.values)]
    for lbl, m1, m2 in pairs:
        n1, n2, o, p = diff_test(m1, m2, gg, vv)
        if np.isnan(o):
            print(f"    {lbl:<58} n={n1}/{n2} INCONCLUSIVE")
            continue
        print(f"    {lbl:<58} n={n1}/{n2}  D5 diff {o*100:+.3f}%  p={p:.3f} {stars(p)}")

    print("\n  additivity check (is the joint cell just the sum of the two marginals?):")
    base = float(gg[vv].mean())
    ea = gg[(r.band == "above").values & vv].mean() - base
    eb = gg[(~r.above_ma20).values & vv].mean() - base
    ej = gg[((r.band == "above") & ~r.above_ma20).values & vv].mean() - base
    print(f"    band=above marginal {ea*100:+.3f}%  +  SPX<MA20 marginal {eb*100:+.3f}%  "
          f"=  {(ea+eb)*100:+.3f}%   vs observed joint {ej*100:+.3f}%   "
          f"(interaction {(ej-ea-eb)*100:+.3f}%)")

    # ---------------------------------------------------------------- era check on the winners
    sect("PART 5 — era stability of the cells that survived (D5 excess within era)")
    eras = [("1990s", 1990, 1999), ("2000s", 2000, 2009), ("2010s", 2010, 2019), ("2020s", 2020, 2099)]
    gg = d["g5"].values
    vv = ~np.isnan(gg)
    yrs = d.index.year.values
    for lbl, m in [("band=above (BB10,2 upper)", (d.band == "above").values),
                   ("band=below (BB10,2 lower)", (d.band == "below").values),
                   ("SPX<MA20", (d.spx < d.spx_ma20).values),
                   ("band=above & SPX<MA20", ((d.band == "above") & (d.spx < d.spx_ma20)).values),
                   ("band=below & SPX>MA20", ((d.band == "below") & (d.spx > d.spx_ma20)).values)]:
        parts = []
        for en, y0, y1 in eras:
            sub = (yrs >= y0) & (yrs <= y1) & vv
            mm = m & sub
            if mm.sum() < 15:
                parts.append(f"{en} n={int(mm.sum())} -")
                continue
            parts.append(f"{en} n={int(mm.sum())} {(gg[mm].mean()-gg[sub].mean())*100:+.2f}%")
        print(f"  {lbl:<30} " + " | ".join(parts))

    # current reading
    last = d.iloc[-1]
    sect(f"CURRENT READING ({d.index[-1].date()})")
    print(f"  VIX {last.vix:.2f}  band={last.band}  BB(10,2) {last['bb10_2.0_lo']:.2f}..{last['bb10_2.0_up']:.2f}")
    lastterm = d.term.dropna()
    print(f"  term (vix/vix3m) last available {lastterm.index[-1].date()}: {lastterm.iloc[-1]:.3f} "
          f"({'backwardation' if lastterm.iloc[-1] >= 1 else 'contango'})  [STALE: {int((d.index > lastterm.index[-1]).sum())} sessions old]")
    print(f"  VVIX {last.vvix:.2f}  1y pctile "
          f"{last.vvix_pct1y*100:.0f}%" if pd.notna(last.vvix_pct1y) else "  VVIX 1y pctile n/a")
    print(f"  VIX 1y pctile {last.vix_pct1y*100:.0f}%")
    if pd.notna(last.spx_ma20):
        print(f"  SPX {last.spx:.2f} vs MA20 {last.spx_ma20:.2f} -> "
              f"{'ABOVE' if last.spx > last.spx_ma20 else 'BELOW'}")
    else:
        print(f"  SPX {last.spx:.2f}  MA20 unavailable")


if __name__ == "__main__":
    main()
