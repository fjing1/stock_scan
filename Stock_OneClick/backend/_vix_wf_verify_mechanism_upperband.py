"""
_vix_wf_verify_mechanism_upperband.py — ADVERSARIAL verification of the claim:

  "The UPPER band IS redundant: bb10_2.0_above's raw D5 excess +0.30% (n=518, p=.017) falls to
   +0.18% (p=.154) under vix_pct1y FE, +0.08% (p=.528) under SPX-5d-return FE, and +0.04%
   (p=.761, 14% of raw) under the full control set — it is a repackaging of 'VIX is high and
   the index just fell'."

Lens: MECHANISM / CONFOUNDING. Is a simpler variable really doing the work, or is the control
regression simply destroying the estimator's power?

The claim is built entirely on p-values crossing .05 inside a fixed-effects regression. Three
things can produce that pattern and only ONE of them is "redundancy":
  (1) genuine subsumption — the simpler variable carries the same signal and more;
  (2) collinearity/over-control — the FE absorbs most of the dummy's variance, the standard
      error explodes, and the point estimate becomes uninformative (absence of evidence);
  (3) estimand mismatch — FE-OLS variance-weights bins, the tradeable estimand is the
      treated-weighted (ATT) within-bin difference. They can differ a lot.

So this script does NOT re-run the same test. It runs the tests that DISTINGUISH those cases:

  A. Absorption + power diagnostics: how much of the dummy survives each control, HAC (Newey-West)
     standard errors and 95% CIs on the Frisch-Waugh coefficient, and the minimum detectable
     effect (MDE) of each specification.
  B. INJECTION POWER TEST: implant a known +0.30% effect into the treated days and ask whether
     the "full control set" specification can detect it. If it cannot, p=.761 means nothing.
  C. ATT / matched estimator (treated-count weighted) vs FE-OLS (variance weighted), same bins.
  D. The head-to-head the claim never runs: frequency-matched simple signals. If the band is a
     repackaging of "VIX high + index just fell", a signal built literally out of those two
     should deliver AT LEAST as much D5 excess on the same number of days.
  E. Disjoint decomposition: band-only, simple-only, both, neither.
  F. Symmetry / reverse Frisch-Waugh: control for the BAND and see whether the simpler variable
     survives. If both die under each other, it is collinearity, not subsumption.
  G. Other confounders the lens names: VRP, realized vol, calendar (month / dow / year FE).
  H. Episode clustering and leave-one-year-out stability of the matched estimate.

Run: ../../vcp_env/bin/python _vix_wf_verify_mechanism_upperband.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _vix_data

RNG = np.random.default_rng(20260910)
N_ROT = 5000
SIG = "bb10_2.0_above"


# ------------------------------------------------------------------ helpers
def stars(p):
    if np.isnan(p):
        return "   "
    return "***" if p < 0.01 else ("** " if p < 0.05 else ("*  " if p < 0.10 else "   "))


def fe_matrix(n, fe_list, extra=None):
    cols = [np.ones(n)]
    for codes in fe_list:
        codes = np.asarray(codes)
        for k in np.unique(codes)[1:]:
            cols.append((codes == k).astype(float))
    if extra is not None:
        cols.extend([np.asarray(c, float) for c in extra])
    return np.column_stack(cols)


def fwl_full(y, X, mask):
    """Frisch-Waugh coefficient on `mask` after partialling out X, plus HAC inference.

    Returns dict with coef, HAC se (Newey-West, Bartlett lag L), t, p_normal, 95% CI,
    absorption R^2 of the dummy on X, and the effective (residual) variance retained."""
    Q, _ = np.linalg.qr(X)
    y_res = y - Q @ (Q.T @ y)
    dv = mask.astype(float)
    dr = dv - Q @ (Q.T @ dv)
    den = float(dr @ dr)
    coef = float(dr @ y_res) / den
    # absorption: how much of the dummy's own variance the controls ate
    dv_c = dv - dv.mean()
    r2_absorb = 1.0 - den / float(dv_c @ dv_c)
    # HAC (Newey-West) se on the FWL slope, lag chosen for 5-day overlapping windows
    e = y_res - coef * dr
    u = dr * e
    L = 10
    s = float(u @ u)
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        s += 2.0 * w * float(u[l:] @ u[:-l])
    var = s / (den ** 2)
    se = float(np.sqrt(max(var, 0.0)))
    t = coef / se if se > 0 else np.nan
    from math import erfc, sqrt
    p_norm = erfc(abs(t) / sqrt(2.0)) if se > 0 else np.nan
    return {"coef": coef, "se": se, "t": t, "p": p_norm,
            "lo": coef - 1.96 * se, "hi": coef + 1.96 * se,
            "r2_absorb": r2_absorb, "den": den, "mde": 1.96 * se,
            "Q": Q, "y_res": y_res}


def rotation_p_fwl(y, X, mask, coef=None):
    Q, _ = np.linalg.qr(X)
    y_res = y - Q @ (Q.T @ y)

    def c(m):
        dr = m - Q @ (Q.T @ m)
        d = float(dr @ dr)
        return float(dr @ y_res) / d if d > 1e-12 else np.nan

    dv = mask.astype(float)
    obs = c(dv) if coef is None else coef
    null = np.array([c(np.roll(dv, off)) for off in RNG.integers(1, len(dv), size=N_ROT)])
    null = null[~np.isnan(null)]
    base = null.mean()
    return float((np.abs(null - base) >= abs(obs - base)).mean()), null


def rotation_p_mean(mask, fwd):
    """plain conditional-mean rotation test (the repo's rotation_pvalue)."""
    valid = ~np.isnan(fwd)
    obs = fwd[mask & valid].mean()
    null = np.empty(N_ROT)
    for i, off in enumerate(RNG.integers(1, len(mask), size=N_ROT)):
        m = np.roll(mask, off) & valid
        null[i] = fwd[m].mean() if m.sum() else np.nan
    null = null[~np.isnan(null)]
    base = np.nanmean(null)
    return float((np.abs(null - base) >= abs(obs - base)).mean())


def matched_diff(mask, bins, y):
    """ATT: treated-count-weighted within-bin (treated - control) difference.
    Returns (diff, n_treated_used, n_bins_used)."""
    num = den = 0.0
    nb = 0
    for b in np.unique(bins):
        s = bins == b
        t, c = s & mask, s & ~mask
        if t.sum() == 0 or c.sum() == 0:
            continue
        num += t.sum() * (y[t].mean() - y[c].mean())
        den += t.sum()
        nb += 1
    return (num / den if den else np.nan), int(den), nb


def rotation_p_matched(mask, bins, y, obs):
    null = np.empty(N_ROT)
    for i, off in enumerate(RNG.integers(1, len(mask), size=N_ROT)):
        null[i] = matched_diff(np.roll(mask, off), bins, y)[0]
    null = null[~np.isnan(null)]
    base = null.mean()
    return float((np.abs(null - base) >= abs(obs - base)).mean())


def runs(mask):
    """number of contiguous episodes in a boolean mask."""
    m = mask.astype(int)
    return int(((m == 1) & (np.r_[0, m[:-1]] == 0)).sum())


def excess(mask, y):
    return y[mask].mean() - y.mean()


# ------------------------------------------------------------------ main
def main():
    d = _vix_data.add_features(_vix_data.load())
    d["rv20"] = d.spx.pct_change().rolling(20).std() * np.sqrt(252)
    d["spx_r20"] = d.spx / d.spx.shift(20) - 1.0
    d["spx_r5"] = d.spx / d.spx.shift(5) - 1.0
    d["spx_r1"] = d.spx.pct_change()
    d["spx_r10"] = d.spx / d.spx.shift(10) - 1.0
    d["vrp"] = d.vix / 100.0 - d.rv20

    keep = ["g5", "vix_pct1y", "bb10_width", "rv20", "spx_r20", "spx_r5", "spx_r1", "spx_r10",
            "vrp", "stretch", "bb10_pctb", SIG, "bb10_2.0_below"]
    a = d[keep].dropna().copy()
    y = a.g5.values
    n = len(a)
    above = a[SIG].astype(bool).values
    base = y.mean()

    print("=" * 100)
    print("SETUP — identical frame to the analysis under test")
    print("=" * 100)
    print(f"  rows {n:,}   {a.index[0].date()} -> {a.index[-1].date()}   unconditional g5 base "
          f"{base*100:+.3f}%")
    print(f"  {SIG}: n={above.sum()}  ({above.sum()/n*100:.1f}% of days) in "
          f"{runs(above)} contiguous episodes")
    print(f"  raw D5 excess {excess(above, y)*100:+.3f}%   rotation p="
          f"{rotation_p_mean(above, y):.3f}")

    q = lambda s, k: pd.qcut(a[s], k, labels=False, duplicates="drop").values
    lvl20 = q("vix_pct1y", 20)
    lvl10 = q("vix_pct1y", 10)
    wid10 = q("bb10_width", 10)
    rv10 = q("rv20", 10)
    r20d = q("spx_r20", 10)
    r5d = q("spx_r5", 10)
    r5v = q("spx_r5", 20)
    FULL = [lvl20, wid10, rv10, r20d, r5d]

    SPECS = [
        ("no control", []),
        ("vix_pct1y vigintile FE (level)", [lvl20]),
        ("SPX 5d-return decile FE", [r5d]),
        ("level + SPX 5d FE", [lvl20, r5d]),
        ("EVERYTHING (level+width+rv+r20+r5) FE", FULL),
    ]

    # ---------------------------------------------------------------- A. power
    print()
    print("=" * 100)
    print("A. IS THE CONTROLLED ESTIMATE INFORMATIVE?  absorption, HAC standard errors, MDE")
    print("=" * 100)
    print(f"  {'specification':<42}{'coef':>9}{'HAC se':>9}{'95% CI':>20}"
          f"{'absorbed':>10}{'MDE':>9}{'rot p':>8}")
    print("  " + "-" * 96)
    store = {}
    for lbl, fes in SPECS:
        X = fe_matrix(n, fes)
        r = fwl_full(y, X, above)
        rp, _ = rotation_p_fwl(y, X, above, r["coef"])
        store[lbl] = (r, rp)
        print(f"  {lbl:<42}{r['coef']*100:>+8.3f}%{r['se']*100:>8.3f}%"
              f"  [{r['lo']*100:+.3f}%,{r['hi']*100:+.3f}%]"
              f"{r['r2_absorb']*100:>9.1f}%{r['mde']*100:>8.3f}%{rp:>8.3f}")
    raw = store["no control"][0]["coef"]
    print()
    print(f"  Does each controlled 95% CI still contain the RAW estimate {raw*100:+.3f}% ?")
    for lbl, _ in SPECS[1:]:
        r = store[lbl][0]
        inside = r["lo"] <= raw <= r["hi"]
        print(f"    {lbl:<42}{'YES — cannot rule out an UNCHANGED effect' if inside else 'no — genuinely shrunk'}")
    print()
    print("  'absorbed' = share of the dummy's own variance eaten by the controls. MDE = the")
    print("  smallest true effect this specification could detect at 95% (1.96 x HAC se).")

    # ---------------------------------------------------------------- B. injection
    print()
    print("=" * 100)
    print("B. INJECTION POWER TEST — implant a KNOWN effect in the treated days, can the")
    print("   'EVERYTHING' specification find it?  (if not, p=.761 is absence of evidence)")
    print("=" * 100)
    Xfull = fe_matrix(n, FULL)
    Xnone = fe_matrix(n, [])
    print(f"  {'implanted true effect':<26}{'spec':<26}{'recovered':>11}{'rot p':>8}"
          f"{'HAC p':>8}   detected?")
    print("  " + "-" * 90)
    for inj in (0.0, 0.0030, 0.0050, 0.0100):
        y_i = y + inj * above.astype(float)
        for slbl, X in (("no control", Xnone), ("EVERYTHING FE", Xfull)):
            r = fwl_full(y_i, X, above)
            rp, _ = rotation_p_fwl(y_i, X, above, r["coef"])
            det = "YES" if rp < 0.05 else "NO  <-- blind"
            print(f"  {inj*100:>+8.2f}%{'':<18}{slbl:<26}{r['coef']*100:>+10.3f}%"
                  f"{rp:>8.3f}{r['p']:>8.3f}   {det}")

    # ---------------------------------------------------------------- C. ATT vs FE-OLS
    print()
    print("=" * 100)
    print("C. ESTIMAND CHECK — treated-weighted matched difference (ATT, the tradeable number)")
    print("   vs the variance-weighted FE-OLS coefficient, on the SAME bins")
    print("=" * 100)
    print(f"  {'bins':<44}{'FE-OLS':>10}{'ATT matched':>13}{'n treated':>11}{'bins used':>11}{'rot p':>8}")
    print("  " + "-" * 96)
    binsets = [("vix_pct1y decile (10)", lvl10),
               ("vix_pct1y vigintile (20)", lvl20),
               ("SPX 5d-return decile (10)", r5d),
               ("SPX 5d-return vigintile (20)", r5v),
               ("level tercile x SPX5d tercile (9 cells)",
                q("vix_pct1y", 3) * 3 + q("spx_r5", 3)),
               ("level quintile x SPX5d quintile (25 cells)",
                q("vix_pct1y", 5) * 5 + q("spx_r5", 5))]
    for lbl, b in binsets:
        fe = fwl_full(y, fe_matrix(n, [b]), above)["coef"]
        md, nt, nb = matched_diff(above, b, y)
        pm = rotation_p_matched(above, b, y, md)
        print(f"  {lbl:<44}{fe*100:>+9.3f}%{md*100:>+12.3f}%{nt:>11}{nb:>11}{pm:>8.3f}{stars(pm)}")

    # ---------------------------------------------------------------- D. head-to-head
    print()
    print("=" * 100)
    print("D. THE HEAD-TO-HEAD THE CLAIM NEVER RUNS — frequency-matched simple signals")
    print("   If the band is a repackaging of 'VIX high + index just fell', a signal made")
    print("   literally of those two should earn AT LEAST as much on the same number of days.")
    print("=" * 100)
    k = int(above.sum())
    r5v_ = a.spx_r5.values
    lv_ = a.vix_pct1y.values
    stretch_ = a.stretch.values
    thr_r5 = np.sort(r5v_)[k - 1]                       # k worst 5d returns
    thr_lv = np.sort(lv_)[-k]                           # k highest VIX percentiles
    simple_fall = r5v_ <= thr_r5
    simple_hi = lv_ >= thr_lv
    # "VIX high AND index just fell", tuned to hit ~k days by walking a joint quantile
    best = None
    for pl in np.arange(0.50, 0.96, 0.005):
        for pr in np.arange(0.02, 0.51, 0.005):
            m = (lv_ >= np.quantile(lv_, pl)) & (r5v_ <= np.quantile(r5v_, pr))
            if abs(m.sum() - k) < abs((best[0].sum() if best else 10 ** 9) - k):
                best = (m, pl, pr)
    joint, pl, pr = best
    cands = [(f"{SIG} (the band)", above),
             (f"SPX 5d return in worst {k} days", simple_fall),
             (f"vix_pct1y in top {k} days", simple_hi),
             (f"VIX pct1y>={pl:.2f} AND SPX5d<=q{pr:.2f}  [the claimed mechanism]", joint),
             ("stretch >= +10% (VIX vs MA10)", stretch_ >= 0.10)]
    print(f"  {'signal':<58}{'n':>6}{'D5 excess':>12}{'win%':>8}{'rot p':>8}")
    print("  " + "-" * 92)
    for lbl, m in cands:
        if m.sum() < 25:
            print(f"  {lbl:<58}{int(m.sum()):>6}  inconclusive (n<25)")
            continue
        print(f"  {lbl:<58}{int(m.sum()):>6}{excess(m, y)*100:>+11.3f}%"
              f"{(y[m] > 0).mean()*100:>7.1f}%{rotation_p_mean(m, y):>8.3f}"
              f"{stars(rotation_p_mean(m, y))}")
    print(f"  {'ALL DAYS (baseline)':<58}{n:>6}{0.0:>+11.3f}%{(y > 0).mean()*100:>7.1f}%")

    # ---------------------------------------------------------------- E. disjoint cells
    print()
    print("=" * 100)
    print("E. DISJOINT DECOMPOSITION — where does the band's return actually live?")
    print("=" * 100)
    for oname, other in (("SPX-5d-worst-k", simple_fall), ("VIX-pct1y-top-k", simple_hi),
                         ("claimed-mechanism joint", joint)):
        print(f"\n  vs {oname}:")
        print(f"    {'cell':<40}{'n':>6}{'D5 mean':>11}{'excess':>10}{'rot p':>8}")
        for lbl, m in ((f"band ONLY (not {oname})", above & ~other),
                       (f"{oname} ONLY (not band)", other & ~above),
                       ("BOTH", above & other),
                       ("neither", ~above & ~other)):
            if m.sum() < 25:
                print(f"    {lbl:<40}{int(m.sum()):>6}   inconclusive (n<25)")
                continue
            p = rotation_p_mean(m, y)
            print(f"    {lbl:<40}{int(m.sum()):>6}{y[m].mean()*100:>10.3f}%"
                  f"{excess(m, y)*100:>+9.3f}%{p:>8.3f}{stars(p)}")

    # ---------------------------------------------------------------- F. symmetry
    print()
    print("=" * 100)
    print("F. SYMMETRY / REVERSE FRISCH-WAUGH — does the 'simpler' variable survive when the")
    print("   BAND is the control?  If both die under each other it is collinearity, not")
    print("   subsumption, and 'the band is redundant' is an arbitrary choice of victim.")
    print("=" * 100)
    bandbins = pd.qcut(a.bb10_pctb.rank(method="first"), 10, labels=False).values
    tests = [("BAND dummy, control = SPX 5d decile FE", above, [r5d]),
             ("SPX-5d-worst-k dummy, control = band dummy", simple_fall, [above.astype(int)]),
             ("SPX-5d-worst-k dummy, control = bb10_pctb decile FE", simple_fall, [bandbins]),
             ("BAND dummy, control = bb10_pctb decile FE (placebo: same info)", above, [bandbins]),
             ("SPX-5d-worst-k dummy, no control", simple_fall, []),
             ("BAND dummy, control = SPX 5d decile FE + band-only reshuffle", above, [r5d])]
    print(f"  {'test':<64}{'coef':>10}{'HAC se':>9}{'rot p':>8}{'absorbed':>10}")
    print("  " + "-" * 96)
    for lbl, m, fes in tests[:5]:
        X = fe_matrix(n, fes)
        r = fwl_full(y, X, m)
        rp, _ = rotation_p_fwl(y, X, m, r["coef"])
        print(f"  {lbl:<64}{r['coef']*100:>+9.3f}%{r['se']*100:>8.3f}%{rp:>8.3f}"
              f"{r['r2_absorb']*100:>9.1f}%")

    # ---------------------------------------------------------------- G. other confounders
    print()
    print("=" * 100)
    print("G. OTHER CANDIDATE CONFOUNDERS THE LENS NAMES")
    print("=" * 100)
    mon = a.index.month.values - 1
    dow = a.index.dayofweek.values
    yr = a.index.year.values
    more = [("VRP (vix/100 - rv20) decile FE", [q("vrp", 10)]),
            ("realized vol 20d decile FE", [rv10]),
            ("calendar: month FE", [mon]),
            ("calendar: day-of-week FE", [dow]),
            ("calendar: YEAR FE (37 dummies)", [yr]),
            ("calendar: month + dow + year FE", [mon, dow, yr]),
            ("SPX 1d return decile FE", [q("spx_r1", 10)]),
            ("SPX 10d return decile FE", [q("spx_r10", 10)]),
            ("SPX 5d return, LINEAR + QUADRATIC (not FE)",
             None)]
    print(f"  {'control':<48}{'coef':>10}{'HAC se':>9}{'95% CI':>21}{'rot p':>8}{'absorbed':>10}")
    print("  " + "-" * 96)
    for lbl, fes in more:
        if fes is None:
            X = fe_matrix(n, [], extra=[a.spx_r5.values, a.spx_r5.values ** 2])
        else:
            X = fe_matrix(n, fes)
        r = fwl_full(y, X, above)
        rp, _ = rotation_p_fwl(y, X, above, r["coef"])
        print(f"  {lbl:<48}{r['coef']*100:>+9.3f}%{r['se']*100:>8.3f}%"
              f"  [{r['lo']*100:+.3f}%,{r['hi']*100:+.3f}%]{rp:>8.3f}{r['r2_absorb']*100:>9.1f}%")

    # ---------------------------------------------------------------- H. stability
    print()
    print("=" * 100)
    print("H. WHERE THE RAW EFFECT LIVES — leave-one-year-out on the raw and the")
    print("   level-matched (vix_pct1y vigintile ATT) estimates")
    print("=" * 100)
    md_all, _, _ = matched_diff(above, lvl20, y)
    rows = []
    for Y in sorted(set(yr)):
        s = yr != Y
        if (above & ~s).sum() == 0:
            continue
        rows.append((Y, int((above & ~s).sum()),
                     excess(above[s], y[s]),
                     matched_diff(above[s], lvl20[s], y[s])[0]))
    rows.sort(key=lambda r: r[3])
    print(f"  full sample: raw {excess(above, y)*100:+.3f}%   level-matched ATT {md_all*100:+.3f}%")
    print(f"  5 years whose removal HURTS the level-matched estimate most (i.e. carry the effect):")
    for Y, nn, rw, mdd in rows[:5]:
        print(f"    drop {Y} (n={nn:>3} band days): raw -> {rw*100:+.3f}%   matched -> {mdd*100:+.3f}%")
    print(f"  5 years whose removal HELPS most:")
    for Y, nn, rw, mdd in rows[-5:]:
        print(f"    drop {Y} (n={nn:>3} band days): raw -> {rw*100:+.3f}%   matched -> {mdd*100:+.3f}%")
    lo = min(r[3] for r in rows)
    hi = max(r[3] for r in rows)
    print(f"  leave-one-year-out range of the level-matched ATT: {lo*100:+.3f}% .. {hi*100:+.3f}%")
    print(f"  years with >=1 band day: {len(rows)} of {len(set(yr))}")


if __name__ == "__main__":
    main()
