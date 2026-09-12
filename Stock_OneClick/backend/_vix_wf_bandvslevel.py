"""
_vix_wf_bandvslevel.py — CONTROL STUDY: do Bollinger Bands on VIX add information beyond the
plain VIX LEVEL (vix_pct1y)?

The suspicion: "VIX below its lower band" may be nothing more than "VIX is low", which needs no
bands. Five tests:
  1. plain-level baseline curve: D5 (g5) excess by vix_pct1y decile
  2. double sort: bb10_2.0_below true vs false WITHIN each vix_pct1y tercile
  3. double sort: bb10_2.0_above true vs false WITHIN each tercile
  4. horse race regressions: g5 ~ pct1y   vs   g5 ~ pct1y + below + above
  5. extra controls: vix_z1y, 10-day VIX change (momentum), and raw stretch

All significance via circular rotation test (reused approach from _vix_ma10_bb_research.py).
Regression coefficient p-values: rotate the dummy's mask, refit via Frisch-Waugh, compare.

Run: ../../vcp_env/bin/python _vix_wf_bandvslevel.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _vix_data

RNG = np.random.default_rng(20260910)
N_ROT = 5000


# ------------------------------------------------------------------ helpers
def stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "   "
    return "***" if p < 0.01 else ("** " if p < 0.05 else ("*  " if p < 0.10 else "   "))


def rotation_pvalue(mask: np.ndarray, fwd: np.ndarray, observed: float,
                    restrict: np.ndarray | None = None) -> float:
    """Circular rotation test (same approach as _vix_ma10_bb_research.rotation_pvalue).
    `restrict`: optional fixed subsample (e.g. a vix_pct1y tercile) — the rotated mask is
    intersected with it, so the null answers 'could a signal with this shape land inside this
    tercile and look this good by luck?'"""
    n = len(mask)
    valid = ~np.isnan(fwd)
    if restrict is not None:
        valid = valid & restrict
    null = np.empty(N_ROT)
    ns = np.empty(N_ROT)
    for i, off in enumerate(RNG.integers(1, n, size=N_ROT)):
        m = np.roll(mask, off) & valid
        k = m.sum()
        ns[i] = k
        null[i] = fwd[m].mean() if k >= 5 else np.nan
    null_ok = null[~np.isnan(null)]
    if not len(null_ok):
        return float("nan")
    base = np.nanmean(null_ok)
    return float((np.abs(null_ok - base) >= abs(observed - base)).mean())


def ols(y: np.ndarray, X: np.ndarray):
    """X already includes an intercept column. Returns coefs, R^2."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return beta, 1.0 - ss_res / ss_tot


def fwl_coef(y_res: np.ndarray, dvec: np.ndarray, Q: np.ndarray) -> float:
    """Frisch-Waugh: coefficient on `dvec` in a regression of y on [controls, dvec],
    given y already residualised on the controls and Q = orthonormal basis of the controls."""
    d_res = dvec - Q @ (Q.T @ dvec)
    den = float(d_res @ d_res)
    return float(d_res @ y_res) / den if den > 1e-12 else np.nan


def dummy_rotation_p(y: np.ndarray, ctrl: np.ndarray, mask: np.ndarray, observed: float) -> float:
    """p-value for a dummy's regression coefficient: rotate the dummy circularly, refit
    (holding all controls fixed), and see how often |coef| is at least as large."""
    Q, _ = np.linalg.qr(ctrl)
    y_res = y - Q @ (Q.T @ y)
    null = np.empty(N_ROT)
    dm = mask.astype(float)
    n = len(dm)
    for i, off in enumerate(RNG.integers(1, n, size=N_ROT)):
        null[i] = fwl_coef(y_res, np.roll(dm, off), Q)
    null = null[~np.isnan(null)]
    base = null.mean()
    return float((np.abs(null - base) >= abs(observed - base)).mean())


# ------------------------------------------------------------------ main
def main():
    d = _vix_data.add_features(_vix_data.load())
    print(f"panel: {len(d):,} rows  {d.index[0].date()} → {d.index[-1].date()}")

    BELOW, ABOVE = "bb10_2.0_below", "bb10_2.0_above"
    d["vix_chg10"] = d.vix / d.vix.shift(10) - 1.0

    need = ["g5", "vix_pct1y", "vix_z1y", "vix_chg10", "stretch", BELOW, ABOVE,
            "bb10_pctb", "bb10_2.0_reentry"]
    a = d.loc[:, need].copy()
    a["below"] = d[BELOW].astype(bool)
    a["above"] = d[ABOVE].astype(bool)
    a["reentry"] = d["bb10_2.0_reentry"].astype(bool)
    a = a.dropna(subset=["g5", "vix_pct1y", "vix_z1y", "vix_chg10", "stretch", "bb10_pctb"])
    pos = d.index.get_indexer(a.index)
    ngaps = int((np.diff(pos) > 1).sum())
    print(f"analysis frame: {len(a):,} rows  {a.index[0].date()} → {a.index[-1].date()}  "
          f"(internal gaps in the calendar: {ngaps} — rotation is effectively circular)")

    g5 = a.g5.values
    base = g5.mean()
    lvl = a.vix_pct1y.values
    below = a.below.values
    above = a.above.values
    reentry = a.reentry.values
    print(f"unconditional D5 (g5) mean over this frame: {base*100:+.3f}%   "
          f"win {(g5>0).mean()*100:.1f}%   n={len(g5):,}")
    print(f"below-band days n={below.sum():,} ({below.mean()*100:.1f}%)   "
          f"above-band days n={above.sum():,} ({above.mean()*100:.1f}%)   "
          f"re-entry days n={reentry.sum():,}")
    print(f"corr(below, vix_pct1y) = {np.corrcoef(below.astype(float), lvl)[0,1]:+.3f}   "
          f"corr(above, vix_pct1y) = {np.corrcoef(above.astype(float), lvl)[0,1]:+.3f}")

    # ---------------------------------------------------------- 1. level decile curve
    print("\n" + "=" * 100)
    print("1. PLAIN-LEVEL BASELINE CURVE — D5 (g5) excess by vix_pct1y decile")
    print("=" * 100)
    dec = pd.qcut(a.vix_pct1y, 10, labels=False, duplicates="drop").values
    print(f"{'decile':<10}{'pct1y range':<22}{'n':>7}{'mean D5':>10}{'excess':>10}{'p':>8}   "
          f"{'win%':>7}{'#below':>8}{'#above':>8}")
    print("-" * 100)
    for k in range(int(dec.max()) + 1):
        m = dec == k
        lo, hi = lvl[m].min(), lvl[m].max()
        cond = g5[m].mean()
        p = rotation_pvalue(m, g5, cond)
        print(f"D{k+1:<9}{f'{lo:.3f}–{hi:.3f}':<22}{m.sum():>7}{cond*100:>9.2f}%"
              f"{(cond-base)*100:>9.2f}%{p:>8.3f}{stars(p)}{(g5[m]>0).mean()*100:>6.1f}%"
              f"{int((below&m).sum()):>8}{int((above&m).sum()):>8}")
    print(f"{'ALL':<10}{'':<22}{len(g5):>7}{base*100:>9.2f}%{0.0:>9.2f}%{'':>8}   "
          f"{(g5>0).mean()*100:>6.1f}%{int(below.sum()):>8}{int(above.sum()):>8}")

    # ---------------------------------------------------------- 2/3. double sorts
    terc = pd.qcut(a.vix_pct1y, 3, labels=["low", "mid", "high"], duplicates="drop").values
    for signame, sig in (("bb10_2.0_below (VIX under LOWER band)", below),
                         ("bb10_2.0_above (VIX over UPPER band)", above),
                         ("bb10_2.0_reentry (back inside from above)", reentry)):
        print("\n" + "=" * 100)
        n_sec = 2 if "below" in signame else (3 if "above" in signame else "3b")
        print(f"{n_sec}. DOUBLE SORT — {signame}  vs  vix_pct1y tercile   (D5 = g5)")
        print("=" * 100)
        print(f"{'tercile':<10}{'pct1y':<16}{'n_sig':>7}{'n_no':>7}{'sig mean':>11}"
              f"{'no-sig mean':>13}{'WITHIN diff':>13}{'p':>8}   {'sig win%':>9}{'no win%':>9}")
        print("-" * 100)
        for t in ("low", "mid", "high"):
            tm = terc == t
            s = sig & tm
            ns = sig & ~np.array(False) & tm
            no = (~sig) & tm
            rng = f"{lvl[tm].min():.2f}–{lvl[tm].max():.2f}"
            if s.sum() < 25:
                print(f"{t:<10}{rng:<16}{int(s.sum()):>7}{int(no.sum()):>7}"
                      f"{'  (n<25: INCONCLUSIVE)':>60}")
                continue
            ms, mn = g5[s].mean(), g5[no].mean()
            p = rotation_pvalue(sig, g5, ms, restrict=tm)
            print(f"{t:<10}{rng:<16}{int(s.sum()):>7}{int(no.sum()):>7}{ms*100:>10.2f}%"
                  f"{mn*100:>12.2f}%{(ms-mn)*100:>12.2f}%{p:>8.3f}{stars(p)}"
                  f"{(g5[s]>0).mean()*100:>8.1f}%{(g5[no]>0).mean()*100:>8.1f}%")
        # pooled within-tercile difference (level-neutralised): n-weighted average of within diffs
        num = den = 0.0
        for t in ("low", "mid", "high"):
            tm = terc == t
            s, no = sig & tm, (~sig) & tm
            if s.sum() >= 25 and no.sum() >= 25:
                num += s.sum() * (g5[s].mean() - g5[no].mean())
                den += s.sum()
        if den:
            print(f"  pooled level-neutral within-tercile diff: {num/den*100:+.3f}%  "
                  f"(weighted over n={int(den)} signal days)")
        print(f"  raw unconditional excess (no level control): "
              f"{(g5[sig].mean()-base)*100:+.3f}%  n={int(sig.sum())}")

        # finer control: 20 equal-count vix_pct1y bins, matched within-bin difference
        b20 = pd.qcut(a.vix_pct1y, 20, labels=False, duplicates="drop").values

        def matched_diff(mask):
            num = den = 0.0
            for k in range(int(b20.max()) + 1):
                bm = b20 == k
                s, no = mask & bm, (~mask) & bm
                if s.sum() >= 5 and no.sum() >= 5:
                    num += s.sum() * (g5[s].mean() - g5[no].mean())
                    den += s.sum()
            return num / den if den else np.nan, den

        md, mdn = matched_diff(sig)
        # rotation null for the matched statistic
        null = np.empty(N_ROT)
        for i, off in enumerate(RNG.integers(1, len(sig), size=N_ROT)):
            null[i] = matched_diff(np.roll(sig, off))[0]
        null = null[~np.isnan(null)]
        pm = float((np.abs(null - null.mean()) >= abs(md - null.mean())).mean())
        print(f"  matched on 20 vix_pct1y bins: within-bin diff {md*100:+.3f}%  "
              f"p={pm:.3f}{stars(pm)}  (n={int(mdn)} signal days matched)")

    # ---------------------------------------------------------- 4/5. horse-race regressions
    print("\n" + "=" * 100)
    print("4/5. HORSE RACE — OLS of g5 on level, then level + band dummies, then + momentum controls")
    print("=" * 100)
    one = np.ones(len(a))
    z = a.vix_z1y.values
    chg = a.vix_chg10.values
    stretch = a.stretch.values
    bl, ab = below.astype(float), above.astype(float)

    models = [
        ("(a) level only:            g5 ~ pct1y", [("pct1y", lvl)], []),
        ("(b) level + bands:         g5 ~ pct1y + below + above", [("pct1y", lvl)], ["below", "above"]),
        ("(c) + z-score:             g5 ~ pct1y + z1y + below + above", [("pct1y", lvl), ("z1y", z)], ["below", "above"]),
        ("(d) + 10d VIX momentum:    g5 ~ pct1y + z1y + vixchg10 + below + above",
         [("pct1y", lvl), ("z1y", z), ("vixchg10", chg)], ["below", "above"]),
        ("(e) + stretch (VIX/MA10):  g5 ~ pct1y + z1y + vixchg10 + stretch + below + above",
         [("pct1y", lvl), ("z1y", z), ("vixchg10", chg), ("stretch", stretch)], ["below", "above"]),
        ("(f) bands only (no level): g5 ~ below + above", [], ["below", "above"]),
        ("(g) continuous %B:         g5 ~ pct1y + bb10_pctb",
         [("pct1y", lvl), ("pctb", a.bb10_pctb.values)], []),
    ]
    dummies = {"below": (bl, below), "above": (ab, above)}

    print(f"{'model':<58}{'R^2':>9}   coefficients (per-unit effect on 5-day SPX return)")
    print("-" * 100)
    for name, ctrls, dums in models:
        cols = [one] + [c for _, c in ctrls] + [dummies[k][0] for k in dums]
        X = np.column_stack(cols)
        beta, r2 = ols(g5, X)
        names = ["const"] + [n for n, _ in ctrls] + dums
        parts = []
        for nm, b in zip(names, beta):
            if nm == "const":
                continue
            parts.append(f"{nm}={b*100:+.3f}%")
        print(f"{name:<58}{r2*100:>8.4f}%   " + "  ".join(parts))
        # rotation p for each dummy, controls held fixed
        for k in dums:
            dvec, dmask = dummies[k]
            other = [dummies[o][0] for o in dums if o != k]
            ctrl = np.column_stack([one] + [c for _, c in ctrls] + other)
            Q, _ = np.linalg.qr(ctrl)
            y_res = g5 - Q @ (Q.T @ g5)
            obs = fwl_coef(y_res, dvec, Q)
            p = dummy_rotation_p(g5, ctrl, dmask, obs)
            print(f"{'':<58}{'':>9}     {k:<6} coef {obs*100:+.3f}%  rotation p={p:.3f} {stars(p)}")

    # incremental R^2
    X_lvl = np.column_stack([one, lvl])
    _, r2_lvl = ols(g5, X_lvl)
    X_full = np.column_stack([one, lvl, bl, ab])
    _, r2_full = ols(g5, X_full)
    X_band = np.column_stack([one, bl, ab])
    _, r2_band = ols(g5, X_band)
    print(f"\n  R^2 level only      : {r2_lvl*100:.4f}%")
    print(f"  R^2 level + bands   : {r2_full*100:.4f}%   incremental from bands: "
          f"{(r2_full-r2_lvl)*100:.4f} pp")
    print(f"  R^2 bands only      : {r2_band*100:.4f}%")

    # ---------------------------------------------------------- 6. horizon robustness
    print("\n" + "=" * 100)
    print("6. HORIZON ROBUSTNESS — below/above dummy coefficient controlling for pct1y (model b)")
    print("=" * 100)
    print(f"{'horizon':<10}{'n':>8}{'below coef':>13}{'p':>8}   {'above coef':>13}{'p':>8}   "
          f"{'raw below exc':>15}{'raw above exc':>15}")
    print("-" * 100)
    for h in (1, 3, 5, 10, 21):
        col = f"g{h}"
        sub = d.loc[a.index, col]
        ok = sub.notna().values
        y = sub.values[ok]
        o = np.ones(ok.sum())
        l2, b2, a2 = lvl[ok], bl[ok], ab[ok]
        ctrl_b = np.column_stack([o, l2, a2])
        ctrl_a = np.column_stack([o, l2, b2])
        Qb, _ = np.linalg.qr(ctrl_b)
        Qa, _ = np.linalg.qr(ctrl_a)
        cb = fwl_coef(y - Qb @ (Qb.T @ y), b2, Qb)
        ca = fwl_coef(y - Qa @ (Qa.T @ y), a2, Qa)
        pb = dummy_rotation_p(y, ctrl_b, below[ok], cb)
        pa = dummy_rotation_p(y, ctrl_a, above[ok], ca)
        bsub = base_h = y.mean()
        rb = y[below[ok]].mean() - base_h
        ra = y[above[ok]].mean() - base_h
        print(f"D{h:<9}{ok.sum():>8}{cb*100:>12.3f}%{pb:>8.3f}{stars(pb)}"
              f"{ca*100:>12.3f}%{pa:>8.3f}{stars(pa)}{rb*100:>14.3f}%{ra*100:>14.3f}%")

    # ---------------------------------------------------------- 7. where do band days live?
    print("\n" + "=" * 100)
    print("7. OVERLAP — distribution of band days across vix_pct1y deciles (is the band just 'low VIX'?)")
    print("=" * 100)
    tab = pd.DataFrame({"dec": dec + 1, "below": below, "above": above, "reentry": reentry})
    piv = tab.groupby("dec")[["below", "above", "reentry"]].sum()
    piv["days"] = tab.groupby("dec").size()
    piv["below%ofdec"] = piv.below / piv.days * 100
    piv["above%ofdec"] = piv.above / piv.days * 100
    print(piv.to_string())
    print(f"\n  share of all below-band days in bottom level tercile: "
          f"{(below & (terc=='low')).sum()/below.sum()*100:.1f}%")
    print(f"  share of all above-band days in top level tercile:    "
          f"{(above & (terc=='high')).sum()/above.sum()*100:.1f}%")


if __name__ == "__main__":
    main()
