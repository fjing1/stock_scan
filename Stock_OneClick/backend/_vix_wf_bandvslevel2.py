"""
_vix_wf_bandvslevel2.py — robustness round for the band-vs-level control study.

Round 1 (_vix_wf_bandvslevel.py) said: bb10_2.0_below survives a LINEAR vix_pct1y control.
Linear is a weak control though — the decile curve is non-monotonic. So here:

  A. level control as DECILE / VIGINTILE FIXED EFFECTS (nonparametric), not a linear term
  B. alternative level definitions: vix_pct2y, log(vix), raw vix quintile FE
  C. STRETCH-decile fixed effects — the band IS a threshold on stretch/sd, so this asks whether
     the sd-normalisation carries anything past "VIX is far under its MA10"
  D. era stability of the LEVEL-NEUTRAL (20-bin matched) below-band effect
  E. independent replication on BB(20,2.0) and BB(10,2.5)
  F. how separable are the two: contingency of below-band vs bottom-decile level

Run: ../../vcp_env/bin/python _vix_wf_bandvslevel2.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _vix_data

RNG = np.random.default_rng(20260911)
N_ROT = 5000


def stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "   "
    return "***" if p < 0.01 else ("** " if p < 0.05 else ("*  " if p < 0.10 else "   "))


def fwl_coef(y_res, dvec, Q):
    d_res = dvec - Q @ (Q.T @ dvec)
    den = float(d_res @ d_res)
    return float(d_res @ y_res) / den if den > 1e-12 else np.nan


def dummy_fe_test(y, fe_codes_list, mask, label, extra=None):
    """Coefficient on `mask` in a regression of y on fixed effects (one-hot of each code array
    in fe_codes_list) + optional extra continuous controls. p via circular rotation of `mask`."""
    cols = [np.ones(len(y))]
    for codes in fe_codes_list:
        codes = np.asarray(codes)
        for k in np.unique(codes)[1:]:            # drop first level (collinear with intercept)
            cols.append((codes == k).astype(float))
    if extra is not None:
        for c in extra:
            cols.append(np.asarray(c, dtype=float))
    X = np.column_stack(cols)
    Q, _ = np.linalg.qr(X)
    y_res = y - Q @ (Q.T @ y)
    dm = mask.astype(float)
    obs = fwl_coef(y_res, dm, Q)
    n = len(dm)
    null = np.empty(N_ROT)
    for i, off in enumerate(RNG.integers(1, n, size=N_ROT)):
        null[i] = fwl_coef(y_res, np.roll(dm, off), Q)
    null = null[~np.isnan(null)]
    p = float((np.abs(null - null.mean()) >= abs(obs - null.mean())).mean())
    print(f"  {label:<62}{obs*100:>+9.3f}%   p={p:.3f} {stars(p)}   n_sig={int(mask.sum())}")
    return obs, p


def main():
    d = _vix_data.add_features(_vix_data.load())
    d["vix_chg10"] = d.vix / d.vix.shift(10) - 1.0
    d["logvix"] = np.log(d.vix)

    keep = ["g5", "g3", "vix_pct1y", "vix_pct2y", "vix_z1y", "vix_chg10", "stretch",
            "logvix", "vix", "bb10_2.0_below", "bb10_2.0_above", "bb20_2.0_below",
            "bb20_2.0_above", "bb10_2.5_below", "bb10_2.0_reentry"]
    a = d[keep].dropna().copy()
    for c in ["bb10_2.0_below", "bb10_2.0_above", "bb20_2.0_below", "bb20_2.0_above",
              "bb10_2.5_below", "bb10_2.0_reentry"]:
        a[c] = a[c].astype(bool)
    y = a.g5.values
    base = y.mean()
    below = a["bb10_2.0_below"].values
    above = a["bb10_2.0_above"].values
    print(f"frame {len(a):,} rows  {a.index[0].date()} → {a.index[-1].date()} "
          f"(vix_pct2y needs 504d warm-up)   baseline g5 {base*100:+.3f}%")
    print(f"below n={below.sum()}  above n={above.sum()}")

    dec1 = pd.qcut(a.vix_pct1y, 10, labels=False, duplicates="drop").values
    vig1 = pd.qcut(a.vix_pct1y, 20, labels=False, duplicates="drop").values
    dec2 = pd.qcut(a.vix_pct2y, 10, labels=False, duplicates="drop").values
    qvix = pd.qcut(a.vix, 10, labels=False, duplicates="drop").values
    decs = pd.qcut(a.stretch, 10, labels=False, duplicates="drop").values
    vigs = pd.qcut(a.stretch, 20, labels=False, duplicates="drop").values
    decm = pd.qcut(a.vix_chg10, 10, labels=False, duplicates="drop").values

    print("\n" + "=" * 100)
    print("A/B. NONPARAMETRIC LEVEL CONTROLS — coefficient on the BELOW-band dummy (D5, g5)")
    print("=" * 100)
    print(f"  {'control set':<62}{'below coef':>10}")
    dummy_fe_test(y, [], below, "no control (raw excess)")
    dummy_fe_test(y, [dec1], below, "vix_pct1y DECILE fixed effects")
    dummy_fe_test(y, [vig1], below, "vix_pct1y VIGINTILE fixed effects (20 bins)")
    dummy_fe_test(y, [dec2], below, "vix_pct2y decile fixed effects")
    dummy_fe_test(y, [qvix], below, "RAW VIX decile fixed effects (absolute level)")
    dummy_fe_test(y, [], below, "linear log(VIX) + z1y", extra=[a.logvix.values, a.vix_z1y.values])
    dummy_fe_test(y, [dec1, decm], below, "pct1y deciles + 10d-VIX-change deciles (level+momentum)")
    dummy_fe_test(y, [dec1], below, "pct1y deciles + linear stretch",
                  extra=[a.stretch.values])

    print("\n" + "=" * 100)
    print("C. STRETCH CONTROL — the band IS a threshold on stretch/sd. Does sd-normalisation add?")
    print("=" * 100)
    dummy_fe_test(y, [decs], below, "stretch DECILE fixed effects")
    dummy_fe_test(y, [vigs], below, "stretch VIGINTILE fixed effects")
    dummy_fe_test(y, [decs, dec1], below, "stretch deciles + vix_pct1y deciles")
    dummy_fe_test(y, [vigs, vig1], below, "stretch vigintiles + vix_pct1y vigintiles")
    # where do below-band days sit in the stretch distribution?
    print("\n  below-band days by stretch decile: " +
          " ".join(f"S{k+1}:{int((below & (decs==k)).sum())}" for k in range(10)))
    print("  above-band days by stretch decile: " +
          " ".join(f"S{k+1}:{int((above & (decs==k)).sum())}" for k in range(10)))
    # head-to-head: bottom stretch decile as a plain no-band signal
    s1 = decs == 0
    print(f"\n  plain 'bottom stretch decile' signal: n={int(s1.sum())}  "
          f"raw excess {(y[s1].mean()-base)*100:+.3f}%   "
          f"(below-band raw excess {(y[below].mean()-base)*100:+.3f}%, n={int(below.sum())})")
    ov = int((below & s1).sum())
    print(f"  overlap: {ov}/{int(below.sum())} below-band days are in the bottom stretch decile "
          f"({ov/below.sum()*100:.0f}%)")
    nb = below & ~s1
    if nb.sum() >= 25:
        print(f"  below-band days OUTSIDE the bottom stretch decile: n={int(nb.sum())}  "
              f"mean {y[nb].mean()*100:+.3f}%  excess {(y[nb].mean()-base)*100:+.3f}%")
    else:
        print(f"  below-band days OUTSIDE bottom stretch decile: n={int(nb.sum())} — INCONCLUSIVE")

    print("\n" + "=" * 100)
    print("D. ERA STABILITY of the LEVEL-NEUTRAL below-band effect (within vix_pct1y vigintile)")
    print("=" * 100)

    def matched(mask, sel):
        num = den = 0.0
        for k in np.unique(vig1):
            bm = (vig1 == k) & sel
            s, no = mask & bm, (~mask) & bm
            if s.sum() >= 3 and no.sum() >= 10:
                num += s.sum() * (y[s].mean() - y[no].mean())
                den += s.sum()
        return (num / den if den else np.nan), den

    yrs = a.index.year
    print(f"  {'era':<12}{'n_below':>9}{'raw excess':>13}{'level-matched diff':>21}")
    for nm, y0, y1 in [("1990s", 1990, 1999), ("2000s", 2000, 2009),
                       ("2010s", 2010, 2019), ("2020s", 2020, 2099)]:
        sel = (yrs >= y0) & (yrs <= y1)
        s = below & sel
        if s.sum() < 10:
            print(f"  {nm:<12}{int(s.sum()):>9}   (too few)")
            continue
        raw = y[s].mean() - y[sel].mean()
        md, mdn = matched(below, sel)
        flag = "" if s.sum() >= 25 else "   <-- n<25, INCONCLUSIVE"
        print(f"  {nm:<12}{int(s.sum()):>9}{raw*100:>12.3f}%{md*100:>20.3f}%{flag}")

    print("\n" + "=" * 100)
    print("E. REPLICATION on other band parameterisations (D5, control = vix_pct1y vigintile FE)")
    print("=" * 100)
    for col in ["bb10_2.0_below", "bb20_2.0_below", "bb10_2.5_below",
                "bb10_2.0_above", "bb20_2.0_above", "bb10_2.0_reentry"]:
        m = a[col].values
        if m.sum() < 25:
            print(f"  {col:<62}   n={int(m.sum())} INCONCLUSIVE")
            continue
        dummy_fe_test(y, [vig1], m, col)

    print("\n" + "=" * 100)
    print("F. SEPARABILITY — 2x2 of below-band x bottom-level-decile (D5 means)")
    print("=" * 100)
    d1 = dec1 == 0
    print(f"  {'cell':<44}{'n':>7}{'mean D5':>11}{'excess':>10}{'win%':>8}")
    for nm, m in [("below-band AND bottom level decile", below & d1),
                  ("below-band, NOT bottom level decile", below & ~d1),
                  ("bottom level decile, NOT below-band", ~below & d1),
                  ("neither", ~below & ~d1)]:
        tag = "" if m.sum() >= 25 else "  (n<25 INCONCLUSIVE)"
        print(f"  {nm:<44}{int(m.sum()):>7}{y[m].mean()*100:>10.2f}%"
              f"{(y[m].mean()-base)*100:>9.2f}%{(y[m]>0).mean()*100:>7.1f}%{tag}")

    # D3 check too, since D3 was the strongest horizon for below in round 1
    y3 = a.g3.values
    print("\n" + "=" * 100)
    print("G. SAME TESTS AT D3 (g3) — below-band coefficient")
    print("=" * 100)
    for lbl, fes, ex in [("no control", [], None),
                         ("vix_pct1y vigintile FE", [vig1], None),
                         ("stretch vigintile FE", [vigs], None),
                         ("pct1y + stretch vigintile FE", [vig1, vigs], None)]:
        dummy_fe_test(y3, fes, below, lbl, extra=ex)


if __name__ == "__main__":
    main()
