"""
_vix_wf_verify_snooping_lowerband3.py — fairness pass: give the claim the SMALLEST defensible
search family and see whether it ever clears FWER .05.

Also: the modern-era question (the whole point of a trading claim) under the same correction.

Run: ../../vcp_env/bin/python _vix_wf_verify_snooping_lowerband3.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import _vix_data  # noqa: E402
from _vix_wf_verify_snooping_lowerband import (  # noqa: E402
    MIN_N, HORIZONS, fwl_rot_all, p_from_null, stars, fe_Q, hr, build,
)


def rw(labels, masks_hs, y_by_h, Q):
    """Romano-Wolf step-down over (label, mask, horizon) triples. Returns DataFrame."""
    nulls, zobs = [], []
    for lbl, m, h in masks_hs:
        c, nl = fwl_rot_all(m, y_by_h[h], Q)
        mu, sd = np.nanmean(nl), max(np.nanstd(nl), 1e-15)
        nulls.append(np.abs(nl - mu) / sd)
        zobs.append(abs((c - mu) / sd))
    Z = np.vstack(nulls)
    zobs = np.array(zobs)
    order = np.argsort(-zobs)
    alive = np.ones(len(zobs), bool)
    adj = np.empty(len(zobs))
    run = 0.0
    for i in order:
        run = max(run, float((np.nanmax(Z[alive], axis=0) >= zobs[i]).mean()))
        adj[i] = run
        alive[i] = False
    return pd.DataFrame({"label": labels, "z": zobs, "rw_p": adj})


def main():
    d = build()
    hcols = [f"g{h}" for h in HORIZONS]
    below_cols = [f"bb{n}_{k}_below" for n in (10, 20) for k in (1.5, 2.0, 2.5)]
    fam = d[hcols + ["vix_pct1y"] + below_cols].dropna().copy()
    yh = {h: fam[f"g{h}"].values for h in HORIZONS}
    nf = len(fam)
    Q = fe_Q(nf, [pd.qcut(fam.vix_pct1y, 20, labels=False, duplicates="drop").values])
    print(f"frame n={nf:,}  {fam.index[0].date()} -> {fam.index[-1].date()}")

    hr("SMALLEST DEFENSIBLE FAMILIES — Romano-Wolf FWER p for bb10_2.0_below at D5")
    m20 = fam["bb10_2.0_below"].astype(bool).values
    fams = {
        "k=1  (a single pre-registered test, no search at all)":
            [("bb10_2.0_below|D5", m20, 5)],
        "k=5  (one signal x the 5 horizons this repo always scans)":
            [(f"bb10_2.0_below|D{h}", m20, h) for h in HORIZONS],
        "k=30 (the 6 below-band variants of _vix_ma10_bb_research x 5 horizons)":
            [(f"{c}|D{h}", fam[c].astype(bool).values, h)
             for c in below_cols for h in HORIZONS
             if fam[c].astype(bool).values.sum() >= MIN_N],
    }
    for nm, trip in fams.items():
        labels = [t[0] for t in trip]
        out = rw(labels, trip, yh, Q)
        t = out[out.label == "bb10_2.0_below|D5"].iloc[0]
        print(f"  {nm:<66} k={len(trip):>3}  FWER p = {t.rw_p:.3f} {stars(t.rw_p)}")
    print("  (round 1: k=70 -> .374, k=210 -> .717, k=255 -> .768)")

    hr("MODERN ERA under the same machinery — is there anything left to correct?")
    yrs = fam.index.year
    for nm, sel in [("full 1990-2026", np.ones(nf, bool)),
                    ("1990-2009", yrs <= 2009),
                    ("2010-2026", yrs >= 2010),
                    ("2015-2026", yrs >= 2015)]:
        sub = fam[sel]
        ms = m20[sel]
        if ms.sum() < MIN_N:
            print(f"  {nm:<20} n_below={int(ms.sum()):>4}  n<{MIN_N} INCONCLUSIVE")
            continue
        Qs = fe_Q(int(sel.sum()),
                  [pd.qcut(sub.vix_pct1y, 20, labels=False, duplicates="drop").values])
        yy = sub.g5.values
        c, nl = fwl_rot_all(ms, yy, Qs)
        # FWER within the k=30 family, refit inside the window
        trip = [(f"{cc}|D{h}", sub[cc].astype(bool).values, h)
                for cc in below_cols for h in HORIZONS
                if sub[cc].astype(bool).values.sum() >= MIN_N]
        out = rw([t[0] for t in trip], trip, {h: sub[f"g{h}"].values for h in HORIZONS}, Qs)
        row = out[out.label == "bb10_2.0_below|D5"]
        fw = float(row.rw_p.iloc[0]) if len(row) else np.nan
        print(f"  {nm:<20} n_below={int(ms.sum()):>4}  coef {c*100:>+7.3f}%  "
              f"nominal p={p_from_null(c, nl):.3f}  FWER(k={len(trip)}) p={fw:.3f} {stars(fw)}")

    hr("HOW MUCH OF THE 1990-2026 EFFECT IS PRE-2010? (below-band days, sum of per-day excess)")
    a = d[["g5", "vix_pct1y", "bb10_2.0_below"]].dropna()
    y = a.g5.values
    b = a["bb10_2.0_below"].astype(bool).values
    base = y.mean()
    yr = a.index.year
    tot = (y[b] - base).sum()
    pre = (y[b & (yr <= 2009)] - base).sum()
    print(f"  total {tot*100:.1f} pp-days over {int(b.sum())} days")
    print(f"  1990-2009: {pre*100:.1f} pp-days from {int((b&(yr<=2009)).sum())} days "
          f"({pre/tot*100:.0f}% of the total from {(b&(yr<=2009)).sum()/b.sum()*100:.0f}% of the days)")
    print(f"  2010-2026: {(tot-pre)*100:.1f} pp-days from {int((b&(yr>=2010)).sum())} days "
          f"({(tot-pre)/tot*100:.0f}%)")


if __name__ == "__main__":
    main()
