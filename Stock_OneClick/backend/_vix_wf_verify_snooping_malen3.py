"""
_vix_wf_verify_snooping_malen3.py — third pass: stress the COUNTER-examples I found, so I don't
refute a claim with an artefact of my own.

The q-sweep produced blocks of lengths that DO separate from SMA10 at matched frequency
(q=0.20: L=5..9 beat L=10; q=0.30: L=13..23 lag L=10; q=0.03: L=12 lags, p=0.0001). Before using
them against the claim, check they are not episode-driven:
  - drop 2008, drop 2020, drop both
  - split by half-sample
  - check the paired null sd (is the small p coming from a real gap or from a tiny crit?)

Run: ../../vcp_env/bin/python _vix_wf_verify_snooping_malen3.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _vix_data
from _vix_wf_verify_snooping_malen import (LENGTHS, excess, paired_rotation_detail,  # noqa: F401
                                           stars, stretch, topq_mask)


def main():
    d = _vix_data.add_features(_vix_data.load())
    vix, fwd5 = d.vix, d.g5.values
    valid5 = ~np.isnan(fwd5)
    n = len(d)
    yrs = d.index.year.values
    s = {L: stretch(vix, L, "sma") for L in LENGTHS}

    cases = [(0.03, 12), (0.05, 11), (0.20, 5), (0.20, 6), (0.20, 8), (0.30, 18), (0.30, 21),
             (0.12, 24), (0.12, 5), (0.12, 20)]
    scen = [("full", np.ones(n, bool)),
            ("drop 2008", yrs != 2008),
            ("drop 2020", yrs != 2020),
            ("drop 2008+2020", (yrs != 2008) & (yrs != 2020)),
            ("1990-2008", np.arange(n) < n // 2),
            ("2008-2026", np.arange(n) >= n // 2)]

    print("=" * 118)
    print("STRESS TEST of every length/frequency cell that separated from SMA10")
    print("  (paired rotation p for 'SMA_L top-q% has the same conditional D5 mean as SMA10 top-q%')")
    print("=" * 118)
    print(f"  {'cell':<16}" + "".join(f"{lbl:>17}" for lbl, _ in scen))
    for q, L in cases:
        mq_L = topq_mask(s[L], q)
        mq_10 = topq_mask(s[10], q)
        out = []
        for lbl, keep in scen:
            f = np.where(keep & valid5, fwd5, np.nan)
            nn = int((mq_L & keep & valid5).sum())
            if nn < 25:
                out.append(f"n={nn} incl.".rjust(17))
                continue
            o, p, c95, _, sd, _, _ = paired_rotation_detail(mq_L, mq_10, f)
            out.append(f"{o*100:+.3f}% p{p:.3f}".rjust(17))
        print(f"  q={q:.2f} L={L:<8}" + "".join(out))

    print("\n  (each cell shows: diff in D5 conditional mean vs SMA10, and the paired rotation p)")

    print("\n" + "=" * 118)
    print("Where does the small p come from — a big gap, or a tiny critical value?")
    print("=" * 118)
    print(f"  {'cell':<14}{'nL':>7}{'jaccard':>10}{'diff':>10}{'null sd':>10}{'crit95':>10}{'p':>9}")
    for q, L in cases:
        mq_L, mq_10 = topq_mask(s[L], q), topq_mask(s[10], q)
        o, p, c95, _, sd, na, _ = paired_rotation_detail(mq_L, mq_10, fwd5)
        jac = (mq_L & mq_10).sum() / max((mq_L | mq_10).sum(), 1)
        print(f"  q={q:.2f} L={L:<6}{na:>7}{jac:>10.3f}{o*100:>9.3f}%{sd*100:>9.3f}%{c95*100:>9.3f}%"
              f"{p:>9.4f}{stars(p)}")

    print("\n" + "=" * 118)
    print("BLOCK COHERENCE at q=0.20 and q=0.30 — the full length curve, so the 'significant block'")
    print("can be read in context rather than as isolated cells.")
    print("=" * 118)
    for q in (0.20, 0.30):
        mq = {L: topq_mask(s[L], q) for L in LENGTHS}
        exc10, n10 = excess(mq[10], fwd5)
        print(f"\n  q={q:.2f} (n~{n10} each), SMA10 excess {exc10*100:+.3f}%")
        print(f"  {'L':>4}" + "".join(f"{L:>7}" for L in LENGTHS))
        vals = [excess(mq[L], fwd5)[0] * 100 for L in LENGTHS]
        ps = [paired_rotation_detail(mq[L], mq[10], fwd5)[1] if L != 10 else np.nan for L in LENGTHS]
        print(f"  {'exc':>4}" + "".join(f"{v:>7.3f}" for v in vals))
        print(f"  {'p':>4}" + "".join((f"{p:>7.3f}" if p == p else f"{'--':>7}") for p in ps))

    print("\n" + "=" * 118)
    print("CONTROL: how many p<0.05 cells does the SAME 222-cell grid produce when the outcome is")
    print("replaced by a phase-randomised surrogate return series (real edge destroyed, all serial")
    print("dependence preserved)? This calibrates 'is 40/222 unusual'.")
    print("=" * 118)
    rng = np.random.default_rng(7)
    counts, minps = [], []
    qs = (0.03, 0.05, 0.10, 0.12, 0.20, 0.30)
    masks = {(q, L): topq_mask(s[L], q) for q in qs for L in LENGTHS}
    for rep in range(20):
        off = int(rng.integers(500, n - 500))
        f = np.roll(fwd5, off)            # circular shift of the OUTCOME = surrogate under the null
        c, mp = 0, 1.0
        for q in qs:
            for L in LENGTHS:
                if L == 10:
                    continue
                p = paired_rotation_detail(masks[(q, L)], masks[(q, 10)], f)[1]
                if p < 0.05:
                    c += 1
                mp = min(mp, p)
        counts.append(c)
        minps.append(mp)
    counts = np.array(counts)
    minps = np.array(minps)
    print(f"  surrogate grids (20 reps): p<0.05 count  median {np.median(counts):.0f}  "
          f"90th pct {np.percentile(counts,90):.0f}  max {counts.max()}   (REAL data: 40)")
    print(f"  surrogate grids: min-p over the 222 cells  median {np.median(minps):.5f}  "
          f"5th pct {np.percentile(minps,5):.5f}  min {minps.min():.5f}   (REAL data: 0.00011)")
    print(f"  fraction of surrogate grids with >=40 significant cells: "
          f"{(counts>=40).mean():.2f}")
    print(f"  fraction of surrogate grids with min-p <= 0.00011: {(minps<=0.00011).mean():.2f}")


if __name__ == "__main__":
    main()
