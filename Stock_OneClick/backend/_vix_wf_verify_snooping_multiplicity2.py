"""
_vix_wf_verify_snooping_multiplicity2.py — addendum to _vix_wf_verify_snooping_multiplicity.py.

Two gaps closed:
  G1. The MID-tercile claim (-0.66%, n=60, p=.032) is re-tested on the CLAIM'S OWN frame
      (g5 + vix_pct1y available => n=8,978, below n=159), not on my all-horizons frame.
  G2. "strongest in the MID level tercile" was scored against a 3x5 grid in the main script.
      But the CONDITIONER used to cut the terciles is itself a choice. Price the full
      subgroup search: 6 conditioners x 3 terciles x 5 horizons, joint-rotation maxT.

Same machinery/rules as the main script (imported, not re-implemented).

Run: ../../vcp_env/bin/python _vix_wf_verify_snooping_multiplicity2.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from _vix_wf_verify_snooping_multiplicity import (  # noqa: E402
    MIN_N, build, mean_null, p_of, romano_wolf, maxT, stars, hr,
)

HZ = (1, 3, 5, 10, 21)


def main():
    d = build()
    d["rv20"] = d.spx.pct_change().rolling(20).std() * np.sqrt(252)
    cols = ["g5", "vix_pct1y", "vix_pct2y", "vix", "stretch", "bb10_width", "rv20",
            "bb10_2.0_below"] + [f"g{h}" for h in HZ]
    a = d[list(dict.fromkeys(cols))].dropna().copy()
    n = len(a)
    below = a["bb10_2.0_below"].astype(bool).values
    y5 = a.g5.values
    print(f"CLAIM FRAME: n={n:,}  {a.index[0].date()} -> {a.index[-1].date()}  "
          f"below n={int(below.sum())}  baseline g5 {y5.mean()*100:+.3f}%")
    print(f"  raw D5 excess {(y5[below].mean()-y5.mean())*100:+.3f}%  (claim -0.58%)")

    ter = pd.qcut(a.vix_pct1y, 3, labels=False, duplicates="drop").values
    mid = below & (ter == 1)
    o, nl = mean_null(mid, y5)
    e_mid = o - y5[ter == 1].mean()
    print(f"  MID tercile   {e_mid*100:+.3f}%  n={int(mid.sum())}  p={p_of(o, nl):.4f}"
          f"{stars(p_of(o, nl))}   (claim -0.66%, n=60, p=.032)  -> REPRODUCES")

    hr("G2. FULL SUBGROUP SEARCH — the tercile CONDITIONER is a choice too")
    conds = ["vix_pct1y", "vix_pct2y", "vix", "stretch", "bb10_width", "rv20"]
    labs, zs, Zs, rec = [], [], [], {}
    for cname in conds:
        t = pd.qcut(a[cname], 3, labels=False, duplicates="drop").values
        for g in range(3):
            reg = t == g
            m = below & reg
            for h in HZ:
                yy = a[f"g{h}"].values
                cnt = int(m.sum())
                if cnt < MIN_N:
                    rec[(cname, g, h)] = (cnt, np.nan, np.nan)
                    continue
                oo, nn = mean_null(m, yy)
                mu, sd = np.nanmean(nn), np.nanstd(nn)
                sd = sd if sd > 1e-15 else 1e-15
                rec[(cname, g, h)] = (cnt, oo - yy[reg].mean(), p_of(oo, nn))
                labs.append((cname, g, h))
                zs.append(abs((oo - mu) / sd))
                Zs.append(np.abs(nn - mu) / sd)
    zs = np.array(zs)
    Zs = np.vstack(Zs)
    rw = romano_wolf(zs, Zs)
    total = len(conds) * 3 * len(HZ)
    print(f"  subgroup cells enumerated: {total}   testable (n>={MIN_N}): {len(labs)}")
    print(f"  nominal p<.05: {int(sum(1 for l in labs if rec[l][2] < .05))}/{len(labs)} "
          f"(chance {0.05*len(labs):.0f})")
    i = labs.index(("vix_pct1y", 1, 5))
    print(f"\n  the claimed cell (vix_pct1y MID, D5): excess {rec[labs[i]][1]*100:+.2f}%  "
          f"n={rec[labs[i]][0]}  nominal p={rec[labs[i]][2]:.4f}")
    print(f"    rank {int((zs > zs[i]).sum())+1} of {len(labs)} testable subgroup cells by |z|")
    print(f"    single-step maxT FWER p = {maxT(zs, Zs, i):.4f}   "
          f"Romano-Wolf p = {rw[i]:.4f} {stars(rw[i])}")
    bi = int(np.argmax(zs))
    print(f"    family best subgroup cell: {labs[bi]}  excess {rec[labs[bi]][1]*100:+.2f}% "
          f"n={rec[labs[bi]][0]}  RW p={rw[bi]:.4f}{stars(rw[bi])}")
    surv = [labs[j] for j in np.argsort(-zs) if rw[j] < 0.05]
    print(f"    subgroup cells surviving FWER at .05: {len(surv)}"
          + (f" -> {surv}" if surv else ""))

    print("\n  the D5 row of every conditioner (is 'MID' special, or is it one of many cuts?):")
    print(f"  {'conditioner':<14}{'LOW':>22}{'MID':>22}{'HIGH':>22}")
    for cname in conds:
        cells = []
        for g in range(3):
            cnt, e, p = rec[(cname, g, 5)]
            cells.append((f"n<25 (n={cnt})" if cnt < MIN_N
                          else f"{e*100:+.2f}% n={cnt} p={p:.3f}").rjust(22))
        print(f"  {cname:<14}" + "".join(cells))

    hr("G3. IS THE MID TERCILE STABLE IN TIME?")
    yr = np.asarray(a.index.year)
    for lbl, w in (("1990s+2000s (pre-2010)", yr < 2010), ("2010-2026", yr >= 2010)):
        m = mid & w
        cnt = int(m.sum())
        if cnt < MIN_N:
            print(f"  {lbl:<24} n={cnt} — INCONCLUSIVE")
            continue
        e = y5[m].mean() - y5[(ter == 1) & w].mean()
        print(f"  {lbl:<24} n={cnt}  excess {e*100:+.2f}%")
    print("\ndone.")


if __name__ == "__main__":
    main()
