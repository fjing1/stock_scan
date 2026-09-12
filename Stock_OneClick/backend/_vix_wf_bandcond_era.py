"""
_vix_wf_bandcond_era.py — era stability + parameter robustness for the two cells that survived
PART 6 of _vix_wf_bandcond.py:
    A) band=below (BB10,2 lower)  &  SPX > MA20   -> tail risk UP
    B) band=above (BB10,2 upper)  &  SPX < MA20   -> tail risk DOWN, mean UP
Every established bullish VIX variant in this repo dies in the 2020s, so the risk-side claims
have to be shown era-by-era before anyone wires them into the scanner.
Run: ../../vcp_env/bin/python _vix_wf_bandcond_era.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

from _vix_wf_bandcond import MIN_N, N_ROT, RNG, build, p_lt2, q05, sect, stars  # noqa: E402

ERAS = [("1990s", 1990, 1999), ("2000s", 2000, 2009),
        ("2010s", 2010, 2019), ("2020s", 2020, 2099)]


def subset_rot(vals, mask, statfn):
    valid = ~np.isnan(vals)
    n = int((mask & valid).sum())
    if n < MIN_N:
        return n, np.nan, np.nan, np.nan
    obs = float(statfn(vals[mask & valid]))
    sb = float(statfn(vals[valid]))
    offs = RNG.integers(1, len(mask), size=N_ROT)
    null = [statfn(vals[np.roll(mask, o) & valid]) for o in offs
            if (np.roll(mask, o) & valid).sum() >= 10]
    null = np.array(null)
    p = float((np.abs(null - null.mean()) >= abs(obs - null.mean())).mean())
    return n, obs, obs - sb, p


def main():
    d = build()
    d = d[d.band.notna() & d.spx_ma20.notna()].copy()
    yrs = d.index.year.values

    sect("ERA STABILITY of the two surviving cells (excess is vs the SAME-ERA, SAME-REGIME baseline)")
    for cell, bandval, reg_above in (("A: band=BELOW lower & SPX>MA20", "below", True),
                                     ("B: band=ABOVE upper & SPX<MA20", "above", False)):
        print(f"\n  {cell}")
        for stat_nm, col, fn in (("mean g5", "g5", np.mean),
                                 ("P(g5<-2%)", "g5", p_lt2),
                                 ("5th pct g5", "g5", q05)):
            parts = []
            for en, y0, y1 in ERAS:
                sub = d[(yrs >= y0) & (yrs <= y1) & (d.above_ma20 == reg_above)]
                vals = sub[col].values
                m = (sub.band == bandval).values
                valid = ~np.isnan(vals)
                n = int((m & valid).sum())
                if n < 15:
                    parts.append(f"{en} n={n} -")
                    continue
                exc = fn(vals[m & valid]) - fn(vals[valid])
                parts.append(f"{en} n={n} {exc*100:+.2f}%")
            print(f"    {stat_nm:<12} " + " | ".join(parts))

    sect("PARAMETER ROBUSTNESS — same two cells with BB(20,2.0) and BB(10,1.5)/BB(10,2.5) instead")
    for tag in ("bb10_2.0", "bb20_2.0", "bb10_1.5", "bb10_2.5", "bb20_1.5", "bb20_2.5"):
        for lbl, side, reg in (("BELOW & SPX>MA20", "below", "SPX>MA20"),
                               ("ABOVE & SPX<MA20", "above", "SPX<MA20")):
            sub = d[d.above_ma20] if reg == "SPX>MA20" else d[~d.above_ma20]
            mm = sub[f"{tag}_{side}"].values.astype(bool)
            row = []
            for stat_nm, col, fn in (("g5", "g5", np.mean), ("P<-2%", "g5", p_lt2)):
                n, obs, exc, p = subset_rot(sub[col].values, mm, fn)
                if n < MIN_N:
                    row.append(f"{stat_nm} n={n} INCONCL")
                else:
                    row.append(f"{stat_nm} n={n} exc {exc*100:+.2f}% p={p:.3f}{stars(p).strip()}")
            print(f"  {tag:<10} {lbl:<20} " + " | ".join(row))

    sect("HOW OFTEN DOES EACH FIRE? (practical value depends on frequency)")
    n = len(d)
    for lbl, m in (("band=below & SPX>MA20", (d.band == "below").values & d.above_ma20.values),
                   ("band=above & SPX<MA20", (d.band == "above").values & (~d.above_ma20).values),
                   ("SPX<MA20 (scanner flag alone)", (~d.above_ma20).values),
                   ("band != inside (any band signal)", (d.band != "inside").values)):
        yrsfire = pd.Series(m, index=d.index).groupby(d.index.year).sum()
        print(f"  {lbl:<34} {int(m.sum()):>5} days = {m.sum()/n*100:4.1f}% of sessions   "
              f"~{m.sum()/ (n/252):.1f} days/yr   years with 0 fires: "
              f"{int((yrsfire == 0).sum())}/{len(yrsfire)}")

    sect("LAST 5 YEARS ONLY (2021-2026) — does anything survive the recent window?")
    rec = d[d.index.year >= 2021]
    print(f"  n={len(rec)}  {rec.index[0].date()} -> {rec.index[-1].date()}")
    for reg, sub in (("SPX>MA20", rec[rec.above_ma20]), ("SPX<MA20", rec[~rec.above_ma20])):
        for stat_nm, col, fn in (("mean g5", "g5", np.mean), ("P(g5<-2%)", "g5", p_lt2)):
            vals = sub[col].values
            valid = ~np.isnan(vals)
            base = fn(vals[valid])
            line = [f"  {reg} {stat_nm:<11} regime base {base*100:+.2f}%"]
            for b in ("below", "above"):
                m = (sub.band == b).values
                nn = int((m & valid).sum())
                if nn < MIN_N:
                    line.append(f"{b} n={nn} INCONCL")
                else:
                    line.append(f"{b} n={nn} exc {(fn(vals[m & valid])-base)*100:+.2f}%")
            print("   | ".join(line))


if __name__ == "__main__":
    main()
