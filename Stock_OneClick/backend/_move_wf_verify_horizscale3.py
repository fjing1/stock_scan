"""
_move_wf_verify_horizscale3.py — final control: is the singles QLIKE gain a STALE-PRICE artifact?

The two biggest single contributors to the h=63 singles gain are FUBO 2019-04-01
(sigma_ewma = 0.000245/day, i.e. 0.02%/day -- a frozen pre-reverse-merger shell quote) and
STI 2023-11-06 (543.00 printed for weeks, then -94%). Those are stale quotes, not markets.

Filter A: drop observations whose trailing 30 closes contain <15 DISTINCT values (stale quote).
Filter B: drop observations with sigma_ewma < 0.002/day (0.2%/day ~= 3% annualised: no real
          single name is that quiet; it means the price stopped printing).
Both filters use only PAST data at t, so they are implementable live and cause no lookahead.

Run: ../../vcp_env/bin/python _move_wf_verify_horizscale3.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import _move_data as D
import _move_lib as L

HS = [2, 5, 21, 63]
IDX = ["SPY", "QQQ", "^GSPC", "IWM", "DIA"]
MIN_OBS, FLOOR = 500, 1e-4
WGRID = np.round(np.concatenate([np.arange(0.02, 1.0, 0.025), [1.0]]), 4)
KGRID = np.array([0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50])
pd.set_option("display.width", 200)


def qlike_el(fc2, f2):
    x = f2 / fc2
    return x - np.log(x) - 1.0


def main():
    t0 = time.time()
    p = D.load()
    close = p["Close"]
    keep = [s for s in close.columns if close[s].notna().sum() >= MIN_OBS and s != "^VIX"]
    keep = [s for s in IDX if s in keep] + [s for s in keep if s not in IDX]
    close = close[keep]
    singles = [s for s in keep if s not in IDX]
    lr = np.log(close).diff()
    years = close.index.year.values
    yrs = sorted(set(years))
    s_e = L.vol_ewma(close, 0.94)
    fwd = {h: L.realized_vol_forward(close, h) for h in HS}
    nuniq = close.rolling(30).apply(lambda v: len(np.unique(v)), raw=True)
    print(f"stale diagnostics: share of single-name rows with <15 distinct closes in the last 30 "
          f"bars = {float((nuniq[singles] < 15).sum().sum() / nuniq[singles].notna().sum().sum()):.4f}"
          f";  share with sigma_ewma<0.002/day = "
          f"{float((s_e[singles] < 0.002).sum().sum() / s_e[singles].notna().sum().sum()):.4f}"
          f"   [{time.time()-t0:.0f}s]")

    out = []
    for lab, cols in [("INDICES", IDX), ("SINGLES", singles)]:
        SE = np.ascontiguousarray(s_e[cols].values)
        NU = np.ascontiguousarray(nuniq[cols].values)
        LRV = np.ascontiguousarray(lr[cols].values)
        for filt in ("none", "distinct>=15", "sigma>=0.002", "both"):
            keepm = np.isfinite(SE) & (SE >= FLOOR)
            if filt in ("distinct>=15", "both"):
                keepm &= np.isfinite(NU) & (NU >= 15)
            if filt in ("sigma>=0.002", "both"):
                keepm &= SE >= 0.002
            for h in HS:
                F = np.ascontiguousarray(fwd[h][cols].values)
                ok = keepm & np.isfinite(F) & (F >= FLOOR)
                acc = []
                for y in yrs[5:]:
                    tr = ok & (years[:, None] < y)
                    te = ok & (years[:, None] == y)
                    if tr.sum() < 1500 or te.sum() < 100:
                        continue
                    LRp = float(np.nanmean(LRV[years < y] ** 2))
                    f2t, s2t = F[tr] ** 2, SE[tr] ** 2
                    sc = [np.log(np.mean(f2t / np.maximum(LRp + (s2t - LRp) * w, FLOOR ** 2)))
                          + np.mean(np.log(np.maximum(LRp + (s2t - LRp) * w, FLOOR ** 2)))
                          for w in WGRID]
                    w = float(WGRID[int(np.argmin(sc))])
                    sck = [np.log(np.mean(f2t / np.maximum(s2t, k * LRp)))
                           + np.mean(np.log(np.maximum(s2t, k * LRp))) for k in KGRID]
                    kf = float(KGRID[int(np.argmin(sck))])
                    b2a_t = np.maximum(LRp + (s2t - LRp) * w, FLOOR ** 2)
                    b2f_t = np.maximum(s2t, kf * LRp)
                    cf, ca, ck = (float(np.mean(f2t / s2t)), float(np.mean(f2t / b2a_t)),
                                  float(np.mean(f2t / b2f_t)))
                    f2e, s2e = F[te] ** 2, SE[te] ** 2
                    acc.append((int(te.sum()),
                                float(qlike_el(cf * s2e, f2e).mean()),
                                float(qlike_el(ca * np.maximum(LRp + (s2e - LRp) * w,
                                                               FLOOR ** 2), f2e).mean()),
                                float(qlike_el(ck * np.maximum(s2e, kf * LRp), f2e).mean())))
                a = np.array(acc)
                nsum = a[:, 0].sum()
                fl, ar, fo = [float((a[:, i] * a[:, 0]).sum() / nsum) for i in (1, 2, 3)]
                out.append(dict(grp=lab, filter=filt, h=h, n=int(nsum), flat94=fl, ar1=ar,
                                floorK=fo, ar1_vs_flat_pct=100 * (1 - ar / fl),
                                floor_vs_flat_pct=100 * (1 - fo / fl)))
        print(f"  {lab} done [{time.time()-t0:.0f}s]")
    T = pd.DataFrame(out)
    print("\n" + "=" * 118)
    print("OOS QLIKE with and without the stale-quote filters (expanding walk-forward, 21 test")
    print("years, every parameter fit on strictly prior years).")
    print("=" * 118)
    for lab in ("INDICES", "SINGLES"):
        g = T[T.grp == lab]
        print(f"\n  {lab}:")
        print(g.pivot_table(index="h", columns="filter",
                            values=["n", "flat94", "ar1", "ar1_vs_flat_pct",
                                    "floor_vs_flat_pct"]).round(4).to_string())
    print(f"\ndone {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
