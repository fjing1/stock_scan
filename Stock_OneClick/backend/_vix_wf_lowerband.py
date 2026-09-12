"""
_vix_wf_lowerband.py — pin down the VIX Bollinger LOWER-band ("complacency") warning.

Builds on _vix_ma10_bb_research.py, which established BB(10,2.0) VIX-below-lower-band -> D5
excess -0.59% (p<.01, n=165) and found it the most era-stable of the VIX-band results.

Six questions:
  1. robustness surface over (n, k)
  2. persistence: entry day vs day-2..5 of the streak (ENTRY or STATE?)
  3. exit: what happens after the episode ends
  4. trend interaction: SPX above / below its own 200d SMA
  5. practical de-risk rule vs buy-hold, full sample and 2010-2026
  6. magnitude vs frequency: full g5 distribution conditional vs unconditional

Method (non-negotiable, per repo history):
  - every number is EXCESS over the SAME-SAMPLE unconditional mean
  - significance = circular rotation test (reuses the logic of rotation_pvalue in
    _vix_ma10_bb_research.py). Here it is done EXHAUSTIVELY over all n-1 non-zero offsets via
    FFT circular cross-correlation, which is the same null but with no Monte-Carlo noise and
    far more than the 2000 required draws. Validated against the repo's loop version below.
  - subgroups with n < 25 are printed as inconclusive and not interpreted
  - g* forward returns only (entry at the NEXT close; VIX settles 16:15 ET)

Run: ../../vcp_env/bin/python _vix_wf_lowerband.py
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

import _vix_data

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20260910)
HZ = [1, 3, 5, 10]
ERAS = [("1990s", 1990, 1999), ("2000s", 2000, 2009),
        ("2010s", 2010, 2019), ("2020s", 2020, 2099)]


# ------------------------------------------------------------------ rotation test
def rot_all(mask: np.ndarray, fwd: np.ndarray):
    """Conditional mean of `fwd` for EVERY circular rotation of `mask` (offsets 1..n-1).

    np.roll(mask, o)[i] == mask[i - o], so
        sum_i mask[i-o] * y[i]  ==  circular cross-correlation of mask with y at lag o
                                ==  ifft(conj(fft(mask)) * fft(y))[o]
    Same null hypothesis as rotation_pvalue() in _vix_ma10_bb_research.py: the signal's count
    and burstiness carry no information about WHEN it fires. Preserves the autocorrelation of
    overlapping forward returns and the clustering of the signal.
    """
    n = len(mask)
    valid = ~np.isnan(fwd)
    y = np.where(valid, np.nan_to_num(fwd), 0.0)
    m = mask.astype(float)
    M = np.fft.rfft(m, n)
    num = np.fft.irfft(np.conj(M) * np.fft.rfft(y, n), n)
    den = np.fft.irfft(np.conj(M) * np.fft.rfft(valid.astype(float), n), n)
    den = np.round(den)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    return means[1:], den[1:]          # drop the identity rotation


def rot_p(mask: np.ndarray, fwd: np.ndarray, observed: float) -> float:
    null, _ = rot_all(mask, fwd)
    null = null[~np.isnan(null)]
    if not len(null):
        return float("nan")
    base = null.mean()
    return float((np.abs(null - base) >= abs(observed - base)).mean())


def rot_p_loop(mask, fwd, observed, n_rot=5000):
    """Byte-for-byte the repo's Monte-Carlo version, kept only to validate rot_p."""
    n = len(mask)
    valid = ~np.isnan(fwd)
    null = np.empty(n_rot)
    offs = RNG.integers(1, n, size=n_rot)
    for i, off in enumerate(offs):
        m = np.roll(mask, off) & valid
        null[i] = fwd[m].mean() if m.sum() else np.nan
    null = null[~np.isnan(null)]
    base = np.nanmean(null)
    return float((np.abs(null - base) >= abs(observed - base)).mean())


def stars(p):
    if p != p:
        return "   "
    return "***" if p < 0.01 else ("** " if p < 0.05 else ("*  " if p < 0.10 else "   "))


def cell(d, mask, h, do_p=True):
    """(n, conditional mean, baseline, excess, p) for horizon h. Baseline = same valid sample."""
    fwd = d[f"g{h}"].values
    valid = ~np.isnan(fwd)
    sel = mask & valid
    k = int(sel.sum())
    if k == 0:
        return dict(n=0, cond=np.nan, base=np.nan, exc=np.nan, p=np.nan)
    cond, base = fwd[sel].mean(), fwd[valid].mean()
    p = rot_p(mask, fwd, cond) if do_p and k >= 25 else np.nan
    return dict(n=k, cond=cond, base=base, exc=cond - base, p=p,
                win=float((fwd[sel] > 0).mean()), win_base=float((fwd[valid] > 0).mean()))


def episodes(mask: np.ndarray) -> int:
    return int((mask & ~np.r_[False, mask[:-1]]).sum())


def streak_day(mask: np.ndarray) -> np.ndarray:
    """1 on the first day of a True run, 2 on the second, ... 0 where False."""
    out = np.zeros(len(mask), dtype=int)
    c = 0
    for i, v in enumerate(mask):
        c = c + 1 if v else 0
        out[i] = c
    return out


# ------------------------------------------------------------------ data
def panel():
    raw = _vix_data.load()
    raw = raw[raw.spx.notna()]              # 2 holiday rows where VIX printed but SPX did not
    d = _vix_data.add_features(raw)
    # custom band grid (add_features only ships n=10,20 / k=1.5,2.0,2.5)
    for n in (5, 10, 15, 20, 30):
        basis = d.vix.rolling(n).mean()
        sd = d.vix.rolling(n).std(ddof=0)
        for k in (1.0, 1.5, 2.0, 2.5):
            lo = basis - k * sd
            d[f"L{n}_{k}"] = (d.vix < lo).where(lo.notna(), False).astype(bool)
    d = d.iloc[30:].copy()                  # every band defined -> all cells share one sample
    return d


def main():
    d = panel()
    N = len(d)
    print(f"sample {d.index[0].date()} -> {d.index[-1].date()}   {N:,} sessions "
          f"(bands all defined; 2 SPX-missing holiday rows dropped)")
    for h in HZ:
        v = d[f"g{h}"].notna()
        print(f"   baseline g{h}: mean {d[f'g{h}'][v].mean()*100:+.3f}%  "
              f"median {d[f'g{h}'][v].median()*100:+.3f}%  win {(d[f'g{h}'][v]>0).mean()*100:.1f}%  n={int(v.sum())}")

    # ---- validate the exhaustive rotation test against the repo's Monte-Carlo version
    m = d["bb10_2.0_below"].values
    f5 = d["g5"].values
    obs = f5[m & ~np.isnan(f5)].mean()
    print(f"\nrotation-test cross-check on BB(10,2.0) below / g5: "
          f"exhaustive p={rot_p(m, f5, obs):.4f}   repo MC(5000) p={rot_p_loop(m, f5, obs):.4f}")

    # ================================================================ 1. ROBUSTNESS SURFACE
    print("\n" + "=" * 118)
    print("1. ROBUSTNESS SURFACE — VIX closes BELOW the lower band, excess over same-sample baseline")
    print("=" * 118)
    print(f"{'band':<14}{'n days':>8}{'epis':>6}{'freq':>7}   " +
          "".join(f"{'D'+str(h)+' exc   p':>22}" for h in HZ))
    print("-" * 118)
    grid = {}
    for n in (5, 10, 15, 20, 30):
        for k in (1.0, 1.5, 2.0, 2.5):
            mask = d[f"L{n}_{k}"].values
            cnt = int(mask.sum())
            row = f"BB({n:>2},{k:.1f})".ljust(14) + f"{cnt:>8}{episodes(mask):>6}{cnt/N*100:>6.1f}%   "
            if cnt < 25:
                print(row + f"  n<25 — inconclusive")
                continue
            cells = []
            for h in HZ:
                c = cell(d, mask, h)
                grid[(n, k, h)] = c
                if c["n"] < 25:
                    cells.append(f"{'n<25':>22}")
                else:
                    cells.append(f"{c['exc']*100:+7.2f}%  p={c['p']:.3f} {stars(c['p'])}".rjust(22))
            print(row + "".join(cells))
    print("\n  plateau check — count of (n,k) cells with excess < 0 and p < .05, by horizon:")
    for h in HZ:
        neg = [(n, k) for (n, k, hh), c in grid.items()
               if hh == h and c["n"] >= 25 and c["exc"] < 0 and c["p"] < 0.05]
        allc = [(n, k) for (n, k, hh), c in grid.items() if hh == h and c["n"] >= 25]
        negs = [c["exc"] for (n, k, hh), c in grid.items() if hh == h and c["n"] >= 25 and c["exc"] < 0]
        print(f"    D{h:<3} {len(neg):>2}/{len(allc)} cells sig-negative; "
              f"{sum(1 for (n,k,hh),c in grid.items() if hh==h and c['n']>=25 and c['exc']<0)}/{len(allc)} "
              f"negative at all; median excess across cells "
              f"{np.median([c['exc'] for (n,k,hh),c in grid.items() if hh==h and c['n']>=25])*100:+.2f}%")

    # era stability of the plateau centre and of a couple of neighbours
    print("\n  era stability (D5 excess vs same-era baseline):")
    for n, k in [(10, 2.0), (20, 2.0), (10, 1.5), (20, 1.5), (30, 2.0), (5, 2.0), (15, 2.0)]:
        mask = d[f"L{n}_{k}"].values
        fwd = d["g5"]
        ok = fwd.notna().values
        parts = []
        for nm, y0, y1 in ERAS:
            sub = (d.index.year >= y0) & (d.index.year <= y1)
            s = mask & sub & ok
            if s.sum() < 25:
                parts.append(f"{nm} n={int(s.sum())} incon.")
            else:
                b = fwd[sub & ok].mean()
                parts.append(f"{nm} n={int(s.sum())} {(fwd[s].mean()-b)*100:+.2f}%")
        print(f"    BB({n},{k}): " + " | ".join(parts))

    # ================================================================ 2. PERSISTENCE
    print("\n" + "=" * 118)
    print("2. PERSISTENCE — is the edge in the ENTRY (first day below) or in the STATE (any day below)?")
    print("=" * 118)
    for n, k in [(10, 2.0), (20, 2.0)]:
        mask = d[f"L{n}_{k}"].values
        sd_ = streak_day(mask)
        print(f"\n  BB({n},{k})   {int(mask.sum())} days in {episodes(mask)} episodes   "
              f"mean episode length {mask.sum()/max(episodes(mask),1):.1f} days")
        print(f"  {'group':<32}{'n':>7}   " + "".join(f"{'D'+str(h)+' exc  p':>21}" for h in HZ))
        groups = [("day 1 only (ENTRY)", sd_ == 1),
                  ("day 2", sd_ == 2), ("day 3", sd_ == 3), ("day 4", sd_ == 4),
                  ("day 5+", sd_ >= 5),
                  ("day >=2 (continuation)", sd_ >= 2),
                  ("ALL days below (STATE)", mask)]
        for lbl, g in groups:
            cnt = int(g.sum())
            if cnt < 25:
                print(f"  {lbl:<32}{cnt:>7}   n<25 — inconclusive")
                continue
            cells = []
            for h in HZ:
                c = cell(d, g, h)
                cells.append((f"{c['exc']*100:+7.2f}% p={c['p']:.3f}{stars(c['p'])}"
                              if c["n"] >= 25 else "n<25").rjust(21))
            print(f"  {lbl:<32}{cnt:>7}   " + "".join(cells))

    # ================================================================ 3. EXIT
    print("\n" + "=" * 118)
    print("3. EXIT — after the episode ends (first close back inside the band), does the warning expire?")
    print("=" * 118)
    for n, k in [(10, 2.0), (20, 2.0)]:
        below = d[f"L{n}_{k}"].values
        exit_lo = below & False
        exit_lo = np.r_[False, below[:-1]] & ~below
        after = {}
        for lag in range(1, 11):
            after[lag] = np.r_[[False] * lag, exit_lo[:-lag]] & ~below
        print(f"\n  BB({n},{k})")
        print(f"  {'event':<32}{'n':>7}   " + "".join(f"{'D'+str(h)+' exc  p':>21}" for h in HZ))
        rows = [("exit day itself (t=0)", exit_lo)]
        for lag in (1, 2, 3, 5, 10):
            rows.append((f"t+{lag} after exit (still out)", after[lag]))
        rows.append(("1-5 days after exit (window)",
                     np.any([after[l] for l in range(1, 6)], axis=0)))
        rows.append(("6-10 days after exit (window)",
                     np.any([after[l] for l in range(6, 11)], axis=0)))
        rows.append(("never below in prior 21d (clean)",
                     ~pd.Series(below).rolling(21, min_periods=1).max().astype(bool).values))
        for lbl, g in rows:
            cnt = int(g.sum())
            if cnt < 25:
                print(f"  {lbl:<32}{cnt:>7}   n<25 — inconclusive")
                continue
            cells = []
            for h in HZ:
                c = cell(d, g, h)
                cells.append((f"{c['exc']*100:+7.2f}% p={c['p']:.3f}{stars(c['p'])}"
                              if c["n"] >= 25 else "n<25").rjust(21))
            print(f"  {lbl:<32}{cnt:>7}   " + "".join(cells))

    # ================================================================ 4. TREND INTERACTION
    print("\n" + "=" * 118)
    print("4. TREND INTERACTION — SPX above / below its own 200d SMA")
    print("=" * 118)
    sma200 = d.spx.rolling(200).mean()
    up = (d.spx > sma200).values & sma200.notna().values
    dn = (d.spx <= sma200).values & sma200.notna().values
    print(f"  regime day counts: above200 {int(up.sum())}  below200 {int(dn.sum())}  "
          f"undefined {int((~up & ~dn).sum())}")
    for h in HZ:
        fwd = d[f"g{h}"].values
        v = ~np.isnan(fwd)
        print(f"    regime baselines g{h}:  above200 {fwd[up & v].mean()*100:+.3f}% (n={int((up&v).sum())})"
              f"   below200 {fwd[dn & v].mean()*100:+.3f}% (n={int((dn&v).sum())})")
    for n, k in [(10, 2.0), (20, 2.0), (10, 1.5)]:
        mask = d[f"L{n}_{k}"].values
        print(f"\n  BB({n},{k}) below lower band, split by trend "
              f"(excess is vs the SAME-REGIME baseline; p rotates the mask then re-intersects the regime)")
        print(f"  {'regime':<32}{'n':>7}   " + "".join(f"{'D'+str(h)+' exc  p':>21}" for h in HZ))
        for lbl, reg in [("SPX > 200d SMA", up), ("SPX <= 200d SMA", dn)]:
            g = mask & reg
            cnt = int(g.sum())
            if cnt < 25:
                print(f"  {lbl:<32}{cnt:>7}   n<25 — inconclusive")
                continue
            cells = []
            for h in HZ:
                fwd = d[f"g{h}"].values
                v = ~np.isnan(fwd)
                sel = g & v
                if sel.sum() < 25:
                    cells.append("n<25".rjust(21))
                    continue
                obs = fwd[sel].mean()
                base = fwd[reg & v].mean()
                # rotate the raw mask, then re-intersect with the (fixed) regime
                nn = len(mask)
                y = np.where(v, np.nan_to_num(fwd), 0.0) * reg
                w = v.astype(float) * reg
                M = np.fft.rfft(mask.astype(float), nn)
                num = np.fft.irfft(np.conj(M) * np.fft.rfft(y, nn), nn)[1:]
                den = np.round(np.fft.irfft(np.conj(M) * np.fft.rfft(w, nn), nn)[1:])
                null = np.where(den >= 25, num / np.where(den > 0, den, 1), np.nan)
                null = null[~np.isnan(null)]
                b0 = null.mean()
                p = float((np.abs(null - b0) >= abs(obs - b0)).mean())
                cells.append(f"{(obs-base)*100:+7.2f}% p={p:.3f}{stars(p)}".rjust(21))
            print(f"  {lbl:<32}{cnt:>7}   " + "".join(cells))

    # ================================================================ 5. PRACTICAL DE-RISK RULE
    print("\n" + "=" * 118)
    print("5. DE-RISK RULE — flat (cash, 0%) for K days starting at the close AFTER the signal")
    print("=" * 118)
    ret = (d.spx.shift(-1) / d.spx - 1.0).fillna(0.0).values   # r_t = close t -> close t+1

    def derisk(mask, K):
        out = np.zeros(len(d), dtype=bool)
        for i in np.flatnonzero(mask):
            out[i + 1: i + 1 + K] = True     # signal at i, act at close i+1
        return out

    def perf(r):
        n = len(r)
        cagr = (1 + r).prod() ** (252 / n) - 1
        vol = r.std(ddof=0) * math.sqrt(252)
        eq = (1 + r).cumprod()
        dd = (eq / np.maximum.accumulate(eq) - 1).min()
        return cagr, vol, (cagr / vol if vol else np.nan), dd

    for w_lbl, wmask in [("FULL 1990-2026", np.ones(len(d), bool)),
                         ("2010-2026", np.asarray(d.index.year >= 2010))]:
        rr = ret[wmask]
        bc, bv, bs, bd = perf(rr)
        print(f"\n  {w_lbl}   n={len(rr):,} sessions")
        print(f"  {'rule':<34}{'CAGR':>9}{'vol':>8}{'Sharpe':>8}{'maxDD':>9}{'days out':>10}"
              f"{'%out':>7}{'avoided ret/day':>18}")
        print(f"  {'buy & hold':<34}{bc*100:>8.2f}%{bv*100:>7.1f}%{bs:>8.2f}{bd*100:>8.1f}%"
              f"{0:>10}{0.0:>6.1f}%{'-':>18}")
        for n, k in [(10, 2.0), (20, 2.0)]:
            for K in (3, 5, 10):
                mask = d[f"L{n}_{k}"].values
                out = derisk(mask, K)
                r2 = np.where(out, 0.0, ret)[wmask]
                o = out[wmask]
                c2, v2, s2, dd2 = perf(r2)
                avoided = rr[o].mean() * 100 if o.sum() else np.nan
                print(f"  {f'flat {K}d after BB({n},{k}) below':<34}{c2*100:>8.2f}%{v2*100:>7.1f}%"
                      f"{s2:>8.2f}{dd2*100:>8.1f}%{int(o.sum()):>10}{o.mean()*100:>6.1f}%"
                      f"{avoided:>17.4f}%")
        print(f"  (buy-hold average day in this window: {rr.mean()*100:+.4f}%)")

    print("\n  same rule but PAID CASH at 0% assumed above. Sitting out only helps if the avoided")
    print("  days average worse than the buy-hold day shown on the last line of each block.")

    # by-era check of the de-risk rule's core claim
    print("\n  avoided-days mean daily return vs that era's baseline (BB(10,2.0), K=5):")
    out = derisk(d["L10_2.0"].values, 5)
    for nm, y0, y1 in ERAS:
        sub = np.asarray((d.index.year >= y0) & (d.index.year <= y1))
        o = out & sub
        if o.sum() < 25:
            print(f"    {nm}: n={int(o.sum())} — inconclusive")
            continue
        print(f"    {nm}: avoided {ret[o].mean()*100:+.4f}%/day vs era baseline "
              f"{ret[sub].mean()*100:+.4f}%/day  (n={int(o.sum())} days out of {int(sub.sum())})")

    # ================================================================ 6. MAGNITUDE VS FREQUENCY
    print("\n" + "=" * 118)
    print("6. MAGNITUDE vs FREQUENCY — full g5 distribution, conditional vs unconditional")
    print("=" * 118)
    fwd = d["g5"].values
    v = ~np.isnan(fwd)
    for n, k in [(10, 2.0), (20, 2.0)]:
        mask = d[f"L{n}_{k}"].values & v
        a, b = fwd[mask], fwd[v]
        qs = [10, 25, 50, 75, 90]
        print(f"\n  BB({n},{k}) below lower band   n={int(mask.sum())} vs baseline n={int(v.sum())}")
        print(f"  {'':<14}{'mean':>9}{'p10':>9}{'p25':>9}{'median':>9}{'p75':>9}{'p90':>9}"
              f"{'win%':>8}{'sd':>8}{'min':>9}")
        print(f"  {'signal':<14}{a.mean()*100:>8.2f}%" +
              "".join(f"{np.percentile(a, q)*100:>8.2f}%" for q in qs) +
              f"{(a>0).mean()*100:>7.1f}%{a.std(ddof=0)*100:>7.2f}%{a.min()*100:>8.2f}%")
        print(f"  {'baseline':<14}{b.mean()*100:>8.2f}%" +
              "".join(f"{np.percentile(b, q)*100:>8.2f}%" for q in qs) +
              f"{(b>0).mean()*100:>7.1f}%{b.std(ddof=0)*100:>7.2f}%{b.min()*100:>8.2f}%")
        print(f"  {'difference':<14}{(a.mean()-b.mean())*100:>8.2f}%" +
              "".join(f"{(np.percentile(a,q)-np.percentile(b,q))*100:>8.2f}%" for q in qs) +
              f"{((a>0).mean()-(b>0).mean())*100:>7.1f}%")
        # rotation p on the MEDIAN and on the WIN RATE, not just the mean
        nn = len(d)
        for stat_name, transform in [("median", None), ("win rate", None)]:
            pass
        # median rotation (loop; medians are not FFT-able)
        med_null = []
        win_null = []
        offs = RNG.integers(1, nn, size=3000)
        base_mask = d[f"L{n}_{k}"].values
        for off in offs:
            mm = np.roll(base_mask, off) & v
            if mm.sum() >= 25:
                med_null.append(np.median(fwd[mm]))
                win_null.append((fwd[mm] > 0).mean())
        med_null, win_null = np.array(med_null), np.array(win_null)
        for nm_, obs_, nul_ in [("median", np.median(a), med_null), ("win rate", (a > 0).mean(), win_null)]:
            b0 = nul_.mean()
            p = float((np.abs(nul_ - b0) >= abs(obs_ - b0)).mean())
            print(f"    rotation p on {nm_:<9} obs {obs_*100:+.2f}%  null-mean {b0*100:+.2f}%  p={p:.3f} "
                  f"({len(nul_)} valid rolls)")
        # tail decomposition: how much of the mean gap survives trimming the worst outcomes
        for trim in (0.01, 0.05, 0.10):
            lo_a, lo_b = np.quantile(a, trim), np.quantile(b, trim)
            at, bt = a[a > lo_a], b[b > lo_b]
            print(f"    drop worst {trim*100:>4.0f}% of each: excess "
                  f"{(at.mean()-bt.mean())*100:+.2f}%  (raw {(a.mean()-b.mean())*100:+.2f}%)")
        # contribution of the single worst episodes
        ep_id = np.cumsum(base_mask & ~np.r_[False, base_mask[:-1]])
        dfe = pd.DataFrame({"ep": ep_id[base_mask & v], "g5": fwd[base_mask & v]})
        agg = dfe.groupby("ep").g5.agg(["mean", "count"]).sort_values("mean")
        tot = a.sum()
        print(f"    {len(agg)} distinct episodes; 3 worst by mean g5:")
        for ep, row in agg.head(3).iterrows():
            dts = d.index[base_mask & v][ep_id[base_mask & v] == ep]
            print(f"      ep{int(ep)} {dts[0].date()}..{dts[-1].date()}  n={int(row['count'])}  "
                  f"mean g5 {row['mean']*100:+.2f}%")
        # excess after removing the single worst episode
        worst = agg.index[0]
        keep = (ep_id[base_mask & v] != worst)
        print(f"    excess excluding that single worst episode: "
              f"{(a[keep].mean()-b.mean())*100:+.2f}%  (n={int(keep.sum())})")

    print("\ndone.")


if __name__ == "__main__":
    main()
