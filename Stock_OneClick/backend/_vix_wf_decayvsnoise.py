"""
_vix_wf_decayvsnoise.py — Is the BULLISH half of the VIX-vs-MA10 / VIX-Bollinger folklore dead,
and is the BEARISH half still alive?

Prior work (_vix_ma10_bb_research.py) found every bullish variant flips NEGATIVE in the 2020s.
This script asks whether that is DECAY (a real regime change) or NOISE (a small, crisis-clustered
sample wandering around zero).

  1. rolling 10-year trailing window, sampled annually -> D5 excess trajectory
  2. formal split test 1990-2014 vs 2015-2026: rotation test inside each subperiod, plus a
     circular-block bootstrap of the DIFFERENCE of the two subperiod excesses, plus a
     rotation test of the difference itself
  3. 2020s excluding COVID (2020-02-01..2020-06-30) — does the negative sign survive?
  4. mechanism: per-era firing frequency and per-era unconditional baseline (a signal can look
     "decayed" purely because the drift it is measured against went up)

Outcome is ALWAYS g5 (SPX forward 5d, entry at the NEXT close — VIX settles 16:15 ET).
Everything is reported as EXCESS over the SAME-SAMPLE unconditional mean.

Run: ../../vcp_env/bin/python _vix_wf_decayvsnoise.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _vix_data

RNG = np.random.default_rng(20260910)
N_ROT = 5000
N_BOOT = 5000
H = 5                      # D5 is the horizon the prior study settled on
MIN_N = 25                 # below this -> inconclusive, do not interpret

SIGNALS = [
    ("stretch>=+10%", lambda d: (d.stretch >= 0.10).values),
    ("stretch>=+20%", lambda d: (d.stretch >= 0.20).values),
    ("BB(10,2) above upper", lambda d: d["bb10_2.0_above"].values.astype(bool)),
    ("BB(10,2) re-entry", lambda d: d["bb10_2.0_reentry"].values.astype(bool)),
    ("BB(10,2) below lower", lambda d: d["bb10_2.0_below"].values.astype(bool)),
]
BEARISH = {"BB(10,2) below lower"}


# ------------------------------------------------------------------ stats
def _stars(p):
    if p is None or np.isnan(p):
        return "   "
    return "***" if p < 0.01 else ("** " if p < 0.05 else ("*  " if p < 0.10 else "   "))


def rot_null(mask: np.ndarray, fwd: np.ndarray, n_rot: int = N_ROT) -> np.ndarray:
    """Circular rotation null: roll the boolean signal mask by a random offset and recompute the
    conditional mean. Identical in definition to rotation_pvalue() in _vix_ma10_bb_research.py
    (np.roll(mask, off) & valid -> fwd[m].mean()), just vectorised so it can be run hundreds of
    times. Preserves both the autocorrelation of overlapping forward returns and the burstiness
    of the signal, which a t-test does not."""
    n = len(mask)
    idx0 = np.flatnonzero(mask)
    if idx0.size == 0:
        return np.array([])
    valid = ~np.isnan(fwd)
    fv = np.where(valid, fwd, 0.0)
    vv = valid.astype(np.float64)
    offs = RNG.integers(1, n, size=n_rot)
    j = (idx0[:, None] + offs[None, :]) % n          # rolled positions, shape (n_sig, n_rot)
    cnt = vv[j].sum(axis=0)
    s = fv[j].sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        null = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    return null[~np.isnan(null)]


def rot_p(mask: np.ndarray, fwd: np.ndarray, observed: float) -> float:
    null = rot_null(mask, fwd)
    if not len(null):
        return float("nan")
    base = null.mean()
    return float((np.abs(null - base) >= abs(observed - base)).mean())


def excess(mask: np.ndarray, fwd: np.ndarray):
    valid = ~np.isnan(fwd)
    sel = mask & valid
    n = int(sel.sum())
    if n == 0:
        return dict(n=0, cond=np.nan, base=np.nan, exc=np.nan, win=np.nan, win_base=np.nan)
    cond = fwd[sel].mean()
    base = fwd[valid].mean()
    return dict(n=n, cond=cond, base=base, exc=cond - base,
                win=float((fwd[sel] > 0).mean()), win_base=float((fwd[valid] > 0).mean()))


def block_boot_excess(mask: np.ndarray, fwd: np.ndarray, L: int, n_boot: int) -> np.ndarray:
    """Circular moving-block bootstrap of the excess (cond mean - same-resample baseline).
    Blocks keep the signal clustering and the overlap of forward windows intact; L is in sessions."""
    n = len(mask)
    valid = ~np.isnan(fwd)
    fv = np.where(valid, fwd, 0.0)
    vv = valid.astype(np.float64)
    mv = (mask & valid).astype(np.float64)
    mfv = np.where(mask & valid, fv, 0.0)
    nb = int(np.ceil(n / L))
    out = np.empty(n_boot)
    starts = RNG.integers(0, n, size=(n_boot, nb))
    off = np.arange(L)
    for b in range(n_boot):
        j = (starts[b][:, None] + off[None, :]).ravel()[:n] % n
        cs, cn = mfv[j].sum(), mv[j].sum()
        bs, bn = fv[j].sum(), vv[j].sum()
        out[b] = (cs / cn - bs / bn) if cn > 0 and bn > 0 else np.nan
    return out[~np.isnan(out)]


# ------------------------------------------------------------------ main
def main():
    d = _vix_data.add_features(_vix_data.load())
    fwd_all = d[f"g{H}"].values
    yrs = d.index.year.values
    print(f"Panel: {d.index[0].date()} -> {d.index[-1].date()}  ({len(d):,} sessions)")
    print(f"g{H} valid rows: {int((~np.isnan(fwd_all)).sum()):,}   "
          f"unconditional mean {np.nanmean(fwd_all)*100:+.3f}%  "
          f"win {np.nanmean(fwd_all > 0)*100:.1f}%")

    masks = {name: fn(d) for name, fn in SIGNALS}
    for name, m in masks.items():
        r = excess(m, fwd_all)
        p = rot_p(m, fwd_all, r["cond"])
        print(f"  full sample  {name:<22} n={r['n']:>5}  exc {r['exc']*100:+.3f}%  "
              f"base {r['base']*100:+.3f}%  p={p:.4f} {_stars(p)}")

    # ---------------------------------------------------------------- 1. rolling 10y
    print(f"\n{'='*118}\n1. ROLLING 10-YEAR TRAILING WINDOW, sampled annually — D{H} excess (%) and "
          f"signal n inside the window\n{'='*118}")
    hdr = f"{'win end':<9}{'sessions':>9}{'base%':>8}  "
    for name, _ in SIGNALS:
        hdr += f"{name[:18]:>22}"
    print(hdr)
    print(f"{'':<9}{'':>9}{'':>8}  " + "".join(f"{'exc%  (n)':>22}" for _ in SIGNALS))
    print("-" * 118)
    traj = {name: [] for name, _ in SIGNALS}
    y0, y1 = int(yrs.min()), int(yrs.max())
    for Y in range(y0 + 9, y1 + 1):
        w = (yrs >= Y - 9) & (yrs <= Y)
        fw = np.where(w, fwd_all, np.nan)
        valid = ~np.isnan(fw)
        base = np.nanmean(fw)
        line = f"{Y:<9}{int(valid.sum()):>9}{base*100:>7.3f}  "
        for name, _ in SIGNALS:
            m = masks[name] & w
            r = excess(m, fw)
            traj[name].append((Y, r["n"], r["exc"]))
            if r["n"] < MIN_N:
                line += f"{'n<25  (' + str(r['n']) + ')':>22}"
            else:
                line += f"{r['exc']*100:>+13.3f}  ({r['n']:>3})"
        print(line)

    print("\n  sign-consistency of the trailing-10y excess across the annual samples above:")
    for name, _ in SIGNALS:
        ok = [(Y, e) for Y, n, e in traj[name] if n >= MIN_N]
        if not ok:
            continue
        es = np.array([e for _, e in ok])
        pos = (es > 0).mean()
        last5 = [f"{Y}:{e*100:+.2f}" for Y, e in ok[-5:]]
        print(f"    {name:<22} windows={len(ok):>3}  frac>0={pos*100:>5.1f}%  "
              f"min {es.min()*100:+.2f}%  max {es.max()*100:+.2f}%  last5 [{', '.join(last5)}]")

    # ---------------------------------------------------------------- 2. split test
    print(f"\n{'='*118}\n2. FORMAL SPLIT — 1990-2014 vs 2015-2026 (D{H} excess), rotation test INSIDE "
          f"each subperiod\n{'='*118}")
    early = yrs <= 2014
    late = yrs >= 2015
    print(f"  early window {d.index[early][0].date()} -> {d.index[early][-1].date()} "
          f"({int(early.sum()):,} sessions)   "
          f"late window {d.index[late][0].date()} -> {d.index[late][-1].date()} "
          f"({int(late.sum()):,} sessions)")
    print(f"\n{'signal':<22}{'n_e':>6}{'exc_e%':>9}{'p_e':>8}   {'n_l':>6}{'exc_l%':>9}{'p_l':>8}"
          f"   {'diff%':>9}{'p_boot':>9}{'p_rot':>8}{'boot 95% CI':>22}")
    print("-" * 118)
    split_res = {}
    for name, _ in SIGNALS:
        m = masks[name]
        row = {}
        for tag, sub in (("e", early), ("l", late)):
            fs = np.where(sub, fwd_all, np.nan)
            r = excess(m & sub, fs)
            # rotation restricted to the subperiod: roll the mask only within that slice
            msub, fsub = m[sub], fwd_all[sub]
            p = rot_p(msub, fsub, r["cond"]) if r["n"] >= MIN_N else float("nan")
            row[tag] = (r, p)
        (re_, pe), (rl, pl) = row["e"], row["l"]
        diff = rl["exc"] - re_["exc"]

        # --- bootstrap the DIFFERENCE of the two subperiod excesses (independent block resamples)
        p_boot, ci = float("nan"), ""
        if re_["n"] >= MIN_N and rl["n"] >= MIN_N:
            L = 63          # ~1 quarter: longer than the 5d overlap and than typical vol clusters
            be = block_boot_excess(m[early], fwd_all[early], L, N_BOOT)
            bl = block_boot_excess(m[late], fwd_all[late], L, N_BOOT)
            k = min(len(be), len(bl))
            dd = bl[:k] - be[:k]
            p_boot = float(2 * min((dd <= 0).mean(), (dd >= 0).mean()))
            ci = f"[{np.percentile(dd,2.5)*100:+.2f}, {np.percentile(dd,97.5)*100:+.2f}]"

        # --- rotation test of the DIFFERENCE: roll the FULL-sample mask, recompute both
        #     subperiod excesses each time. Null = "when the signal fires carries no information
        #     about which era it works in."
        p_rotd = float("nan")
        if re_["n"] >= MIN_N and rl["n"] >= MIN_N:
            n = len(m)
            idx0 = np.flatnonzero(m)
            valid = ~np.isnan(fwd_all)
            fv = np.where(valid, fwd_all, 0.0)
            offs = RNG.integers(1, n, size=N_ROT)
            j = (idx0[:, None] + offs[None, :]) % n
            nulls = {}
            for tag, sub in (("e", early), ("l", late)):
                subf = (sub & valid).astype(np.float64)
                subfv = np.where(sub & valid, fv, 0.0)
                cnt = subf[j].sum(axis=0)
                s = subfv[j].sum(axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    cond = np.where(cnt >= MIN_N, s / np.maximum(cnt, 1), np.nan)
                nulls[tag] = cond - np.nanmean(np.where(sub, fwd_all, np.nan))
            nd = nulls["l"] - nulls["e"]
            nd = nd[~np.isnan(nd)]
            if len(nd):
                p_rotd = float((np.abs(nd - nd.mean()) >= abs(diff - nd.mean())).mean())

        split_res[name] = dict(early=re_, late=rl, pe=pe, pl=pl, diff=diff,
                               p_boot=p_boot, p_rot=p_rotd)
        print(f"{name:<22}{re_['n']:>6}{re_['exc']*100:>+9.3f}{pe:>8.3f}   "
              f"{rl['n']:>6}{rl['exc']*100:>+9.3f}{pl:>8.3f}   "
              f"{diff*100:>+9.3f}{p_boot:>9.3f}{p_rotd:>8.3f}{ci:>22}")
    print("\n  exc_e / exc_l = excess over the SAME-SUBPERIOD unconditional mean. p_e / p_l are")
    print("  rotation tests run inside that subperiod only. p_boot = circular block bootstrap")
    print("  (L=63 sessions) of the difference; p_rot = rotation test of the difference itself.")

    # block-length sensitivity for the bootstrap difference
    print("\n  block-length sensitivity of p_boot (L = 21 / 63 / 126 sessions):")
    for name, _ in SIGNALS:
        m = masks[name]
        if split_res[name]["early"]["n"] < MIN_N or split_res[name]["late"]["n"] < MIN_N:
            continue
        ps = []
        for L in (21, 63, 126):
            be = block_boot_excess(m[early], fwd_all[early], L, 2000)
            bl = block_boot_excess(m[late], fwd_all[late], L, 2000)
            k = min(len(be), len(bl))
            dd = bl[:k] - be[:k]
            ps.append(2 * min((dd <= 0).mean(), (dd >= 0).mean()))
        print(f"    {name:<22} " + "  ".join(f"L={L}: p={p:.3f}" for L, p in zip((21, 63, 126), ps)))

    # ---------------------------------------------------------------- 3. 2020s ex-COVID
    print(f"\n{'='*118}\n3. THE 2020s WITHOUT COVID — drop 2020-02-01..2020-06-30\n{'='*118}")
    covid = (d.index >= "2020-02-01") & (d.index <= "2020-06-30")
    print(f"  COVID block = {int(covid.sum())} sessions "
          f"({d.index[covid][0].date()} -> {d.index[covid][-1].date()})")
    periods = [
        ("2020-2026 (all)", yrs >= 2020),
        ("2020-2026 ex-COVID", (yrs >= 2020) & ~covid),
        ("2015-2019", (yrs >= 2015) & (yrs <= 2019)),
        ("2015-2026 ex-COVID", (yrs >= 2015) & ~covid),
        ("2021-2026 (no 2020 at all)", yrs >= 2021),
    ]
    print(f"\n{'period':<28}{'signal':<22}{'n':>6}{'cond%':>9}{'base%':>9}{'exc%':>9}{'p_rot':>8}"
          f"{'win%':>8}{'winbase%':>10}")
    print("-" * 118)
    for pname, sub in periods:
        for name, _ in SIGNALS:
            m = masks[name] & sub
            fs = np.where(sub, fwd_all, np.nan)
            r = excess(m, fs)
            if r["n"] < MIN_N:
                print(f"{pname:<28}{name:<22}{r['n']:>6}   INCONCLUSIVE (n<25)")
                continue
            p = rot_p(masks[name][sub], fwd_all[sub], r["cond"])
            print(f"{pname:<28}{name:<22}{r['n']:>6}{r['cond']*100:>+9.3f}{r['base']*100:>+9.3f}"
                  f"{r['exc']*100:>+9.3f}{p:>8.3f}{r['win']*100:>8.1f}{r['win_base']*100:>10.1f}")
        print()

    # COVID block on its own, for reference (n may be tiny -> flagged)
    print("  the COVID block itself (reference only):")
    for name, _ in SIGNALS:
        r = excess(masks[name] & covid, np.where(covid, fwd_all, np.nan))
        tag = "INCONCLUSIVE (n<25)" if r["n"] < MIN_N else \
              f"cond {r['cond']*100:+.2f}%  base {r['base']*100:+.2f}%  exc {r['exc']*100:+.2f}%"
        print(f"    {name:<22} n={r['n']:>3}  {tag}")

    # ---------------------------------------------------------------- 4. mechanism / base rates
    print(f"\n{'='*118}\n4. MECHANISM — has the BASE RATE changed? firing frequency and the "
          f"unconditional D{H} drift, per era\n{'='*118}")
    eras = [("1990s", 1990, 1999), ("2000s", 2000, 2009), ("2010s", 2010, 2019),
            ("2020s", 2020, 2099), ("2015-2026", 2015, 2099)]
    print(f"{'era':<12}{'sessions':>9}{'base D5%':>10}{'base win%':>11}{'VIX mean':>10}"
          f"{'VIX med':>9}   " + "".join(f"{s[:16]:>18}" for s, _ in SIGNALS))
    print(f"{'':<12}{'':>9}{'':>10}{'':>11}{'':>10}{'':>9}   "
          + "".join(f"{'fires% (n)':>18}" for _ in SIGNALS))
    print("-" * 130)
    for ename, a, b in eras:
        sub = (yrs >= a) & (yrs <= b)
        fs = np.where(sub, fwd_all, np.nan)
        v = ~np.isnan(fs)
        line = (f"{ename:<12}{int(v.sum()):>9}{np.nanmean(fs)*100:>10.3f}"
                f"{np.nanmean(fs[v] > 0)*100:>11.1f}{d.vix.values[sub].mean():>10.2f}"
                f"{np.median(d.vix.values[sub]):>9.2f}   ")
        for name, _ in SIGNALS:
            n = int((masks[name] & sub & v).sum())
            line += f"{n/max(int(v.sum()),1)*100:>12.1f} ({n:>3})"
        print(line)

    print(f"\n{'era':<12}" + "".join(f"{s[:16]:>20}" for s, _ in SIGNALS))
    print(f"{'':<12}" + "".join(f"{'cond% / exc%':>20}" for _ in SIGNALS))
    print("-" * 112)
    for ename, a, b in eras:
        sub = (yrs >= a) & (yrs <= b)
        fs = np.where(sub, fwd_all, np.nan)
        line = f"{ename:<12}"
        for name, _ in SIGNALS:
            r = excess(masks[name] & sub, fs)
            if r["n"] < MIN_N:
                line += f"{'n<25':>20}"
            else:
                line += f"{r['cond']*100:>+9.3f} /{r['exc']*100:>+8.3f}"
        print(line)

    # decomposition: how much of the change in the CONDITIONAL mean is just the baseline moving?
    print("\n  decomposition, 1990-2014 -> 2015-2026 (percentage points of D5):")
    fe = np.where(early, fwd_all, np.nan)
    fl = np.where(late, fwd_all, np.nan)
    db = np.nanmean(fl) - np.nanmean(fe)
    print(f"    unconditional baseline moved {db*100:+.3f} pp "
          f"({np.nanmean(fe)*100:+.3f}% -> {np.nanmean(fl)*100:+.3f}%)")
    for name, _ in SIGNALS:
        re_ = excess(masks[name] & early, fe)
        rl = excess(masks[name] & late, fl)
        if re_["n"] < MIN_N or rl["n"] < MIN_N:
            print(f"    {name:<22} INCONCLUSIVE")
            continue
        dc = rl["cond"] - re_["cond"]
        print(f"    {name:<22} conditional moved {dc*100:+.3f} pp, of which baseline "
              f"{db*100:+.3f} pp -> genuine edge change {(dc-db)*100:+.3f} pp")

    # ---------------------------------------------------------------- 5. power check
    print(f"\n{'='*118}\n5. POWER CHECK — with the late-sample n, what excess could we even have "
          f"detected?\n{'='*118}")
    for name, _ in SIGNALS:
        m, sub = masks[name], late
        r = excess(m & sub, np.where(sub, fwd_all, np.nan))
        if r["n"] < MIN_N:
            print(f"  {name:<22} INCONCLUSIVE (n={r['n']})")
            continue
        null = rot_null(m[sub], fwd_all[sub])
        sd = null.std()
        print(f"  {name:<22} n={r['n']:>4}  rotation-null sd of the conditional mean = "
              f"{sd*100:.3f}pp -> 2-sigma detectable excess ~{2*sd*100:.2f}pp; "
              f"observed excess {r['exc']*100:+.3f}pp ({abs(r['exc'])/sd:.2f} sigma)")
        e_full = excess(m, fwd_all)["exc"]
        print(f"{'':<24}full-sample excess {e_full*100:+.3f}pp would be "
              f"{abs(e_full)/sd:.2f} sigma at this n — "
              f"{'DETECTABLE' if abs(e_full)/sd >= 2 else 'NOT detectable'} in the late window")

    print(f"\n{'='*118}\nCURRENT READING ({d.index[-1].date()})\n{'='*118}")
    last = d.iloc[-1]
    print(f"  VIX {last.vix:.2f} | MA10 {last.ma10:.2f} | stretch {last.stretch*100:+.1f}% | "
          f"BB(10,2) {last['bb10_2.0_lo']:.2f}..{last['bb10_2.0_up']:.2f} "
          f"(above={bool(last['bb10_2.0_above'])}, below={bool(last['bb10_2.0_below'])})")


if __name__ == "__main__":
    main()
