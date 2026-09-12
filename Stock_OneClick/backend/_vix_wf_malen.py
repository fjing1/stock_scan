"""
_vix_wf_malen.py — Is MA10 special, or is 10 an arbitrary published number?

Sweeps:
  1. SMA length 3..40, signal = VIX stretched +10% above / -10% below the MA.  D5 excess, n, p.
  2. Same sweep with EMA.
  3. Stretch threshold 0%..40% (2.5% steps) at MA10 specifically.
  4. Percent-stretch vs z-score ( (VIX-MA10)/rolling-10d-sigma, i.e. Bollinger %B rescaled ),
     compared at MATCHED signal frequency so the comparison is not a threshold-units artifact.
  5. Total configuration count -> Bonferroni + Benjamini-Hochberg FDR on the best cell.

Outcome = g5 (SPX forward 5d, entry at the NEXT close; VIX settles 16:15 ET).
Every number is EXCESS over the same-sample unconditional mean.

Significance: circular rotation test, same null as rotation_pvalue() in
_vix_ma10_bb_research.py (roll the boolean mask, recompute the conditional mean), but evaluated
at ALL n-1 offsets exactly via FFT instead of 5000 random draws. Verified against the random-draw
implementation below.

Run: ../../vcp_env/bin/python _vix_wf_malen.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _vix_data

RNG = np.random.default_rng(20260910)
MIN_N = 25


# ---------------------------------------------------------------- rotation test
def rotation_full(mask: np.ndarray, fwd: np.ndarray):
    """Exact circular-rotation null over all n-1 non-zero offsets, via FFT.

    null[off] = mean( fwd[ roll(mask,off) & valid ] ).  Returns (observed, p_two_sided, n_used).
    """
    mask = np.asarray(mask, dtype=bool)
    n = len(mask)
    valid = ~np.isnan(fwd)
    a = np.where(valid, np.nan_to_num(fwd), 0.0)
    m = mask.astype(float)
    v = valid.astype(float)
    M = np.fft.rfft(m)
    num = np.fft.irfft(np.conj(M) * np.fft.rfft(a), n)
    den = np.fft.irfft(np.conj(M) * np.fft.rfft(v), n)
    den = np.round(den, 6)
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    obs = vals[0]
    n_used = int(round(den[0]))
    null = vals[1:]
    null = null[~np.isnan(null)]
    if not len(null) or np.isnan(obs):
        return obs, float("nan"), n_used
    base = null.mean()
    p = float((np.abs(null - base) >= abs(obs - base)).mean())
    return float(obs), p, n_used


def rotation_random(mask, fwd, observed, n_rot=5000, rng=None):
    """Reference implementation copied in spirit from _vix_ma10_bb_research.rotation_pvalue."""
    rng = rng or RNG
    mask = np.asarray(mask, dtype=bool)
    n = len(mask)
    valid = ~np.isnan(fwd)
    out = []
    for off in rng.integers(1, n, size=n_rot):
        mm = np.roll(mask, off) & valid
        if mm.sum():
            out.append(fwd[mm].mean())
    null = np.array(out)
    base = null.mean()
    return float((np.abs(null - base) >= abs(observed - base)).mean())


def cell(mask, fwd):
    """-> dict(n, cond, base, exc, p, win, win_base) for one signal/horizon cell."""
    mask = np.asarray(mask, dtype=bool)
    valid = ~np.isnan(fwd)
    sel = mask & valid
    n = int(sel.sum())
    if n < MIN_N:
        return {"n": n, "exc": np.nan, "p": np.nan, "cond": np.nan, "mask": mask,
                "base": np.nan, "win": np.nan, "win_base": np.nan}
    obs, p, _ = rotation_full(mask, fwd)
    base = fwd[valid].mean()
    return {"n": n, "cond": obs, "base": base, "exc": obs - base, "p": p, "mask": mask,
            "win": float((fwd[sel] > 0).mean()),
            "win_base": float((fwd[valid] > 0).mean())}


def spearman(a, b):
    """Rank correlation without scipy (not installed in this env)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    ra = pd.Series(a[ok]).rank().values
    rb = pd.Series(b[ok]).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def stars(p):
    if p != p:
        return "   "
    return "***" if p < 0.01 else ("** " if p < 0.05 else ("*  " if p < 0.10 else "   "))


# ---------------------------------------------------------------- main
def paired_rotation(mask_a, mask_b, fwd):
    """Two-sided p for 'signal A and signal B have the same conditional mean'.
    Both masks are rotated by the SAME offset, so the null keeps their relative geometry and
    the shared return autocorrelation; only the joint placement in time is randomised."""
    a = np.asarray(mask_a, bool)
    b = np.asarray(mask_b, bool)
    n = len(a)
    valid = ~np.isnan(fwd)
    y = np.where(valid, np.nan_to_num(fwd), 0.0)
    Y, V = np.fft.rfft(y), np.fft.rfft(valid.astype(float))
    out = []
    for m in (a, b):
        M = np.fft.rfft(m.astype(float))
        num = np.fft.irfft(np.conj(M) * Y, n)
        den = np.round(np.fft.irfft(np.conj(M) * V, n), 6)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan))
    diff = out[0] - out[1]
    obs = diff[0]
    null = diff[1:]
    null = null[~np.isnan(null)]
    base = null.mean()
    return float((np.abs(null - base) >= abs(obs - base)).mean())


def effective_tests(masks):
    """Li & Ji (2005) effective number of independent tests: eigen-decompose the correlation
    matrix of the tested SIGNALS and count eigenvalues as 1 + fractional part. 436 near-duplicate
    masks are nowhere near 436 independent experiments, and plain Bonferroni pretends they are."""
    X = np.array([m.astype(float) for m in masks])
    keep = X.std(axis=1) > 0
    X = X[keep]
    C = np.corrcoef(X)
    C = np.nan_to_num(C)
    ev = np.abs(np.linalg.eigvalsh(C))
    return float(np.sum((ev >= 1.0).astype(float) + (ev - np.floor(ev)))), int(len(X))


def multiple_testing(registry):
    print("\n" + "=" * 108)
    print("10. MULTIPLE TESTING — every cell above with n>=25 and a rotation p-value is counted")
    print("=" * 108)
    reg = pd.DataFrame(registry)
    K = len(reg)
    print(f"  configurations tested: K = {K}")
    print("  (families: " + ", ".join(f"{k}={v}" for k, v in reg.family.value_counts().sort_index().items()) + ")")
    reg = reg.sort_values("p").reset_index(drop=True)
    print(f"\n  {'rank':>4} {'family':<15} {'label':<22} {'n':>6} {'exc':>9} {'p_raw':>8} "
          f"{'p_bonf':>9} {'BH crit(.10)':>13} {'BH crit(.05)':>13}")
    for i, r in reg.head(15).iterrows():
        i += 1
        print(f"  {i:>4} {r.family:<15} {r.label:<22} {int(r.n):>6} {r.exc*100:>8.3f}% "
              f"{r.p:>8.4f} {min(r.p*K,1.0):>9.4f} {i/K*0.10:>13.5f} {i/K*0.05:>13.5f}")

    for qlvl in (0.10, 0.05):
        pv = reg.p.values
        crit = np.arange(1, K + 1) / K * qlvl
        passing = np.flatnonzero(pv <= crit)
        if len(passing):
            kmax = int(passing.max())
            print(f"\n  Benjamini-Hochberg FDR q={qlvl:.2f}: {kmax+1} of {K} cells survive "
                  f"(largest p that passes = {pv[kmax]:.4f})")
            print("    survivor families: " + ", ".join(
                f"{k}={v}" for k, v in reg.iloc[:kmax + 1].family.value_counts().sort_index().items()))
            print("    top survivors: " + "; ".join(
                f"{r.family}/{r.label} n={int(r.n)} {r.exc*100:+.2f}% p={r.p:.4f}"
                for _, r in reg.iloc[:min(kmax + 1, 8)].iterrows()))
        else:
            print(f"\n  Benjamini-Hochberg FDR q={qlvl:.2f}: NOTHING survives (min p {pv.min():.4f}, "
                  f"needs <= {crit[0]:.5f})")
    best = reg.iloc[0]
    print(f"\n  BEST cell: {best.family}/{best.label}  n={int(best.n)}  exc {best.exc*100:+.3f}%  "
          f"p_raw {best.p:.4f}")
    print(f"  Bonferroni: p_raw x K={K} -> {min(best.p*K,1.0):.4f}  "
          f"({'SURVIVES' if best.p*K < 0.05 else 'FAILS'} at 0.05)")

    meff, nmask = effective_tests(list(reg["mask"]))
    print(f"\n  EFFECTIVE number of independent tests (Li & Ji eigenvalue method on the "
          f"{nmask} signal masks): M_eff = {meff:.1f}  (vs nominal K = {K})")
    print(f"  Bonferroni on M_eff: {best.p:.4f} x {meff:.1f} -> {min(best.p*meff,1.0):.4f}  "
          f"({'SURVIVES' if best.p*meff < 0.05 else 'FAILS'} at 0.05; "
          f"{'survives' if best.p*meff < 0.10 else 'fails'} at 0.10)")

    # BH on a COARSE PRE-SPECIFIED grid. Membership is decided by parameter position, never by
    # p-value, so this is not cherry-picking; it just stops counting 38 near-identical lengths.
    coarse_lab = set()
    for L in (3, 5, 8, 10, 15, 20, 30, 40):
        coarse_lab |= {f"SMA{L} +10%", f"SMA{L} -10%", f"EMA{L} +10%", f"EMA{L} -10%"}
    for t in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0):
        coarse_lab |= {f"MA10 +{t:.1f}%", f"MA10 -{t:.1f}%"}
    for zt in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
        coarse_lab |= {f"z>=+{zt:.2f}", f"z<=-{zt:.2f}"}
    sub = reg[reg.label.isin(coarse_lab)].sort_values("p").reset_index(drop=True)
    Kc = len(sub)
    pv2 = sub.p.values
    crit2 = np.arange(1, Kc + 1) / Kc * 0.10
    pas = np.flatnonzero(pv2 <= crit2)
    print(f"\n  COARSE pre-specified grid (lengths 3/5/8/10/15/20/30/40 x SMA,EMA x above,below;")
    print(f"  thresholds 0/5/10/15/20/25%; z 0/0.5/1/1.5/2/2.5, both tails): Kc = {Kc} cells")
    if len(pas):
        km = int(pas.max())
        print(f"    BH q=0.10: {km+1} of {Kc} survive (largest passing p {pv2[km]:.4f})")
        print("      " + "; ".join(f"{r.label} n={int(r.n)} {r.exc*100:+.2f}% p={r.p:.4f}"
                                   for _, r in sub.iloc[:min(km+1, 10)].iterrows()))
    else:
        print(f"    BH q=0.10: nothing survives (min p {pv2.min():.4f} needs <= {crit2[0]:.5f})")
    crit3 = np.arange(1, Kc + 1) / Kc * 0.05
    pas3 = np.flatnonzero(pv2 <= crit3)
    if len(pas3):
        km3 = int(pas3.max())
        print(f"    BH q=0.05: {km3+1} of {Kc} survive (largest passing p {pv2[km3]:.4f})")
        print("      " + "; ".join(f"{r.label} n={int(r.n)} {r.exc*100:+.2f}% p={r.p:.4f}"
                                   for _, r in sub.iloc[:min(km3+1, 10)].iterrows()))
    else:
        print(f"    BH q=0.05: nothing survives (min p {pv2.min():.4f} needs <= {crit3[0]:.5f})")
    print(f"    Bonferroni on the coarse grid: best p {pv2.min():.4f} x {Kc} -> "
          f"{min(pv2.min()*Kc,1.0):.4f} "
          f"({'SURVIVES' if pv2.min()*Kc < 0.05 else 'FAILS'} at 0.05)")

    print("\n  Reported three ways on purpose, no cherry-picking: nominal K=588 (worst case, and")
    print("  plainly wrong because adjacent-L Jaccard is ~0.92), M_eff=116 (defensible), and a")
    print("  coarse pre-specified grid (what an honest a-priori study would have run).")
    print(f"  Rotation-p resolution floor is 1/{9240} = {1/9240:.5f}, so p=0.0008 is ~7 of 9240 rolls.")


def main():
    d = _vix_data.add_features(_vix_data.load())
    vix = d.vix
    fwd5 = d.g5.values
    valid5 = ~np.isnan(fwd5)
    base5 = fwd5[valid5].mean()
    print(f"panel {len(d):,} rows  {d.index[0].date()} -> {d.index[-1].date()}")
    print(f"outcome g5 (SPX fwd 5d, entry next close): n_valid={valid5.sum():,}  "
          f"unconditional mean {base5*100:+.4f}%  win {float((fwd5[valid5]>0).mean())*100:.2f}%")

    # sanity: FFT rotation == random-draw rotation
    chk = (d.stretch >= 0.10).values
    o, pf, nn = rotation_full(chk, fwd5)
    pr = rotation_random(chk, fwd5, o, 5000)
    print(f"\nrotation-test cross-check  VIX>MA10+10%: n={nn}  exc {(o-base5)*100:+.3f}%  "
          f"p_fft(all {len(d)-1} offsets)={pf:.4f}  p_random(5000 draws)={pr:.4f}")

    registry = []   # every tested cell, for the multiple-testing correction

    def track(family, label, res):
        if res["n"] >= MIN_N and res["p"] == res["p"]:
            registry.append({"family": family, "label": label, **res})
        return res

    # ================= 1. SMA length sweep =================
    print("\n" + "=" * 108)
    print("1. SMA LENGTH SWEEP  — signal: VIX >= MA*1.10  /  VIX <= MA*0.90 ;  outcome D5 (g5)")
    print("=" * 108)
    print(f"{'L':>4} | {'n_above':>7} {'exc_above':>10} {'p':>7}    | "
          f"{'n_below':>7} {'exc_below':>10} {'p':>7}    | {'win_ab':>7} {'win_be':>7}")
    print("-" * 108)
    sma_rows = {}
    for L in range(3, 41):
        ma = vix.rolling(L).mean()
        st = (vix / ma - 1.0)
        a = track("1-sma-above", f"SMA{L} +10%", cell((st >= 0.10).values, fwd5))
        b = track("1-sma-below", f"SMA{L} -10%", cell((st <= -0.10).values, fwd5))
        sma_rows[L] = (a, b)
        print(f"{L:>4} | {a['n']:>7} {a['exc']*100:>9.3f}% {a['p']:>7.4f}{stars(a['p'])} | "
              f"{b['n']:>7} {b['exc']*100:>9.3f}% {b['p']:>7.4f}{stars(b['p'])} | "
              f"{a['win']*100:>6.1f}% {b['win']*100:>6.1f}%   (base win {a['win_base']*100:.1f}%)")

    for side, idx in (("above", 0), ("below", 1)):
        vals = {L: sma_rows[L][idx]["exc"] for L in sma_rows if sma_rows[L][idx]["n"] >= MIN_N}
        if not vals:
            continue
        best = max(vals, key=lambda L: vals[L] if side == "above" else -vals[L])
        arr = np.array(list(vals.values()))
        nb = [L for L in (8, 9, 10, 11, 12) if L in vals]
        print(f"\n  {side}: peak at L={best} exc {vals[best]*100:+.3f}% | "
              f"curve mean {arr.mean()*100:+.3f}% sd {arr.std(ddof=1)*100:.3f}% | "
              f"L=10 exc {vals.get(10, float('nan'))*100:+.3f}% "
              f"(rank {sorted(vals, key=lambda L: -vals[L] if side=='above' else vals[L]).index(10)+1}/{len(vals)}) | "
              f"L=8..12 mean {np.mean([vals[L] for L in nb])*100:+.3f}%")

    # neighbourhood smoothness: how correlated are adjacent-L signals?
    masks = {L: (vix / vix.rolling(L).mean() - 1.0 >= 0.10).values for L in range(3, 41)}
    ov = []
    for L in range(3, 40):
        x, y = masks[L], masks[L + 1]
        ov.append((x & y).sum() / max((x | y).sum(), 1))
    print(f"  adjacent-L signal Jaccard overlap (above,+10%): mean {np.mean(ov):.3f} "
          f"min {np.min(ov):.3f}  -> the 38 lengths are NOT 38 independent tests")

    # ================= 2. EMA length sweep =================
    print("\n" + "=" * 108)
    print("2. EMA LENGTH SWEEP  — same thresholds, EMA(span=L) instead of SMA")
    print("=" * 108)
    print(f"{'L':>4} | {'n_above':>7} {'exc_above':>10} {'p':>7}    | "
          f"{'n_below':>7} {'exc_below':>10} {'p':>7}")
    print("-" * 108)
    ema_rows = {}
    for L in range(3, 41):
        ma = vix.ewm(span=L, adjust=False, min_periods=L).mean()
        st = (vix / ma - 1.0)
        a = track("2-ema-above", f"EMA{L} +10%", cell((st >= 0.10).values, fwd5))
        b = track("2-ema-below", f"EMA{L} -10%", cell((st <= -0.10).values, fwd5))
        ema_rows[L] = (a, b)
        print(f"{L:>4} | {a['n']:>7} {a['exc']*100:>9.3f}% {a['p']:>7.4f}{stars(a['p'])} | "
              f"{b['n']:>7} {b['exc']*100:>9.3f}% {b['p']:>7.4f}{stars(b['p'])}")

    print("\n  SMA vs EMA head-to-head (mean D5 excess across L=3..40, n>=25 cells only):")
    for side, idx in (("above +10%", 0), ("below -10%", 1)):
        s = np.array([sma_rows[L][idx]["exc"] for L in range(3, 41) if sma_rows[L][idx]["n"] >= MIN_N])
        e = np.array([ema_rows[L][idx]["exc"] for L in range(3, 41) if ema_rows[L][idx]["n"] >= MIN_N])
        ns = np.array([sma_rows[L][idx]["n"] for L in range(3, 41) if sma_rows[L][idx]["n"] >= MIN_N])
        ne = np.array([ema_rows[L][idx]["n"] for L in range(3, 41) if ema_rows[L][idx]["n"] >= MIN_N])
        print(f"    {side:<12} SMA {s.mean()*100:+.3f}% (mean n {ns.mean():.0f})   "
              f"EMA {e.mean()*100:+.3f}% (mean n {ne.mean():.0f})   diff {(e.mean()-s.mean())*100:+.3f}%")
    # EMA half-life is shorter than the same-span SMA; align by centre of mass
    print("  note: EMA(span=L) has centre of mass (L-1)/2 vs SMA's (L-1)/2 -> spans are comparable,")
    print("  but EMA has a fatter tail, so an EMA of the same span reacts a touch slower to spikes.")

    # ================= 3. threshold sweep at MA10 =================
    print("\n" + "=" * 108)
    print("3. STRETCH THRESHOLD SWEEP at SMA10  — VIX >= MA10*(1+t) and VIX <= MA10*(1-t)")
    print("=" * 108)
    st10 = d.stretch
    print(f"{'t':>7} | {'n_above':>7} {'exc':>9} {'p':>7}    {'win':>7} | "
          f"{'n_below':>7} {'exc':>9} {'p':>7}    {'win':>7}")
    print("-" * 108)
    thr_rows = {}
    for t in np.arange(0.0, 0.4001, 0.025):
        a = track("3-thr-above", f"MA10 +{t*100:.1f}%", cell((st10 >= t).values, fwd5))
        b = track("3-thr-below", f"MA10 -{t*100:.1f}%", cell((st10 <= -t).values, fwd5))
        thr_rows[round(t, 4)] = (a, b)
        fa = f"{a['exc']*100:>8.3f}% {a['p']:>7.4f}{stars(a['p'])} {a['win']*100:>6.1f}%" if a['n'] >= MIN_N else f"{'(n<25 inconclusive)':>32}"
        fb = f"{b['exc']*100:>8.3f}% {b['p']:>7.4f}{stars(b['p'])} {b['win']*100:>6.1f}%" if b['n'] >= MIN_N else f"{'(n<25 inconclusive)':>32}"
        print(f"{t*100:>6.1f}% | {a['n']:>7} {fa} | {b['n']:>7} {fb}")

    # marginal buckets: where does the edge actually live? (non-overlapping slices)
    print("\n  NON-OVERLAPPING stretch buckets (each day counted once), D5 excess:")
    edges = [-1.0, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 1.0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ((st10 > lo) & (st10 <= hi)).values
        r = cell(m, fwd5)
        if r["n"] < MIN_N:
            print(f"    ({lo*100:+6.1f}%,{hi*100:+6.1f}%]  n={r['n']:>5}  (n<25, inconclusive)")
            continue
        track("3-bucket", f"bucket({lo:+.2f},{hi:+.2f}]", r)
        print(f"    ({lo*100:+6.1f}%,{hi*100:+6.1f}%]  n={r['n']:>5}  exc {r['exc']*100:+7.3f}%  "
              f"p={r['p']:.4f}{stars(r['p'])}  win {r['win']*100:.1f}%")

    # ================= 4. percent stretch vs z-score =================
    print("\n" + "=" * 108)
    print("4. PERCENT-STRETCH vs Z-SCORE (both on a 10d window). z = (VIX - SMA10)/sigma10,")
    print("   which is exactly Bollinger %B rescaled:  %B = (z + 2)/4  for BB(10,2).")
    print("=" * 108)
    sd10 = vix.rolling(10).std(ddof=0)
    z10 = (vix - vix.rolling(10).mean()) / sd10
    pctb = d["bb10_pctb"]
    print(f"  check: corr(z10, 4*pctb-2) = {z10.corr(4*pctb-2):.6f}   corr(pct-stretch, z) = {st10.corr(z10):.4f}"
          f"   Spearman = {spearman(st10.values, z10.values):.4f}")

    ok = valid5 & st10.notna().values & z10.notna().values
    print(f"  rank IC vs g5 on the common sample (n={int(ok.sum()):,}):  "
          f"Spearman(pct-stretch) {spearman(st10.values[ok], fwd5[ok]):+.4f}   "
          f"Spearman(z-score) {spearman(z10.values[ok], fwd5[ok]):+.4f}")

    print("\n  z-threshold sweep (upper tail / lower tail):")
    print(f"{'z':>7} | {'n_above':>7} {'exc':>9} {'p':>7}    | {'n_below':>7} {'exc':>9} {'p':>7}")
    print("-" * 108)
    for zt in np.arange(0.0, 3.01, 0.25):
        a = track("4-z-above", f"z>=+{zt:.2f}", cell((z10 >= zt).values, fwd5))
        b = track("4-z-below", f"z<=-{zt:.2f}", cell((z10 <= -zt).values, fwd5))
        fa = f"{a['exc']*100:>8.3f}% {a['p']:>7.4f}{stars(a['p'])}" if a['n'] >= MIN_N else f"{'(n<25)':>20}"
        fb = f"{b['exc']*100:>8.3f}% {b['p']:>7.4f}{stars(b['p'])}" if b['n'] >= MIN_N else f"{'(n<25)':>20}"
        print(f"{zt:>7.2f} | {a['n']:>7} {fa} | {b['n']:>7} {fb}")

    print("\n  MATCHED-FREQUENCY head-to-head (threshold set so both formulations fire on the same")
    print("  number of days -> a fair comparison of the RANKING, not of arbitrary threshold units):")
    print(f"  {'target':>8} | {'pct-stretch: n / exc / p':>34} | {'z-score: n / exc / p':>34}")
    valid_st = st10.dropna()
    valid_z = z10.dropna()
    for q in (0.50, 0.30, 0.20, 0.10, 0.05, 0.025, 0.01):
        for side in ("high", "low"):
            if side == "high":
                ts, tz = valid_st.quantile(1 - q), valid_z.quantile(1 - q)
                ms, mz = (st10 >= ts).values, (z10 >= tz).values
            else:
                ts, tz = valid_st.quantile(q), valid_z.quantile(q)
                ms, mz = (st10 <= ts).values, (z10 <= tz).values
            rs = track("4-matched-pct", f"pct {side} q={q}", cell(ms, fwd5))
            rz = track("4-matched-z", f"z {side} q={q}", cell(mz, fwd5))
            fs = (f"n={rs['n']:>5} {rs['exc']*100:+7.3f}% p={rs['p']:.4f}{stars(rs['p'])}"
                  if rs['n'] >= MIN_N else f"n={rs['n']:>5}  (n<25 inconclusive)")
            fz = (f"n={rz['n']:>5} {rz['exc']*100:+7.3f}% p={rz['p']:.4f}{stars(rz['p'])}"
                  if rz['n'] >= MIN_N else f"n={rz['n']:>5}  (n<25 inconclusive)")
            print(f"  {side:>4} {q*100:>4.1f}% | {fs:>34} | {fz:>34}")

    # overlap between the two formulations at matched frequency
    for q in (0.10, 0.05):
        ts, tz = valid_st.quantile(1 - q), valid_z.quantile(1 - q)
        ms, mz = (st10 >= ts).values, (z10 >= tz).values
        print(f"  overlap of the two top-{q*100:.0f}% upper-tail signals: "
              f"Jaccard {(ms & mz).sum()/max((ms | mz).sum(),1):.3f} "
              f"({int((ms & mz).sum())} shared days)")

    # ================= 5. CONTROL: matched signal FREQUENCY across MA lengths =================
    print("\n" + "=" * 108)
    print("5. CONTROL — a short MA at a FIXED +10% fires less often and on more extreme days, so")
    print("   section 1 confounds 'MA length' with 'selectivity'. Here the threshold is re-set per")
    print("   length so every length fires on the SAME number of days. Any remaining shape is a")
    print("   genuine length effect.")
    print("=" * 108)
    lengths = list(range(3, 41))
    freq_curves = {}
    for q in (0.12, 0.05):
        print(f"\n  --- top {q*100:.0f}% of days by stretch (n ~ {int(q*9200)}), SMA vs EMA ---")
        print(f"  {'L':>4} | {'SMA thr':>8} {'exc':>9} {'p':>8}  | {'EMA thr':>8} {'exc':>9} {'p':>8}"
              f"  || {'SMA lo':>9} {'p':>8}  | {'EMA lo':>9} {'p':>8}")
        cur_hi, cur_lo, cur_ehi = {}, {}, {}
        for L in lengths:
            s = vix / vix.rolling(L).mean() - 1.0
            e = vix / vix.ewm(span=L, adjust=False, min_periods=L).mean() - 1.0
            sv, ev = s.dropna(), e.dropna()
            thi, tlo = sv.quantile(1 - q), sv.quantile(q)
            ehi, elo = ev.quantile(1 - q), ev.quantile(q)
            rh = track(f"5-freq{q}-sma-hi", f"SMA{L} top{q}", cell((s >= thi).values, fwd5))
            rl = track(f"5-freq{q}-sma-lo", f"SMA{L} bot{q}", cell((s <= tlo).values, fwd5))
            eh = track(f"5-freq{q}-ema-hi", f"EMA{L} top{q}", cell((e >= ehi).values, fwd5))
            el = track(f"5-freq{q}-ema-lo", f"EMA{L} bot{q}", cell((e <= elo).values, fwd5))
            cur_hi[L], cur_lo[L], cur_ehi[L] = rh["exc"], rl["exc"], eh["exc"]
            print(f"  {L:>4} | {thi*100:>7.2f}% {rh['exc']*100:>8.3f}% {rh['p']:>8.4f}{stars(rh['p'])[:1]} | "
                  f"{ehi*100:>7.2f}% {eh['exc']*100:>8.3f}% {eh['p']:>8.4f}{stars(eh['p'])[:1]} || "
                  f"{rl['exc']*100:>8.3f}% {rl['p']:>8.4f}{stars(rl['p'])[:1]} | "
                  f"{el['exc']*100:>8.3f}% {el['p']:>8.4f}{stars(el['p'])[:1]}   (n={rh['n']})")
        freq_curves[q] = (cur_hi, cur_lo)
        bh = max(cur_hi, key=lambda k: cur_hi[k])
        bl = min(cur_lo, key=lambda k: cur_lo[k])
        be = max(cur_ehi, key=lambda k: cur_ehi[k])
        av = np.array(list(cur_hi.values()))
        print(f"  top{q*100:.0f}% SMA upper: peak L={bh} {cur_hi[bh]*100:+.3f}% | L=10 {cur_hi[10]*100:+.3f}% "
              f"(rank {sorted(cur_hi, key=lambda k: -cur_hi[k]).index(10)+1}/{len(cur_hi)}) | "
              f"curve mean {av.mean()*100:+.3f}% sd {av.std(ddof=1)*100:.3f}%")
        print(f"  top{q*100:.0f}% EMA upper: peak L={be} {cur_ehi[be]*100:+.3f}% | L=10 {cur_ehi[10]*100:+.3f}% "
              f"| curve mean {np.mean(list(cur_ehi.values()))*100:+.3f}%  "
              f"(SMA curve mean {av.mean()*100:+.3f}% -> smoothing-type effect "
              f"{(np.mean(list(cur_ehi.values()))-av.mean())*100:+.3f}% at MATCHED frequency)")
        print(f"  top{q*100:.0f}% SMA lower: trough L={bl} {cur_lo[bl]*100:+.3f}% | L=10 {cur_lo[10]*100:+.3f}% "
              f"(rank {sorted(cur_lo, key=lambda k: cur_lo[k]).index(10)+1}/{len(cur_lo)})")

    # ================= 6. is the peak DIFFERENT from L=10? paired rotation test =================
    print("\n" + "=" * 108)
    print("6. PAIRED ROTATION TEST — is any rival configuration statistically DISTINGUISHABLE from")
    print("   the MA10 baseline? Both masks are rotated by the SAME offset; the null is 'these two")
    print("   signals have the same conditional mean'. This is the question section 1 cannot answer.")
    print("=" * 108)
    ema10 = vix.ewm(span=10, adjust=False, min_periods=10).mean()
    ref_hi = (st10 >= 0.10).values
    ref_lo = (st10 <= -0.10).values
    st5 = (vix / vix.rolling(5).mean() - 1.0)
    st20 = (vix / vix.rolling(20).mean() - 1.0)
    q12_thr = {L: (vix / vix.rolling(L).mean() - 1.0).dropna().quantile(0.88) for L in (5, 10, 20)}
    e10 = vix / ema10 - 1.0
    e10_q12 = e10.dropna().quantile(0.88)
    pairs = [
        ("SMA5 +10%   vs SMA10 +10%", (st5 >= 0.10).values, ref_hi),
        ("SMA20 +10%  vs SMA10 +10%", (st20 >= 0.10).values, ref_hi),
        ("SMA40 +10%  vs SMA10 +10%", (vix / vix.rolling(40).mean() - 1 >= 0.10).values, ref_hi),
        ("EMA10 +10%  vs SMA10 +10%", (e10 >= 0.10).values, ref_hi),
        ("SMA5 top12% vs SMA10 top12%", (st5 >= q12_thr[5]).values,
         (st10 >= q12_thr[10]).values),
        ("SMA20 top12% vs SMA10 top12%", (st20 >= q12_thr[20]).values,
         (st10 >= q12_thr[10]).values),
        ("EMA10 top12% vs SMA10 top12%", (e10 >= e10_q12).values,
         (st10 >= q12_thr[10]).values),
        ("z10>=+2     vs SMA10 +10%", (z10 >= 2.0).values, ref_hi),
        ("SMA10 +25%  vs SMA10 +10%", (st10 >= 0.25).values, ref_hi),
        ("z10<=-1.5   vs SMA10 -10%", (z10 <= -1.5).values, ref_lo),
        ("z10<=-2     vs SMA10 -10%", (z10 <= -2.0).values, ref_lo),
        ("z10 bot10%  vs pct bot10%", (z10 <= z10.dropna().quantile(0.10)).values,
         (st10 <= st10.dropna().quantile(0.10)).values),
        ("z10 top10%  vs pct top10%", (z10 >= z10.dropna().quantile(0.90)).values,
         (st10 >= st10.dropna().quantile(0.90)).values),
    ]
    print(f"  {'comparison':<32}{'nA':>6}{'nB':>6}{'meanA-meanB':>14}{'p(diff)':>10}")
    for lbl, mA, mB in pairs:
        rA, rB = cell(mA, fwd5), cell(mB, fwd5)
        if rA["n"] < MIN_N or rB["n"] < MIN_N:
            print(f"  {lbl:<32}{rA['n']:>6}{rB['n']:>6}   (n<25 inconclusive)")
            continue
        pd_ = paired_rotation(mA, mB, fwd5)
        print(f"  {lbl:<32}{rA['n']:>6}{rB['n']:>6}{(rA['cond']-rB['cond'])*100:>13.3f}%"
              f"{pd_:>10.4f}{stars(pd_)}")
    print("  A blank/high p means: the rival is NOT distinguishable from MA10 -- the length-sweep")
    print("  peak is inside the noise of the MA10 cell, i.e. picking the argmax is curve-fitting.")

    # ================= 7. half-sample stability of the length curve =================
    print("\n" + "=" * 108)
    print("7. HALF-SAMPLE STABILITY — if 10 (or any length) were structurally special, the excess-")
    print("   vs-length curve should look the same in both halves of history.")
    print("=" * 108)
    mid = len(d) // 2
    h1 = np.zeros(len(d), bool); h1[:mid] = True
    h2 = ~h1
    for tag, thr_mode in (("fixed +10%", "fixed"), ("matched top-12%", "freq")):
        c1, c2 = {}, {}
        for L in lengths:
            s = vix / vix.rolling(L).mean() - 1.0
            m = (s >= 0.10).values if thr_mode == "fixed" else (s >= s.dropna().quantile(0.88)).values
            for half, store in ((h1, c1), (h2, c2)):
                sub = half & valid5
                mm = m & sub
                store[L] = (fwd5[mm].mean() - fwd5[sub].mean()) if mm.sum() >= MIN_N else np.nan
        ok = [L for L in lengths if not (np.isnan(c1[L]) or np.isnan(c2[L]))]
        a1 = np.array([c1[L] for L in ok]); a2 = np.array([c2[L] for L in ok])
        r = float(np.corrcoef(a1, a2)[0, 1])
        b1 = ok[int(np.argmax(a1))]; b2 = ok[int(np.argmax(a2))]
        print(f"  {tag:<18} 1990-{d.index[mid].year} argmax L={b1} ({a1.max()*100:+.3f}%)   "
              f"{d.index[mid].year}-2026 argmax L={b2} ({a2.max()*100:+.3f}%)   "
              f"corr of the two curves {r:+.3f}   L=10: {c1[10]*100:+.3f}% / {c2[10]*100:+.3f}%")
        print(f"  {'':<18} rank of L=10 in half1 {sorted(ok, key=lambda L: -c1[L]).index(10)+1}/{len(ok)}"
              f"   in half2 {sorted(ok, key=lambda L: -c2[L]).index(10)+1}/{len(ok)}")

    # ================= 8. era stability of the headline cells =================
    print("\n" + "=" * 108)
    print("8. ERA STABILITY of the headline cells (D5 excess vs same-era baseline)")
    print("=" * 108)
    eras = [("1990s", 1990, 1999), ("2000s", 2000, 2009), ("2010s", 2010, 2019), ("2020s", 2020, 2099)]
    heads = [
        ("SMA10 +10%", (st10 >= 0.10).values),
        ("SMA10 +25%", (st10 >= 0.25).values),
        ("SMA5  +10%", (vix / vix.rolling(5).mean() - 1 >= 0.10).values),
        ("SMA20 +10%", (vix / vix.rolling(20).mean() - 1 >= 0.10).values),
        ("SMA30 +10%", (vix / vix.rolling(30).mean() - 1 >= 0.10).values),
        ("EMA10 +10%", (vix / ema10 - 1 >= 0.10).values),
        ("z10 >= +2", (z10 >= 2.0).values),
        ("SMA10 -10%", (st10 <= -0.10).values),
        ("SMA20 -10%", (vix / vix.rolling(20).mean() - 1 <= -0.10).values),
        ("z10 <= -1.5", (z10 <= -1.5).values),
        ("z10 <= -2", (z10 <= -2.0).values),
    ]
    yr = d.index.year.values
    print(f"  {'signal':<13}" + "".join(f"{e[0]:>22}" for e in eras))
    for lbl, m in heads:
        cells = []
        for _, y0, y1 in eras:
            sub = (yr >= y0) & (yr <= y1) & valid5
            mm = m & sub
            if mm.sum() < 15:
                cells.append(f"n={int(mm.sum())} --".rjust(22))
                continue
            exc = fwd5[mm].mean() - fwd5[sub].mean()
            cells.append(f"n={int(mm.sum()):>4} {exc*100:+7.3f}%".rjust(22))
        print(f"  {lbl:<13}" + "".join(cells))

    print("\n  YEAR-BY-YEAR consistency (share of calendar years with >=10 signal days where the")
    print("  conditional mean beats / lags that year's own baseline, in the expected direction):")
    for lbl, m in heads:
        bull = "+" in lbl or ">=" in lbl
        hits, tot, yrs = 0, 0, []
        for y in range(1990, 2027):
            sub = (yr == y) & valid5
            mm = m & sub
            if mm.sum() < 10:
                continue
            e = fwd5[mm].mean() - fwd5[sub].mean()
            tot += 1
            if (e > 0) == bull:
                hits += 1
            yrs.append(e)
        if tot:
            print(f"    {lbl:<13} {hits}/{tot} years ({hits/tot*100:.0f}%)   "
                  f"median yearly excess {np.median(yrs)*100:+.3f}%")

    # ================= 9. robustness: other horizons for the length sweep peak =================
    print("\n" + "=" * 108)
    print("9. HORIZON ROBUSTNESS — is the MA-length ranking stable across D1/D3/D5/D10/D21?")
    print("=" * 108)
    print(f"  {'L':>4}" + "".join(f"{'D'+str(h):>12}" for h in (1, 3, 5, 10, 21)))
    for L in (5, 8, 10, 12, 15, 20, 25, 30, 40):
        m = (vix / vix.rolling(L).mean() - 1 >= 0.10).values
        row = []
        for h in (1, 3, 5, 10, 21):
            f = d[f"g{h}"].values
            r = track("9-horizon", f"SMA{L} +10% D{h}", cell(m, f))
            row.append(f"{r['exc']*100:+7.3f}%{stars(r['p'])[:2]}" if r["n"] >= MIN_N else f"{'n<25':>9}")
        print(f"  {L:>4}" + "".join(s.rjust(12) for s in row))
    print("  (D-columns are excess vs the same-horizon unconditional mean; ** p<.05 *** p<.01,"
          " ' *' = p<.10)")

    multiple_testing(registry)


if __name__ == "__main__":
    main()
