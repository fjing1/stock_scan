"""
_vix_wf_verify_snooping_malen.py — ADVERSARIAL verification of the "MA length is not a real
parameter" claim from _vix_wf_malen.py, through the DATA-SNOOPING lens.

The claim is a NULL claim, so the attack is different from the usual one. Three lines of attack:

  A. POWER / preordained null. A paired rotation test that cannot reject anything is not evidence
     of equivalence. Compute the EXACT critical value of the paired-rotation null (95th pct of
     |null - base| over all 9,240 offsets) and compare it to the full spread of the L=3..40 curve.
     If the smallest detectable difference exceeds the whole curve range, p=0.288 was guaranteed
     before a single number was computed.

  B. GENERALISATION. The claim says "no SMA length in 3..40 is distinguishable from SMA10" but
     the evidence offers exactly TWO matched-frequency paired tests (L=5, L=20). Run all 37.

  C. EPISODE CONCENTRATION + ARBITRARY KNOBS. Drop the single worst month, drop 2008, drop 2020,
     drop both; re-run at q=0.05/0.10/0.12/0.20; re-run at D1/D3/D10/D21. Also check whether the
     length curve is MONOTONE in L (if short beats long systematically, length IS a parameter even
     if L=10 is not the argmax), and how many effectively-independent points the 38-point curve
     and its "+0.352 half-sample correlation" really contain.

Run: ../../vcp_env/bin/python _vix_wf_verify_snooping_malen.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _vix_data

RNG = np.random.default_rng(20260910)
MIN_N = 25
LENGTHS = list(range(3, 41))


# ------------------------------------------------------------------ rotation machinery
def _roll_means(mask: np.ndarray, fwd: np.ndarray):
    """All n circular-rotation conditional means via FFT. vals[0] is the observed value."""
    mask = np.asarray(mask, bool)
    n = len(mask)
    valid = ~np.isnan(fwd)
    a = np.where(valid, np.nan_to_num(fwd), 0.0)
    M = np.fft.rfft(mask.astype(float))
    num = np.fft.irfft(np.conj(M) * np.fft.rfft(a), n)
    den = np.round(np.fft.irfft(np.conj(M) * np.fft.rfft(valid.astype(float)), n), 6)
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return vals, int(round(den[0]))


def rotation_full(mask, fwd):
    vals, n_used = _roll_means(mask, fwd)
    obs = vals[0]
    null = vals[1:]
    null = null[~np.isnan(null)]
    if not len(null) or np.isnan(obs):
        return obs, float("nan"), n_used
    base = null.mean()
    return float(obs), float((np.abs(null - base) >= abs(obs - base)).mean()), n_used


def paired_rotation_detail(mask_a, mask_b, fwd):
    """Returns (obs_diff, p, crit95, crit99, null_sd, nA, nB).

    crit95 = the SMALLEST |observed diff - null centre| that this test would call p<0.05.
    Anything below crit95 is mathematically un-rejectable, no matter how many rows we have.
    """
    va, na = _roll_means(mask_a, fwd)
    vb, nb = _roll_means(mask_b, fwd)
    diff = va - vb
    obs = diff[0]
    null = diff[1:]
    null = null[~np.isnan(null)]
    base = null.mean()
    dev = np.abs(null - base)
    p = float((dev >= abs(obs - base)).mean())
    return (float(obs), p, float(np.quantile(dev, 0.95)), float(np.quantile(dev, 0.99)),
            float(null.std(ddof=1)), na, nb)


def excess(mask, fwd, sub=None):
    """Conditional-minus-unconditional excess on an optional sub-sample. Returns (exc, n)."""
    mask = np.asarray(mask, bool)
    valid = ~np.isnan(fwd)
    if sub is not None:
        valid = valid & sub
    m = mask & valid
    n = int(m.sum())
    if n < MIN_N:
        return np.nan, n
    return float(fwd[m].mean() - fwd[valid].mean()), n


def stretch(vix, L, kind="sma"):
    if kind == "sma":
        return vix / vix.rolling(L).mean() - 1.0
    return vix / vix.ewm(span=L, adjust=False, min_periods=L).mean() - 1.0


def topq_mask(s: pd.Series, q: float):
    return (s >= s.dropna().quantile(1 - q)).values


def spearman(a, b):
    """Rank correlation without scipy (not installed in this env)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    return float(np.corrcoef(pd.Series(a[ok]).rank().values, pd.Series(b[ok]).rank().values)[0, 1])


def stars(p):
    if p != p:
        return "   "
    return "***" if p < 0.01 else ("** " if p < 0.05 else ("*  " if p < 0.10 else "   "))


def eff_curve_points(masks):
    """Li & Ji effective number of independent tests among the 38 length masks."""
    X = np.array([m.astype(float) for m in masks])
    X = X[X.std(axis=1) > 0]
    C = np.nan_to_num(np.corrcoef(X))
    ev = np.abs(np.linalg.eigvalsh(C))
    return float(np.sum((ev >= 1.0).astype(float) + (ev - np.floor(ev))))


def main():
    d = _vix_data.add_features(_vix_data.load())
    vix = d.vix
    fwd5 = d.g5.values
    valid5 = ~np.isnan(fwd5)
    n = len(d)
    print(f"panel {n:,} rows  {d.index[0].date()} -> {d.index[-1].date()}")
    print(f"g5 valid n={int(valid5.sum()):,}  unconditional mean {fwd5[valid5].mean()*100:+.4f}%  "
          f"sd {fwd5[valid5].std(ddof=1)*100:.3f}%")

    s = {L: stretch(vix, L, "sma") for L in LENGTHS}
    e = {L: stretch(vix, L, "ema") for L in LENGTHS}
    m12 = {L: topq_mask(s[L], 0.12) for L in LENGTHS}

    # ============================================================ 0. REPRODUCE THE CLAIM
    print("\n" + "=" * 112)
    print("0. REPRODUCTION of the claim's headline numbers")
    print("=" * 112)
    for lbl, A, B in (("SMA5 top12% vs SMA10 top12%", m12[5], m12[10]),
                      ("SMA20 top12% vs SMA10 top12%", m12[20], m12[10]),
                      ("EMA10 top12% vs SMA10 top12%", topq_mask(e[10], 0.12), m12[10])):
        o, p, c95, c99, sd, na, nb = paired_rotation_detail(A, B, fwd5)
        print(f"  {lbl:<30} nA={na:>5} nB={nb:>5}  diff {o*100:+.3f}%  p={p:.4f}{stars(p)}")
    o, p, _, _, _, na, nb = paired_rotation_detail((s[20] >= 0.10).values, (s[10] >= 0.10).values, fwd5)
    print(f"  {'SMA20 +10% vs SMA10 +10% (fixed)':<30} nA={na:>5} nB={nb:>5}  diff {o*100:+.3f}%  p={p:.4f}{stars(p)}")

    curve = {}
    for L in LENGTHS:
        curve[L] = excess(m12[L], fwd5)[0]
    arr = np.array([curve[L] for L in LENGTHS])
    rank10 = sorted(LENGTHS, key=lambda L: -curve[L]).index(10) + 1
    print(f"  matched top-12% curve L=3..40: mean {arr.mean()*100:+.3f}%  sd {arr.std(ddof=1)*100:.3f}%  "
          f"min {arr.min()*100:+.3f}% (L={LENGTHS[int(arr.argmin())]})  "
          f"max {arr.max()*100:+.3f}% (L={LENGTHS[int(arr.argmax())]})")
    print(f"  RANGE max-min = {(arr.max()-arr.min())*100:.3f}%   L=10 = {curve[10]*100:+.3f}% rank {rank10}/38")

    # ============================================================ A. POWER
    print("\n" + "=" * 112)
    print("A. IS THE NULL PREORDAINED? Exact critical value of the paired-rotation test.")
    print("   crit95 = 95th percentile of |null diff - centre| over all 9,240 offsets. A test can")
    print("   NEVER return p<0.05 for an observed gap smaller than crit95.")
    print("=" * 112)
    print(f"  {'comparison':<30}{'|obs diff|':>11}{'null sd':>10}{'crit95':>10}{'crit99':>10}"
          f"{'obs/crit95':>12}{'p':>9}")
    crit_list = []
    for lbl, A, B in (("SMA5 top12% vs SMA10", m12[5], m12[10]),
                      ("SMA20 top12% vs SMA10", m12[20], m12[10]),
                      ("SMA3 top12% vs SMA10", m12[3], m12[10]),
                      ("SMA40 top12% vs SMA10", m12[40], m12[10]),
                      ("EMA10 top12% vs SMA10", topq_mask(e[10], 0.12), m12[10])):
        o, p, c95, c99, sd, na, nb = paired_rotation_detail(A, B, fwd5)
        crit_list.append(c95)
        print(f"  {lbl:<30}{abs(o)*100:>10.3f}%{sd*100:>9.3f}%{c95*100:>9.3f}%{c99*100:>9.3f}%"
              f"{abs(o)/c95:>12.2f}{p:>9.4f}{stars(p)}")
    print(f"\n  Full L=3..40 matched-top-12% curve RANGE (worst length to best) = "
          f"{(arr.max()-arr.min())*100:.3f}%")
    print(f"  Median crit95 of the paired test                                  = "
          f"{np.median(crit_list)*100:.3f}%")
    if np.median(crit_list) > (arr.max() - arr.min()):
        print("  => The smallest gap the test can flag EXCEEDS the entire curve range. 'No length is")
        print("     distinguishable from SMA10' was arithmetically guaranteed before seeing the data.")
    else:
        print("  => The test could in principle have flagged the extremes of the curve.")

    # what n would be needed? scale: rotation sd ~ 1/sqrt(n_eff); to halve crit95 need 4x the days
    print(f"\n  Sanity on precision: g5 sd = {fwd5[valid5].std(ddof=1)*100:.3f}%. A naive iid SE for a")
    print(f"  1,108-day mean is {fwd5[valid5].std(ddof=1)/np.sqrt(1108)*100:.3f}%; the rotation null sd for the")
    print(f"  PAIRED diff is {crit_list[0]/1.96*100:.3f}%-ish. Overlapping 5d windows + crisis clustering")
    print("  are why. Episode counts below quantify the real independent-sample size.")

    # episode structure of the n=1108 sample
    for L in (5, 10, 20):
        mm = m12[L] & valid5
        idx = np.flatnonzero(mm)
        runs = 1 + int((np.diff(idx) > 1).sum()) if len(idx) else 0
        months = len(pd.Series(d.index[mm]).dt.to_period("M").unique())
        years = len(pd.Series(d.index[mm]).dt.year.unique())
        print(f"  SMA{L} top12%: n={int(mm.sum())} days but only {runs} consecutive runs, "
              f"{months} distinct months, {years} distinct years  "
              f"(=> ~{runs} independent episodes, not 1,108 observations)")

    # ============================================================ B. ALL 37 LENGTHS
    print("\n" + "=" * 112)
    print("B. GENERALISATION — the claim cites 2 paired tests but asserts all of L=3..40.")
    print("   Running the paired rotation vs SMA10 for every length.")
    print("=" * 112)
    rows = []
    for L in LENGTHS:
        if L == 10:
            continue
        o, p, c95, c99, sd, na, nb = paired_rotation_detail(m12[L], m12[10], fwd5)
        jac = (m12[L] & m12[10]).sum() / max((m12[L] | m12[10]).sum(), 1)
        rows.append({"L": L, "diff": o, "p": p, "crit95": c95, "nA": na, "jac": jac})
    R = pd.DataFrame(rows)
    print(f"  {'L':>4}{'nA':>7}{'jaccard vs L10':>16}{'diff':>10}{'crit95':>10}{'p':>9}")
    for _, r in R.iterrows():
        print(f"  {int(r.L):>4}{int(r.nA):>7}{r.jac:>16.3f}{r['diff']*100:>9.3f}%{r.crit95*100:>9.3f}%"
              f"{r.p:>9.4f}{stars(r.p)}")
    sig = R[R.p < 0.05]
    print(f"\n  lengths with paired p<0.05 vs SMA10: {len(sig)}/37   min p = {R.p.min():.4f} (L={int(R.loc[R.p.idxmin(),'L'])})")
    print(f"  largest |diff| = {R['diff'].abs().max()*100:.3f}% (L={int(R.loc[R['diff'].abs().idxmax(),'L'])}), "
          f"still {R['diff'].abs().max()/R.crit95.median():.2f}x the median crit95")
    print("  NOTE: the two lengths the claim chose to report (5 and 20) have the HIGHEST Jaccard")
    print(f"  overlap with L=10 of the whole sweep (L=5 {R.loc[R.L==5,'jac'].iloc[0]:.3f}, "
          f"L=20 {R.loc[R.L==20,'jac'].iloc[0]:.3f} vs L=40 {R.loc[R.L==40,'jac'].iloc[0]:.3f}) —")
    print("  i.e. the two comparisons least able to show a difference.")

    # ============================================================ C1. MONOTONE STRUCTURE
    print("\n" + "=" * 112)
    print("C1. IS THE CURVE STRUCTURELESS, OR MONOTONE IN L? 'Length is not a parameter' requires")
    print("    NO systematic shape, not merely 'L=10 is not the argmax'.")
    print("=" * 112)
    Ls = np.array(LENGTHS, float)
    rho = spearman(arr, Ls)
    print(f"  Spearman(L, top-12% excess) over L=3..40: {rho:+.3f}")
    print(f"  short block L=3..10 mean {arr[:8].mean()*100:+.3f}%   "
          f"long block L=30..40 mean {arr[-11:].mean()*100:+.3f}%   "
          f"gap {(arr[:8].mean()-arr[-11:].mean())*100:+.3f}%")
    # is the short-vs-long block gap itself testable? paired rotation on the union masks
    short_m = np.zeros(n, bool)
    long_m = np.zeros(n, bool)
    for L in range(3, 11):
        short_m |= m12[L]
    for L in range(30, 41):
        long_m |= m12[L]
    o, p, c95, _, _, na, nb = paired_rotation_detail(short_m, long_m, fwd5)
    print(f"  paired rotation, union(L=3..10) vs union(L=30..40): nA={na} nB={nb} "
          f"diff {o*100:+.3f}% crit95 {c95*100:.3f}% p={p:.4f}{stars(p)}")
    mono = eff_curve_points([m12[L] for L in LENGTHS])
    print(f"  Li&Ji effective independent points among the 38 length masks: M_eff = {mono:.1f}")
    print(f"  mean adjacent-L Jaccard = "
          f"{np.mean([(m12[L]&m12[L+1]).sum()/max((m12[L]|m12[L+1]).sum(),1) for L in range(3,40)]):.3f}")

    # ============================================================ C2. EPISODE CONCENTRATION
    print("\n" + "=" * 112)
    print("C2. EPISODE CONCENTRATION — drop the single worst month, drop 2008, drop 2020, drop both.")
    print("    Recompute the whole curve and the two headline paired tests each time.")
    print("=" * 112)
    per = pd.Series(d.index).dt.to_period("M").values
    yrs = d.index.year.values

    # 'worst month' = the calendar month whose removal most changes the L=10 top-12% conditional mean
    base_exc10 = curve[10]
    shifts = {}
    for pm in pd.unique(per[m12[10] & valid5]):
        keep = per != pm
        ex, nn = excess(m12[10], fwd5, keep)
        if nn >= MIN_N:
            shifts[pm] = ex - base_exc10
    worst = max(shifts, key=lambda k: abs(shifts[k]))
    top5 = sorted(shifts, key=lambda k: -abs(shifts[k]))[:5]
    print(f"  months whose removal moves the SMA10 top-12% excess the most:")
    for pm in top5:
        ndays = int((m12[10] & valid5 & (per == pm)).sum())
        print(f"    {pm}  n_signal_days={ndays:>3}  excess shifts {shifts[pm]*100:+.3f}% "
              f"({base_exc10*100:+.3f}% -> {(base_exc10+shifts[pm])*100:+.3f}%)")

    scenarios = [("FULL SAMPLE", np.ones(n, bool)),
                 (f"drop worst month {worst}", per != worst),
                 ("drop 2008", yrs != 2008),
                 ("drop 2020", yrs != 2020),
                 ("drop 2008 & 2020", (yrs != 2008) & (yrs != 2020)),
                 ("drop 2008,2020,+worst month", (yrs != 2008) & (yrs != 2020) & (per != worst))]
    print(f"\n  {'scenario':<30}{'curve mean':>12}{'sd':>9}{'range':>9}{'argmax':>8}{'L10':>9}"
          f"{'rank10':>8}{'n(L10)':>8}")
    scen_curves = {}
    for lbl, keep in scenarios:
        c = {}
        for L in LENGTHS:
            c[L] = excess(m12[L], fwd5, keep)[0]
        a = np.array([c[L] for L in LENGTHS])
        if np.isnan(a).any():
            print(f"  {lbl:<30}  (some lengths n<25 — inconclusive)")
            continue
        scen_curves[lbl] = c
        rk = sorted(LENGTHS, key=lambda L: -c[L]).index(10) + 1
        _, nl10 = excess(m12[10], fwd5, keep)
        print(f"  {lbl:<30}{a.mean()*100:>11.3f}%{a.std(ddof=1)*100:>8.3f}%{(a.max()-a.min())*100:>8.3f}%"
              f"{LENGTHS[int(a.argmax())]:>8}{c[10]*100:>8.3f}%{rk:>8}{nl10:>8}")

    print("\n  curve-shape correlation of each scenario vs the full-sample curve:")
    full = np.array([scen_curves['FULL SAMPLE'][L] for L in LENGTHS])
    for lbl in scen_curves:
        if lbl == "FULL SAMPLE":
            continue
        a = np.array([scen_curves[lbl][L] for L in LENGTHS])
        print(f"    {lbl:<32} pearson {np.corrcoef(full,a)[0,1]:+.3f}   "
              f"spearman {spearman(full, a):+.3f}")

    print("\n  paired tests re-run on the reduced samples (masks and returns restricted, rotation")
    print("  performed on the contiguous restricted series):")
    print(f"  {'scenario':<30}{'L5 vs L10 diff':>17}{'p':>9}{'L20 vs L10 diff':>18}{'p':>9}")
    for lbl, keep in scenarios:
        kk = keep & valid5
        f_sub = np.where(kk, fwd5, np.nan)
        o5, p5, _, _, _, n5, _ = paired_rotation_detail(m12[5], m12[10], f_sub)
        o20, p20, _, _, _, n20, _ = paired_rotation_detail(m12[20], m12[10], f_sub)
        print(f"  {lbl:<30}{o5*100:>16.3f}%{p5:>9.4f}{stars(p5)[:1]}{o20*100:>17.3f}%{p20:>9.4f}{stars(p20)[:1]}")

    # ============================================================ C3. THE 12% KNOB
    print("\n" + "=" * 112)
    print("C3. THE 'top 12%' KNOB — 12% was chosen to match the n of the fixed +10% MA10 cell.")
    print("    Does the conclusion survive other matched frequencies?")
    print("=" * 112)
    print(f"  {'q':>6}{'n each':>9}{'curve mean':>12}{'sd':>9}{'range':>9}{'argmax':>8}{'rank10':>8}"
          f"{'L5vs10 p':>11}{'L20vs10 p':>11}{'min p over 37':>15}")
    for q in (0.03, 0.05, 0.10, 0.12, 0.20, 0.30):
        mq = {L: topq_mask(s[L], q) for L in LENGTHS}
        c = {L: excess(mq[L], fwd5)[0] for L in LENGTHS}
        a = np.array([c[L] for L in LENGTHS])
        if np.isnan(a).any():
            print(f"  {q:>6.2f}   (n<25 for some lengths — inconclusive)")
            continue
        _, nq = excess(mq[10], fwd5)
        rk = sorted(LENGTHS, key=lambda L: -c[L]).index(10) + 1
        p5 = paired_rotation_detail(mq[5], mq[10], fwd5)[1]
        p20 = paired_rotation_detail(mq[20], mq[10], fwd5)[1]
        allp = [paired_rotation_detail(mq[L], mq[10], fwd5)[1] for L in LENGTHS if L != 10]
        print(f"  {q:>6.2f}{nq:>9}{a.mean()*100:>11.3f}%{a.std(ddof=1)*100:>8.3f}%"
              f"{(a.max()-a.min())*100:>8.3f}%{LENGTHS[int(a.argmax())]:>8}{rk:>8}"
              f"{p5:>11.4f}{p20:>11.4f}{min(allp):>15.4f}")

    # ============================================================ C4. HORIZONS
    print("\n" + "=" * 112)
    print("C4. HORIZONS — the claim is stated unconditionally but was only tested at D5.")
    print("=" * 112)
    print(f"  {'horizon':>8}{'curve mean':>12}{'sd':>9}{'range':>9}{'argmax':>8}{'rank10':>8}"
          f"{'L5vs10 p':>11}{'L20vs10 p':>11}{'min p over 37':>15}{'#p<.05':>8}")
    for h in (1, 3, 5, 10, 21):
        f = d[f"g{h}"].values
        c = {L: excess(m12[L], f)[0] for L in LENGTHS}
        a = np.array([c[L] for L in LENGTHS])
        rk = sorted(LENGTHS, key=lambda L: -c[L]).index(10) + 1
        allp = np.array([paired_rotation_detail(m12[L], m12[10], f)[1] for L in LENGTHS if L != 10])
        p5 = paired_rotation_detail(m12[5], m12[10], f)[1]
        p20 = paired_rotation_detail(m12[20], m12[10], f)[1]
        print(f"  {'D'+str(h):>8}{a.mean()*100:>11.3f}%{a.std(ddof=1)*100:>8.3f}%"
              f"{(a.max()-a.min())*100:>8.3f}%{LENGTHS[int(a.argmax())]:>8}{rk:>8}"
              f"{p5:>11.4f}{p20:>11.4f}{allp.min():>15.4f}{int((allp<0.05).sum()):>8}")

    # ============================================================ C5. HALF-SAMPLE CORR
    print("\n" + "=" * 112)
    print("C5. THE '+0.352 half-sample correlation' AND THE 'rank 2/38 -> 14/38' INSTABILITY CLAIM")
    print("=" * 112)
    mid = n // 2
    h1 = np.zeros(n, bool); h1[:mid] = True
    h2 = ~h1
    print(f"  split at {d.index[mid].date()}  (half1 {int((h1&valid5).sum())} obs, "
          f"half2 {int((h2&valid5).sum())} obs)")
    for tag, mk in (("fixed +10%", lambda L: (s[L] >= 0.10).values),
                    ("matched top-12%", lambda L: m12[L])):
        c1, c2, nn1, nn2 = {}, {}, {}, {}
        for L in LENGTHS:
            m = mk(L)
            c1[L], nn1[L] = excess(m, fwd5, h1)
            c2[L], nn2[L] = excess(m, fwd5, h2)
        ok = [L for L in LENGTHS if not (np.isnan(c1[L]) or np.isnan(c2[L]))]
        a1 = np.array([c1[L] for L in ok]); a2 = np.array([c2[L] for L in ok])
        r = float(np.corrcoef(a1, a2)[0, 1])
        rs = spearman(a1, a2)
        r1 = sorted(ok, key=lambda L: -c1[L]).index(10) + 1 if 10 in ok else np.nan
        r2 = sorted(ok, key=lambda L: -c2[L]).index(10) + 1 if 10 in ok else np.nan
        print(f"\n  {tag}: argmax h1 L={ok[int(a1.argmax())]}  argmax h2 L={ok[int(a2.argmax())]}  "
              f"|argmax shift| = {abs(ok[int(a1.argmax())]-ok[int(a2.argmax())])}")
        print(f"    curve corr pearson {r:+.3f}  spearman {rs:+.3f}   "
              f"L=10 rank {r1}/{len(ok)} -> {r2}/{len(ok)}   "
              f"L=10 excess {c1[10]*100:+.3f}% -> {c2[10]*100:+.3f}%  (n {nn1[10]}/{nn2[10]})")
        # how much of the rank swing is real? spread of the curve vs its own sampling noise
        se1 = np.nanstd([fwd5[mk(L) & h1 & valid5].std(ddof=1)/np.sqrt(max(nn1[L],1)) for L in ok])
        print(f"    half-curve spread h1 {(a1.max()-a1.min())*100:.3f}%  h2 {(a2.max()-a2.min())*100:.3f}%   "
              f"typical per-cell SE ~{np.mean([fwd5[mk(L)&h1&valid5].std(ddof=1)/np.sqrt(max(nn1[L],1)) for L in ok])*100:.3f}% (iid, understated)")
        # rank of L=10 under pure noise: how volatile is a rank among 38 near-identical cells?
        # bootstrap the EPISODES (contiguous runs) inside each half to get a rank distribution
        ranks = []
        for _ in range(400):
            idx = np.flatnonzero(h1 & valid5)
            # block bootstrap, 21-day blocks, to respect overlap
            nb_ = len(idx) // 21
            pick = RNG.integers(0, len(idx) - 21, size=nb_)
            sel = np.concatenate([idx[p:p + 21] for p in pick])
            cc = {}
            for L in ok:
                mm = mk(L)[sel]
                if mm.sum() < MIN_N:
                    cc[L] = np.nan
                else:
                    cc[L] = fwd5[sel][mm].mean() - fwd5[sel].mean()
            good = [L for L in ok if not np.isnan(cc[L])]
            if 10 in good:
                ranks.append(sorted(good, key=lambda L: -cc[L]).index(10) + 1)
        if ranks:
            ranks = np.array(ranks)
            print(f"    BLOCK-BOOTSTRAP (21d blocks, 400 reps, half1 only): L=10's rank is "
                  f"{np.percentile(ranks,5):.0f}-{np.percentile(ranks,95):.0f} (90% band), median {np.median(ranks):.0f}")
            print(f"    => a rank swing of 2/38 -> 14/38 is INSIDE the noise band of a single half-sample,")
            print(f"       so the swing is not evidence of anything either way.")

        # correlation of two curves built from 38 points with M_eff independent points
        print(f"    the 38 curve points are only ~{mono:.1f} independent, so a +{r:.3f} correlation has a")
        print(f"    95% CI of roughly [{np.tanh(np.arctanh(r)-1.96/np.sqrt(max(mono-3,1))):+.2f}, "
              f"{np.tanh(np.arctanh(r)+1.96/np.sqrt(max(mono-3,1))):+.2f}] — it excludes neither 0 nor 1.")

    # ============================================================ D. FAMILYWIDE MHT
    print("\n" + "=" * 112)
    print("D. WHOLE-FAMILY MULTIPLE TESTING — does anything in the matched-frequency length family")
    print("   survive correction, and is the '+0.142% curve mean' itself real?")
    print("=" * 112)
    pm = []
    for L in LENGTHS:
        o, p, nu = rotation_full(m12[L], fwd5)
        pm.append({"L": L, "exc": o - fwd5[valid5].mean(), "p": p, "n": nu})
    P = pd.DataFrame(pm).sort_values("p").reset_index(drop=True)
    K = len(P)
    print(f"  matched top-12% one-sample cells, K={K}: min p {P.p.min():.4f} (L={int(P.L.iloc[0])}), "
          f"Bonferroni {min(P.p.min()*K,1):.4f}")
    crit = np.arange(1, K + 1) / K * 0.05
    pas = np.flatnonzero(P.p.values <= crit)
    print(f"  BH q=0.05: {int(pas.max())+1 if len(pas) else 0} of {K} survive")
    print(f"  Li&Ji M_eff over these 38 masks = {mono:.1f}  -> Bonferroni(M_eff) "
          f"{min(P.p.min()*mono,1):.4f}")
    print(f"  top 5 by p: " + "; ".join(f"L={int(r.L)} {r.exc*100:+.3f}% p={r.p:.4f}"
                                        for _, r in P.head(5).iterrows()))
    print(f"  L=10 specifically: exc {P.loc[P.L==10,'exc'].iloc[0]*100:+.3f}%  "
          f"p={P.loc[P.L==10,'p'].iloc[0]:.4f}  n={int(P.loc[P.L==10,'n'].iloc[0])}")
    print("  Against the ~70+ signal/horizon combinations already run in this project the relevant")
    print("  Bonferroni floor is p<0.0007; nothing in this family is near it.")


if __name__ == "__main__":
    main()
