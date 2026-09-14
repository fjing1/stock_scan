"""
_vix_wf_verify_snooping_malen2.py — addendum to _vix_wf_verify_snooping_malen.py.

Two follow-ups the first pass raised:

  1. The 'top 12%' knob. At q=0.03/0.05/0.20/0.30 some lengths DO separate from L=10 at p<0.05.
     Which ones, how big, and do they survive Bonferroni over the 37 lengths (and over the whole
     37 x 6 q-grid)? Are they isolated spikes (multiplicity artefact) or a coherent block of
     adjacent lengths (real)?

  2. EQUIVALENCE, not non-rejection. 'Not distinguishable' only supports 'length is not a real
     parameter' if the confidence interval on the difference EXCLUDES economically meaningful
     gaps. Build rotation-based two-sided 90%/95% intervals for every L vs L=10 and compare the
     interval half-width to the size of the L=10 edge itself.

Run: ../../vcp_env/bin/python _vix_wf_verify_snooping_malen2.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _vix_data
from _vix_wf_verify_snooping_malen import (LENGTHS, _roll_means, excess,  # noqa: F401
                                           paired_rotation_detail, rotation_full, spearman,
                                           stars, stretch, topq_mask)


def paired_ci(mask_a, mask_b, fwd, lvl=0.90):
    """Rotation-calibrated CI for meanA-meanB: obs +/- quantile of the centred null."""
    va, na = _roll_means(mask_a, fwd)
    vb, nb = _roll_means(mask_b, fwd)
    diff = va - vb
    obs = diff[0]
    null = diff[1:]
    null = null[~np.isnan(null)]
    hw = float(np.quantile(np.abs(null - null.mean()), lvl))
    return float(obs), hw, na, nb


def main():
    d = _vix_data.add_features(_vix_data.load())
    vix, fwd5 = d.vix, d.g5.values
    valid5 = ~np.isnan(fwd5)
    s = {L: stretch(vix, L, "sma") for L in LENGTHS}

    print("=" * 112)
    print("1. WHICH lengths separate from L=10 once the arbitrary 'top 12%' knob is moved?")
    print("   (all 37 lengths x 6 frequencies = 222 paired rotation tests)")
    print("=" * 112)
    allrows = []
    for q in (0.03, 0.05, 0.10, 0.12, 0.20, 0.30):
        mq = {L: topq_mask(s[L], q) for L in LENGTHS}
        base_exc, nq = excess(mq[10], fwd5)
        row = []
        for L in LENGTHS:
            if L == 10:
                continue
            o, p, c95, _, sd, na, _ = paired_rotation_detail(mq[L], mq[10], fwd5)
            allrows.append({"q": q, "L": L, "diff": o, "p": p, "n": na, "crit95": c95})
            row.append((L, o, p))
        sig = [(L, o, p) for L, o, p in row if p < 0.05]
        print(f"\n  q={q:.2f}  n_each~{nq}  L=10 own excess {base_exc*100:+.3f}%   "
              f"lengths with p<0.05 vs L=10: {len(sig)}/37")
        if sig:
            blocks = ",".join(str(L) for L, _, _ in sig)
            print(f"    significant L: {blocks}")
            for L, o, p in sorted(sig, key=lambda t: t[2])[:6]:
                ex, _ = excess(mq[L], fwd5)
                print(f"      L={L:>2}  L-excess {ex*100:+.3f}%  vs L10 {base_exc*100:+.3f}%  "
                      f"diff {o*100:+.3f}%  p={p:.4f}  bonf37={min(p*37,1):.4f}  bonf222={min(p*222,1):.4f}")
    A = pd.DataFrame(allrows)
    print(f"\n  ACROSS THE WHOLE 222-CELL GRID: {int((A.p<0.05).sum())} cells p<0.05 "
          f"(expected under a true null ~{0.05*222:.0f}), {int((A.p<0.01).sum())} cells p<0.01 "
          f"(expected ~{0.01*222:.0f})")
    print(f"  min p = {A.p.min():.5f} at q={A.loc[A.p.idxmin(),'q']:.2f} L={int(A.loc[A.p.idxmin(),'L'])}"
          f"  -> Bonferroni x222 = {min(A.p.min()*222,1):.4f}")
    # are the significant cells adjacent (coherent) or scattered (multiplicity noise)?
    for q in sorted(A.q.unique()):
        sl = A[(A.q == q) & (A.p < 0.05)].L.values
        if len(sl) >= 2:
            runs = 1 + int((np.diff(np.sort(sl)) > 1).sum())
            print(f"    q={q:.2f}: {len(sl)} significant lengths in {runs} contiguous block(s) "
                  f"-> {'coherent' if runs <= 2 else 'scattered'}")
        elif len(sl) == 1:
            print(f"    q={q:.2f}: 1 isolated significant length (L={sl[0]})")

    print("\n" + "=" * 112)
    print("2. EQUIVALENCE — what gaps does 'not distinguishable' still permit?")
    print("   Rotation-calibrated 90% interval on (L excess - L10 excess) at matched top-12%.")
    print("=" * 112)
    m12 = {L: topq_mask(s[L], 0.12) for L in LENGTHS}
    own10, n10 = excess(m12[10], fwd5)
    print(f"  reference: SMA10 top-12% own excess {own10*100:+.3f}% (n={n10})")
    print(f"  {'L':>4}{'diff':>10}{'90% CI':>22}{'CI half-width':>15}{'HW / L10 edge':>15}")
    hws = []
    for L in (3, 5, 7, 15, 20, 24, 30, 40):
        o, hw, na, _ = paired_ci(m12[L], m12[10], fwd5, 0.90)
        hws.append(hw)
        print(f"  {L:>4}{o*100:>9.3f}%   [{(o-hw)*100:+6.3f}%, {(o+hw)*100:+6.3f}%]"
              f"{hw*100:>14.3f}%{hw/abs(own10):>15.2f}x")
    print(f"\n  median 90%-CI half-width = {np.median(hws)*100:.3f}%, i.e. "
          f"{np.median(hws)/abs(own10)*100:.0f}% of the SMA10 signal's ENTIRE edge.")
    print("  => the data cannot rule out that a different length delivers roughly HALF AGAIN to")
    print("     DOUBLE the SMA10 edge. Non-rejection here is a resolution limit, not equivalence.")

    # formal TOST-style: what fraction of lengths have a CI that EXCLUDES a +/-0.10% gap?
    exc_cnt = 0
    tot = 0
    for L in LENGTHS:
        if L == 10:
            continue
        o, hw, _, _ = paired_ci(m12[L], m12[10], fwd5, 0.90)
        tot += 1
        if abs(o) + hw <= 0.0010:
            exc_cnt += 1
    print(f"\n  TOST at a +/-0.10% equivalence margin (half the SMA10 edge): {exc_cnt}/{tot} lengths")
    print("  can actually be declared EQUIVALENT to SMA10. The rest are simply unresolved.")

    # 3. how much power would we need?
    print("\n" + "=" * 112)
    print("3. HOW MUCH DATA WOULD SETTLE IT? rotation-null sd scales ~1/sqrt(n_days)")
    print("=" * 112)
    o5, p5, c95_5, _, sd5, _, _ = paired_rotation_detail(m12[5], m12[10], fwd5)
    for mult in (1, 2, 4, 9, 16):
        print(f"    {mult:>2}x history ({36.7*mult:>5.0f} years): crit95 ~ {c95_5/np.sqrt(mult)*100:.3f}%  "
              f"-> observed +0.064% gap would be {'DETECTABLE' if 0.00064 > c95_5/np.sqrt(mult) else 'still invisible'}")
    print("  The L=5-vs-L=10 gap that the claim calls 'noise' needs roughly "
          f"{(c95_5/0.00064)**2:.0f}x the available history to resolve; i.e. the experiment can")
    print("  never be run. That makes the null uninformative rather than established.")

    # 4. does the CHOICE of D5 matter for the FIXED-threshold result the claim leans on?
    print("\n" + "=" * 112)
    print("4. THE ONE SIGNIFICANT COMPARISON THE CLAIM EXPLAINS AWAY (fixed +10%, SMA20 vs SMA10,")
    print("   -0.143% p=0.024): is 'selectivity not length' the only reading?")
    print("=" * 112)
    for L in (5, 20, 30, 40):
        mA = (s[L] >= 0.10).values
        mB = (s[10] >= 0.10).values
        o, p, c95, _, _, na, nb = paired_rotation_detail(mA, mB, fwd5)
        exA, _ = excess(mA, fwd5)
        exB, _ = excess(mB, fwd5)
        print(f"  SMA{L:>2} +10% vs SMA10 +10%: nA={na:>5} nB={nb:>5}  "
              f"excA {exA*100:+.3f}%  excB {exB*100:+.3f}%  diff {o*100:+.3f}%  p={p:.4f}{stars(p)}")
    print("  At a FIXED threshold, longer MA -> more signal days -> lower average intensity.")
    ns = {L: int((s[L] >= 0.10).sum()) for L in (5, 10, 20, 40)}
    print(f"  signal days at a fixed +10%: L=5 {ns[5]}, L=10 {ns[10]}, L=20 {ns[20]}, L=40 {ns[40]}")
    print("  The claim's 'it was selectivity' reading is supported by the monotone n, but note that")
    print("  under EITHER reading the practitioner-relevant rule (fixed +10% on MA10 vs MA20) DOES")
    print("  differ: -0.143%, p=0.024. Frequency-matching removes the difference by construction,")
    print("  because it removes the only thing the length was doing.")


if __name__ == "__main__":
    main()
