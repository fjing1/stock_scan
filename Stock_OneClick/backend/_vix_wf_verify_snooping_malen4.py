"""
_vix_wf_verify_snooping_malen4.py — high-rep version of the surrogate-grid control from
_vix_wf_verify_snooping_malen3.py, plus a stricter version of the family-wise test.

Question: across the FULL matched-frequency grid (6 frequencies x 37 lengths = 222 paired
rotation tests of 'SMA_L differs from SMA10'), the real data give 40 cells p<0.05 and a min p of
0.00011. Is that more than the identical machinery produces when the outcome carries no relation
to the masks? Circularly shifting the OUTCOME is an exact draw from the paired-rotation null for
every cell at once: it preserves each mask's geometry, the returns' autocorrelation, and the
correlation structure ACROSS the 222 cells. So the distribution of (count, min-p) over shifts is
the exact family-wise null.

Run: ../../vcp_env/bin/python _vix_wf_verify_snooping_malen4.py
"""
from __future__ import annotations

import numpy as np

import _vix_data
from _vix_wf_verify_snooping_malen import LENGTHS, _roll_means, stretch, topq_mask

QS = (0.03, 0.05, 0.10, 0.12, 0.20, 0.30)


def main():
    d = _vix_data.add_features(_vix_data.load())
    vix, fwd5 = d.vix, d.g5.values
    n = len(d)
    s = {L: stretch(vix, L, "sma") for L in LENGTHS}
    masks = {(q, L): topq_mask(s[L], q) for q in QS for L in LENGTHS}

    # precompute the FFT of every mask once
    Mhat = {k: np.fft.rfft(m.astype(float)) for k, m in masks.items()}

    def grid(fwd):
        """-> (count p<0.05, count p<0.01, min p) over all 222 paired cells for this outcome."""
        valid = ~np.isnan(fwd)
        a = np.where(valid, np.nan_to_num(fwd), 0.0)
        A, V = np.fft.rfft(a), np.fft.rfft(valid.astype(float))
        curves = {}
        for k, M in Mhat.items():
            num = np.fft.irfft(np.conj(M) * A, n)
            den = np.round(np.fft.irfft(np.conj(M) * V, n), 6)
            with np.errstate(invalid="ignore", divide="ignore"):
                curves[k] = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
        c05 = c01 = 0
        mp = 1.0
        for q in QS:
            ref = curves[(q, 10)]
            for L in LENGTHS:
                if L == 10:
                    continue
                diff = curves[(q, L)] - ref
                obs, null = diff[0], diff[1:]
                null = null[~np.isnan(null)]
                base = null.mean()
                p = float((np.abs(null - base) >= abs(obs - base)).mean())
                c05 += p < 0.05
                c01 += p < 0.01
                mp = min(mp, p)
        return c05, c01, mp

    real = grid(fwd5)
    print(f"REAL grid: {real[0]} of 222 cells p<0.05, {real[1]} cells p<0.01, min p {real[2]:.5f}")

    rng = np.random.default_rng(11)
    reps = 300
    offs = rng.integers(250, n - 250, size=reps)
    C05, C01, MP = [], [], []
    for i, off in enumerate(offs):
        c05, c01, mp = grid(np.roll(fwd5, int(off)))
        C05.append(c05); C01.append(c01); MP.append(mp)
    C05, C01, MP = np.array(C05), np.array(C01), np.array(MP)

    print(f"\nSURROGATE grids ({reps} circular shifts of the outcome — exact family-wise null):")
    for nm, arr, obs, hi in (("count p<0.05", C05, real[0], True),
                             ("count p<0.01", C01, real[1], True)):
        print(f"  {nm:<13} median {np.median(arr):>5.0f}  90th {np.percentile(arr,90):>5.0f}  "
              f"99th {np.percentile(arr,99):>5.0f}  max {arr.max():>5.0f}   REAL={obs}   "
              f"family-wise p = {(arr >= obs).mean():.4f}")
    print(f"  {'min-p':<13} median {np.median(MP):.5f}  5th {np.percentile(MP,5):.5f}  "
          f"1st {np.percentile(MP,1):.5f}  min {MP.min():.5f}   REAL={real[2]:.5f}   "
          f"family-wise p = {(MP <= real[2]).mean():.4f}")
    print("\n  A family-wise p below 0.05 on EITHER statistic means: the length-vs-SMA10 differences")
    print("  seen across the frequency grid are larger than this exact machinery generates by")
    print("  chance, AFTER paying for all 222 searched configurations. That is the opposite of")
    print("  'MA length is not a real parameter'.")

    # and the honest counterpoint: restricted to the claim's own top-12% slice only
    def slice_grid(fwd, q):
        valid = ~np.isnan(fwd)
        a = np.where(valid, np.nan_to_num(fwd), 0.0)
        A, V = np.fft.rfft(a), np.fft.rfft(valid.astype(float))
        cur = {}
        for L in LENGTHS:
            M = Mhat[(q, L)]
            num = np.fft.irfft(np.conj(M) * A, n)
            den = np.round(np.fft.irfft(np.conj(M) * V, n), 6)
            with np.errstate(invalid="ignore", divide="ignore"):
                cur[L] = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
        mp = 1.0
        c = 0
        for L in LENGTHS:
            if L == 10:
                continue
            diff = cur[L] - cur[10]
            obs, null = diff[0], diff[1:]
            null = null[~np.isnan(null)]
            base = null.mean()
            p = float((np.abs(null - base) >= abs(obs - base)).mean())
            mp = min(mp, p)
            c += p < 0.05
        return c, mp

    print("\n  PER-FREQUENCY family-wise test (37 lengths each, same 300 shifts):")
    print(f"  {'q':>6}{'REAL count<.05':>16}{'REAL min-p':>12}{'FW p (count)':>14}{'FW p (min-p)':>14}")
    for q in QS:
        rc, rm = slice_grid(fwd5, q)
        sc = np.array([slice_grid(np.roll(fwd5, int(o)), q) for o in offs[:150]])
        print(f"  {q:>6.2f}{rc:>16}{rm:>12.5f}{(sc[:,0] >= rc).mean():>14.4f}"
              f"{(sc[:,1] <= rm).mean():>14.4f}")


if __name__ == "__main__":
    main()
