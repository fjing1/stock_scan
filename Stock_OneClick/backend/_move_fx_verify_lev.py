"""
_move_fx_verify_lev.py — adversarial verification of the factor blocks that survived the screening
pass (LEV = leverage/semivariance, VOL = volume), plus an honest re-test of VIXC on the span where
its inputs actually exist.

WHY: the screen reported LEV at +0.0046..+0.0370 delta BSS2 and VOL at +0.0024 for single names at
h=1. Before either goes into the model those numbers have to survive:
  1. IDENTICAL ROWS. The screen substituted nan_to_num(...,0) for missing features at predict time.
     In LOG space a 0 means sigma = 1 (100% per day), which is catastrophic, and it is exactly what
     produced the absurd VIXC numbers (+0.0766 / -0.1618 / +0.1103). Here the row mask is computed
     ONCE per comparison and both baseline and candidate are fit and scored on it.
  2. PERMUTATION CONTROL. The candidate columns are shuffled across rows, preserving their marginal
     distribution and parameter count while destroying the timing. A real factor must beat its twin.
  3. YEAR-BLOCK BOOTSTRAP. Overlapping windows and a ~1-correlated cross-section mean pooled n is
     not independent n; resample whole calendar years.
  4. CRISIS CONCENTRATION. Drop 2008/2009/2020 and re-measure.
  5. NON-OVERLAPPING windows (every h-th row).

VIXC is handled separately: its inputs start 2006 (^VIX3M) / 2011 (^VIX9D) and END 2026-07-17 in
yfinance, so even a positive result is not live-deployable today. That constraint is reported.

Run: ../../vcp_env/bin/python _move_fx_verify_lev.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import _move_fx_screen as S
import _move_lib as L

T0 = time.time()
RNG = np.random.default_rng(11)
NBOOT = 400
MIN_TRAIN_YEARS = 5
SHRINK = 40.0


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def walk(base, extra, lr, tgt, yr, sid, h, thr, nonoverlap=False):
    """Walk-forward P(|move|>=thr) on a pre-masked, all-finite sample.
    extra=None gives the baseline. Returns (pred, actual, per-symbol clim, year)."""
    ones = np.ones((len(lr), 1))
    a_dn, a_up = np.log(1 - thr), np.log(1 + thr)
    big = ((lr <= a_dn) | (lr >= a_up)).astype(float)
    nsym = int(sid.max()) + 1
    P, PS, BIG, YY = [], [], [], []
    years = sorted(set(yr))
    for ty in years[MIN_TRAIN_YEARS:]:
        tr, te = yr < ty, yr == ty
        if nonoverlap:
            te = te & (np.arange(len(yr)) % h == 0)
        if tr.sum() < 4000 or te.sum() < 30:
            continue
        Xtr = np.hstack([ones[tr], base[tr]] + ([extra[tr]] if extra is not None else []))
        Xte = np.hstack([ones[te], base[te]] + ([extra[te]] if extra is not None else []))
        b = S.ols(Xtr, tgt[tr])
        if b is None:
            continue
        s_tr, s_te = np.exp(Xtr @ b) * np.sqrt(h), np.exp(Xte @ b) * np.sqrt(h)
        P.append(S.probs(np.sort(lr[tr] / s_tr), s_te, thr))
        cnt = np.bincount(sid[tr], weights=big[tr], minlength=nsym)
        tot = np.bincount(sid[tr], minlength=nsym).astype(float)
        PS.append(((cnt + SHRINK * big[tr].mean()) / (tot + SHRINK))[sid[te]])
        BIG.append(big[te]); YY.append(np.full(int(te.sum()), ty))
    if not P:
        return None
    return np.concatenate(P), np.concatenate(BIG), np.concatenate(PS), np.concatenate(YY)


def bss2(p, bg, ps):
    return 1 - np.nanmean((p - bg) ** 2) / np.nanmean((ps - bg) ** 2)


def boot(p0, p1, bg, ps, yy):
    yrs = np.unique(yy)
    idx = {u: np.flatnonzero(yy == u) for u in yrs}
    out = np.empty(NBOOT)
    for b in range(NBOOT):
        m = np.concatenate([idx[u] for u in RNG.choice(yrs, len(yrs), replace=True)])
        out[b] = bss2(p1[m], bg[m], ps[m]) - bss2(p0[m], bg[m], ps[m])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float((out > 0).mean())


def compare(d, blk, h, thr, drop_crisis=False):
    """One clean comparison on a single shared row mask."""
    base, lr, tgt, yr, sid = d["base"], d["lr"], d["tgt"], d["yr"], d["sid"]
    X = d[blk]
    ok = (np.isfinite(base).all(1) & np.isfinite(X).all(1)
          & np.isfinite(lr) & np.isfinite(tgt))
    if drop_crisis:
        ok &= ~np.isin(yr, [2008, 2009, 2020])
    if ok.sum() < 20000:
        return None
    i = np.flatnonzero(ok)
    b_, x_, l_, t_, y_, s_ = base[i], X[i], lr[i], tgt[i], yr[i], sid[i]
    r0 = walk(b_, None, l_, t_, y_, s_, h, thr)
    r1 = walk(b_, x_, l_, t_, y_, s_, h, thr)
    rp = walk(b_, x_[RNG.permutation(len(x_))], l_, t_, y_, s_, h, thr)
    rn0 = walk(b_, None, l_, t_, y_, s_, h, thr, nonoverlap=True)
    rn1 = walk(b_, x_, l_, t_, y_, s_, h, thr, nonoverlap=True)
    if not (r0 and r1):
        return None
    return dict(n=len(r1[0]), base=bss2(*r0[:3]), d=bss2(*r1[:3]) - bss2(*r0[:3]),
                d_perm=(bss2(*rp[:3]) - bss2(*r0[:3])) if rp else np.nan,
                d_nonov=(bss2(*rn1[:3]) - bss2(*rn0[:3])) if (rn0 and rn1) else np.nan,
                d_ece=L.ece(r1[0], r1[1].astype(bool)) - L.ece(r0[0], r0[1].astype(bool)),
                boot=boot(r0[0], r1[0], r1[1], r1[2], r1[3]))


def main():
    cache = S.build(stride=3)
    rows = []
    for (g, h), d in sorted(cache.items()):
        for thr in (0.02, 0.05):
            for blk in ("LEV", "VOL", "VIXC", "GAP", "JUMP"):
                if blk not in d or np.isfinite(d[blk]).all(1).mean() < 0.10:
                    continue
                c = compare(d, blk, h, thr)
                if c is None:
                    continue
                cc = compare(d, blk, h, thr, drop_crisis=True)
                lo, hi, pgt = c["boot"]
                rows.append(dict(group=g, h=h, thr=thr, block=blk, n=c["n"], base=c["base"],
                                 d=c["d"], d_perm=c["d_perm"], lo=lo, hi=hi, p_gt0=pgt,
                                 d_nocrisis=cc["d"] if cc else np.nan,
                                 d_nonov=c["d_nonov"], d_ece=c["d_ece"]))
                log(f"{g:<7} h={h:<3} thr={thr*100:.0f}% {blk:<5} n={c['n']:>7,} "
                    f"d={c['d']:+.4f} CI[{lo:+.4f},{hi:+.4f}] P>0={pgt:.2f} "
                    f"perm={c['d_perm']:+.4f} noCrisis={(cc['d'] if cc else float('nan')):+.4f} "
                    f"nonOv={c['d_nonov']:+.4f} dECE={c['d_ece']:+.4f}")
    df = pd.DataFrame(rows)
    df.to_pickle("_move_fx_verify_lev.pkl")
    print("\n" + "=" * 126)
    print("VERIFIED delta BSS2 -- identical rows, permutation control, year-block bootstrap")
    print("perm must be ~0 for the gain to be information rather than free parameters")
    print("=" * 126)
    if not df.empty:
        show = df[["group", "h", "thr", "block", "n", "base", "d", "d_perm", "lo", "hi",
                   "p_gt0", "d_nocrisis", "d_nonov", "d_ece"]]
        print(show.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("\nsaved _move_fx_verify_lev.pkl")


if __name__ == "__main__":
    main()
