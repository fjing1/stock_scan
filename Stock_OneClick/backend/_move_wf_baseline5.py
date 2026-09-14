"""
_move_wf_baseline5.py — how much of the baseline's miscalibration is FIXABLE?

The harness showed a monotone reliability sag: in the top EWMA-sigma decile realized vol is only
0.82-0.94x the forecast, in the bottom decile 1.08-1.32x. That is textbook vol mean reversion, and
it means the provisional sigma_hat is a biased forecast. This script fits the standard
Mincer-Zarnowitz correction ON TRAIN YEARS ONLY

    log sigma_h_real = a_h + b_h * log(sigma_ewma_1 * sqrt(h))     ->  sigma_hat' = exp(a+b*log(.))

and re-runs the identical walk-forward with sigma_hat' (variant EMPMZ). b_h < 1 shrinks extreme
forecasts toward the mean. Everything is still strictly out of sample: a_h, b_h and the z quantiles
all come from years < test year, with the last h train days purged.

This is a PROBE to size the remaining headroom, not a proposal to ship an unvalidated tweak.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _move_lib as L
import _move_wf_baseline as W

HORIZONS = W.HORIZONS
THRS = (0.02, 0.05)


def hdr(s):
    print("\n" + "=" * 100)
    print(s)
    print("=" * 100, flush=True)


def run(close_g, h):
    """Walk-forward for one group/horizon: EMP (baseline), EMPMZ (bias-corrected sigma),
    CLIMSYM (per-symbol base rate). Returns dict[(thr, model)] -> arrays."""
    A = W.flatten(close_g, h)
    idx = A["index"]
    real_h = L.realized_vol_forward(close_g, h) * math.sqrt(h)      # h-day realized sigma
    m = np.zeros((len(close_g), close_g.shape[1]), bool)
    m[A["pos"], A["sym"]] = True
    rv = real_h.to_numpy()[m]
    if h == 1:
        rv = (np.abs(A["fwd_l"]) * math.sqrt(math.pi / 2))           # single-|r| estimator

    lut = {k: i for i, k in enumerate(L.BUCKETS)}
    actual = {thr: np.array([lut[x] for x in
                             L.bucketize(pd.Series(A["fwd_s"]), thr).to_numpy()], dtype=np.int8)
              for thr in THRS}
    store = {(thr, mm): {"p": [], "y": [], "pos": [], "sym": [], "rawmin": []}
             for thr in THRS for mm in ("EMP", "EMPMZ", "CLIMSYM")}
    coefs = []
    for y, trm, _ in L.walk_forward_years(idx, W.MIN_TRAIN_YEARS):
        last = np.flatnonzero(trm)[-1]
        tr = (A["yr"] < y) & (A["pos"] <= last - h)
        te = A["yr"] == y
        if tr.sum() < 5000 or te.sum() == 0:
            continue
        # ---- Mincer-Zarnowitz on train only
        x = np.log(A["sig_h"][tr])
        yv = np.log(np.where(rv[tr] > 0, rv[tr], np.nan))
        ok = np.isfinite(x) & np.isfinite(yv)
        b, a = np.polyfit(x[ok], yv[ok], 1)
        coefs.append((y, a, b, int(ok.sum())))
        s_mz_tr = np.exp(a + b * np.log(A["sig_h"][tr]))
        s_mz_te = np.exp(a + b * np.log(A["sig_h"][te]))

        z_srt = np.sort(A["fwd_l"][tr] / A["sig_h"][tr])
        z_mz_srt = np.sort(A["fwd_l"][tr] / s_mz_tr)

        for thr in THRS:
            a_tr = actual[thr][tr]
            clim = L.climatology(a_tr.astype(int), k=4)
            cs = np.bincount(A["sym"][tr].astype(np.int64) * 4 + a_tr.astype(np.int64),
                             minlength=A["n_sym"] * 4).reshape(A["n_sym"], 4).astype(float)
            cs = cs + 40.0 * clim[None, :]
            cs /= cs.sum(axis=1, keepdims=True)
            cand = {"EMP": W.probs_empirical(z_srt, A["sig_h"][te], thr),
                    "EMPMZ": W.probs_empirical(z_mz_srt, s_mz_te, thr),
                    "CLIMSYM": W.probs_floor(cs[A["sym"][te]])}
            for mm, (p, rmn) in cand.items():
                st = store[(thr, mm)]
                st["p"].append(p.astype(np.float32)); st["y"].append(actual[thr][te])
                st["pos"].append(A["pos"][te]); st["sym"].append(A["sym"][te])
                st["rawmin"].append(rmn.astype(np.float32))
    out = {k: {kk: np.concatenate(v) for kk, v in st.items()} for k, st in store.items()}
    return out, coefs


def sc(rm, rb, sel=None):
    s = slice(None) if sel is None else sel
    p, yv, pb = rm["p"][s].astype(float), rm["y"][s].astype(int), rb["p"][s].astype(float)
    pm, pbm = p[:, 0] + p[:, 3], pb[:, 0] + pb[:, 3]
    hit = (yv == 0) | (yv == 3)
    b2, b2c = float(((pm - hit) ** 2).mean()), float(((pbm - hit) ** 2).mean())
    return dict(n=len(yv), bss=L.skill_score(L.brier_multi(p, yv), L.brier_multi(pb, yv)),
                bss2=L.skill_score(b2, b2c), ece_up=L.ece(p[:, 3], yv == 3),
                ece_move=L.ece(pm, hit))


def main():
    close, idx_syms, singles = W.load_panel()
    groups = {"INDEX": close[idx_syms], "SINGLES": close[singles]}

    hdr("H1 — Mincer-Zarnowitz slope b_h fitted on TRAIN years (b<1 => EWMA overshoots)")
    print(f"{'grp':<8}{'h':>3}{'first_yr_b':>11}{'last_yr_b':>10}{'mean_b':>8}{'mean_a':>9}"
          f"{'n_fits':>8}")
    R = {}
    for g, cg in groups.items():
        for h in HORIZONS:
            o, co = run(cg, h)
            R[(g, h)] = o
            bs = [c[2] for c in co]
            print(f"{g:<8}{h:>3}{bs[0]:>11.3f}{bs[-1]:>10.3f}{np.mean(bs):>8.3f}"
                  f"{np.mean([c[1] for c in co]):>9.3f}{len(co):>8}", flush=True)

    hdr("H2 — EMP (provisional) vs EMPMZ (bias-corrected sigma), all OOS, vs CLIMSYM baseline")
    print(f"{'grp':<8}{'h':>3}{'thr':>6}{'n':>10}"
          f"{'BSS4_EMP':>10}{'BSS4_MZ':>9}{'BSS2_EMP':>10}{'BSS2_MZ':>9}"
          f"{'ECEup_EMP':>11}{'ECEup_MZ':>10}{'ECEmv_EMP':>11}{'ECEmv_MZ':>10}")
    for g in groups:
        for h in HORIZONS:
            for thr in THRS:
                o = R[(g, h)]
                e = sc(o[(thr, "EMP")], o[(thr, "CLIMSYM")])
                z = sc(o[(thr, "EMPMZ")], o[(thr, "CLIMSYM")])
                print(f"{g:<8}{h:>3}{thr:>6.0%}{e['n']:>10,}"
                      f"{e['bss']:>10.4f}{z['bss']:>9.4f}{e['bss2']:>10.4f}{z['bss2']:>9.4f}"
                      f"{e['ece_up']:>11.4f}{z['ece_up']:>10.4f}"
                      f"{e['ece_move']:>11.4f}{z['ece_move']:>10.4f}", flush=True)

    hdr("H3 — reliability of P(|r|>=2%) after the MZ correction (compare to B3 in pass 2)")
    for g in groups:
        for h in (5, 21):
            o = R[(g, h)][(0.02, "EMPMZ")]
            p, yv = o["p"].astype(float), o["y"].astype(int)
            pm, hit = p[:, 0] + p[:, 3], (yv == 0) | (yv == 3)
            print(f"\n--- {g} h={h} EMPMZ P(|r|>=2%)  ECE={L.ece(pm, hit):.4f}  n={len(yv):,} ---")
            print(L.reliability(pm, hit, 10).to_string(float_format=lambda v: f"{v:9.4f}"))

    hdr("H4 — RECENT WINDOW 2021-2026, EMPMZ vs CLIMSYM, date-block bootstrap CI")
    print(f"{'grp':<8}{'h':>3}{'thr':>6}{'n':>10}{'BSS4':>8}{'lo95':>8}{'hi95':>8}"
          f"{'BSS2':>8}{'ECEup':>8}{'ECEmv':>8}")
    recent = np.flatnonzero(close.index.year >= 2021)
    for g in groups:
        for h in HORIZONS:
            for thr in THRS:
                o = R[(g, h)]
                rm, rb = o[(thr, "EMPMZ")], o[(thr, "CLIMSYM")]
                sel = np.isin(rm["pos"], recent)
                sm = {k: v[sel] for k, v in rm.items()}
                sb = {k: v[sel] for k, v in rb.items()}
                m = sc(sm, sb)
                _, lo, hi, _ = W.block_bootstrap_bss(sm, sb, h, nboot=300)
                print(f"{g:<8}{h:>3}{thr:>6.0%}{m['n']:>10,}{m['bss']:>8.4f}{lo:>8.4f}{hi:>8.4f}"
                      f"{m['bss2']:>8.4f}{m['ece_up']:>8.4f}{m['ece_move']:>8.4f}", flush=True)


if __name__ == "__main__":
    main()
