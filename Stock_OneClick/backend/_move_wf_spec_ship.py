#!/usr/bin/env python
"""
_move_wf_spec_ship.py -- produces every constant and every promised number for the FINAL
move_prob.py spec. Reuses the already-built panel cache and harness in _move_wf_spec_final.py.

stages: q1 (VIX for singles?) | ship (coefficients + z tables) | thr (threshold sweep)
        | sub (subgroup calibration, MZ slope, per-symbol risk) | earn | boot | live | tests
"""
from __future__ import annotations
import sys, math, time, json, pickle
import numpy as np
import pandas as pd
import _move_data as D
import _move_lib as L
from _move_wf_spec_final import (FLOOR, CEIL, HS, MIN_TRAIN_YEARS, SPECS, clog, ols, design,
                                 chi_log_corr, ncdf, EmpShape, GridShape, UGRID, probs, bidx,
                                 brier4, brier2, ll, ece_bucket, ece_move, clim_sym, load_built)

T0 = time.time()
np.seterr(all="ignore")
SPECS = dict(SPECS)
SPECS["har6v"] = ("rv_d", "rv_w", "rv_m", "rv_q", "yz21", "e97", "siv")
SPECS["har4v"] = ("rv_d", "rv_w", "rv_m", "yz21", "siv")
SPECS["harc4v"] = ("ac_d", "ac_w", "ac_m", "ac_q", "siv")
SPECS["e94v"] = ("e94", "siv")
FEATNAME = {"rv_d": "rv_d", "rv_w": "rv_w", "rv_m": "rv_m", "rv_q": "rv_q", "yz21": "yz21",
            "e97": "e97", "siv": "s_iv", "e94": "e94", "park21": "park21",
            "ac_d": "ac_d", "ac_w": "ac_w", "ac_m": "ac_m", "ac_q": "ac_q"}


def dsg(g, cols, rows):
    X = np.empty((int(rows.sum()), len(cols) + 1))
    X[:, 0] = 1.0
    for k, c in enumerate(cols):
        X[:, k + 1] = clog(g[c][rows])
    return X


def fitpred(g, spec, h, tr, te):
    cols = SPECS[spec]
    y = clog(g[f"tv{h}"]) + chi_log_corr(h)
    b = ols(dsg(g, cols, tr), y[tr])
    return b


def sig_of(g, spec, b, rows, h):
    """per-day sigma forecast * sqrt(h) == total h-day sigma of log returns"""
    return np.exp(dsg(g, SPECS[spec], rows) @ b) * math.sqrt(h)


def wf_iter(B, grp, h, specs, thr=0.02):
    """yield per-test-year (a_te, sigmas dict, train z dict, extras)"""
    g = B[grp]
    fwd, tv, flog = g[f"fwd{h}"], g[f"tv{h}"], g[f"flog{h}"]
    okS = np.isfinite(fwd) & np.isfinite(flog)
    okV = np.isfinite(tv)
    years = sorted(B["ymap"])
    for y in years[MIN_TRAIN_YEARS:]:
        te = okS & (g["year"] == y)
        if te.sum() < 50:
            continue
        i0 = np.min(g["date_i"][te])
        prior = g["year"] < y
        trV = okV & prior & (g["date_i"] + h < i0)
        trS = okS & prior & (g["date_i"] + h < i0)
        if trV.sum() < 500 or trS.sum() < 500:
            continue
        out = {}
        for sp in specs:
            b = fitpred(g, sp, h, trV, te)
            out[sp] = (sig_of(g, sp, b, trS, h), sig_of(g, sp, b, te, h), b)
        yield y, te, trS, trV, out


def score(p, a):
    return dict(n=len(a), ll=ll(p, a), b4=brier4(p, a), b2=brier2(p, a),
                ece_mv=ece_move(p, a), ece_up=ece_bucket(p, a, 3), ece_dn=ece_bucket(p, a, 0),
                p_mv=float((p[:, 0] + p[:, 3]).mean()), o_mv=float(((a == 0) | (a == 3)).mean()),
                p_up=float(p[:, 3].mean()), o_up=float((a == 3).mean()),
                p_dn=float(p[:, 0].mean()), o_dn=float((a == 0).mean()))


def run_models(B, grp, h, specs, thr=0.02, shapes=("emp",), extra=None):
    """Composed walk-forward. Returns (a, date_i, year, sym_i, sigdict, P)"""
    g = B[grp]
    nsym = len(g["syms"])
    A, DT, YR, SY, SG = [], [], [], [], {}
    P = {}
    for y, te, trS, trV, sig in wf_iter(B, grp, h, specs, thr):
        fwd, flog = g[f"fwd{h}"], g[f"flog{h}"]
        a_te, a_tr = bidx(fwd[te], thr), bidx(fwd[trS], thr)
        pool = np.bincount(a_tr, minlength=4).astype(float); pool /= pool.sum()
        P.setdefault("clim_pool", []).append(np.tile(pool, (int(te.sum()), 1)))
        psym, _ = clim_sym(a_tr, g["sym_i"][trS], g["sym_i"][te], nsym)
        P.setdefault("clim_sym", []).append(psym)
        for sp, (s_tr, s_te, b) in sig.items():
            z_tr = flog[trS] / s_tr
            loc = float(np.median(z_tr))
            q1, q3 = np.percentile(z_tr, [25, 75])
            sc = float((q3 - q1) / 1.349)
            u_tr = (z_tr - loc) / sc
            for sh in shapes:
                F = EmpShape(u_tr) if sh == "emp" else (GridShape(u_tr, UGRID) if sh == "grid" else ncdf)
                lo = 0.0 if sh == "noloc" else loc
                if sh == "noloc":
                    F = EmpShape(u_tr)
                P.setdefault(f"{sp}|{sh}", []).append(probs(s_te, lo, sc, thr, F))
            SG.setdefault(sp, []).append(s_te)
        if extra is not None:
            extra(y, te, trS, sig, g, P)
        A.append(a_te); DT.append(g["date_i"][te]); YR.append(np.full(int(te.sum()), y))
        SY.append(g["sym_i"][te])
    return (np.concatenate(A), np.concatenate(DT), np.concatenate(YR), np.concatenate(SY),
            {k: np.concatenate(v) for k, v in SG.items()}, {k: np.vstack(v) for k, v in P.items()})


# =============================================================== stage q1: does VIX help singles?
def stage_q1(B):
    print("=== Q1: does log(VIX) help, in the COMPOSED pipeline (thr=2%, emp shape)? ===")
    print(f"{'grp':7s} {'h':>3s} {'model':10s} {'n':>10s} {'R2logvol':>9s} {'BSS4':>8s} "
          f"{'BSS2':>8s} {'ECEmv':>7s} {'MZslope':>8s}")
    for grp in ("INDEX", "SINGLE"):
        for h in HS:
            g = B[grp]
            a, dt, yr, sy, SG, P = run_models(B, grp, h, ["har6", "har6v"], shapes=("emp",))
            b4s, b2s = brier4(P["clim_sym"], a), brier2(P["clim_sym"], a)
            # OOS R2 on log vol + MZ slope, scored on the same rows
            tv = g[f"tv{h}"]
            for sp in ("har6", "har6v"):
                # rebuild target on scored rows
                rows_ok = []
                yv, pv = [], []
                for y, te, trS, trV, sig in wf_iter(B, grp, h, [sp]):
                    m = np.isfinite(tv[te])
                    tgt = clog(tv[te][m]) + chi_log_corr(h)
                    prd = np.log(sig[sp][1][m] / math.sqrt(h))
                    trm = float(np.mean(clog(tv[trV]) + chi_log_corr(h)))
                    yv.append(tgt); pv.append(prd); rows_ok.append(np.full(m.sum(), trm))
                yv, pv, bm = np.concatenate(yv), np.concatenate(pv), np.concatenate(rows_ok)
                r2 = 1 - np.sum((yv - pv) ** 2) / np.sum((yv - bm) ** 2)
                X = np.column_stack([np.ones(len(pv)), pv])
                mz = float(np.linalg.lstsq(X, yv, rcond=None)[0][1])
                p = P[f"{sp}|emp"]
                print(f"{grp:7s} {h:3d} {sp:10s} {len(a):10,d} {r2:9.4f} "
                      f"{1-brier4(p,a)/b4s:+8.4f} {1-brier2(p,a)/b2s:+8.4f} "
                      f"{ece_move(p,a):7.4f} {mz:8.4f}")


# =============================================================== stage ship
LADDER = {"INDEX": ["har6v", "har6", "har4v", "har4", "harc4v", "harc4", "e94v", "e94", "park21"],
          "SINGLE": ["har6v", "har6", "har4v", "har4", "harc4v", "harc4", "e94v", "e94", "park21"]}


def stage_ship(B, cut=2026):
    out = {}
    print(f"=== SHIP CONSTANTS: fit on all rows with year < {cut}, purged ===")
    for grp in ("INDEX", "SINGLE"):
        g = B[grp]
        i0 = int(np.min(g["date_i"][g["year"] >= cut])) if (g["year"] >= cut).any() else 10 ** 9
        for h in HS:
            tv, fwd, flog = g[f"tv{h}"], g[f"fwd{h}"], g[f"flog{h}"]
            trV = np.isfinite(tv) & (g["year"] < cut) & (g["date_i"] + h < i0)
            trS = np.isfinite(flog) & (g["year"] < cut) & (g["date_i"] + h < i0)
            for sp in LADDER[grp]:
                b = ols(dsg(g, SPECS[sp], trV), clog(tv[trV]) + chi_log_corr(h))
                out[("coef", grp, h, sp)] = (b, int(trV.sum()))
            # z table from the primary spec
            sp = "har6v" if grp == "INDEX" else "har6"
            b = out[("coef", grp, h, sp)][0]
            s_tr = sig_of(g, sp, b, trS, h)
            z = flog[trS] / s_tr
            loc = float(np.median(z)); q1, q3 = np.percentile(z, [25, 75])
            sc = float((q3 - q1) / 1.349)
            u = np.sort((z - loc) / sc)
            G = np.searchsorted(u, UGRID, side="right") / len(u)
            out[("z", grp, h)] = dict(loc=loc, scale=sc, n=len(z), G=G,
                                      std=float(z.std(ddof=1)), grid=UGRID,
                                      mean=float(z.mean()),
                                      q=[float(np.percentile(z, p)) for p in (1, 5, 25, 50, 75, 95, 99)])
    # print coefficients
    for grp in ("INDEX", "SINGLE"):
        print(f"\n--- {grp} coefficients (order: const, then features) ---")
        for sp in LADDER[grp]:
            print(f"  spec {sp}: [{', '.join(FEATNAME[c] for c in SPECS[sp])}]")
            for h in HS:
                b, n = out[("coef", grp, h, sp)]
                print(f"    h={h:<3d} n_fit={n:>9,d}  " + "  ".join(f"{v:+.4f}" for v in b))
    print("\n--- z tables (log-return z = log(1+r_h) / sigma_hat_h) ---")
    for grp in ("INDEX", "SINGLE"):
        for h in HS:
            d = out[("z", grp, h)]
            print(f"  {grp:7s} h={h:<3d} n={d['n']:>9,d}  loc(median)={d['loc']:+.4f}  "
                  f"scale(IQR/1.349)={d['scale']:.4f}  std={d['std']:.4f}  mean={d['mean']:+.4f}")
            print(f"          pct 1/5/25/50/75/95/99 of z: " + " ".join(f"{v:+.3f}" for v in d["q"]))
    print("\n--- G(u) grid, u = (z-loc)/scale ---")
    print("  u      " + " ".join(f"{v:>7.2f}" for v in UGRID))
    for grp in ("INDEX", "SINGLE"):
        for h in HS:
            d = out[("z", grp, h)]
            print(f"  {grp[:3]} h{h:<2d} " + " ".join(f"{v:7.5f}" for v in d["G"]))
    with open("/tmp/_move_ship.pkl", "wb") as fh:
        pickle.dump(out, fh)
    return out


# =============================================================== stage thr
def stage_thr(B):
    print("=== THRESHOLD SWEEP, recommended spec, walk-forward OOS vs CLIMSYM ===")
    print(f"{'grp':7s} {'h':>3s} {'thr':>5s} {'n':>10s} {'ratio':>6s} {'BSS4':>8s} {'BSS2':>8s} "
          f"{'LLskl':>7s} {'ECEmv':>7s} {'ECEup':>7s} {'ECEdn':>7s} {'p_mv':>6s} {'o_mv':>6s} "
          f"{'nofloor%':>8s}")
    rows = []
    for grp in ("INDEX", "SINGLE"):
        sp = "har6v" if grp == "INDEX" else "har6"
        for h in HS:
            for thr in (0.01, 0.02, 0.03, 0.05, 0.08):
                a, dt, yr, sy, SG, P = run_models(B, grp, h, [sp], thr=thr, shapes=("emp",))
                p = P[f"{sp}|emp"]
                b4s, b2s, lls = brier4(P["clim_sym"], a), brier2(P["clim_sym"], a), ll(P["clim_sym"], a)
                sig = SG[sp]
                ratio = np.log(1 + thr) / sig
                nofl = float(np.mean((p[:, 0] <= 1.01e-4) | (p[:, 3] <= 1.01e-4)))
                r = dict(grp=grp, h=h, thr=thr, n=len(a), ratio=float(np.median(ratio)),
                         bss4=1 - brier4(p, a) / b4s, bss2=1 - brier2(p, a) / b2s,
                         llskill=1 - ll(p, a) / lls, ece_mv=ece_move(p, a),
                         ece_up=ece_bucket(p, a, 3), ece_dn=ece_bucket(p, a, 0),
                         p_mv=float((p[:, 0] + p[:, 3]).mean()),
                         o_mv=float(((a == 0) | (a == 3)).mean()), nofloor=nofl)
                rows.append(r)
                print(f"{grp:7s} {h:3d} {thr:5.0%} {len(a):10,d} {r['ratio']:6.2f} "
                      f"{r['bss4']:+8.4f} {r['bss2']:+8.4f} {r['llskill']:+7.4f} "
                      f"{r['ece_mv']:7.4f} {r['ece_up']:7.4f} {r['ece_dn']:7.4f} "
                      f"{r['p_mv']:6.3f} {r['o_mv']:6.3f} {nofl:8.4f}")
    pd.DataFrame(rows).to_pickle("/tmp/_move_thr.pkl")


# =============================================================== stage sub
def stage_sub(B):
    print("=== SUBGROUP CALIBRATION + PER-SYMBOL RISK (thr=2%, recommended spec) ===")
    store = {}
    for grp in ("INDEX", "SINGLE"):
        sp = "har6v" if grp == "INDEX" else "har6"
        g = B[grp]
        for h in HS:
            a, dt, yr, sy, SG, P = run_models(B, grp, h, [sp], shapes=("emp", "grid", "norm"))
            store[(grp, h)] = (a, dt, yr, sy, SG[sp], P)
    # --- A) own long-run vol bands (singles)
    print("\n--- A) SINGLE, by own long-run annualized vol (cc252 at t * sqrt(252)) ---")
    g = B["SINGLE"]
    bands = [(0, .25), (.25, .35), (.35, .45), (.45, .55), (.55, .70), (.70, .90), (.90, 1.20), (1.20, 9)]
    for h in (1, 5, 21):
        a, dt, yr, sy, sig, P = store[("SINGLE", h)]
        # recover cc252 aligned to the scored rows: re-walk to get row indices
        idx = _scored_rows(B, "SINGLE", h)
        v = g["cc252"][idx] * math.sqrt(252)
        p = P["har6|emp"]
        print(f"  h={h}: {'band':>12s} {'share':>6s} {'n':>9s} {'pmv':>6s} {'omv':>6s} "
              f"{'ratio':>6s} {'err_dn':>7s} {'err_up':>7s} {'BSS4':>8s}")
        for lo, hi in bands:
            m = (v >= lo) & (v < hi)
            if m.sum() < 500:
                continue
            b4s = brier4(P["clim_sym"][m], a[m])
            pm, om = float((p[m, 0] + p[m, 3]).mean()), float(((a[m] == 0) | (a[m] == 3)).mean())
            print(f"        {lo:5.2f}-{hi:<6.2f} {m.mean():6.3f} {int(m.sum()):9,d} {pm:6.3f} "
                  f"{om:6.3f} {om/pm:6.3f} {float((a[m]==0).mean()-p[m,0].mean()):+7.4f} "
                  f"{float((a[m]==3).mean()-p[m,3].mean()):+7.4f} "
                  f"{1-brier4(p[m],a[m])/b4s:+8.4f}")
    # --- B) cross-sectional vol tercile: does conditioning loc/scale help?
    print("\n--- B) SINGLE, cross-sectional sigma_hat tercile: pooled vs conditioned loc/scale ---")
    for h in HS:
        res = _xsec_test(B, "SINGLE", h)
        print(f"  h={h:3d} " + res)
    # --- C) per-symbol BSS4 distribution
    print("\n--- C) SINGLE per-symbol BSS4 vs CLIMSYM (>=250 scored rows) ---")
    print(f"  {'h':>3s} {'nsym':>5s} {'p05':>8s} {'p25':>8s} {'med':>8s} {'p75':>8s} {'p95':>8s} "
          f"{'frac<=0':>8s} {'medECEup':>9s}")
    for h in HS:
        a, dt, yr, sy, sig, P = store[("SINGLE", h)]
        p = P["har6|emp"]
        vals, eces = [], []
        for s in np.unique(sy):
            m = sy == s
            if m.sum() < 250:
                continue
            b4s = brier4(P["clim_sym"][m], a[m])
            vals.append(1 - brier4(p[m], a[m]) / b4s)
            eces.append(ece_bucket(p[m], a[m], 3, nb=5))
        vals = np.array(vals); eces = np.array([e for e in eces if np.isfinite(e)])
        print(f"  {h:3d} {len(vals):5d} " + " ".join(f"{np.percentile(vals,q):+8.4f}"
              for q in (5, 25, 50, 75, 95)) + f" {float((vals<=0).mean()):8.3f} "
              f"{np.median(eces):9.4f}")
    # --- D) per-year and crisis behaviour
    print("\n--- D) per-year BSS4 (thr=2%) sign count + worst years ---")
    for grp in ("INDEX", "SINGLE"):
        sp = "har6v" if grp == "INDEX" else "har6"
        for h in HS:
            a, dt, yr, sy, sig, P = store[(grp, h)]
            p = P[f"{sp}|emp"]
            rec = []
            for y in np.unique(yr):
                m = yr == y
                b4s = brier4(P["clim_sym"][m], a[m])
                rec.append((int(y), 1 - brier4(p[m], a[m]) / b4s,
                            float((p[m, 3]).mean() - (a[m] == 3).mean()),
                            float((p[m, 0] + p[m, 3]).mean() - ((a[m] == 0) | (a[m] == 3)).mean())))
            r = pd.DataFrame(rec, columns=["y", "bss4", "err_up", "err_mv"])
            worst = r.nsmallest(3, "bss4")
            print(f"  {grp:7s} h={h:3d}  pos {int((r.bss4>0).sum())}/{len(r)}  "
                  f"median {r.bss4.median():+.4f}  worst: " +
                  " ".join(f"{int(w.y)}({w.bss4:+.3f},up{w.err_up:+.3f})" for w in worst.itertuples()))
    # --- E) variance floor test
    print("\n--- E) does a hard variance floor on sigma_hat help? (SINGLE, thr=2%) ---")
    for h in (1, 21):
        _floor_test(B, "SINGLE", h)
    with open("/tmp/_move_sub.pkl", "wb") as fh:
        pickle.dump({k: (v[0], v[1], v[2], v[3], v[4]) for k, v in store.items()}, fh)
    return store


def _scored_rows(B, grp, h):
    g = B[grp]
    fwd, flog = g[f"fwd{h}"], g[f"flog{h}"]
    okS = np.isfinite(fwd) & np.isfinite(flog)
    idx = []
    years = sorted(B["ymap"])
    for y in years[MIN_TRAIN_YEARS:]:
        te = okS & (g["year"] == y)
        if te.sum() < 50:
            continue
        i0 = np.min(g["date_i"][te])
        if (np.isfinite(g[f"tv{h}"]) & (g["year"] < y) & (g["date_i"] + h < i0)).sum() < 500:
            continue
        idx.append(np.nonzero(te)[0])
    return np.concatenate(idx)


def _xsec_test(B, grp, h, thr=0.02):
    """Compare pooled loc/scale vs loc/scale conditioned on the cross-sectional sigma_hat tercile."""
    g = B[grp]
    sp = "har6"
    A, Pp, Pc, Ps, TER = [], [], [], [], []
    for y, te, trS, trV, sig in wf_iter(B, grp, h, [sp], thr):
        s_tr, s_te, b = sig[sp]
        flog, fwd = g[f"flog{h}"], g[f"fwd{h}"]
        z_tr = flog[trS] / s_tr
        a_te = bidx(fwd[te], thr)
        # tercile of sigma_hat WITHIN each date (point-in-time cross-section)
        def terc(sv, di):
            o = np.argsort(di, kind="stable")
            t = np.empty(len(sv), dtype=np.int8)
            st = 0
            for e in np.r_[np.nonzero(np.diff(di[o]))[0] + 1, len(o)]:
                ix = o[st:e]
                r = np.argsort(np.argsort(sv[ix]))
                t[ix] = np.minimum((r * 3) // max(len(ix), 1), 2)
                st = e
            return t
        t_tr = terc(s_tr, g["date_i"][trS]); t_te = terc(s_te, g["date_i"][te])
        loc = float(np.median(z_tr)); q1, q3 = np.percentile(z_tr, [25, 75]); sc = (q3 - q1) / 1.349
        Fp = EmpShape((z_tr - loc) / sc)
        Pp.append(probs(s_te, loc, sc, thr, Fp))
        pc = np.empty((int(te.sum()), 4))
        for k in range(3):
            mt, mv = t_tr == k, t_te == k
            if mt.sum() < 200 or mv.sum() == 0:
                pc[mv] = probs(s_te[mv], loc, sc, thr, Fp) if mv.sum() else 0
                continue
            zk = z_tr[mt]
            lk = float(np.median(zk)); a1, a3 = np.percentile(zk, [25, 75]); sk = (a3 - a1) / 1.349
            pc[mv] = probs(s_te[mv], lk, sk, thr, EmpShape((zk - lk) / sk))
        Pc.append(pc)
        pool = np.bincount(bidx(fwd[trS], thr), minlength=4).astype(float); pool /= pool.sum()
        psym, _ = clim_sym(bidx(fwd[trS], thr), g["sym_i"][trS], g["sym_i"][te], len(g["syms"]))
        Ps.append(psym); A.append(a_te); TER.append(t_te)
    a = np.concatenate(A); pp = np.vstack(Pp); pc = np.vstack(Pc); ps = np.vstack(Ps)
    ter = np.concatenate(TER)
    b4s = brier4(ps, a)
    s = (f"n={len(a):>9,d} pooled BSS4 {1-brier4(pp,a)/b4s:+.4f} cond BSS4 "
         f"{1-brier4(pc,a)/b4s:+.4f}  LL {ll(pp,a):.5f}->{ll(pc,a):.5f}  "
         f"pooled err_up by tercile ")
    for k in range(3):
        m = ter == k
        s += f"T{k+1} {float((a[m]==3).mean()-pp[m,3].mean()):+.4f} "
    s += " | cond "
    for k in range(3):
        m = ter == k
        s += f"T{k+1} {float((a[m]==3).mean()-pc[m,3].mean()):+.4f} "
    return s


# =============================================================== stage xs2: SHIPPABLE conditioning
def _band_test(B, grp, h, nb, mode, thr=0.02, ret_const=False):
    """mode='xsec' -> within-date cross-sectional rank of sigma_hat (needs the whole universe)
       mode='abs'  -> absolute bin of sigma_hat_d, edges = train quantiles (per-ticker usable)
    Returns metrics + the shippable (edges, loc[], scale[]) from the LAST fold."""
    g = B[grp]
    sp = "har6v" if grp == "INDEX" else "har6"
    A, Pp, Pc, PS, BN = [], [], [], [], []
    const = None
    for y, te, trS, trV, sig in wf_iter(B, grp, h, [sp], thr):
        s_tr, s_te, b = sig[sp]
        flog, fwd = g[f"flog{h}"], g[f"fwd{h}"]
        z_tr = flog[trS] / s_tr
        if mode == "xsec":
            def rk(sv, di):
                o = np.argsort(di, kind="stable")
                t = np.empty(len(sv), dtype=np.int8); st = 0
                for e in np.r_[np.nonzero(np.diff(di[o]))[0] + 1, len(o)]:
                    ix = o[st:e]
                    r = np.argsort(np.argsort(sv[ix]))
                    t[ix] = np.minimum((r * nb) // max(len(ix), 1), nb - 1); st = e
                return t
            b_tr, b_te = rk(s_tr, g["date_i"][trS]), rk(s_te, g["date_i"][te])
            edges = None
        else:
            sd_tr = s_tr / math.sqrt(h); sd_te = s_te / math.sqrt(h)
            edges = np.percentile(sd_tr, np.linspace(0, 100, nb + 1)[1:-1])
            b_tr = np.digitize(sd_tr, edges); b_te = np.digitize(sd_te, edges)
        loc = float(np.median(z_tr)); q1, q3 = np.percentile(z_tr, [25, 75]); sc = (q3 - q1) / 1.349
        Fp = EmpShape((z_tr - loc) / sc)
        Pp.append(probs(s_te, loc, sc, thr, Fp))
        pc = np.zeros((int(te.sum()), 4)); LO, SC = [], []
        for k in range(nb):
            mt, mv = b_tr == k, b_te == k
            if mt.sum() < 300:
                LO.append(loc); SC.append(sc)
                if mv.any():
                    pc[mv] = probs(s_te[mv], loc, sc, thr, Fp)
                continue
            zk = z_tr[mt]
            lk = float(np.median(zk)); a1, a3 = np.percentile(zk, [25, 75]); sk = (a3 - a1) / 1.349
            LO.append(lk); SC.append(sk)
            if mv.any():
                pc[mv] = probs(s_te[mv], lk, sk, thr, EmpShape((zk - lk) / sk))
        const = (edges, LO, SC, loc, sc)
        Pc.append(pc)
        psym, _ = clim_sym(bidx(fwd[trS], thr), g["sym_i"][trS], g["sym_i"][te], len(g["syms"]))
        PS.append(psym); A.append(bidx(fwd[te], thr)); BN.append(b_te)
    a = np.concatenate(A); pp = np.vstack(Pp); pc = np.vstack(Pc); ps = np.vstack(PS)
    bn = np.concatenate(BN); b4s = brier4(ps, a)
    if ret_const:
        return const
    eu = [float((a[bn == k] == 3).mean() - pc[bn == k, 3].mean()) for k in range(nb)]
    eu0 = [float((a[bn == k] == 3).mean() - pp[bn == k, 3].mean()) for k in range(nb)]
    return dict(n=len(a), bss_pool=1 - brier4(pp, a) / b4s, bss_cond=1 - brier4(pc, a) / b4s,
                ll_pool=ll(pp, a), ll_cond=ll(pc, a), ece_pool=ece_bucket(pp, a, 3),
                ece_cond=ece_bucket(pc, a, 3), eu=eu, eu0=eu0,
                mabs0=float(np.mean(np.abs(eu0))), mabs=float(np.mean(np.abs(eu))))


def stage_xs2(B):
    print("=== SHIPPABLE loc/scale CONDITIONING: xsec-rank vs ABSOLUTE sigma band ===")
    for grp in ("SINGLE", "INDEX"):
        for h in HS:
            for mode in ("xsec", "abs"):
                for nb in (3, 5):
                    if grp == "INDEX" and mode == "xsec":
                        continue
                    r = _band_test(B, grp, h, nb, mode)
                    print(f"  {grp:6s} h={h:3d} {mode:4s} nb={nb} n={r['n']:>9,d} "
                          f"BSS4 {r['bss_pool']:+.4f}->{r['bss_cond']:+.4f}  "
                          f"LL {r['ll_pool']:.5f}->{r['ll_cond']:.5f}  "
                          f"ECEup {r['ece_pool']:.4f}->{r['ece_cond']:.4f}  "
                          f"mean|err_up| {r['mabs0']:.4f}->{r['mabs']:.4f}")
    print("\n--- shippable constants for the chosen form (abs, nb=5, fit on <2026) ---")
    for grp in ("SINGLE", "INDEX"):
        for h in HS:
            e, LO, SC, l0, s0 = _band_test(B, grp, h, 5, "abs", ret_const=True)
            print(f"  {grp:6s} h={h:<3d} pooled loc {l0:+.4f} scale {s0:.4f}")
            print(f"          edges(ann%) " + " ".join(f"{v*math.sqrt(252)*100:.2f}" for v in e))
            print(f"          loc   " + " ".join(f"{v:+.4f}" for v in LO))
            print(f"          scale " + " ".join(f"{v:.4f}" for v in SC))


def _floor_test(B, grp, h, thr=0.02):
    g = B[grp]
    sp = "har6"
    res = {}
    for kf in (0.0, 0.3, 0.5):
        A, P, PS = [], [], []
        for y, te, trS, trV, sig in wf_iter(B, grp, h, [sp], thr):
            s_tr, s_te, b = sig[sp]
            lr_tr = g["cc252"][trS] * math.sqrt(h)
            lr_te = g["cc252"][te] * math.sqrt(h)
            st = np.maximum(s_tr, kf * lr_tr); sv = np.maximum(s_te, kf * lr_te)
            flog, fwd = g[f"flog{h}"], g[f"fwd{h}"]
            z = flog[trS] / st
            loc = float(np.median(z)); q1, q3 = np.percentile(z, [25, 75]); sc = (q3 - q1) / 1.349
            P.append(probs(sv, loc, sc, thr, EmpShape((z - loc) / sc)))
            psym, _ = clim_sym(bidx(fwd[trS], thr), g["sym_i"][trS], g["sym_i"][te], len(g["syms"]))
            PS.append(psym); A.append(bidx(fwd[te], thr))
        a = np.concatenate(A); p = np.vstack(P); ps = np.vstack(PS)
        res[kf] = (1 - brier4(p, a) / brier4(ps, a), ll(p, a), ece_move(p, a))
    print(f"  {grp} h={h}: " + "  ".join(
        f"k={k}: BSS4 {v[0]:+.4f} LL {v[1]:.5f} ECEmv {v[2]:.4f}" for k, v in res.items()))


# =============================================================== stage earn
def stage_earn(B):
    print("=== EARNINGS: residual multiplier against the HAR sigma, + composed OOS gain ===")
    path = "/Users/feijing/github.com/stock_scan/gold_pine_script/pead_earnings_cache.json"
    raw = json.load(open(path))
    g = B["SINGLE"]
    syms = list(g["syms"])
    sidx = {s: i for i, s in enumerate(syms)}
    dates = B["index"]
    dpos = {d.date(): i for i, d in enumerate(dates)}
    ev = {}
    for s, v in raw.items():
        if s not in sidx:
            continue
        ds = v.get("dates", v) if isinstance(v, dict) else v
        rows = []
        for d in ds:
            try:
                ds0 = d[0] if isinstance(d, (list, tuple)) else d
                t = pd.Timestamp(str(ds0)[:10]).date()
            except Exception:
                continue
            if t in dpos:
                rows.append(dpos[t])
        if len(rows) >= 5:
            ev[sidx[s]] = np.array(sorted(set(rows)))
    print(f"  earnings coverage: {len(ev)} of {len(syms)} single names "
          f"({sum(len(v) for v in ev.values()):,} events)")
    cov = {k: (v.min(), v.max()) for k, v in ev.items()}
    nd = len(dates)
    flagm = {}
    for h in HS:
        F = np.zeros((nd, len(syms)), dtype=bool)
        for j, bs in ev.items():
            for b in bs:
                # reaction bar b in [t+1,t+h] -> t in [b-h,b-1];  bar b+1 in [t+1,t+h] -> t in [b-h+1,b]
                F[max(b - h, 0):b + 1, j] = True
        flagm[h] = F
    for h in HS:
        F = flagm[h][:, :]
        fl = F[g["date_i"], g["sym_i"]]
        inc = np.zeros(len(fl), dtype=bool)
        for j, (a0, a1) in cov.items():
            m = g["sym_i"] == j
            inc |= m & (g["date_i"] >= a0) & (g["date_i"] <= a1)
        okv = np.isfinite(g[f"tv{h}"]) & inc
        # residual multiplier: realized fwd vol / HAR forecast, flagged vs unflagged, per symbol
        a, dt, yr, sy, SG, P = run_models(B, "SINGLE", h, ["har6"], shapes=("emp",))
        idx = _scored_rows(B, "SINGLE", h)
        sig = SG["har6"]
        flS, incS = fl[idx], inc[idx]
        tvS = g[f"tv{h}"][idx] * math.sqrt(h)
        mult = []
        for j in ev:
            m = incS & (g["sym_i"][idx] == j) & np.isfinite(tvS)
            if (m & flS).sum() < 20 or (m & ~flS).sum() < 100:
                continue
            r = tvS[m] / sig[m]
            mult.append(np.median(r[flS[m]]) / np.median(r[~flS[m]]))
        p = P["har6|emp"]
        b4s = brier4(P["clim_sym"], a)
        mf = incS & flS
        mu = incS & ~flS
        print(f"  h={h:3d} n_sym={len(mult):3d} residual sigma mult (median over syms) "
              f"{np.median(mult):.3f}  [p25 {np.percentile(mult,25):.3f} p75 "
              f"{np.percentile(mult,75):.3f}]  frac>1 {np.mean(np.array(mult)>1):.2f}")
        print(f"        flagged n={int(mf.sum()):>7,d} pred|mv| {float((p[mf,0]+p[mf,3]).mean()):.4f} "
              f"obs {float(((a[mf]==0)|(a[mf]==3)).mean()):.4f} gap "
              f"{float(((a[mf]==0)|(a[mf]==3)).mean()-(p[mf,0]+p[mf,3]).mean()):+.4f} | "
              f"unflagged n={int(mu.sum()):>7,d} gap "
              f"{float(((a[mu]==0)|(a[mu]==3)).mean()-(p[mu,0]+p[mu,3]).mean()):+.4f}")
        # composed test with the multiplier applied, fit on prior years
        _earn_apply(B, h, fl, inc)


def _earn_apply(B, h, fl, inc, thr=0.02):
    g = B["SINGLE"]
    sp = "har6"
    A, P0, P1, PS = [], [], [], []
    mults = []
    for y, te, trS, trV, sig in wf_iter(B, "SINGLE", h, [sp], thr):
        s_tr, s_te, b = sig[sp]
        flog, fwd, tv = g[f"flog{h}"], g[f"fwd{h}"], g[f"tv{h}"]
        # multiplier from TRAIN rows only (ratio of median |z| flagged vs unflagged)
        mt = inc[trS]; ft = fl[trS]
        zz = np.abs(flog[trS] / s_tr)
        if (mt & ft).sum() > 200 and (mt & ~ft).sum() > 200:
            mm = float(np.median(zz[mt & ft]) / np.median(zz[mt & ~ft]))
        else:
            mm = 1.0
        mults.append(mm)
        adj_tr = np.where(inc[trS] & fl[trS], mm, 1.0)
        adj_te = np.where(inc[te] & fl[te], mm, 1.0)
        for tag, at, av, P in (("base", 1.0, 1.0, P0), ("earn", adj_tr, adj_te, P1)):
            st, sv = s_tr * at, s_te * av
            z = flog[trS] / st
            loc = float(np.median(z)); q1, q3 = np.percentile(z, [25, 75]); sc = (q3 - q1) / 1.349
            P.append(probs(sv, loc, sc, thr, EmpShape((z - loc) / sc)))
        psym, _ = clim_sym(bidx(fwd[trS], thr), g["sym_i"][trS], g["sym_i"][te], len(g["syms"]))
        PS.append(psym); A.append(bidx(fwd[te], thr))
    a = np.concatenate(A); p0 = np.vstack(P0); p1 = np.vstack(P1); ps = np.vstack(PS)
    idx = _scored_rows(B, "SINGLE", h)
    m = inc[idx]
    f = m & fl[idx]
    b4s = brier4(ps, a)
    print(f"        WF multiplier (median of {len(mults)} annual fits) {np.median(mults):.3f}; "
          f"COVERED subset n={int(m.sum()):,}: BSS4 {1-brier4(p0[m],a[m])/brier4(ps[m],a[m]):+.4f} -> "
          f"{1-brier4(p1[m],a[m])/brier4(ps[m],a[m]):+.4f}   LL {ll(p0[m],a[m]):.5f} -> "
          f"{ll(p1[m],a[m]):.5f}   ON FLAGGED n={int(f.sum()):,}: LL {ll(p0[f],a[f]):.5f} -> "
          f"{ll(p1[f],a[f]):.5f} (clim {ll(ps[f],a[f]):.5f})")


# =============================================================== stage boot
def stage_boot(B, reps=400):
    print(f"=== DATE-BLOCK BOOTSTRAP CIs ({reps} reps, block=63 dates, cluster by DATE) ===")
    rng = np.random.default_rng(7)
    print(f"{'grp':7s} {'h':>3s} {'win':>6s} {'metric':8s} {'point':>9s} {'lo':>9s} {'hi':>9s}")
    for grp in ("INDEX", "SINGLE"):
        sp = "har6v" if grp == "INDEX" else "har6"
        for h in HS:
            a, dt, yr, sy, SG, P = run_models(B, grp, h, [sp], shapes=("emp",))
            p, ps = P[f"{sp}|emp"], P["clim_sym"]
            for win, mask in (("all", np.ones(len(a), bool)), ("2021+", yr >= 2021)):
                ud = np.unique(dt[mask])
                # block bootstrap over dates
                blocks = 63
                starts = np.arange(0, len(ud))
                order = {d: k for k, d in enumerate(ud)}
                pos = np.array([order[d] for d in dt[mask]])
                bysort = np.argsort(pos, kind="stable")
                pos_s = pos[bysort]
                bnd = np.searchsorted(pos_s, np.arange(len(ud) + 1))
                am, pm, psm = a[mask][bysort], p[mask][bysort], ps[mask][bysort]
                nb = max(1, len(ud) // blocks)
                out4, out2 = [], []
                for _ in range(reps):
                    st = rng.integers(0, len(ud), nb)
                    sel = np.concatenate([np.arange(bnd[s], bnd[min(s + blocks, len(ud))])
                                          for s in st])
                    if len(sel) < 100:
                        continue
                    out4.append(1 - brier4(pm[sel], am[sel]) / brier4(psm[sel], am[sel]))
                    out2.append(1 - brier2(pm[sel], am[sel]) / brier2(psm[sel], am[sel]))
                for nm, v, pt in (("BSS4", out4, 1 - brier4(pm, am) / brier4(psm, am)),
                                  ("BSS2", out2, 1 - brier2(pm, am) / brier2(psm, am))):
                    q = np.percentile(v, [2.5, 97.5])
                    print(f"{grp:7s} {h:3d} {win:>6s} {nm:8s} {pt:+9.4f} {q[0]:+9.4f} {q[1]:+9.4f}")


# =============================================================== stage live
def stage_live(B):
    print("=== LIVE WORKED EXAMPLE at the last closed bar ===")
    ship = pickle.load(open("/tmp/_move_ship.pkl", "rb"))
    p = D.load()
    dates = p["Close"].index
    print(f"  last bar {dates[-1].date()}")
    for grp, names in (("INDEX", ["SPY", "QQQ"]), ("SINGLE", ["AAPL", "NVDA", "KO", "XOM", "BLNK"])):
        g = B[grp]
        sp = "har6v" if grp == "INDEX" else "har6"
        last = int(len(dates) - 1)
        for nm in names:
            if nm not in g["syms"]:
                print(f"  {nm}: not in {grp} panel"); continue
            j = list(g["syms"]).index(nm)
            m = (g["sym_i"] == j)
            if not m.any():
                continue
            di = g["date_i"][m]
            k = np.nonzero(m)[0][np.argmax(di)]
            print(f"\n  --- {nm} ({grp}) as of {dates[g['date_i'][k]].date()} ---")
            fs = "  ".join(f"{FEATNAME[c]}={g[c][k]*math.sqrt(252)*100:.1f}%" for c in SPECS[sp])
            print(f"    features (annualized): {fs}")
            for h in HS:
                b = ship[("coef", grp, h, sp)][0]
                x = np.r_[1.0, [math.log(min(max(g[c][k], FLOOR), CEIL)) for c in SPECS[sp]]]
                sd = math.exp(float(x @ b))
                sh = sd * math.sqrt(h)
                zt = ship[("z", grp, h)]
                u = np.sort(np.array([]))  # not needed; use grid
                GS = _grid_shape(zt)
                for thr in (0.02,):
                    pr = probs(np.array([sh]), zt["loc"], zt["scale"], thr, GS)[0]
                    print(f"    h={h:<3d} sigma_d={sd*math.sqrt(252)*100:5.2f}%ann  "
                          f"sigma_h={sh*100:5.2f}%  k={math.log(1+thr)/sh:5.2f}  "
                          f"P(dn>2%)={pr[0]:.3f} P(-2..0)={pr[1]:.3f} P(0..+2)={pr[2]:.3f} "
                          f"P(up>2%)={pr[3]:.3f}  P(|mv|>2%)={pr[0]+pr[3]:.3f}  "
                          f"dn/up={pr[0]/max(pr[3],1e-9):.2f}")


def _grid_shape(zt):
    grid, G = zt["grid"], np.maximum.accumulate(np.clip(zt["G"], 1e-6, 1 - 1e-6))
    bl = (grid[1] - grid[0]) / max(math.log(G[1] / G[0]), 1e-9)
    br = (grid[-1] - grid[-2]) / max(math.log((1 - G[-2]) / (1 - G[-1])), 1e-9)

    def F(x):
        x = np.asarray(x, float)
        out = np.interp(x, grid, G)
        out = np.where(x < grid[0], G[0] * np.exp((x - grid[0]) / bl), out)
        out = np.where(x > grid[-1], 1 - (1 - G[-1]) * np.exp(-(x - grid[-1]) / br), out)
        return np.clip(out, 1e-9, 1 - 1e-9)
    return F




# =============================================================== stage final: the SHIPPED pipeline
def earn_flags(B):
    path = "/Users/feijing/github.com/stock_scan/gold_pine_script/pead_earnings_cache.json"
    raw = json.load(open(path))
    g = B["SINGLE"]; syms = list(g["syms"]); sidx = {s: i for i, s in enumerate(syms)}
    dpos = {d.date(): i for i, d in enumerate(B["index"])}
    ev = {}
    for s, v in raw.items():
        if s not in sidx:
            continue
        rows = []
        for d in (v.get("dates", v) if isinstance(v, dict) else v):
            try:
                t = pd.Timestamp(str(d[0] if isinstance(d, (list, tuple)) else d)[:10]).date()
            except Exception:
                continue
            if t in dpos:
                rows.append(dpos[t])
        if len(rows) >= 5:
            ev[sidx[s]] = np.array(sorted(set(rows)))
    nd = len(B["index"])
    inc = np.zeros(len(g["date_i"]), bool)
    for j, bs in ev.items():
        inc |= (g["sym_i"] == j) & (g["date_i"] >= bs.min()) & (g["date_i"] <= bs.max())
    fl = {}
    for h in HS:
        F = np.zeros((nd, len(syms)), bool)
        for j, bs in ev.items():
            for b in bs:
                F[max(b - h, 0):b + 1, j] = True
        fl[h] = F[g["date_i"], g["sym_i"]]
    return ev, inc, fl


def shipped(B, grp, h, thr=0.02, nb=5, use_band=None, use_earn=True, shape="grid",
            band_shape=False, EF=None):
    """Walk-forward run of the exact shipped pipeline. Returns (a, dt, yr, sy, sig, p, ps, pp, bn)."""
    g = B[grp]
    sp = "har6v" if grp == "INDEX" else "har6"
    if use_band is None:
        use_band = (grp == "SINGLE")
    A, DT, YR, SY, SG, P, PS, PP, BN, M = [], [], [], [], [], [], [], [], [], []
    inc = fl = None
    if grp == "SINGLE" and use_earn and EF is not None:
        _, inc, flh = EF; fl = flh[h]
    for y, te, trS, trV, sig in wf_iter(B, grp, h, [sp], thr):
        s_tr, s_te, b = sig[sp]
        flog, fwd = g[f"flog{h}"], g[f"fwd{h}"]
        if fl is not None:
            zz = np.abs(flog[trS] / s_tr); mt = inc[trS]; ft = fl[trS]
            mm = (float(np.median(zz[mt & ft]) / np.median(zz[mt & ~ft]))
                  if (mt & ft).sum() > 200 and (mt & ~ft).sum() > 200 else 1.0)
            s_tr = s_tr * np.where(inc[trS] & ft, mm, 1.0)
            s_te = s_te * np.where(inc[te] & fl[te], mm, 1.0)
            M.append(mm)
        z_tr = flog[trS] / s_tr
        loc0 = float(np.median(z_tr)); q1, q3 = np.percentile(z_tr, [25, 75]); sc0 = (q3 - q1) / 1.349
        mk = lambda u: (GridShape(u, UGRID) if shape == "grid"
                        else (EmpShape(u) if shape == "emp" else ncdf))
        F0 = mk((z_tr - loc0) / sc0)
        if not use_band:
            p = probs(s_te, loc0, sc0, thr, F0)
            bt = np.zeros(int(te.sum()), np.int8)
        else:
            ed = np.percentile(s_tr / math.sqrt(h), np.linspace(0, 100, nb + 1)[1:-1])
            b_tr = np.digitize(s_tr / math.sqrt(h), ed); bt = np.digitize(s_te / math.sqrt(h), ed)
            p = np.zeros((int(te.sum()), 4))
            for k in range(nb):
                m1, m2 = b_tr == k, bt == k
                if not m2.any():
                    continue
                if m1.sum() < 300:
                    p[m2] = probs(s_te[m2], loc0, sc0, thr, F0); continue
                zk = z_tr[m1]
                lk = float(np.median(zk)); a1, a3 = np.percentile(zk, [25, 75]); sk = (a3 - a1) / 1.349
                Fk = mk((zk - lk) / sk) if band_shape else F0
                p[m2] = probs(s_te[m2], lk, sk, thr, Fk)
        a_tr = bidx(fwd[trS], thr)
        pool = np.bincount(a_tr, minlength=4).astype(float); pool /= pool.sum()
        psym, _ = clim_sym(a_tr, g["sym_i"][trS], g["sym_i"][te], len(g["syms"]))
        A.append(bidx(fwd[te], thr)); DT.append(g["date_i"][te]); SY.append(g["sym_i"][te])
        YR.append(np.full(int(te.sum()), y)); SG.append(s_te); P.append(p); PS.append(psym)
        PP.append(np.tile(pool, (int(te.sum()), 1))); BN.append(bt)
    return (np.concatenate(A), np.concatenate(DT), np.concatenate(YR), np.concatenate(SY),
            np.concatenate(SG), np.vstack(P), np.vstack(PS), np.vstack(PP), np.concatenate(BN),
            (float(np.median(M)) if M else 1.0))


def _boot(a, p, ps, dt, mask, reps=400, blk=63, seed=11):
    rng = np.random.default_rng(seed)
    ud = np.unique(dt[mask]); order = {d: k for k, d in enumerate(ud)}
    pos = np.array([order[d] for d in dt[mask]]); srt = np.argsort(pos, kind="stable")
    bnd = np.searchsorted(pos[srt], np.arange(len(ud) + 1))
    am, pm, sm = a[mask][srt], p[mask][srt], ps[mask][srt]
    nbk = max(1, len(ud) // blk); o4, o2 = [], []
    for _ in range(reps):
        st = rng.integers(0, len(ud), nbk)
        sel = np.concatenate([np.arange(bnd[s], bnd[min(s + blk, len(ud))]) for s in st])
        o4.append(1 - brier4(pm[sel], am[sel]) / brier4(sm[sel], am[sel]))
        o2.append(1 - brier2(pm[sel], am[sel]) / brier2(sm[sel], am[sel]))
    return np.percentile(o4, [2.5, 97.5]), np.percentile(o2, [2.5, 97.5])


def stage_final(B):
    EF = earn_flags(B)
    print("=== THE SHIPPED PIPELINE, walk-forward OOS, thr=2% ===")
    print("variant key: band=5 abs sigma bands for loc/scale (SINGLE only), earn=earnings sigma mult,"
          " grid=shipped G table")
    hdr = (f"{'grp':7s} {'h':>3s} {'variant':22s} {'n':>9s} {'BSS4':>8s} {'BSS2':>8s} "
           f"{'LLskl':>7s} {'BSS4p':>8s} {'ECEmv':>7s} {'ECEup':>7s} {'ECEdn':>7s} {'mult':>5s}")
    print(hdr)
    keep = {}
    for grp in ("INDEX", "SINGLE"):
        for h in HS:
            variants = [("plain grid", dict(use_band=False, use_earn=False)),
                        ("band", dict(use_band=True, use_earn=False)),
                        ("band+bandshape", dict(use_band=True, use_earn=False, band_shape=True))]
            if grp == "SINGLE":
                variants += [("band+earn (SHIPPED)", dict(use_band=True, use_earn=True))]
            else:
                variants = variants[:1] + [("band", dict(use_band=True, use_earn=False))]
                variants[0] = ("plain grid (SHIPPED)", dict(use_band=False, use_earn=False))
            for nm, kw in variants:
                a, dt, yr, sy, sig, p, ps, pp, bn, mm = shipped(B, grp, h, EF=EF, **kw)
                print(f"{grp:7s} {h:3d} {nm:22s} {len(a):9,d} "
                      f"{1-brier4(p,a)/brier4(ps,a):+8.4f} {1-brier2(p,a)/brier2(ps,a):+8.4f} "
                      f"{1-ll(p,a)/ll(ps,a):+7.4f} {1-brier4(p,a)/brier4(pp,a):+8.4f} "
                      f"{ece_move(p,a):7.4f} {ece_bucket(p,a,3):7.4f} {ece_bucket(p,a,0):7.4f} "
                      f"{mm:5.3f}")
                if "SHIPPED" in nm:
                    keep[(grp, h)] = (a, dt, yr, sy, sig, p, ps, pp, bn)
    print("\n=== SHIPPED MODEL: windows, CIs, non-overlap  (thr=2%) ===")
    print(f"{'grp':7s} {'h':>3s} {'window':10s} {'n':>9s} {'ndate':>6s} {'BSS4':>8s} "
          f"{'CI4':>18s} {'BSS2':>8s} {'CI2':>18s} {'ECEmv':>7s}")
    for (grp, h), (a, dt, yr, sy, sig, p, ps, pp, bn) in sorted(keep.items()):
        for wnm, mask in (("all", np.ones(len(a), bool)), ("2021+", yr >= 2021),
                          ("nonoverlap", np.isin(dt, np.unique(dt)[::max(h, 1)]))):
            c4, c2 = _boot(a, p, ps, dt, mask, reps=300)
            print(f"{grp:7s} {h:3d} {wnm:10s} {int(mask.sum()):9,d} "
                  f"{len(np.unique(dt[mask])):6,d} "
                  f"{1-brier4(p[mask],a[mask])/brier4(ps[mask],a[mask]):+8.4f} "
                  f"[{c4[0]:+.4f},{c4[1]:+.4f}] "
                  f"{1-brier2(p[mask],a[mask])/brier2(ps[mask],a[mask]):+8.4f} "
                  f"[{c2[0]:+.4f},{c2[1]:+.4f}] {ece_move(p[mask],a[mask]):7.4f}")
    print("\n=== RELIABILITY of P(|move|>=2%), shipped model, 10 equal-count bins ===")
    for (grp, h), (a, dt, yr, sy, sig, p, ps, pp, bn) in sorted(keep.items()):
        r = L.reliability(p[:, 0] + p[:, 3], ((a == 0) | (a == 3)).astype(float), 10)
        print(f"  {grp:7s} h={h:<3d} pred " + " ".join(f"{v:.3f}" for v in r.mean_pred) )
        print(f"  {'':7s} {'':5s} obs  " + " ".join(f"{v:.3f}" for v in r.observed))
    print("\n=== up/down split honesty: predicted vs observed up_big share of the big mass ===")
    for (grp, h), (a, dt, yr, sy, sig, p, ps, pp, bn) in sorted(keep.items()):
        recs = []
        for y in np.unique(yr):
            m = yr == y
            pu = p[m, 3].sum() / (p[m, 0] + p[m, 3]).sum()
            ou = (a[m] == 3).sum() / max(((a[m] == 0) | (a[m] == 3)).sum(), 1)
            recs.append((int(y), pu, ou, pu - ou))
        d = pd.DataFrame(recs, columns=["y", "pred", "obs", "err"])
        w = d.iloc[d.err.abs().values.argsort()[::-1][:3]]
        print(f"  {grp:7s} h={h:<3d} mean|err| {d.err.abs().mean():.4f}  worst: " +
              " ".join(f"{int(r.y)} {r.pred:.3f}v{r.obs:.3f}({r.err:+.3f})" for r in w.itertuples()))
    with open("/tmp/_move_final.pkl", "wb") as fh:
        pickle.dump(keep, fh)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "ship"
    B = load_built()
    fn = {"q1": stage_q1, "ship": stage_ship, "thr": stage_thr, "sub": stage_sub,
          "earn": stage_earn, "boot": stage_boot, "live": stage_live,
          "xs2": stage_xs2, "final": stage_final}[stage]
    fn(B)
    print(f"\n[{stage} done in {time.time()-T0:.0f}s]")
