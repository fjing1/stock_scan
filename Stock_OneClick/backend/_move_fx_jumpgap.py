#!/usr/bin/env python
"""
_move_fx_jumpgap.py -- does the JUMP / OVERNIGHT-vs-INTRADAY factor family improve move_prob.py?

Baseline reproduced from _move_validate.py exactly (same walk-forward, same kappa protocol, same
per-symbol shrunk climatology reference). Every extra factor is fit on strictly prior years.

stages:
  build   -- build + pickle the feature cache (one pass over the cached panel; no downloads)
  base    -- reproduce the shipped baseline numbers
  spec    -- run the factor ladder, report delta BSS2 / delta ECE / years improved
  boot    -- date-block bootstrap CI on the winning deltas
  shape   -- test 4: is the z distribution fatter after a jump day?
  attrib  -- test 5: how much of the h=1 index calibration error is overnight risk?
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _move_data as D
import _move_lib as L
import move_prob as M

T0 = time.time()
np.seterr(all="ignore")

CACHE = Path(__file__).with_name("_move_fx_jumpgap_cache.pkl")
MIN_TRAIN_YEARS = 5
INNER_FOLDS = 5
SHRINK = 40.0
HS = (1, 5, 10, 21)
THRS = (0.02, 0.05)


def clog(x):
    return np.log(np.clip(x, *M.VOL_CLIP))


def rms(s, n):
    return np.sqrt(s.pow(2).rolling(n).mean())


# ---------------------------------------------------------------- feature construction
BASE = ["rv_d", "rv_w", "rv_m", "rv_q", "ewma97"]


def sym_features(c, o, hi, lo, vix):
    """All candidate features at bar t, computable from bars <= t. Returns a dict of np arrays."""
    f = {}
    r = np.log(c).diff()                       # close-to-close log return
    on = np.log(o / c.shift(1))                # overnight gap
    oc = np.log(c / o)                         # intraday open-to-close
    rv1 = L.vol_parkinson(hi, lo, 1)

    # --- shipped baseline block (must match move_prob.build_features bit for bit)
    f["rv_d"] = clog(rv1)
    f["rv_w"] = clog(rv1.rolling(5).mean())
    f["rv_m"] = clog(rv1.rolling(22).mean())
    f["rv_q"] = clog(rv1.rolling(63).mean())
    f["ewma97"] = clog(L.vol_ewma(c, 0.97))
    if vix is not None:
        f["logvix"] = np.log(np.clip(vix.reindex(c.index).ffill(limit=3) / 100 / np.sqrt(252),
                                     *M.VOL_CLIP))

    # --- (1) explicit overnight / intraday decomposition
    f["on_d"] = clog(on.abs())
    f["on_w"] = clog(rms(on, 5))
    f["on_m"] = clog(rms(on, 22))
    f["on_q"] = clog(rms(on, 63))
    f["co_d"] = clog(oc.abs())
    f["co_w"] = clog(rms(oc, 5))
    f["co_m"] = clog(rms(oc, 22))
    f["co_q"] = clog(rms(oc, 63))
    # plain close-to-close (gap-aware but undecomposed) -- the cheap comparator
    f["cc_d"] = clog(r.abs())
    f["cc_w"] = clog(rms(r, 5))
    f["cc_m"] = clog(rms(r, 22))
    f["cc_q"] = clog(rms(r, 63))

    # --- (2) gap ratio as a state variable
    cc21 = rms(r, 21)
    park21 = L.vol_parkinson(hi, lo, 21)
    park63 = L.vol_parkinson(hi, lo, 63)
    f["gr21"] = np.log(np.clip(cc21 / park21, 0.25, 4.0))
    f["gr63"] = np.log(np.clip(rms(r, 63) / park63, 0.25, 4.0))
    ov22 = on.pow(2).rolling(22).mean()
    cv22 = oc.pow(2).rolling(22).mean()
    f["osh"] = (ov22 / (ov22 + cv22)).clip(0.0, 1.0)          # overnight variance share

    # --- (3) jump / bipower decomposition
    bp = (r.abs() * r.abs().shift(1)) * (np.pi / 2)
    bv5, bv22, bv63 = np.sqrt(bp.rolling(5).mean()), np.sqrt(bp.rolling(22).mean()), \
        np.sqrt(bp.rolling(63).mean())
    f["bv_w"], f["bv_m"], f["bv_q"] = clog(bv5), clog(bv22), clog(bv63)
    rv22v, rv63v = rms(r, 22).pow(2), rms(r, 63).pow(2)
    j22 = (rv22v - bv22.pow(2)).clip(lower=0.0)
    j63 = (rv63v - bv63.pow(2)).clip(lower=0.0)
    f["jsh_m"] = (j22 / rv22v).clip(0.0, 1.0)
    f["jsh_q"] = (j63 / rv63v).clip(0.0, 1.0)
    f["jv_m"] = np.log1p(np.sqrt(j22) * math.sqrt(252))        # log(1+jump vol, annualized)
    # standardized size of today's move vs the continuous part -> "jump day" detector
    zj = (r.abs() / bv22).clip(0.0, 25.0)
    f["zj_d"] = np.log(np.clip(zj, 0.02, 25.0))
    f["zj5"] = np.log(np.clip(zj.rolling(5).max(), 0.02, 25.0))
    f["_zj_raw"] = zj                                          # diagnostics only, never a feature
    f["_on2"] = on.pow(2)
    f["_r2"] = r.pow(2)
    return f


ALLF = None


def build():
    p = D.load()
    C, O, H, Lo = p["Close"], p["Open"], p["High"], p["Low"]
    vix = C["^VIX"] if "^VIX" in C.columns else None
    usable = [s for s in C.columns if s != "^VIX" and C[s].notna().sum() >= 500]
    groups = {"index": [s for s in usable if s in M.INDEX_LIKE],
              "single": [s for s in usable if s not in M.INDEX_LIKE]}
    dpos = {d: i for i, d in enumerate(C.index)}
    out = {"dates": C.index}
    for g, syms in groups.items():
        acc, keep_syms = {}, []
        for i, s in enumerate(syms):
            c = C[s].dropna()
            if len(c) < 400:
                continue
            j = len(keep_syms)
            keep_syms.append(s)
            f = sym_features(c, O[s].reindex(c.index), H[s].reindex(c.index),
                             Lo[s].reindex(c.index), vix if g == "index" else None)
            base_ok = np.ones(len(c), bool)
            for k in BASE + (["logvix"] if g == "index" and vix is not None else []):
                base_ok &= np.isfinite(np.asarray(f[k], float))
            for k, v in f.items():
                acc.setdefault(k, []).append(np.asarray(v, float)[base_ok])
            acc.setdefault("yr", []).append(c.index.year.values[base_ok])
            acc.setdefault("sid", []).append(np.full(int(base_ok.sum()), j))
            acc.setdefault("di", []).append(np.array([dpos[d] for d in c.index])[base_ok])
            for h in HS:
                acc.setdefault(f"lr{h}", []).append(
                    np.log(c.shift(-h) / c).values[base_ok])
                acc.setdefault(f"tgt{h}", []).append(
                    clog(L.realized_vol_forward(c, h)).values[base_ok])
        G = {k: np.concatenate(v) for k, v in acc.items()}
        G["syms"] = keep_syms
        out[g] = G
        print(f"  {g:<7} syms={len(keep_syms):>4}  rows={len(G['yr']):>9,}  "
              f"years {G['yr'].min()}-{G['yr'].max()}")
    pd.to_pickle(out, CACHE)
    return out


def load():
    if not CACHE.exists():
        return build()
    return pd.read_pickle(CACHE)


# ---------------------------------------------------------------- walk-forward engine
def design(G, cols, mask):
    n = int(mask.sum())
    X = np.empty((n, len(cols) + 1))
    X[:, 0] = 1.0
    for k, c in enumerate(cols):
        X[:, k + 1] = G[c][mask]
    return X


def ols(X, y):
    keep = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if keep.sum() < 200:
        return None
    b, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
    return b


def bucket_idx(lr, thr):
    a_dn, a_up = np.log(1 - thr), np.log(1 + thr)
    return np.where(lr <= a_dn, 0, np.where(lr <= 0, 1, np.where(lr < a_up, 2, 3)))


def brier4(p, a):
    oh = np.zeros_like(p)
    oh[np.arange(len(a)), a] = 1.0
    return float(((p - oh) ** 2).sum(axis=1).mean())


def run_spec(G, h, cols, rows, thrs=THRS, want_sig=False):
    """Walk-forward the FULL shipped pipeline with feature set `cols` on row subset `rows`.

    Identical protocol to _move_validate.run: OLS log-vol fit, kappa = median of the last 5 inner
    annual folds (each fit on strictly prior years), empirical z table * kappa, 4-bucket probs,
    1e-4 clip + renormalize. One fit per year instead of one per (test year x inner fold): the
    inner-fold fit at year y uses the identical training mask (yr < y) as the test fit at ty = y,
    so it is the same regression -- reusing it is exact, not an approximation."""
    X = design(G, cols, rows)
    lr = G[f"lr{h}"][rows]
    tgt = G[f"tgt{h}"][rows]
    yr = G["yr"][rows]
    sid = G["sid"][rows]
    di = G["di"][rows]
    okrow = np.isfinite(lr)
    X, lr, tgt, yr, sid, di = X[okrow], lr[okrow], tgt[okrow], yr[okrow], sid[okrow], di[okrow]
    fin = np.isfinite(tgt)
    years = sorted(set(yr.tolist()))
    test_years = years[MIN_TRAIN_YEARS:]
    acts = {thr: bucket_idx(lr, thr) for thr in thrs}
    res = {thr: dict(P=[], PS=[], A=[], Y=[], DT=[]) for thr in thrs}
    SIG, KAP = [], {}
    kaps = []
    for y in years:
        tr = yr < y
        ntr = int(tr.sum())
        if ntr < 2000:
            continue
        beta = ols(X[tr & fin], tgt[tr & fin])
        if beta is None:
            continue
        sig_tr = np.exp(X[tr] @ beta) * np.sqrt(h)
        z = np.sort(lr[tr] / sig_tr)
        va = yr == y
        # kappa available to a forecaster standing at the start of year y: inner folds iy < y ONLY.
        # Snapshot BEFORE appending year y's own fold, otherwise year y validates its own kappa.
        kappa = float(np.median(kaps[-INNER_FOLDS:])) if kaps else 1.0
        if ntr >= 5000 and va.sum() >= 50 and len(z) >= 5000:
            kaps.append(M._calibrate_kappa(z, np.exp(X[va] @ beta) * np.sqrt(h), lr[va]))
        if y not in test_years or va.sum() < 50:
            continue
        KAP[y] = kappa
        zt = z * kappa
        sig_te = np.exp(X[va] @ beta) * np.sqrt(h)
        if want_sig:
            SIG.append(sig_te)
        F = lambda v: np.searchsorted(zt, v, side="right") / len(zt)
        f0 = F(0.0)
        for thr in thrs:
            a_dn, a_up = np.log(1 - thr), np.log(1 + thr)
            f_dn, f_up = F(a_dn / sig_te), F(a_up / sig_te)
            p = np.column_stack([f_dn, f0 - f_dn, f_up - f0, 1 - f_up])
            p = np.clip(p, 1e-4, None)
            p /= p.sum(axis=1, keepdims=True)
            a_tr, a_te = acts[thr][tr], acts[thr][va]
            pc = np.bincount(a_tr, minlength=4) / len(a_tr)
            ps = np.empty((int(va.sum()), 4))
            s_te = sid[va]
            for s in np.unique(s_te):
                m = sid[tr] == s
                cnt = np.bincount(a_tr[m], minlength=4).astype(float) if m.any() else np.zeros(4)
                ps[s_te == s] = (cnt + SHRINK * pc) / (cnt.sum() + SHRINK)
            R = res[thr]
            R["P"].append(p); R["PS"].append(ps); R["A"].append(a_te)
            R["Y"].append(np.full(int(va.sum()), y)); R["DT"].append(di[va])
    out = {}
    for thr in thrs:
        R = res[thr]
        if not R["P"]:
            continue
        out[thr] = dict(p=np.vstack(R["P"]), ps=np.vstack(R["PS"]),
                        a=np.concatenate(R["A"]), yr=np.concatenate(R["Y"]),
                        dt=np.concatenate(R["DT"]))
    out["kappa"] = KAP
    if want_sig:
        out["sig"] = np.concatenate(SIG) if SIG else None
    return out


def metrics(d):
    p, ps, a = d["p"], d["ps"], d["a"]
    big = np.isin(a, [0, 3]).astype(float)
    pm, pms = p[:, 0] + p[:, 3], ps[:, 0] + ps[:, 3]
    bs2 = lambda q: float(((q - big) ** 2).mean())
    return dict(n=len(a), bss2=1 - bs2(pm) / bs2(pms), ece=L.ece(pm, big.astype(bool)),
                ece_base=L.ece(pms, big.astype(bool)),
                bss4=1 - brier4(p, a) / brier4(ps, a),
                obs=float(big.mean()), pred=float(pm.mean()))


def per_year_bss2(d):
    p, ps, a, yr = d["p"], d["ps"], d["a"], d["yr"]
    big = np.isin(a, [0, 3]).astype(float)
    pm, pms = p[:, 0] + p[:, 3], ps[:, 0] + ps[:, 3]
    o = {}
    for u in np.unique(yr):
        m = yr == u
        o[int(u)] = 1 - ((pm[m] - big[m]) ** 2).mean() / ((pms[m] - big[m]) ** 2).mean()
    return o


# ---------------------------------------------------------------- stages
def cols_for(g, extra):
    c = list(BASE)
    if g == "index":
        c.append("logvix")
    return c + list(extra)


LADDER = [
    ("base", ()),
    # (1) explicit overnight / intraday decomposition
    ("+on_wmq", ("on_w", "on_m", "on_q")),
    ("+on/co full", ("on_d", "on_w", "on_m", "on_q", "co_d", "co_w", "co_m", "co_q")),
    ("+cc_wmq (cheap)", ("cc_w", "cc_m", "cc_q")),
    # (2) gap ratio state variable
    ("+gr21", ("gr21",)),
    ("+gr21+gr63", ("gr21", "gr63")),
    ("+osh", ("osh",)),
    # (3) jump / bipower
    ("+bv_wmq", ("bv_w", "bv_m", "bv_q")),
    ("+bv+jsh", ("bv_w", "bv_m", "bv_q", "jsh_m", "jsh_q")),
    ("+jsh", ("jsh_m", "jsh_q")),
    ("+jv_m", ("jv_m",)),
    ("+zj_d+zj5", ("zj_d", "zj5")),
    # combined best-of
    ("+on_wmq+gr21", ("on_w", "on_m", "on_q", "gr21")),
    ("+on_wmq+jsh", ("on_w", "on_m", "on_q", "jsh_m", "jsh_q")),
]


def extra_ok(G, ladder=LADDER):
    need = sorted({c for _, e in ladder for c in e})
    m = np.ones(len(G["yr"]), bool)
    for c in need:
        m &= np.isfinite(G[c])
    return m


def stage_base(B):
    print("=== BASELINE REPRODUCTION (must match the shipped table) ===")
    print(f"{'grp':<7}{'h':>3}{'thr':>6}{'n':>10}{'BSS2':>9}{'ECE':>8}{'ECEbase':>9}"
          f"{'obs':>7}{'pred':>7}")
    for g in ("index", "single"):
        G = B[g]
        rows = np.ones(len(G["yr"]), bool)
        for h in HS:
            r = run_spec(G, h, cols_for(g, ()), rows)
            for thr in THRS:
                m = metrics(r[thr])
                print(f"{g:<7}{h:>3}{thr:>6.0%}{m['n']:>10,}{m['bss2']:>+9.4f}{m['ece']:>8.4f}"
                      f"{m['ece_base']:>9.4f}{m['obs']:>7.3f}{m['pred']:>7.3f}")
    print("\n--- same thing on the reduced row set (all extra factors finite) ---")
    for g in ("index", "single"):
        G = B[g]
        rows = extra_ok(G)
        print(f"  {g}: rows {int(rows.sum()):,} of {len(rows):,} "
              f"({100*rows.mean():.2f}% kept)")
        for h in HS:
            r = run_spec(G, h, cols_for(g, ()), rows)
            m2 = metrics(r[0.02])
            print(f"    h={h:<3} thr=2%  n={m2['n']:>9,}  BSS2={m2['bss2']:+.4f}  "
                  f"ECE={m2['ece']:.4f}")


def stage_spec(B, only=None):
    print("=== FACTOR LADDER: delta vs baseline, identical rows, everything refit per year ===")
    store = {}
    for g in ("index", "single"):
        G = B[g]
        rows = extra_ok(G)
        for h in HS:
            base = run_spec(G, h, cols_for(g, ()), rows)
            bm = {thr: metrics(base[thr]) for thr in THRS}
            by = {thr: per_year_bss2(base[thr]) for thr in THRS}
            print(f"\n--- {g} h={h}  (n={bm[0.02]['n']:,}) ---")
            print(f"  {'spec':<18}{'thr':>5}{'BSS2':>9}{'dBSS2':>9}{'ECE':>8}{'dECE':>9}"
                  f"{'yrs+':>7}")
            for thr in THRS:
                print(f"  {'base':<18}{thr:>5.0%}{bm[thr]['bss2']:>+9.4f}{'':>9}"
                      f"{bm[thr]['ece']:>8.4f}")
            for name, ex in LADDER[1:]:
                if only and name not in only:
                    continue
                r = run_spec(G, h, cols_for(g, ex), rows)
                for thr in THRS:
                    m = metrics(r[thr])
                    py = per_year_bss2(r[thr])
                    nplus = sum(1 for y in py if py[y] > by[thr][y])
                    print(f"  {name:<18}{thr:>5.0%}{m['bss2']:>+9.4f}"
                          f"{m['bss2']-bm[thr]['bss2']:>+9.4f}{m['ece']:>8.4f}"
                          f"{m['ece']-bm[thr]['ece']:>+9.4f}{nplus:>4}/{len(py)}")
                    store[(g, h, thr, name)] = dict(bss2=m["bss2"], d=m["bss2"] - bm[thr]["bss2"],
                                                    ece=m["ece"], dece=m["ece"] - bm[thr]["ece"],
                                                    nplus=nplus, nyr=len(py), n=m["n"])
            store[(g, h, 0.02, "base")] = bm[0.02]
            store[(g, h, 0.05, "base")] = bm[0.05]
    pd.to_pickle(store, "/tmp/_move_fx_jumpgap_spec.pkl")


def _boot_delta(a, p1, p2, ps, dt, reps=500, blk=63, seed=17):
    """Date-block bootstrap on the BSS2 delta. Blocks of 63 consecutive trading dates >> h."""
    rng = np.random.default_rng(seed)
    big = np.isin(a, [0, 3]).astype(float)
    q1, q2, qs = p1[:, 0] + p1[:, 3], p2[:, 0] + p2[:, 3], ps[:, 0] + ps[:, 3]
    ud = np.unique(dt)
    order = {d: k for k, d in enumerate(ud)}
    pos = np.array([order[d] for d in dt])
    srt = np.argsort(pos, kind="stable")
    bnd = np.searchsorted(pos[srt], np.arange(len(ud) + 1))
    e1 = ((q1 - big) ** 2)[srt]
    e2 = ((q2 - big) ** 2)[srt]
    es = ((qs - big) ** 2)[srt]
    nb = max(1, len(ud) // blk)
    out = []
    for _ in range(reps):
        st = rng.integers(0, len(ud), nb)
        sel = np.concatenate([np.arange(bnd[s], bnd[min(s + blk, len(ud))]) for s in st])
        d = es[sel].mean()
        out.append((1 - e2[sel].mean() / d) - (1 - e1[sel].mean() / d))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def stage_boot(B, picks=None):
    picks = picks or [("index", 1, "+on/co full"), ("index", 1, "+gr21"),
                      ("single", 1, "+on/co full"), ("single", 1, "+gr21"),
                      ("index", 5, "+on/co full"), ("single", 5, "+on/co full")]
    print("=== DATE-BLOCK BOOTSTRAP on delta BSS2 (500 reps, block=63 dates) ===")
    print(f"  {'grp':<7}{'h':>3}{'spec':<18}{'thr':>5}{'dBSS2':>9}{'CI95':>22}{'ndate':>8}")
    for g, h, name in picks:
        G = B[g]
        rows = extra_ok(G)
        ex = dict(LADDER)[name]
        b = run_spec(G, h, cols_for(g, ()), rows)
        r = run_spec(G, h, cols_for(g, ex), rows)
        for thr in THRS:
            d1, d2 = b[thr], r[thr]
            lo, hi = _boot_delta(d1["a"], d1["p"], d2["p"], d1["ps"], d1["dt"])
            dd = metrics(d2)["bss2"] - metrics(d1)["bss2"]
            print(f"  {g:<7}{h:>3}{name:<18}{thr:>5.0%}{dd:>+9.4f}"
                  f"   [{lo:+.4f},{hi:+.4f}]{len(np.unique(d1['dt'])):>8,}")


def stage_coef(B):
    print("=== COEFFICIENTS, full-sample fit (IN-SAMPLE, for interpretation only) ===")
    for g in ("index", "single"):
        G = B[g]
        rows = extra_ok(G)
        for h in (1, 21):
            for name in ("base", "+on/co full", "+gr21", "+bv+jsh"):
                ex = dict(LADDER)[name]
                cols = cols_for(g, ex)
                lrv = G[f"lr{h}"][rows]
                ok = np.isfinite(lrv)
                X = design(G, cols, rows)[ok]
                y = G[f"tgt{h}"][rows][ok]
                b = ols(X, y)
                yy = y[np.isfinite(y)]
                pr = (X @ b)[np.isfinite(y)]
                r2 = 1 - np.sum((yy - pr) ** 2) / np.sum((yy - yy.mean()) ** 2)
                print(f"  {g:<7} h={h:<3} {name:<14} R2(IN-SAMPLE)={r2:.4f}  const={b[0]:+.3f}  "
                      + "  ".join(f"{c}={v:+.3f}" for c, v in zip(cols, b[1:])))


def stage_shape(B):
    """Test 4: is the z distribution fatter after a jump day?  z is built from the SHIPPED
    baseline sigma, walk-forward, then split by a state variable known at t."""
    print("=== TEST 4: z-SHAPE conditional on a jump day (walk-forward sigma, thr-free) ===")
    for g in ("index", "single"):
        G = B[g]
        rows = extra_ok(G)
        for h in (1, 5, 21):
            r = run_spec(G, h, cols_for(g, ()), rows, thrs=(0.02,), want_sig=True)
            d = r[0.02]
            # rebuild the aligned raw state variable on the scored rows
            sel = _scored_index(G, h, rows)
            zj = G["_zj_raw"][sel]
            lr = G[f"lr{h}"][sel]
            sig = r["sig"]
            z = lr / sig
            jm = zj > 2.5
            def st(v):
                return (len(v), float(np.mean(np.abs(v) > 2)), float(np.mean(np.abs(v) > 3)),
                        float(np.percentile(v, 1)), float(np.percentile(v, 99)),
                        float(np.std(v)))
            a, b = st(z[jm]), st(z[~jm])
            print(f"  {g:<7} h={h:<3} jump-day rows {a[0]:>8,} ({jm.mean():.3f}) | "
                  f"P(|z|>2) {a[1]:.4f} vs {b[1]:.4f}   P(|z|>3) {a[2]:.4f} vs {b[2]:.4f}   "
                  f"sd {a[5]:.3f} vs {b[5]:.3f}   q01 {a[3]:+.2f}/{b[3]:+.2f} "
                  f"q99 {a[4]:+.2f}/{b[4]:+.2f}")
    print("\n--- does a jump-conditional z TABLE improve probabilities? (walk-forward) ---")
    for g in ("index", "single"):
        G = B[g]
        rows = extra_ok(G)
        for h in HS:
            b0, b1 = _cond_z(G, g, h, rows)
            print(f"  {g:<7} h={h:<3} thr=2%  BSS2 pooled-z {b0[0]:+.4f} -> jump-cond-z "
                  f"{b1[0]:+.4f}  ({b1[0]-b0[0]:+.4f})   ECE {b0[1]:.4f} -> {b1[1]:.4f}")


def _scored_index(G, h, rows):
    """Row indices (into the full group arrays) of the rows actually scored by run_spec."""
    idx = np.nonzero(rows)[0]
    lr = G[f"lr{h}"][rows]
    idx = idx[np.isfinite(lr)]
    yr = G["yr"][idx]
    years = sorted(set(G["yr"][rows][np.isfinite(lr)].tolist()))
    test_years = set(years[MIN_TRAIN_YEARS:])
    keep = np.array([y in test_years for y in yr])
    # drop years with <50 rows or <2000 training rows, matching run_spec
    out = []
    for y in sorted(test_years):
        m = yr == y
        if m.sum() < 50 or (yr < y).sum() < 2000:
            continue
        out.append(idx[m])
    return np.concatenate(out)


def _cond_z(G, g, h, rows, thr=0.02, jcut=2.5):
    """Same pipeline, but the empirical z table is split by whether bar t was a jump day."""
    cols = cols_for(g, ())
    X = design(G, cols, rows)
    lr, tgt = G[f"lr{h}"][rows], G[f"tgt{h}"][rows]
    yr, sid, zj = G["yr"][rows], G["sid"][rows], G["_zj_raw"][rows]
    ok = np.isfinite(lr)
    X, lr, tgt, yr, sid, zj = X[ok], lr[ok], tgt[ok], yr[ok], sid[ok], zj[ok]
    fin = np.isfinite(tgt)
    jj = zj > jcut
    years = sorted(set(yr.tolist()))
    act = bucket_idx(lr, thr)
    P0, P1, PS, A = [], [], [], []
    kaps = []
    for y in years:
        tr = yr < y
        if tr.sum() < 2000:
            continue
        beta = ols(X[tr & fin], tgt[tr & fin])
        if beta is None:
            continue
        sig_tr = np.exp(X[tr] @ beta) * np.sqrt(h)
        zall = lr[tr] / sig_tr
        z = np.sort(zall)
        va = yr == y
        kap = float(np.median(kaps[-INNER_FOLDS:])) if kaps else 1.0
        if tr.sum() >= 5000 and va.sum() >= 50 and len(z) >= 5000:
            kaps.append(M._calibrate_kappa(z, np.exp(X[va] @ beta) * np.sqrt(h), lr[va]))
        if y not in set(years[MIN_TRAIN_YEARS:]) or va.sum() < 50:
            continue
        sig_te = np.exp(X[va] @ beta) * np.sqrt(h)
        a_dn, a_up = np.log(1 - thr), np.log(1 + thr)

        def probs(zt, s):
            F = lambda v: np.searchsorted(zt, v, side="right") / len(zt)
            f0 = F(0.0)
            f_dn, f_up = F(a_dn / s), F(a_up / s)
            p = np.column_stack([f_dn, f0 - f_dn, f_up - f0, 1 - f_up])
            p = np.clip(p, 1e-4, None)
            return p / p.sum(axis=1, keepdims=True)

        zt0 = z * kap
        P0.append(probs(zt0, sig_te))
        # conditional tables
        p1 = np.empty((int(va.sum()), 4))
        for flag in (True, False):
            mt, mv = (jj[tr] == flag), (jj[va] == flag)
            if mv.sum() == 0:
                continue
            if mt.sum() < 2000:
                p1[mv] = probs(zt0, sig_te[mv])
                continue
            p1[mv] = probs(np.sort(zall[mt]) * kap, sig_te[mv])
        P1.append(p1)
        a_tr, a_te = act[tr], act[va]
        pc = np.bincount(a_tr, minlength=4) / len(a_tr)
        ps = np.empty((int(va.sum()), 4))
        s_te = sid[va]
        for s in np.unique(s_te):
            m = sid[tr] == s
            cnt = np.bincount(a_tr[m], minlength=4).astype(float) if m.any() else np.zeros(4)
            ps[s_te == s] = (cnt + SHRINK * pc) / (cnt.sum() + SHRINK)
        PS.append(ps); A.append(a_te)
    out = []
    a = np.concatenate(A); ps = np.vstack(PS)
    big = np.isin(a, [0, 3]).astype(float)
    pms = ps[:, 0] + ps[:, 3]
    den = ((pms - big) ** 2).mean()
    for P in (P0, P1):
        p = np.vstack(P)
        pm = p[:, 0] + p[:, 3]
        out.append((1 - ((pm - big) ** 2).mean() / den, L.ece(pm, big.astype(bool))))
    return out


def stage_attrib(B):
    """Test 5: how much of the index h=1 calibration error is overnight risk?"""
    print("=== TEST 5: overnight attribution of the residual h=1 calibration error ===")
    p = D.load()
    C, O = p["Close"], p["Open"]
    for s in ("SPY", "QQQ", "^GSPC"):
        if s not in C.columns:
            continue
        c = C[s].dropna()
        o = O[s].reindex(c.index)
        on = np.log(o / c.shift(1)).dropna()
        r = np.log(c).diff().reindex(on.index)
        oc = np.log(c / o).reindex(on.index)
        hi, lo = p["High"][s].reindex(c.index), p["Low"][s].reindex(c.index)
        pk = L.vol_parkinson(hi, lo, len(on)) if False else None
        share = float((on ** 2).sum() / (r ** 2).sum())
        pkall = math.sqrt(float((np.log(hi / lo) ** 2).reindex(on.index).sum())
                          / (4 * math.log(2)) / len(on))
        ccall = math.sqrt(float((r ** 2).mean()))
        print(f"  {s:<7} n={len(on):,}  overnight variance share {share:.3f}   "
              f"cc vol {ccall*math.sqrt(252)*100:.2f}%ann  parkinson {pkall*math.sqrt(252)*100:.2f}%ann"
              f"  ratio {ccall/pkall:.3f}")
    for g in ("index", "single"):
        G = B[g]
        rows = extra_ok(G)
        for h in (1, 5):
            r = run_spec(G, h, cols_for(g, ()), rows, thrs=(0.02,), want_sig=True)
            d = r[0.02]
            sel = _scored_index(G, h, rows)
            osh = G["osh"][sel]
            p, a = d["p"], d["a"]
            big = np.isin(a, [0, 3]).astype(float)
            pm = p[:, 0] + p[:, 3]
            ece = L.ece(pm, big.astype(bool))
            # conditional bias by trailing overnight-share quintile (state known at t)
            q = np.quantile(osh, np.linspace(0, 1, 6)[1:-1])
            bn = np.digitize(osh, q)
            gaps, ws = [], []
            parts = []
            for k in range(5):
                m = bn == k
                gp = float(big[m].mean() - pm[m].mean())
                gaps.append(gp); ws.append(m.mean())
                parts.append(f"Q{k+1} {osh[m].mean():.2f}: gap {gp:+.4f} (n={int(m.sum()):,})")
            cond = float(np.sum(np.array(ws) * np.abs(gaps)))
            print(f"\n  {g} h={h}: overall ECE {ece:.4f}; mean|bias| across overnight-share "
                  f"quintiles {cond:.4f}  -> {100*cond/max(ece,1e-9):.0f}% of ECE is systematic "
                  f"in the overnight state")
            for s in parts:
                print("      " + s)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "base"
    if stage == "build":
        build()
    else:
        B = load()
        {"base": stage_base, "spec": stage_spec, "boot": stage_boot, "coef": stage_coef,
         "shape": stage_shape, "attrib": stage_attrib}[stage](B)
    print(f"\n[{stage} done in {time.time()-T0:.0f}s]")
