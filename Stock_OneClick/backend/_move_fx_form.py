"""
_move_fx_form.py -- MODEL FORM study for move_prob.py.

Question: is the shipped two-stage plug-in (HAR log-vol OLS -> empirical z table) leaving
accuracy on the table purely through its FUNCTIONAL FORM, using the same inputs?

Variants tested (all strictly walk-forward, every parameter refit on prior calendar years):
  BASE    shipped: OLS log-sigma HAR + pooled empirical z table * kappa
  LOGIT   direct binary logistic (IRLS) on the same log features, per (h, thr)
  MNL     4-class multinomial logit (Newton) on the same log features
  PLATT   BASE then logistic recalibration of logit(p_base) on train years
  ISO     BASE then isotonic (PAVA) recalibration of p_base on train years
  SQ      BASE + squared terms and short*long interaction in the log-vol regression
  SPL     BASE + piecewise-linear (knot) Mincer-Zarnowitz correction on log sigma_hat
  FE      BASE + per-symbol shrunk intercept in the log-vol regression
  GARCH   GARCH(1,1) MLE (numpy, no arch) replacing stage 1, index group
  GJR     GJR-GARCH(1,1) MLE replacing stage 1, index group

Scoring: BSS2 = Brier skill on P(|move| >= thr) vs each ticker's OWN train-year base rate
shrunk 40 pseudo-counts toward pooled; ECE on the same probability; per-test-year win counts;
date-block bootstrap CIs (block > h) for the headline deltas.

Usage:  ../../vcp_env/bin/python _move_fx_form.py <stage>
stages: cache | base | main | garch | coh | boot
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _move_lib as L
import move_prob as M

CACHE = Path(__file__).with_name("_move_fx_form_cache.pkl")
RES = Path(__file__).with_name("_move_fx_form_res.pkl")
HORIZONS = (1, 5, 10, 21)
THRS = (0.02, 0.05)
MIN_TRAIN_YEARS = 5
T0 = time.time()
np.seterr(all="ignore")


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ cache
def build_cache():
    """Exactly move_prob.build_cache's rows, plus the date position `dt` (for purging and for
    date-clustered bootstrap) and the per-row symbol id."""
    import _move_data as D
    panel = D.load()
    C, H, Lo = panel["Close"], panel["High"], panel["Low"]
    vix = C["^VIX"]
    master = C.index
    pos_of = pd.Series(np.arange(len(master)), index=master)

    usable = [s for s in C.columns if s != "^VIX" and C[s].notna().sum() >= 500]
    groups = {"index": [s for s in usable if s in M.INDEX_LIKE],
              "single": [s for s in usable if s not in M.INDEX_LIKE]}
    cache = {}
    for g, syms in groups.items():
        cols = M.FEATS_OHLC + (["logvix"] if g == "index" else [])
        acc = {h: dict(X=[], lr=[], tgt=[], yr=[], sid=[], dt=[]) for h in HORIZONS}
        kept = []
        for i, s in enumerate(syms):
            c = C[s].dropna()
            if len(c) < 400:
                continue
            kept.append(s)
            f = M.build_features(c, H[s].reindex(c.index), Lo[s].reindex(c.index), vix)
            Xi = np.column_stack([np.ones(len(f))] + [f[cc].values for cc in cols])
            p = pos_of.reindex(c.index).values
            yrv = c.index.year.values
            for h in HORIZONS:
                li = np.log(c.shift(-h) / c).values
                ti = M._clip_log(L.realized_vol_forward(c, h)).values
                ok = np.isfinite(Xi).all(axis=1) & np.isfinite(li)
                a = acc[h]
                a["X"].append(Xi[ok]); a["lr"].append(li[ok]); a["tgt"].append(ti[ok])
                a["yr"].append(yrv[ok]); a["dt"].append(p[ok])
                a["sid"].append(np.full(int(ok.sum()), len(kept) - 1, np.int32))
        for h in HORIZONS:
            a = acc[h]
            cache[(g, h)] = dict(X=np.vstack(a["X"]),
                                 lr=np.concatenate(a["lr"]),
                                 tgt=np.concatenate(a["tgt"]),
                                 yr=np.concatenate(a["yr"]).astype(np.int16),
                                 sid=np.concatenate(a["sid"]),
                                 dt=np.concatenate(a["dt"]).astype(np.int32),
                                 cols=cols, symbols=kept)
            log(f"  {g:<7} h={h:<3} rows={len(cache[(g,h)]['lr']):>9,} nsym={len(kept)}")
    cache["_master"] = master
    pd.to_pickle(cache, CACHE)
    return cache


def load_cache():
    if not CACHE.exists():
        return build_cache()
    return pd.read_pickle(CACHE)


# ------------------------------------------------------------------ estimators
def ols(X, y):
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    b, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    return b


def irls_logit(X, y, w=None, ridge=1e-6, iters=25, tol=1e-8):
    """Binary logistic by IRLS / Newton, numpy only. Returns coefficient vector."""
    n, k = X.shape
    b = np.zeros(k)
    b[0] = math.log(max(y.mean(), 1e-6) / max(1 - y.mean(), 1e-6))
    R = ridge * np.eye(k)
    R[0, 0] = 0.0
    for _ in range(iters):
        eta = X @ b
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
        wgt = np.clip(p * (1 - p), 1e-8, None)
        if w is not None:
            wgt = wgt * w
        g = X.T @ ((y - p) if w is None else w * (y - p))
        Hs = X.T @ (X * wgt[:, None]) + R
        try:
            step = np.linalg.solve(Hs, g)
        except np.linalg.LinAlgError:
            break
        b = b + step
        if np.max(np.abs(step)) < tol:
            break
    return b


def predict_logit(X, b):
    return 1.0 / (1.0 + np.exp(-np.clip(X @ b, -35, 35)))


def softmax_newton(X, Y4, ridge=1e-5, iters=30, tol=1e-7):
    """4-class multinomial logit, class 3 as reference. Block Newton (one class at a time,
    using the standard diagonal-dominant bound Hessian 0.5*X'X per class -- stable and cheap)."""
    n, k = X.shape
    K = Y4.shape[1]
    B = np.zeros((k, K - 1))
    base = np.bincount(Y4.argmax(1), minlength=K).astype(float) + 1
    base /= base.sum()
    B[0, :] = np.log(base[:K - 1] / base[K - 1])
    R = ridge * np.eye(k)
    R[0, 0] = 0.0
    XtX = X.T @ X
    Hb = np.linalg.inv(0.5 * XtX + R)      # bound Hessian, shared across classes
    for _ in range(iters):
        eta = np.column_stack([X @ B, np.zeros(n)])
        eta -= eta.max(axis=1, keepdims=True)
        E = np.exp(eta)
        P = E / E.sum(axis=1, keepdims=True)
        G = X.T @ (Y4[:, :K - 1] - P[:, :K - 1]) - R @ B
        step = Hb @ G
        B = B + step
        if np.max(np.abs(step)) < tol:
            break
    return B


def predict_softmax(X, B):
    n = X.shape[0]
    eta = np.column_stack([X @ B, np.zeros(n)])
    eta -= eta.max(axis=1, keepdims=True)
    E = np.exp(eta)
    return E / E.sum(axis=1, keepdims=True)


def pava(x, y, w):
    """Isotonic regression (pool-adjacent-violators) of y on sorted-by-x, weights w.
    Returns (knot_x, knot_y) step function, non-decreasing."""
    o = np.argsort(x, kind="mergesort")
    xs, ys, ws = x[o], y[o], w[o]
    # collapse ties in x
    vals, idx = np.unique(xs, return_inverse=True)
    sy = np.bincount(idx, weights=ys * ws)
    sw = np.bincount(idx, weights=ws)
    lvl = sy / sw
    # PAVA
    v, wt, sz = [], [], []
    for i in range(len(vals)):
        v.append(lvl[i]); wt.append(sw[i]); sz.append(1)
        while len(v) > 1 and v[-2] > v[-1]:
            nv = (v[-1] * wt[-1] + v[-2] * wt[-2]) / (wt[-1] + wt[-2])
            nw = wt[-1] + wt[-2]
            ns = sz[-1] + sz[-2]
            v.pop(); v.pop(); wt.pop(); wt.pop(); sz.pop(); sz.pop()
            v.append(nv); wt.append(nw); sz.append(ns)
    out = np.repeat(np.array(v), np.array(sz))
    return vals, out


def apply_iso(knot_x, knot_y, x):
    i = np.searchsorted(knot_x, x, side="left")
    i = np.clip(i, 0, len(knot_y) - 1)
    return knot_y[i]


# ------------------------------------------------------------------ probability plumbing
def probs4(z_sorted, sig_h, thr):
    """Four-bucket probabilities from an empirical z table, exactly as move_prob does."""
    a_dn, a_up = math.log(1 - thr), math.log(1 + thr)
    nz = len(z_sorted)
    F_dn = np.searchsorted(z_sorted, a_dn / sig_h, side="right") / nz
    F_0 = np.searchsorted(z_sorted, 0.0, side="right") / nz
    F_up = np.searchsorted(z_sorted, a_up / sig_h, side="right") / nz
    p = np.column_stack([F_dn, np.maximum(F_0 - F_dn, 0),
                         np.maximum(F_up - F_0, 0), np.maximum(1 - F_up, 0)])
    p = np.clip(p, 1e-4, None)
    return p / p.sum(axis=1, keepdims=True)


def p_move_from4(p):
    return p[:, 0] + p[:, 3]


def hit_of(lr, thr):
    return (lr <= math.log(1 - thr)) | (lr >= math.log(1 + thr))


def bidx4(lr, thr):
    """0 down_big 1 down_small 2 up_small 3 up_big, on SIMPLE-return thresholds in log space."""
    a_dn, a_up = math.log(1 - thr), math.log(1 + thr)
    return np.where(lr <= a_dn, 0, np.where(lr <= 0, 1, np.where(lr < a_up, 2, 3))).astype(np.int8)


def calibrate_kappa(z_in, sig_val, lr_val, thrs=(0.01, 0.02, 0.03, 0.05)):
    grid = np.linspace(0.90, 1.40, 26)
    best, best_err = 1.0, np.inf
    for k in grid:
        zk = z_in * k
        err = 0.0
        for thr in thrs:
            a_dn, a_up = math.log(1 - thr), math.log(1 + thr)
            f_dn = np.searchsorted(zk, a_dn / sig_val, side="right") / len(zk)
            f_up = np.searchsorted(zk, a_up / sig_val, side="right") / len(zk)
            err += abs((f_dn + (1 - f_up)).mean()
                       - float(((lr_val <= a_dn) | (lr_val >= a_up)).mean()))
        if err < best_err:
            best, best_err = float(k), err
    return best


def clim_sym_pmove(sid_tr, b_tr, sid_te, n_sym, thr_k=4, pseudo=40.0):
    """Per-symbol train base rate over 4 buckets, shrunk `pseudo` counts toward pooled.
    Returns P(|move|>=thr) for the test rows."""
    pool = np.bincount(b_tr.astype(np.int64), minlength=thr_k).astype(float)
    pool /= pool.sum()
    cs = np.bincount(sid_tr.astype(np.int64) * thr_k + b_tr.astype(np.int64),
                     minlength=n_sym * thr_k).reshape(n_sym, thr_k).astype(float)
    cs += pseudo * pool[None, :]
    cs /= cs.sum(axis=1, keepdims=True)
    cs = np.clip(cs, 1e-4, None)
    cs /= cs.sum(axis=1, keepdims=True)
    return (cs[:, 0] + cs[:, 3])[sid_te]


def bss2(p, hit, pref):
    num = float(((p - hit) ** 2).mean())
    den = float(((pref - hit) ** 2).mean())
    return 1.0 - num / den if den > 0 else float("nan")




# ------------------------------------------------------------------ fast kappa
def calibrate_kappa_fast(z_sorted, sig_val, lr_val, thrs=(0.01, 0.02, 0.03, 0.05)):
    """Same objective as move_prob._calibrate_kappa but scales the QUERY instead of the table
    (F_{z*k}(a/s) == F_z(a/(s*k))), so it never materialises 26 copies of a 1M-row z array."""
    grid = np.linspace(0.90, 1.40, 26)
    nz = len(z_sorted)
    qs, obs = [], []
    for thr in thrs:
        a_dn, a_up = math.log(1 - thr), math.log(1 + thr)
        qs.append((a_dn / sig_val, a_up / sig_val))
        obs.append(float(((lr_val <= a_dn) | (lr_val >= a_up)).mean()))
    best, best_err = 1.0, np.inf
    for k in grid:
        err = 0.0
        for (qd, qu), ob in zip(qs, obs):
            f_dn = np.searchsorted(z_sorted, qd / k, side="right").mean() / nz
            f_up = np.searchsorted(z_sorted, qu / k, side="right").mean() / nz
            err += abs(f_dn + (1 - f_up) - ob)
        if err < best_err:
            best, best_err = float(k), err
    return best


# ------------------------------------------------------------------ walk-forward engine
class Gram:
    """Per-year X'X / X'y accumulator so any expanding-window OLS is exact and instant.
    `tail` is the purge block: rows of the last train year whose forward window reaches into
    the test year."""

    def __init__(self, X, y, yr, dt, h):
        self.years = np.array(sorted(set(yr.tolist())))
        self.G, self.c, self.Gt, self.ct = {}, {}, {}, {}
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        for yy in self.years:
            m = (yr == yy) & ok
            Xi, yi = X[m], y[m]
            self.G[yy], self.c[yy] = Xi.T @ Xi, Xi.T @ yi
            last = dt[yr == yy].max()
            mt = m & (dt > last - h)
            Xt, yt = X[mt], y[mt]
            self.Gt[yy], self.ct[yy] = Xt.T @ Xt, Xt.T @ yt

    def fit(self, upto_year):
        ys = [y for y in self.years if y < upto_year]
        G = sum(self.G[y] for y in ys) - self.Gt[ys[-1]]
        c = sum(self.c[y] for y in ys) - self.ct[ys[-1]]
        return np.linalg.solve(G + 1e-9 * np.eye(len(c)), c)


class SymAcc:
    """Per-(year, symbol) sums of X, y and counts, with the same purge tail. Lets the per-symbol
    shrunk intercept be recomputed for any expanding window without touching the raw rows."""

    def __init__(self, X, y, yr, dt, sid, nsym, h):
        self.years = np.array(sorted(set(yr.tolist())))
        k = X.shape[1]
        self.SX, self.Sy, self.N = {}, {}, {}
        self.SXt, self.Syt, self.Nt = {}, {}, {}
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        for yy in self.years:
            for tail in (False, True):
                m = (yr == yy) & ok
                if tail:
                    last = dt[yr == yy].max()
                    m = m & (dt > last - h)
                s = sid[m].astype(np.int64)
                SX = np.column_stack([np.bincount(s, weights=X[m][:, j], minlength=nsym)
                                      for j in range(k)])
                Sy = np.bincount(s, weights=y[m], minlength=nsym)
                N = np.bincount(s, minlength=nsym).astype(float)
                if tail:
                    self.SXt[yy], self.Syt[yy], self.Nt[yy] = SX, Sy, N
                else:
                    self.SX[yy], self.Sy[yy], self.N[yy] = SX, Sy, N

    def window(self, upto_year):
        ys = [y for y in self.years if y < upto_year]
        SX = sum(self.SX[y] for y in ys) - self.SXt[ys[-1]]
        Sy = sum(self.Sy[y] for y in ys) - self.Syt[ys[-1]]
        N = sum(self.N[y] for y in ys) - self.Nt[ys[-1]]
        return SX, Sy, N

    def fe(self, upto_year, beta, lam):
        SX, Sy, N = self.window(upto_year)
        resid = np.where(N > 0, (Sy - SX @ beta) / np.maximum(N, 1), 0.0)
        return resid * (N / (N + lam))


def sq_design(X, ncol_base):
    """Base design + squares of every feature + short*long (rv_d * rv_q) interaction."""
    f = X[:, 1:]
    return np.column_stack([X, f ** 2, (f[:, 0] * f[:, 3])[:, None]])


def spline_design(s, knots):
    return np.column_stack([np.ones(len(s)), s] + [np.maximum(s - k, 0.0) for k in knots])


def platt_fit(p, y):
    x = np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1 - np.clip(p, 1e-5, 1 - 1e-5)))
    return irls_logit(np.column_stack([np.ones(len(x)), x]), y.astype(float), iters=30)


def platt_apply(b, p):
    x = np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1 - np.clip(p, 1e-5, 1 - 1e-5)))
    return predict_logit(np.column_stack([np.ones(len(x)), x]), b)


def iso_fit(p, y, nbin=50):
    """Binned isotonic: equal-count bins on p, PAVA on the bin hit rates, linear interpolation
    between bin centres. Binning keeps it stable and makes the map cheap to apply."""
    o = np.argsort(p, kind="mergesort")
    ps, ys = p[o], y[o].astype(float)
    edges = np.linspace(0, len(ps), nbin + 1).astype(int)
    cx = np.array([ps[a:b].mean() for a, b in zip(edges[:-1], edges[1:]) if b > a])
    cy = np.array([ys[a:b].mean() for a, b in zip(edges[:-1], edges[1:]) if b > a])
    cw = np.array([b - a for a, b in zip(edges[:-1], edges[1:]) if b > a], float)
    _, fit = pava(np.arange(len(cx), dtype=float), cy, cw)
    return cx, fit


def iso_apply(m, p):
    cx, cy = m
    return np.interp(p, cx, cy)


SIGVARS = ("BASE", "SQ", "FE250", "FE1000", "SPL")
ALLVARS = SIGVARS + ("LOGIT", "MNL", "PLATT", "ISO")
NCAL = 5            # calendar years of out-of-sample p_base used to fit a recalibration map
KFOLDS = 8          # inner folds whose kappa is medianed, exactly as move_prob.fit


def train_mask(yr, dt, y, h):
    prev = yr < y
    if not prev.any():
        return prev
    return prev & (dt <= dt[prev].max() - h)


def run_group(g, h, cache, variants=ALLVARS, verbose=True):
    """One expanding-window pass over calendar years. For year y every variant is fit on years
    < y (purged by h days), the width kappa is the median of the inner-fold kappas of years < y,
    and any recalibration map is fit on the OOS predictions of the previous NCAL years."""
    d = cache[(g, h)]
    X, lr, tgt, yr, sid, dt = d["X"], d["lr"], d["tgt"], d["yr"], d["sid"], d["dt"]
    nsym, rt = len(d["symbols"]), math.sqrt(h)
    years = np.array(sorted(set(yr.tolist())))
    first_test = years[MIN_TRAIN_YEARS]
    Xsq = sq_design(X, X.shape[1]) if "SQ" in variants else None
    grams = {"BASE": Gram(X, tgt, yr, dt, h)}
    if "SQ" in variants:
        grams["SQ"] = Gram(Xsq, tgt, yr, dt, h)
    sacc = SymAcc(X, tgt, yr, dt, sid, nsym, h) if any(v.startswith("FE") for v in variants) else None
    b4 = {t: bidx4(lr, t) for t in THRS}
    hit = {t: hit_of(lr, t) for t in THRS}

    kap = {v: {} for v in SIGVARS}
    hist = {t: [] for t in THRS}            # OOS (year, p_base, hit) for recalibration maps
    store = {(v, t): [] for v in variants for t in THRS}
    meta = {k: [] for k in ("dt", "sid", "yr", "sig", "ref2", "ref5", "hit2", "hit5")}
    warm = {}
    for y in years:
        tr = train_mask(yr, dt, y, h)
        te = yr == y
        if tr.sum() < 5000 or te.sum() == 0:
            continue
        Xtr, Xte = X[tr], X[te]
        beta = grams["BASE"].fit(y)
        shat_tr, shat_te = Xtr @ beta, Xte @ beta
        sig = {"BASE": (np.exp(shat_tr) * rt, np.exp(shat_te) * rt)}
        if "SQ" in variants:
            bq = grams["SQ"].fit(y)
            sig["SQ"] = (np.exp(Xsq[tr] @ bq) * rt, np.exp(Xsq[te] @ bq) * rt)
        for lam, nm in ((250.0, "FE250"), (1000.0, "FE1000")):
            if nm in variants:
                a = sacc.fe(y, beta, lam)
                sig[nm] = (np.exp(shat_tr + a[sid[tr]]) * rt, np.exp(shat_te + a[sid[te]]) * rt)
        if "SPL" in variants:
            kn = np.quantile(shat_tr, [0.2, 0.4, 0.6, 0.8])
            bs = ols(spline_design(shat_tr, kn), tgt[tr])
            sig["SPL"] = (np.exp(spline_design(shat_tr, kn) @ bs) * rt,
                          np.exp(spline_design(shat_te, kn) @ bs) * rt)

        is_test = y >= first_test
        p_out = {}
        for v, (s_tr, s_te) in sig.items():
            z = np.sort(lr[tr] / s_tr)
            prior = [kap[v][yy] for yy in years if yy < y and yy in kap[v]][-KFOLDS:]
            k = float(np.median(prior)) if prior else 1.0
            kap[v][y] = calibrate_kappa_fast(z, s_te, lr[te])   # for LATER years only
            zk = z * k
            for t in THRS:
                p_out[(v, t)] = p_move_from4(probs4(zk, s_te, t))
        if "LOGIT" in variants:
            for t in THRS:
                key = ("LOGIT", t)
                b = irls_logit(Xtr, hit[t][tr].astype(float), iters=8)
                warm[key] = b
                p_out[key] = predict_logit(Xte, b)
        if "MNL" in variants:
            t = 0.02
            Y4 = np.zeros((int(tr.sum()), 4))
            Y4[np.arange(int(tr.sum())), b4[t][tr]] = 1.0
            B = softmax_newton(Xtr, Y4, iters=60)
            P = predict_softmax(Xte, B)
            p_out[("MNL", t)] = P[:, 0] + P[:, 3]
            p_out[("MNL", 0.05)] = np.full(int(te.sum()), np.nan)
        for t in THRS:
            ph = [x for x in hist[t] if x[0] >= y - NCAL]
            if ph and ("PLATT" in variants or "ISO" in variants):
                pc = np.concatenate([x[1] for x in ph]).astype(float)
                yc = np.concatenate([x[2] for x in ph])
                if "PLATT" in variants:
                    p_out[("PLATT", t)] = platt_apply(platt_fit(pc, yc), p_out[("BASE", t)])
                if "ISO" in variants:
                    p_out[("ISO", t)] = iso_apply(iso_fit(pc, yc), p_out[("BASE", t)])
            else:
                for v in ("PLATT", "ISO"):
                    if v in variants:
                        p_out[(v, t)] = p_out[("BASE", t)].copy()
            hist[t].append((y, p_out[("BASE", t)].astype(np.float32), hit[t][te]))

        if not is_test:
            continue
        meta["dt"].append(dt[te]); meta["sid"].append(sid[te]); meta["yr"].append(yr[te])
        meta["sig"].append(sig["BASE"][1].astype(np.float32))
        for t in THRS:
            meta[f"hit{int(t*100)}"].append(hit[t][te])
            meta[f"ref{int(t*100)}"].append(
                clim_sym_pmove(sid[tr], b4[t][tr], sid[te], nsym).astype(np.float32))
            for v in variants:
                store[(v, t)].append(np.clip(p_out[(v, t)], 2e-4, 1 - 2e-4).astype(np.float32))

    R = {k: np.concatenate(v) for k, v in meta.items() if v}
    for t in THRS:
        R[("hit", t)] = R[f"hit{int(t*100)}"]
        for v in variants:
            R[(v, t)] = np.concatenate(store[(v, t)])
    R["kappa"] = {v: kap[v] for v in SIGVARS if kap[v]}
    R["n_sym"] = nsym
    if verbose:
        for t in THRS:
            line = f"  {g:<7} h={h:<3} thr={t:.0%} n={len(R['dt']):>9,}  "
            for v in variants:
                b = bss2(R[(v, t)].astype(float), R[("hit", t)], R[f"ref{int(t*100)}"].astype(float))
                line += f"{v}={b:+.4f} "
            log(line)
    return R


# ------------------------------------------------------------------ scoring / reporting
def year_bss2(R, v, t):
    ref = R[f"ref{int(t*100)}"].astype(float)
    hit = R[("hit", t)]
    p = R[(v, t)].astype(float)
    out = {}
    for y in np.unique(R["yr"]):
        m = R["yr"] == y
        out[int(y)] = bss2(p[m], hit[m], ref[m])
    return out


def summarise(R, g, h, variants):
    rows = []
    for t in THRS:
        ref = R[f"ref{int(t*100)}"].astype(float)
        hit = R[("hit", t)]
        yb = {v: year_bss2(R, v, t) for v in variants}
        for v in variants:
            p = R[(v, t)].astype(float)
            if not np.isfinite(p).all():
                rows.append(dict(grp=g, h=h, thr=t, var=v, n=len(p), bss2=np.nan, ece=np.nan,
                                 d_bss2=np.nan, d_ece=np.nan, wins=np.nan, nyr=np.nan))
                continue
            b, e = bss2(p, hit, ref), L.ece(p, hit)
            w = sum(1 for y in yb[v] if yb[v][y] > yb["BASE"][y] + 1e-12)
            rows.append(dict(grp=g, h=h, thr=t, var=v, n=len(p), bss2=b, ece=e,
                             d_bss2=b - bss2(R[("BASE", t)].astype(float), hit, ref),
                             d_ece=e - L.ece(R[("BASE", t)].astype(float), hit),
                             wins=w, nyr=len(yb[v])))
    return rows


def stage_main():
    cache = load_cache()
    allrows = []
    for g in ("index", "single"):
        for h in HORIZONS:
            R = run_group(g, h, cache)
            allrows += summarise(R, g, h, ALLVARS)
            pd.to_pickle({k: v for k, v in R.items()
                          if k in ("dt", "sid", "yr", "sig", "ref2", "ref5", "hit2", "hit5")
                          or (isinstance(k, tuple) and k[0] in ALLVARS)},
                         Path(__file__).with_name(f"_move_fx_form_R_{g}{h}.pkl"))
            del R
    df = pd.DataFrame(allrows)
    df.to_csv(Path(__file__).with_name("_move_fx_form_main.csv"), index=False)
    print("\n" + "=" * 110)
    print("MAIN TABLE -- delta BSS2 / delta ECE vs the shipped baseline (walk-forward 2006-2026)")
    print("=" * 110)
    for t in THRS:
        print(f"\n--- thr={t:.0%} ---")
        print(f"{'grp':<7}{'h':>3}{'n':>10}{'BASE':>9}" +
              "".join(f"{v:>10}" for v in ALLVARS if v != "BASE"))
        for g in ("index", "single"):
            for h in HORIZONS:
                s = df[(df.grp == g) & (df.h == h) & (df.thr == t)].set_index("var")
                line = f"{g:<7}{h:>3}{int(s.loc['BASE','n']):>10,}{s.loc['BASE','bss2']:>9.4f}"
                for v in ALLVARS:
                    if v == "BASE":
                        continue
                    line += f"{s.loc[v,'d_bss2']:>+10.4f}"
                print(line)
        print(f"\n{'grp':<7}{'h':>3}{'ECE_BASE':>10}" +
              "".join(f"{'d'+v:>10}" for v in ALLVARS if v != "BASE"))
        for g in ("index", "single"):
            for h in HORIZONS:
                s = df[(df.grp == g) & (df.h == h) & (df.thr == t)].set_index("var")
                line = f"{g:<7}{h:>3}{s.loc['BASE','ece']:>10.4f}"
                for v in ALLVARS:
                    if v == "BASE":
                        continue
                    line += f"{s.loc[v,'d_ece']:>+10.4f}"
                print(line)
        print(f"\nyears improved out of {int(df.nyr.max())} (thr={t:.0%})")
        print(f"{'grp':<7}{'h':>3}" + "".join(f"{v:>10}" for v in ALLVARS if v != "BASE"))
        for g in ("index", "single"):
            for h in HORIZONS:
                s = df[(df.grp == g) & (df.h == h) & (df.thr == t)].set_index("var")
                print(f"{g:<7}{h:>3}" + "".join(f"{s.loc[v,'wins']:>10.0f}"
                                                for v in ALLVARS if v != "BASE"))


