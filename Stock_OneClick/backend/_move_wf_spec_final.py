#!/usr/bin/env python
"""
_move_wf_spec_final.py — end-to-end walk-forward validation of the FINAL move_prob.py stack.

Purpose: the six upstream studies each validated ONE layer against a DIFFERENT provisional
scaler. This script runs the composed pipeline (HAR log-vol forecast -> z table -> 4 buckets)
strictly walk-forward and produces the numbers the shipping spec promises, plus the shipped
constants (coefficients, loc/scale, z-CDF grid).

Stages (argv[1]): build | race | thr | ship | earn | boot | all
"""
from __future__ import annotations
import sys, math, time, json, pickle
import numpy as np
import pandas as pd
import _move_data as D
import _move_lib as L

FLOOR, CEIL = 1e-3, 0.5          # sigma clip band before logs (vol-forecast study's band)
HS = (1, 5, 10, 21)
MIN_TRAIN_YEARS = 5
MIN_BARS = 300                   # EWMA/HAR warm-up + rv_q(63) + room
CACHE = "/tmp/_move_spec_final.pkl"
EARN_JSON = "/Users/feijing/github.com/stock_scan/gold_pine_script/pead_earnings_cache.json"


# ---------------------------------------------------------------- small math utils (no scipy)
def digamma(x: float) -> float:
    r = 0.0
    while x < 6.0:
        r -= 1.0 / x
        x += 1.0
    f = 1.0 / (x * x)
    return (r + math.log(x) - 0.5 / x
            - f * (1.0 / 12 - f * (1.0 / 120 - f * (1.0 / 252 - f * (1.0 / 240)))))


def chi_log_corr(h: int) -> float:
    """Add this to E[log(sample sd of nu+1 obs)] to get log(sigma). nu = h-1 (nu=1 for h=1)."""
    nu = max(h - 1, 1)
    return -0.5 * (digamma(nu / 2.0) - math.log(nu / 2.0))


_A = (0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429)


def ncdf(x):
    """Vectorised standard normal CDF, A&S 26.2.17, |err| < 7.5e-8."""
    x = np.asarray(x, dtype=float)
    s = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.2316419 * ax)
    poly = t * (_A[0] + t * (_A[1] + t * (_A[2] + t * (_A[3] + t * _A[4]))))
    tail = np.exp(-0.5 * ax * ax) / math.sqrt(2 * math.pi) * poly
    return np.where(s >= 0, 1.0 - tail, tail)


def clog(x):
    return np.log(np.clip(x, FLOOR, CEIL))


def ols(X, y):
    XtX = X.T @ X
    return np.linalg.solve(XtX + 1e-9 * np.eye(X.shape[1]) * np.trace(XtX) / X.shape[1], X.T @ y)


# ---------------------------------------------------------------- panel build
FEATS = ("rv_d", "rv_w", "rv_m", "rv_q", "yz21", "e97", "e94", "park21",
         "ac_d", "ac_w", "ac_m", "ac_q", "siv", "cc252")


def build():
    t0 = time.time()
    p = D.load()
    O, Hi, Lo, C = (p[k] for k in ("Open", "High", "Low", "Close"))
    cov = C.notna().sum()
    keep = [s for s in C.columns if cov[s] >= 500]
    dropped_neg = [s for s in keep if (C[s].dropna() <= 0).any()]
    keep = [s for s in keep if s not in dropped_neg]
    IDX = [s for s in ("SPY", "QQQ", "IWM", "DIA", "^GSPC") if s in keep]
    SNG = [s for s in keep if s not in IDX and s != "^VIX"]
    syms = IDX + SNG
    vix = C["^VIX"]
    O, Hi, Lo, C = (df[syms] for df in (O, Hi, Lo, C))

    lr = L._logret(C)
    rv1 = L.vol_parkinson(Hi, Lo, 1)
    ar = lr.abs()
    f = {
        "rv_d": rv1, "rv_w": rv1.rolling(5).mean(), "rv_m": rv1.rolling(22).mean(),
        "rv_q": rv1.rolling(63).mean(),
        "yz21": L.vol_yang_zhang(O, Hi, Lo, C, 21),
        "e97": L.vol_ewma(C, 0.97), "e94": L.vol_ewma(C, 0.94),
        "park21": L.vol_parkinson(Hi, Lo, 21),
        "ac_d": ar, "ac_w": ar.rolling(5).mean(), "ac_m": ar.rolling(22).mean(),
        "ac_q": ar.rolling(63).mean(),
        "cc252": L.vol_cc(C, 252),
    }
    siv = vix / 100.0 / math.sqrt(252.0)
    f["siv"] = pd.DataFrame(np.repeat(siv.values[:, None], len(syms), axis=1),
                            index=C.index, columns=syms)

    # usability mask: features present, enough history, not a stale/frozen quote
    nbar = C.notna().cumsum()
    nz30 = (lr.abs() > 1e-12).rolling(30).sum()
    ok = (nbar >= MIN_BARS) & C.notna()
    for k in ("rv_q", "yz21", "e97", "e94", "park21", "ac_q", "cc252", "siv"):
        ok &= f[k].notna()
    stale = (nz30 < 10) | (f["e94"] < 0.002)
    usable = ok & ~stale

    tg = {}
    for h in HS:
        tg[f"fwd{h}"] = C.shift(-h) / C - 1.0
        tg[f"flog{h}"] = np.log(C.shift(-h) / C)
        tv = ar.shift(-1) if h == 1 else L.realized_vol_forward(C, h)
        tg[f"tv{h}"] = tv

    out = {"IDX": IDX, "SNG": SNG, "index": C.index, "dropped_neg": dropped_neg,
           "n_ok": int(ok.values.sum()), "n_stale": int((ok & stale).values.sum())}
    for grp, gs in (("INDEX", IDX), ("SINGLE", SNG)):
        m = usable[gs].values
        ii, jj = np.nonzero(m)
        g = {"date_i": ii.astype(np.int32), "sym_i": jj.astype(np.int32),
             "year": C.index.year.values[ii].astype(np.int16), "syms": gs}
        for k in FEATS:
            g[k] = f[k][gs].values[ii, jj]
        for k, df in tg.items():
            g[k] = df[gs].values[ii, jj]
        out[grp] = g
    out["ymap"] = {y: i for i, y in enumerate(sorted(set(C.index.year)))}
    print(f"[build] {time.time()-t0:.1f}s  symbols kept {len(syms)} "
          f"(INDEX {len(IDX)}, SINGLE {len(SNG)}); dropped non-positive close: {dropped_neg}")
    print(f"[build] usable rows: INDEX {len(out['INDEX']['date_i']):,}  "
          f"SINGLE {len(out['SINGLE']['date_i']):,}   stale/frozen rows removed "
          f"{out['n_stale']:,} of {out['n_ok']:,} ({100*out['n_stale']/out['n_ok']:.2f}%)")
    with open(CACHE, "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    return out


def load_built():
    try:
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return build()


# ---------------------------------------------------------------- scale models
SPECS = {
    "har6":   ("rv_d", "rv_w", "rv_m", "rv_q", "yz21", "e97"),
    "har4":   ("rv_d", "rv_w", "rv_m", "yz21"),
    "har6v":  ("rv_d", "rv_w", "rv_m", "rv_q", "yz21", "e97", "siv"),
    "har4v":  ("rv_d", "rv_w", "rv_m", "yz21", "siv"),
    "harc4":  ("ac_d", "ac_w", "ac_m", "ac_q"),
    "e94":    ("e94",),
    "e97s":   ("e97",),
    "park21": ("park21",),
}


def design(g, cols, rows):
    X = np.empty((rows.sum(), len(cols) + 1))
    X[:, 0] = 1.0
    for k, c in enumerate(cols):
        X[:, k + 1] = clog(g[c][rows])
    return X


def fit_scale(g, spec, h, tr, te):
    """OLS in log space on the h-day forward realized-vol target. Returns sigma_hat (per-DAY)
    on train rows and test rows, plus coefficients."""
    cols = SPECS[spec]
    y = clog(g[f"tv{h}"]) + chi_log_corr(h)
    b = ols(design(g, cols, tr), y[tr])
    return b


def predict_scale(g, spec, b, rows, h):
    return np.exp(design(g, SPECS[spec], rows) @ b) * math.sqrt(h)


# ---------------------------------------------------------------- shape models
class EmpShape:
    """Empirical CDF of standardized u, evaluated by searchsorted."""
    def __init__(self, u):
        self.u = np.sort(u)
        self.n = len(u)

    def __call__(self, x):
        return np.searchsorted(self.u, x, side="right") / self.n


class GridShape:
    """Shippable table: G on a fixed u grid, linear interp, exponential tails."""
    def __init__(self, u, grid):
        s = np.sort(u)
        n = len(s)
        self.grid = np.asarray(grid, float)
        self.G = np.searchsorted(s, self.grid, side="right") / n
        self.G = np.maximum.accumulate(np.clip(self.G, 0.5 / n, 1 - 0.5 / n))
        gl = self.G[:2]
        self.bl = (grid[1] - grid[0]) / max(math.log(gl[1] / gl[0]), 1e-9)
        gr = 1 - self.G[-2:]
        self.br = (grid[-1] - grid[-2]) / max(math.log(gr[0] / gr[1]), 1e-9)

    def __call__(self, x):
        x = np.asarray(x, float)
        out = np.interp(x, self.grid, self.G)
        lo = x < self.grid[0]
        if lo.any():
            out = np.where(lo, self.G[0] * np.exp((x - self.grid[0]) / self.bl), out)
        hi = x > self.grid[-1]
        if hi.any():
            out = np.where(hi, 1 - (1 - self.G[-1]) * np.exp(-(x - self.grid[-1]) / self.br), out)
        return np.clip(out, 1e-9, 1 - 1e-9)


NORM = ncdf
UGRID = np.array([-6, -5, -4.5, -4, -3.5, -3, -2.75, -2.5, -2.25, -2, -1.75, -1.5, -1.25, -1.0,
                  -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2, 2.25, 2.5,
                  2.75, 3, 3.5, 4, 4.5, 5, 6], float)


def probs(sig, loc, scale, thr, F):
    a_dn = np.log(1 - thr) / sig
    a_up = np.log(1 + thr) / sig
    Fd = F((a_dn - loc) / scale)
    F0 = F((0.0 - loc) / scale)
    Fu = F((a_up - loc) / scale)
    p = np.column_stack([Fd, F0 - Fd, Fu - F0, 1.0 - Fu])
    p = np.clip(p, 1e-4, None)
    return p / p.sum(1, keepdims=True)


# ---------------------------------------------------------------- scoring helpers
def bidx(fwd, thr):
    return np.where(fwd <= -thr, 0, np.where(fwd <= 0, 1, np.where(fwd < thr, 2, 3))).astype(int)


def brier4(p, a):
    oh = np.zeros_like(p)
    oh[np.arange(len(a)), a] = 1.0
    return float(((p - oh) ** 2).sum(1).mean())


def brier2(p, a):
    pm = p[:, 0] + p[:, 3]
    ym = ((a == 0) | (a == 3)).astype(float)
    return float(((pm - ym) ** 2).mean())


def ll(p, a):
    return float(-np.log(np.clip(p[np.arange(len(a)), a], 1e-12, 1)).mean())


def ece_bucket(p, a, k, nb=10):
    return L.ece(p[:, k], (a == k).astype(float), nb)


def ece_move(p, a, nb=10):
    return L.ece(p[:, 0] + p[:, 3], ((a == 0) | (a == 3)).astype(float), nb)


def clim_sym(a_tr, sym_tr, a_te_syms, nsym, pseudo=40.0):
    pool = np.bincount(a_tr, minlength=4).astype(float)
    pool /= pool.sum()
    cnt = np.zeros((nsym, 4))
    np.add.at(cnt, (sym_tr, a_tr), 1.0)
    ps = (cnt + pseudo * pool) / (cnt.sum(1, keepdims=True) + pseudo)
    return ps[a_te_syms], pool


# ---------------------------------------------------------------- the walk-forward race
def race(B, thr=0.02, verbose=True, want=None, hs=HS):
    """Returns dict[(grp,h,model)] -> metrics; also per-year skill records."""
    res, peryear, store = {}, [], {}
    years = sorted(B["ymap"])
    test_years = years[MIN_TRAIN_YEARS:]
    for grp in ("INDEX", "SINGLE"):
        g = B[grp]
        nsym = len(g["syms"])
        for h in hs:
            fwd = g[f"fwd{h}"]
            tv = g[f"tv{h}"]
            flog = g[f"flog{h}"]
            okS = np.isfinite(fwd) & np.isfinite(flog)      # scoring rows
            okV = np.isfinite(tv)                           # vol-fit rows
            acc = {}
            for y in test_years:
                te = okS & (g["year"] == y)
                if te.sum() < 50:
                    continue
                i0 = np.min(g["date_i"][te])
                prior = (g["year"] < y)
                trV = okV & prior & (g["date_i"] + h < i0)   # purged
                trS = okS & prior & (g["date_i"] + h < i0)
                if trV.sum() < 500 or trS.sum() < 500:
                    continue
                a_te = bidx(fwd[te], thr)
                a_tr = bidx(fwd[trS], thr)
                # ---- baselines
                pool = np.bincount(a_tr, minlength=4).astype(float)
                pool /= pool.sum()
                mods = {"clim_pool": np.tile(pool, (te.sum(), 1))}
                psym, _ = clim_sym(a_tr, g["sym_i"][trS], g["sym_i"][te], nsym)
                mods["clim_sym"] = psym
                # ---- scale models
                sig = {}
                for spec in (want or ["har6", "har6v", "har4", "har4v", "e94", "harc4", "park21", "e97s"]):
                    if spec.endswith("v") and grp != "INDEX":
                        continue
                    b = fit_scale(g, spec, h, trV, te)
                    sig[spec] = (predict_scale(g, spec, b, trS, h),
                                 predict_scale(g, spec, b, te, h), b)
                # raw ewma*sqrt(h), no fit at all (the provisional scaler)
                sig["e94raw"] = (g["e94"][trS] * math.sqrt(h), g["e94"][te] * math.sqrt(h), None)
                # ---- shapes on top of each scale
                for spec, (s_tr, s_te, _b) in sig.items():
                    z_tr = flog[trS] / s_tr
                    loc = float(np.median(z_tr))
                    q1, q3 = np.percentile(z_tr, [25, 75])
                    scale = float((q3 - q1) / 1.349)
                    u_tr = (z_tr - loc) / scale
                    Femp = EmpShape(u_tr)
                    combos = {f"{spec}|emp": (loc, scale, Femp)}
                    if spec in ("har6", "har6v", "e94raw"):
                        combos[f"{spec}|grid"] = (loc, scale, GridShape(u_tr, UGRID))
                        combos[f"{spec}|norm"] = (loc, scale, NORM)
                        combos[f"{spec}|noloc"] = (0.0, scale, Femp)
                    for nm, (lo, sc, F) in combos.items():
                        mods[nm] = probs(s_te, lo, sc, thr, F)
                    if spec in ("har6", "har6v"):
                        store.setdefault((grp, h), {})[y] = (s_te, z_tr, loc, scale)
                # ---- accumulate
                for nm, p in mods.items():
                    acc.setdefault(nm, []).append(p)
                acc.setdefault("_a", []).append(a_te)
                acc.setdefault("_date", []).append(g["date_i"][te])
                acc.setdefault("_year", []).append(np.full(te.sum(), y))
                acc.setdefault("_sig", []).append(sig.get("har6v", sig["har6"])[1])
            if not acc:
                continue
            a = np.concatenate(acc.pop("_a"))
            dt = np.concatenate(acc.pop("_date"))
            yr = np.concatenate(acc.pop("_year"))
            sg = np.concatenate(acc.pop("_sig"))
            P = {nm: np.vstack(v) for nm, v in acc.items()}
            base4 = brier4(P["clim_sym"], a)
            base2 = brier2(P["clim_sym"], a)
            basell = ll(P["clim_sym"], a)
            b4p, b2p, llp = brier4(P["clim_pool"], a), brier2(P["clim_pool"], a), ll(P["clim_pool"], a)
            for nm, p in P.items():
                res[(grp, h, nm)] = dict(
                    n=len(a), ndate=len(np.unique(dt)), ll=ll(p, a), brier4=brier4(p, a),
                    brier2=brier2(p, a),
                    bss4=1 - brier4(p, a) / base4, bss2=1 - brier2(p, a) / base2,
                    llskill=1 - ll(p, a) / basell,
                    bss4_pool=1 - brier4(p, a) / b4p, bss2_pool=1 - brier2(p, a) / b2p,
                    llskill_pool=1 - ll(p, a) / llp,
                    ece_up=ece_bucket(p, a, 3), ece_dn=ece_bucket(p, a, 0),
                    ece_mv=ece_move(p, a),
                    p_up=float(p[:, 3].mean()), p_dn=float(p[:, 0].mean()),
                    p_mv=float((p[:, 0] + p[:, 3]).mean()),
                    o_up=float((a == 3).mean()), o_dn=float((a == 0).mean()),
                    o_mv=float(((a == 0) | (a == 3)).mean()))
                for y in np.unique(yr):
                    m = yr == y
                    if m.sum() < 50:
                        continue
                    peryear.append(dict(grp=grp, h=h, model=nm, year=int(y), n=int(m.sum()),
                                        bss4=1 - brier4(p[m], a[m]) / max(brier4(P["clim_sym"][m], a[m]), 1e-12),
                                        llskill=1 - ll(p[m], a[m]) / max(ll(P["clim_sym"][m], a[m]), 1e-12)))
            store.setdefault((grp, h), {})["_score"] = (a, dt, yr, sg, P)
            if verbose:
                print(f"  [{grp} h={h}] n={len(a):,} dates={len(np.unique(dt)):,} done "
                      f"({time.time()-T0:.0f}s)")
    return res, pd.DataFrame(peryear), store


def show(res, order, thr):
    print(f"\n=== WALK-FORWARD OOS, thr={thr:.0%}, vs PER-SYMBOL climatology (CLIMSYM) ===")
    print(f"{'grp':7s} {'h':>3s} {'model':16s} {'n':>10s} {'LL':>7s} {'BSS4':>8s} {'BSS2':>8s} "
          f"{'LLskl':>7s} {'ECEmv':>7s} {'ECEup':>7s} {'ECEdn':>7s} {'pmv':>6s} {'omv':>6s}")
    for grp in ("INDEX", "SINGLE"):
        for h in HS:
            for nm in order:
                r = res.get((grp, h, nm))
                if not r:
                    continue
                print(f"{grp:7s} {h:3d} {nm:16s} {r['n']:10,d} {r['ll']:7.4f} {r['bss4']:+8.4f} "
                      f"{r['bss2']:+8.4f} {r['llskill']:+7.4f} {r['ece_mv']:7.4f} "
                      f"{r['ece_up']:7.4f} {r['ece_dn']:7.4f} {r['p_mv']:6.3f} {r['o_mv']:6.3f}")


T0 = time.time()

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "build":
        build()
        sys.exit()
    B = load_built()
    if stage in ("race", "all"):
        res, py, store = race(B, 0.02)
        order = ["clim_pool", "har6v|emp", "har6|emp", "har6v|grid", "har6|grid", "har6v|norm",
                 "har6|norm", "har6v|noloc", "har6|noloc", "har4v|emp", "har4|emp",
                 "e94|emp", "e94raw|emp", "e94raw|norm", "harc4|emp", "park21|emp", "e97s|emp"]
        show(res, order, 0.02)
        with open("/tmp/_move_spec_race.pkl", "wb") as fh:
            pickle.dump({"res": res, "py": py}, fh)
        print("\n=== PER-YEAR: count of 21 test years with BSS4 > 0 vs CLIMSYM (thr=2%) ===")
        for grp in ("INDEX", "SINGLE"):
            for h in HS:
                s = py[(py.grp == grp) & (py.h == h)]
                if s.empty:
                    continue
                line = f"{grp:7s} h={h:3d}  "
                for nm in ("har6v|emp", "har6|emp", "e94raw|emp", "har6|norm", "clim_pool"):
                    q = s[s.model == nm]
                    if q.empty:
                        continue
                    line += f"{nm}: {int((q.bss4 > 0).sum())}/{len(q)}   "
                print(line)
        print("\n=== RECENT WINDOW (test years >= 2021), thr=2% ===")
        for grp in ("INDEX", "SINGLE"):
            for h in HS:
                s = py[(py.grp == grp) & (py.h == h) & (py.year >= 2021)]
                if s.empty:
                    continue
                line = f"{grp:7s} h={h:3d}  "
                for nm in ("har6v|emp", "har6|emp", "e94raw|emp"):
                    q = s[s.model == nm]
                    if q.empty:
                        continue
                    w = q.n / q.n.sum()
                    line += f"{nm}: BSS4 {float((q.bss4*w).sum()):+.4f} ({int((q.bss4>0).sum())}/{len(q)}y)  "
                print(line)
