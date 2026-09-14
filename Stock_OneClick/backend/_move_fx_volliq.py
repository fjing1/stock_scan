"""
_move_fx_volliq.py — does VOLUME / LIQUIDITY add anything to move_prob.py?

Factor family under test (first study that can use the panel's Volume field):
  1. volume level    : log mean dollar volume over 5 / 22 / 63 days
  2. volume surprise : log(V_t / mean22(V)), log(mean5 / mean22)
  3. volume trend    : log(mean22 / mean63)            ("drying up")
  4. Amihud illiq    : log mean_22(|logret| / dollar_volume)
  5. liquidity subgroup calibration of the SHIPPED baseline (obs/pred by dollar-volume decile)

Method is byte-for-byte the shipped validation protocol (_move_validate.py): expanding
walk-forward by calendar year, >=5 training years, every parameter (HAR betas, empirical z
table, kappa width) refit on strictly prior years, scored against each ticker's OWN train-year
base rate shrunk 40 pseudo-counts toward pooled.

Speed trick: per-year Gram matrices (X'X, X'y) are accumulated ONCE over the full column block,
so every model variant / every expanding window is a KxK solve instead of a re-scan of 1M rows.

Run:  ../../vcp_env/bin/python _move_fx_volliq.py <stage>
      stages: cache | repro | feat | liq | trend | boot | all
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _move_data as D
import _move_lib as L
import move_prob as M

CACHE = Path(__file__).with_name("_move_fx_volliq_cache.pkl")
MIN_TRAIN_YEARS = 5
INNER_FOLDS = 5
SHRINK = 40.0
THRS = (0.02, 0.05)

# ---------------------------------------------------------------- volume features
VOL_FEATS = ["ldv5", "ldv22", "ldv63", "vs_d", "vs_w", "vt_q", "amih", "vvol", "zr22", "clip22"]


def volume_features(close: pd.Series, vol: pd.Series,
                    high: pd.Series | None = None, low: pd.Series | None = None) -> pd.DataFrame:
    """All computable from bars up to and including t. Ratios are clipped before log for the
    same reason move_prob clips Parkinson: 0.9% of volume cells are exactly 0 (halts) and an
    unclipped log(0) is a -inf that one row would otherwise dominate the regression with."""
    v = vol.reindex(close.index).astype(float)
    dv = (close * v).clip(lower=1e3)            # dollar volume, floored at $1k
    r = np.log(close).diff().abs()

    m5, m22, m63 = v.rolling(5).mean(), v.rolling(22).mean(), v.rolling(63).mean()
    f = pd.DataFrame(index=close.index)
    f["ldv5"] = np.log(dv.rolling(5).mean().clip(lower=1e3))
    f["ldv22"] = np.log(dv.rolling(22).mean().clip(lower=1e3))
    f["ldv63"] = np.log(dv.rolling(63).mean().clip(lower=1e3))
    f["vs_d"] = np.log((v / m22).clip(0.05, 20.0))
    f["vs_w"] = np.log((m5 / m22).clip(0.05, 20.0))
    f["vt_q"] = np.log((m22 / m63).clip(0.05, 20.0))
    # Amihud: |return| per $1M traded, 22-day mean, in logs
    f["amih"] = np.log((r / (dv / 1e6)).rolling(22).mean().clip(1e-9, 1e4))
    f["vvol"] = np.log(np.log(v.clip(lower=1.0)).rolling(22).std(ddof=1).clip(1e-3, 5.0))
    # Illiquidity-artefact flags: share of the last 22 bars with high == low (Parkinson exactly 0)
    # and share whose 1-bar Parkinson hits move_prob's lower clip. Both tell the model that this
    # name's rv_d input is mechanically biased DOWN, which is exactly where thin names live.
    if high is not None and low is not None:
        zr = (high.reindex(close.index) == low.reindex(close.index)).astype(float)
        pk = L.vol_parkinson(high.reindex(close.index), low.reindex(close.index), 1)
        f["zr22"] = zr.rolling(22).mean()
        f["clip22"] = (pk <= M.VOL_CLIP[0]).astype(float).rolling(22).mean()
    else:
        f["zr22"] = 0.0
        f["clip22"] = 0.0
    return f


# ---------------------------------------------------------------- cache
def build_cache(panel):
    C, H, Lo, V = panel["Close"], panel["High"], panel["Low"], panel["Volume"]
    vix = C["^VIX"] if "^VIX" in C.columns else None
    master = {d: i for i, d in enumerate(C.index)}
    usable = [s for s in C.columns if s != "^VIX" and C[s].notna().sum() >= 500]
    groups = {"index": [s for s in usable if s in M.INDEX_LIKE],
              "single": [s for s in usable if s not in M.INDEX_LIKE]}
    cache = {}
    for g, syms in groups.items():
        base = M.FEATS_OHLC + (["logvix"] if (g == "index" and vix is not None) else [])
        cols = base + VOL_FEATS
        per_h = {h: dict(X=[], lr=[], tgt=[], yr=[], sid=[], dt=[]) for h in M.HORIZONS}
        for i, s in enumerate(syms):
            c = C[s].dropna()
            if len(c) < 400:
                continue
            f = M.build_features(c, H[s].reindex(c.index), Lo[s].reindex(c.index), vix)
            fv = volume_features(c, V[s], H[s], Lo[s])
            ff = pd.concat([f, fv], axis=1)
            Xi = np.column_stack([np.ones(len(ff))] + [ff[k].values for k in cols])
            dti = np.array([master[d] for d in c.index], dtype=np.int32)
            yri = c.index.year.values
            for h in M.HORIZONS:
                li = np.log(c.shift(-h) / c).values
                ti = M._clip_log(L.realized_vol_forward(c, h)).values
                ok = np.isfinite(Xi).all(axis=1) & np.isfinite(li) & np.isfinite(ti)
                d = per_h[h]
                d["X"].append(Xi[ok]); d["lr"].append(li[ok]); d["tgt"].append(ti[ok])
                d["yr"].append(yri[ok]); d["sid"].append(np.full(int(ok.sum()), i, np.int32))
                d["dt"].append(dti[ok])
        for h in M.HORIZONS:
            d = per_h[h]
            if not d["X"]:
                continue
            cache[(g, h)] = dict(
                X=np.vstack(d["X"]), lr=np.concatenate(d["lr"]), tgt=np.concatenate(d["tgt"]),
                yr=np.concatenate(d["yr"]), sid=np.concatenate(d["sid"]),
                dt=np.concatenate(d["dt"]), cols=cols, base=base, symbols=syms)
            print(f"  cached {g:<7} h={h:<3} rows={len(cache[(g, h)]['lr']):>9,} "
                  f"cols={len(cols)}")
    pd.to_pickle(cache, CACHE)
    return cache


def load_cache():
    if not CACHE.exists():
        return build_cache(D.load())
    return pd.read_pickle(CACHE)


# ---------------------------------------------------------------- fast OLS via per-year Gram
class Gram:
    """Per-calendar-year X'X and X'y over the FULL column block. Any variant's OLS on any
    expanding year window is then a KxK solve on a sub-block. Exact, not an approximation."""

    def __init__(self, X, y, yr):
        self.years = np.array(sorted(set(yr)))
        self.G, self.b, self.n = {}, {}, {}
        for u in self.years:
            m = yr == u
            Xu = X[m]
            self.G[u] = Xu.T @ Xu
            self.b[u] = Xu.T @ y[m]
            self.n[u] = int(m.sum())

    def fit(self, sel_years, ci):
        """ci: column indices (0 = intercept). Returns beta or None."""
        G = np.zeros((len(ci), len(ci)))
        b = np.zeros(len(ci))
        n = 0
        ix = np.ix_(ci, ci)
        for u in sel_years:
            G += self.G[u][ix]
            b += self.b[u][ci]
            n += self.n[u]
        if n < 200:
            return None, 0
        try:
            beta = np.linalg.solve(G + 1e-10 * np.eye(len(ci)) * np.trace(G) / len(ci), b)
        except np.linalg.LinAlgError:
            return None, 0
        return beta, n


def bucket_idx(lr, thr):
    a_dn, a_up = np.log(1 - thr), np.log(1 + thr)
    return np.where(lr <= a_dn, 0, np.where(lr <= 0, 1, np.where(lr < a_up, 2, 3)))


def brier4(p, a):
    oh = np.zeros_like(p); oh[np.arange(len(a)), a] = 1.0
    return ((p - oh) ** 2).sum(axis=1).mean()


def probs_from(z_sorted, sig, thr):
    a_dn, a_up = np.log(1 - thr), np.log(1 + thr)
    n = len(z_sorted)
    F = lambda v: np.searchsorted(z_sorted, v, side="right") / n
    f_dn, f_0, f_up = F(a_dn / sig), F(0.0), F(a_up / sig)
    p = np.column_stack([f_dn, f_0 - f_dn, f_up - f_0, 1 - f_up])
    p = np.clip(p, 1e-4, None)
    return p / p.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- walk-forward
def test_years_of(d, gram):
    years = list(gram.years)
    out = []
    for ty in years[MIN_TRAIN_YEARS:]:
        tr, te = d["yr"] < ty, d["yr"] == ty
        if tr.sum() < 2000 or te.sum() < 50:
            continue
        out.append(ty)
    return out


def prep_ref(d, gram, thrs=THRS):
    """Model-INDEPENDENT reference block: actuals, per-symbol shrunk climatology, year, date.
    Identical for every variant, so it is computed once and reused."""
    yr, sid, lr = d["yr"], d["sid"], d["lr"]
    n_sym = int(sid.max()) + 1
    tys = test_years_of(d, gram)
    ref = {}
    for thr in thrs:
        a_all = bucket_idx(lr, thr)
        PS, A, YY, DD = [], [], [], []
        for ty in tys:
            tr, te = yr < ty, yr == ty
            a_tr, a_te = a_all[tr], a_all[te]
            pc = np.bincount(a_tr, minlength=4) / len(a_tr)
            cnt = np.bincount(sid[tr] * 4 + a_tr,
                              minlength=n_sym * 4).reshape(n_sym, 4).astype(float)
            ps_sym = (cnt + SHRINK * pc) / (cnt.sum(1, keepdims=True) + SHRINK)
            PS.append(ps_sym[sid[te]]); A.append(a_te)
            YY.append(np.full(int(te.sum()), ty)); DD.append(d["dt"][te])
        ref[thr] = dict(ps=np.vstack(PS), a=np.concatenate(A),
                        yy=np.concatenate(YY), dd=np.concatenate(DD))
    return ref, tys


def walk_forward_p(d, ci, gram, tys, thrs=THRS, kcache=None):
    """Model probabilities per threshold, rows stacked in the same order as prep_ref."""
    lr, yr = d["lr"], d["yr"]
    years = list(gram.years)
    Xc = np.ascontiguousarray(d["X"][:, ci])
    sq = np.sqrt(d["h"])
    kcache = {} if kcache is None else kcache
    out = {thr: [] for thr in thrs}

    for ty in tys:
        tr, te = yr < ty, yr == ty
        beta, _ = gram.fit([u for u in years if u < ty], ci)
        sig_tr = np.exp(Xc[tr] @ beta) * sq
        sig_te = np.exp(Xc[te] @ beta) * sq

        # kappa: median over INNER folds strictly inside the training window.
        # Fold iy's value depends only on iy (beta and z both come from yr < iy), never on the
        # outer test year, so it is memoised across test years -- 3x fewer 1M-row sorts.
        ks = []
        for iy in [u for u in years if u < ty][-INNER_FOLDS:]:
            if iy not in kcache:
                itr, iva = yr < iy, yr == iy
                if itr.sum() < 5000 or iva.sum() < 50:
                    kcache[iy] = None
                else:
                    bi, _ = gram.fit([u for u in years if u < iy], ci)
                    zi = np.sort(lr[itr] / (np.exp(Xc[itr] @ bi) * sq))
                    kcache[iy] = (M._calibrate_kappa(zi, np.exp(Xc[iva] @ bi) * sq, lr[iva])
                                  if len(zi) >= 5000 else None)
            if kcache[iy] is not None:
                ks.append(kcache[iy])
        kappa = float(np.median(ks)) if ks else 1.0
        z_tr = np.sort(lr[tr] / sig_tr) * kappa
        for thr in thrs:
            out[thr].append(probs_from(z_tr, sig_te, thr))
    return {thr: np.vstack(v) for thr, v in out.items()}


def score(p, r):
    ps, a, yy = r["ps"], r["a"], r["yy"]
    big = np.isin(a, [0, 3]).astype(float)
    pm, pms = p[:, 0] + p[:, 3], ps[:, 0] + ps[:, 3]
    se = (pm - big) ** 2
    se_b = (pms - big) ** 2
    per_yr = {u: 1 - se[yy == u].mean() / se_b[yy == u].mean() for u in np.unique(yy)}
    return dict(bss2=1 - se.mean() / se_b.mean(), ece=L.ece(pm, big.astype(bool)), n=len(a),
                obs=big.mean(), pred=pm.mean(), se=se, se_b=se_b, big=big, pm=pm,
                per_yr=per_yr, bss4=1 - brier4(p, a) / brier4(ps, a))


# ---------------------------------------------------------------- variants
def variant_cols(d, extra):
    cols = d["cols"]
    ci = [0] + [cols.index(c) + 1 for c in d["base"]]
    ci += [cols.index(c) + 1 for c in extra]
    return ci


VARIANTS = {
    "base":        [],
    "+vs_d":       ["vs_d"],
    "+vsurp":      ["vs_d", "vs_w"],
    "+ldv22":      ["ldv22"],
    "+ldvblk":     ["ldv5", "ldv22", "ldv63"],
    "+amih":       ["amih"],
    "+vt_q":       ["vt_q"],
    "+vtrend":     ["vs_w", "vt_q"],
    "+vvol":       ["vvol"],
    "+all":        ["ldv5", "ldv22", "ldv63", "vs_d", "vs_w", "vt_q", "amih", "vvol"],
}

VARIANTS2 = {
    "+zr22":       ["zr22"],
    "+clip22":     ["clip22"],
    "+zrblk":      ["zr22", "clip22"],
    "+ldv22+zr22": ["ldv22", "zr22"],
    "+allx":       VOL_FEATS,
}


# ---------------------------------------------------------------- liquidity subgroup calibration
N_DEC = 10


def liq_deciles(d, tys):
    """Point-in-time dollar-volume decile of every test row: breakpoints from TRAIN years only."""
    ldv = d["X"][:, d["cols"].index("ldv22") + 1]
    yr = d["yr"]
    out = []
    for ty in tys:
        tr, te = yr < ty, yr == ty
        q = np.quantile(ldv[tr], np.linspace(0, 1, N_DEC + 1)[1:-1])
        out.append(np.searchsorted(q, ldv[te], side="right"))
    return np.concatenate(out)


def stage_liq(cache, thrs=THRS):
    """Is the SHIPPED baseline miscalibrated for thin names? obs/pred by liquidity decile."""
    print("\n=== STAGE liq: baseline obs/pred of P(|move|>=thr) by dollar-volume decile ===")
    rows = []
    for (g, h), d in sorted(cache.items()):
        if g != "single":
            continue
        d["h"] = h
        gram = Gram(d["X"], d["tgt"], d["yr"])
        ref, tys = prep_ref(d, gram, thrs)
        pb = walk_forward_p(d, variant_cols(d, []), gram, tys, thrs, kcache={})
        dec = liq_deciles(d, tys)
        for thr in thrs:
            s = score(pb[thr], ref[thr])
            big, pm = s["big"], s["pm"]
            ldv = None
            for k in range(N_DEC):
                m = dec == k
                if m.sum() < 100:
                    continue
                obs, pred = big[m].mean(), pm[m].mean()
                se_b = s["se_b"][m].mean()
                rows.append(dict(group=g, h=h, thr=thr, dec=k, n=int(m.sum()),
                                 obs=obs, pred=pred, ratio=obs / pred if pred else np.nan,
                                 gap=obs - pred,
                                 bss2=1 - s["se"][m].mean() / se_b,
                                 ece=L.ece(pm[m], big[m].astype(bool))))
        print(f"  {g} h={h} done")
    df = pd.DataFrame(rows)
    df.to_pickle("_move_fx_volliq_liq.pkl")
    pd.set_option("display.width", 250)
    for thr in thrs:
        print(f"\n  --- thr={thr*100:.0f}%  (dec 0 = thinnest 10% by 22d dollar volume) ---")
        piv = df[df.thr == thr].pivot(index="dec", columns="h", values="ratio")
        npv = df[df.thr == thr].pivot(index="dec", columns="h", values="n")
        print("  obs/pred ratio by horizon:")
        print(piv.round(4).to_string())
        print("  n:")
        print(npv.to_string())
    return df


def stage_decfix(cache, thrs=THRS):
    """Two candidate fixes for any thin-name miscalibration, both refit on prior years only:
       dec_z   -- a separate empirical z table per liquidity decile (shape fix)
       dec_sig -- a per-decile multiplicative sigma correction (scale fix)"""
    print("\n=== STAGE decfix: per-liquidity-decile recalibration ===")
    out = []
    for (g, h), d in sorted(cache.items()):
        if g != "single":
            continue
        d["h"] = h
        lr, yr = d["lr"], d["yr"]
        gram = Gram(d["X"], d["tgt"], d["yr"])
        ref, tys = prep_ref(d, gram, thrs)
        ci = variant_cols(d, [])
        Xc = np.ascontiguousarray(d["X"][:, ci])
        ldv = d["X"][:, d["cols"].index("ldv22") + 1]
        sq = np.sqrt(h)
        years = list(gram.years)
        kcache = {}
        P = {"dec_z": {t: [] for t in thrs}, "dec_sig": {t: [] for t in thrs}}
        for ty in tys:
            tr, te = yr < ty, yr == ty
            beta, _ = gram.fit([u for u in years if u < ty], ci)
            sig_tr, sig_te = np.exp(Xc[tr] @ beta) * sq, np.exp(Xc[te] @ beta) * sq
            ks = []
            for iy in [u for u in years if u < ty][-INNER_FOLDS:]:
                if iy not in kcache:
                    itr, iva = yr < iy, yr == iy
                    if itr.sum() < 5000 or iva.sum() < 50:
                        kcache[iy] = None
                    else:
                        bi, _ = gram.fit([u for u in years if u < iy], ci)
                        zi = np.sort(lr[itr] / (np.exp(Xc[itr] @ bi) * sq))
                        kcache[iy] = M._calibrate_kappa(zi, np.exp(Xc[iva] @ bi) * sq, lr[iva])
                if kcache[iy] is not None:
                    ks.append(kcache[iy])
            kappa = float(np.median(ks)) if ks else 1.0

            q = np.quantile(ldv[tr], np.linspace(0, 1, N_DEC + 1)[1:-1])
            dtr = np.searchsorted(q, ldv[tr], side="right")
            dte = np.searchsorted(q, ldv[te], side="right")
            z_all = np.sort(lr[tr] / sig_tr) * kappa
            ztr_raw = lr[tr] / sig_tr
            lr_tr = lr[tr]

            pz = {t: np.empty((int(te.sum()), 4)) for t in thrs}
            psg = {t: np.empty((int(te.sum()), 4)) for t in thrs}
            for k in range(N_DEC):
                mtr, mte = dtr == k, dte == k
                if mte.sum() == 0:
                    continue
                # (a) decile-specific z table
                zk = np.sort(ztr_raw[mtr]) * kappa if mtr.sum() >= 5000 else z_all
                # (b) decile-specific sigma multiplier, chosen on TRAIN rows of this decile
                best, berr = 1.0, np.inf
                for mlt in np.linspace(0.80, 1.30, 26):
                    err = 0.0
                    for t in thrs:
                        a_dn, a_up = np.log(1 - t), np.log(1 + t)
                        s2 = sig_tr[mtr] * mlt
                        F = lambda v: np.searchsorted(z_all, v, side="right") / len(z_all)
                        pmv = F(a_dn / s2) + (1 - F(a_up / s2))
                        err += abs(pmv.mean() - float(((lr_tr[mtr] <= a_dn)
                                                       | (lr_tr[mtr] >= a_up)).mean()))
                    if err < berr:
                        best, berr = float(mlt), err
                for t in thrs:
                    pz[t][mte] = probs_from(zk, sig_te[mte], t)
                    psg[t][mte] = probs_from(z_all, sig_te[mte] * best, t)
            for t in thrs:
                P["dec_z"][t].append(pz[t]); P["dec_sig"][t].append(psg[t])

        pb = walk_forward_p(d, ci, gram, tys, thrs, kcache={})
        for t in thrs:
            b = score(pb[t], ref[t])
            for name in ("dec_z", "dec_sig"):
                s = score(np.vstack(P[name][t]), ref[t])
                nyr = sum(s["per_yr"][u] > b["per_yr"][u] for u in s["per_yr"])
                out.append(dict(group=g, h=h, thr=t, var=name, n=s["n"],
                                bss2=s["bss2"], bss2_base=b["bss2"], dbss2=s["bss2"] - b["bss2"],
                                ece=s["ece"], ece_base=b["ece"], dece=s["ece"] - b["ece"],
                                yrs_up=nyr, yrs=len(s["per_yr"])))
        print(f"  single h={h} done")
    df = pd.DataFrame(out)
    df.to_pickle("_move_fx_volliq_decfix.pkl")
    pd.set_option("display.width", 250)
    print(df.to_string())
    return df


def stage_repro(cache):
    print("\n=== STAGE repro: baseline on the volume-restricted row set ===")
    tgt = {("index", 1): .192, ("index", 5): .133, ("index", 10): .092, ("index", 21): .034,
           ("single", 1): .114, ("single", 5): .055, ("single", 10): .033, ("single", 21): .018}
    rows = []
    for (g, h), d in sorted(cache.items()):
        d["h"] = h
        gram = Gram(d["X"], d["tgt"], d["yr"])
        ref, tys = prep_ref(d, gram)
        p = walk_forward_p(d, variant_cols(d, []), gram, tys)
        s = score(p[0.02], ref[0.02])
        rows.append(dict(group=g, h=h, n=s["n"], bss2=s["bss2"], ece=s["ece"],
                         want=tgt[(g, h)], diff=s["bss2"] - tgt[(g, h)]))
        print(f"  {g:<7} h={h:<3} n={s['n']:>9,}  BSS2={s['bss2']:+.4f} "
              f"(shipped {tgt[(g,h)]:+.3f}, diff {s['bss2']-tgt[(g,h)]:+.4f})  ECE={s['ece']:.4f}")
    return pd.DataFrame(rows)


def stage_feat(cache, variants=None, tag="feat"):
    variants = variants or VARIANTS
    print(f"\n=== STAGE {tag}: delta BSS2 / delta ECE vs shipped baseline ===")
    out = []
    for (g, h), d in sorted(cache.items()):
        d["h"] = h
        t0 = time.time()
        gram = Gram(d["X"], d["tgt"], d["yr"])
        ref, tys = prep_ref(d, gram)
        pb = walk_forward_p(d, variant_cols(d, []), gram, tys, kcache={})
        base_s = {thr: score(pb[thr], ref[thr]) for thr in THRS}
        for name, extra in variants.items():
            if name == "base":
                continue
            pv = walk_forward_p(d, variant_cols(d, extra), gram, tys, kcache={})
            for thr in THRS:
                s, b = score(pv[thr], ref[thr]), base_s[thr]
                nyr = sum(s["per_yr"][u] > b["per_yr"][u] for u in s["per_yr"])
                out.append(dict(group=g, h=h, thr=thr, var=name, n=s["n"],
                                bss2=s["bss2"], bss2_base=b["bss2"],
                                dbss2=s["bss2"] - b["bss2"],
                                ece=s["ece"], ece_base=b["ece"], dece=s["ece"] - b["ece"],
                                yrs_up=nyr, yrs=len(s["per_yr"])))
        print(f"  {g:<7} h={h:<3} done in {time.time()-t0:.0f}s")
    df = pd.DataFrame(out)
    df.to_pickle(f"_move_fx_volliq_{tag}.pkl")
    return df


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "cache":
        build_cache(D.load())
    else:
        c = load_cache()
        pd.set_option("display.width", 250)
        if stage in ("repro", "all"):
            stage_repro(c)
        if stage in ("feat", "all"):
            df = stage_feat(c)
            print(df.sort_values(["thr", "group", "h", "dbss2"], ascending=[1, 1, 1, 0]).to_string())
        if stage in ("feat2", "all"):
            df = stage_feat(c, VARIANTS2, tag="feat2")
            print(df.sort_values(["thr", "group", "h", "dbss2"], ascending=[1, 1, 1, 0]).to_string())
        if stage in ("liq", "all"):
            stage_liq(c)
        if stage in ("decfix", "all"):
            stage_decfix(c)
