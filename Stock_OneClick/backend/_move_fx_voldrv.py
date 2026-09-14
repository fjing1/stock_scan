"""
_move_fx_voldrv.py -- does the VOLATILITY-DERIVATIVES COMPLEX improve move_prob.py?

Factor family under test (the previous round used only spot ^VIX and flagged term structure as
UNTESTED):
  1. HORIZON MATCHING   ^VIX9D for h=1/5, ^VIX3M for h=21, instead of one 30-day ^VIX for all h
  2. TERM STRUCTURE     log(VIX/VIX3M) as its own feature (backwardation = acute stress)
  3. VVIX               vol-of-vol -- does it forecast the SHAPE (z tails) rather than the scale?
  4. VRP                log(VIX / recent realized vol) as a state variable
  5. SPILLOVER          does any of the above add for SINGLE NAMES beyond spot VIX?
  6. AVAILABILITY       usable spans + the fallback ladder

BASELINE REPRODUCED EXACTLY (move_prob.py scale-and-shape):
  sigma_h = exp(OLS in log space) * sqrt(h), per (asset class, horizon)
  features clipped into [1e-3, 0.5] BEFORE log: rv_d rv_w rv_m rv_q ewma97 (+ logvix for INDEX)
  z = log(close[t+h]/close[t]) / sigma_h  (LOG space); z table = sorted z over the group * kappa
  P(down) = F(log(1-thr)/sigma_h), P(up) = 1 - F(log(1+thr)/sigma_h), clip 1e-4, renormalise
  single-name sigma_h *= EARN_MULT[h] when a report falls in (t, t+h]

MEASUREMENT (identical for every factor, or the numbers are not comparable):
  walk-forward by calendar year, expanding, >=5 train years, every parameter refit on strictly
  prior years.  Reference = EACH TICKER's OWN train-year base rate shrunk 40 pseudo-counts toward
  the pooled rate.  Metrics: BSS2 on P(|move|>=thr), ECE on the same, per-test-year win count.
  CIs: DATE-block bootstrap, block = 63 sessions (> every h), whole cross-section moves together.

APPLES-TO-APPLES RULE: ^VIX9D starts 2011, ^VIX3M 2006-07, ^VVIX 2007-01, and all three STOP at
2026-07-17 in the Yahoo feed.  Every comparison inside one experiment is therefore run on the
INTERSECTION of the rows where all competing specs have finite features, so the baseline and the
challenger always see the identical sample.  Rows are never forward-filled.

Run:  ../../vcp_env/bin/python _move_fx_voldrv.py <stage>
      stages: base | horiz | slope | vvix | vrp | single | avail | all
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _move_data as D
import _move_lib as L
import move_prob as M
import _move_fx_voldrv_fetch as FX

T0 = time.time()
np.seterr(all="ignore")

HS = (1, 5, 10, 21)
THRS = (0.02, 0.05)
MIN_TRAIN_YEARS = 5
FLOOR = 1e-4
SHRINK = 40.0
BOOT_BLK = 63
NBOOT = 400
EARN_PATH = "/Users/feijing/github.com/stock_scan/gold_pine_script/pead_earnings_cache.json"
CACHE = Path(__file__).with_name("_move_fx_voldrv_cache.pkl")


def hdr(s):
    print("\n" + "=" * 108)
    print(s)
    print("=" * 108, flush=True)


# ============================================================ market-wide vol-derivative features
def market_feats(index: pd.DatetimeIndex, C: pd.DataFrame, H: pd.DataFrame, Lo: pd.DataFrame):
    """Every market-wide (same for all symbols) regressor, aligned to the panel's date index.
    NO ffill anywhere: a missing session stays NaN and the row simply drops out of any spec that
    uses it."""
    V = FX.load().reindex(index)
    vix_panel = C["^VIX"]                       # the panel's own ^VIX, identical source
    f = {}
    sq = math.sqrt(252.0)

    def liv(s):                                 # index level -> log per-day sigma, clipped
        return M._clip_log(s / 100.0 / sq)

    f["lv"] = liv(vix_panel)                            # baseline logvix (30d)
    f["lv9"] = liv(V["^VIX9D"])                         # 9-day
    f["lv3m"] = liv(V["^VIX3M"])                        # 3-month
    f["lv6m"] = liv(V["^VIX6M"])                        # 6-month
    # term-structure slopes (dimensionless log ratios; NOT clipped -- they are not vols)
    f["ts_9_30"] = np.log(V["^VIX9D"] / vix_panel)
    f["ts_30_3m"] = np.log(vix_panel / V["^VIX3M"])     # >0 = backwardation = acute stress
    f["ts_3m_6m"] = np.log(V["^VIX3M"] / V["^VIX6M"])
    f["lvvix"] = np.log(V["^VVIX"] / 100.0)             # vol-of-vol, log level
    # variance risk premium: log(implied 30d / realized 21d of the market)
    spy = "SPY" if "SPY" in C.columns else "^GSPC"
    rv21_mkt = L.vol_parkinson(H[spy], Lo[spy], 21)
    f["lrv_mkt"] = M._clip_log(rv21_mkt)
    f["vrp"] = f["lv"] - f["lrv_mkt"]
    f["lvxn"] = liv(V["^VXN"])
    return pd.DataFrame(f, index=index), V


# ============================================================ panel -> flat arrays
def build(verbose=True):
    p = D.load()
    C, Hi, Lo = p["Close"], p["High"], p["Low"]
    idx = C.index
    MF, V = market_feats(idx, C, Hi, Lo)
    pos_of = {d: i for i, d in enumerate(idx)}

    usable = [s for s in C.columns if s != "^VIX" and C[s].notna().sum() >= 500]
    groups = {"INDEX": [s for s in usable if s in M.INDEX_LIKE],
              "SINGLE": [s for s in usable if s not in M.INDEX_LIKE]}
    mnames = list(MF.columns)
    out = {"index": idx, "mnames": mnames, "V": V}

    for g, syms in groups.items():
        POS, SID, FB, LR, TG = [], [], [], {h: [] for h in HS}, {h: [] for h in HS}
        for i, s in enumerate(syms):
            c = C[s].dropna()
            if len(c) < 400:
                continue
            # identical to move_prob.build_features: per-symbol, on the symbol's own dropna index
            fr = M.build_features(c, Hi[s].reindex(c.index), Lo[s].reindex(c.index), vix=None)
            FB.append(fr[M.FEATS_OHLC].to_numpy())
            POS.append(np.array([pos_of[d] for d in c.index], dtype=np.int32))
            SID.append(np.full(len(c), i, dtype=np.int32))
            for h in HS:
                LR[h].append(np.log(c.shift(-h) / c).to_numpy())
                TG[h].append(M._clip_log(L.realized_vol_forward(c, h)).to_numpy())
        pos = np.concatenate(POS)
        Fb = np.vstack(FB)
        Fm = MF.to_numpy()[pos]                  # market-wide block, broadcast by date
        A = {"pos": pos, "sid": np.concatenate(SID),
             "yr": idx.year.to_numpy()[pos].astype(np.int16),
             "F": np.hstack([Fb, Fm]).astype(np.float64),
             "fnames": list(M.FEATS_OHLC) + mnames,
             "syms": syms,
             "lr": {h: np.concatenate(LR[h]) for h in HS},
             "tgt": {h: np.concatenate(TG[h]) for h in HS}}
        out[g] = A
        if verbose:
            print(f"  {g:<7} {len(syms):>3} symbols  rows={len(pos):,}")

    out["SINGLE"]["earn"] = earn_flags(out["SINGLE"], idx)
    pd.to_pickle(out, CACHE)
    return out


def earn_flags(A, idx):
    """bool[h] -> per-row flag: a reported earnings date falls in (t, t+h]."""
    raw = json.load(open(EARN_PATH))
    sidx = {s: i for i, s in enumerate(A["syms"])}
    dpos = {d.date(): i for i, d in enumerate(idx)}
    ev = {}
    for s, rows in raw.items():
        if s not in sidx:
            continue
        got = sorted({dpos[t] for t in
                      (pd.Timestamp(str(r[0] if isinstance(r, (list, tuple)) else r)[:10]).date()
                       for r in rows) if t in dpos})
        if len(got) >= 5:
            ev[sidx[s]] = np.array(got)
    nd, ns = len(idx), len(A["syms"])
    out = {}
    for h in HS:
        Fm = np.zeros((nd, ns), bool)
        for j, bs in ev.items():
            for b in bs:
                Fm[max(b - h, 0):b, j] = True          # t in [b-h, b-1]  <=>  b in (t, t+h]
        out[h] = Fm[A["pos"], A["sid"]]
    print(f"  earnings: {len(ev)} of {ns} single names, "
          f"{sum(len(v) for v in ev.values()):,} events")
    return out


def load(rebuild=False):
    if rebuild or not CACHE.exists():
        return build()
    return pd.read_pickle(CACHE)


# ============================================================ estimator + scoring primitives
def ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def design(A, cols, rows):
    ci = [A["fnames"].index(c) for c in cols]
    X = np.empty((int(rows.sum()), len(cols) + 1))
    X[:, 0] = 1.0
    X[:, 1:] = A["F"][np.ix_(rows, ci)]
    return X


def probs_emp(z_sorted, sig, thr):
    n = len(z_sorted)
    a_dn, a_up = math.log(1 - thr), math.log(1 + thr)
    Fdn = np.searchsorted(z_sorted, a_dn / sig, side="right") / n
    Fup = np.searchsorted(z_sorted, a_up / sig, side="right") / n
    F0 = np.searchsorted(z_sorted, 0.0, side="right") / n
    p = np.column_stack([Fdn, F0 - Fdn, Fup - F0, 1.0 - Fup])
    p = np.clip(p, FLOOR, None)
    return p / p.sum(axis=1, keepdims=True)


def bidx(fwd_simple, thr):
    return np.where(fwd_simple <= -thr, 0,
                    np.where(fwd_simple <= 0, 1, np.where(fwd_simple < thr, 2, 3))).astype(np.int8)


def clim_sym(a_tr, sid_tr, sid_te, nsym):
    pool = np.bincount(a_tr, minlength=4).astype(float)
    pool /= pool.sum()
    cs = np.bincount(sid_tr.astype(np.int64) * 4 + a_tr.astype(np.int64),
                     minlength=nsym * 4).reshape(nsym, 4).astype(float)
    cs += SHRINK * pool[None, :]
    cs /= cs.sum(axis=1, keepdims=True)
    p = np.clip(cs[sid_te], FLOOR, None)
    return p / p.sum(axis=1, keepdims=True)


def bss2(p, a, pref):
    hit = ((a == 0) | (a == 3)).astype(float)
    m, b = (p[:, 0] + p[:, 3]) - hit, (pref[:, 0] + pref[:, 3]) - hit
    return 1.0 - float((m ** 2).mean()) / float((b ** 2).mean())


def ece2(p, a, nb=10):
    hit = ((a == 0) | (a == 3)).astype(float)
    return L.ece(p[:, 0] + p[:, 3], hit, nb)


def _wf_kappa(A, cols, h, tr, trv, lr, tgt, yr, test_year, n_folds=8):
    """Median kappa over up to `n_folds` INNER expanding folds, all inside the training years.

    Exactly move_prob.fit's procedure (a single fold is regime-dependent: the last calm year
    returns 0.90-0.98 while the median across folds returns 1.05-1.13), but re-run for every
    walk-forward test year so no test-year information ever touches the width calibration."""
    iy = [q for q in sorted(set(yr[tr].tolist())) if q < test_year][-n_folds:]
    ks = []
    for ty in iy:
        itr = tr & (yr < ty)
        iva = tr & (yr == ty)
        if itr.sum() < 5000 or iva.sum() < 50:
            continue
        itv = itr & np.isfinite(tgt)
        if itv.sum() < 500:
            continue
        b = ols(design(A, cols, itv), tgt[itv])
        z_in = np.sort(lr[itr] / (np.exp(design(A, cols, itr) @ b) * math.sqrt(h)))
        if len(z_in) < 5000:
            continue
        s_va = np.exp(design(A, cols, iva) @ b) * math.sqrt(h)
        ks.append(M._calibrate_kappa(z_in, s_va, lr[iva]))
    return float(np.median(ks)) if ks else 1.0


# ============================================================ the walk-forward engine
def run(A, h, specs, thrs=THRS, kappa=1.0, earn=None, shape_by=None, nshape=3,
        row_mask=None, min_train=5000, gate=None):
    """Walk-forward one (group, horizon) for a dict {name: [feature cols]}.

    All specs share the SAME rows: the intersection of finite features across every spec, so the
    baseline and the challenger are never scored on different samples.

    shape_by: {specname: feature} -> instead of one pooled z table, cut the train z into `nshape`
    buckets by that feature (edges = train quantiles) and give each bucket its own z table.
    This is how a factor is allowed to move the SHAPE rather than the scale.
    """
    lr, tgt = A["lr"][h], A["tgt"][h]
    fwd_s = np.expm1(lr)
    allc = sorted({c for v in specs.values() for c in v})
    ci = [A["fnames"].index(c) for c in allc]
    finite = np.isfinite(A["F"][:, ci]).all(axis=1) & np.isfinite(lr)
    if row_mask is not None:
        finite &= row_mask
    if shape_by:
        for f in set(shape_by.values()):
            finite &= np.isfinite(A["F"][:, A["fnames"].index(f)])
    yr, pos, sid = A["yr"], A["pos"], A["sid"]
    nsym = len(A["syms"])
    years = sorted(set(yr[finite]))
    if len(years) <= MIN_TRAIN_YEARS:
        return None
    em = np.ones(len(lr)) if earn is None else np.where(earn, M.EARN_MULT[h], 1.0)

    keys = list(specs)
    store = {k: {t: [] for t in thrs} for k in keys + ["CLIMSYM"]}
    meta = {"yr": [], "pos": [], "a": {t: [] for t in thrs},
            "r2num": {k: [] for k in keys}, "r2den": [], "n": 0}
    for y in years[MIN_TRAIN_YEARS:]:
        te = finite & (yr == y)
        tr = finite & (yr < y)
        if te.sum() < 50 or tr.sum() < min_train:
            continue
        trv = tr & np.isfinite(tgt)
        if trv.sum() < 500:
            continue
        sig_tr, sig_te, kap = {}, {}, {}
        ybar = float(tgt[trv].mean())
        tev = te & np.isfinite(tgt)
        for k, cols in specs.items():
            if gate and k in gate:
                gcol, gmin, gfall = gate[k]
                # IDENTIFICATION GATE.  Until enough TRAIN rows actually observe the extra series,
                # its column is degenerate (all-zero + a constant dummy) and lstsq's minimum-norm
                # solution silently splits the intercept between const and the dummy -- which then
                # vanishes on test rows where the dummy is 0 and blows sigma up.  Measured cost of
                # NOT gating: dBSS2 -0.104 in test year 2011 alone.  Fall back until identified.
                if int(np.isfinite(A["F"][:, A["fnames"].index(gcol)][tr]).sum()) < gmin:
                    cols = gfall
            b = ols(design(A, cols, trv), tgt[trv])
            sig_tr[k] = np.exp(design(A, cols, tr) @ b) * math.sqrt(h)
            sig_te[k] = np.exp(design(A, cols, te) @ b) * math.sqrt(h)
            pred_v = design(A, cols, tev) @ b          # predicted log per-day sigma
            meta["r2num"][k].append(float(((tgt[tev] - pred_v) ** 2).sum()))
            # kappa: the shipped z-table WIDTH multiplier, re-estimated for EVERY test year on
            # INNER expanding folds drawn only from the training years (never on test).  Per spec,
            # so a challenger is never handed the baseline's width calibration or vice versa.
            kap[k] = 1.0 if kappa != "wf" else _wf_kappa(A, cols, h, tr, trv, lr, tgt, yr, y)
        meta["r2den"].append(float(((tgt[tev] - ybar) ** 2).sum()))
        meta.setdefault("kappa", {}).setdefault(y, dict(kap))

        for t in thrs:
            a_tr = bidx(fwd_s[tr], t)
            store["CLIMSYM"][t].append(clim_sym(a_tr, sid[tr], sid[te], nsym))
        for k in keys:
            kappa_k = kap[k] if kappa == "wf" else kappa
            s_tr = sig_tr[k] * em[tr]
            s_te = sig_te[k] * em[te]
            if shape_by and k in shape_by:
                fcol = A["F"][:, A["fnames"].index(shape_by[k])]
                edges = np.percentile(fcol[tr], np.linspace(0, 100, nshape + 1)[1:-1])
                b_tr, b_te = np.digitize(fcol[tr], edges), np.digitize(fcol[te], edges)
                ztab = {}
                z_all = np.sort(lr[tr] / s_tr) * kappa_k
                for q in range(nshape):
                    m = b_tr == q
                    ztab[q] = np.sort(lr[tr][m] / s_tr[m]) * kappa_k if m.sum() > 2000 else z_all
                for t in thrs:
                    pp = np.empty((int(te.sum()), 4))
                    for q in range(nshape):
                        mv = b_te == q
                        if mv.any():
                            pp[mv] = probs_emp(ztab[q], s_te[mv], t)
                    store[k][t].append(pp)
            else:
                zs = np.sort(lr[tr] / s_tr) * kappa_k
                for t in thrs:
                    store[k][t].append(probs_emp(zs, s_te, t))
        meta["yr"].append(np.full(int(te.sum()), y))
        meta["pos"].append(pos[te])
        for t in thrs:
            meta["a"][t].append(bidx(fwd_s[te], t))
    if not meta["yr"]:
        return None
    R = {"yr": np.concatenate(meta["yr"]), "pos": np.concatenate(meta["pos"]),
         "a": {t: np.concatenate(meta["a"][t]) for t in thrs},
         "p": {k: {t: np.vstack(store[k][t]) for t in thrs} for k in keys + ["CLIMSYM"]},
         "r2": {k: 1.0 - sum(meta["r2num"][k]) / sum(meta["r2den"]) for k in keys},
         "kappa": meta.get("kappa", {}),
         "years": sorted(set(np.concatenate(meta["yr"]).tolist()))}
    return R


# ============================================================ reporting helpers
def table(R, keys, thrs=THRS, ref="CLIMSYM"):
    rows = []
    for t in thrs:
        a, pr = R["a"][t], R["p"][ref][t]
        for k in keys:
            p = R["p"][k][t]
            rows.append(dict(thr=t, spec=k, n=len(a), bss2=bss2(p, a, pr), ece=ece2(p, a),
                             r2=R["r2"][k]))
    return pd.DataFrame(rows)


def year_wins(R, ka, kb, t):
    """(#years spec ka beats kb on BSS2, total years, mean delta)."""
    a, pr = R["a"][t], R["p"]["CLIMSYM"][t]
    pa, pb = R["p"][ka][t], R["p"][kb][t]
    w, n, ds = 0, 0, []
    for y in R["years"]:
        m = R["yr"] == y
        if m.sum() < 30:
            continue
        d = bss2(pa[m], a[m], pr[m]) - bss2(pb[m], a[m], pr[m])
        ds.append(d)
        w += d > 0
        n += 1
    return w, n, float(np.mean(ds)) if ds else np.nan


def boot_delta(R, ka, kb, t, reps=NBOOT, blk=BOOT_BLK, seed=11):
    """DATE-block bootstrap of delta-BSS2.  Blocks are non-overlapping tiles of `blk` consecutive
    trading dates (63 > every horizon tested, so overlapping forward windows stay inside a block);
    the whole cross-section on a date travels together, because 231 stocks are ~1-correlated on a
    crash day and pooled n is not independent n.  Per-block error SUMS are precomputed, so a rep
    is a sum over ~80 numbers rather than a 1M-row gather."""
    rng = np.random.default_rng(seed)
    a, pr = R["a"][t], R["p"]["CLIMSYM"][t]
    pa, pb = R["p"][ka][t], R["p"][kb][t]
    hit = ((a == 0) | (a == 3)).astype(float)
    ea = ((pa[:, 0] + pa[:, 3]) - hit) ** 2
    eb = ((pb[:, 0] + pb[:, 3]) - hit) ** 2
    er = ((pr[:, 0] + pr[:, 3]) - hit) ** 2
    ud = np.unique(R["pos"])
    rank = np.searchsorted(ud, R["pos"])            # date -> 0..n_dates-1
    tile = rank // blk
    nbk = int(tile.max()) + 1
    Sa = np.bincount(tile, weights=ea, minlength=nbk)
    Sb = np.bincount(tile, weights=eb, minlength=nbk)
    Sr = np.bincount(tile, weights=er, minlength=nbk)
    Cn = np.bincount(tile, minlength=nbk).astype(float)
    out = np.empty(reps)
    for i in range(reps):
        q = rng.integers(0, nbk, nbk)
        c = Cn[q].sum()
        r = Sr[q].sum() / c
        out[i] = (1 - (Sa[q].sum() / c) / r) - (1 - (Sb[q].sum() / c) / r)
    lo, hi = np.percentile(out, [2.5, 97.5])
    rr = er.mean()
    pt = (1 - ea.mean() / rr) - (1 - eb.mean() / rr)
    return pt, float(lo), float(hi), float((out > 0).mean()), len(ud)


def show(R, keys, base, thrs=THRS, label="", boot_for=()):
    tb = table(R, keys, thrs)
    print(f"{'thr':>5} {'spec':<20} {'n':>9} {'ndate':>6} {'R2vol':>8} {'BSS2':>8} {'dBSS2':>8} "
          f"{'ECE':>7} {'dECE':>8} {'yrs+':>7}")
    nd = len(np.unique(R["pos"]))
    for t in thrs:
        sub = tb[tb.thr == t].set_index("spec")
        b = sub.loc[base]
        for k in keys:
            r = sub.loc[k]
            if k == base:
                print(f"{t:>5.0%} {k:<20} {int(r.n):>9,} {nd:>6,} {r.r2:>8.4f} {r.bss2:>8.4f} "
                      f"{'--':>8} {r.ece:>7.4f} {'--':>8} {'--':>7}")
            else:
                w, ny, _ = year_wins(R, k, base, t)
                print(f"{t:>5.0%} {k:<20} {int(r.n):>9,} {nd:>6,} {r.r2:>8.4f} {r.bss2:>8.4f} "
                      f"{r.bss2-b.bss2:>+8.4f} {r.ece:>7.4f} {r.ece-b.ece:>+8.4f} {w:>3}/{ny:<3}")
    for k in boot_for:
        for t in thrs:
            pt, lo, hi, pg, ndte = boot_delta(R, k, base, t)
            print(f"      boot dBSS2 {k} vs {base} @thr={t:.0%}: {pt:+.4f} "
                  f"95%CI[{lo:+.4f},{hi:+.4f}] P(>0)={pg:.3f} n_dates={ndte:,}")


# ============================================================ stage: reproduce the baseline
BASE = {"INDEX": list(M.FEATS_OHLC) + ["lv"], "SINGLE": list(M.FEATS_OHLC)}
# kappa policy, verified in stage_base:
#   INDEX  -- re-estimated walk-forward every test year (it is what halves index ECE; cheap here,
#             the group is 31k rows).  Fit per spec, so no spec inherits another's width.
#   SINGLE -- held at 1.0.  Walk-forward kappa for singles lands at 0.98-1.02 and changes BSS2 by
#             0.0016 / ECE by 0.001 while costing ~140 s per (horizon, spec); it is applied
#             identically to baseline and challenger either way, so it cannot move a DELTA.
KAP = {"INDEX": "wf", "SINGLE": 1.0}
TARGET = {("INDEX", 0.02): [.192, .133, .092, .034], ("SINGLE", 0.02): [.114, .055, .033, .018]}
TARGET_ECE = {("INDEX", 0.02): [.007, .016, .018, .026],
              ("SINGLE", 0.02): [.013, .007, .007, .010]}


def stage_base(B, kappa=None):
    hdr(f"BASELINE REPRODUCTION (kappa={kappa or KAP}) -- must match the shipped numbers "
        f"within ~0.002")
    print(f"{'grp':<7}{'h':>3}{'thr':>6}{'n':>10}{'ndate':>7}{'R2vol':>8}{'BSS2':>8}{'target':>8}"
          f"{'diff':>8}{'ECE':>8}{'tgtECE':>8}{'diff':>8}")
    got = {}
    for g in ("INDEX", "SINGLE"):
        A = B[g]
        for j, h in enumerate(HS):
            R = run(A, h, {"base": BASE[g]}, kappa=(KAP[g] if kappa is None else kappa),
                    earn=(A["earn"][h] if g == "SINGLE" else None))
            got[(g, h)] = R
            for t in THRS:
                a = R["a"][t]
                v = bss2(R["p"]["base"][t], a, R["p"]["CLIMSYM"][t])
                e = ece2(R["p"]["base"][t], a)
                tg = TARGET.get((g, t), [np.nan] * 4)[j]
                te = TARGET_ECE.get((g, t), [np.nan] * 4)[j]
                print(f"{g:<7}{h:>3}{t:>6.0%}{len(a):>10,}{len(np.unique(R['pos'])):>7,}"
                      f"{R['r2']['base']:>8.4f}{v:>8.4f}{tg:>8.3f}{v-tg:>+8.4f}"
                      f"{e:>8.4f}{te:>8.3f}{e-te:>+8.4f}", flush=True)
    return got


# ============================================================ stage: availability / fallback
def stage_avail(B):
    hdr("AVAILABILITY OF THE VOLATILITY-DERIVATIVES COMPLEX (no ffill; holes stay holes)")
    idx = B["index"]
    V = B["V"]
    print(f"panel: {len(idx)} sessions {idx[0].date()} -> {idx[-1].date()}")
    print(f"{'series':<9}{'first':>12}{'last':>12}{'n_obs':>8}{'miss_in_span':>14}"
          f"{'longest_hole':>14}{'sessions_after_last':>20}")
    for t in V.columns:
        s = V[t].reindex(idx)
        nn = s.notna().to_numpy()
        fv, lv = s.first_valid_index(), s.last_valid_index()
        span = (idx >= fv) & (idx <= lv)
        d = np.flatnonzero(nn & span)
        gaps = np.diff(d) - 1 if len(d) > 1 else np.array([0])
        print(f"{t:<9}{str(fv.date()):>12}{str(lv.date()):>12}{int(nn.sum()):>8,}"
              f"{int((~nn & span).sum()):>14}{int(gaps.max()):>14}{int((idx > lv).sum()):>20}")
    print("\nUsable walk-forward TEST years per series (needs >=5 prior calendar years of that "
          "series):")
    for t in V.columns:
        s = V[t].reindex(idx)
        ys = sorted(set(idx.year[s.notna()]))
        print(f"  {t:<9} data years {ys[0]}-{ys[-1]}  -> test years {ys[MIN_TRAIN_YEARS]}-{ys[-1]}"
              f"  ({len(ys)-MIN_TRAIN_YEARS} of 21)")
    print("\nCoverage of each feature on the SINGLE-group scored rows:")
    A = B["SINGLE"]
    for c in B["mnames"]:
        v = A["F"][:, A["fnames"].index(c)]
        print(f"  {c:<10} finite {np.isfinite(v).mean():.4f}")


# ============================================================ stage: horizon matching
def stage_horiz(B):
    hdr("1. HORIZON MATCHING -- does a horizon-matched implied vol beat the one-size-fits-all VIX?")
    print("common sample = rows where VIX, VIX9D, VIX3M, VIX6M are ALL present "
          "(2011-01-03..2026-07-17); every spec sees identical rows.")
    F = M.FEATS_OHLC
    specs = {"base(VIX)": list(F) + ["lv"],
             "sub VIX9D": list(F) + ["lv9"],
             "sub VIX3M": list(F) + ["lv3m"],
             "sub VIX6M": list(F) + ["lv6m"],
             "VIX+VIX9D": list(F) + ["lv", "lv9"],
             "VIX+VIX3M": list(F) + ["lv", "lv3m"],
             "VIX+9D+3M": list(F) + ["lv", "lv9", "lv3m"]}
    for h in HS:
        print(f"\n--- INDEX h={h} ---")
        R = run(B["INDEX"], h, specs, kappa=KAP["INDEX"])
        show(R, list(specs), "base(VIX)",
             boot_for=("sub VIX9D", "sub VIX3M", "VIX+9D+3M"))
    hdr("1b. HORIZON MATCHING on the FULL VIX-era sample for the baseline (unequal samples, "
        "shown only to confirm the common-sample restriction is not what kills the challenger)")
    for h in (1, 21):
        Rb = run(B["INDEX"], h, {"base(VIX)": list(F) + ["lv"]}, kappa=KAP["INDEX"])
        print(f"  INDEX h={h:<3} baseline on ALL rows: n={len(Rb['a'][0.02]):,} "
              f"R2={Rb['r2']['base(VIX)']:.4f} BSS2(2%)="
              f"{bss2(Rb['p']['base(VIX)'][0.02], Rb['a'][0.02], Rb['p']['CLIMSYM'][0.02]):.4f}")


# ============================================================ stage: term-structure slope
def stage_slope(B):
    hdr("2. TERM STRUCTURE SLOPE as its own feature, beyond the level")
    F = M.FEATS_OHLC
    specs = {"base(VIX)": list(F) + ["lv"],
             "+slope30/3m": list(F) + ["lv", "ts_30_3m"],
             "+slope9/30": list(F) + ["lv", "ts_9_30"],
             "+both slopes": list(F) + ["lv", "ts_9_30", "ts_30_3m"],
             "+slope3m/6m": list(F) + ["lv", "ts_30_3m", "ts_3m_6m"]}
    for h in HS:
        print(f"\n--- INDEX h={h} ---")
        R = run(B["INDEX"], h, specs, kappa=KAP["INDEX"])
        show(R, list(specs), "base(VIX)", boot_for=("+slope30/3m", "+both slopes"))
    hdr("2b. SLOPE as a SHAPE conditioner (3 z tables cut by the train quantiles of the slope) "
        "-- does backwardation fatten the standardised tails?")
    for h in HS:
        R = run(B["INDEX"], h, {"base(VIX)": list(F) + ["lv"], "shape|slope": list(F) + ["lv"]},
                shape_by={"shape|slope": "ts_30_3m"}, nshape=3, kappa=KAP["INDEX"])
        print(f"\n--- INDEX h={h} ---")
        show(R, ["base(VIX)", "shape|slope"], "base(VIX)")


# ============================================================ stage: VVIX
def stage_vvix(B):
    hdr("3. VVIX -- scale effect, then SHAPE effect (the genuinely new question)")
    F = M.FEATS_OHLC
    specs = {"base(VIX)": list(F) + ["lv"],
             "+VVIX": list(F) + ["lv", "lvvix"],
             "+VVIX+slope": list(F) + ["lv", "lvvix", "ts_30_3m"]}
    for h in HS:
        print(f"\n--- INDEX h={h}  (scale) ---")
        R = run(B["INDEX"], h, specs, kappa=KAP["INDEX"])
        show(R, list(specs), "base(VIX)", boot_for=("+VVIX",))
    hdr("3b. VVIX as a SHAPE conditioner: separate empirical z tables by VVIX state")
    for nsh in (3, 5):
        for h in HS:
            R = run(B["INDEX"], h,
                    {"base(VIX)": list(F) + ["lv"], f"shape|VVIX x{nsh}": list(F) + ["lv"]},
                    shape_by={f"shape|VVIX x{nsh}": "lvvix"}, nshape=nsh, kappa=KAP["INDEX"])
            print(f"\n--- INDEX h={h} nshape={nsh} ---")
            show(R, ["base(VIX)", f"shape|VVIX x{nsh}"], "base(VIX)")
    hdr("3c. DESCRIPTIVE (IN-SAMPLE): does the standardised z actually have fatter tails when "
        "VVIX is high?  z = log fwd return / sigma_hat from the baseline fit on prior years.")
    for h in (1, 5, 21):
        R = run(B["INDEX"], h, {"base(VIX)": list(F) + ["lv"]})
        a = R["a"][0.02]
        vv = B["INDEX"]["F"][:, B["INDEX"]["fnames"].index("lvvix")]
        print(f"  h={h}: (tail diagnostics come from 3b's conditional tables) n={len(a):,}")


# ============================================================ stage: variance risk premium
def stage_vrp(B):
    hdr("4. VARIANCE RISK PREMIUM  log(VIX / realized21) -- level, nonlinearity, shape")
    F = M.FEATS_OHLC
    print("NOTE: for the INDEX group log(VIX) and the symbol's own rv_m are BOTH already in the\n"
          "baseline, so a LINEAR vrp = lv - lrv_mkt lies in the span of the existing design for\n"
          "SPY/^GSPC up to the difference between market and own realized vol.  Any gain here has\n"
          "to come from the non-spanned part, the nonlinearity, or the shape.")
    specs = {"base(VIX)": list(F) + ["lv"],
             "+vrp": list(F) + ["lv", "vrp"],
             "+mkt rv21": list(F) + ["lv", "lrv_mkt"]}
    for h in HS:
        print(f"\n--- INDEX h={h} ---")
        R = run(B["INDEX"], h, specs, kappa=KAP["INDEX"])
        show(R, list(specs), "base(VIX)")
    hdr("4b. VRP as a SHAPE conditioner (3 and 5 z tables by train quantiles of vrp)")
    for nsh in (3, 5):
        for h in HS:
            R = run(B["INDEX"], h,
                    {"base(VIX)": list(F) + ["lv"], f"shape|vrp x{nsh}": list(F) + ["lv"]},
                    shape_by={f"shape|vrp x{nsh}": "vrp"}, nshape=nsh, kappa=KAP["INDEX"])
            print(f"\n--- INDEX h={h} nshape={nsh} ---")
            show(R, ["base(VIX)", f"shape|vrp x{nsh}"], "base(VIX)")


# ============================================================ stage: single names
def stage_single(B):
    hdr("5. SPILLOVER TO SINGLE NAMES -- does the term structure / VVIX add beyond spot VIX?")
    A = B["SINGLE"]
    F = M.FEATS_OHLC
    specs = {"base(shipped)": list(F),
             "+VIX": list(F) + ["lv"],
             "+VIX+slope": list(F) + ["lv", "ts_30_3m"],
             "+VIX+VVIX": list(F) + ["lv", "lvvix"],
             "+VIX+vrp": list(F) + ["lv", "vrp"],
             "+VIX9D": list(F) + ["lv9"],
             "+VIX+9D+3M": list(F) + ["lv", "lv9", "lv3m"],
             "+VIX+slope+VVIX": list(F) + ["lv", "ts_30_3m", "lvvix"]}
    for h in HS:
        print(f"\n--- SINGLE h={h} ---")
        R = run(A, h, specs, earn=A["earn"][h])
        show(R, list(specs), "base(shipped)",
             boot_for=("+VIX", "+VIX+slope", "+VIX+slope+VVIX"))
        print("      --- incremental over +VIX (is the derivatives complex worth anything once "
              "spot VIX is in?) ---")
        for k in ("+VIX+slope", "+VIX+VVIX", "+VIX+vrp", "+VIX+9D+3M", "+VIX+slope+VVIX"):
            for t in THRS:
                pt, lo, hi, pg, nd = boot_delta(R, k, "+VIX", t)
                w, ny, _ = year_wins(R, k, "+VIX", t)
                print(f"      {k:<18} thr={t:.0%}  dBSS2 vs +VIX {pt:+.4f} "
                      f"95%CI[{lo:+.4f},{hi:+.4f}] P(>0)={pg:.3f}  yrs+ {w}/{ny}")
    hdr("5b. SINGLE NAMES: shape conditioning on VVIX / slope (does the derivatives complex move "
        "the single-name z tails?)")
    for h in HS:
        R = run(A, h, {"+VIX": list(F) + ["lv"], "shape|VVIX": list(F) + ["lv"],
                       "shape|slope": list(F) + ["lv"]},
                shape_by={"shape|VVIX": "lvvix", "shape|slope": "ts_30_3m"}, nshape=3,
                earn=A["earn"][h])
        print(f"\n--- SINGLE h={h} ---")
        show(R, ["+VIX", "shape|VVIX", "shape|slope"], "+VIX")


STAGES = {"base": stage_base, "avail": stage_avail, "horiz": stage_horiz, "slope": stage_slope,
          "vvix": stage_vvix, "vrp": stage_vrp, "single": stage_single}



# ============================================================ stage: adversarial control
def stage_ctrl(B):
    hdr("6. IS IT REALLY *HORIZON* MATCHING?  Adversarial controls.\n"
        "   If a second 30-DAY vol index (^VXN) or a pure VIX MOMENTUM term buys the same gain,\n"
        "   then the story is 'a second, fresher market-vol reading', not 'horizon matching'.")
    F = M.FEATS_OHLC
    A = B["INDEX"]
    # VIX momentum: log(VIX_t / VIX_{t-5}) -- built here so it lands in the same feature block
    if "vixmom" not in A["fnames"]:
        idx = B["index"]
        v = np.log(pd.read_pickle(Path(__file__).with_name("_move_panel.pkl"))["Close"]["^VIX"])
        mom = (v - v.shift(5)).to_numpy()
        for g in ("INDEX", "SINGLE"):
            G = B[g]
            G["F"] = np.hstack([G["F"], mom[G["pos"]][:, None]])
            G["fnames"] = G["fnames"] + ["vixmom"]
    specs = {"base(VIX)": list(F) + ["lv"],
             "+VIX9D (9d)": list(F) + ["lv", "lv9"],
             "+VXN (30d ctrl)": list(F) + ["lv", "lvxn"],
             "+VIXmom5 (ctrl)": list(F) + ["lv", "vixmom"],
             "+VXN+VIXmom": list(F) + ["lv", "lvxn", "vixmom"],
             "+VIX9D+VXN+mom": list(F) + ["lv", "lv9", "lvxn", "vixmom"]}
    print("\n### 6a. COMMON SAMPLE (VIX9D era, 2011+) -- all six specs on identical rows")
    for h in HS:
        print(f"\n--- INDEX h={h} ---")
        R = run(B["INDEX"], h, specs, kappa=KAP["INDEX"])
        show(R, list(specs), "base(VIX)", boot_for=("+VIX9D (9d)", "+VXN (30d ctrl)"))
    print("\n### 6b. FULL 21-YEAR SAMPLE, controls only (no VIX9D needed)")
    sp2 = {k: v for k, v in specs.items() if "9D" not in k}
    for h in HS:
        print(f"\n--- INDEX h={h} ---")
        R = run(B["INDEX"], h, sp2, kappa=KAP["INDEX"])
        show(R, list(sp2), "base(VIX)", boot_for=("+VXN (30d ctrl)", "+VIXmom5 (ctrl)"))


# ============================================================ stage: shippable fallback ladder
def run_ladder(A, h, primary, fallback, gate, kappa=1.0, earn=None, thrs=THRS):
    """The move_prob fallback discipline: the primary spec and the degraded spec are SEPARATE
    fits with SEPARATE z tables (dropping a column while keeping the intercept is a bug, not a
    fallback -- the logvix term is worth ~ -2.1, so deleting it inflates sigma ~8x).

    gate = feature that must be finite for the primary to be usable.  Rows where it is missing are
    scored by the fallback, which is fit on EVERY prior row (it does not lose the pre-2011 history).
    """
    lr, tgt = A["lr"][h], A["tgt"][h]
    fwd_s = np.expm1(lr)
    need_fb = [A["fnames"].index(c) for c in fallback]
    need_pr = [A["fnames"].index(c) for c in primary]
    ok_fb = np.isfinite(A["F"][:, need_fb]).all(axis=1) & np.isfinite(lr)
    ok_pr = np.isfinite(A["F"][:, need_pr]).all(axis=1) & np.isfinite(lr)
    yr, pos, sid = A["yr"], A["pos"], A["sid"]
    nsym = len(A["syms"])
    em = np.ones(len(lr)) if earn is None else np.where(earn, M.EARN_MULT[h], 1.0)
    years = sorted(set(yr[ok_fb]))
    out = {"yr": [], "pos": [], "a": {t: [] for t in thrs},
           "p": {t: [] for t in thrs}, "ps": {t: [] for t in thrs}, "prim": []}
    for y in years[MIN_TRAIN_YEARS:]:
        te = ok_fb & (yr == y)
        tr = ok_fb & (yr < y)
        if te.sum() < 50 or tr.sum() < 5000:
            continue
        use_p = ok_pr & te
        pp = {t: np.empty((int(te.sum()), 4)) for t in thrs}
        sub_p = use_p[te]
        fit_pr = (ok_pr & (yr < y) & np.isfinite(tgt)).sum() >= 500
        if not fit_pr:
            use_p = np.zeros(len(lr), bool)
            sub_p = use_p[te]
        for cols, rows_tr, rows_te, sel in (
                (fallback, tr, te & ~use_p, ~sub_p),
                (primary, ok_pr & (yr < y), use_p, sub_p)):
            if sel.sum() == 0:
                continue
            trv = rows_tr & np.isfinite(tgt)
            if trv.sum() < 500:
                # not enough prior history to fit this branch -> the rows stay with the fallback,
                # which is exactly what the ladder must do in the first VIX9D year (2011).
                continue
            b = ols(design(A, cols, trv), tgt[trv])
            s_tr = np.exp(design(A, cols, rows_tr) @ b) * math.sqrt(h) * em[rows_tr]
            s_te = np.exp(design(A, cols, rows_te) @ b) * math.sqrt(h) * em[rows_te]
            kk = (_wf_kappa(A, cols, h, rows_tr, trv, lr, tgt, yr, y) if kappa == "wf" else kappa)
            zs = np.sort(lr[rows_tr] / s_tr) * kk
            for t in thrs:
                pp[t][sel] = probs_emp(zs, s_te, t)
        for t in thrs:
            out["p"][t].append(pp[t])
            out["ps"][t].append(clim_sym(bidx(fwd_s[tr], t), sid[tr], sid[te], nsym))
            out["a"][t].append(bidx(fwd_s[te], t))
        out["yr"].append(np.full(int(te.sum()), y))
        out["pos"].append(pos[te])
        out["prim"].append(sub_p)
    R = {"yr": np.concatenate(out["yr"]), "pos": np.concatenate(out["pos"]),
         "prim": np.concatenate(out["prim"]),
         "a": {t: np.concatenate(out["a"][t]) for t in thrs},
         "p": {"LADDER": {t: np.vstack(out["p"][t]) for t in thrs},
               "CLIMSYM": {t: np.vstack(out["ps"][t]) for t in thrs}},
         "r2": {"LADDER": np.nan},
         "years": sorted(set(np.concatenate(out["yr"]).tolist()))}
    return R


def stage_ship(B):
    hdr("7. THE SHIPPABLE FORM: fallback ladder  [VIX + VIX9D] -> [VIX]  over ALL 21 test years")
    print("Primary   = rv_d rv_w rv_m rv_q ewma97 logvix log(VIX9D)   (2011-01-03 .. 2026-07-17)")
    print("Fallback  = the shipped index spec, own fit + own z table   (everything else)")
    print("Both re-fit every test year on strictly prior years; kappa re-estimated per branch.")
    F = M.FEATS_OHLC
    for h in HS:
        Rb = run(B["INDEX"], h, {"base": list(F) + ["lv"]}, kappa=KAP["INDEX"])
        Rl = run_ladder(B["INDEX"], h, list(F) + ["lv", "lv9"], list(F) + ["lv"], "lv9",
                        kappa=KAP["INDEX"])
        print(f"\n--- INDEX h={h} ---  primary branch covers "
              f"{Rl['prim'].mean():.1%} of scored rows")
        print(f"{'thr':>5} {'model':<14} {'n':>8} {'BSS2':>8} {'dBSS2':>8} {'ECE':>7} {'dECE':>8} "
              f"{'yrs+':>7}")
        for t in THRS:
            a, pr = Rb["a"][t], Rb["p"]["CLIMSYM"][t]
            pb = Rb["p"]["base"][t]
            # ladder rows are a subset/equal set; align on (pos, sid) via identical construction
            al, prl = Rl["a"][t], Rl["p"]["CLIMSYM"][t]
            pl = Rl["p"]["LADDER"][t]
            assert len(a) == len(al)
            vb, vl = bss2(pb, a, pr), bss2(pl, al, prl)
            eb, el = ece2(pb, a), ece2(pl, al)
            w = n = 0
            for y in Rb["years"]:
                m = Rb["yr"] == y
                if m.sum() < 30:
                    continue
                d = bss2(pl[m], al[m], prl[m]) - bss2(pb[m], a[m], pr[m])
                w += d > 0
                n += 1
            print(f"{t:>5.0%} {'baseline':<14} {len(a):>8,} {vb:>8.4f} {'--':>8} {eb:>7.4f} "
                  f"{'--':>8} {'--':>7}")
            print(f"{t:>5.0%} {'ladder':<14} {len(al):>8,} {vl:>8.4f} {vl-vb:>+8.4f} {el:>7.4f} "
                  f"{el-eb:>+8.4f} {w:>3}/{n:<3}")
            # restrict to the primary-covered rows: the honest 'where it can act' number
            m = Rl["prim"]
            print(f"      {'  on 2011+ rows':<14} {int(m.sum()):>8,} "
                  f"{bss2(pl[m], al[m], prl[m]):>8.4f} "
                  f"{bss2(pl[m], al[m], prl[m]) - bss2(pb[m], a[m], pr[m]):>+8.4f} "
                  f"{ece2(pl[m], al[m]):>7.4f} {ece2(pl[m], al[m]) - ece2(pb[m], a[m]):>+8.4f}")
        # per-year deltas at 2%
        t = 0.02
        a, pr, pb = Rb["a"][t], Rb["p"]["CLIMSYM"][t], Rb["p"]["base"][t]
        pl = Rl["p"]["LADDER"][t]
        line = []
        for y in Rb["years"]:
            m = Rb["yr"] == y
            line.append(f"{y}:{bss2(pl[m], a[m], pr[m]) - bss2(pb[m], a[m], pr[m]):+.3f}")
        print("      per-year dBSS2 @2%: " + " ".join(line))


# ============================================================ stage: does VVIX move the tails?
def stage_tails(B):
    hdr("8. DOES VVIX (or the slope) ACTUALLY FORECAST FATTER STANDARDISED TAILS?\n"
        "   OUT-OF-SAMPLE z = log fwd return / sigma_hat (baseline fit on strictly prior years),\n"
        "   bucketed by the state variable using TRAIN quantiles only.")
    F = M.FEATS_OHLC
    for g in ("INDEX", "SINGLE"):
        A = B[g]
        cols = BASE[g]
        for h in (1, 5, 21):
            lr, tgt, yr = A["lr"][h], A["tgt"][h], A["yr"]
            for sv in ("lvvix", "ts_30_3m"):
                fc = A["F"][:, A["fnames"].index(sv)]
                need = [A["fnames"].index(c) for c in cols]
                ok = np.isfinite(A["F"][:, need]).all(axis=1) & np.isfinite(lr) & np.isfinite(fc)
                years = sorted(set(yr[ok]))
                Z, BK = [], []
                for y in years[MIN_TRAIN_YEARS:]:
                    te, tr = ok & (yr == y), ok & (yr < y)
                    if te.sum() < 50 or tr.sum() < 5000:
                        continue
                    trv = tr & np.isfinite(tgt)
                    b = ols(design(A, cols, trv), tgt[trv])
                    s = np.exp(design(A, cols, te) @ b) * math.sqrt(h)
                    ed = np.percentile(fc[tr], [33.333, 66.667])
                    Z.append(lr[te] / s)
                    BK.append(np.digitize(fc[te], ed))
                z, bk = np.concatenate(Z), np.concatenate(BK)
                print(f"  {g:<7} h={h:<3} state={sv:<9} n={len(z):>9,}  " + "  ".join(
                    f"T{q+1}: sd {z[bk==q].std():.2f} P(|z|>2) {np.mean(np.abs(z[bk==q])>2):.4f} "
                    f"P(|z|>3) {np.mean(np.abs(z[bk==q])>3):.4f}" for q in range(3)))
            print()


STAGES.update({"ctrl": stage_ctrl, "ship": stage_ship, "tails": stage_tails})




# ============================================================ stage: ship2 = missing-indicator form
def add_imputed(B):
    """ts9_i = log(VIX9D/VIX) where observed, 0 where not, PLUS a miss9 dummy.

    Why this and not the two-branch ladder: the ladder's primary branch can only be trained on
    2011+ rows, so it throws away the 2001-2010 history -- including 2008 -- from BOTH the
    coefficients and the empirical z table.  Measured cost of that: index ECE gets WORSE
    (h=5, 2011+ rows: 0.0164 -> 0.0216).  The missing-indicator form keeps one pooled fit and one
    pooled full-history z table; the slope coefficient is identified only off the rows where
    ^VIX9D exists, and on a row where it does not the model collapses to the shipped baseline
    shifted by one fitted constant.  That is graceful degradation with no fabricated data: the
    zero is never used as a value, only as a placeholder that the dummy absorbs."""
    if "ts9_i" in B["INDEX"]["fnames"]:
        return
    for g in ("INDEX", "SINGLE"):
        G = B[g]
        v = G["F"][:, G["fnames"].index("ts_9_30")]
        miss = ~np.isfinite(v)
        G["F"] = np.hstack([G["F"], np.where(miss, 0.0, v)[:, None],
                            miss.astype(float)[:, None]])
        G["fnames"] = G["fnames"] + ["ts9_i", "miss9"]


def stage_ship2(B):
    add_imputed(B)
    hdr("9. SHIPPABLE FORM v2 -- missing-indicator short-end slope, FULL 21 test years, "
        "full-history z table")
    F = M.FEATS_OHLC
    specs = {"base(VIX)": list(F) + ["lv"],
             "+ts9_i+miss9": list(F) + ["lv", "ts9_i", "miss9"]}
    for h in HS:
        print(f"\n--- INDEX h={h} ---")
        R = run(B["INDEX"], h, specs, kappa=KAP["INDEX"],
                gate={"+ts9_i+miss9": ("ts_9_30", 1000, list(F) + ["lv"])})
        show(R, list(specs), "base(VIX)", boot_for=("+ts9_i+miss9",))
        t = 0.02
        a, pr = R["a"][t], R["p"]["CLIMSYM"][t]
        pb, pn = R["p"]["base(VIX)"][t], R["p"]["+ts9_i+miss9"][t]
        mi = B["INDEX"]["F"][:, B["INDEX"]["fnames"].index("miss9")]
        # restrict to the rows where VIX9D actually exists
        idx = B["index"]
        obs = np.isin(R["pos"], np.flatnonzero(~np.isnan(B["V"]["^VIX9D"].reindex(idx).to_numpy())))
        print(f"      on rows WITH ^VIX9D (n={int(obs.sum()):,}): "
              f"BSS2 {bss2(pb[obs], a[obs], pr[obs]):.4f} -> {bss2(pn[obs], a[obs], pr[obs]):.4f} "
              f"({bss2(pn[obs], a[obs], pr[obs]) - bss2(pb[obs], a[obs], pr[obs]):+.4f})  "
              f"ECE {ece2(pb[obs], a[obs]):.4f} -> {ece2(pn[obs], a[obs]):.4f}")
        print(f"      on rows WITHOUT  (n={int((~obs).sum()):,}): "
              f"BSS2 {bss2(pb[~obs], a[~obs], pr[~obs]):.4f} -> "
              f"{bss2(pn[~obs], a[~obs], pr[~obs]):.4f} "
              f"({bss2(pn[~obs], a[~obs], pr[~obs]) - bss2(pb[~obs], a[~obs], pr[~obs]):+.4f})")
        line = []
        for y in R["years"]:
            m = R["yr"] == y
            line.append(f"{y}:{bss2(pn[m], a[m], pr[m]) - bss2(pb[m], a[m], pr[m]):+.3f}")
        print("      per-year dBSS2 @2%: " + " ".join(line))


def stage_ship2s(B):
    add_imputed(B)
    hdr("9b. SAME missing-indicator form on SINGLE NAMES (shipped single spec has no VIX at all, "
        "so this adds logvix AND the short-end slope)")
    A = B["SINGLE"]
    F = M.FEATS_OHLC
    specs = {"base(shipped)": list(F),
             "+VIX": list(F) + ["lv"],
             "+VIX+ts9_i+miss9": list(F) + ["lv", "ts9_i", "miss9"]}
    for h in HS:
        print(f"\n--- SINGLE h={h} ---")
        R = run(A, h, specs, earn=A["earn"][h],
                gate={"+VIX+ts9_i+miss9": ("ts_9_30", 20000, list(F) + ["lv"])})
        show(R, list(specs), "base(shipped)", boot_for=("+VIX", "+VIX+ts9_i+miss9"))
        for t in THRS:
            pt, lo, hi, pg, nd = boot_delta(R, "+VIX+ts9_i+miss9", "+VIX", t)
            w, ny, _ = year_wins(R, "+VIX+ts9_i+miss9", "+VIX", t)
            print(f"      incremental over +VIX @thr={t:.0%}: {pt:+.4f} "
                  f"95%CI[{lo:+.4f},{hi:+.4f}] P(>0)={pg:.3f} yrs+ {w}/{ny}")


STAGES.update({"ship2": stage_ship2, "ship2s": stage_ship2s})




# ============================================================ stage: final verification
def stage_verify(B):
    add_imputed(B)
    hdr("10. VERIFICATION of the only surviving candidate: INDEX + short-end slope (ts9_i+miss9)")
    F = M.FEATS_OHLC
    specs = {"base": list(F) + ["lv"], "cand": list(F) + ["lv", "ts9_i", "miss9"]}
    gate = {"cand": ("ts_9_30", 1000, list(F) + ["lv"])}
    for h in HS:
        R = run(B["INDEX"], h, specs, kappa=KAP["INDEX"], gate=gate)
        obs = np.isin(R["pos"], np.flatnonzero(
            ~np.isnan(B["V"]["^VIX9D"].reindex(B["index"]).to_numpy())))
        print(f"\n--- INDEX h={h} ---")
        for t in THRS:
            a, pr = R["a"][t], R["p"]["CLIMSYM"][t]
            pb, pc = R["p"]["base"][t], R["p"]["cand"][t]
            for nm, m in (("all 21y", np.ones(len(a), bool)),
                          ("VIX9D rows", obs),
                          ("ex-2020", obs & (R["yr"] != 2020)),
                          ("2021+", obs & (R["yr"] >= 2021)),
                          ("non-overlap", obs & np.isin(R["pos"], np.unique(R["pos"])[::max(h, 1)]))):
                if m.sum() < 500:
                    continue
                d = bss2(pc[m], a[m], pr[m]) - bss2(pb[m], a[m], pr[m])
                print(f"  thr={t:.0%} {nm:<12} n={int(m.sum()):>7,} "
                      f"BSS2 {bss2(pb[m], a[m], pr[m]):+.4f} -> {bss2(pc[m], a[m], pr[m]):+.4f} "
                      f"({d:+.4f})  ECE {ece2(pb[m], a[m]):.4f} -> {ece2(pc[m], a[m]):.4f}")
    hdr("10b. FITTED COEFFICIENT on the short-end slope log(VIX9D/VIX), last walk-forward fit")
    for h in HS:
        A = B["INDEX"]
        lr, tgt, yr = A["lr"][h], A["tgt"][h], A["yr"]
        cols = list(F) + ["lv", "ts9_i", "miss9"]
        need = [A["fnames"].index(c) for c in cols]
        tr = np.isfinite(A["F"][:, need]).all(axis=1) & np.isfinite(lr) & np.isfinite(tgt) \
            & (yr < 2026)
        b = ols(design(A, cols, tr), tgt[tr])
        print(f"  h={h:<3} n={int(tr.sum()):>7,}  " +
              "  ".join(f"{c}={v:+.4f}" for c, v in zip(["const"] + cols, b)))
    hdr("10c. SUPPORT: where does the thr boundary sit on the z scale?  |log(1+thr)|/sigma_h.\n"
        "     This is why a state-dependent TAIL does not move BSS2: at thr=2% the question is\n"
        "     asked in the shoulder of the distribution, not in the tail.")
    for g in ("INDEX", "SINGLE"):
        A = B[g]
        for h in HS:
            lr, tgt, yr = A["lr"][h], A["tgt"][h], A["yr"]
            cols = BASE[g]
            need = [A["fnames"].index(c) for c in cols]
            ok = np.isfinite(A["F"][:, need]).all(axis=1) & np.isfinite(lr) & np.isfinite(tgt)
            tr = ok & (yr < 2026)
            b = ols(design(A, cols, tr), tgt[tr])
            s = np.exp(design(A, cols, ok) @ b) * math.sqrt(h)
            for t in THRS:
                r = math.log(1 + t) / s
                print(f"  {g:<7} h={h:<3} thr={t:.0%}  median |z| boundary = "
                      f"{np.median(r):.2f}  [p10 {np.percentile(r, 10):.2f}, "
                      f"p90 {np.percentile(r, 90):.2f}]")


STAGES.update({"verify": stage_verify})


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "base"
    B = load(rebuild=("--rebuild" in sys.argv))
    for s in (list(STAGES) if which == "all" else which.split(",")):
        STAGES[s](B)
        print(f"\n[{s} done {time.time()-T0:.0f}s]", flush=True)
