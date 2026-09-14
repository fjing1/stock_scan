"""
_move_wf_verify_harvol2.py — round 2 of the adversarial verification (see _move_wf_verify_harvol.py).

Round 1 showed the ewma94 comparison survives purging, crisis-drops, date-clustering,
non-overlapping subsamples and prospective spec selection. Round 2 attacks the parts that
looked soft, plus the pooling issue:

  A. "BEATS EVERY SINGLE-ESTIMATOR CANDIDATE" — at h=1 INDEX the margin over gk5 was +0.0005.
     Test it: date-collapsed Newey-West t on claim-vs-best-single, and WIDEN the single-estimator
     grid with the short windows the original never tried (gk2/gk3/gk4, park3, yz3, rs3, ewma85-91).
     The original grid was arbitrary; if a single estimator outside it wins, that half of the claim
     is grid-dependent.
  B. POOLING / FAKE n — refit EVERY model SEPARATELY PER SYMBOL on that symbol's own prior years,
     so no cross-sectional pooling at all. Report the fraction of the 231 names where the claim
     spec beats ewma94, and a symbol-clustered bootstrap.
  C. CLIP-FLOOR SENSITIVITY — the arbitrary FLOOR=1e-3 puts a point mass on ~20% of index h=1
     rows. Re-run the whole thing at FLOOR = 1e-4 and 3e-3.
  D. h=1 TARGET SPECIFICATION — |log r[t+1]| is a very noisy sigma proxy. Re-score h=1 against
     1-bar Parkinson of day t+1 and Garman-Klass of day t+1 instead.
  E. QLIKE — is the win loss-function-specific? Score the standard variance QLIKE too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _move_lib as L
import _move_wf_volest as V
from _move_wf_verify_harvol import (SINGLES, MULTI, CLAIM, BASE, build, gram, gadd, gzero,
                                    solve, hac_t)

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 80)
HORIZONS = (1, 5, 10, 21)

EXTRA_SINGLES = ["gk2", "gk3", "gk4", "park2", "park3", "yz3", "yz4", "rs3", "cc3",
                 "ewma85", "ewma88", "ewma91", "ewma96", "ewma98"]


def extend(E, o, hi, lo, c):
    for n in (2, 3, 4):
        E[f"gk{n}"] = L.vol_garman_klass(o, hi, lo, c, n)
    for n in (2, 3):
        E[f"park{n}"] = L.vol_parkinson(hi, lo, n)
    for n in (3, 4):
        E[f"yz{n}"] = L.vol_yang_zhang(o, hi, lo, c, n)
    E["rs3"] = L.vol_rogers_satchell(o, hi, lo, c, 3)
    E["cc3"] = L.vol_cc(c, 3)
    for lam in (0.85, 0.88, 0.91, 0.96, 0.98):
        E[f"ewma{int(round(lam*100))}"] = L.vol_ewma(c, lam)
    return E


def wf2(d, allf, specs, floor, ceil, quiet=True, all_years=None):
    """Walk-forward with per-date aggregates + QLIKE, for an arbitrary clip band already applied."""
    fi = {f: j for j, f in enumerate(allf)}
    y, X, di, yr = d["y"], d["X"], d["di"], d["yr"]
    ylist = sorted(set(yr.tolist())) if all_years is None else all_years
    k = len(allf)
    gy = {yy: gram(X, y, yr == yy) for yy in ylist if (yr == yy).sum() > 0}
    cum, run = {}, gzero(k)
    for yy in ylist:
        cum[yy] = run
        if yy in gy:
            run = gadd(run, gy[yy])
    nd = d["T"]
    names = list(specs)
    sse_d = {s: np.zeros(nd) for s in names}
    ql = {s: 0.0 for s in names}
    cnt_d = np.zeros(nd)
    dev2_d = np.zeros(nd)
    n_tot = 0
    used = []
    for yy in ylist[5:]:
        te = yr == yy
        trc = cum[yy]
        if te.sum() < 50 or trc["n"] < 500:
            continue
        used.append(yy)
        mu = trc["sy"] / trc["n"]
        yt, Xt, dt = y[te], X[te], di[te]
        np.add.at(dev2_d, dt, (yt - mu) ** 2)
        np.add.at(cnt_d, dt, 1.0)
        n_tot += int(te.sum())
        v_a = np.clip(np.exp(yt), floor, ceil) ** 2
        for s, fl in specs.items():
            idx = [fi[f] for f in fl]
            beta = solve(trc, idx)
            pred = beta[0] + Xt[:, idx] @ beta[1:]
            np.add.at(sse_d[s], dt, (yt - pred) ** 2)
            v_f = np.clip(np.exp(pred), floor, ceil) ** 2
            ql[s] += float((v_a / v_f - np.log(v_a / v_f) - 1.0).sum())
    return dict(sse_d=sse_d, cnt_d=cnt_d, dev2_d=dev2_d, n=n_tot, years=used,
                qlike={s: ql[s] / n_tot for s in names})


def r2_(res, s, dm=None):
    m = res["cnt_d"] > 0
    if dm is not None:
        m = m & dm
    return 1.0 - res["sse_d"][s][m].sum() / res["dev2_d"][m].sum()


# ------------------------------------------------------------------ per-symbol (no pooling)
def per_symbol(E, allf, tgt, cols, years, specs, floor, ceil, min_train=400, seed=3):
    fi = {f: j for j, f in enumerate(allf)}
    ylist = sorted(set(years.tolist()))
    rows = []
    names = list(specs)
    tot = {s: 0.0 for s in names}
    sst_tot = 0.0
    n_tot = 0
    per_sym = {}
    for sym in cols:
        tv = tgt[sym].values.astype(np.float64)
        y = np.log(np.clip(tv, floor, ceil))
        X = np.column_stack([np.log(np.clip(E[f][sym].values.astype(np.float64), floor, ceil))
                             for f in allf])
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        y, X, yr = y[ok], X[ok], years[ok]
        if len(y) < min_train + 100:
            continue
        k = len(allf)
        gy = {yy: gram(X, y, yr == yy) for yy in ylist if (yr == yy).sum() > 0}
        cum, run = {}, gzero(k)
        for yy in ylist:
            cum[yy] = run
            if yy in gy:
                run = gadd(run, gy[yy])
        sse = {s: 0.0 for s in names}
        sst = 0.0
        n = 0
        for yy in ylist[5:]:
            te = yr == yy
            trc = cum[yy]
            if te.sum() < 50 or trc["n"] < min_train:
                continue
            mu = trc["sy"] / trc["n"]
            yt, Xt = y[te], X[te]
            sst += float(((yt - mu) ** 2).sum())
            n += int(te.sum())
            for s, fl in specs.items():
                idx = [fi[f] for f in fl]
                beta = solve(trc, idx)
                sse[s] += float(((yt - (beta[0] + Xt[:, idx] @ beta[1:])) ** 2).sum())
        if n < 500 or sst <= 0:
            continue
        per_sym[sym] = {s: 1 - sse[s] / sst for s in names}
        per_sym[sym]["n"] = n
        for s in names:
            tot[s] += sse[s]
        sst_tot += sst
        n_tot += n
    P = pd.DataFrame(per_sym).T
    pooled = {s: 1 - tot[s] / sst_tot for s in names}
    # symbol-clustered bootstrap of the pooled gap
    rng = np.random.default_rng(seed)
    syms = list(per_sym)
    gaps = np.empty(2000)
    gv = (P[CLAIM] - P[BASE]).values
    for b in range(2000):
        gaps[b] = gv[rng.integers(0, len(syms), len(syms))].mean()
    return P, pooled, n_tot, gaps


def main():
    o, hi, lo, c = V.load_panel()
    idx_cols = [s for s in V.IDX if s in c.columns]
    sng_cols = [s for s in c.columns if s not in V.NOT_SINGLE]
    E = extend(V.build_estimators(o, hi, lo, c), o, hi, lo, c)
    allf = sorted(E)
    years = c.index.year.values
    ylist_full = sorted(set(years.tolist()))
    dy = years

    SP = {s: [s] for s in SINGLES + EXTRA_SINGLES}
    SP.update({k: v for k, v in MULTI.items() if k in (CLAIM, "har4+yz21", "har3", "HARc4")})

    def target(h, kind="std"):
        if h != 1:
            return L.realized_vol_forward(c, h)
        if kind == "std":
            return np.log(c).diff().abs().shift(-1)
        if kind == "park":
            return L.vol_parkinson(hi, lo, 1).shift(-1)
        if kind == "gk":
            return L.vol_garman_klass(o, hi, lo, c, 1).shift(-1)

    # =============================================================== A. widened single grid
    print("=" * 130)
    print("A.  'BEATS EVERY SINGLE-ESTIMATOR CANDIDATE' — original grid had 21 singles and the")
    print("    h=1 INDEX margin was +0.0005 over gk5. Widen the grid with 14 short-window /")
    print("    extra-lambda singles the original never tried, and t-test the winner's margin.")
    store = {}
    rows = []
    for h in HORIZONS:
        tgt = target(h)
        for uni, cols in (("INDEX", idx_cols), ("SINGLE", sng_cols), ("SPY", ["SPY"])):
            d = build(E, allf, tgt, cols, years)
            r = wf2(d, allf, SP, V.FLOOR, V.CEIL, all_years=ylist_full)
            store[(h, uni)] = (r, d)
            orig = {s: r2_(r, s) for s in SINGLES}
            wide = {s: r2_(r, s) for s in SINGLES + EXTRA_SINGLES}
            bo, bw = max(orig, key=orig.get), max(wide, key=wide.get)
            cl = r2_(r, CLAIM)
            _, t_o, nd_, _ = hac_t(r, CLAIM, bo, h)
            _, t_w, _, _ = hac_t(r, CLAIM, bw, h)
            rows.append({"h": h, "uni": uni, "n": r["n"], "claim": cl,
                         "best_orig_grid": bo, "r2": orig[bo], "gap": cl - orig[bo], "t": t_o,
                         "best_WIDE_grid": bw, "r2_w": wide[bw], "gap_w": cl - wide[bw], "t_w": t_w,
                         "beats": cl > wide[bw]})
    A = pd.DataFrame(rows)
    print(A.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    print("\n    top-6 singles in the WIDE grid, per cell:")
    for (h, uni), (r, _) in store.items():
        w = pd.Series({s: r2_(r, s) for s in SINGLES + EXTRA_SINGLES}).sort_values(ascending=False)
        print(f"      h={h:2d} {uni:6s} claim={r2_(r, CLAIM):.4f} | " +
              "  ".join(f"{k}={v:.4f}" for k, v in w.head(6).items()))

    # =============================================================== E. QLIKE
    print("\n" + "=" * 130)
    print("E.  QLIKE on variance (lower is better) — is the win loss-function specific?")
    rows = []
    for (h, uni), (r, _) in store.items():
        q = r["qlike"]
        best_single = min((s for s in SINGLES + EXTRA_SINGLES), key=lambda s: q[s])
        rows.append({"h": h, "uni": uni, "qlike_claim": q[CLAIM], "qlike_ewma94": q[BASE],
                     "claim_better_than_ewma94": q[CLAIM] < q[BASE],
                     "best_single": best_single, "qlike_best_single": q[best_single],
                     "claim_better_than_all_singles": q[CLAIM] < q[best_single],
                     "qlike_HARc4": q["HARc4"], "qlike_har3": q["har3"]})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:9.5f}"))

    # =============================================================== C. clip-floor sensitivity
    print("\n" + "=" * 130)
    print("C.  CLIP-FLOOR SENSITIVITY. FLOOR=1e-3 is arbitrary and clips ~20% of index h=1 rows.")
    rows = []
    for floor in (1e-4, 1e-3, 3e-3):
        for h in HORIZONS:
            tgt = target(h)
            for uni, cols in (("INDEX", idx_cols), ("SINGLE", sng_cols)):
                d = build_floor(E, allf, tgt, cols, years, floor, V.CEIL)
                r = wf2(d, allf, {CLAIM: MULTI[CLAIM], BASE: [BASE], "gk5": ["gk5"],
                                  "har3": MULTI["har3"]}, floor, V.CEIL, all_years=ylist_full)
                rows.append({"floor": floor, "h": h, "uni": uni, "n": r["n"],
                             "claim": r2_(r, CLAIM), "ewma94": r2_(r, BASE), "gk5": r2_(r, "gk5"),
                             "gap_vs_ewma94": r2_(r, CLAIM) - r2_(r, BASE),
                             "gap_vs_gk5": r2_(r, CLAIM) - r2_(r, "gk5"),
                             "t_vs_ewma94": hac_t(r, CLAIM, BASE, h)[1],
                             "t_vs_gk5": hac_t(r, CLAIM, "gk5", h)[1]})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    # =============================================================== D. h=1 target spec
    print("\n" + "=" * 130)
    print("D.  h=1 TARGET SPECIFICATION. |log r[t+1]| is a pi^2/8-variance-noise proxy. Re-score")
    print("    h=1 against 1-bar Parkinson and 1-bar Garman-Klass of day t+1.")
    rows = []
    for kind in ("std", "park", "gk"):
        tgt = target(1, kind)
        for uni, cols in (("INDEX", idx_cols), ("SINGLE", sng_cols), ("SPY", ["SPY"])):
            d = build(E, allf, tgt, cols, years)
            r = wf2(d, allf, SP, V.FLOOR, V.CEIL, all_years=ylist_full)
            wide = {s: r2_(r, s) for s in SINGLES + EXTRA_SINGLES}
            bw = max(wide, key=wide.get)
            rows.append({"target": kind, "uni": uni, "n": r["n"], "claim": r2_(r, CLAIM),
                         "ewma94": r2_(r, BASE), "gap_vs_ewma94": r2_(r, CLAIM) - r2_(r, BASE),
                         "t_vs_ewma94": hac_t(r, CLAIM, BASE, 1)[1],
                         "best_wide_single": bw, "r2_bws": wide[bw],
                         "gap_vs_bws": r2_(r, CLAIM) - wide[bw],
                         "t_vs_bws": hac_t(r, CLAIM, bw, 1)[1]})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    # =============================================================== B. per-symbol, no pooling
    print("\n" + "=" * 130)
    print("B.  NO POOLING AT ALL: every spec refit separately on EACH symbol's own prior years")
    print("    (>=400 train rows, >=500 scored rows). Kills the 231-correlated-stocks issue.")
    sp2 = {CLAIM: MULTI[CLAIM], BASE: [BASE], "gk5": ["gk5"], "har3": MULTI["har3"],
           "HARc4": MULTI["HARc4"], "ewma97": ["ewma97"], "park21": ["park21"]}
    for h in HORIZONS:
        tgt = target(h)
        P, pooled, nn, gaps = per_symbol(E, allf, tgt, sng_cols, years, sp2, V.FLOOR, V.CEIL)
        g = P[CLAIM] - P[BASE]
        gs = P[CLAIM] - P["gk5"]
        gh = P[CLAIM] - P["har3"]
        print(f"  h={h:2d} SINGLE per-symbol fits: {len(P)} symbols, n={nn:,} scored rows")
        print(f"     pooled-SSE R2: claim={pooled[CLAIM]:.4f} ewma94={pooled[BASE]:.4f} "
              f"gk5={pooled['gk5']:.4f} har3={pooled['har3']:.4f} HARc4={pooled['HARc4']:.4f} "
              f"ewma97={pooled['ewma97']:.4f} park21={pooled['park21']:.4f}")
        print(f"     claim beats ewma94 in {int((g>0).sum())}/{len(P)} symbols  "
              f"mean gap {g.mean():+.4f} median {g.median():+.4f}  "
              f"symbol-clustered boot 95% [{np.percentile(gaps,2.5):+.4f},"
              f"{np.percentile(gaps,97.5):+.4f}]")
        print(f"     claim beats gk5   in {int((gs>0).sum())}/{len(P)} symbols  mean {gs.mean():+.4f}"
              f" | claim beats har3 in {int((gh>0).sum())}/{len(P)} symbols mean {gh.mean():+.4f}")
    for h in (1, 21):
        tgt = target(h)
        P, pooled, nn, gaps = per_symbol(E, allf, tgt, idx_cols, years, sp2, V.FLOOR, V.CEIL)
        print(f"  h={h:2d} INDEX  per-symbol fits ({list(P.index)}):")
        print(P.to_string(float_format=lambda v: f"{v:8.4f}"))


def build_floor(E, allf, tgt, cols, years, floor, ceil):
    T, S = len(tgt.index), len(cols)
    tv = tgt[cols].values.astype(np.float64)
    y = np.log(np.clip(tv, floor, ceil)).ravel()
    X = np.empty((T * S, len(allf)), dtype=np.float64)
    for j, f in enumerate(allf):
        X[:, j] = np.log(np.clip(E[f][cols].values.astype(np.float64), floor, ceil)).ravel()
    di = np.repeat(np.arange(T), S).astype(np.int32)
    yr = np.repeat(years, S)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    return dict(y=y[ok], X=X[ok], di=di[ok], yr=yr[ok], T=T,
                tclip=np.zeros(int(ok.sum()), bool), fclip=np.zeros(int(ok.sum()), bool))


if __name__ == "__main__":
    main()
