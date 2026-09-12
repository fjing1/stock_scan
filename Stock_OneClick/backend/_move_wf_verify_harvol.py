"""
_move_wf_verify_harvol.py — ADVERSARIAL verification of the "HAR beats everything" vol-forecast claim.

CLAIM UNDER TEST
  "A log-space HAR regression on 1-bar-Parkinson realized vol (d/w/m/q) plus Yang-Zhang(21) and
   EWMA(0.97) beats every single-estimator candidate and beats EWMA(0.94) at every horizon
   h in {1,5,10,21}, out of sample, in both the index and single-name universes."

Attacks implemented (each prints its own verdict block):
  T0  reproduce the headline r2_adj numbers with an independent implementation
  T1  "beats EVERY single-estimator candidate" — full ranking, all 20 singles, every cell
  T2  SIMPLER EXPLANATION — boring controls with the SAME number of free parameters and NO range
      data at all (multi-window close-to-close regression, multi-lambda EWMA regression).
      If those match the claim spec, the range/Parkinson/YZ ingredients are decorative.
  T3  FAKE SAMPLE SIZE — collapse to one loss differential per DATE (kills cross-sectional
      correlation), then Newey-West HAC(2h) t-test; plus a strictly NON-OVERLAPPING every-h-th-date
      subsample scored iid.
  T4  FRAGILITY — per-year gaps; drop 2008/2009/2020 from scoring; drop them from TRAINING too;
      INDEX broken out per series (SPY and ^GSPC are near-duplicates, so n=20,512 is fake).
  T5  LOOKAHEAD — purged walk-forward. In the original, a training row t in late December has a
      target spanning r[t+1..t+h], i.e. into the TEST year. Purge those rows and re-run.
  T6  CLIPPING ARTIFACT — the FLOOR=1e-3 clip puts a point mass at log(1e-3) on both the target
      and the 1-bar-Parkinson feature for halted/illiquid bars. Score only unclipped rows.
  T7  SPEC-SELECTION LOOKAHEAD — the winning spec was chosen by reading the full 2006-2026 OOS
      table. Re-do with prospective (nested) selection using only prior OOS years.
  T8  SUBGROUP CHECK (Simpson) — R2 gap and forecast bias inside prior-vol quintiles.

Sample discipline: every candidate is scored on the IDENTICAL rows (all 30 estimators finite),
exactly as the original did, so nothing here is a coverage artifact.
Run: ../../vcp_env/bin/python _move_wf_verify_harvol.py
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

import _move_lib as L
import _move_wf_volest as V

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)

HORIZONS = (1, 5, 10, 21)
CLAIM = "har4+yz21+ewma97"
BASE = "ewma94"

# ---- specs -----------------------------------------------------------------------------------
SINGLES = ["cc5", "cc10", "cc21", "cc63", "cc126", "ewma90", "ewma94", "ewma97",
           "park5", "park10", "park21", "gk5", "gk10", "gk21",
           "rs5", "rs10", "rs21", "yz5", "yz10", "yz21", "lr"]

MULTI = {
    CLAIM:              ["har_d", "har_w", "har_m", "har_q", "yz21", "ewma97"],
    "har4+yz21":         ["har_d", "har_w", "har_m", "har_q", "yz21"],
    "har4":              ["har_d", "har_w", "har_m", "har_q"],
    "har3":              ["har_d", "har_w", "har_m"],
    "HARc4":             ["harc_d", "harc_w", "harc_m", "harc_q"],
    "HARc3":             ["harc_d", "harc_w", "harc_m"],
    # ---- CONTROLS: no range data at all, same order of parameter count -----------------------
    "CTL_cc4win":        ["cc5", "cc21", "cc63", "cc126"],          # 5 params, close-only
    "CTL_cc4win+ewma94": ["cc5", "cc21", "cc63", "cc126", "ewma94"],  # 6 params, close-only
    "CTL_ewma3":         ["ewma90", "ewma94", "ewma97"],            # 4 params, close-only
    "CTL_ewma94+cc126":  ["ewma94", "cc126"],                        # 3 params, the dumbest combo
    "CTL_cc21+cc126":    ["cc21", "cc126"],                          # 3 params
    # ---- does the range data add anything ON TOP of the close-only control? ------------------
    "CTL_cc4win+yz21":   ["cc5", "cc21", "cc63", "cc126", "yz21"],
}
SPECS = {s: [s] for s in SINGLES}
SPECS.update(MULTI)


# ---- data ------------------------------------------------------------------------------------
def build(E, allf, tgt, cols, years):
    T, S = len(tgt.index), len(cols)
    tv = tgt[cols].values.astype(np.float64)
    tclip = ((tv < V.FLOOR) | (tv > V.CEIL)).ravel()
    y = np.log(np.clip(tv, V.FLOOR, V.CEIL)).ravel()
    X = np.empty((T * S, len(allf)), dtype=np.float64)
    fclip = np.zeros(T * S, dtype=bool)
    for j, f in enumerate(allf):
        v = E[f][cols].values.astype(np.float64)
        fclip |= ((v < V.FLOOR) | (v > V.CEIL)).ravel()
        X[:, j] = np.log(np.clip(v, V.FLOOR, V.CEIL)).ravel()
    di = np.repeat(np.arange(T), S).astype(np.int32)
    yr = np.repeat(years, S)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    return dict(y=y[ok], X=X[ok], di=di[ok], yr=yr[ok],
                tclip=tclip[ok], fclip=fclip[ok], T=T)


# ---- OLS via cumulative Gram matrices (exact, and lets us add/remove year blocks cheaply) ----
def gram(X, y, mask):
    Xm, ym = X[mask], y[mask]
    return dict(n=float(len(ym)), sx=Xm.sum(0), sy=float(ym.sum()),
                G=Xm.T @ Xm, b=Xm.T @ ym, sy2=float(ym @ ym))


def gadd(a, b, sign=1.0):
    return dict(n=a["n"] + sign * b["n"], sx=a["sx"] + sign * b["sx"], sy=a["sy"] + sign * b["sy"],
                G=a["G"] + sign * b["G"], b=a["b"] + sign * b["b"], sy2=a["sy2"] + sign * b["sy2"])


def gzero(k):
    return dict(n=0.0, sx=np.zeros(k), sy=0.0, G=np.zeros((k, k)), b=np.zeros(k), sy2=0.0)


def solve(cum, idx):
    k = len(idx)
    M = np.empty((k + 1, k + 1))
    v = np.empty(k + 1)
    M[0, 0] = cum["n"]
    M[0, 1:] = cum["sx"][idx]
    M[1:, 0] = cum["sx"][idx]
    M[1:, 1:] = cum["G"][np.ix_(idx, idx)]
    v[0] = cum["sy"]
    v[1:] = cum["b"][idx]
    return np.linalg.lstsq(M, v, rcond=None)[0]


# ---- one walk-forward pass, aggregating everything we will ever need -------------------------
def wf(d, allf, specs, h, purge=0, drop_train_years=(), date_year=None, quint_feat="cc21"):
    """Returns per-date and per-quintile aggregates of squared error for every spec."""
    fi = {f: j for j, f in enumerate(allf)}
    y, X, di, yr = d["y"], d["X"], d["di"], d["yr"]
    years = sorted(set(yr.tolist()))
    k = len(allf)

    # per-year grams, plus grams of the purge tail (last `purge` trading dates of each year)
    gy, gtail = {}, {}
    for yy in years:
        m = yr == yy
        gy[yy] = gram(X, y, m)
        if purge:
            dmax = di[m].max()
            gtail[yy] = gram(X, y, m & (di > dmax - purge))
    cum = {}
    run = gzero(k)
    for yy in years:
        cum[yy] = run                       # strictly prior years
        run = gadd(run, gy[yy])

    nd = d["T"]
    names = list(specs)
    sse_d = {s: np.zeros(nd) for s in names}
    cnt_d = np.zeros(nd)
    dev2_d = np.zeros(nd)
    # quintile bins from the TRAIN distribution of the chosen feature (causal)
    qj = fi[quint_feat]
    nq = 5
    sse_q = {s: np.zeros(nq) for s in names}
    res_q = {s: np.zeros(nq) for s in names}
    dev2_q = np.zeros(nq)
    cnt_q = np.zeros(nq)
    n_tot = 0
    used_years = []
    # unclipped-only aggregates
    sse_u = {s: 0.0 for s in names}
    dev2_u, cnt_u = 0.0, 0

    for yy in years[5:]:
        te = yr == yy
        trc = cum[yy]
        if purge and (yy - 1) in gtail:
            trc = gadd(trc, gtail[yy - 1], -1.0)
        for dy in drop_train_years:
            if dy < yy:
                trc = gadd(trc, gy[dy], -1.0)
        if te.sum() < 50 or trc["n"] < 500:
            continue
        used_years.append(yy)
        mu = trc["sy"] / trc["n"]
        yt = y[te]
        Xt = X[te]
        dt = di[te]
        dev = yt - mu
        np.add.at(dev2_d, dt, dev ** 2)
        np.add.at(cnt_d, dt, 1.0)
        n_tot += int(te.sum())
        # quintile edges from train rows (prior years only)
        tr_mask = yr < yy
        edges = np.quantile(X[tr_mask, qj], [0.2, 0.4, 0.6, 0.8])
        qb = np.digitize(Xt[:, qj], edges)
        np.add.at(dev2_q, qb, dev ** 2)
        np.add.at(cnt_q, qb, 1.0)
        unc = ~(d["tclip"][te] | d["fclip"][te])
        dev2_u += float((dev[unc] ** 2).sum())
        cnt_u += int(unc.sum())
        for s, fl in specs.items():
            idx = [fi[f] for f in fl]
            beta = solve(trc, idx)
            pred = beta[0] + Xt[:, idx] @ beta[1:]
            e = (yt - pred) ** 2
            np.add.at(sse_d[s], dt, e)
            np.add.at(sse_q[s], qb, e)
            np.add.at(res_q[s], qb, yt - pred)
            sse_u[s] += float(e[unc].sum())
    return dict(sse_d=sse_d, cnt_d=cnt_d, dev2_d=dev2_d, n=n_tot, years=used_years,
                sse_q=sse_q, res_q=res_q, dev2_q=dev2_q, cnt_q=cnt_q,
                sse_u=sse_u, dev2_u=dev2_u, cnt_u=cnt_u)


def r2_from(res, spec, dmask=None):
    m = np.ones(len(res["cnt_d"]), bool) if dmask is None else dmask
    m = m & (res["cnt_d"] > 0)
    sst = res["dev2_d"][m].sum()
    return 1.0 - res["sse_d"][spec][m].sum() / sst if sst else np.nan


def n_from(res, dmask=None):
    m = np.ones(len(res["cnt_d"]), bool) if dmask is None else dmask
    return int(res["cnt_d"][m & (res["cnt_d"] > 0)].sum())


# ---- inference: date-collapsed Newey-West on the loss differential ---------------------------
def hac_t(res, a, b, h, dmask=None):
    m = (res["cnt_d"] > 0)
    if dmask is not None:
        m = m & dmask
    dl = (res["sse_d"][a][m] - res["sse_d"][b][m]) / res["cnt_d"][m]   # per-date mean loss diff
    n = len(dl)
    if n < 30:
        return np.nan, np.nan, n, np.nan
    x = dl - dl.mean()
    lag = max(1, 2 * h)
    g0 = float(x @ x) / n
    s = g0
    for j in range(1, min(lag, n - 1) + 1):
        gj = float(x[j:] @ x[:-j]) / n
        s += 2.0 * (1 - j / (lag + 1)) * gj
    se = np.sqrt(max(s, 1e-30) / n)
    return float(dl.mean()), float(dl.mean() / se), n, float(se)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    o, hi, lo, c = V.load_panel()
    idx_cols = [s for s in V.IDX if s in c.columns]
    sng_cols = [s for s in c.columns if s not in V.NOT_SINGLE]
    E = V.build_estimators(o, hi, lo, c)
    allf = sorted(E)
    years = c.index.year.values
    dyear = pd.Series(years, index=np.arange(len(c)))

    # sanity: Gram-based OLS == the original lstsq OLS
    tgt5 = L.realized_vol_forward(c, 5)
    d = build(E, allf, tgt5, idx_cols, years)
    m = d["yr"] < 2010
    g = gram(d["X"], d["y"], m)
    fi = {f: j for j, f in enumerate(allf)}
    ii = [fi[f] for f in MULTI[CLAIM]]
    b1 = solve(g, ii)
    b2 = V.ols(d["X"][m][:, ii], d["y"][m])
    print(f"[sanity] Gram-OLS vs lstsq-OLS max abs coef diff = {np.abs(b1 - b2).max():.2e}  "
          f"(must be ~0)")
    del d

    UNI = [("INDEX", idx_cols), ("SINGLE", sng_cols), ("SPY", ["SPY"]),
           ("IWM", ["IWM"]), ("QQQ", ["QQQ"])]

    store = {}
    for h in HORIZONS:
        tgt = (np.log(c).diff().abs().shift(-1) if h == 1 else L.realized_vol_forward(c, h))
        for uni, cols in UNI:
            d = build(E, allf, tgt, cols, years)
            res = wf(d, allf, SPECS, h)
            store[(h, uni)] = res
            store[(h, uni, "d")] = d if uni in ("INDEX", "SINGLE") else None
            print(f"  built h={h:2d} {uni:6s} n={res['n']:>9,} dates={int((res['cnt_d']>0).sum())} "
                  f"years {res['years'][0]}-{res['years'][-1]}")
            if uni not in ("INDEX", "SINGLE"):
                del d

    # ======================================================================== T0 reproduce
    print("\n" + "=" * 118)
    print("T0  REPRODUCTION of the claimed headline numbers (OOS R2 on log vol, walk-forward by year)")
    print("    claimed: h=1 INDEX .202/.161  SINGLE .197/.174 | h=5 .488/.416 .507/.471 |"
          " h=10 .559/.485 .618/.578 | h=21 .530/.478 .673/.632")
    rows = []
    for h in HORIZONS:
        for uni in ("INDEX", "SINGLE"):
            r = store[(h, uni)]
            rows.append({"h": h, "uni": uni, "n": r["n"],
                         CLAIM: r2_from(r, CLAIM), BASE: r2_from(r, BASE),
                         "gap": r2_from(r, CLAIM) - r2_from(r, BASE)})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    # ======================================================================== T1 every single
    print("\n" + "=" * 118)
    print("T1  DOES IT BEAT EVERY SINGLE-ESTIMATOR CANDIDATE?  (best single per cell, and the gap)")
    rows = []
    for h in HORIZONS:
        for uni in ("INDEX", "SINGLE", "SPY"):
            r = store[(h, uni)]
            sr = {s: r2_from(r, s) for s in SINGLES}
            bs = max(sr, key=sr.get)
            rows.append({"h": h, "uni": uni, "claim": r2_from(r, CLAIM), "best_single": bs,
                         "r2_best_single": sr[bs], "gap": r2_from(r, CLAIM) - sr[bs],
                         "beats_all": r2_from(r, CLAIM) > sr[bs]})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    # ======================================================================== T2 controls
    print("\n" + "=" * 118)
    print("T2  SIMPLER EXPLANATION: close-only multi-window controls (NO range data, similar")
    print("    parameter count).  Columns are R2; 'vs claim' rows show control minus claim.")
    tab = {}
    for h in HORIZONS:
        for uni in ("INDEX", "SINGLE"):
            r = store[(h, uni)]
            tab[(uni, h)] = {s: r2_from(r, s) for s in
                             [CLAIM, "har4+yz21", "har4", "har3", "HARc4", "HARc3",
                              "CTL_cc4win", "CTL_cc4win+ewma94", "CTL_cc4win+yz21", "CTL_ewma3",
                              "CTL_ewma94+cc126", "CTL_cc21+cc126", "ewma94", "ewma97", "yz21"]}
    T = pd.DataFrame(tab)
    print(T.to_string(float_format=lambda v: f"{v:8.4f}"))
    print("\n    control MINUS claim  (>=0 means the boring close-only model is as good or better):")
    Tg = T.sub(T.loc[CLAIM], axis=1)
    print(Tg.drop(index=[CLAIM]).to_string(float_format=lambda v: f"{v:+8.4f}"))
    print("\n    share of the claim's advantage over ewma94 that a CLOSE-ONLY model already gets:")
    for (uni, h) in T.columns:
        tot = T.loc[CLAIM, (uni, h)] - T.loc["ewma94", (uni, h)]
        for ctl in ("HARc4", "CTL_cc4win+ewma94", "CTL_ewma94+cc126"):
            got = T.loc[ctl, (uni, h)] - T.loc["ewma94", (uni, h)]
            print(f"      {uni:6s} h={h:2d} {ctl:20s} {got/tot*100:6.1f}%   "
                  f"(claim gap {tot:+.4f}, control gap {got:+.4f})")

    # ======================================================================== T3 inference
    print("\n" + "=" * 118)
    print("T3  HONEST INFERENCE.  (a) collapse to ONE loss differential per DATE (removes the")
    print("    278-stock cross-sectional double count), Newey-West HAC(2h) t-stat on the mean.")
    print("    (b) strictly NON-OVERLAPPING every-h-th date, iid t-test. Negative mean = claim wins.")
    rows = []
    for h in HORIZONS:
        for uni in ("INDEX", "SINGLE", "SPY"):
            r = store[(h, uni)]
            md, t, nd_, se = hac_t(r, CLAIM, BASE, h)
            dm = np.zeros(len(r["cnt_d"]), bool)
            live = np.flatnonzero(r["cnt_d"] > 0)
            dm[live[::h]] = True
            md2, t2, nd2, se2 = hac_t(r, CLAIM, BASE, 1, dmask=dm)
            rows.append({"h": h, "uni": uni, "n_dates": nd_, "mean_dloss": md, "HAC_t": t,
                         "R2gap_all": r2_from(r, CLAIM) - r2_from(r, BASE),
                         "nonov_dates": nd2, "nonov_mean_dloss": md2, "nonov_t": t2,
                         "R2gap_nonov": r2_from(r, CLAIM, dm) - r2_from(r, BASE, dm),
                         "n_nonov": n_from(r, dm)})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:9.4f}"))

    # ======================================================================== T4 fragility
    print("\n" + "=" * 118)
    print("T4  FRAGILITY.  (a) per-year OOS R2 gap claim-minus-ewma94; (b) drop crisis years.")
    for h in HORIZONS:
        for uni in ("INDEX", "SINGLE"):
            r = store[(h, uni)]
            g = []
            for yy in r["years"]:
                dm = (dyear.values == yy)
                g.append(r2_from(r, CLAIM, dm) - r2_from(r, BASE, dm))
            g = np.array(g)
            print(f"  h={h:2d} {uni:6s} wins {int((g>0).sum())}/{len(g)} years  "
                  f"median gap {np.median(g):+.4f}  worst {g.min():+.4f} ({r['years'][int(g.argmin())]})  "
                  f"best {g.max():+.4f} ({r['years'][int(g.argmax())]})")
    print()
    for drop, tag in (((2008, 2009), "drop 2008-09"), ((2020,), "drop 2020"),
                      ((2008, 2009, 2020), "drop 2008-09+2020")):
        rows = []
        for h in HORIZONS:
            for uni in ("INDEX", "SINGLE"):
                r = store[(h, uni)]
                dm = ~np.isin(dyear.values, drop)
                rows.append({"h": h, "uni": uni, "n": n_from(r, dm),
                             CLAIM: r2_from(r, CLAIM, dm), BASE: r2_from(r, BASE, dm),
                             "gap": r2_from(r, CLAIM, dm) - r2_from(r, BASE, dm),
                             "HAC_t": hac_t(r, CLAIM, BASE, h, dmask=dm)[1]})
        print(f"  -- SCORING {tag}")
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    print("\n  -- INDEX is 4 series and SPY/^GSPC are near-duplicates; per-series R2 gap:")
    rows = []
    for h in HORIZONS:
        for uni in ("SPY", "QQQ", "IWM"):
            r = store[(h, uni)]
            md, t, nd_, _ = hac_t(r, CLAIM, BASE, h)
            rows.append({"h": h, "series": uni, "n": r["n"], CLAIM: r2_from(r, CLAIM),
                         BASE: r2_from(r, BASE), "gap": r2_from(r, CLAIM) - r2_from(r, BASE),
                         "HAC_t": t})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    # ======================================================================== T5 purge
    print("\n" + "=" * 118)
    print("T5  LOOKAHEAD: purged walk-forward. A December training row's h-day target reaches INTO")
    print("    the test year, so test-period realizations are in the training set. Purge the last")
    print("    h+5 trading dates of the training data and also drop crisis years from TRAINING.")
    rows = []
    for h in HORIZONS:
        for uni in ("INDEX", "SINGLE"):
            d = store[(h, uni, "d")]
            rp = wf(d, allf, {CLAIM: MULTI[CLAIM], BASE: [BASE]}, h, purge=h + 5)
            rows.append({"h": h, "uni": uni, "variant": f"purge {h+5}d", "n": rp["n"],
                         CLAIM: r2_from(rp, CLAIM), BASE: r2_from(rp, BASE),
                         "gap": r2_from(rp, CLAIM) - r2_from(rp, BASE),
                         "HAC_t": hac_t(rp, CLAIM, BASE, h)[1]})
            rd = wf(d, allf, {CLAIM: MULTI[CLAIM], BASE: [BASE]}, h, purge=h + 5,
                    drop_train_years=(2008, 2009, 2020))
            dm = ~np.isin(dyear.values, (2008, 2009, 2020))
            rows.append({"h": h, "uni": uni, "variant": "purge+notrain0809/20", "n": n_from(rd, dm),
                         CLAIM: r2_from(rd, CLAIM, dm), BASE: r2_from(rd, BASE, dm),
                         "gap": r2_from(rd, CLAIM, dm) - r2_from(rd, BASE, dm),
                         "HAC_t": hac_t(rd, CLAIM, BASE, h, dmask=dm)[1]})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    # ======================================================================== T6 clipping
    print("\n" + "=" * 118)
    print("T6  CLIPPING ARTIFACT: score only rows where NEITHER the target NOR any feature hit the")
    print("    [1e-3, 0.5] clip band (halted / zero-range bars create a shared point mass).")
    rows = []
    for h in HORIZONS:
        for uni in ("INDEX", "SINGLE"):
            r = store[(h, uni)]
            f_ = lambda s: 1.0 - r["sse_u"][s] / r["dev2_u"]
            rows.append({"h": h, "uni": uni, "n_unclipped": r["cnt_u"],
                         "pct_clipped": 100 * (1 - r["cnt_u"] / r["n"]),
                         CLAIM: f_(CLAIM), BASE: f_(BASE), "gap": f_(CLAIM) - f_(BASE),
                         "gap_allrows": r2_from(r, CLAIM) - r2_from(r, BASE)})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:9.4f}"))

    # ======================================================================== T7 selection
    print("\n" + "=" * 118)
    print("T7  SPEC-SELECTION LOOKAHEAD: the claim spec was picked from 33 candidates AFTER seeing")
    print("    the whole 2006-2026 OOS table. Prospective rule: each test year pick the spec with")
    print("    the best cumulative OOS R2 over PRIOR test years only (>=3 prior years needed).")
    for h in HORIZONS:
        for uni in ("INDEX", "SINGLE"):
            r = store[(h, uni)]
            ys = r["years"]
            cand = list(SPECS)
            csse = {s: 0.0 for s in cand}
            csst = 0.0
            picks, sse_sel, sse_claim, sse_base, sst_sel = [], 0.0, 0.0, 0.0, 0.0
            for i, yy in enumerate(ys):
                dm = (dyear.values == yy)
                mm = dm & (r["cnt_d"] > 0)
                if i >= 3:
                    best = min(cand, key=lambda s: csse[s])
                    picks.append(best)
                    sse_sel += r["sse_d"][best][mm].sum()
                    sse_claim += r["sse_d"][CLAIM][mm].sum()
                    sse_base += r["sse_d"][BASE][mm].sum()
                    sst_sel += r["dev2_d"][mm].sum()
                for s in cand:
                    csse[s] += r["sse_d"][s][mm].sum()
                csst += r["dev2_d"][mm].sum()
            from collections import Counter
            cnt = Counter(picks)
            print(f"  h={h:2d} {uni:6s}  prospective-pick R2={1-sse_sel/sst_sel:.4f}  "
                  f"claim R2={1-sse_claim/sst_sel:.4f}  ewma94 R2={1-sse_base/sst_sel:.4f}   "
                  f"picks: {dict(cnt.most_common(5))}")

    # ======================================================================== T8 subgroups
    print("\n" + "=" * 118)
    print("T8  SUBGROUPS (Simpson check): OOS R2 and mean log-forecast bias inside prior-vol")
    print("    (cc21) quintiles, quintile edges taken from training data only. q0=calmest.")
    for h in HORIZONS:
        for uni in ("INDEX", "SINGLE"):
            r = store[(h, uni)]
            line = []
            for q in range(5):
                rc = 1 - r["sse_q"][CLAIM][q] / r["dev2_q"][q]
                rb = 1 - r["sse_q"][BASE][q] / r["dev2_q"][q]
                bc = r["res_q"][CLAIM][q] / r["cnt_q"][q]
                bb = r["res_q"][BASE][q] / r["cnt_q"][q]
                line.append(f"q{q}: n={int(r['cnt_q'][q]):>7,} R2 {rc:+.3f}/{rb:+.3f} "
                            f"gap {rc-rb:+.3f} bias {bc:+.3f}/{bb:+.3f}")
            print(f"  h={h:2d} {uni:6s}")
            for s in line:
                print("      " + s)


if __name__ == "__main__":
    main()
