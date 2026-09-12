"""
_move_wf_verify_harvol3.py — round 3: nail the two cracks found in rounds 1-2.

  F. WHICH SPEC PRODUCED THE REPORTED NUMBERS?  The claim names ONE spec
     (har_d/w/m/q + yz21 + ewma97).  Its actual INDEX R2 is BELOW every number quoted as evidence
     for the INDEX universe, while the SINGLE numbers match exactly.  Print every HAR variant on
     the identical sample so we can see which spec each quoted number came from.
  G. "BEATS EVERY SINGLE-ESTIMATOR CANDIDATE" — date-block bootstrap in the SAME style the
     original used (1000 draws, 21-date blocks) but against the BEST SINGLE ESTIMATOR instead of
     ewma94, on both the original grid and a widened grid. Report P(claim > single).
  H. QLIKE significance — the round-2 point estimates reversed in several cells. Date-collapsed
     Newey-West HAC(2h) t-test on the QLIKE differential, plus non-overlapping subsample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _move_lib as L
import _move_wf_volest as V
from _move_wf_verify_harvol import (SINGLES, MULTI, CLAIM, BASE, build, gram, gadd, gzero,
                                    solve, hac_t)
from _move_wf_verify_harvol2 import EXTRA_SINGLES, extend

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 80)
HORIZONS = (1, 5, 10, 21)
BLOCK = 21

HARVARS = {CLAIM: MULTI[CLAIM], "har4+yz21": MULTI["har4+yz21"], "har4": MULTI["har4"],
           "har3": MULTI["har3"], "har3+yz21": ["har_d", "har_w", "har_m", "yz21"],
           "har5": ["har_d", "har_w", "har_m", "har_q", "har_y"],
           "HARc4": MULTI["HARc4"], "HARc3": MULTI["HARc3"]}


def wf3(d, allf, specs, floor, ceil, all_years):
    """Per-date squared-error and per-date QLIKE aggregates for every spec."""
    fi = {f: j for j, f in enumerate(allf)}
    y, X, di, yr = d["y"], d["X"], d["di"], d["yr"]
    k = len(allf)
    gy = {yy: gram(X, y, yr == yy) for yy in all_years if (yr == yy).sum() > 0}
    cum, run = {}, gzero(k)
    for yy in all_years:
        cum[yy] = run
        if yy in gy:
            run = gadd(run, gy[yy])
    nd = d["T"]
    names = list(specs)
    sse_d = {s: np.zeros(nd) for s in names}
    ql_d = {s: np.zeros(nd) for s in names}
    cnt_d, dev2_d = np.zeros(nd), np.zeros(nd)
    n_tot, used = 0, []
    for yy in all_years[5:]:
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
            np.add.at(ql_d[s], dt, v_a / v_f - np.log(v_a / v_f) - 1.0)
    return dict(sse_d=sse_d, ql_d=ql_d, cnt_d=cnt_d, dev2_d=dev2_d, n=n_tot, years=used)


def r2_(r, s, dm=None):
    m = r["cnt_d"] > 0
    if dm is not None:
        m = m & dm
    return 1.0 - r["sse_d"][s][m].sum() / r["dev2_d"][m].sum()


def qk_(r, s, dm=None):
    m = r["cnt_d"] > 0
    if dm is not None:
        m = m & dm
    return r["ql_d"][s][m].sum() / r["cnt_d"][m].sum()


def hac_q(r, a, b, h, dm=None):
    m = r["cnt_d"] > 0
    if dm is not None:
        m = m & dm
    dl = (r["ql_d"][a][m] - r["ql_d"][b][m]) / r["cnt_d"][m]
    n = len(dl)
    x = dl - dl.mean()
    lag = max(1, 2 * h)
    s = float(x @ x) / n
    for j in range(1, min(lag, n - 1) + 1):
        s += 2.0 * (1 - j / (lag + 1)) * float(x[j:] @ x[:-j]) / n
    return float(dl.mean()), float(dl.mean() / np.sqrt(max(s, 1e-30) / n)), n


def date_block_boot(r, names, n_boot=1000, seed=7, block=BLOCK):
    """Same recipe the original used: resample blocks of `block` consecutive trading dates."""
    live = np.flatnonzero(r["cnt_d"] > 0)
    blk = live // block
    ub = np.unique(blk)
    groups = [live[blk == b] for b in ub]
    rng = np.random.default_rng(seed)
    out = {s: np.empty(n_boot) for s in names}
    for i in range(n_boot):
        sel = np.concatenate([groups[j] for j in rng.integers(0, len(ub), len(ub))])
        sst = r["dev2_d"][sel].sum()
        for s in names:
            out[s][i] = 1.0 - r["sse_d"][s][sel].sum() / sst
    return out


def main():
    o, hi, lo, c = V.load_panel()
    idx_cols = [s for s in V.IDX if s in c.columns]
    sng_cols = [s for s in c.columns if s not in V.NOT_SINGLE]
    E = extend(V.build_estimators(o, hi, lo, c), o, hi, lo, c)
    allf = sorted(E)
    years = c.index.year.values
    ay = sorted(set(years.tolist()))

    SP = dict(HARVARS)
    SP.update({s: [s] for s in SINGLES + EXTRA_SINGLES})

    store = {}
    for h in HORIZONS:
        tgt = (np.log(c).diff().abs().shift(-1) if h == 1 else L.realized_vol_forward(c, h))
        for uni, cols in (("INDEX", idx_cols), ("SINGLE", sng_cols), ("SPY", ["SPY"])):
            d = build(E, allf, tgt, cols, years)
            store[(h, uni)] = wf3(d, allf, SP, V.FLOOR, V.CEIL, ay)
            del d

    # ---------------------------------------------------------------- F
    print("=" * 132)
    print("F.  WHICH SPEC PRODUCED THE QUOTED EVIDENCE NUMBERS?")
    print("    quoted: h=1 INDEX .202 SINGLE .197 | h=5 .488 .507 | h=10 .559 .618 | h=21 .530 .673")
    tab = {}
    for h in HORIZONS:
        for uni in ("INDEX", "SINGLE"):
            r = store[(h, uni)]
            tab[(uni, h)] = {s: r2_(r, s) for s in HARVARS}
            tab[(uni, h)]["ewma94"] = r2_(r, BASE)
            tab[(uni, h)]["n"] = r["n"]
    T = pd.DataFrame(tab)
    print(T.to_string(float_format=lambda v: f"{v:9.4f}"))
    print("\n    which HAR variant is the per-cell best, and how far is the NAMED spec below it:")
    for h in HORIZONS:
        for uni in ("INDEX", "SINGLE"):
            r = store[(h, uni)]
            v = {s: r2_(r, s) for s in HARVARS}
            b = max(v, key=v.get)
            print(f"      h={h:2d} {uni:6s} best={b:18s} {v[b]:.4f}   named({CLAIM})={v[CLAIM]:.4f}"
                  f"   named minus best = {v[CLAIM]-v[b]:+.4f}")

    # ---------------------------------------------------------------- G
    print("\n" + "=" * 132)
    print("G.  DATE-BLOCK BOOTSTRAP (1000 draws, 21-date blocks) of the NAMED spec against the best")
    print("    SINGLE estimator — the half of the claim the original only checked against ewma94.")
    rows = []
    for h in HORIZONS:
        for uni in ("INDEX", "SINGLE", "SPY"):
            r = store[(h, uni)]
            og = {s: r2_(r, s) for s in SINGLES}
            wg = {s: r2_(r, s) for s in SINGLES + EXTRA_SINGLES}
            bo, bw = max(og, key=og.get), max(wg, key=wg.get)
            dr = date_block_boot(r, [CLAIM, bo, bw, BASE])
            for tag, k in (("orig-grid best single", bo), ("wide-grid best single", bw),
                           ("ewma94", BASE)):
                dif = dr[CLAIM] - dr[k]
                rows.append({"h": h, "uni": uni, "vs": f"{tag}={k}", "gap": dif.mean(),
                             "lo95": np.percentile(dif, 2.5), "hi95": np.percentile(dif, 97.5),
                             "P(claim>rival)": float((dif > 0).mean())})
    G = pd.DataFrame(rows)
    print(G.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))

    # ---------------------------------------------------------------- H
    print("\n" + "=" * 132)
    print("H.  QLIKE ON VARIANCE — the loss the vol-forecasting literature uses, and the one that")
    print("    matters for turning sigma into P(|r|>2%). Negative dQ = named spec wins.")
    rows = []
    for h in HORIZONS:
        for uni in ("INDEX", "SINGLE", "SPY"):
            r = store[(h, uni)]
            qs = {s: qk_(r, s) for s in SINGLES + EXTRA_SINGLES}
            bs = min(qs, key=qs.get)
            live = np.flatnonzero(r["cnt_d"] > 0)
            dm = np.zeros(len(r["cnt_d"]), bool)
            dm[live[::h]] = True
            m1, t1, nd1 = hac_q(r, CLAIM, BASE, h)
            m2, t2, _ = hac_q(r, CLAIM, bs, h)
            m3, t3, nd3 = hac_q(r, CLAIM, BASE, 1, dm=dm)
            rows.append({"h": h, "uni": uni, "Q_claim": qk_(r, CLAIM), "Q_ewma94": qk_(r, BASE),
                         "dQ_vs_ewma94": m1, "t": t1, "win": m1 < 0,
                         "nonov_dQ": m3, "nonov_t": t3,
                         "best_single": bs, "Q_bs": qs[bs], "dQ_vs_bs": m2, "t_bs": t2,
                         "win_bs": m2 < 0})
    H = pd.DataFrame(rows)
    print(H.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
    print("\n    QLIKE scoreboard for the NAMED spec (out of 12 cells incl. SPY):")
    print(f"      beats ewma94:              {int(H.win.sum())}/12")
    print(f"      beats best single:         {int(H.win_bs.sum())}/12")
    print(f"      beats ewma94 with |t|>2:   {int(((H.dQ_vs_ewma94<0)&(H.t.abs()>2)).sum())}/12")
    print(f"      LOSES  to ewma94 with |t|>2: {int(((H.dQ_vs_ewma94>0)&(H.t.abs()>2)).sum())}/12")

    print("\n" + "=" * 132)
    print("I.  R2 scoreboard for the NAMED spec, all 12 cells, with the honest date-clustered test")
    rows = []
    for h in HORIZONS:
        for uni in ("INDEX", "SINGLE", "SPY"):
            r = store[(h, uni)]
            wg = {s: r2_(r, s) for s in SINGLES + EXTRA_SINGLES}
            bw = max(wg, key=wg.get)
            _, tb, _, _ = hac_t(r, CLAIM, bw, h)
            _, te_, _, _ = hac_t(r, CLAIM, BASE, h)
            rows.append({"h": h, "uni": uni, "beats_ewma94": r2_(r, CLAIM) > r2_(r, BASE),
                         "t_ewma94": te_, "sig_ewma94": abs(te_) > 2,
                         "beats_best_single": r2_(r, CLAIM) > wg[bw], "which": bw,
                         "gap_bs": r2_(r, CLAIM) - wg[bw], "t_bs": tb, "sig_bs": abs(tb) > 2})
    S = pd.DataFrame(rows)
    print(S.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
    print(f"\n      beats ewma94 on R2:                       {int(S.beats_ewma94.sum())}/12"
          f"  (significant: {int((S.beats_ewma94 & S.sig_ewma94).sum())}/12)")
    print(f"      beats best single estimator on R2:        {int(S.beats_best_single.sum())}/12"
          f"  (significant: {int((S.beats_best_single & S.sig_bs).sum())}/12)")


if __name__ == "__main__":
    main()
