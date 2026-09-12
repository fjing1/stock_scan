"""
_move_wf_verify_parkbias2.py — part 2 of the adversarial check on the "Parkinson bias is fixable" claim.

Part 1 (_move_wf_verify_parkbias.py) established:
  * every quoted number reproduces exactly off the claim's own artifact _move_wf_volest.pkl
  * but park21 is NOT the top-corr single estimator at h=21 INDEX (park10=gk10=0.7047 > 0.6922)
  * park21 - ewma94 at h=21 INDEX = +0.0134, date-block 95% CI [-0.0136,+0.0419], 11/21 years
  * sign flips to -0.0029 on non-overlapping windows and to -0.0022 on ^GSPC alone
  * the offered overnight-share mechanism mispredicts the SINGLE ratio by -16.8% (wrong direction)

Part 2 answers the three things that decide whether the CORE mechanism ("level bias is fixable,
noise is not") survives once the post-hoc estimator pick is removed:

  B1  ORACLE vs TRAIN level fix. A "fixable rescaling" needs the multiplier to be knowable in
      advance. Score a + ln(est) with `a` from (i) prior years (honest) vs (ii) the test year
      (oracle). The gap is the part of the level bias that is NOT fixable, because the multiplier
      drifts (part 1: INDEX cc21/park21 ratio trends 1.07 -> 1.28 over the sample).
  B2  HONEST OOS ESTIMATOR SELECTION. The claim picked park21 after seeing all 20 out-of-sample
      results. Redo it properly: each test year, pick the argmax-r2_adj estimator on prior years
      only, separately within {range-based} and {close-only}. Then range-vs-close is a fair fight
      with no post-hoc pick on either side.
  B3  BIAS ACCOUNTING. Decompose r2_adj - r2_raw exactly into
        level term   = (E[y-x])^2 / var(y)        <- "rescaling", the claim's story
        shrink term  = residual                    <- "noise", the claim says this is NOT the issue
      per estimator, and check whether the shrink term is estimator-specific or a universal freebie.
  B4  2026 is a PARTIAL year (n=308 index rows). Drop it and recompute the pooled INDEX gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _move_data as D
import _move_lib as L

FLOOR, CEIL = 1e-3, 0.5
MIN_COV = 500
IDX = ["SPY", "QQQ", "IWM", "^GSPC"]
NOT_SINGLE = {"SPY", "QQQ", "IWM", "DIA", "^GSPC", "^VIX"}
BLOCK = 21
pd.set_option("display.width", 240)

RANGE_EST = ["park5", "park10", "park21", "gk5", "gk10", "gk21",
             "rs5", "rs10", "rs21", "yz5", "yz10", "yz21"]
CLOSE_EST = ["cc5", "cc10", "cc21", "cc63", "cc126", "ewma90", "ewma94", "ewma97"]


def load_panel():
    p = D.load()
    o, h, l, c = p["Open"], p["High"], p["Low"], p["Close"]
    cov = c.notna().sum()
    keep = [s for s in c.columns if cov[s] >= MIN_COV]
    o, h, l, c = o[keep].copy(), h[keep].copy(), l[keep].copy(), c[keep].copy()
    bad = (h < l) | (h <= 0) | (l <= 0) | (o <= 0) | (c <= 0)
    for df in (o, h, l, c):
        df[bad] = np.nan
    return o, h, l, c


def build_est(o, h, l, c):
    E = {}
    for n in (5, 10, 21, 63, 126):
        E[f"cc{n}"] = L.vol_cc(c, n)
    for lam in (0.90, 0.94, 0.97):
        E[f"ewma{int(lam*100)}"] = L.vol_ewma(c, lam)
    for n in (5, 10, 21):
        E[f"park{n}"] = L.vol_parkinson(h, l, n)
        E[f"gk{n}"] = L.vol_garman_klass(o, h, l, c, n)
        E[f"rs{n}"] = L.vol_rogers_satchell(o, h, l, c, n)
        E[f"yz{n}"] = L.vol_yang_zhang(o, h, l, c, n)
    return E


def stack(E, feats, tgt, cols):
    T, S = len(tgt.index), len(cols)
    tv = tgt[cols].values.astype(np.float64)
    y = np.log(np.clip(tv, FLOOR, CEIL)).ravel()
    X = np.empty((T * S, len(feats)))
    for j, f in enumerate(feats):
        X[:, j] = np.log(np.clip(E[f][cols].values.astype(np.float64), FLOOR, CEIL)).ravel()
    di = np.repeat(np.arange(T), S)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    return y[ok], X[ok], di[ok]


def boot_gap(di, dev, ea, eb, n_boot=2000, seed=11, block=BLOCK):
    """Date-block bootstrap of the R2 gap implied by two per-obs squared-error vectors."""
    rng = np.random.default_rng(seed)
    blk = di // block
    ub, inv = np.unique(blk, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    st = np.searchsorted(inv[order], np.arange(len(ub)))
    en = np.append(st[1:], len(order))
    idx_by = [order[st[i]:en[i]] for i in range(len(ub))]
    nb = len(ub)
    d = np.empty(n_boot)
    for b in range(n_boot):
        sel = np.concatenate([idx_by[i] for i in rng.integers(0, nb, nb)])
        s = float((dev[sel] ** 2).sum())
        d[b] = (float(eb[sel].sum()) - float(ea[sel].sum())) / s
    return d, nb


def run(E, feats, tgt, cols, years, drop_years=()):
    y, X, di = stack(E, feats, tgt, cols)
    yr = years[di]
    ys = sorted(set(years))
    sse = {f: dict(raw=0.0, lvl=0.0, lvl_oracle=0.0, adj=0.0, adj_oracle=0.0) for f in feats}
    sst = 0.0
    err = {f: {k: [] for k in ("raw", "lvl", "lvl_oracle", "adj")} for f in feats}
    # OOS-selected composites
    sel = {"OOSpick_range": dict(sse=0.0, err=[], picks=[]),
           "OOSpick_close": dict(sse=0.0, err=[], picks=[]),
           "OOSpick_any": dict(sse=0.0, err=[], picks=[])}
    keep_di, keep_dev = [], []
    prev_r2 = {}           # r2_adj of each estimator on the PREVIOUS test year(s), for OOS picking
    cum = {f: dict(ssr=0.0, sst=0.0) for f in feats}
    rows = []
    for yy in ys[5:]:
        tr, te = yr < yy, yr == yy
        if te.sum() < 20 or tr.sum() < 500 or yy in drop_years:
            continue
        mu = y[tr].mean()
        dev = y[te] - mu
        sst_y = float((dev ** 2).sum())
        sst += sst_y
        # --- pick the OOS winner using ONLY prior-year test performance (walk-forward selection)
        picks = {}
        for tag, pool in (("OOSpick_range", RANGE_EST), ("OOSpick_close", CLOSE_EST),
                          ("OOSpick_any", RANGE_EST + CLOSE_EST)):
            if cum[pool[0]]["sst"] > 0:
                scores = {f: 1 - cum[f]["ssr"] / cum[f]["sst"] for f in pool}
                picks[tag] = max(scores, key=scores.get)
            else:
                picks[tag] = pool[0]
        for f in feats:
            j = feats.index(f)
            xt, xe = X[tr, j], X[te, j]
            a_tr = float((y[tr] - xt).mean())
            a_or = float((y[te] - xe).mean())                       # ORACLE level (cheating)
            A = np.column_stack([np.ones(tr.sum()), xt])
            b, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
            Ao = np.column_stack([np.ones(te.sum()), xe])
            bo, *_ = np.linalg.lstsq(Ao, y[te], rcond=None)          # ORACLE affine (cheating)
            pr = {"raw": xe, "lvl": a_tr + xe, "lvl_oracle": a_or + xe,
                  "adj": b[0] + b[1] * xe, "adj_oracle": bo[0] + bo[1] * xe}
            for k, p in pr.items():
                e = (y[te] - p) ** 2
                sse[f][k] += float(e.sum())
                if k in err[f]:
                    err[f][k].append(e)
            cum[f]["ssr"] += float(((y[te] - pr["adj"]) ** 2).sum())
            cum[f]["sst"] += sst_y
        for tag, pk in picks.items():
            j = feats.index(pk)
            A = np.column_stack([np.ones(tr.sum()), X[tr, j]])
            b, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
            e = (y[te] - (b[0] + b[1] * X[te, j])) ** 2
            sel[tag]["sse"] += float(e.sum())
            sel[tag]["err"].append(e)
            sel[tag]["picks"].append((yy, pk))
        keep_di.append(di[te]); keep_dev.append(dev)
        rows.append({"year": yy, "n": int(te.sum()), **{f"pick_{k}": v for k, v in picks.items()}})
    out = {}
    for f in feats:
        out[f] = {k: 1 - sse[f][k] / sst for k in sse[f]}
        out[f]["n"] = int(sum(len(e) for e in err[f]["raw"]))
    for tag in sel:
        out[tag] = {"adj": 1 - sel[tag]["sse"] / sst, "raw": np.nan, "lvl": np.nan,
                    "lvl_oracle": np.nan, "adj_oracle": np.nan, "n": np.nan}
    res = pd.DataFrame(out).T
    E_ = {f: {k: np.concatenate(v) for k, v in err[f].items()} for f in feats}
    for tag in sel:
        E_[tag] = {"adj": np.concatenate(sel[tag]["err"])}
    return res, E_, np.concatenate(keep_di), np.concatenate(keep_dev), pd.DataFrame(rows), sel


def main():
    o, hi, lo, c = load_panel()
    idx_cols = [s for s in IDX if s in c.columns]
    sng_cols = [s for s in c.columns if s not in NOT_SINGLE]
    E = build_est(o, hi, lo, c)
    years = c.index.year.values
    FEATS = CLOSE_EST + RANGE_EST
    print(f"indices={idx_cols} singles={len(sng_cols)} feats={len(FEATS)}")

    for h in (5, 21):
        tgt = L.realized_vol_forward(c, h)
        for lab, cols in (("INDEX", idx_cols), ("SINGLE", sng_cols)):
            res, ERR, di, dev, picks, sel = run(E, FEATS, tgt, cols, years)
            print("\n" + "=" * 130)
            print(f"h={h} {lab}   n={int(res.loc['park21','n']):,}")

            # ---------- B3 bias accounting (exact)
            print("\nB3  BIAS ACCOUNTING (all OOS).  level_gain = r2_lvl - r2_raw  (pure RESCALING)")
            print("    shrink_gain = r2_adj - r2_lvl  (pure NOISE shrinkage).  unfixable_level = ")
            print("    r2_lvl_oracle - r2_lvl (level you could NOT know in advance).")
            t = res.loc[FEATS, ["raw", "lvl", "lvl_oracle", "adj", "adj_oracle"]].copy()
            t.columns = ["r2_raw", "r2_lvl", "r2_lvlOR", "r2_adj", "r2_adjOR"]
            t["level_gain"] = t.r2_lvl - t.r2_raw
            t["shrink_gain"] = t.r2_adj - t.r2_lvl
            t["unfixable_level"] = t.r2_lvlOR - t.r2_lvl
            t["oracle_headroom"] = t.r2_adjOR - t.r2_adj
            print(t.sort_values("r2_adj", ascending=False).to_string(
                float_format=lambda v: f"{v:9.4f}"))

            # ---------- B2 honest OOS selection
            print("\nB2  HONEST OOS SELECTION (winner chosen on prior test years only, no post-hoc pick)")
            for tag in ("OOSpick_range", "OOSpick_close", "OOSpick_any"):
                print(f"   {tag:16s} r2_adj={res.loc[tag,'adj']:.4f}")
            print(f"   {'park21 (post-hoc)':16s} r2_adj={res.loc['park21','adj']:.4f}")
            print(f"   {'ewma94':16s} r2_adj={res.loc['ewma94','adj']:.4f}")
            d, nb = boot_gap(di, dev, ERR["OOSpick_range"]["adj"], ERR["OOSpick_close"]["adj"])
            print(f"   RANGE-pick minus CLOSE-pick = {d.mean():+.4f}  95% "
                  f"[{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}] "
                  f"P(range better)={np.mean(d>0):.3f}   (n_blocks={nb})")
            d2, _ = boot_gap(di, dev, ERR["park21"]["adj"], ERR["OOSpick_close"]["adj"])
            print(f"   park21 minus CLOSE-pick     = {d2.mean():+.4f}  95% "
                  f"[{np.percentile(d2,2.5):+.4f},{np.percentile(d2,97.5):+.4f}] "
                  f"P(park21 better)={np.mean(d2>0):.3f}")
            d3, _ = boot_gap(di, dev, ERR["OOSpick_range"]["adj"], ERR["park21"]["adj"])
            print(f"   RANGE-pick minus park21     = {d3.mean():+.4f}  95% "
                  f"[{np.percentile(d3,2.5):+.4f},{np.percentile(d3,97.5):+.4f}]")
            print("   picks per year:")
            print("     range:", " ".join(f"{y}:{p}" for y, p in sel["OOSpick_range"]["picks"]))
            print("     close:", " ".join(f"{y}:{p}" for y, p in sel["OOSpick_close"]["picks"]))
            print("     any  :", " ".join(f"{y}:{p}" for y, p in sel["OOSpick_any"]["picks"]))

            # ---------- B1 fixability
            print("\nB1  FIXABILITY of the level: how much of the level bias is knowable in advance?")
            for k in ("park21", "park10", "gk10", "cc21", "ewma94", "ewma97"):
                lg = t.loc[k, "level_gain"]; un = t.loc[k, "unfixable_level"]
                tot = lg + un
                print(f"   {k:8s} level available OOS={lg:+.4f}  still-missing (oracle-only)"
                      f"={un:+.4f}  -> {100*lg/tot if tot else np.nan:5.1f}% of the level bias is fixable")

    # ---------- B4 partial 2026
    print("\n" + "=" * 130)
    print("B4  2026 IS A PARTIAL YEAR. Drop it (and then also 2008/2009/2020) from the INDEX cell.")
    tgt = L.realized_vol_forward(c, 21)
    for tag, dy in (("all years", ()), ("drop 2026", (2026,)),
                    ("drop 2026+crises", (2026, 2008, 2009, 2020))):
        res, ERR, di, dev, _, _ = run(E, FEATS, tgt, idx_cols, years, drop_years=dy)
        d, nb = boot_gap(di, dev, ERR["park21"]["adj"], ERR["ewma94"]["adj"])
        d2, _ = boot_gap(di, dev, ERR["park21"]["adj"], ERR["gk10"]["adj"])
        print(f"  {tag:18s} n={int(res.loc['park21','n']):,}  park21={res.loc['park21','adj']:.4f} "
              f"ewma94={res.loc['ewma94','adj']:.4f} gk10={res.loc['gk10','adj']:.4f} "
              f"park10={res.loc['park10','adj']:.4f} | park21-ewma94={d.mean():+.4f} "
              f"95%[{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}] P={np.mean(d>0):.3f}"
              f" | park21-gk10={d2.mean():+.4f} P={np.mean(d2>0):.3f}")


if __name__ == "__main__":
    main()
