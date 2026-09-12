"""
_move_wf_verify_parkbias.py — ADVERSARIAL verification of the "Parkinson bias is fixable" claim.

CLAIM UNDER TEST
  "The range estimators' downward level bias is a fixable RESCALING problem, not a NOISE problem:
   Parkinson has the HIGHEST correlation with forward vol of any single estimator, and after an
   affine correction fit on prior years it beats close-to-close and EWMA(0.94), even though
   uncorrected it looks worse than close-to-close."
  Offered: cc/park geo-ratio 1.179 INDEX (n=24,964) / 1.063 SINGLE (n=1,141,638); overnight share
  27.6% -> sqrt(1/(1-.276))=1.175 explains it; corr(ln rv_fwd21, ln park21)=.692 IDX / .798 SNG,
  highest of 20 singles; h=21 INDEX R2_raw .334 (< cc21 .338) but R2_adj .488 > ewma94 .478 >
  cc21 .440; "bias worth .154 R2, fully recovered by a+b*ln(est) with b=.805".

ATTACKS RUN (each prints a VERDICT line)
  A1 DECOMPOSITION  The claim's own word is "RESCALING". A pure rescaling is a+1*ln(est) (a level
     shift). The reported b=0.805 is SHRINKAGE, i.e. a noise correction. Split the R2 gain into
     level-only (b forced to 1, a from train) vs slope shrinkage. If shrinkage does the work, the
     claim's mechanism is self-refuting.
  A2 LOOKAHEAD/PURGE  train mask is `year < y`, but at h=21 the last 21 train rows have targets
     that reach INTO the test year. Re-run with a 21-day embargo.
  A3 SAMPLE SIZE  n=24,964 is 4 near-identical index series x 6,241 overlapping windows. Re-score
     on NON-OVERLAPPING rows only (every h-th date) and date-block bootstrap the park21-vs-ewma94
     and park21-vs-cc21 gaps. Paired, clustered by date.
  A4 SIMPLER EXPLANATION  park21 may win on correlation only because it is the SMOOTHEST estimator,
     not because it uses the range. Controls: cc63/cc126 (longer close-only window), mad21 (mean
     |log r|, close-only but outlier-robust), park63/park126 (is 21 even the best Parkinson?),
     hl21 (raw mean log(H/L), no Parkinson constant).
  A5 FRAGILITY  drop 2008-2009 and 2020 test years; split the 4 index series apart; run every
     horizon and both universes (the claim quotes only h=21 INDEX for the R2 ordering).
  A6 IS THE BIAS ACTUALLY A CONSTANT?  a "fixable rescaling" needs a STABLE multiplier. Report the
     cc/park geo-ratio by year and by vol quintile. Also test the offered overnight-share mechanism
     on SINGLE names, where the sign of the prediction is checkable.

Conventions inherited from _move_wf_volest.py so numbers are comparable: FLOOR/CEIL clip, MIN_COV,
log-space scoring, R2 denominator = TRAIN mean of log target, walk-forward expanding years.
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
CRISIS = {2008, 2009, 2020}
pd.set_option("display.width", 220)


def load_panel():
    p = D.load()
    o, h, l, c = p["Open"], p["High"], p["Low"], p["Close"]
    cov = c.notna().sum()
    keep = [s for s in c.columns if cov[s] >= MIN_COV]
    o, h, l, c = o[keep].copy(), h[keep].copy(), l[keep].copy(), c[keep].copy()
    bad = (h < l) | (h <= 0) | (l <= 0) | (o <= 0) | (c <= 0)
    for df in (o, h, l, c):
        df[bad] = np.nan
    print(f"panel: {len(keep)} syms x {len(c)} rows {c.index[0].date()}->{c.index[-1].date()} "
          f"(dropped {len(cov)-len(keep)} <{MIN_COV} closes; {int(bad.values.sum())} OHLC-bad bars nulled)")
    return o, h, l, c


def build_est(o, h, l, c):
    """Original 20 singles + the A4 controls."""
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
    ORIG20 = list(E)                                            # exactly the claim's 20 singles
    # ---- A4 controls (all causal, all computable at t)
    E["park63"] = L.vol_parkinson(h, l, 63)
    E["park126"] = L.vol_parkinson(h, l, 126)
    r = np.log(c).diff().abs()
    E["mad21"] = r.rolling(21).mean() * np.sqrt(np.pi / 2)       # close-only, outlier-robust
    E["mad63"] = r.rolling(63).mean() * np.sqrt(np.pi / 2)
    E["hl21"] = np.log(h / l).rolling(21).mean()                 # range, no Parkinson constant
    E["yz63"] = L.vol_yang_zhang(o, h, l, c, 63)
    E["gk63"] = L.vol_garman_klass(o, h, l, c, 63)
    return E, ORIG20


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


def wf(E, feats, tgt, cols, years, h, purge=0, drop_crisis=False, nonoverlap=False):
    """Expanding walk-forward. Returns per-candidate SSE for raw / level-only / affine,
    plus the pooled arrays needed for bootstrapping and correlation."""
    y, X, di = stack(E, feats, tgt, cols)
    yr = years[di]
    dates_ok = np.ones(len(y), bool)
    if nonoverlap:
        dates_ok = (di % h) == 0
    sse = {f: {"raw": 0.0, "lvl": 0.0, "adj": 0.0} for f in feats}
    sst = 0.0
    keep_di, keep_dev, keep_err, keep_y, keep_x = [], [], {f: {} for f in feats}, [], {f: [] for f in feats}
    for f in feats:
        keep_err[f] = {"raw": [], "lvl": [], "adj": []}
    per_year = []
    coefs = {f: [] for f in feats}
    for yy in sorted(set(years)):
        te = (yr == yy) & dates_ok
        tr = yr < yy
        if purge and te.any():
            # embargo: a train row at date d has a target spanning d+1..d+h, so any train row with
            # d >= first_test_date - h peeks into the test year. Drop them.
            tr = tr & (di < di[yr == yy].min() - purge)
        if drop_crisis and yy in CRISIS:
            continue
        if te.sum() < 20 or tr.sum() < 500 or yy < sorted(set(years))[5]:
            continue
        mu = y[tr].mean()
        dev = y[te] - mu
        sst += float((dev ** 2).sum())
        row = {"year": yy, "n": int(te.sum())}
        for f in feats:
            j = feats.index(f)
            xt, xe = X[tr, j], X[te, j]
            a_lvl = float((y[tr] - xt).mean())
            A = np.column_stack([np.ones(tr.sum()), xt])
            b, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
            pr = {"raw": xe, "lvl": a_lvl + xe, "adj": b[0] + b[1] * xe}
            for k, p in pr.items():
                e = (y[te] - p) ** 2
                sse[f][k] += float(e.sum())
                keep_err[f][k].append(e)
            coefs[f].append((b[0], b[1], a_lvl, np.exp(a_lvl)))
            row[f] = 1.0 - float(((y[te] - pr["adj"]) ** 2).sum()) / float((dev ** 2).sum())
            keep_x[f].append(xe)
        per_year.append(row)
        keep_di.append(di[te]); keep_dev.append(dev); keep_y.append(y[te])
    out = {}
    yall = np.concatenate(keep_y)
    for f in feats:
        xall = np.concatenate(keep_x[f])
        cf = np.array(coefs[f])
        out[f] = {
            "r2_raw": 1 - sse[f]["raw"] / sst,
            "r2_lvl": 1 - sse[f]["lvl"] / sst,
            "r2_adj": 1 - sse[f]["adj"] / sst,
            "corr": float(np.corrcoef(yall, xall)[0, 1]),
            "slope": cf[:, 1].mean(), "icept": cf[:, 0].mean(),
            "mult": cf[:, 3].mean(),
            "n": len(yall),
        }
    boot = {"di": np.concatenate(keep_di), "dev": np.concatenate(keep_dev),
            "err": {f: {k: np.concatenate(v) for k, v in keep_err[f].items()} for f in feats},
            "sst": sst}
    return pd.DataFrame(out).T, pd.DataFrame(per_year), boot


def block_boot(boot, names, key="adj", n_boot=2000, seed=7, block=BLOCK):
    rng = np.random.default_rng(seed)
    di, dev = boot["di"], boot["dev"]
    blk = di // block
    ub, inv = np.unique(blk, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(ub)))
    ends = np.append(starts[1:], len(order))
    idx_by = [order[starts[i]:ends[i]] for i in range(len(ub))]
    nb = len(ub)
    draws = {k: np.empty(n_boot) for k in names}
    for b in range(n_boot):
        sel = np.concatenate([idx_by[i] for i in rng.integers(0, nb, nb)])
        s = float((dev[sel] ** 2).sum())
        for k in names:
            draws[k][b] = 1.0 - float(boot["err"][k][key][sel].sum()) / s
    return draws, nb


def main():
    o, hi, lo, c = load_panel()
    idx_cols = [s for s in IDX if s in c.columns]
    sng_cols = [s for s in c.columns if s not in NOT_SINGLE]
    E, ORIG20 = build_est(o, hi, lo, c)
    years = c.index.year.values
    CTRL = ["park63", "park126", "mad21", "mad63", "hl21", "yz63", "gk63"]
    FEATS = ORIG20 + CTRL
    print(f"indices={idx_cols}  singles={len(sng_cols)}  candidates={len(FEATS)} "
          f"({len(ORIG20)} original + {len(CTRL)} controls)")

    # ================================================================= A6a offered mechanism
    print("\n" + "=" * 118)
    print("A6a  THE OFFERED MECHANISM: overnight variance share -> cc/park ratio")
    for lab, cols in (("INDEX", idx_cols), ("SINGLE", sng_cols)):
        oc = np.log(o[cols] / c[cols].shift(1)).values ** 2
        cc = np.log(c[cols]).diff().values ** 2
        m = np.isfinite(oc) & np.isfinite(cc)
        share = np.nanmean(oc[m]) / np.nanmean(cc[m])
        pred = np.sqrt(1 / (1 - share))
        cc21, pk21 = L.vol_cc(c, 21)[cols], L.vol_parkinson(hi, lo, 21)[cols]
        rt = np.log(cc21 / pk21).values
        rt = rt[np.isfinite(rt)]
        act = float(np.exp(rt.mean()))
        print(f"  {lab:6s} overnight share={share:.4f}  predicted ratio={pred:.4f}  "
              f"ACTUAL geo-ratio cc21/park21={act:.4f}  n={len(rt):,}  "
              f"err={100*(act/pred-1):+.2f}%")

    # ================================================================= main grid
    store = {}
    for h in (5, 10, 21):
        tgt = L.realized_vol_forward(c, h)
        for lab, cols in (("INDEX", idx_cols), ("SINGLE", sng_cols)):
            store[(h, lab, "base")] = wf(E, FEATS, tgt, cols, years, h)

    KEY = ["park21", "cc21", "ewma94", "ewma97", "yz21", "gk21", "rs21", "cc63", "cc126",
           "mad21", "mad63", "hl21", "park63", "park126", "yz63", "gk63", "cc5"]

    print("\n" + "=" * 118)
    print("REPRODUCTION + A1 DECOMPOSITION   r2_raw (b=1,a=0) | r2_lvl (b=1, a=train: PURE RESCALING)")
    print("                                  | r2_adj (a+b*ln est: rescale + SHRINKAGE)")
    for h in (5, 10, 21):
        for lab in ("INDEX", "SINGLE"):
            res, py, _ = store[(h, lab, "base")]
            t = res.loc[KEY, ["r2_raw", "r2_lvl", "r2_adj", "corr", "slope", "mult", "n"]].copy()
            t["gain_lvl"] = t.r2_lvl - t.r2_raw
            t["gain_shrink"] = t.r2_adj - t.r2_lvl
            t["pct_from_shrink"] = 100 * t.gain_shrink / (t.r2_adj - t.r2_raw)
            print(f"\n-- h={h} {lab}   n={int(t.n.iloc[0]):,}   test years {py.year.min()}-{py.year.max()}")
            print(t.sort_values("r2_adj", ascending=False).to_string(
                float_format=lambda v: f"{v:9.4f}"))

    print("\n" + "=" * 118)
    print("A4  IS park21 REALLY THE TOP CORRELATION?  ranked corr(ln rv_fwd, ln est)")
    for h in (5, 21):
        for lab in ("INDEX", "SINGLE"):
            res, _, _ = store[(h, lab, "base")]
            s20 = res.loc[ORIG20, "corr"].sort_values(ascending=False)
            sall = res["corr"].sort_values(ascending=False)
            print(f"\n-- h={h} {lab}")
            print("   original-20 top5 :", "  ".join(f"{k}={v:.4f}" for k, v in s20.head(5).items()))
            print("   +controls   top8 :", "  ".join(f"{k}={v:.4f}" for k, v in sall.head(8).items()))
            print(f"   park21 rank among all {len(sall)}: "
                  f"{1+list(sall.index).index('park21')};  among original 20: "
                  f"{1+list(s20.index).index('park21')}")
            b20 = res.loc[ORIG20, "r2_adj"].sort_values(ascending=False)
            ball = res["r2_adj"].sort_values(ascending=False)
            print("   r2_adj orig-20 top5:", "  ".join(f"{k}={v:.4f}" for k, v in b20.head(5).items()))
            print("   r2_adj +ctrl  top5:", "  ".join(f"{k}={v:.4f}" for k, v in ball.head(5).items()))

    # ================================================================= A3 significance
    print("\n" + "=" * 118)
    print("A3  DATE-BLOCK BOOTSTRAP (2000 draws, blocks of 21 trading dates; whole dates move")
    print("    together so cross-sectional correlation is respected). PAIRED gaps in R2_adj.")
    for h in (5, 10, 21):
        for lab in ("INDEX", "SINGLE"):
            res, _, boot = store[(h, lab, "base")]
            nm = ["park21", "cc21", "ewma94", "ewma97", "cc63", "mad21", "park63"]
            dr, nb = block_boot(boot, nm)
            print(f"\n-- h={h} {lab}  n_obs={int(res.loc['park21','n']):,}  n_blocks={nb}")
            for opp in ("ewma94", "cc21", "ewma97", "cc63", "mad21", "park63"):
                d = dr["park21"] - dr[opp]
                print(f"   park21 - {opp:8s} = {d.mean():+.4f}  95% "
                      f"[{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}]  "
                      f"P(park21 better)={np.mean(d>0):.3f}")

    print("\n" + "=" * 118)
    print("A3b NON-OVERLAPPING WINDOWS ONLY (score every h-th date; fit still on all prior rows)")
    for h in (5, 21):
        for lab in ("INDEX", "SINGLE"):
            tgt = L.realized_vol_forward(c, h)
            cols = idx_cols if lab == "INDEX" else sng_cols
            res, _, boot = wf(E, FEATS, tgt, cols, years, h, nonoverlap=True)
            nm = ["park21", "cc21", "ewma94", "ewma97", "cc63", "mad21"]
            dr, nb = block_boot(boot, nm, block=1)   # already non-overlapping -> resample dates
            print(f"\n-- h={h} {lab}  n_obs={int(res.loc['park21','n']):,}  n_dates={nb}")
            print("   " + "  ".join(f"{k}:{res.loc[k,'r2_adj']:.4f}" for k in nm))
            for opp in ("ewma94", "cc21", "ewma97", "cc63"):
                d = dr["park21"] - dr[opp]
                print(f"   park21 - {opp:8s} = {d.mean():+.4f}  95% "
                      f"[{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}]  "
                      f"P(better)={np.mean(d>0):.3f}")

    # ================================================================= A2 purge
    print("\n" + "=" * 118)
    print("A2  PURGED WALK-FORWARD (embargo last h train rows: their targets reach into test year)")
    for h in (21,):
        tgt = L.realized_vol_forward(c, h)
        for lab, cols in (("INDEX", idx_cols), ("SINGLE", sng_cols)):
            r0 = store[(h, lab, "base")][0]
            r1, _, _ = wf(E, FEATS, tgt, cols, years, h, purge=h)
            nm = ["park21", "cc21", "ewma94", "ewma97"]
            print(f"\n-- h={h} {lab}")
            for k in nm:
                print(f"   {k:8s} r2_adj base={r0.loc[k,'r2_adj']:.4f} purged={r1.loc[k,'r2_adj']:.4f} "
                      f"delta={r1.loc[k,'r2_adj']-r0.loc[k,'r2_adj']:+.5f}")

    # ================================================================= A5 fragility
    print("\n" + "=" * 118)
    print("A5a FRAGILITY: drop 2008, 2009, 2020 TEST years entirely")
    for h in (5, 21):
        tgt = L.realized_vol_forward(c, h)
        for lab, cols in (("INDEX", idx_cols), ("SINGLE", sng_cols)):
            r0 = store[(h, lab, "base")][0]
            r1, _, boot = wf(E, FEATS, tgt, cols, years, h, drop_crisis=True)
            nm = ["park21", "cc21", "ewma94", "ewma97", "cc63", "mad21"]
            dr, nb = block_boot(boot, nm)
            print(f"\n-- h={h} {lab}  no-crisis n={int(r1.loc['park21','n']):,} (base "
                  f"{int(r0.loc['park21','n']):,})")
            for k in nm:
                print(f"   {k:8s} base={r0.loc[k,'r2_adj']:.4f}  no-crisis={r1.loc[k,'r2_adj']:.4f}"
                      f"  raw={r1.loc[k,'r2_raw']:.4f}  lvl={r1.loc[k,'r2_lvl']:.4f}"
                      f"  corr={r1.loc[k,'corr']:.4f}")
            for opp in ("ewma94", "cc21", "ewma97", "cc63", "mad21"):
                d = dr["park21"] - dr[opp]
                print(f"     park21-{opp:8s}={d.mean():+.4f} P(better)={np.mean(d>0):.3f}")

    print("\n" + "=" * 118)
    print("A5b FRAGILITY: the 'INDEX' cell is 4 near-identical series. Split them.")
    for h in (21,):
        tgt = L.realized_vol_forward(c, h)
        for s in idx_cols:
            res, _, boot = wf(E, FEATS, tgt, [s], years, h)
            nm = ["park21", "cc21", "ewma94", "ewma97", "cc63", "mad21"]
            dr, nb = block_boot(boot, nm, n_boot=1000)
            best = res["r2_adj"].idxmax()
            print(f"\n-- h={h} {s}  n={int(res.loc['park21','n']):,}  best of all={best} "
                  f"({res['r2_adj'].max():.4f})")
            print("   " + "  ".join(f"{k}:{res.loc[k,'r2_adj']:.4f}" for k in nm))
            for opp in ("ewma94", "cc21"):
                d = dr["park21"] - dr[opp]
                print(f"   park21-{opp:8s}={d.mean():+.4f} 95%[{np.percentile(d,2.5):+.4f},"
                      f"{np.percentile(d,97.5):+.4f}] P(better)={np.mean(d>0):.3f}")
        print("\n   corr(SPY,^GSPC) log-ret:",
              float(np.log(c[['SPY','^GSPC']]).diff().corr().iloc[0,1]))

    # ================================================================= A6b stability of multiplier
    print("\n" + "=" * 118)
    print("A6b IS THE 'FIXABLE' MULTIPLIER ACTUALLY STABLE? cc21/park21 geo-ratio by year and by")
    print("    vol quintile. A rescaling fix needs ONE number; drift means it is not just a level.")
    for lab, cols in (("INDEX", idx_cols), ("SINGLE", sng_cols)):
        cc21, pk21 = L.vol_cc(c, 21)[cols], L.vol_parkinson(hi, lo, 21)[cols]
        lr = np.log(cc21.clip(lower=FLOOR) / pk21.clip(lower=FLOOR))
        st = lr.stack(future_stack=True).dropna()
        yrs = st.index.get_level_values(0).year
        g = st.groupby(yrs).agg(["mean", "size"])
        g["ratio"] = np.exp(g["mean"])
        print(f"\n-- {lab} by year (ratio):")
        print("   " + "  ".join(f"{y}:{v:.3f}" for y, v in g["ratio"].items()))
        print(f"   min={g.ratio.min():.3f} max={g.ratio.max():.3f} "
              f"sd={g.ratio.std():.3f} range/mean={100*(g.ratio.max()-g.ratio.min())/g.ratio.mean():.1f}%")
        pk = pk21.stack(future_stack=True).reindex(st.index)
        q = pd.qcut(pk, 5, labels=False)
        gq = st.groupby(q).agg(["mean", "size"])
        print(f"-- {lab} by park21 quintile (ratio): " +
              "  ".join(f"Q{int(k)+1}:{np.exp(v):.3f}(n={int(n):,})"
                        for k, (v, n) in gq.iterrows()))
        # also: does the OOS affine slope b vary a lot year to year?
    print("\n" + "=" * 118)
    print("A6c per-year OOS affine slope b for park21 vs cc21 vs ewma94 (h=21)")
    tgt = L.realized_vol_forward(c, 21)
    for lab, cols in (("INDEX", idx_cols), ("SINGLE", sng_cols)):
        y, X, di = stack(E, FEATS, tgt, cols)
        yr = years[di]
        rows = []
        for yy in sorted(set(years))[5:]:
            tr, te = yr < yy, yr == yy
            if te.sum() < 50 or tr.sum() < 500:
                continue
            rec = {"year": yy}
            for k in ("park21", "cc21", "ewma94"):
                j = FEATS.index(k)
                # IN-SAMPLE (test-year) slope, to see what the prior-year fit was aiming at
                A = np.column_stack([np.ones(te.sum()), X[te, j]])
                b, *_ = np.linalg.lstsq(A, y[te], rcond=None)
                At = np.column_stack([np.ones(tr.sum()), X[tr, j]])
                bt, *_ = np.linalg.lstsq(At, y[tr], rcond=None)
                rec[f"{k}_b_train"] = bt[1]
                rec[f"{k}_b_test"] = b[1]
            rows.append(rec)
        t = pd.DataFrame(rows)
        print(f"\n-- {lab}  (b_train = what was used OOS; b_test = the ideal, IN-SAMPLE)")
        print(t.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
        for k in ("park21", "cc21", "ewma94"):
            print(f"   {k:8s} b_train mean={t[f'{k}_b_train'].mean():.3f} "
                  f"b_test sd={t[f'{k}_b_test'].std():.3f} "
                  f"mean|b_test-b_train|={ (t[f'{k}_b_test']-t[f'{k}_b_train']).abs().mean():.3f}")

    # ================================================================= per-year win table
    print("\n" + "=" * 118)
    print("A5c PER-YEAR OOS r2_adj, h=21: how many years does park21 actually beat ewma94/cc21?")
    for lab in ("INDEX", "SINGLE"):
        _, py, _ = store[(21, lab, "base")]
        sub = py[["year", "n", "park21", "cc21", "ewma94", "ewma97", "cc63", "mad21"]]
        print(f"\n-- {lab}")
        print(sub.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
        print(f"   park21>ewma94 in {int((sub.park21>sub.ewma94).sum())}/{len(sub)} years; "
              f"park21>cc21 in {int((sub.park21>sub.cc21).sum())}/{len(sub)}; "
              f"park21>cc63 in {int((sub.park21>sub.cc63).sum())}/{len(sub)}; "
              f"park21>mad21 in {int((sub.park21>sub.mad21).sum())}/{len(sub)}")


if __name__ == "__main__":
    main()
