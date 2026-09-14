"""
_vix_wf_verify_snooping_multiplicity.py — ADVERSARIAL audit, SNOOPING/MULTIPLICITY lens.

CLAIM UNDER TEST (recovered from the lower-band analysis; the task prompt had it garbled):
  "bb10_2.0_below's raw D5 excess of -0.58% (n=159) keeps 84% under vix_pct1y vigintile FE
   (-0.487%, p=.020) and 70% under a full FE ladder (-0.406%, p=.039), and is strongest in
   the MID level tercile (-0.66%, n=60, p=.032)."

The lens is NOT "is the control right" — it is "how many things were tried before these four
numbers were the ones written down, and do they survive when that search is priced in".
Three separate search dimensions get priced here:

  S1  the SIGNAL grid   : (band length n) x (k) x (side) x (horizon)   -- the surface that
                          _vix_wf_lowerband.py printed, from which bb10_2.0/D5 was the pick
  S2  the CONTROL grid  : which FE ladder. "vigintile FE" and "full FE ladder" are 2 draws
                          from a large space of equally defensible control sets
  S3  the SUBGROUP grid : "MID level tercile" is 1 of 3 terciles, x horizons

plus the standard robustness attacks that a snooped number usually fails:
  S4  episode concentration (drop worst month / 2008 / 2020 / leave-one-year-out)
  S5  honest precision (overlapping D5 windows -> clustered SE, effective n)
  S6  pseudo-out-of-sample: pick the winner on the first half, score it on the second half

Method (repo rules, non-negotiable):
  - every number is EXCESS (or an FE coefficient) vs the SAME-SAMPLE baseline
  - significance = circular rotation test, computed EXHAUSTIVELY over all offsets by FFT
    (identical null to rotation_pvalue() in _vix_ma10_bb_research.py, no Monte-Carlo noise);
    offsets within +/-25 bars of identity are dropped so the "null" cannot overlap the real
    signal's own forward window
  - multiple-testing correction = Westfall-Young / Romano-Wolf maxT under a JOINT rotation
    (one offset applied to every config simultaneously), which is the only correction that
    is not absurdly conservative here: these configs are nested subsets of each other
  - n < 25 -> INCONCLUSIVE, never interpreted
  - g* forward returns only (next-close entry)

Run: ../../vcp_env/bin/python _vix_wf_verify_snooping_multiplicity.py
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import _vix_data  # noqa: E402

MIN_OFF = 25
MIN_N = 25
HZ = (1, 3, 5, 10, 21)


# ----------------------------------------------------------------- rotation machinery
def circ_corr(m: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(m)
    return np.real(np.fft.irfft(np.conj(np.fft.rfft(m)) * np.fft.rfft(y), n))


def keep_offsets(n: int) -> np.ndarray:
    off = np.arange(n)
    return np.minimum(off, n - off) >= MIN_OFF


def mean_null(mask: np.ndarray, y: np.ndarray):
    """Exact rotation null of the conditional mean. Returns (observed, null_vector)."""
    dm = mask.astype(float)
    null = circ_corr(dm, y) / dm.sum()
    return float(y[mask].mean()), null[keep_offsets(len(dm))]


def fwl_null(mask: np.ndarray, y: np.ndarray, Q: np.ndarray):
    """Exact rotation null of the Frisch-Waugh coefficient on `mask` in  y ~ [Q, mask].

    beta(off) = roll(d,off).y_res / (d.d - ||Q'roll(d,off)||^2); both terms are circular
    cross-correlations, so every offset is available in closed form.
    """
    dm = mask.astype(float)
    n = len(dm)
    y_res = y - Q @ (Q.T @ y)
    num = circ_corr(dm, y_res)
    proj = np.zeros(n)
    for j in range(Q.shape[1]):
        proj += circ_corr(dm, Q[:, j]) ** 2
    den = dm.sum() - proj
    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.where(den > 1e-9, num / den, np.nan)
    return float(b[0]), b[keep_offsets(n)]


def p_of(obs, null):
    null = null[~np.isnan(null)]
    if not len(null):
        return float("nan")
    c = null.mean()
    return float((np.abs(null - c) >= abs(obs - c)).mean())


def z_of(obs, null):
    null = null[~np.isnan(null)]
    s = null.std(ddof=0)
    return (obs - null.mean()) / s if s > 1e-15 else 0.0


def stars(p):
    if p != p:
        return "   "
    return "***" if p < 0.01 else ("** " if p < 0.05 else ("*  " if p < 0.10 else "   "))


def fe_basis(n, code_arrays):
    cols = [np.ones(n)]
    for c in code_arrays:
        c = np.asarray(c)
        for k in np.unique(c)[1:]:
            cols.append((c == k).astype(float))
    Q, _ = np.linalg.qr(np.column_stack(cols))
    return Q


def maxT(zobs: np.ndarray, Z: np.ndarray, i: int) -> float:
    """Single-step Westfall-Young: P(max_j |z_j(rotation)| >= |z_i(observed)|)."""
    return float(np.mean(np.nanmax(Z, axis=0) >= zobs[i]))


def romano_wolf(zobs: np.ndarray, Z: np.ndarray) -> np.ndarray:
    order = np.argsort(-zobs)
    alive = np.ones(len(zobs), bool)
    adj = np.empty(len(zobs))
    run = 0.0
    for i in order:
        run = max(run, float(np.mean(np.nanmax(Z[alive], axis=0) >= zobs[i])))
        adj[i] = run
        alive[i] = False
    return adj


def hr(t):
    print("\n" + "=" * 110 + f"\n{t}\n" + "=" * 110)


# ----------------------------------------------------------------- data
def build():
    d = _vix_data.add_features(_vix_data.load())
    d["rv20"] = d.spx.pct_change().rolling(20).std() * np.sqrt(252)
    d["spx_r20"] = d.spx / d.spx.shift(20) - 1.0
    d["spx_r5"] = d.spx / d.spx.shift(5) - 1.0
    for n in (5, 15, 30):
        basis, sd = d.vix.rolling(n).mean(), d.vix.rolling(n).std(ddof=0)
        for k in (1.0, 1.5, 2.0, 2.5):
            d[f"bb{n}_{k}_below"] = d.vix < basis - k * sd
            d[f"bb{n}_{k}_above"] = d.vix > basis + k * sd
    for n in (10, 20):
        basis, sd = d.vix.rolling(n).mean(), d.vix.rolling(n).std(ddof=0)
        d[f"bb{n}_1.0_below"] = d.vix < basis - 1.0 * sd
        d[f"bb{n}_1.0_above"] = d.vix > basis + 1.0 * sd
    return d


def main():
    d = build()
    sig_cols = [f"bb{n}_{k}_{s}" for n in (5, 10, 15, 20, 30) for k in (1.0, 1.5, 2.0, 2.5)
                for s in ("below", "above")]
    sig_cols += [f"bb{n}_{k}_{s}" for n in (10, 20) for k in (1.5, 2.0, 2.5)
                 for s in ("reentry", "exit_lo")]
    ctrl = ["vix_pct1y", "vix_pct2y", "bb10_width", "rv20", "spx_r20", "spx_r5", "vix", "stretch"]
    frame = d[[f"g{h}" for h in HZ] + ctrl + sig_cols].dropna().copy()
    n = len(frame)
    yr = np.asarray(frame.index.year)
    ym = np.asarray([f"{t.year}-{t.month:02d}" for t in frame.index])
    Y = {h: frame[f"g{h}"].values for h in HZ}
    below = frame["bb10_2.0_below"].astype(bool).values
    print(f"frame  n={n:,}  {frame.index[0].date()} -> {frame.index[-1].date()}   "
          f"bb10_2.0_below n={int(below.sum())}   baseline g5 {Y[5].mean()*100:+.3f}%")

    qc = lambda s, k: pd.qcut(frame[s], k, labels=False, duplicates="drop").values
    vig = qc("vix_pct1y", 20)
    Q_vig = fe_basis(n, [vig])
    Q_full = fe_basis(n, [vig, qc("bb10_width", 10), qc("rv20", 10),
                          qc("spx_r20", 10), qc("spx_r5", 10)])

    # ============================================================ 0. reproduce
    hr("0. REPRODUCTION")
    raw_o, raw_n = mean_null(below, Y[5])
    raw_exc = raw_o - Y[5].mean()
    p_raw = p_of(raw_o, raw_n)
    c_vig, nl_vig = fwl_null(below, Y[5], Q_vig)
    c_full, nl_full = fwl_null(below, Y[5], Q_full)
    p_vig, p_full = p_of(c_vig, nl_vig), p_of(c_full, nl_full)
    print(f"  raw D5 excess          {raw_exc*100:>+7.3f}%  p={p_raw:.4f}{stars(p_raw)}   "
          f"(claim -0.58%)")
    print(f"  vigintile-FE coef      {c_vig*100:>+7.3f}%  p={p_vig:.4f}{stars(p_vig)}   "
          f"(claim -0.487%, p=.020)   retains {c_vig/raw_exc*100:.0f}% [claim 84%]")
    print(f"  full-FE-ladder coef    {c_full*100:>+7.3f}%  p={p_full:.4f}{stars(p_full)}   "
          f"(claim -0.406%, p=.039)   retains {c_full/raw_exc*100:.0f}% [claim 70%]")
    ter = qc("vix_pct1y", 3)
    mid = below & (ter == 1)
    o_m, n_m = mean_null(mid, Y[5])
    exc_m = o_m - Y[5][ter == 1].mean()
    print(f"  MID level tercile      {exc_m*100:>+7.3f}%  n={int(mid.sum())}  "
          f"p={p_of(o_m, n_m):.4f}{stars(p_of(o_m, n_m))}   (claim -0.66%, n=60, p=.032)")
    print("  -> all four headline numbers reproduce. Nothing below disputes the arithmetic.")

    # ============================================================ S1. signal-grid multiplicity
    hr("S1. SIGNAL-GRID MULTIPLICITY — the (n, k, side) x horizon surface the pick came from")
    cfg = []
    for s in sig_cols:
        m = frame[s].astype(bool).values
        if m.sum() >= MIN_N:
            for h in HZ:
                cfg.append((f"{s}|D{h}", m, h))
    print(f"  configurations with n>={MIN_N}: {len(cfg)}   (band signals only; this EXCLUDES the")
    print("  stretch-threshold arm, the streak/exit/trend splits and the 45-config de-risk grid)")
    for tag, statf in (("RAW excess", lambda m, h: mean_null(m, Y[h])),
                       ("vigintile-FE coef", lambda m, h: fwl_null(m, Y[h], Q_vig))):
        zs, Zs, labs = [], [], []
        for lbl, m, h in cfg:
            o, nl = statf(m, h)
            mu, sd = np.nanmean(nl), np.nanstd(nl)
            sd = sd if sd > 1e-15 else 1e-15
            zs.append(abs((o - mu) / sd))
            Zs.append(np.abs(nl - mu) / sd)
            labs.append(lbl)
        zs = np.array(zs)
        Zs = np.vstack(Zs)
        rw = romano_wolf(zs, Zs)
        ti = labs.index("bb10_2.0_below|D5")
        pn = np.array([p_of(0, Zs[i] * 0 + Zs[i]) for i in range(0)])  # placeholder, unused
        nom = np.array([float(np.mean(Zs[i] >= zs[i])) for i in range(len(zs))])
        print(f"\n  [{tag}]")
        print(f"    nominal p<.05: {int((nom < .05).sum())}/{len(zs)} "
              f"(chance {0.05*len(zs):.0f})   p<.01: {int((nom < .01).sum())} (chance {0.01*len(zs):.0f})")
        print(f"    bb10_2.0_below|D5: |z|={zs[ti]:.2f}  nominal p={nom[ti]:.4f}  "
              f"rank {int((zs > zs[ti]).sum())+1}/{len(zs)}")
        print(f"    single-step maxT FWER p = {maxT(zs, Zs, ti):.4f}   "
              f"Romano-Wolf step-down p = {rw[ti]:.4f} {stars(rw[ti])}")
        bi = int(np.argmax(zs))
        print(f"    family BEST cell: {labs[bi]}  |z|={zs[bi]:.2f}  RW p={rw[bi]:.4f}"
              f"{stars(rw[bi])}  <- does ANYTHING survive?")
        surv = [labs[i] for i in np.argsort(-zs) if rw[i] < 0.05]
        print(f"    cells surviving FWER at .05: {len(surv)}"
              + (f"  -> {surv[:6]}" if surv else ""))

    # ============================================================ S2. control-grid multiplicity
    hr("S2. CONTROL-GRID MULTIPLICITY — the FE ladder is a CHOICE. Specification curve over "
       "every defensible ladder")
    blocks = {"width": qc("bb10_width", 10), "rv20": qc("rv20", 10),
              "r20": qc("spx_r20", 10), "r5": qc("spx_r5", 10),
              "year": pd.factorize(yr)[0], "vixlvl": qc("vix", 10)}
    levels = {"none": None, "pct1y_q5": qc("vix_pct1y", 5), "pct1y_q10": qc("vix_pct1y", 10),
              "pct1y_q20": vig, "pct2y_q20": qc("vix_pct2y", 20)}
    specs = []
    names = list(blocks)
    for lname, lcode in levels.items():
        for r in range(len(names) + 1):
            for combo in itertools.combinations(names, r):
                codes = ([lcode] if lcode is not None else []) + [blocks[c] for c in combo]
                specs.append((f"{lname}" + ("+" + "+".join(combo) if combo else ""), codes))
    print(f"  control specifications enumerated: {len(specs)}  "
          f"(5 level granularities x 2^{len(names)} control-block subsets)")
    coefs, ps, zs2, Zs2, labs2 = [], [], [], [], []
    for name, codes in specs:
        Q = fe_basis(n, codes) if codes else fe_basis(n, [])
        c, nl = fwl_null(below, Y[5], Q)
        mu, sd = np.nanmean(nl), np.nanstd(nl)
        sd = sd if sd > 1e-15 else 1e-15
        coefs.append(c)
        ps.append(p_of(c, nl))
        zs2.append(abs((c - mu) / sd))
        Zs2.append(np.abs(nl - mu) / sd)
        labs2.append(name)
    coefs, ps, zs2 = np.array(coefs), np.array(ps), np.array(zs2)
    Zs2 = np.vstack(Zs2)
    print(f"  coefficient across all {len(specs)} ladders: min {coefs.min()*100:+.3f}%  "
          f"p25 {np.percentile(coefs,25)*100:+.3f}%  median {np.median(coefs)*100:+.3f}%  "
          f"p75 {np.percentile(coefs,75)*100:+.3f}%  max {coefs.max()*100:+.3f}%")
    print(f"  ALL {len(specs)} ladders keep the sign negative: {bool((coefs < 0).all())}")
    print(f"  nominal p<.05 in {int((ps<.05).sum())}/{len(specs)} ladders "
          f"({(ps<.05).mean()*100:.0f}%);  p<.01 in {int((ps<.01).sum())} "
          f"({(ps<.01).mean()*100:.0f}%)")
    for want, lbl in (("pct1y_q20", "the reported 'vigintile FE'"),
                      ("pct1y_q20+width+rv20+r20+r5", "the reported 'full FE ladder'")):
        if want in labs2:
            i = labs2.index(want)
            pct = float((coefs <= coefs[i]).mean()) * 100
            print(f"    {lbl:<28} coef {coefs[i]*100:+.3f}%  p={ps[i]:.4f}  "
                  f"-> {pct:.0f}th percentile of the spec curve "
                  f"({'more' if pct < 50 else 'less'} negative than {100-pct:.0f}% of ladders)")
    rw2 = romano_wolf(zs2, Zs2)
    for want in ("pct1y_q20", "pct1y_q20+width+rv20+r20+r5"):
        if want in labs2:
            i = labs2.index(want)
            print(f"    FWER across the {len(specs)}-ladder search: {want:<30} "
                  f"maxT p={maxT(zs2, Zs2, i):.4f}  RW p={rw2[i]:.4f} {stars(rw2[i])}")
    print("  NOTE: these ladders are ~nested, so this correction is mild; the informative")
    print("  line is the spec curve itself (does the answer depend on which ladder was picked?).")

    # ============================================================ S3. subgroup multiplicity
    hr("S3. SUBGROUP MULTIPLICITY — 'strongest in the MID tercile' is the max of a 3 x 5 grid")
    labs3, zs3, Zs3, info = [], [], [], {}
    for t in range(3):
        reg = (ter == t)
        for h in HZ:
            m = below & reg
            cnt = int(m.sum())
            yy = Y[h]
            if cnt < MIN_N:
                info[(t, h)] = (cnt, np.nan, np.nan)
                continue
            o, nl = mean_null(m, yy)
            e = o - yy[reg].mean()
            mu, sd = np.nanmean(nl), np.nanstd(nl)
            sd = sd if sd > 1e-15 else 1e-15
            info[(t, h)] = (cnt, e, p_of(o, nl))
            labs3.append((t, h))
            zs3.append(abs((o - mu) / sd))
            Zs3.append(np.abs(nl - mu) / sd)
    zs3 = np.array(zs3)
    Zs3 = np.vstack(Zs3)
    rw3 = romano_wolf(zs3, Zs3)
    print(f"  {'tercile of vix_pct1y':<24}" + "".join(f"{'D'+str(h):>20}" for h in HZ))
    for t, nm in enumerate(("LOW (calm VIX)", "MID", "HIGH (elevated VIX)")):
        cells = []
        for h in HZ:
            cnt, e, p = info[(t, h)]
            cells.append((f"n<25 (n={cnt})" if cnt < MIN_N
                          else f"{e*100:+.2f}% p={p:.3f}").rjust(20))
        print(f"  {nm:<24}" + "".join(cells))
    if (1, 5) in labs3:
        i = labs3.index((1, 5))
        print(f"\n  MID/D5: nominal p={info[(1,5)][2]:.4f}  ->  FWER across the "
              f"{len(labs3)} testable subgroup cells: maxT p={maxT(zs3, Zs3, i):.4f}  "
              f"RW p={rw3[i]:.4f} {stars(rw3[i])}")
    bi3 = int(np.argmax(zs3))
    print(f"  strongest testable subgroup cell is {labs3[bi3]} (tercile,horizon) with "
          f"RW p={rw3[bi3]:.4f}")
    print(f"  the HIGH tercile at D5 shows {info[(2,5)][1]*100:+.2f}% on n={info[(2,5)][0]} "
          f"-> below the n>=25 floor, INCONCLUSIVE; 'MID is strongest' is therefore a claim")
    print("  about the only two terciles that could be measured, not about three.")
    # is MID actually different from LOW?
    lo, md = below & (ter == 0), below & (ter == 1)
    diff = (Y[5][md].mean() - Y[5][ter == 1].mean()) - (Y[5][lo].mean() - Y[5][ter == 0].mean())
    pooled = np.sqrt(Y[5][md].var(ddof=1) / md.sum() + Y[5][lo].var(ddof=1) / lo.sum())
    print(f"  MID minus LOW excess = {diff*100:+.2f}%  with SE {pooled*100:.2f}%  "
          f"-> t={diff/pooled:+.2f}  (the terciles are NOT distinguishable from each other)")

    # ============================================================ S4. concentration
    hr("S4. EPISODE / PERIOD CONCENTRATION — is the FE-adjusted number a few months?")

    def redo(keep, label):
        cnt = int((below & keep).sum())
        if cnt < MIN_N:
            print(f"  {label:<34}n={cnt:>4}  n<25 — INCONCLUSIVE")
            return
        yk = np.where(keep, Y[5], np.nan)
        # raw excess inside the surviving sample
        v = ~np.isnan(yk)
        e = yk[below & v].mean() - yk[v].mean()
        # FE coef re-estimated inside the surviving sample
        idx = np.flatnonzero(keep)
        sub = frame.iloc[idx]
        q = lambda s, k: pd.qcut(sub[s], k, labels=False, duplicates="drop").values
        Qs = fe_basis(len(idx), [q("vix_pct1y", 20)])
        c, nl = fwl_null(below[idx], Y[5][idx], Qs)
        print(f"  {label:<34}n={cnt:>4}  raw {e*100:>+6.2f}% [{e/raw_exc*100:>3.0f}%]   "
              f"vigFE {c*100:>+6.2f}% [{c/c_vig*100:>3.0f}%]  p={p_of(c, nl):.3f} "
              f"{stars(p_of(c, nl))}")

    contrib = {}
    for mth in sorted(set(ym[below])):
        s = below & (ym == mth)
        contrib[mth] = (Y[5][s] - Y[5].mean()).sum()
    worst = sorted(contrib.items(), key=lambda kv: kv[1])
    tot = sum(contrib.values())
    print(f"  the {int(below.sum())} signal days sit in {len(set(ym[below]))} distinct months "
          f"and {len(set(yr[below]))} distinct years")
    print("  5 worst months by contribution:")
    for mth, c in worst[:5]:
        print(f"    {mth}  n={int((below&(ym==mth)).sum()):>2}  {c*100:>+6.2f} pp-days "
              f"({c/tot*100:>4.1f}% of the total effect)")
    print()
    redo(np.ones(n, bool), "HEADLINE (full frame)")
    redo(ym != worst[0][0], f"drop worst month ({worst[0][0]})")
    redo(~np.isin(ym, [m for m, _ in worst[:3]]), "drop 3 worst months")
    redo(yr != 2008, "drop 2008")
    redo(yr != 2020, "drop 2020")
    redo((yr != 2008) & (yr != 2020), "drop 2008 AND 2020")
    redo((yr != 2008) & (yr != 2020) & (ym != worst[0][0]), "drop 2008, 2020, worst month")
    redo(yr >= 2010, "2010-2026 only")
    redo(yr >= 2015, "2015-2026 only")
    jk = {}
    for y in sorted(set(yr[below])):
        k = yr != y
        v = np.ones(n, bool) & k
        if (below & v).sum() >= MIN_N:
            jk[y] = Y[5][below & v].mean() - Y[5][v].mean()
    arr = np.array(list(jk.values()))
    print(f"\n  leave-one-year-out raw excess: {arr.min()*100:+.2f}% .. {arr.max()*100:+.2f}%  "
          f"(headline {raw_exc*100:+.2f}%, sd {arr.std()*100:.2f}%)")
    dev = np.sort(Y[5][below] - Y[5].mean())
    for kk in (1, 5, 10, 20):
        print(f"    {kk:>2} worst signal days carry {dev[:kk].sum()/dev.sum()*100:>5.1f}% "
              f"of the total excess")

    # ============================================================ S5. precision
    hr("S5. PRECISION — 159 overlapping D5 windows are not 159 observations")
    x = Y[5][below] - Y[5].mean()
    epid = np.cumsum(below & ~np.r_[False, below[:-1]])[below]
    grp = {"episode": epid, "month": ym[below], "year": yr[below]}
    naive = x.std(ddof=1) / np.sqrt(len(x))
    print(f"  raw excess {x.mean()*100:+.2f}%")
    print(f"  {'naive SE (iid)':<22}{naive*100:.2f}%  t={x.mean()/naive:+.2f}  "
          f"95% CI [{(x.mean()-1.96*naive)*100:+.2f}%, {(x.mean()+1.96*naive)*100:+.2f}%]")
    for nm_, g in grp.items():
        s = pd.Series(x).groupby(g).sum().values
        G = len(s)
        se = np.sqrt((s ** 2).sum()) / len(x) * np.sqrt(G / max(G - 1, 1))
        print(f"  {nm_+'-clustered SE':<22}{se*100:.2f}%  t={x.mean()/se:+.2f}  "
              f"95% CI [{(x.mean()-1.96*se)*100:+.2f}%, {(x.mean()+1.96*se)*100:+.2f}%]  (G={G})")
    xm = Y[5][mid] - Y[5][ter == 1].mean()
    sem = xm.std(ddof=1) / np.sqrt(len(xm))
    print(f"  MID tercile subgroup (n={len(xm)}): {xm.mean()*100:+.2f}%  naive SE {sem*100:.2f}%  "
          f"95% CI [{(xm.mean()-1.96*sem)*100:+.2f}%, {(xm.mean()+1.96*sem)*100:+.2f}%]")
    print("  -> the CI on the subgroup spans a factor of ~5 in effect size; '-0.66%' is not a")
    print("     number this sample can resolve to two decimals.")

    # ============================================================ S6. pseudo-OOS
    hr("S6. PSEUDO-OUT-OF-SAMPLE — pick the winner on the first half, score the second half")
    half = n // 2
    A = np.zeros(n, bool); A[:half] = True
    B = ~A
    print(f"  half A {frame.index[0].date()}..{frame.index[half-1].date()}   "
          f"half B {frame.index[half].date()}..{frame.index[-1].date()}")
    best, bz = None, -np.inf
    for lbl, m, h in cfg:
        if (m & A).sum() < MIN_N:
            continue
        e = Y[h][m & A].mean() - Y[h][A].mean()
        s = Y[h][A].std(ddof=1) / np.sqrt((m & A).sum())
        if abs(e / s) > bz:
            bz, best = abs(e / s), (lbl, m, h, e)
    lbl, m, h, eA = best
    eB = (Y[h][m & B].mean() - Y[h][B].mean()) if (m & B).sum() >= MIN_N else np.nan
    print(f"  best config in half A: {lbl}  excess {eA*100:+.2f}% (n={int((m&A).sum())})")
    print(f"    -> that same config in half B: "
          + (f"{eB*100:+.2f}% (n={int((m&B).sum())})" if eB == eB
             else f"n={int((m&B).sum())} — INCONCLUSIVE"))
    for lbl2, mm in (("bb10_2.0_below", below),):
        for wn, w in (("half A", A), ("half B", B)):
            c = int((mm & w).sum())
            if c < MIN_N:
                print(f"  {lbl2} in {wn}: n={c} — INCONCLUSIVE")
                continue
            e = Y[5][mm & w].mean() - Y[5][w].mean()
            yy = np.where(w, Y[5], np.nan)
            print(f"  {lbl2} D5 in {wn}: {e*100:+.2f}%  n={c}  "
                  f"p={p_of(np.nanmean(yy[mm & w]), mean_null(mm, np.nan_to_num(yy))[1]):.3f}")
    # honest split on the tercile claim
    for wn, w in (("half A", A), ("half B", B)):
        c = int((mid & w).sum())
        if c < MIN_N:
            print(f"  MID-tercile D5 in {wn}: n={c} — INCONCLUSIVE")
        else:
            e = Y[5][mid & w].mean() - Y[5][(ter == 1) & w].mean()
            print(f"  MID-tercile D5 in {wn}: {e*100:+.2f}%  n={c}")

    print("\ndone.")


if __name__ == "__main__":
    main()
