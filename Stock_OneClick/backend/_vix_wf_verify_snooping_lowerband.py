"""
_vix_wf_verify_snooping_lowerband.py — ADVERSARIAL / DATA-SNOOPING audit of the claim:

  "bb10_2.0_below's raw D5 excess of -0.58% (n=159) keeps 84% under vix_pct1y vigintile FE
   (-0.487%, p=.020) and 70% under a full FE ladder (-0.406%, p=.039), and is strongest in
   the MID level tercile (-0.66%, n=60, p=.032)."

Lens: SNOOPING. Controls fix CONFOUNDING, not MULTIPLICITY. The question here is only:
how big was the family this number was picked out of, and does it survive when that family
is priced in, when the worst episodes are removed, and when precision is measured honestly?

Sections
  0. reproduce the headline numbers (same frame, same estimator)
  1. enumerate the ACTUAL search family (band grid x horizons) and count nominal winners
  2. Romano-Wolf step-down FWER correction over the family, JOINT circular rotation null
     (same offset for every config, so cross-signal dependence is preserved)
  3. episode / period concentration: drop worst month, drop 2008, drop 2020, drop each year
  4. honest precision: overlapping windows -> effective independent blocks, cluster SE, CI
  5. the MID-tercile subgroup: multiplicity across the 3x5 subgroup grid it was picked from
  6. pseudo out-of-sample: choose on the first half, score on the second

Method notes
  - Rotation null is computed EXHAUSTIVELY over every non-trivial offset via FFT circular
    cross-correlation (identical null to _vix_ma10_bb_research.rotation_pvalue, but exact
    instead of 5,000 Monte-Carlo draws). Offsets within +/-25 bars of identity are excluded
    so the "null" does not overlap the real signal's own forward windows.
  - FE-adjusted coefficients via Frisch-Waugh, and the rotation null of the FWL coefficient
    is also computed exactly in closed form (see fwl_rot_all).
  - Every number is EXCESS over the same-sample unconditional mean. n reported everywhere.
  - n < 25 -> printed as INCONCLUSIVE, never interpreted.

Run: ../../vcp_env/bin/python _vix_wf_verify_snooping_lowerband.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import _vix_data  # noqa: E402

MIN_OFF = 25          # exclude near-identity rotations (forward window is up to 21 bars)
MIN_N = 25            # repo rule: subgroups below this are inconclusive
HORIZONS = (1, 3, 5, 10, 21)


# ------------------------------------------------------------------ rotation machinery
def circ_corr(m: np.ndarray, y: np.ndarray) -> np.ndarray:
    """arr[off] = sum_t np.roll(m, off)[t] * y[t], for every off in 0..n-1, via FFT."""
    n = len(m)
    return np.real(np.fft.irfft(np.conj(np.fft.rfft(m)) * np.fft.rfft(y), n))


def offset_mask(n: int) -> np.ndarray:
    off = np.arange(n)
    return (np.minimum(off, n - off) >= MIN_OFF)


def mean_rot_all(mask: np.ndarray, y: np.ndarray):
    """Exact rotation null of the CONDITIONAL MEAN. Returns (obs, null_array)."""
    d = mask.astype(float)
    n = len(d)
    s = circ_corr(d, y)
    cnt = d.sum()
    null = s / cnt
    keep = offset_mask(n)
    return float(y[mask].mean()), null[keep]


def fwl_rot_all(mask: np.ndarray, y: np.ndarray, Q: np.ndarray | None):
    """Exact rotation null of the Frisch-Waugh coefficient on `mask` in y ~ [Q, mask].

    num(off) = roll(d,off).y_res           (y_res already orthogonal to col(Q))
    den(off) = d.d - sum_j (roll(d,off).q_j)^2
    Both are circular cross-correlations -> closed form for all offsets.
    """
    d = mask.astype(float)
    n = len(d)
    if Q is None or Q.shape[1] == 0:
        y_res = y - y.mean()
        ones = np.ones(n)
        num = circ_corr(d, y_res)
        den = d.sum() - circ_corr(d, ones / np.sqrt(n)) ** 2
    else:
        y_res = y - Q @ (Q.T @ y)
        num = circ_corr(d, y_res)
        proj = np.zeros(n)
        for j in range(Q.shape[1]):
            proj += circ_corr(d, Q[:, j]) ** 2
        den = d.sum() - proj
    with np.errstate(divide="ignore", invalid="ignore"):
        coefs = np.where(den > 1e-9, num / den, np.nan)
    keep = offset_mask(n)
    return float(coefs[0]), coefs[keep]


def p_from_null(obs: float, null: np.ndarray) -> float:
    null = null[~np.isnan(null)]
    if not len(null):
        return float("nan")
    base = null.mean()
    return float((np.abs(null - base) >= abs(obs - base)).mean())


def zstat(obs: float, null: np.ndarray) -> float:
    null = null[~np.isnan(null)]
    sd = null.std(ddof=0)
    return (obs - null.mean()) / sd if sd > 1e-15 else 0.0


def stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "   "
    return "***" if p < 0.01 else ("** " if p < 0.05 else ("*  " if p < 0.10 else "   "))


def fe_Q(n: int, code_arrays) -> np.ndarray:
    cols = [np.ones(n)]
    for codes in code_arrays:
        codes = np.asarray(codes)
        for k in np.unique(codes)[1:]:
            cols.append((codes == k).astype(float))
    X = np.column_stack(cols)
    Q, _ = np.linalg.qr(X)
    return Q


def hr(title, ch="="):
    print("\n" + ch * 104)
    print(title)
    print(ch * 104)


# ------------------------------------------------------------------ data
def build():
    d = _vix_data.add_features(_vix_data.load())
    d["rv20"] = d.spx.pct_change().rolling(20).std() * np.sqrt(252)
    d["spx_r20"] = d.spx / d.spx.shift(20) - 1.0
    d["spx_r5"] = d.spx / d.spx.shift(5) - 1.0
    # extra band lengths used by _vix_wf_lowerband.py's (n,k) surface -> part of the family
    for n in (5, 15, 30):
        basis = d.vix.rolling(n).mean()
        sd = d.vix.rolling(n).std(ddof=0)
        for k in (1.0, 1.5, 2.0, 2.5):
            t = f"bb{n}_{k}"
            d[f"{t}_above"] = d.vix > basis + k * sd
            d[f"{t}_below"] = d.vix < basis - k * sd
    for n in (10, 20):
        basis = d.vix.rolling(n).mean()
        sd = d.vix.rolling(n).std(ddof=0)
        d[f"bb{n}_1.0_above"] = d.vix > basis + 1.0 * sd
        d[f"bb{n}_1.0_below"] = d.vix < basis - 1.0 * sd
    return d


def main():
    d = build()
    print(f"panel {len(d):,} rows  {d.index[0].date()} -> {d.index[-1].date()}")

    # ============================================================ 0. reproduce headline
    hr("0. REPRODUCTION — headline numbers on the claim's own frame (g5, vix_pct1y available)")
    keep = ["g5", "vix_pct1y", "bb10_2.0_below", "bb10_2.0_above", "bb20_2.0_below",
            "bb10_width", "rv20", "spx_r20", "spx_r5", "stretch", "vix"]
    a = d[keep].dropna().copy()
    y = a.g5.values
    n = len(y)
    below = a["bb10_2.0_below"].astype(bool).values
    base = y.mean()
    print(f"  frame n={n:,}  {a.index[0].date()} -> {a.index[-1].date()}  "
          f"baseline g5 {base*100:+.3f}%   below n={int(below.sum())}")

    QQ = lambda s, k: pd.qcut(a[s], k, labels=False, duplicates="drop").values
    vig = QQ("vix_pct1y", 20)
    Q_vig = fe_Q(n, [vig])
    Q_full = fe_Q(n, [vig, QQ("bb10_width", 10), QQ("rv20", 10),
                      QQ("spx_r20", 10), QQ("spx_r5", 10)])

    raw, raw_null = mean_rot_all(below, y)
    print(f"  raw excess                    {(raw-base)*100:>+8.3f}%   p={p_from_null(raw, raw_null):.4f}"
          f"{stars(p_from_null(raw, raw_null))}  (claim: -0.58%)")
    c1, n1 = fwl_rot_all(below, y, Q_vig)
    print(f"  vix_pct1y vigintile FE coef   {c1*100:>+8.3f}%   p={p_from_null(c1, n1):.4f}"
          f"{stars(p_from_null(c1, n1))}  (claim: -0.487%, p=.020)")
    c2, n2 = fwl_rot_all(below, y, Q_full)
    print(f"  full FE ladder coef           {c2*100:>+8.3f}%   p={p_from_null(c2, n2):.4f}"
          f"{stars(p_from_null(c2, n2))}  (claim: -0.406%, p=.039)")
    print(f"  -> retention vs raw: vigintile {c1/(raw-base)*100:.0f}%, full ladder "
          f"{c2/(raw-base)*100:.0f}%   [claim said 84% / 70%]")

    # ============================================================ 1. the search family
    hr("1. HOW BIG WAS THE SEARCH FAMILY? — every (band length, k, side) x horizon actually tried")
    # frame where ALL horizons exist, so the family is scored on one common sample
    hcols = [f"g{h}" for h in HORIZONS]
    fam_keep = hcols + ["vix_pct1y", "stretch"]
    sig_names = []
    for bn in (5, 10, 15, 20, 30):
        for k in (1.0, 1.5, 2.0, 2.5):
            for side in ("below", "above"):
                col = f"bb{bn}_{k}_{side}"
                if col in d.columns:
                    sig_names.append(col)
    for bn in (10, 20):
        for k in (1.5, 2.0, 2.5):
            for side in ("reentry", "exit_lo"):
                sig_names.append(f"bb{bn}_{k}_{side}")
    fam = d[fam_keep + sig_names].dropna().copy()
    yf = {h: fam[f"g{h}"].values for h in HORIZONS}
    nf = len(fam)
    vigf = pd.qcut(fam.vix_pct1y, 20, labels=False, duplicates="drop").values
    Qf = fe_Q(nf, [vigf])
    print(f"  common frame n={nf:,}  {fam.index[0].date()} -> {fam.index[-1].date()}")
    print(f"  band signals enumerated: {len(sig_names)}  x  {len(HORIZONS)} horizons "
          f"= {len(sig_names)*len(HORIZONS)} configurations")
    # stretch-threshold family (the other arm of _vix_ma10_bb_research)
    stretch_sigs = {}
    for t in (0.05, 0.10, 0.15, 0.20, 0.25):
        stretch_sigs[f"stretch>+{int(t*100)}%"] = (fam.stretch.values > t)
    for t in (0.05, 0.10, 0.15, 0.20):
        stretch_sigs[f"stretch<-{int(t*100)}%"] = (fam.stretch.values < -t)
    print(f"  plus stretch thresholds: {len(stretch_sigs)} x {len(HORIZONS)} = "
          f"{len(stretch_sigs)*len(HORIZONS)} configurations")

    configs = []   # (label, mask, horizon)
    for s in sig_names:
        m = fam[s].astype(bool).values
        if m.sum() >= MIN_N:
            for h in HORIZONS:
                configs.append((f"{s}|D{h}", m, h))
    for lbl, m in stretch_sigs.items():
        if m.sum() >= MIN_N:
            for h in HORIZONS:
                configs.append((f"{lbl}|D{h}", m, h))
    print(f"  configurations with n>={MIN_N}: {len(configs)}")

    rows = []
    for lbl, m, h in configs:
        yy = yf[h]
        obs, null = mean_rot_all(m, yy)
        p_raw = p_from_null(obs, null)
        c, cn = fwl_rot_all(m, yy, Qf)
        rows.append(dict(label=lbl, n=int(m.sum()), h=h,
                         raw_exc=obs - yy.mean(), p_raw=p_raw, z_raw=zstat(obs, null),
                         fe_coef=c, p_fe=p_from_null(c, cn), z_fe=zstat(c, cn)))
    R = pd.DataFrame(rows)
    for tag, pc, zc in (("RAW excess", "p_raw", "z_raw"), ("level-FE coef", "p_fe", "z_fe")):
        k05 = int((R[pc] < 0.05).sum())
        k01 = int((R[pc] < 0.01).sum())
        print(f"\n  [{tag}] nominal winners in the family of {len(R)}:")
        print(f"     p<.05: {k05}  (expected by pure chance {0.05*len(R):.1f})   "
              f"p<.01: {k01}  (expected {0.01*len(R):.1f})")
        tgt = R[R.label == "bb10_2.0_below|D5"].iloc[0]
        rank = int((R[zc].abs() > abs(tgt[zc])).sum()) + 1
        print(f"     bb10_2.0_below|D5 nominal p={tgt[pc]:.4f}, |z|={abs(tgt[zc]):.2f} "
              f"-> rank {rank} of {len(R)} by |z|")
    print("\n  top 12 of the family by |z| (level-FE coef):")
    top = R.reindex(R.z_fe.abs().sort_values(ascending=False).index).head(12)
    print(f"    {'config':<30}{'n':>6}{'FE coef':>11}{'p':>8}{'|z|':>7}")
    for _, r in top.iterrows():
        print(f"    {r.label:<30}{r.n:>6}{r.fe_coef*100:>+10.3f}%{r.p_fe:>8.3f}{abs(r.z_fe):>7.2f}")

    # ============================================================ 2. Romano-Wolf FWER
    hr("2. MULTIPLE-TESTING CORRECTION — Romano-Wolf step-down, JOINT rotation null "
       "(one offset applied to every config at once)")

    def rw_adjust(sub: pd.DataFrame, statfun) -> pd.DataFrame:
        """sub: rows of the family. statfun(label,mask,h) -> (obs, null_array).
        Returns sub with rw_p column (step-down FWER-adjusted)."""
        nulls, zobs = [], []
        for lbl, m, h in configs:
            if lbl not in set(sub.label):
                continue
            obs, null = statfun(lbl, m, h)
            mu, sd = np.nanmean(null), np.nanstd(null)
            sd = sd if sd > 1e-15 else 1e-15
            nulls.append(np.abs(null - mu) / sd)
            zobs.append(abs((obs - mu) / sd))
        Z = np.vstack(nulls)                       # (k configs, n offsets)
        zobs = np.array(zobs)
        labels = [lbl for lbl, m, h in configs if lbl in set(sub.label)]
        order = np.argsort(-zobs)
        alive = np.ones(len(zobs), bool)
        radj = np.empty(len(zobs))
        running = 0.0
        for i in order:
            maxz = np.nanmax(Z[alive], axis=0)
            p = float((maxz >= zobs[i]).mean())
            running = max(running, p)              # enforce monotonicity
            radj[i] = running
            alive[i] = False
        return pd.DataFrame({"label": labels, "rw_p": radj, "z": zobs})

    masks_by_label = {lbl: (m, h) for lbl, m, h in configs}

    def stat_raw(lbl, m, h):
        return mean_rot_all(m, yf[h])

    def stat_fe(lbl, m, h):
        return fwl_rot_all(m, yf[h], Qf)

    families = {
        "A. below-band variants only (14 sigs x 5 h)":
            [c for c in R.label if "_below|" in c],
        "B. all band signals (below/above/reentry/exit) x 5 h":
            [c for c in R.label if c.startswith("bb")],
        "C. everything searched incl. stretch thresholds":
            list(R.label),
    }
    for famname, labs in families.items():
        sub = R[R.label.isin(labs)]
        for tag, sf in (("RAW", stat_raw), ("level-FE", stat_fe)):
            adj = rw_adjust(sub, sf)
            t = adj[adj.label == "bb10_2.0_below|D5"].iloc[0]
            nrej = int((adj.rw_p < 0.05).sum())
            print(f"  {famname:<52} k={len(sub):>3} [{tag:<8}]  "
                  f"bb10_2.0_below|D5 FWER p={t.rw_p:.3f}{stars(t.rw_p)}  "
                  f"| whole family: {nrej} config(s) survive FWER .05")
    # Bonferroni / BH for reference on the level-FE arm
    pv = np.sort(R.p_fe.values)
    m_ = len(pv)
    bh = pv <= (np.arange(1, m_ + 1) / m_) * 0.05
    tgt = R[R.label == "bb10_2.0_below|D5"].iloc[0]
    print(f"\n  reference: Bonferroni over k={m_}: bb10_2.0_below|D5 adj p = "
          f"{min(1.0, tgt.p_fe*m_):.3f}")
    print(f"  reference: Benjamini-Hochberg FDR .05 over k={m_}: "
          f"{int(bh.sum())} discoveries; threshold p<= "
          f"{(pv[bh].max() if bh.any() else 0):.4f}; target p={tgt.p_fe:.4f} "
          f"-> {'PASSES' if tgt.p_fe <= (pv[bh].max() if bh.any() else 0) else 'FAILS'}")

    # ============================================================ 3. concentration
    hr("3. IS IT A HANDFUL OF EPISODES? — drop worst month, drop 2008, drop 2020, drop each year")
    idx = np.flatnonzero(below)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    groups = np.split(idx, breaks + 1)
    ym = a.index.to_period("M")
    print(f"  {int(below.sum())} below-band days in {len(groups)} runs, "
          f"{len(set(ym[below]))} distinct calendar months, "
          f"{len(set(a.index.year[below]))} distinct years")

    def recompute(keepmask_days: np.ndarray, label: str, Qmat=Q_vig):
        """Drop rows entirely (keepmask_days=True to keep) and refit FE + rotation on the subframe."""
        yy = y[keepmask_days]
        sub_below = below[keepmask_days]
        if sub_below.sum() < MIN_N:
            print(f"  {label:<48} n_below={int(sub_below.sum()):>4}   n<{MIN_N} INCONCLUSIVE")
            return np.nan, np.nan
        vv = pd.qcut(pd.Series(a.vix_pct1y.values[keepmask_days]), 20,
                     labels=False, duplicates="drop").values
        Qs = fe_Q(len(yy), [vv])
        c, nl = fwl_rot_all(sub_below, yy, Qs)
        p = p_from_null(c, nl)
        rawx = yy[sub_below].mean() - yy.mean()
        print(f"  {label:<48} n_below={int(sub_below.sum()):>4}  raw exc {rawx*100:>+7.3f}%  "
              f"FE coef {c*100:>+7.3f}%  p={p:.3f} {stars(p)}")
        return c, p

    c_full, p_full = recompute(np.ones(n, bool), "FULL SAMPLE (reference)")

    # worst single calendar month for the signal
    contrib = {}
    for m_ in sorted(set(ym[below])):
        sel = below & np.asarray(ym == m_)
        contrib[m_] = (sel.sum(), y[sel].mean())
    worst_m = min(contrib, key=lambda k: contrib[k][1] * contrib[k][0])
    print(f"\n  worst month by total contribution: {worst_m}  "
          f"(n={contrib[worst_m][0]} below-days, mean g5 {contrib[worst_m][1]*100:+.2f}%)")
    top5 = sorted(contrib.items(), key=lambda kv: kv[1][0] * kv[1][1])[:5]
    print("  5 most damaging months: " +
          ", ".join(f"{k} n={v[0]} {v[1]*100:+.2f}%" for k, v in top5))
    recompute(~np.asarray(ym == worst_m), f"DROP worst month ({worst_m})")
    m3 = np.ones(n, bool)
    for k, _ in top5[:3]:
        m3 &= ~np.asarray(ym == k)
    recompute(m3, "DROP 3 most damaging months")
    m5 = np.ones(n, bool)
    for k, _ in top5:
        m5 &= ~np.asarray(ym == k)
    recompute(m5, "DROP 5 most damaging months")

    yrs = a.index.year
    recompute(yrs != 2008, "DROP 2008")
    recompute(yrs != 2020, "DROP 2020")
    recompute((yrs != 2008) & (yrs != 2020), "DROP 2008 and 2020")
    recompute(~np.isin(yrs, [2007, 2008, 2009]), "DROP the GFC window 2007-2009")
    recompute(~np.isin(yrs, [2008, 2020, 1998, 2011]), "DROP 1998, 2008, 2011, 2020 (crisis yrs)")

    print("\n  leave-one-YEAR-out (FE coef, only years that contain >=1 below-day):")
    line = []
    jk = {}
    for yr in sorted(set(yrs[below])):
        yy = y[yrs != yr]
        sb = below[yrs != yr]
        if sb.sum() < MIN_N:
            continue
        vv = pd.qcut(pd.Series(a.vix_pct1y.values[yrs != yr]), 20, labels=False,
                     duplicates="drop").values
        c = fwl_rot_all(sb, yy, fe_Q(len(yy), [vv]))[0]
        jk[yr] = c
        line.append(f"{yr}:{c*100:+.2f}")
    print("    " + "  ".join(line))
    jkv = np.array(list(jk.values()))
    print(f"    range {jkv.min()*100:+.3f}% .. {jkv.max()*100:+.3f}%  "
          f"sign flips: {int((jkv > 0).sum())} of {len(jkv)}")

    # per-year contribution share: how concentrated is the total?
    tot = 0.0
    per_year = {}
    for yr in sorted(set(yrs[below])):
        sel = below & (yrs == yr)
        per_year[yr] = sel.sum() * (y[sel].mean() - base)
        tot += per_year[yr]
    srt = sorted(per_year.items(), key=lambda kv: kv[1])
    cum = np.cumsum([v for _, v in srt]) / tot
    print(f"\n  concentration of the TOTAL raw effect (sum of per-day excess = {tot*100:.2f} pp-days):")
    print(f"    worst 1 year  = {cum[0]*100:5.1f}% of the total  ({srt[0][0]})")
    print(f"    worst 2 years = {cum[1]*100:5.1f}%  ({srt[0][0]}, {srt[1][0]})")
    print(f"    worst 3 years = {cum[2]*100:5.1f}%  ({', '.join(str(k) for k,_ in srt[:3])})")
    print(f"    worst 5 years = {cum[4]*100:5.1f}%")
    nyr = len(srt)
    print(f"    (uniform would give {1/nyr*100:.1f}% / {2/nyr*100:.1f}% / {3/nyr*100:.1f}% / "
          f"{5/nyr*100:.1f}% across {nyr} years)")

    # ============================================================ 4. precision
    hr("4. PRECISION — do 159 overlapping days support a +/-0.5% claim?")
    pos = np.flatnonzero(below)
    blocks, cur = [], [pos[0]]
    for p_ in pos[1:]:
        if p_ - cur[-1] <= 5:        # g5 windows overlap within 5 bars
            cur.append(p_)
        else:
            blocks.append(cur)
            cur = [p_]
    blocks.append(cur)
    print(f"  below-band days: {len(pos)}")
    print(f"  non-overlapping 5d blocks (gap > 5 bars): {len(blocks)}")
    print(f"  distinct months: {len(set(ym[below]))}   distinct quarters: "
          f"{len(set(a.index.to_period('Q')[below]))}   distinct years: {len(set(yrs[below]))}")
    blk_means = np.array([y[b].mean() for b in blocks])
    blk_w = np.array([len(b) for b in blocks], float)
    wm = float((blk_means * blk_w).sum() / blk_w.sum())
    se_cluster = float(np.sqrt(((blk_w ** 2) * (blk_means - wm) ** 2).sum()) / blk_w.sum())
    print(f"  block-clustered mean {wm*100:+.3f}%  cluster SE {se_cluster*100:.3f}%  "
          f"-> 95% CI on the RAW mean [{(wm-1.96*se_cluster)*100:+.3f}%, "
          f"{(wm+1.96*se_cluster)*100:+.3f}%]")
    print(f"  raw EXCESS {(raw-base)*100:+.3f}%  -> 95% CI "
          f"[{(wm-base-1.96*se_cluster)*100:+.3f}%, {(wm-base+1.96*se_cluster)*100:+.3f}%]")
    sd_fe = np.nanstd(n1)
    print(f"  rotation-null SD of the FE coef: {sd_fe*100:.3f}%  -> implied 95% band "
          f"+/-{1.96*sd_fe*100:.3f}%; observed coef {c1*100:+.3f}%")
    print(f"  share of below-band days that are the FIRST day of a run: "
          f"{sum(1 for g in groups if len(g))}/{int(below.sum())} runs -> mean run length "
          f"{below.sum()/len(groups):.2f} days")

    # block bootstrap over episodes for the FE coefficient
    rng = np.random.default_rng(20260909)
    Qb = Q_vig
    y_res_b = y - Qb @ (Qb.T @ y)
    boot = np.empty(4000)
    for i in range(4000):
        pick = rng.integers(0, len(blocks), len(blocks))
        mm = np.zeros(n, bool)
        for j in pick:
            mm[blocks[j]] = True
        dvec = mm.astype(float)
        dr = dvec - Qb @ (Qb.T @ dvec)
        den = float(dr @ dr)
        boot[i] = (dr @ y_res_b) / den if den > 1e-9 else np.nan
    boot = boot[~np.isnan(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  episode block-bootstrap of the FE coef (4000 draws, resample the "
          f"{len(blocks)} blocks): 95% CI [{lo*100:+.3f}%, {hi*100:+.3f}%]  "
          f"P(coef>0) = {(boot > 0).mean()*100:.1f}%")

    # ============================================================ 5. the MID tercile
    hr("5. THE 'MID TERCILE IS STRONGEST' SUBGROUP — priced for the grid it was picked from")
    terc = pd.qcut(a.vix_pct1y, 3, labels=["low", "mid", "high"], duplicates="drop").values
    # rebuild the horizon x tercile grid on the common-all-horizons frame
    hk = [f"g{h}" for h in HORIZONS] + ["vix_pct1y", "bb10_2.0_below"]
    b2 = d[hk].dropna().copy()
    tb = pd.qcut(b2.vix_pct1y, 3, labels=["low", "mid", "high"], duplicates="drop").values
    below_b = b2["bb10_2.0_below"].astype(bool).values
    print(f"  subgroup frame n={len(b2):,}  below n={int(below_b.sum())}")
    print(f"  {'cell':<22}{'n':>6}{'within-terc diff':>19}{'p (nominal)':>13}")
    cells = []
    for h in HORIZONS:
        yy = b2[f"g{h}"].values
        for t in ("low", "mid", "high"):
            tm = tb == t
            s = below_b & tm
            if s.sum() < MIN_N:
                if h == 5:
                    print(f"  {t+' D'+str(h):<22}{int(s.sum()):>6}   n<{MIN_N} INCONCLUSIVE")
                continue
            no = (~below_b) & tm
            diff = yy[s].mean() - yy[no].mean()
            # rotation null restricted to the tercile
            dsig = below_b.astype(float)
            cnt = circ_corr(dsig, tm.astype(float))
            ssum = circ_corr(dsig, np.where(tm, yy, 0.0))
            with np.errstate(invalid="ignore", divide="ignore"):
                null_in = np.where(cnt >= 5, ssum / cnt, np.nan)
            tot_s = np.where(tm, yy, 0.0).sum()
            tot_c = tm.sum()
            null_out = (tot_s - ssum) / (tot_c - cnt)
            null = (null_in - null_out)[offset_mask(len(yy))]
            p = p_from_null(diff, null)
            cells.append(dict(cell=f"{t} D{h}", n=int(s.sum()), diff=diff, p=p,
                              z=zstat(diff, null), null=null))
            if h == 5:
                print(f"  {t+' D'+str(h):<22}{int(s.sum()):>6}{diff*100:>18.3f}%{p:>13.3f}"
                      f"{stars(p)}")
    C = pd.DataFrame([{k: v for k, v in c.items() if k != "null"} for c in cells])
    nulls_arr = np.vstack([np.abs((c["null"] - np.nanmean(c["null"])) /
                                  max(np.nanstd(c["null"]), 1e-15)) for c in cells])
    zo = C.z.abs().values
    maxz = np.nanmax(nulls_arr, axis=0)
    mid5 = C[C.cell == "mid D5"]
    if len(mid5):
        i = mid5.index[0]
        fw = float((maxz >= zo[i]).mean())
        print(f"\n  the grid this cell was chosen from: {len(C)} cells "
              f"(3 terciles x {len(HORIZONS)} horizons, n>={MIN_N})")
        print(f"  'mid D5' nominal p={C.p[i]:.3f}, |z|={zo[i]:.2f}  ->  max-stat FWER p over the "
              f"grid = {fw:.3f} {stars(fw)}")
        print(f"  Bonferroni over just the 3 terciles at D5: {min(1.0, C.p[i]*3):.3f}")
        print(f"  best cell in the grid: {C.loc[zo.argmax(),'cell']} "
              f"(|z|={zo.max():.2f}, n={C.loc[zo.argmax(),'n']}) — the mid-D5 cell is rank "
              f"{int((zo > zo[i]).sum())+1} of {len(C)}")
        # stability of the mid cell
        tm = tb == "mid"
        s = below_b & tm
        ymb = b2.index.to_period("M")
        cm = {}
        for mo in sorted(set(ymb[s])):
            sel = s & np.asarray(ymb == mo)
            cm[mo] = (sel.sum(), b2.g5.values[sel].mean())
        wm_ = min(cm, key=lambda k: cm[k][0] * cm[k][1])
        keepd = ~np.asarray(ymb == wm_)
        yy = b2.g5.values[keepd]
        s2 = below_b[keepd] & (tb[keepd] == "mid")
        no2 = (~below_b[keepd]) & (tb[keepd] == "mid")
        print(f"  mid-tercile, DROP its worst month ({wm_}, n={cm[wm_][0]}): "
              f"n={int(s2.sum())} diff {(yy[s2].mean()-yy[no2].mean())*100:+.3f}%"
              f"{'  <-- n<25 INCONCLUSIVE' if s2.sum() < MIN_N else ''}")

    # ============================================================ 6. pseudo-OOS
    hr("6. PSEUDO OUT-OF-SAMPLE — the family was searched on the whole history; split it")
    for nm, sel in [("1990-2007 (first half)", yrs <= 2007),
                    ("2008-2026 (second half)", yrs >= 2008),
                    ("1990-2009", yrs <= 2009),
                    ("2010-2026 (true OOS: post-discovery-era data)", yrs >= 2010),
                    ("2015-2026", yrs >= 2015)]:
        yy = y[sel]
        sb = below[sel]
        if sb.sum() < MIN_N:
            print(f"  {nm:<48} n_below={int(sb.sum()):>4}   n<{MIN_N} INCONCLUSIVE")
            continue
        vv = pd.qcut(pd.Series(a.vix_pct1y.values[sel]), 20, labels=False,
                     duplicates="drop").values
        c, nl = fwl_rot_all(sb, yy, fe_Q(len(yy), [vv]))
        p = p_from_null(c, nl)
        print(f"  {nm:<48} n_below={int(sb.sum()):>4}  raw exc "
              f"{(yy[sb].mean()-yy.mean())*100:>+7.3f}%  FE coef {c*100:>+7.3f}%  "
              f"p={p:.3f} {stars(p)}")

    # in the second half, re-run the whole family and see if bb10_2.0_below|D5 is still the pick
    hr("6b. WOULD THE SEARCH HAVE PICKED THIS CONFIG OUT-OF-SAMPLE?", "-")
    for nm, sel_f in [("first half 1990-2007", fam.index.year <= 2007),
                      ("second half 2008-2026", fam.index.year >= 2008)]:
        yy5 = fam.g5.values[sel_f]
        vv = pd.qcut(pd.Series(fam.vix_pct1y.values[sel_f]), 20, labels=False,
                     duplicates="drop").values
        Qs = fe_Q(len(yy5), [vv])
        res = []
        for lbl, m, h in configs:
            if h != 5:
                continue
            ms = m[sel_f]
            if ms.sum() < MIN_N:
                continue
            c, nl = fwl_rot_all(ms, yy5, Qs)
            res.append((lbl, int(ms.sum()), c, zstat(c, nl)))
        res.sort(key=lambda r: -abs(r[3]))
        rk = [i for i, r in enumerate(res) if r[0] == "bb10_2.0_below|D5"]
        print(f"  {nm}: D5 configs with n>={MIN_N}: {len(res)};  "
              f"bb10_2.0_below|D5 rank = {(rk[0]+1) if rk else 'n/a (n<25)'}")
        print("     top 5: " + ", ".join(f"{r[0]}({r[2]*100:+.2f}%,|z|{abs(r[3]):.2f})"
                                         for r in res[:5]))


if __name__ == "__main__":
    main()
