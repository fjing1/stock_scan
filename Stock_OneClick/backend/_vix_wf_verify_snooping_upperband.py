"""
_vix_wf_verify_snooping_upperband.py — ADVERSARIAL verification, SNOOPING lens.

CLAIM UNDER TEST:
  "The UPPER band IS redundant: bb10_2.0_above's raw D5 excess +0.30% (n=518, p=.017) falls to
   +0.18% (p=.154) once vix_pct1y is controlled nonparametrically, to +0.08% (p=.528) under SPX
   5-day-return fixed effects, and to +0.04% (p=.761, only 14% of raw) under the full control set
   — it is a repackaging of 'VIX is high and the index just fell'."

The snooping attack has THREE prongs, because the claim is a NULL claim built out of p-values:

  P1. SPECIFICATION SEARCH FOR A NULL. _vix_wf_bandvslevel4.py ran NINE control sets on the ABOVE
      dummy. The claim quotes the three with the largest p-values and ignores the four where the
      dummy stays significant at p<.05. Census every spec (the 9 that were run plus more equally
      defensible ones) and show the full distribution of (coef, p), not the tail of it.

  P2. THE ATTENUATION IS NOT MEASURED, IT IS ASSUMED. "0.30 -> 0.04, only 14% of raw" treats two
      noisy point estimates as if their ratio were precise. Two tests:
        (a) ROTATION TEST ON THE ATTENUATION ITSELF: controls shrink *every* mask, not just this
            one. Roll the mask, recompute raw AND controlled coef for the rotated mask, and ask
            how often a random-timed mask of the same shape gets attenuated at least as hard.
        (b) BLOCK BOOTSTRAP CI on raw, on controlled, and on the DIFFERENCE and RATIO.
      If the CI on the difference straddles 0 and the CI on the ratio spans [0,1] and beyond,
      "falls to 14% of raw" is an unmeasurable quantity dressed up as a finding.

  P3. THE STARTING POINT WAS NEVER SIGNIFICANT. p=.017 is quoted as if it were an established
      effect that the controls then destroy. Price the whole family of signal x horizon tests
      this project ran (130 combos) with Bonferroni and Benjamini-Hochberg. If the raw +0.30%
      does not survive family-wide correction, there was no effect to explain away, and the
      control study is a demonstration about noise.

  Plus the requested episode-concentration checks (worst month, 2008, 2020) and an over-control
  diagnostic (how much of the dummy's own variance the "full control set" eats).

METHOD
  Exact rotation null: instead of 5000 random offsets, compute the statistic at ALL n circular
  offsets via FFT. For a dummy d, controls X with orthonormal basis Q, and y residualised to
  y_res (Q^T y_res = 0):
      numerator(off)   = roll(d,off) . y_res                      -> one circular cross-correlation
      denominator(off) = ||d||^2 - sum_j (q_j . roll(d,off))^2     -> one per control column
  so every offset's Frisch-Waugh coefficient is available at once. This is the same null as
  _vix_ma10_bb_research.rotation_pvalue / _vix_wf_bandvslevel4.test, just evaluated exhaustively
  (n=8,978 offsets) instead of sampled 5,000 times.

Run: ../../vcp_env/bin/python _vix_wf_verify_snooping_upperband.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import _vix_data

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20260914)
N_BOOT = 2000
BLOCK = 21          # bootstrap block length: covers the 5-day overlap + VIX signal clustering


# ---------------------------------------------------------------- exact rotation machinery
def circ_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """r[off] = sum_i a[(i+off) mod n] * b[i]  for all off, via FFT."""
    n = len(a)
    return np.real(np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b))))[:n]


def fe_matrix(n, fe_list=(), extra=None):
    cols = [np.ones(n)]
    for codes in fe_list:
        codes = np.asarray(codes)
        for k in np.unique(codes)[1:]:
            cols.append((codes == k).astype(float))
    if extra is not None:
        cols.extend([np.asarray(c, float) for c in extra])
    return np.column_stack(cols)


def rot_coefs_all(y: np.ndarray, X: np.ndarray, d: np.ndarray):
    """Frisch-Waugh coefficient on d, and the same coefficient for ALL n circular rotations of d.
    Returns (observed, null_array_length_n, Q, y_res)."""
    Q, _ = np.linalg.qr(X)
    y_res = y - Q @ (Q.T @ y)
    d = np.asarray(d, float)
    num = circ_corr(d, y_res)                       # roll(d,off) . y_res  for every off
    dd = float(d @ d)
    proj = np.zeros(len(d))
    for j in range(Q.shape[1]):
        proj += circ_corr(d, Q[:, j]) ** 2          # (q_j . roll(d,off))^2
    den = dd - proj
    with np.errstate(divide="ignore", invalid="ignore"):
        null = np.where(den > 1e-9, num / den, np.nan)
    return float(null[0]), null, Q, y_res


def p_from_null(obs, null):
    null = null[~np.isnan(null)]
    base = null.mean()
    return float((np.abs(null - base) >= abs(obs - base)).mean()), float(null.std())


def stars(p):
    return "***" if p < 0.01 else ("** " if p < 0.05 else ("*  " if p < 0.10 else "   "))


# ---------------------------------------------------------------- data (same frame as level4)
def build():
    d = _vix_data.add_features(_vix_data.load())
    d["rv20"] = d.spx.pct_change(fill_method=None).rolling(20).std() * np.sqrt(252)
    d["spx_r20"] = d.spx / d.spx.shift(20) - 1.0
    d["spx_r5"] = d.spx / d.spx.shift(5) - 1.0
    d["spx_r1"] = d.spx / d.spx.shift(1) - 1.0
    d["spx_r10"] = d.spx / d.spx.shift(10) - 1.0
    d["vix_chg10"] = d.vix / d.vix.shift(10) - 1.0
    d["vrp"] = d.vix / 100.0 - d.rv20
    # NOTE: exactly the columns _vix_wf_bandvslevel4.py requires, so the frame is bit-identical
    # (n=8,978). vix_pct2y / vix_z1y / stretch are added WITHOUT entering the dropna, because
    # vix_pct2y's 504-day warm-up would silently shrink the frame and change the claim's numbers.
    keep = ["g5", "vix_pct1y", "bb10_width", "rv20", "spx_r20", "spx_r5", "vrp",
            "bb10_2.0_below", "bb10_2.0_above"]
    a = d[keep].dropna().copy()
    extra_cols = ["vix_pct2y", "vix_z1y", "stretch", "vix", "vix_chg10", "spx_r1", "spx_r10",
                  "bb20_2.0_above"]
    for c in extra_cols:
        a[c] = d.loc[a.index, c]
    for c in ["bb10_2.0_above", "bb20_2.0_above", "bb10_2.0_below"]:
        a[c] = a[c].astype(bool)
    return d, a


def main():
    d_all, a = build()
    y = a.g5.values
    above = a["bb10_2.0_above"].values
    n = len(a)
    base = y.mean()
    print(f"frame {n:,} rows  {a.index[0].date()} -> {a.index[-1].date()}   "
          f"baseline g5 {base*100:+.3f}%   above-band n={int(above.sum())} "
          f"({above.mean()*100:.1f}% of days)")
    print(f"raw conditional mean {y[above].mean()*100:+.3f}%  -> raw EXCESS over same-sample "
          f"baseline {(y[above].mean()-base)*100:+.3f}%")

    # NaN-safe quantile binning: a missing value gets its own FE bin (-1) rather than dropping the
    # row, so every spec is fit on the identical 8,978-row frame the claim used.
    def QQ(s, k):
        v = a[s]
        c = pd.qcut(v, k, labels=False, duplicates="drop").values
        c = np.where(np.isnan(c), -1.0, c)
        return c
    lvl20 = QQ("vix_pct1y", 20)
    lvl10 = QQ("vix_pct1y", 10)
    wid = QQ("bb10_width", 10)
    rv = QQ("rv20", 10)
    r20 = QQ("spx_r20", 10)
    r5 = QQ("spx_r5", 10)
    r1 = QQ("spx_r1", 10)
    r10 = QQ("spx_r10", 10)
    vrp = QQ("vrp", 10)
    p2y = QQ("vix_pct2y", 10)
    z1y = QQ("vix_z1y", 10)
    stre = QQ("stretch", 10)
    vlv = QQ("vix", 10)
    chg = QQ("vix_chg10", 10)

    # ============================================================ P1. specification census
    print("\n" + "=" * 108)
    print("P1. SPECIFICATION CENSUS — every defensible control set for the ABOVE dummy (D5, g5)")
    print("     [R] = one of the 9 specs actually run in _vix_wf_bandvslevel4.py")
    print("     [Q] = one of the 3 specs QUOTED in the claim")
    print("=" * 108)
    specs = [
        ("no control (the 'raw' number)",                       [],                 None, "RQ"),
        ("vix_pct1y VIGINTILE FE (level, 20 bins)",             [lvl20],            None, "RQ"),
        ("vix_pct1y DECILE FE (level, 10 bins)",                [lvl10],            None, ""),
        ("vix_pct2y decile FE",                                 [p2y],              None, ""),
        ("vix_z1y decile FE",                                   [z1y],              None, ""),
        ("RAW VIX decile FE (absolute level)",                  [vlv],              None, ""),
        ("linear vix_pct1y (round-1 spec)",                     [],   [a.vix_pct1y.values], ""),
        ("bb10_width decile FE",                                [wid],              None, "R"),
        ("SPX realized-vol(20d) decile FE",                     [rv],               None, "R"),
        ("SPX trailing 20d return decile FE",                   [r20],              None, "R"),
        ("SPX trailing 10d return decile FE",                   [r10],              None, ""),
        ("SPX trailing 5d return decile FE",                    [r5],               None, "RQ"),
        ("SPX trailing 1d return decile FE",                    [r1],               None, ""),
        ("VRP decile FE",                                       [vrp],              None, "R"),
        ("stretch decile FE",                                   [stre],             None, ""),
        ("10d VIX change decile FE",                            [chg],              None, ""),
        ("level + width + realized-vol FE",                     [lvl20, wid, rv],   None, "R"),
        ("level + SPX 20d return FE",                           [lvl20, r20],       None, ""),
        ("level + realized-vol FE",                             [lvl20, rv],        None, ""),
        ("EVERYTHING (level+width+rv+r20+r5) FE",               [lvl20, wid, rv, r20, r5], None, "RQ"),
        ("EVERYTHING but with r10 instead of r5",               [lvl20, wid, rv, r20, r10], None, ""),
        ("EVERYTHING but level as DECILES",                     [lvl10, wid, rv, r20, r5], None, ""),
        ("EVERYTHING minus the SPX-return FEs",                 [lvl20, wid, rv],   None, ""),
    ]
    print(f"  {'control set':<48}{'tag':>4}{'coef':>10}{'p(exact rot)':>14}{'k':>5}"
          f"{'%of raw':>9}{'nullSD':>9}")
    print("-" * 108)
    census = []
    raw_coef = None
    for lbl, fes, extra, tag in specs:
        X = fe_matrix(n, fes, extra)
        obs, null, _, _ = rot_coefs_all(y, X, above)
        p, sd = p_from_null(obs, null)
        if raw_coef is None:
            raw_coef = obs
        census.append((lbl, tag, obs, p, X.shape[1]))
        print(f"  {lbl:<48}{tag:>4}{obs*100:>+9.3f}%{p:>10.3f} {stars(p)}{X.shape[1]:>5}"
              f"{obs/raw_coef*100:>8.0f}%{sd*100:>8.3f}%")
    ran = [c for c in census if "R" in c[1]]
    quoted = [c for c in census if "Q" in c[1]]
    sig_ran = [c for c in ran if c[3] < 0.05]
    allspec = census
    sig_all = [c for c in allspec if c[3] < 0.05]
    print(f"\n  of the {len(ran)} specs ACTUALLY RUN in bandvslevel4: {len(sig_ran)} leave the ABOVE "
          f"dummy significant at p<.05 -> {[c[0][:34] for c in sig_ran]}")
    print(f"  the claim quotes {len(quoted)} of them ({[c[0][:26] for c in quoted]}) — i.e. the raw "
          f"one plus the {len(quoted)-1} highest-p specs.")
    cf = np.array([c[2] for c in allspec[1:]])
    print(f"  across all {len(allspec)-1} control sets: coef median {np.median(cf)*100:+.3f}%  "
          f"range {cf.min()*100:+.3f}%..{cf.max()*100:+.3f}%  "
          f"share with p<.05 {len(sig_all)-1}/{len(allspec)-1}  "
          f"share still POSITIVE {np.mean(cf>0)*100:.0f}%")

    # ============================================================ P2a. rotation test on ATTENUATION
    print("\n" + "=" * 108)
    print("P2a. IS THE ATTENUATION SPECIFIC TO THIS SIGNAL? Controls shrink big coefficients")
    print("     mechanically. Rotate the mask; keep only the rotations whose RAW coefficient is as")
    print("     large as the real one (|raw| >= observed); then ask what fraction of THOSE lose as")
    print("     much of their coefficient as the real mask does. A large p = the shrinkage is what")
    print("     the control does to ANY well-timed mask, not proof that the level 'explains' the band.")
    print("=" * 108)
    X0 = fe_matrix(n, [], None)
    _, null_raw, _, _ = rot_coefs_all(y, X0, above)
    print(f"  {'control set':<44}{'raw':>8}{'ctrl':>8}{'kept ratio':>12}{'n comparable':>14}"
          f"{'median ratio':>14}{'p(ratio)':>10}")
    print("-" * 108)
    for lbl, fes, extra in [("vix_pct1y vigintile FE", [lvl20], None),
                            ("SPX trailing 5d return decile FE", [r5], None),
                            ("EVERYTHING (level+width+rv+r20+r5)", [lvl20, wid, rv, r20, r5], None)]:
        X = fe_matrix(n, fes, extra)
        obs_c, null_c, _, _ = rot_coefs_all(y, X, above)
        ok = ~np.isnan(null_c) & ~np.isnan(null_raw)
        obs_raw = null_raw[0]
        big = ok & (np.abs(null_raw) >= abs(obs_raw))
        with np.errstate(divide="ignore", invalid="ignore"):
            rat_null = null_c[big] / null_raw[big]
        rat_obs = obs_c / obs_raw
        p = float((rat_null <= rat_obs).mean())
        print(f"  {lbl:<44}{obs_raw*100:>+7.3f}%{obs_c*100:>+7.3f}%{rat_obs*100:>11.0f}%"
              f"{int(big.sum()):>14}{np.median(rat_null)*100:>13.0f}%{p:>10.3f}")

    # ============================================================ P2b. block bootstrap CIs
    print("\n" + "=" * 108)
    print(f"P2b. BLOCK BOOTSTRAP ({N_BOOT} draws, {BLOCK}-day circular blocks) — CI on raw, on")
    print("     controlled, and on the DIFFERENCE and RATIO the claim reports as '14% of raw'")
    print("=" * 108)

    def boot_pair(Xc):
        nb = int(np.ceil(n / BLOCK))
        out = np.empty((N_BOOT, 2))
        Xr = fe_matrix(n, [], None)
        for b in range(N_BOOT):
            starts = RNG.integers(0, n, size=nb)
            idx = (starts[:, None] + np.arange(BLOCK)[None, :]).ravel()[:n] % n
            yb, db = y[idx], above[idx].astype(float)
            for j, XX in enumerate((Xr, Xc)):
                Xb = XX[idx]
                sol, *_ = np.linalg.lstsq(Xb, np.column_stack([yb, db]), rcond=None)
                res = np.column_stack([yb, db]) - Xb @ sol
                yr, dr = res[:, 0], res[:, 1]
                den = float(dr @ dr)
                out[b, j] = (dr @ yr) / den if den > 1e-9 else np.nan
        return out

    for lbl, fes in [("vix_pct1y vigintile FE", [lvl20]),
                     ("SPX trailing 5d return decile FE", [r5]),
                     ("EVERYTHING (level+width+rv+r20+r5)", [lvl20, wid, rv, r20, r5])]:
        Xc = fe_matrix(n, fes, None)
        bp = boot_pair(Xc)
        bp = bp[~np.isnan(bp).any(axis=1)]
        rawb, ctrb = bp[:, 0], bp[:, 1]
        diff = rawb - ctrb
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.abs(rawb) > 1e-6, ctrb / rawb, np.nan)
        ratio = ratio[~np.isnan(ratio)]
        q = lambda v, p: np.percentile(v, p)
        print(f"\n  {lbl}")
        print(f"    raw coef        {null_raw[0]*100:+.3f}%   95% CI "
              f"[{q(rawb,2.5)*100:+.3f}%, {q(rawb,97.5)*100:+.3f}%]")
        obs_c = rot_coefs_all(y, Xc, above)[0]
        print(f"    controlled coef {obs_c*100:+.3f}%   95% CI "
              f"[{q(ctrb,2.5)*100:+.3f}%, {q(ctrb,97.5)*100:+.3f}%]")
        print(f"    ATTENUATION (raw-ctrl) {(null_raw[0]-obs_c)*100:+.3f}%   95% CI "
              f"[{q(diff,2.5)*100:+.3f}%, {q(diff,97.5)*100:+.3f}%]   "
              f"P(attenuation<=0) = {np.mean(diff<=0):.3f}")
        print(f"    RATIO ctrl/raw  {obs_c/null_raw[0]*100:.0f}%   95% CI "
              f"[{q(ratio,2.5)*100:.0f}%, {q(ratio,97.5)*100:.0f}%]   "
              f"P(ratio>=0.50)={np.mean(ratio>=0.5):.3f}  P(ratio>=1)={np.mean(ratio>=1):.3f}")
        # minimum detectable effect: what coef would this controlled test have needed for p<.05?
        _, nullc = rot_coefs_all(y, Xc, above)[0], rot_coefs_all(y, Xc, above)[1]
        sdc = np.nanstd(nullc)
        print(f"    controlled test's rotation-null SD {sdc*100:.3f}% -> it can only detect effects "
              f"|coef| > {1.96*sdc*100:.3f}% at p<.05")

    # ============================================================ P2c. over-control diagnostic
    print("\n" + "=" * 108)
    print("P2c. OVER-CONTROL DIAGNOSTIC — how much of the ABOVE dummy's own variance does each")
    print("     control set absorb? (a control that eats the treatment cannot test the treatment)")
    print("=" * 108)
    dm = above.astype(float)
    print(f"  {'control set':<48}{'R2 of dummy on ctrls':>22}{'resid var left':>16}{'SE inflation':>14}")
    for lbl, fes in [("vix_pct1y vigintile FE", [lvl20]),
                     ("SPX trailing 5d return decile FE", [r5]),
                     ("SPX trailing 20d return decile FE", [r20]),
                     ("EVERYTHING (level+width+rv+r20+r5)", [lvl20, wid, rv, r20, r5])]:
        X = fe_matrix(n, fes, None)
        Q, _ = np.linalg.qr(X)
        dres = dm - Q @ (Q.T @ dm)
        r2 = 1 - float(dres @ dres) / float(((dm - dm.mean()) ** 2).sum())
        print(f"  {lbl:<48}{r2*100:>21.1f}%{(1-r2)*100:>15.1f}%{1/np.sqrt(max(1-r2,1e-9)):>13.2f}x")
    print("\n  distribution of above-band days across SPX trailing-5d-return deciles:")
    cnt = [int(((r5 == k) & above).sum()) for k in range(10)]
    tot = [int((r5 == k).sum()) for k in range(10)]
    print("    decile (worst 5d -> best):  " + " ".join(f"D{k+1}:{cnt[k]:>3}" for k in range(10)))
    print(f"    {cnt[0]/above.sum()*100:.0f}% of all above-band days sit in the single worst "
          f"trailing-5d-return decile; {(cnt[0]+cnt[1])/above.sum()*100:.0f}% in the worst two.")
    print(f"    conversely those two deciles are {(cnt[0]+cnt[1])/(tot[0]+tot[1])*100:.0f}% "
          f"above-band days.")

    # ============================================================ P2d. the single fragile knob
    print("\n" + "=" * 108)
    print("P2d. WHICH KNOB IS DOING THE WORK? The 'full control set' is level+width+rv+r20+r5.")
    print("     Swap ONLY the short-horizon SPX-return control and refit; leave-one-out the rest.")
    print("=" * 108)
    horiz = {"r1": r1, "r5": r5, "r10": r10, "r20": r20}
    print(f"  {'full control set, short-return leg = ':<52}{'coef':>9}{'p':>9}{'% of raw':>10}")
    for hn, hc in [("r1 (1-day)", r1), ("r5 (5-day, the reported choice)", r5),
                   ("r10 (10-day)", r10), ("r20 (20-day)", r20), ("none", None)]:
        fes = [lvl20, wid, rv, r20] + ([hc] if hc is not None else [])
        X = fe_matrix(n, fes, None)
        o, nl, _, _ = rot_coefs_all(y, X, above)
        p, _ = p_from_null(o, nl)
        print(f"  {hn:<52}{o*100:>+8.3f}%{p:>9.3f}{o/raw_coef*100:>9.0f}%")
    print(f"\n  leave-one-control-out from the full set (all five legs, drop one at a time):")
    legs = [("level (pct1y vig)", lvl20), ("bb10_width", wid), ("realized vol", rv),
            ("SPX r20", r20), ("SPX r5", r5)]
    for i, (nm, _) in enumerate(legs):
        fes = [c for j, (_, c) in enumerate(legs) if j != i]
        X = fe_matrix(n, fes, None)
        o, nl, _, _ = rot_coefs_all(y, X, above)
        p, _ = p_from_null(o, nl)
        print(f"    drop {nm:<24}-> coef {o*100:+.3f}%  p={p:.3f}  ({o/raw_coef*100:.0f}% of raw)")
    print(f"\n  within-bin matched difference on SPX trailing-5d-return deciles "
          f"(only bins with >=25 signal and >=25 non-signal days are interpretable):")
    num = den = 0.0
    for k in range(10):
        bm = r5 == k
        s, no = above & bm, (~above) & bm
        tag = "" if s.sum() >= 25 else "  <-- n<25 INCONCLUSIVE"
        dv = y[s].mean() - y[no].mean() if s.sum() else np.nan
        if s.sum() >= 25:
            num += s.sum() * dv
            den += s.sum()
        print(f"    r5 decile {k+1:>2}  n_sig={int(s.sum()):>4}  n_no={int(no.sum()):>5}  "
              f"diff {dv*100:>+7.3f}%{tag}")
    print(f"    pooled over the {int(den)} signal days in INTERPRETABLE bins: {num/den*100:+.3f}% "
          f"({num/den/raw_coef*100:.0f}% of raw)")

    # ============================================================ P3. family-wide multiple testing
    print("\n" + "=" * 108)
    print("P3. MULTIPLE TESTING — price the raw p=.017 against the whole family this project ran")
    print("=" * 108)
    dd = _vix_data.add_features(_vix_data.load())
    sig_defs = []
    for thr in (0.05, 0.10, 0.15, 0.20, 0.25):
        sig_defs.append((f"stretch>=+{thr*100:.0f}%", (dd.stretch >= thr)))
    for thr in (-0.05, -0.10, -0.15):
        sig_defs.append((f"stretch<={thr*100:.0f}%", (dd.stretch <= thr)))
    sig_defs.append(("stretch>0", dd.stretch > 0))
    sig_defs.append(("stretch<0", dd.stretch < 0))
    for nn in (10, 20):
        for k in (1.5, 2.0, 2.5):
            sig_defs.append((f"BB({nn},{k}) above", dd[f"bb{nn}_{k}_above"]))
            sig_defs.append((f"BB({nn},{k}) reentry", dd[f"bb{nn}_{k}_reentry"]))
    for nn in (10, 20):
        sig_defs.append((f"BB({nn},2.0) below", dd[f"bb{nn}_2.0_below"]))
        sig_defs.append((f"BB({nn},2.0) exit_lo", dd[f"bb{nn}_2.0_exit_lo"]))

    rows = []
    for h in (1, 3, 5, 10, 21):
        fwd = dd[f"g{h}"].values
        valid = ~np.isnan(fwd)
        f0 = np.where(valid, np.nan_to_num(fwd), 0.0)
        vv = valid.astype(float)
        for lbl, s in sig_defs:
            m = np.asarray(s.fillna(False).astype(bool).values, float)
            if (m.astype(bool) & valid).sum() < 25:
                continue
            num = circ_corr(m, f0)
            den = circ_corr(m, vv)
            with np.errstate(divide="ignore", invalid="ignore"):
                nulls = np.where(den >= 5, num / den, np.nan)
            obs = nulls[0]
            bmean = np.nanmean(nulls)
            p = float((np.abs(nulls[~np.isnan(nulls)] - bmean) >= abs(obs - bmean)).mean())
            exc = obs - fwd[valid].mean()
            rows.append({"sig": lbl, "h": h, "n": int((m.astype(bool) & valid).sum()),
                         "exc": exc, "p": p})
    fam = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    M = len(fam)
    fam["bonf"] = np.minimum(fam.p * M, 1.0)
    fam["bh_crit"] = (fam.index + 1) / M * 0.05
    kmax = np.flatnonzero(fam.p.values <= fam.bh_crit.values)
    bh_cut = fam.p.values[kmax.max()] if len(kmax) else 0.0
    fam["bh_sig"] = fam.p <= bh_cut
    print(f"  family = {M} signal x horizon tests (the grid this project actually searched).")
    print(f"  Bonferroni threshold for alpha=.05: p < {0.05/M:.5f}")
    print(f"  Benjamini-Hochberg (FDR 5%) cut: p <= {bh_cut:.5f}  -> {int(fam.bh_sig.sum())} discoveries")
    print(f"\n  top 12 of the family by raw p:")
    print(f"  {'signal':<24}{'h':>3}{'n':>7}{'excess':>10}{'raw p':>9}{'Bonf p':>9}{'BH sig':>8}")
    for _, r in fam.head(12).iterrows():
        print(f"  {r.sig:<24}{int(r.h):>3}{int(r.n):>7}{r.exc*100:>+9.3f}%{r.p:>9.4f}"
              f"{r.bonf:>9.3f}{str(bool(r.bh_sig)):>8}")
    tgt = fam[(fam.sig == "BB(10,2.0) above") & (fam.h == 5)]
    if len(tgt):
        t = tgt.iloc[0]
        rank = int(fam.index[(fam.sig == "BB(10,2.0) above") & (fam.h == 5)][0]) + 1
        print(f"\n  >>> THE CLAIM'S STARTING POINT, BB(10,2.0) above @ D5: excess {t.exc*100:+.3f}% "
              f"raw p={t.p:.4f}  rank {rank}/{M}")
        print(f"      Bonferroni-adjusted p = {t.bonf:.3f}   BH-significant at FDR 5%: {bool(t.bh_sig)}")

    # ============================================================ P4. episode concentration
    print("\n" + "=" * 108)
    print("P4. EPISODE CONCENTRATION — drop the worst month, drop 2008, drop 2020")
    print("     (recomputed for BOTH the raw coefficient and the claim's 'full control set')")
    print("=" * 108)
    Xfull = fe_matrix(n, [lvl20, wid, rv, r20, r5], None)
    Xlvl = fe_matrix(n, [lvl20], None)
    ym = np.asarray(a.index.to_period("M").astype(str))

    def coefs_on(sel):
        """raw + level-FE + full-control coefficients on a subsample (FE refit within subsample)."""
        ys, ds = y[sel], above[sel].astype(float)
        if ds.sum() < 25:
            return None
        out = []
        for XX in (fe_matrix(sel.sum(), [], None), Xlvl[sel], Xfull[sel]):
            sol, *_ = np.linalg.lstsq(XX, np.column_stack([ys, ds]), rcond=None)
            res = np.column_stack([ys, ds]) - XX @ sol
            yr, dr = res[:, 0], res[:, 1]
            den = float(dr @ dr)
            out.append((dr @ yr) / den if den > 1e-9 else np.nan)
        return out, int(ds.sum())

    # which single calendar month contributes most to the raw effect?
    contrib = {}
    for m in np.unique(ym):
        s = (ym == m) & above
        if s.sum() == 0:
            continue
        contrib[m] = (y[s].mean() - base) * s.sum()
    worst = max(contrib, key=contrib.get)
    wm = (ym == worst)
    print(f"  single month contributing most to the raw ABOVE effect: {worst} "
          f"(n={int((wm & above).sum())} above-days, "
          f"mean g5 {y[wm & above].mean()*100:+.2f}%)")
    yrs = a.index.year.values
    scen = [("full sample", np.ones(n, bool)),
            (f"drop worst month ({worst})", ~wm),
            ("drop 2008", yrs != 2008),
            ("drop 2020", yrs != 2020),
            ("drop 2008 + 2020", (yrs != 2008) & (yrs != 2020)),
            ("drop 2008+2020+worst month", (yrs != 2008) & (yrs != 2020) & ~wm),
            ("drop GFC window 2008-2009", (yrs != 2008) & (yrs != 2009)),
            ("2010s+2020s only", yrs >= 2010)]
    print(f"\n  {'subsample':<34}{'n_above':>9}{'raw':>10}{'level FE':>11}{'FULL ctrl':>11}"
          f"{'ctrl/raw':>10}")
    print("-" * 108)
    for lbl, sel in scen:
        r = coefs_on(sel)
        if r is None:
            print(f"  {lbl:<34}   n<25 INCONCLUSIVE")
            continue
        (c0, c1, c2), k = r
        print(f"  {lbl:<34}{k:>9}{c0*100:>+9.3f}%{c1*100:>+10.3f}%{c2*100:>+10.3f}%"
              f"{c2/c0*100 if abs(c0)>1e-9 else float('nan'):>9.0f}%")

    # leave-one-episode-out jackknife on the RAW and the FULL-CONTROL coefficient
    idx = np.flatnonzero(above)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    groups = np.split(idx, breaks + 1)
    print(f"\n  {len(groups)} distinct above-band episodes covering {int(above.sum())} days "
          f"(median length {int(np.median([len(g) for g in groups]))}, "
          f"max {max(len(g) for g in groups)})")

    def jk(X):
        Q, _ = np.linalg.qr(X)
        yr = y - Q @ (Q.T @ y)
        full = None
        vals = []
        dfull = above.astype(float)
        dres = dfull - Q @ (Q.T @ dfull)
        full = float(dres @ yr) / float(dres @ dres)
        for g in groups:
            m = above.copy()
            m[g] = False
            dv = m.astype(float)
            dr = dv - Q @ (Q.T @ dv)
            vals.append(float(dr @ yr) / float(dr @ dr))
        return full, np.array(vals)

    for lbl, X in [("raw (no control)", fe_matrix(n, [], None)),
                   ("level vigintile FE", Xlvl),
                   ("FULL control set", Xfull)]:
        f, v = jk(X)
        w = int(np.argmin(np.abs(v)))
        print(f"  jackknife {lbl:<22} full {f*100:+.3f}%  range {v.min()*100:+.3f}%..{v.max()*100:+.3f}%"
              f"  median {np.median(v)*100:+.3f}%  sign flips on dropping 1 episode: "
              f"{int((np.sign(v)!=np.sign(f)).sum())}/{len(v)}")

    # ============================================================ P5. the corroborations
    print("\n" + "=" * 108)
    print("P5. THE CLAIM'S OWN CORROBORATIONS — recomputed")
    print("=" * 108)
    terc = pd.qcut(a.vix_pct1y, 3, labels=["low", "mid", "high"], duplicates="drop").values
    num = den = 0.0
    print(f"  {'tercile':<10}{'n_sig':>7}{'sig mean':>11}{'no-sig':>11}{'WITHIN diff':>13}")
    for t in ("low", "mid", "high"):
        tm = terc == t
        s, no = above & tm, (~above) & tm
        if s.sum() < 25:
            print(f"  {t:<10}{int(s.sum()):>7}   n<25 INCONCLUSIVE")
            continue
        dv = y[s].mean() - y[no].mean()
        num += s.sum() * dv
        den += s.sum()
        print(f"  {t:<10}{int(s.sum()):>7}{y[s].mean()*100:>10.3f}%{y[no].mean()*100:>10.3f}%"
              f"{dv*100:>+12.3f}%")
    print(f"  POOLED level-neutral within-tercile diff: {num/den*100:+.3f}%  (n={int(den)}) "
          f"= {num/den/raw_coef*100:.0f}% of the raw coefficient {raw_coef*100:+.3f}%")
    print("  -> the claim cites this table as corroboration of redundancy, but the three within-")
    print("     tercile diffs are all POSITIVE and their pooled value is close to the raw effect;")
    print("     the p>=.11 there reflects tercile sample sizes, not attenuation.")

    b20 = a["bb20_2.0_above"].values
    for lbl, m in [("BB(20,2.0) above (the claim's replication)", b20)]:
        X = fe_matrix(n, [lvl20], None)
        o0, nl0, _, _ = rot_coefs_all(y, fe_matrix(n, [], None), m)
        p0, _ = p_from_null(o0, nl0)
        o1, nl1, _, _ = rot_coefs_all(y, X, m)
        p1, _ = p_from_null(o1, nl1)
        print(f"\n  {lbl}: n={int(m.sum())}  raw {o0*100:+.3f}% (p={p0:.3f})  "
              f"level-FE {o1*100:+.3f}% (p={p1:.3f})")
        print(f"    the claim quotes '+0.050%, p=.696' for BB(20,2.0) above as an independent")
        print(f"    replication — but its RAW coefficient is {o0*100:+.3f}%, so it never had an")
        print(f"    effect to lose. It replicates 'no signal', not 'signal explained by level'.")

    # ============================================================ P6. where the raw effect lives
    print("\n" + "=" * 108)
    print("P6. THE COMPETING (AND SIMPLER) EXPLANATION — the raw effect is a pre-2010 artifact,")
    print("     which the claim never tests. Era split of the RAW coefficient, no controls.")
    print("=" * 108)
    print(f"  {'era':<16}{'n_days':>8}{'n_above':>9}{'raw coef':>11}{'p':>8}{'level FE':>11}{'p':>8}"
          f"{'FULL ctrl':>11}{'p':>8}{'ctrl/raw':>10}")
    print("-" * 108)
    for nm, y0, y1 in [("1990s", 1990, 1999), ("2000s", 2000, 2009),
                       ("2010s", 2010, 2019), ("2020s", 2020, 2099),
                       ("pre-2010", 1990, 2009), ("2010 onward", 2010, 2099)]:
        sel = (yrs >= y0) & (yrs <= y1)
        if (above & sel).sum() < 25:
            print(f"  {nm:<16}{int(sel.sum()):>8}{int((above&sel).sum()):>9}   n<25 INCONCLUSIVE")
            continue
        ys, ds = y[sel], above[sel].astype(float)
        cells = []
        for XX in (fe_matrix(int(sel.sum()), [], None), Xlvl[sel], Xfull[sel]):
            o, nl, _, _ = rot_coefs_all(ys, XX, ds)
            p, _ = p_from_null(o, nl)
            cells.append((o, p))
        print(f"  {nm:<16}{int(sel.sum()):>8}{int(ds.sum()):>9}"
              + "".join(f"{o*100:>+10.3f}%{p:>8.3f}" for o, p in cells)
              + f"{cells[2][0]/cells[0][0]*100:>9.0f}%")
    print("\n  If the whole raw coefficient is pre-2010, then 'the level explains it' is not the")
    print("  reason the band is untradeable — 'it stopped happening 16 years ago' is.")


if __name__ == "__main__":
    main()
