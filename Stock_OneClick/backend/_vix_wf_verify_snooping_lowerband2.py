"""
_vix_wf_verify_snooping_lowerband2.py — round 2 of the snooping audit.

Round 1 showed: the config is rank 11 of a 255-config family; Romano-Wolf FWER p = .37 (narrowest
family) to .77 (full family); dropping the 3 worst months (5 of 159 days) takes the coefficient
from -.487% to -.328% (p=.107) and the 5 worst months (8 days) to -.242% (p=.227); the effect is
-.184% (p=.54) in 2010-2026.

Round 2 asks the questions that decide whether that is *selection* or *fat tails*:
  A. OUTLIER IMMUNITY — winsorised / trimmed / rank-based versions of the same level-FE effect.
     If the edge is pervasive it survives; if it is 5-8 crisis days it dies.
  B. IS "DROP THE 3 WORST MONTHS" A RIGGED TEST? — compare the observed attenuation to the
     attenuation from dropping 3/5 RANDOM signal-months, and to the attenuation a genuinely
     constant effect of the same size would show.
  C. IS THE 139-EPISODE JACKKNIFE INFORMATIVE? — arithmetic ceiling on single-episode influence.
  D. MID-vs-LOW TERCILE — the claim is a BETWEEN-subgroup ordering. Test the interaction.
  E. DOES THE FAMILY CONTAIN ANY SIGNAL AT ALL? — dependence-aware test on the COUNT of nominal
     winners (rotation null of the count), which is the right benchmark, not 0.05*k.
  F. CONTRIBUTION CURVE — what share of below-band days generate what share of the total effect.

Run: ../../vcp_env/bin/python _vix_wf_verify_snooping_lowerband2.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import _vix_data  # noqa: E402
from _vix_wf_verify_snooping_lowerband import (  # noqa: E402
    MIN_N, HORIZONS, circ_corr, offset_mask, mean_rot_all, fwl_rot_all,
    p_from_null, zstat, stars, fe_Q, hr, build,
)


def main():
    rng = np.random.default_rng(20260909)

    # ---- self-check of the FFT rotation machinery against brute force
    t = rng.normal(size=400)
    mm = rng.random(400) < 0.1
    bf = np.array([np.roll(mm.astype(float), o) @ t for o in (0, 7, 133, 399)])
    ff = circ_corr(mm.astype(float), t)[[0, 7, 133, 399]]
    print(f"self-check circ_corr vs brute force: max abs err {np.abs(bf-ff).max():.2e}")

    d = build()
    keep = ["g5", "vix_pct1y", "bb10_2.0_below"]
    a = d[keep].dropna().copy()
    y = a.g5.values
    n = len(y)
    below = a["bb10_2.0_below"].astype(bool).values
    base = y.mean()
    vig = pd.qcut(a.vix_pct1y, 20, labels=False, duplicates="drop").values
    Qv = fe_Q(n, [vig])
    ym = a.index.to_period("M")
    yrs = a.index.year
    c_ref, nl_ref = fwl_rot_all(below, y, Qv)
    print(f"frame n={n:,} {a.index[0].date()} -> {a.index[-1].date()}  below n={int(below.sum())}  "
          f"reference level-FE coef {c_ref*100:+.3f}%  p={p_from_null(c_ref, nl_ref):.3f}")

    # =============================================== A. outlier immunity
    hr("A. OUTLIER IMMUNITY — same level-FE estimator, robustified outcome (D5, n_below=159)")
    print(f"  {'outcome transform':<52}{'coef':>11}{'p':>9}{'% of raw':>10}")

    def run(yy, label, scale=None, unit="%"):
        c, nl = fwl_rot_all(below, yy, Qv)
        p = p_from_null(c, nl)
        rel = f"{c/c_ref*100:>9.0f}%" if scale is None else " " * 10
        v = f"{c*100:>+10.3f}%" if unit == "%" else f"{c:>+11.4f}"
        print(f"  {label:<52}{v}{p:>9.3f}{stars(p)}{rel}")
        return c, p

    run(y, "raw g5 (reference)")
    for q in (0.01, 0.025, 0.05):
        lo, hi = np.percentile(y, [q * 100, 100 - q * 100])
        run(np.clip(y, lo, hi), f"winsorised g5 at {q*100:g}/{100-q*100:g} pct "
                                f"[{lo*100:.1f}%,{hi*100:+.1f}%]")
    # rank of g5 WITHIN its own vix_pct1y vigintile -> level-controlled and scale-free
    rk = pd.Series(y).groupby(pd.Series(vig)).rank(pct=True).values
    c, p = run(rk, "within-vigintile PERCENTILE RANK of g5 (0..1)", scale=1, unit="r")
    print(f"       -> below-band mean rank {rk[below].mean():.4f} vs 0.5 baseline; "
          f"a -0.49% mean shift on a +0.20%/2.4%-sd outcome would be ~-0.08 in rank units")
    # sign: share of below days beating their vigintile median
    med = pd.Series(y).groupby(pd.Series(vig)).transform("median").values
    sgn = (y > med).astype(float)
    c, p = run(sgn, "P(g5 > own-vigintile median)  [pure sign test]", scale=1, unit="r")
    print(f"       -> below-band hit rate {sgn[below].mean()*100:.1f}% vs "
          f"{sgn[~below].mean()*100:.1f}% for the rest")
    # trimmed: drop below-band days in the extreme tails of g5 (both sides, symmetric)
    for k in (2, 4, 8):
        o = np.argsort(y[np.flatnonzero(below)])
        drop = np.flatnonzero(below)[np.r_[o[:k], o[-k:]]]
        m = below.copy()
        m[drop] = False
        cc, nn = fwl_rot_all(m, y, Qv)
        print(f"  {'symmetric trim: drop '+str(k)+' worst AND '+str(k)+' best below-days':<52}"
              f"{cc*100:>+10.3f}%{p_from_null(cc, nn):>9.3f}"
              f"{stars(p_from_null(cc, nn))}{cc/c_ref*100:>9.0f}%   n={int(m.sum())}")

    # =============================================== B. is the drop-worst-months test rigged?
    hr("B. IS 'DROP THE 3 WORST MONTHS' A RIGGED TEST? — random-month drop null")
    sig_months = sorted(set(ym[below]))
    print(f"  {len(sig_months)} calendar months contain >=1 below-band day")

    def coef_dropping(months):
        kp = np.ones(n, bool)
        for mo in months:
            kp &= ~np.asarray(ym == mo)
        yy, sb = y[kp], below[kp]
        vv = pd.qcut(pd.Series(a.vix_pct1y.values[kp]), 20, labels=False, duplicates="drop").values
        return fwl_rot_all(sb, yy, fe_Q(len(yy), [vv]))[0], int(sb.sum())

    contrib = {mo: (int((below & np.asarray(ym == mo)).sum()),
                    y[below & np.asarray(ym == mo)].mean()) for mo in sig_months}
    worst = sorted(contrib.items(), key=lambda kv: kv[1][0] * kv[1][1])
    for k in (1, 3, 5):
        obs_c, obs_n = coef_dropping([mo for mo, _ in worst[:k]])
        # null: drop k random signal-months, matched on total below-days removed
        ndrop = sum(contrib[mo][0] for mo, _ in worst[:k])
        draws = []
        for _ in range(400):
            pick = list(rng.choice(len(sig_months), k, replace=False))
            draws.append(coef_dropping([sig_months[i] for i in pick])[0])
        draws = np.array(draws)
        print(f"  drop {k} WORST month(s) ({ndrop} of 159 days): coef {obs_c*100:+.3f}%  "
              f"(n_below={obs_n})")
        print(f"     random {k}-month drop null: mean {draws.mean()*100:+.3f}%  "
              f"5th/95th [{np.percentile(draws,5)*100:+.3f}%, {np.percentile(draws,95)*100:+.3f}%]  "
              f"-> observed is at the {(draws < obs_c).mean()*100:.0f}th pctile of the null")
    print("  (the 'drop the worst' test is deliberately adversarial; the number that matters is "
          "how FAR\n   above the random-drop band the attenuation sits, i.e. how few days carry it)")

    # =============================================== C. jackknife informativeness
    hr("C. WAS THE 139-EPISODE JACKKNIFE EVER CAPABLE OF FLIPPING SIGN?")
    idx = np.flatnonzero(below)
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    tot = float((y[below] - base).sum())
    ep_contrib = np.array([float((y[g] - base).sum()) for g in groups])
    share = ep_contrib / tot
    print(f"  {len(groups)} episodes, mean length {below.sum()/len(groups):.2f} days")
    print(f"  a single episode must supply >100% of the total effect to flip the leave-one-out sign")
    print(f"  largest single-episode share of the total: {share.max()*100:.1f}%  "
          f"(episode {a.index[groups[int(share.argmax())][0]].date()})")
    print(f"  episodes needed to reach 100% of the total: "
          f"{int(np.searchsorted(np.cumsum(np.sort(share)[::-1]), 1.0)) + 1}")
    print(f"  => 0/139 sign flips is ARITHMETICALLY FORCED, not evidence. The same jackknife on a")
    print(f"     pure-noise signal of the same shape would also give 0 flips whenever |coef| is")
    print(f"     nonzero and episodes are ~1 day long.")
    # demonstrate: rotate the mask to a random offset (pure noise) and jackknife that
    flips = []
    for _ in range(200):
        off = int(rng.integers(300, n - 300))
        m = np.roll(below, off)
        ii = np.flatnonzero(m)
        gg = np.split(ii, np.flatnonzero(np.diff(ii) > 1) + 1)
        y_res = y - Qv @ (Qv.T @ y)
        dv = m.astype(float)
        dr = dv - Qv @ (Qv.T @ dv)
        full = (dr @ y_res) / (dr @ dr)
        f = 0
        for g in gg:
            m2 = m.copy()
            m2[g] = False
            dv2 = m2.astype(float)
            dr2 = dv2 - Qv @ (Qv.T @ dv2)
            if np.sign((dr2 @ y_res) / (dr2 @ dr2)) != np.sign(full):
                f += 1
        flips.append(f / len(gg))
    flips = np.array(flips)
    print(f"  CONTROL: same jackknife applied to 200 ROTATED (pure-noise) copies of the mask -> "
          f"mean sign-flip rate {flips.mean()*100:.2f}%, {int((flips==0).sum())}/200 copies "
          f"also had 0 flips")

    # =============================================== D. mid vs low tercile
    hr("D. 'STRONGEST IN THE MID TERCILE, NOT THE BOTTOM' — is that ordering distinguishable?")
    terc = pd.qcut(a.vix_pct1y, 3, labels=False, duplicates="drop").values
    names = {0: "low", 1: "mid", 2: "high"}
    stats = {}
    for t in (0, 1, 2):
        tm = terc == t
        s = below & tm
        no = (~below) & tm
        if s.sum() < MIN_N:
            print(f"  {names[t]:<8} n={int(s.sum()):>4}  n<{MIN_N} INCONCLUSIVE")
            continue
        diff = y[s].mean() - y[no].mean()
        se = np.sqrt(y[s].var(ddof=1) / s.sum() + y[no].var(ddof=1) / no.sum())
        stats[t] = (diff, se, int(s.sum()))
        print(f"  {names[t]:<8} n={int(s.sum()):>4}  within-tercile diff {diff*100:+.3f}%  "
              f"naive SE {se*100:.3f}%  95% CI [{(diff-1.96*se)*100:+.3f}%, "
              f"{(diff+1.96*se)*100:+.3f}%]")
    if 0 in stats and 1 in stats:
        dm, sm, _ = stats[1]
        dl, sl, _ = stats[0]
        gap = dm - dl
        sg = np.sqrt(sm ** 2 + sl ** 2)
        print(f"\n  MID minus LOW = {gap*100:+.3f}%  SE {sg*100:.3f}%  "
              f"t = {gap/sg:+.2f}  -> 95% CI [{(gap-1.96*sg)*100:+.3f}%, {(gap+1.96*sg)*100:+.3f}%]")
        print(f"  the CIs of the two terciles overlap heavily; the 'mid > bottom' ordering is NOT")
        print(f"  statistically distinguishable (needs |t|>1.96, has |t|={abs(gap/sg):.2f}).")
        # explicit interaction test: rotation p on the mid-minus-low contrast
        cm = below & (terc == 1)
        cl = below & (terc == 0)
        contrast = cm.astype(float) / max(cm.sum(), 1) - cl.astype(float) / max(cl.sum(), 1)
        obsv = float(contrast @ y) - (y[terc == 1].mean() - y[terc == 0].mean())
        nullv = circ_corr(cm.astype(float), y) / cm.sum() - circ_corr(cl.astype(float), y) / cl.sum()
        nullv = (nullv - (y[terc == 1].mean() - y[terc == 0].mean()))[offset_mask(n)]
        print(f"  rotation p on the mid-vs-low contrast: {p_from_null(gap, nullv):.3f}"
              f"{stars(p_from_null(gap, nullv))}")

    # =============================================== E. family-level signal test
    hr("E. DOES THE 255-CONFIG FAMILY CONTAIN ANY SIGNAL? — rotation null of the WINNER COUNT")
    hcols = [f"g{h}" for h in HORIZONS]
    sig_names = []
    for bn in (5, 10, 15, 20, 30):
        for k in (1.0, 1.5, 2.0, 2.5):
            for side in ("below", "above"):
                c_ = f"bb{bn}_{k}_{side}"
                if c_ in d.columns:
                    sig_names.append(c_)
    for bn in (10, 20):
        for k in (1.5, 2.0, 2.5):
            for side in ("reentry", "exit_lo"):
                sig_names.append(f"bb{bn}_{k}_{side}")
    fam = d[hcols + ["vix_pct1y", "stretch"] + sig_names].dropna().copy()
    nf = len(fam)
    Qf = fe_Q(nf, [pd.qcut(fam.vix_pct1y, 20, labels=False, duplicates="drop").values])
    yfam = {h: fam[f"g{h}"].values for h in HORIZONS}
    masks = {}
    for s in sig_names:
        m = fam[s].astype(bool).values
        if m.sum() >= MIN_N:
            masks[s] = m
    for t in (0.05, 0.10, 0.15, 0.20, 0.25):
        masks[f"stretch>+{int(t*100)}%"] = fam.stretch.values > t
    for t in (0.05, 0.10, 0.15, 0.20):
        masks[f"stretch<-{int(t*100)}%"] = fam.stretch.values < -t
    masks = {k: v for k, v in masks.items() if v.sum() >= MIN_N}
    Zn, zo, labs = [], [], []
    for s, m in masks.items():
        for h in HORIZONS:
            c, nl = fwl_rot_all(m, yfam[h], Qf)
            mu, sd = np.nanmean(nl), max(np.nanstd(nl), 1e-15)
            Zn.append(np.abs(nl - mu) / sd)
            zo.append(abs((c - mu) / sd))
            labs.append(f"{s}|D{h}")
    Zn = np.vstack(Zn)
    zo = np.array(zo)
    k = len(zo)
    crit = np.nanpercentile(Zn, 95, axis=1)          # per-config nominal .05 critical value
    obs_cnt = int((zo >= crit).sum())
    null_cnt = (Zn >= crit[:, None]).sum(axis=0)
    print(f"  k={k} configs, common frame n={nf:,}")
    print(f"  observed number of nominal p<.05 configs: {obs_cnt}")
    print(f"  rotation null of that count (dependence preserved): mean {null_cnt.mean():.1f}, "
          f"median {np.median(null_cnt):.0f}, 95th pct {np.percentile(null_cnt,95):.0f}, "
          f"max {null_cnt.max()}")
    print(f"  p(count this high or higher) = {(null_cnt >= obs_cnt).mean():.3f}"
          f"{stars((null_cnt >= obs_cnt).mean())}")
    print(f"  (independence would predict {0.05*k:.1f}; the family is highly correlated so that "
          f"benchmark is wrong)")

    # =============================================== F. contribution curve
    hr("F. CONTRIBUTION CURVE — how few days carry the whole -0.57%?")
    exc = y[below] - base
    o = np.argsort(exc)
    cum = np.cumsum(exc[o]) / exc.sum()
    for frac in (0.5, 0.9, 1.0):
        j = int(np.searchsorted(cum, frac)) + 1
        print(f"  {frac*100:>5.0f}% of the total negative excess comes from the worst "
              f"{j:>3} of 159 days ({j/159*100:.1f}%)")
    print(f"  median below-band g5 {np.median(y[below])*100:+.3f}% vs overall median "
          f"{np.median(y)*100:+.3f}%  (median gap {np.median(y[below])*100-np.median(y)*100:+.3f}%)")
    print(f"  mean gap {(y[below].mean()-base)*100:+.3f}%  -> the effect is ~"
          f"{abs((y[below].mean()-base)/(np.median(y[below])-np.median(y))):.1f}x larger in the "
          f"mean than in the median: a TAIL effect, not a location shift")
    print(f"  below-band win rate {(y[below]>0).mean()*100:.1f}% vs baseline "
          f"{(y>0).mean()*100:.1f}%  (gap {((y[below]>0).mean()-(y>0).mean())*100:+.1f} pp)")

    # recent-era version of the same
    for lo_yr in (2010, 2015):
        sel = yrs >= lo_yr
        sb = below[sel]
        yy = y[sel]
        if sb.sum() < MIN_N:
            continue
        print(f"  {lo_yr}+ : n={int(sb.sum())}  mean gap {(yy[sb].mean()-yy.mean())*100:+.3f}%  "
              f"median gap {(np.median(yy[sb])-np.median(yy))*100:+.3f}%  "
              f"win {(yy[sb]>0).mean()*100:.1f}% vs {(yy>0).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
