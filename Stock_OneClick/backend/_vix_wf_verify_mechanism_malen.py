"""
_vix_wf_verify_mechanism_malen.py — ADVERSARIAL verification of the "MA length is not a real
parameter" claim from _vix_wf_malen.py, attacked on MECHANISM / CONFOUNDING.

Claim under test:
  "once every length fires on the same number of days (top 12%, n~1108 each), no SMA length in
   3..40 is statistically distinguishable from SMA10 (paired rotation p=0.288 for L=5, p=0.205
   for L=20), and L=10's rank swings 2/38 -> 14/38 across halves (curve corr +0.352)."

Attacks:
  A. REPLICATION of the three headline paired p-values.
  B. POWER AUDIT: the paired rotation test compares two masks that share most of their days.
     What difference COULD it detect? Critical |diff| at alpha=.05 vs the observed spread of the
     whole length curve. Plus positive controls (known-different signals) and a synthetic
     injection to measure the true minimum detectable effect.
  C. SIMPLER-VARIABLE HORSE RACE at matched frequency (n~1108 each): VIX percentile, raw VIX
     level, 5d VIX change, trailing SPX returns (1/5/10/21d), SPX drawdown from 20d high,
     realized vol. Is "VIX stretched above its MA" just "SPX fell recently"?
  D. CONTROLS: residualise g5 within deciles of each simpler variable, then re-run (i) the SMA10
     top-12% cell and (ii) the WHOLE length curve. If the length curve only exists as a shadow of
     the control variable, the claim is right for the wrong reason; if a length effect EMERGES
     after control, the claim is wrong.
  E. DISAGREEMENT-DAY test: compare lengths only on the days where they actually disagree
     (max-power version of the paired test).
  F. EXACT null distribution of the half-sample curve correlation and of L=10's rank swing,
     computed over all 9,240 rotations via FFT. Is +0.352 / a 12-place rank move unusual, or is
     it exactly what noise produces?
  G. Era mechanism: VIX mean-reversion half-life per era vs the argmax MA length per era.
  H. Lookahead check: full-sample top-12% thresholds vs expanding-window thresholds.

Outcome throughout: g5 (SPX forward 5d, entry NEXT close). Every number is EXCESS over the
same-sample unconditional mean. Rotation inference reused from _vix_wf_malen.py in spirit.

Run: ../../vcp_env/bin/python _vix_wf_verify_mechanism_malen.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import _vix_data

warnings.filterwarnings("ignore", category=FutureWarning)

RNG = np.random.default_rng(20260910)
MIN_N = 25
LENGTHS = list(range(3, 41))


# ------------------------------------------------------------------ rotation machinery
def rot_curves(mask: np.ndarray, y: np.ndarray, valid: np.ndarray):
    """Return array of length n: conditional mean of y over (roll(mask,off) & valid), all offsets."""
    n = len(mask)
    a = np.where(valid, np.nan_to_num(y), 0.0)
    M = np.fft.rfft(mask.astype(float))
    num = np.fft.irfft(np.conj(M) * np.fft.rfft(a), n)
    den = np.round(np.fft.irfft(np.conj(M) * np.fft.rfft(valid.astype(float)), n), 6)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def rotation_full(mask, y):
    mask = np.asarray(mask, bool)
    valid = ~np.isnan(y)
    vals = rot_curves(mask, y, valid)
    obs = vals[0]
    null = vals[1:][~np.isnan(vals[1:])]
    if not len(null) or np.isnan(obs):
        return obs, float("nan"), int((mask & valid).sum())
    base = null.mean()
    return float(obs), float((np.abs(null - base) >= abs(obs - base)).mean()), int((mask & valid).sum())


def cell(mask, y):
    mask = np.asarray(mask, bool)
    valid = ~np.isnan(y)
    n = int((mask & valid).sum())
    if n < MIN_N:
        return {"n": n, "exc": np.nan, "p": np.nan, "cond": np.nan, "base": np.nan}
    obs, p, _ = rotation_full(mask, y)
    base = y[valid].mean()
    return {"n": n, "cond": obs, "base": base, "exc": obs - base, "p": p}


def paired_rotation(mask_a, mask_b, y, want_crit=False):
    """p for 'A and B have the same conditional mean'; both rolled by the same offset.
    If want_crit, also return the two-sided 5% critical |diff| implied by the null (= the MDE
    of this test at alpha=.05, ~50% power; and the 80%-power MDE ~ crit + 0.84*sd)."""
    a, b = np.asarray(mask_a, bool), np.asarray(mask_b, bool)
    valid = ~np.isnan(y)
    ca = rot_curves(a, y, valid)
    cb = rot_curves(b, y, valid)
    diff = ca - cb
    obs = diff[0]
    null = diff[1:][~np.isnan(diff[1:])]
    base = null.mean()
    p = float((np.abs(null - base) >= abs(obs - base)).mean())
    if not want_crit:
        return p
    crit = float(np.quantile(np.abs(null - base), 0.95))
    sd = float(null.std(ddof=1))
    return p, obs, crit, sd


def stars(p):
    if p != p:
        return "   "
    return "***" if p < 0.01 else ("** " if p < 0.05 else ("*  " if p < 0.10 else "   "))


def jac(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return float((a & b).sum() / max((a | b).sum(), 1))


def residualise(y, ctrl, nb=10):
    """y minus the mean of y within each decile of ctrl (deciles from the valid sample)."""
    y = np.asarray(y, float)
    ctrl = np.asarray(ctrl, float)
    ok = ~np.isnan(y) & ~np.isnan(ctrl)
    out = np.full(len(y), np.nan)
    if ok.sum() < 100:
        return out
    edges = np.quantile(ctrl[ok], np.linspace(0, 1, nb + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(ctrl, edges) - 1
    idx = np.clip(idx, 0, nb - 1)
    for b in range(nb):
        m = ok & (idx == b)
        if m.sum():
            out[m] = y[m] - y[m].mean()
    return out


def main():
    d = _vix_data.add_features(_vix_data.load())
    vix, spx = d.vix, d.spx
    n = len(d)
    y = d.g5.values
    valid = ~np.isnan(y)
    base = y[valid].mean()
    print(f"panel {n:,} rows  {d.index[0].date()} -> {d.index[-1].date()}")
    print(f"outcome g5: n_valid={valid.sum():,}  unconditional mean {base*100:+.4f}%")

    st = {L: (vix / vix.rolling(L).mean() - 1.0) for L in LENGTHS}
    q88 = {L: st[L].dropna().quantile(0.88) for L in LENGTHS}
    m12 = {L: (st[L] >= q88[L]).values for L in LENGTHS}
    ema10 = vix.ewm(span=10, adjust=False, min_periods=10).mean()
    e10 = vix / ema10 - 1.0
    m12_e10 = (e10 >= e10.dropna().quantile(0.88)).values

    # ============================================================ A. replication
    print("\n" + "=" * 104)
    print("A. REPLICATION of the three headline paired p-values (top-12% matched frequency)")
    print("=" * 104)
    for lbl, mA in (("SMA5", m12[5]), ("SMA20", m12[20]), ("EMA10", m12_e10)):
        rA, rB = cell(mA, y), cell(m12[10], y)
        p, obs, crit, sd = paired_rotation(mA, m12[10], y, want_crit=True)
        print(f"  {lbl:>6} top12% vs SMA10 top12%: nA={rA['n']} nB={rB['n']}  "
              f"diff {obs*100:+.3f}%  p={p:.4f}  |  Jaccard {jac(mA, m12[10]):.3f}")

    # ============================================================ B. power audit
    print("\n" + "=" * 104)
    print("B. POWER AUDIT — what difference could this paired test actually detect?")
    print("=" * 104)
    curve = {L: cell(m12[L], y)["exc"] for L in LENGTHS}
    arr = np.array([curve[L] for L in LENGTHS])
    print(f"  matched top-12% length curve: mean {arr.mean()*100:+.3f}%  sd {arr.std(ddof=1)*100:.3f}%  "
          f"min {arr.min()*100:+.3f}% (L={LENGTHS[int(np.argmin(arr))]})  "
          f"max {arr.max()*100:+.3f}% (L={LENGTHS[int(np.argmax(arr))]})  "
          f"FULL RANGE {(arr.max()-arr.min())*100:.3f}%")
    print(f"\n  {'pair':<26}{'Jaccard':>9}{'symdiff':>9}{'obs diff':>11}{'crit|d|@.05':>13}"
          f"{'MDE 80%pw':>11}{'p':>9}")
    for L in (3, 5, 8, 15, 20, 30, 40):
        p, obs, crit, sd = paired_rotation(m12[L], m12[10], y, want_crit=True)
        sym = int((m12[L] ^ m12[10]).sum())
        print(f"  SMA{L} vs SMA10 top12%{'':<{max(0,6-len(str(L)))}}{jac(m12[L],m12[10]):>9.3f}{sym:>9}"
              f"{obs*100:>10.3f}%{crit*100:>12.3f}%{(crit+0.84*sd)*100:>10.3f}%{p:>9.4f}{stars(p)}")
    print("\n  POSITIVE CONTROLS (signals that really are different -> can the test see them?):")
    bot12 = {L: (st[L] <= st[L].dropna().quantile(0.12)).values for L in (10,)}
    ctrls = [
        ("SMA10 top12% vs SMA10 bot12%", m12[10], bot12[10]),
        ("SMA10 top12% vs random 12%",   m12[10],
         (pd.Series(RNG.random(n)) <= 0.12).values),
        ("SMA10 top12% vs SMA10 top2%",  m12[10],
         (st[10] >= st[10].dropna().quantile(0.98)).values),
    ]
    for lbl, a, b in ctrls:
        p, obs, crit, sd = paired_rotation(a, b, y, want_crit=True)
        print(f"  {lbl:<32} diff {obs*100:+.3f}%  crit {crit*100:.3f}%  p={p:.4f}{stars(p)}  "
              f"Jaccard {jac(a,b):.3f}  (nB={int((np.asarray(b)&valid).sum())})")

    print("\n  SYNTHETIC INJECTION — take SMA10's top-12% mask, swap a fraction of its days for the")
    print("  days with the best/worst realised g5 (so the true diff is KNOWN), and see when the")
    print("  paired test rejects. This measures the real MDE, not an asymptotic guess.")
    idx_sel = np.flatnonzero(m12[10] & valid)
    order = np.argsort(y[idx_sel])          # worst -> best inside the signal
    pool = np.flatnonzero(~m12[10] & valid)
    pool_order = pool[np.argsort(y[pool])]  # worst -> best outside
    print(f"  {'swap k days':>12}{'true diff':>12}{'p':>10}")
    for k in (10, 25, 50, 100, 200, 400):
        b = m12[10].copy()
        b[idx_sel[order[:k]]] = False            # drop the k worst
        b[pool_order[-k:]] = True                # add the k best outside days
        rb = cell(b, y)
        p = paired_rotation(b, m12[10], y)
        print(f"  {k:>12}{(rb['cond']-cell(m12[10],y)['cond'])*100:>11.3f}%{p:>10.4f}{stars(p)}")

    # ============================================================ C. horse race
    print("\n" + "=" * 104)
    print("C. SIMPLER-VARIABLE HORSE RACE at matched frequency (top/bottom 12%, n~1108 each)")
    print("=" * 104)
    r1 = spx.pct_change(1)
    r5 = spx / spx.shift(5) - 1.0
    r10 = spx / spx.shift(10) - 1.0
    r21 = spx / spx.shift(21) - 1.0
    dd20 = spx / spx.rolling(20).max() - 1.0
    rv10 = spx.pct_change().rolling(10).std(ddof=0) * np.sqrt(252)
    vixchg5 = vix / vix.shift(5) - 1.0
    cands = [
        ("SMA10 stretch  hi", st[10], "hi"),
        ("raw VIX level  hi", vix, "hi"),
        ("VIX 1y pctile  hi", d.vix_pct1y, "hi"),
        ("VIX 5d change  hi", vixchg5, "hi"),
        ("SPX 1d ret     lo", r1, "lo"),
        ("SPX 5d ret     lo", r5, "lo"),
        ("SPX 10d ret    lo", r10, "lo"),
        ("SPX 21d ret    lo", r21, "lo"),
        ("SPX dd-20d-hi  lo", dd20, "lo"),
        ("realised vol10 hi", rv10, "hi"),
    ]
    masks = {}
    print(f"  {'signal':<20}{'n':>6}{'D5 exc':>10}{'p':>9}   {'Jac vs SMA10-stretch':>21}")
    for lbl, s, side in cands:
        thr = s.dropna().quantile(0.88 if side == "hi" else 0.12)
        m = (s >= thr).values if side == "hi" else (s <= thr).values
        masks[lbl] = m
        r = cell(m, y)
        print(f"  {lbl:<20}{r['n']:>6}{r['exc']*100:>9.3f}%{r['p']:>9.4f}{stars(r['p'])}"
              f"{jac(m, m12[10]):>18.3f}")

    print("\n  2x2 decomposition — SMA10-stretch top12% vs SPX-5d-return bottom12% "
          "(each day counted once):")
    A, B = m12[10], masks["SPX 5d ret     lo"]
    for lbl, m in (("both", A & B), ("stretch only", A & ~B), ("spx-drop only", ~A & B),
                   ("neither", ~A & ~B)):
        r = cell(m, y)
        txt = (f"exc {r['exc']*100:+7.3f}%  p={r['p']:.4f}{stars(r['p'])}"
               if r["n"] >= MIN_N else "(n<25 inconclusive)")
        print(f"    {lbl:<14} n={r['n']:>5}  {txt}")

    # ============================================================ D. controls
    print("\n" + "=" * 104)
    print("D. CONTROL — residualise g5 inside deciles of each simpler variable, then re-test")
    print("=" * 104)
    controls = [("none", None), ("SPX 5d ret", r5.values), ("SPX 10d ret", r10.values),
                ("SPX 21d ret", r21.values), ("SPX dd-20d", dd20.values),
                ("raw VIX", vix.values), ("VIX 1y pct", d.vix_pct1y.values),
                ("realised vol10", rv10.values), ("VIX 5d chg", vixchg5.values)]
    print(f"  {'control':<16}{'SMA10 top12% exc':>18}{'p':>9}   "
          f"{'curve mean':>11}{'curve sd':>10}{'argmax':>8}{'L10 rank':>10}{'range':>9}")
    resid_store = {}
    for lbl, c in controls:
        yy = y.copy() if c is None else residualise(y, c)
        resid_store[lbl] = yy
        r = cell(m12[10], yy)
        cur = np.array([cell(m12[L], yy)["exc"] for L in LENGTHS])
        rank10 = int(np.argsort(-cur).tolist().index(LENGTHS.index(10))) + 1
        print(f"  {lbl:<16}{r['exc']*100:>17.3f}%{r['p']:>9.4f}{stars(r['p'])}"
              f"{cur.mean()*100:>10.3f}%{cur.std(ddof=1)*100:>9.3f}%"
              f"{LENGTHS[int(np.argmax(cur))]:>8}{rank10:>10}{(cur.max()-cur.min())*100:>8.3f}%")
    print("  (exc on residualised g5 = the part of the signal NOT explained by the control var.)")

    print("\n  Paired L-vs-10 tests ON THE RESIDUALISED OUTCOME (control = SPX 5d ret):")
    yy = resid_store["SPX 5d ret"]
    for L in (3, 5, 20, 40):
        p, obs, crit, sd = paired_rotation(m12[L], m12[10], yy, want_crit=True)
        print(f"    SMA{L:<2} vs SMA10: diff {obs*100:+.3f}%  crit {crit*100:.3f}%  p={p:.4f}{stars(p)}")

    # ============================================================ E. disagreement days
    print("\n" + "=" * 104)
    print("E. DISAGREEMENT-DAY TEST — compare lengths only where they actually disagree")
    print("=" * 104)
    print(f"  {'pair':<18}{'n(L only)':>10}{'mean':>9}{'n(10 only)':>11}{'mean':>9}"
          f"{'diff':>9}{'p':>9}")
    for L in (3, 4, 5, 8, 15, 20, 30, 40):
        aonly = m12[L] & ~m12[10]
        bonly = m12[10] & ~m12[L]
        na, nb = int((aonly & valid).sum()), int((bonly & valid).sum())
        if min(na, nb) < MIN_N:
            print(f"  SMA{L:<3}vs SMA10 {na:>10}{'':>9}{nb:>11}{'':>9}   (n<25 inconclusive)")
            continue
        ma_, mb_ = y[aonly & valid].mean(), y[bonly & valid].mean()
        p = paired_rotation(aonly, bonly, y)
        print(f"  SMA{L:<3}vs SMA10 {na:>10}{ma_*100:>8.3f}%{nb:>11}{mb_*100:>8.3f}%"
              f"{(ma_-mb_)*100:>8.3f}%{p:>9.4f}{stars(p)}")

    # ============================================================ F. half-sample null
    print("\n" + "=" * 104)
    print("F. HALF-SAMPLE CURVE CORRELATION — exact null over all 9,240 rotations")
    print("=" * 104)
    mid = n // 2
    h1 = np.zeros(n, bool); h1[:mid] = True
    h2 = ~h1
    v1, v2 = valid & h1, valid & h2
    b1, b2 = y[v1].mean(), y[v2].mean()
    C1 = np.array([rot_curves(m12[L], y, v1) for L in LENGTHS])   # 38 x n
    C2 = np.array([rot_curves(m12[L], y, v2) for L in LENGTHS])
    # counts per offset to enforce MIN_N
    N1 = np.array([np.round(np.fft.irfft(np.conj(np.fft.rfft(m12[L].astype(float)))
                                         * np.fft.rfft(v1.astype(float)), n)) for L in LENGTHS])
    N2 = np.array([np.round(np.fft.irfft(np.conj(np.fft.rfft(m12[L].astype(float)))
                                         * np.fft.rfft(v2.astype(float)), n)) for L in LENGTHS])
    okoff = (np.nanmin(N1, axis=0) >= MIN_N) & (np.nanmin(N2, axis=0) >= MIN_N) \
        & ~np.isnan(C1).any(axis=0) & ~np.isnan(C2).any(axis=0)
    A1, A2 = C1 - b1, C2 - b2
    A1c = A1 - A1.mean(axis=0, keepdims=True)
    A2c = A2 - A2.mean(axis=0, keepdims=True)
    corr = (A1c * A2c).sum(axis=0) / np.sqrt((A1c ** 2).sum(axis=0) * (A2c ** 2).sum(axis=0))
    obs_corr = corr[0]
    null_corr = corr[1:][okoff[1:]]
    i10 = LENGTHS.index(10)
    rank1 = (-A1).argsort(axis=0).argsort(axis=0)[i10] + 1
    rank2 = (-A2).argsort(axis=0).argsort(axis=0)[i10] + 1
    print(f"  observed: half1 1990-{d.index[mid].year} argmax L={LENGTHS[int(np.argmax(A1[:,0]))]}, "
          f"half2 argmax L={LENGTHS[int(np.argmax(A2[:,0]))]}, "
          f"L=10 rank {rank1[0]}/38 -> {rank2[0]}/38, curve corr {obs_corr:+.3f}")
    print(f"  null (mask rotated, {len(null_corr):,} valid offsets): mean corr {null_corr.mean():+.3f}  "
          f"sd {null_corr.std():.3f}  q05 {np.quantile(null_corr,0.05):+.3f}  "
          f"q50 {np.quantile(null_corr,0.50):+.3f}  q95 {np.quantile(null_corr,0.95):+.3f}")
    print(f"  P(null corr >= observed {obs_corr:+.3f}) = {(null_corr >= obs_corr).mean():.4f}")
    rs = np.abs(rank1[1:][okoff[1:]] - rank2[1:][okoff[1:]])
    print(f"  |rank swing| of L=10: observed {abs(int(rank1[0])-int(rank2[0]))}  |  "
          f"null median {np.median(rs):.0f}  mean {rs.mean():.1f}  "
          f"P(null swing >= observed) = {(rs >= abs(int(rank1[0])-int(rank2[0]))).mean():.3f}")
    # how much of the curve's cross-L variance is even estimable?
    print(f"  per-point noise: sd across rotations of a single length's half-sample excess = "
          f"{np.nanstd(A1[i10][okoff]) * 100:.3f}% (half1), {np.nanstd(A2[i10][okoff]) * 100:.3f}% (half2)"
          f"  vs cross-L sd of the FULL-sample curve {arr.std(ddof=1)*100:.3f}%")

    # ============================================================ G. era mechanism
    print("\n" + "=" * 104)
    print("G. MECHANISM — does the best MA length track the VIX's own mean-reversion speed?")
    print("=" * 104)
    lv = np.log(vix)
    print(f"  {'era':<12}{'n':>7}{'AR1(logVIX)':>13}{'half-life d':>13}{'argmax L top12%':>17}"
          f"{'L10 exc':>10}")
    for lbl, y0, y1 in (("1990-2008", 1990, 2008), ("2009-2026", 2009, 2026),
                        ("1990s", 1990, 1999), ("2000s", 2000, 2009),
                        ("2010s", 2010, 2019), ("2020s", 2020, 2099)):
        sub = ((d.index.year >= y0) & (d.index.year <= y1))
        xx = lv[sub]
        a = np.polyfit(xx.values[:-1], xx.values[1:], 1)[0]
        hl = np.log(0.5) / np.log(a) if 0 < a < 1 else np.nan
        sv = sub & valid
        bb = y[sv].mean()
        cur, ns = {}, {}
        for L in LENGTHS:
            mm = m12[L] & sv
            ns[L] = int(mm.sum())
            cur[L] = (y[mm].mean() - bb) if mm.sum() >= MIN_N else np.nan
        good = [L for L in LENGTHS if not np.isnan(cur[L])]
        am = max(good, key=lambda L: cur[L]) if good else np.nan
        print(f"  {lbl:<12}{int(sub.sum()):>7}{a:>13.4f}{hl:>13.1f}"
              f"{f'L={am} ({cur[am]*100:+.3f}%)':>17}{cur.get(10, np.nan)*100:>9.3f}%")

    # ============================================================ H. lookahead
    print("\n" + "=" * 104)
    print("H. LOOKAHEAD — full-sample top-12% thresholds vs EXPANDING-window (point-in-time)")
    print("=" * 104)
    print(f"  {'L':>4}{'full-sample exc':>18}{'n':>7}   {'expanding exc':>16}{'n':>7}{'p':>9}")
    for L in (3, 5, 10, 20, 40):
        s = st[L]
        exp_thr = s.expanding(min_periods=500).quantile(0.88).shift(1)
        me = (s >= exp_thr).values & ~np.isnan(exp_thr.values)
        rf, re = cell(m12[L], y), cell(me, y)
        print(f"  {L:>4}{rf['exc']*100:>17.3f}%{rf['n']:>7}   {re['exc']*100:>15.3f}%{re['n']:>7}"
              f"{re['p']:>9.4f}{stars(re['p'])}")
    print("\n  Expanding-window length curve (point-in-time thresholds, no lookahead):")
    cur_e = {}
    for L in LENGTHS:
        s = st[L]
        exp_thr = s.expanding(min_periods=500).quantile(0.88).shift(1)
        me = (s >= exp_thr).values & ~np.isnan(exp_thr.values)
        cur_e[L] = cell(me, y)["exc"]
    ce = np.array([cur_e[L] for L in LENGTHS])
    print(f"    mean {ce.mean()*100:+.3f}%  sd {ce.std(ddof=1)*100:.3f}%  "
          f"argmax L={LENGTHS[int(np.argmax(ce))]} ({ce.max()*100:+.3f}%)  "
          f"L=10 {cur_e[10]*100:+.3f}% rank {sorted(LENGTHS, key=lambda L: -cur_e[L]).index(10)+1}/38  "
          f"corr with full-sample curve {np.corrcoef(ce, arr)[0,1]:+.3f}")

    # ============================================================ I. global structure test
    print("\n" + "=" * 104)
    print("I. GLOBAL TEST — is there ANY length structure? Rotate ALL 38 masks by the SAME offset")
    print("   (preserves their mutual overlap geometry) and look at the cross-L spread of the")
    print("   resulting curve. This is the omnibus version of the 38 underpowered pairwise tests.")
    print("=" * 104)
    for ylbl, yy in (("raw g5", y), ("g5 | SPX5d-decile", resid_store["SPX 5d ret"]),
                     ("g5 | VIX1ypct-decile", resid_store["VIX 1y pct"])):
        vv = ~np.isnan(yy)
        CF = np.array([rot_curves(m12[L], yy, vv) for L in LENGTHS])
        NF = np.array([np.round(np.fft.irfft(np.conj(np.fft.rfft(m12[L].astype(float)))
                                             * np.fft.rfft(vv.astype(float)), n)) for L in LENGTHS])
        ok = (np.nanmin(NF, axis=0) >= MIN_N) & ~np.isnan(CF).any(axis=0)
        sd_off = CF.std(axis=0, ddof=1)
        rng_off = CF.max(axis=0) - CF.min(axis=0)
        obs_sd, obs_rng = sd_off[0], rng_off[0]
        nl_sd, nl_rng = sd_off[1:][ok[1:]], rng_off[1:][ok[1:]]
        print(f"  {ylbl:<22} cross-L sd obs {obs_sd*100:.3f}%  null median {np.median(nl_sd)*100:.3f}% "
              f"q95 {np.quantile(nl_sd,0.95)*100:.3f}%  p={(nl_sd >= obs_sd).mean():.4f}"
              f"   | range obs {obs_rng*100:.3f}% p={(nl_rng >= obs_rng).mean():.4f}")

    print("\n  Paired tests on the CURVE EXTREMES (the biggest length gaps in the data), and a")
    print("  pooled short-MA vs long-MA test (more power than any single pair):")
    order_curve = sorted(LENGTHS, key=lambda L: -curve[L])
    Lbest, Lworst = order_curve[0], order_curve[-1]
    extra = [(Lbest, Lworst), (5, 24), (5, 20), (5, 30), (3, 24)]
    for a_, b_ in extra:
        p, obs, crit, sd = paired_rotation(m12[a_], m12[b_], y, want_crit=True)
        print(f"    SMA{a_:<2} vs SMA{b_:<2}: obs {obs*100:+.3f}%  crit@.05 {crit*100:.3f}%  "
              f"p={p:.4f}{stars(p)}  Jaccard {jac(m12[a_], m12[b_]):.3f}")
    short = np.array([m12[L] for L in (4, 5, 6, 7)]).mean(axis=0) >= 0.5
    long_ = np.array([m12[L] for L in (22, 24, 26, 28)]).mean(axis=0) >= 0.5
    rs_, rl_ = cell(short, y), cell(long_, y)
    p = paired_rotation(short, long_, y)
    print(f"    pooled SHORT(4-7) n={rs_['n']} exc {rs_['exc']*100:+.3f}% vs "
          f"LONG(22-28) n={rl_['n']} exc {rl_['exc']*100:+.3f}%  diff "
          f"{(rs_['cond']-rl_['cond'])*100:+.3f}%  p={p:.4f}{stars(p)}  "
          f"Jaccard {jac(short, long_):.3f}")
    so, lo = short & ~long_, long_ & ~short
    print(f"    disagreement only: short-only n={int((so&valid).sum())} mean {y[so&valid].mean()*100:+.3f}%"
          f" vs long-only n={int((lo&valid).sum())} mean {y[lo&valid].mean()*100:+.3f}%"
          f"  diff {(y[so&valid].mean()-y[lo&valid].mean())*100:+.3f}%  "
          f"p={paired_rotation(so, lo, y):.4f}")

    print("\n  Did the fixed-threshold SMA20-vs-SMA10 gap really 'vanish', or did the test just")
    print("  get wider? (effect size vs critical value, fixed +10% vs matched top-12%)")
    f20, f10 = (st[20] >= 0.10).values, (st[10] >= 0.10).values
    p, obs, crit, sd = paired_rotation(f20, f10, y, want_crit=True)
    print(f"    fixed +10%   : nA={int((f20&valid).sum())} nB={int((f10&valid).sum())} "
          f"diff {obs*100:+.3f}%  crit {crit*100:.3f}%  p={p:.4f}{stars(p)}  Jaccard {jac(f20,f10):.3f}")
    p, obs, crit, sd = paired_rotation(m12[20], m12[10], y, want_crit=True)
    print(f"    matched top12%: nA={cell(m12[20],y)['n']} nB={cell(m12[10],y)['n']} "
          f"diff {obs*100:+.3f}%  crit {crit*100:.3f}%  p={p:.4f}{stars(p)}  "
          f"Jaccard {jac(m12[20],m12[10]):.3f}")
    # frequency-match SMA20 to SMA10's EXACT fixed-threshold count instead of a 12% quantile
    k = int((f10 & valid).sum())
    thr20 = st[20].dropna().nlargest(k).min()
    m20k = (st[20] >= thr20).values
    p, obs, crit, sd = paired_rotation(m20k, f10, y, want_crit=True)
    print(f"    SMA20 re-cut to SMA10's own n={k}: diff {obs*100:+.3f}%  crit {crit*100:.3f}%  "
          f"p={p:.4f}{stars(p)}  (SMA20 thr {thr20*100:.2f}% vs SMA10 thr 10.00%)")

    # ============================================================ J. monotone trend in L
    print("\n" + "=" * 104)
    print("J. MONOTONE-TREND TEST — 'shorter MA is better' is ONE hypothesis with one d.o.f., not")
    print("   38 pairwise picks. Statistic = Spearman(L, excess) across L=3..40; null = same-offset")
    print("   rotation of all 38 masks (so multiplicity across L is built into the null).")
    print("=" * 104)
    Lrank = pd.Series(LENGTHS, dtype=float).rank().values
    Lrank = (Lrank - Lrank.mean()) / Lrank.std()

    def trend_test(yy, lbl, subset=None):
        vv = ~np.isnan(yy)
        if subset is not None:
            vv = vv & subset
        CF = np.array([rot_curves(m12[L], yy, vv) for L in LENGTHS])
        NF = np.array([np.round(np.fft.irfft(np.conj(np.fft.rfft(m12[L].astype(float)))
                                             * np.fft.rfft(vv.astype(float)), n)) for L in LENGTHS])
        okc = (np.nanmin(NF, axis=0) >= MIN_N) & ~np.isnan(CF).any(axis=0)
        R = np.apply_along_axis(lambda c: pd.Series(c).rank().values, 0, CF)
        R = (R - R.mean(axis=0)) / R.std(axis=0)
        rho = (R * Lrank[:, None]).mean(axis=0)
        obs = rho[0]
        null = rho[1:][okc[1:]]
        p_lo = float((null <= obs).mean())
        print(f"  {lbl:<26} Spearman(L, exc) = {obs:+.3f}   null mean {null.mean():+.3f} "
              f"sd {null.std():.3f}  q05 {np.quantile(null,0.05):+.3f}   "
              f"P(null <= obs) = {p_lo:.4f}{stars(2*min(p_lo,1-p_lo))}")
        return obs, p_lo

    trend_test(y, "raw g5 (full sample)")
    trend_test(resid_store["SPX 5d ret"], "g5 | SPX5d-decile")
    trend_test(resid_store["VIX 1y pct"], "g5 | VIX1ypct-decile")
    trend_test(resid_store["raw VIX"], "g5 | rawVIX-decile")
    trend_test(y, "raw g5, half1 (1990-2008)", subset=h1)
    trend_test(y, "raw g5, half2 (2008-2026)", subset=h2)
    yr = d.index.year.values
    trend_test(y, "raw g5, 2009-2026 only", subset=(yr >= 2009))
    trend_test(y, "raw g5, ex-2008 & ex-2020", subset=~((yr == 2008) | (yr == 2020)))
    ce_rho = spearman_simple(LENGTHS, [cur_e[L] for L in LENGTHS])
    print(f"  {'expanding-window curve':<26} Spearman(L, exc) = {ce_rho:+.3f} "
          f"(point-in-time thresholds, no rotation null run)")


def spearman_simple(a, b):
    ra = pd.Series(a, dtype=float).rank().values
    rb = pd.Series(b, dtype=float).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


if __name__ == "__main__":
    main()
