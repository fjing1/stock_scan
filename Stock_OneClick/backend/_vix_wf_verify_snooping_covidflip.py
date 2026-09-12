"""
_vix_wf_verify_snooping_covidflip.py — ADVERSARIAL verification of the claim:

  "The 2020s negative sign on every bullish variant is 24 COVID days, not decay: excluding
   2020-02-01..2020-06-30 flips stretch>=+10% from -0.172pp to +0.210pp excess and
   stretch>=+20% from -0.365pp to +0.800pp (p=0.029), and all four bullish signals flip positive."

LENS: data snooping. The exclusion window (start, end) is itself a searched parameter, and
"drop the worst months" mechanically improves any mean. Tests run here:

  T0  reproduce the headline numbers.
  T1  exclusion-window grid: how much of +0.800pp depends on picking Feb01..Jun30 2020?
  T2  placebo "drop the best contiguous 5-month window" applied to EVERY era, and the null
      distribution of that improvement over all 5-month windows (the snooping-corrected p).
  T3  drop single worst month / drop 2008 / drop 2020 / drop both, full sample and per era.
  T4  family-wise multiple testing: Romano-Wolf / White-reality-check style max-statistic over
      the whole signal x horizon x era family, using SHARED circular rotations.
  T5  precision: block-bootstrap CI on the ex-COVID 2020s excess (is n=72 enough for "+0.800pp"?).
  T6  decay vs COVID: is ex-COVID 2020s statistically distinguishable from the pre-2020 excess,
      and does it survive splitting 2020s ex-COVID in half?

Run: ../../vcp_env/bin/python _vix_wf_verify_snooping_covidflip.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _vix_data

RNG = np.random.default_rng(20260910)
HORIZONS = [1, 3, 5, 10, 21]
ERAS = [("1990s", 1990, 1999), ("2000s", 2000, 2009),
        ("2010s", 2010, 2019), ("2020s", 2020, 2099)]
MIN_N = 25


def hr(t=""):
    print("\n" + "=" * 108)
    if t:
        print(t)
        print("=" * 108)


# ------------------------------------------------------------------ fast rotation machinery
def circ_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """c[off] = sum_j a[j] * b[(j+off) % n]  ==  sum_i roll(a,off)[i] * b[i]."""
    n = len(a)
    return np.fft.irfft(np.conj(np.fft.rfft(a)) * np.fft.rfft(b), n)


def rotation_null_means(mask: np.ndarray, fwd: np.ndarray, keep: np.ndarray, min_n=MIN_N):
    """Conditional mean of fwd over (rotated mask) & keep & valid, for EVERY circular offset.
    Exact enumeration of the rotation null (n-1 draws) instead of 5000 random ones."""
    valid = ~np.isnan(fwd)
    w = (keep & valid).astype(np.float64)
    y = np.where(valid, np.nan_to_num(fwd), 0.0) * w
    m = mask.astype(np.float64)
    num = circ_corr(m, y)
    den = circ_corr(m, w)
    den_r = np.rint(den)
    out = np.where(den_r >= min_n, num / np.where(den_r > 0, den, np.nan), np.nan)
    return out[1:]                       # drop offset 0 (the observed configuration)


def rot_p(mask, fwd, keep, observed, min_n=MIN_N):
    null = rotation_null_means(mask, fwd, keep, min_n)
    null = null[~np.isnan(null)]
    if len(null) < 100:
        return float("nan")
    base = null.mean()
    return float((np.abs(null - base) >= abs(observed - base)).mean())


def stats(d, mask, keep, h=5, min_n=MIN_N, with_p=True):
    """excess of conditional mean over the SAME-SAMPLE unconditional mean, restricted to `keep`."""
    fwd = d[f"g{h}"].values
    valid = ~np.isnan(fwd)
    k = keep & valid
    sel = mask & k
    n = int(sel.sum())
    if n < min_n:
        return dict(n=n, exc=np.nan, cond=np.nan, base=np.nan, p=np.nan, ok=False)
    cond = fwd[sel].mean()
    base = fwd[k].mean()
    p = rot_p(mask, fwd, keep, cond, min_n) if with_p else np.nan
    return dict(n=n, exc=cond - base, cond=cond, base=base, p=p, ok=True)


# ------------------------------------------------------------------ data + signals
d = _vix_data.add_features(_vix_data.load())
idx = d.index
N = len(d)
print(f"panel: {N:,} rows  {idx[0].date()} -> {idx[-1].date()}")

SIGNALS = {
    "stretch>=+10%":       (d.stretch >= 0.10).values,
    "stretch>=+20%":       (d.stretch >= 0.20).values,
    "stretch>=+25%":       (d.stretch >= 0.25).values,
    "BB(10,2.0) above":    d["bb10_2.0_above"].values,
    "BB(10,2.0) re-entry": d["bb10_2.0_reentry"].values,
}
BULL4 = ["stretch>=+10%", "stretch>=+20%", "BB(10,2.0) above", "BB(10,2.0) re-entry"]

is20s = np.asarray(idx.year >= 2020)
COVID = ((idx >= "2020-02-01") & (idx <= "2020-06-30"))
keep20 = is20s
keep20_ex = is20s & ~COVID

# ------------------------------------------------------------------ T0 reproduce
hr("T0 — REPRODUCTION of the claim's headline numbers (2020-2026 block, g5)")
print(f"{'signal':<22}{'n all':>7}{'exc all':>10}{'p':>8}   {'n exC':>7}{'exc exC':>10}{'p':>8}"
      f"   {'ndrop':>6}{'drop mean g5':>14}{'drop worst':>12}")
print("-" * 108)
t0 = {}
for s in BULL4:
    m = SIGNALS[s]
    a = stats(d, m, keep20)
    b = stats(d, m, keep20_ex)
    dropm = m & COVID & ~np.isnan(d.g5.values)
    dg = d.g5.values[dropm]
    t0[s] = (a, b)
    print(f"{s:<22}{a['n']:>7}{a['exc']*100:>9.3f}pp{a['p']:>8.3f}   "
          f"{b['n']:>7}{b['exc']*100:>9.3f}pp{b['p']:>8.3f}   "
          f"{dropm.sum():>6}{dg.mean()*100:>13.2f}%{dg.min()*100:>11.1f}%")
print("\n  NOTE the deletion fraction: stretch>=+20% loses "
      f"{(SIGNALS['stretch>=+20%'] & COVID).sum()}/{(SIGNALS['stretch>=+20%'] & is20s).sum()} "
      f"= {(SIGNALS['stretch>=+20%'] & COVID).sum()/(SIGNALS['stretch>=+20%'] & is20s).sum()*100:.0f}% "
      "of its 2020s observations.")

# ------------------------------------------------------------------ T1 window grid
hr("T1 — HOW MUCH IS THE WINDOW ITSELF SNOOPED? grid of exclusion windows (2020s block, g5)")
starts = ["2020-01-01", "2020-02-01", "2020-02-15", "2020-03-01", "2020-03-15"]
ends = ["2020-03-31", "2020-04-30", "2020-05-31", "2020-06-30", "2020-07-31",
        "2020-09-30", "2020-12-31", "2021-12-31"]
for s in ["stretch>=+10%", "stretch>=+20%"]:
    m = SIGNALS[s]
    print(f"\n  {s}  (all-2020s excess {t0[s][0]['exc']*100:+.3f}pp, n={t0[s][0]['n']})")
    print("    excl start |" + "".join(f"{e[2:]:>13}" for e in ends))
    for st in starts:
        cells = []
        for en in ends:
            kp = is20s & ~((idx >= st) & (idx <= en))
            r = stats(d, m, kp, with_p=False)
            cells.append(f"{r['exc']*100:+7.3f}(n{r['n']})" if r["ok"] else f"{'n<25':>13}")
        print(f"    {st[2:]} |" + "".join(f"{c:>13}" for c in cells))

# how many window configs were available, and where does the reported one rank?
hr("T1b — rank of the CHOSEN window among all plausible contiguous exclusion windows in 2020-2021")
months = pd.period_range("2020-01", "2021-12", freq="M")
for s in ["stretch>=+10%", "stretch>=+20%"]:
    m = SIGNALS[s]
    recs = []
    for i in range(len(months)):
        for j in range(i, len(months)):
            st = months[i].start_time
            en = months[j].end_time
            kp = is20s & ~((idx >= st) & (idx <= en))
            r = stats(d, m, kp, with_p=False)
            if r["ok"]:
                recs.append((r["exc"], f"{months[i]}..{months[j]}", r["n"]))
    recs.sort(reverse=True)
    chosen = [r for r in recs if r[1] == "2020-02..2020-06"]
    rk = recs.index(chosen[0]) + 1 if chosen else -1
    print(f"\n  {s}: {len(recs)} valid contiguous month-windows searched-over.")
    print(f"    best   : {recs[0][0]*100:+.3f}pp  ({recs[0][1]}, n={recs[0][2]})")
    print(f"    chosen : {chosen[0][0]*100:+.3f}pp  (2020-02..2020-06, n={chosen[0][2]})  "
          f"rank {rk}/{len(recs)} = top {rk/len(recs)*100:.0f}%")
    print(f"    median : {np.median([r[0] for r in recs])*100:+.3f}pp   "
          f"worst: {recs[-1][0]*100:+.3f}pp ({recs[-1][1]})")

# ------------------------------------------------------------------ T2 placebo across eras
hr("T2 — PLACEBO: 'drop the most flattering contiguous 5-month window' applied to EVERY era")
print("  If deleting the worst 5 months improves EVERY era by a similar amount, then 'COVID")
print("  explains the 2020s' is not an explanation -- it is arithmetic.\n")
all_months = pd.period_range(idx[0].to_period("M"), idx[-1].to_period("M"), freq="M")
mstart = np.array([p.start_time for p in all_months])
mend = np.array([p.end_time for p in all_months])


def window_scan(mask, keep, k_months=5):
    """excess after deleting each contiguous k-month window inside `keep`."""
    out = []
    for i in range(len(all_months) - k_months + 1):
        w = ((idx >= mstart[i]) & (idx <= mend[i + k_months - 1]))
        if not (w & keep).any():
            continue
        r = stats(d, mask, keep & ~w, with_p=False)
        if r["ok"]:
            out.append((r["exc"], str(all_months[i]), r["n"], w))
    return out


for s in BULL4:
    m = SIGNALS[s]
    print(f"  {s}")
    for nm, y0, y1 in ERAS:
        kp = np.asarray((idx.year >= y0) & (idx.year <= y1))
        base = stats(d, m, kp, with_p=False)
        if not base["ok"]:
            print(f"    {nm:<7} n={base['n']:<4} INCONCLUSIVE (n<25)")
            continue
        sc = window_scan(m, kp)
        if not sc:
            print(f"    {nm:<7} n={base['n']:<4} exc {base['exc']*100:+.3f}pp  (no valid drop)")
            continue
        vals = np.array([x[0] for x in sc])
        best = max(sc, key=lambda x: x[0])
        # snooping-corrected p: fraction of 5-month windows whose deletion helps at least as much
        print(f"    {nm:<7} n={base['n']:<4} exc {base['exc']*100:+7.3f}pp -> best-drop "
              f"{best[0]*100:+7.3f}pp (+{(best[0]-base['exc'])*100:.3f}pp, drop {best[1]}..) "
              f"| median-drop {np.median(vals)*100:+7.3f}pp | 90th pct {np.percentile(vals,90)*100:+7.3f}pp")
    print()

hr("T2b — snooping-corrected p for the COVID drop: among all 5-month deletions in 2020-2026,")
print("      how unusual is the Feb-Jun-2020 deletion, and how often does a random deletion")
print("      also flip the sign to positive?\n")
for s in BULL4:
    m = SIGNALS[s]
    kp = is20s
    base = stats(d, m, kp, with_p=False)
    sc = window_scan(m, kp)
    vals = np.array([x[0] for x in sc])
    cov = [x for x in sc if x[1] == "2020-02"]
    covv = cov[0][0] if cov else np.nan
    frac_pos = float((vals > 0).mean())
    frac_ge = float((vals >= covv).mean()) if cov else np.nan
    print(f"  {s:<22} base {base['exc']*100:+7.3f}pp | covid-drop {covv*100:+7.3f}pp | "
          f"{frac_pos*100:5.1f}% of ALL 5-month deletions flip it positive | "
          f"covid rank p={frac_ge:.3f}  ({len(vals)} windows)")

# ------------------------------------------------------------------ T3 episode concentration
hr("T3 — EPISODE CONCENTRATION: drop worst month / drop 2008 / drop 2020 (FULL SAMPLE, g5)")
full = np.ones(N, bool)
print(f"{'signal':<22}{'full exc':>11}{'p':>8}{'-worst mo':>12}{'(month)':>10}"
      f"{'-2008':>10}{'-2020':>10}{'-08&-20':>10}")
print("-" * 108)
drop08 = np.asarray(idx.year != 2008)
drop20 = np.asarray(idx.year != 2020)
for s in BULL4:
    m = SIGNALS[s]
    f = stats(d, m, full)
    # worst single month = the one whose deletion most improves the excess
    best_e, best_lbl = -9, "-"
    for i in range(len(all_months)):
        w = ((idx >= mstart[i]) & (idx <= mend[i]))
        if not (m & w).any():
            continue
        r = stats(d, m, full & ~w, with_p=False)
        if r["ok"] and r["exc"] > best_e:
            best_e, best_lbl = r["exc"], str(all_months[i])
    e08 = stats(d, m, drop08, with_p=False)
    e20 = stats(d, m, drop20, with_p=False)
    eb = stats(d, m, drop08 & drop20, with_p=False)
    print(f"{s:<22}{f['exc']*100:>10.3f}pp{f['p']:>8.3f}{best_e*100:>11.3f}pp{best_lbl:>10}"
          f"{e08['exc']*100:>9.3f}pp{e20['exc']*100:>9.3f}pp{eb['exc']*100:>9.3f}pp")
print("\n  (n after each deletion:)")
for s in BULL4:
    m = SIGNALS[s]
    print(f"    {s:<22} full n={stats(d,m,full,with_p=False)['n']:<5} "
          f"-2008 n={stats(d,m,drop08,with_p=False)['n']:<5} "
          f"-2020 n={stats(d,m,drop20,with_p=False)['n']:<5} "
          f"-both n={stats(d,m,drop08&drop20,with_p=False)['n']}")

# ------------------------------------------------------------------ T4 family-wise correction
hr("T4 — FAMILY-WISE MULTIPLE TESTING (max-statistic over shared circular rotations)")
family = []
for s in SIGNALS:
    for h in HORIZONS:
        for nm, kp in [("full", full), ("1990s", np.asarray(idx.year <= 1999)),
                       ("2000s", np.asarray((idx.year >= 2000) & (idx.year <= 2009))),
                       ("2010s", np.asarray((idx.year >= 2010) & (idx.year <= 2019))),
                       ("2020s", is20s), ("2020s-exCOVID", keep20_ex)]:
            family.append((s, h, nm, kp))

obs, nulls, labels = [], [], []
for s, h, nm, kp in family:
    m = SIGNALS[s]
    fwd = d[f"g{h}"].values
    valid = ~np.isnan(fwd)
    k = kp & valid
    if (m & k).sum() < MIN_N:
        continue
    cond = fwd[m & k].mean()
    base = fwd[k].mean()
    nl = rotation_null_means(m, fwd, kp)
    good = ~np.isnan(nl)
    if good.sum() < 500:
        continue
    nl = np.where(good, nl, np.nan)
    centre = np.nanmean(nl)
    obs.append(abs(cond - centre))
    nulls.append(np.abs(nl - centre))
    labels.append((s, h, nm, cond - base, int((m & k).sum())))

NL = np.vstack(nulls)                       # configs x offsets
valid_cols = ~np.isnan(NL).any(axis=0)
NLv = NL[:, valid_cols]
maxnull = NLv.max(axis=0)
print(f"  family size actually tested here: {len(labels)} configs "
      f"(5 signals x 5 horizons x 6 samples, n>=25)")
print(f"  shared rotation offsets usable: {NLv.shape[1]:,}\n")
print(f"{'signal':<22}{'H':>3}{'sample':<16}{'n':>6}{'excess':>10}{'raw p':>8}{'FWE p':>8}")
print("-" * 108)
rows = []
for i, (s, h, nm, exc, n) in enumerate(labels):
    raw = float((NLv[i] >= obs[i]).mean())
    fwe = float((maxnull >= obs[i]).mean())
    rows.append((raw, fwe, s, h, nm, exc, n))
for raw, fwe, s, h, nm, exc, n in sorted(rows)[:18]:
    print(f"{s:<22}{h:>3}{nm:<16}{n:>6}{exc*100:>9.3f}pp{raw:>8.3f}{fwe:>8.3f}")
print("\n  the claim's headline config:")
for raw, fwe, s, h, nm, exc, n in rows:
    if s == "stretch>=+20%" and h == 5 and nm == "2020s-exCOVID":
        print(f"    stretch>=+20% D5 2020s-exCOVID: n={n} excess {exc*100:+.3f}pp  "
              f"raw p={raw:.3f}  family-wise p={fwe:.3f}")
        bonf = min(1.0, raw * len(labels))
        print(f"    Bonferroni over the {len(labels)} configs tested here: p={bonf:.3f}")
        print(f"    Bonferroni over the ~70 configs this project has tried: p={min(1.0, raw*70):.3f}")
# Benjamini-Hochberg
ps = np.array([r[0] for r in rows])
order = np.argsort(ps)
mm = len(ps)
bh = np.empty(mm)
prev = 1.0
for rank in range(mm - 1, -1, -1):
    i = order[rank]
    prev = min(prev, ps[i] * mm / (rank + 1))
    bh[i] = prev
sig = [(rows[i], bh[i]) for i in range(mm) if bh[i] < 0.10]
print(f"\n  Benjamini-Hochberg q<0.10 survivors out of {mm} configs: {len(sig)}")
for r, q in sorted(sig, key=lambda x: x[1])[:12]:
    print(f"    {r[2]:<22} D{r[3]:<3}{r[4]:<16} n={r[6]:<5} exc {r[5]*100:+7.3f}pp  q={q:.3f}")

# ------------------------------------------------------------------ T5 precision
hr("T5 — PRECISION: is n=72 enough to assert '+0.800pp'? (stationary block bootstrap, 20d blocks)")


def block_boot_ci(mask, keep, h=5, nboot=20000, blk=20):
    fwd = d[f"g{h}"].values
    k = keep & (~np.isnan(fwd))
    ii = np.flatnonzero(k)
    n = len(ii)
    nb = int(np.ceil(n / blk))
    out = np.empty(nboot)
    starts_ = RNG.integers(0, n, size=(nboot, nb))
    offs = np.arange(blk)
    for b in range(nboot):
        pick = (starts_[b][:, None] + offs[None, :]).ravel()[:n] % n
        jj = ii[pick]
        mm_ = mask[jj]
        if mm_.sum() < 5:
            out[b] = np.nan
            continue
        out[b] = fwd[jj][mm_].mean() - fwd[jj].mean()
    out = out[~np.isnan(out)]
    return np.percentile(out, [2.5, 50, 97.5]), float((out <= 0).mean())


for s in BULL4:
    m = SIGNALS[s]
    r = stats(d, m, keep20_ex, with_p=False)
    if not r["ok"]:
        print(f"  {s:<22} n={r['n']} INCONCLUSIVE")
        continue
    ci, pneg = block_boot_ci(m, keep20_ex)
    sd = np.nanstd(d.g5.values[m & keep20_ex & ~np.isnan(d.g5.values)], ddof=1)
    print(f"  {s:<22} n={r['n']:<4} exc {r['exc']*100:+7.3f}pp  "
          f"95% CI [{ci[0]*100:+.2f}, {ci[2]*100:+.2f}]pp  P(exc<=0)={pneg:.3f}  "
          f"| sd of g5 on signal days {sd*100:.2f}%  naive SE {sd/np.sqrt(r['n'])*100:.2f}pp")

# ------------------------------------------------------------------ T6 decay vs covid
hr("T6 — 'NOT DECAY' TEST: ex-COVID 2020s vs the pre-2020 record, and 2020s split in half")
for s in BULL4:
    m = SIGNALS[s]
    pre = np.asarray(idx.year <= 2019)
    a = stats(d, m, pre)
    b = stats(d, m, keep20_ex)
    print(f"\n  {s}")
    print(f"    1990-2019          n={a['n']:<5} exc {a['exc']*100:+7.3f}pp  p={a['p']:.3f}")
    print(f"    2020-2026 exCOVID  n={b['n']:<5} exc {b['exc']*100:+7.3f}pp  p={b['p']:.3f}"
          if b["ok"] else f"    2020-2026 exCOVID  n={b['n']} INCONCLUSIVE")
    for lo, hi in [(2020, 2022), (2023, 2026)]:
        kp = np.asarray((idx.year >= lo) & (idx.year <= hi)) & ~COVID
        r = stats(d, m, kp, with_p=False)
        tag = f"{lo}-{hi} exCOVID"
        print(f"    {tag:<19}n={r['n']:<5} " +
              (f"exc {r['exc']*100:+7.3f}pp" if r["ok"] else "INCONCLUSIVE (n<25)"))
    # also: 2010s vs 2020s-exCOVID difference
    t10 = stats(d, m, np.asarray((idx.year >= 2010) & (idx.year <= 2019)), with_p=False)
    if t10["ok"] and b["ok"]:
        print(f"    2010s exc {t10['exc']*100:+.3f}pp  ->  2020s-exCOVID {b['exc']*100:+.3f}pp "
              f"(delta {(b['exc']-t10['exc'])*100:+.3f}pp)")

hr("T7 — SANITY: what does the SAME exclusion logic do to the earlier eras?")
print("  Delete Sep2008-Jan2009 (the 2008 analogue of Feb-Jun 2020) from the 2000s block:\n")
gfc = ((idx >= "2008-09-01") & (idx <= "2009-01-31"))
k00 = np.asarray((idx.year >= 2000) & (idx.year <= 2009))
for s in BULL4:
    m = SIGNALS[s]
    a = stats(d, m, k00, with_p=False)
    b = stats(d, m, k00 & ~gfc, with_p=False)
    if not a["ok"]:
        print(f"  {s:<22} n={a['n']} INCONCLUSIVE")
        continue
    print(f"  {s:<22} 2000s n={a['n']:<4} exc {a['exc']*100:+7.3f}pp  -> ex-GFC n={b['n']:<4} "
          + (f"exc {b['exc']*100:+7.3f}pp  (delta {(b['exc']-a['exc'])*100:+.3f}pp)"
             if b["ok"] else "INCONCLUSIVE"))

print("\ndone.")
