"""
_move_wf_xsec.py — cross-sectional / earnings / idiosyncratic study for the move-probability system.

Answers: (1) how much does bucket climatology differ index vs pooled singles vs sector,
(2) does an earnings flag matter, (3) market vs idiosyncratic share at h=5,
(4) does a pooled model mis-calibrate by own-vol / price decile, (5) survivorship proxy.

Run:  ../../vcp_env/bin/python _move_wf_xsec.py <section>
      sections: clim | z | earn | idio | calib | surv | all
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _move_data as D
import _move_lib as L

THR = 0.02
HORIZONS = (1, 5, 10, 21)
INDICES = ["SPY", "QQQ", "IWM", "DIA", "^GSPC"]
SECTOR_BUCKETS = ["TECH_STOCKS", "HEALTHCARE_STOCKS", "FINANCIAL_STOCKS", "CONSUMER_DISCRETIONARY",
                  "CONSUMER_STAPLES", "ENERGY_STOCKS", "MATERIALS_INDUSTRIALS", "UTILITIES",
                  "REAL_ESTATE_REITS", "COMMUNICATION_SERVICES"]
RNG = np.random.default_rng(20260911)


# ------------------------------------------------------------------ setup
def setup(drop_bad=True):
    p = D.load()
    close = p["Close"]
    cov = close.notna().sum()
    keep = cov[cov >= 500].index.tolist()
    # DATA HYGIENE: CBIO has 2,189 NEGATIVE adjusted closes (a yfinance back-adjustment artifact).
    # np.log() turns those into NaN so the vol estimators quietly drop them, but
    # forward_simple_return happily returns finite garbage (min -1.82, i.e. "-182%"), which then
    # lands in the down_big bucket. Any _move_* study that does not drop it is contaminated.
    bad = [c for c in keep if (close[c].dropna() <= 0).any()]
    if bad:
        print(f"  !! dropping {len(bad)} symbol(s) with non-positive adjusted closes: "
              f"{bad} (rows affected: {int((close[bad] <= 0).sum().sum())})")
        if drop_bad:
            keep = [c for c in keep if c not in bad]
    p = {f: v.loc[:, [c for c in keep if c in v.columns]] for f, v in p.items()}
    close = p["Close"]
    singles = [s for s in keep if s not in INDICES and s != "^VIX"]

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import stock_symbols_1243 as S
    # Buckets in stock_symbols_1243.py OVERLAP heavily (115/231 panel singles sit in >1 bucket;
    # every utility is also listed under ENERGY_STOCKS). Tie-break = smallest bucket wins, which
    # is deterministic and puts utilities/comm-services where GICS would.
    sizes = {b: len(set(getattr(S, b, []))) for b in SECTOR_BUCKETS}
    memb = defaultdict(set)
    for b in SECTOR_BUCKETS:
        for s in set(getattr(S, b, [])):
            if s in singles:
                memb[s].add(b)
    sector = {s: min(v, key=lambda b: sizes[b]) for s, v in memb.items()}
    exclusive = {s: list(v)[0] for s, v in memb.items() if len(v) == 1}
    return p, singles, sector, exclusive


def logret(df):
    return np.log(df).diff()


# ------------------------------------------------------------------ helpers
def stack_obs(close, syms, h, thr=THR):
    """Long-form (date, sym, fwd_ret, bucket_idx) for the given symbols/horizon."""
    fwd = L.forward_simple_return(close[syms], h)
    m = fwd.stack(dropna=True)
    idx_map = {b: i for i, b in enumerate(L.BUCKETS)}
    b = L.bucketize(m, thr).map(idx_map).astype(int)
    return pd.DataFrame({"fwd": m.values, "bk": b.values},
                        index=pd.MultiIndex.from_tuples(m.index, names=["date", "sym"]))


def block_boot_pbig(obs, h, n_rep=400, block=63):
    """Block bootstrap of P(|move|>thr): resample contiguous DATE blocks, keep every symbol on a
    sampled date together. Handles both the h-day window overlap and cross-sectional correlation."""
    dates = obs.index.get_level_values("date")
    udates = np.array(sorted(dates.unique()))
    pos = pd.Series(np.arange(len(udates)), index=udates)
    grp = obs.groupby(level="date")
    hits = grp["bk"].apply(lambda x: np.isin(x, [0, 3]).sum())
    cnts = grp["bk"].size()
    hv, cv = hits.reindex(udates).fillna(0).values, cnts.reindex(udates).fillna(0).values
    nb = max(1, len(udates) // block)
    out = []
    for _ in range(n_rep):
        starts = RNG.integers(0, max(1, len(udates) - block), nb)
        sel = np.concatenate([np.arange(s, min(s + block, len(udates))) for s in starts])
        c = cv[sel].sum()
        out.append(hv[sel].sum() / c if c else np.nan)
    return float(np.nanstd(out))


SIG_FLOOR = 0.002   # 0.2%/day ~ 3.2% annualised. 571/1.15M rows (0.05%) hit it; several are
                    # literally sigma_hat == 0 (halted / flat bars) which otherwise make z infinite.


def sigma_hat(p, syms, n=20):
    """Provisional per-day vol forecast known at t: Yang-Zhang(20) with an EWMA fallback."""
    o, hi, lo, c = (p[f][syms] for f in ("Open", "High", "Low", "Close"))
    yz = L.vol_yang_zhang(o, hi, lo, c, n)
    ew = L.vol_ewma(c, 0.94)
    return yz.where(yz.notna() & (yz > 0), ew).clip(lower=SIG_FLOOR)


# ================================================================== 1. climatology
def sec_clim(p, singles, sector, exclusive):
    close = p["Close"]
    print("=" * 100)
    print("SECTION 1  BUCKET CLIMATOLOGY  (thr=2%, simple fwd returns, full sample 2001-2026)")
    print("=" * 100)

    def row(name, obs, h, se=None):
        n = len(obs)
        fr = np.bincount(obs.bk.values, minlength=4) / n
        pbig = fr[0] + fr[3]
        s = f"{name:26s} h={h:<3d} n={n:>9,d}  down_big={fr[0]:.4f} down_sm={fr[1]:.4f} " \
            f"up_sm={fr[2]:.4f} up_big={fr[3]:.4f}  P|mv|>2%={pbig:.4f}"
        if se is not None:
            s += f" +-{se:.4f}"
        print(s)
        return pbig, fr

    res = {}
    for h in HORIZONS:
        print("-" * 100)
        for ix in INDICES:
            o = stack_obs(close, [ix], h)
            pb, fr = row(ix, o, h)
            res[(ix, h)] = (pb, fr, len(o))
        o = stack_obs(close, singles, h)
        se = block_boot_pbig(o, h)
        pb, fr = row(f"POOLED {len(singles)} singles", o, h, se)
        res[("POOLED", h)] = (pb, fr, len(o))
        # vix for reference
        if "^VIX" in p["Close"].columns:
            ov = stack_obs(p["Close"], ["^VIX"], h)
            row("^VIX (reference only)", ov, h)

    print("\n" + "=" * 100)
    print("SECTOR SPREAD  (smallest-bucket-wins assignment)")
    print("=" * 100)
    rows = []
    for h in HORIZONS:
        for b in SECTOR_BUCKETS:
            syms = [s for s in singles if sector.get(s) == b]
            if len(syms) < 5:
                print(f"  {b} h={h}: only {len(syms)} symbols, skipped")
                continue
            o = stack_obs(close, syms, h)
            n = len(o)
            fr = np.bincount(o.bk.values, minlength=4) / n
            se = block_boot_pbig(o, h, n_rep=200)
            rows.append(dict(h=h, sector=b, nsym=len(syms), n=n, down_big=fr[0], up_big=fr[3],
                             pbig=fr[0] + fr[3], se=se))
    sdf = pd.DataFrame(rows)
    for h in HORIZONS:
        sub = sdf[sdf.h == h].sort_values("pbig")
        print(f"\n--- h={h} ---")
        print(sub[["sector", "nsym", "n", "down_big", "up_big", "pbig", "se"]].to_string(index=False,
              float_format=lambda x: f"{x:.4f}"))
        lo, hi = sub.pbig.iloc[0], sub.pbig.iloc[-1]
        print(f"  spread: min {lo:.4f} ({sub.sector.iloc[0]})  max {hi:.4f} ({sub.sector.iloc[-1]})"
              f"  ratio {hi/lo:.2f}x   pooled {res[('POOLED',h)][0]:.4f}  SPY {res[('SPY',h)][0]:.4f}"
              f"  singles/SPY ratio {res[('POOLED',h)][0]/res[('SPY',h)][0]:.2f}x")

    # exclusive-membership robustness
    print("\nROBUSTNESS: sectors restricted to symbols in EXACTLY ONE bucket (h=5)")
    for b in SECTOR_BUCKETS:
        syms = [s for s, bb in exclusive.items() if bb == b]
        if len(syms) < 5:
            print(f"  {b:26s} nsym={len(syms)} -> skipped (all members are multi-bucket)")
            continue
        o = stack_obs(close, syms, 5)
        fr = np.bincount(o.bk.values, minlength=4) / len(o)
        print(f"  {b:26s} nsym={len(syms):3d} n={len(o):>8,d} P|mv|>2%={fr[0]+fr[3]:.4f}")

    # per-symbol dispersion
    print("\nPER-SYMBOL dispersion of P(|move|>2%) at h=5 (231 singles, each full-history):")
    ps = []
    for s in singles:
        o = stack_obs(close, [s], 5)
        if len(o) < 400:
            continue
        fr = np.bincount(o.bk.values, minlength=4) / len(o)
        ps.append((s, len(o), fr[0] + fr[3]))
    pdf = pd.DataFrame(ps, columns=["sym", "n", "pbig"]).sort_values("pbig")
    q = pdf.pbig.quantile([0, .05, .25, .5, .75, .95, 1]).round(4)
    print(f"  n symbols={len(pdf)}  quantiles: {q.to_dict()}")
    print(f"  lowest 6: {pdf.head(6)[['sym','pbig']].values.tolist()}")
    print(f"  highest 6: {pdf.tail(6)[['sym','pbig']].values.tolist()}")
    return res, sdf, pdf


# ================================================================== 2. z-distribution shape
def sec_z(p, singles, sector):
    close = p["Close"]
    print("\n" + "=" * 100)
    print("SECTION 1b  IS ONE POOLED z-DISTRIBUTION DEFENSIBLE?")
    print("  z = r_h_simple / (sigma_hat_YZ20(t) * sqrt(h)); sigma_hat known at t, mu=0")
    print("=" * 100)
    sh = sigma_hat(p, singles)
    shi = sigma_hat(p, INDICES)
    for h in HORIZONS:
        print(f"\n--- h={h} ---")
        rows = []

        def zstats(name, syms, sh_src, close_src):
            fwd = L.forward_simple_return(close_src[syms], h)
            den = sh_src[syms] * np.sqrt(h)
            z = (fwd / den).replace([np.inf, -np.inf], np.nan).stack(dropna=True)
            if len(z) < 500:
                return None
            zv = z.values
            # raw std is destroyed by a handful of genuine +400% biotech pops, so report a robust
            # scale (MAD*1.4826) and P(|z|>c) alongside it -- those are what calibration cares about.
            mad = np.median(np.abs(zv - np.median(zv))) * 1.4826
            return dict(name=name, nsym=len(syms), n=len(zv), std=zv.std(ddof=1), robust_sd=mad,
                        skew=pd.Series(zv).skew(), kurt=pd.Series(zv).kurt(),
                        q01=np.quantile(zv, .01), q05=np.quantile(zv, .05),
                        q95=np.quantile(zv, .95), q99=np.quantile(zv, .99),
                        p_abs_gt2=np.mean(np.abs(zv) > 2), p_abs_gt3=np.mean(np.abs(zv) > 3),
                        p_lt_m3=np.mean(zv < -3))

        for ix in INDICES:
            r = zstats(ix, [ix], shi, close)
            if r:
                rows.append(r)
        rows.append(zstats("POOLED singles", singles, sh, close))
        for b in SECTOR_BUCKETS:
            syms = [s for s in singles if sector.get(s) == b]
            if len(syms) >= 5:
                rows.append(zstats(b, syms, sh, close))
        zdf = pd.DataFrame([r for r in rows if r])
        print(zdf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        sec_only = zdf[zdf.name.isin(SECTOR_BUCKETS)]
        print(f"  sector spread of robust_sd(z): {sec_only.robust_sd.min():.3f}-"
              f"{sec_only.robust_sd.max():.3f}"
              f"  of q01: {sec_only.q01.min():.2f}..{sec_only.q01.max():.2f}"
              f"  of P(|z|>2): {sec_only.p_abs_gt2.min():.4f}-{sec_only.p_abs_gt2.max():.4f}"
              f"  ratio {sec_only.p_abs_gt2.max()/sec_only.p_abs_gt2.min():.2f}x")

    # per-symbol z shape dispersion at h=5 -- the real test of "does sigma_hat absorb the difference"
    print("\nPER-SYMBOL z SHAPE at h=5 vs PER-SYMBOL P(|move|>2%): does standardizing collapse it?")
    fwd = L.forward_simple_return(close[singles], 5)
    z = (fwd / (sh[singles] * np.sqrt(5))).replace([np.inf, -np.inf], np.nan)
    rows = []
    for s in singles:
        zz = z[s].dropna()
        rr = fwd[s].dropna()
        if len(zz) < 400:
            continue
        zv = zz.values
        mad = np.median(np.abs(zv - np.median(zv))) * 1.4826
        rows.append((s, len(zz), mad, np.mean(np.abs(zv) > 2), np.mean(zv < -2),
                     np.mean(np.abs(rr) > THR), np.log(close[s]).diff().std() * np.sqrt(252)))
    q = pd.DataFrame(rows, columns=["sym", "n", "zsd", "pz2", "pzdn2", "praw", "annvol"])
    print(f"  n symbols={len(q)}")
    for col, lab in (("praw", "raw   P(|move|>2%)"), ("zsd", "robust_sd(z)     "),
                     ("pz2", "P(|z|>2)         "), ("pzdn2", "P(z<-2)          ")):
        v = q[col]
        print(f"  {lab}: mean {v.mean():.4f}  sd {v.std():.4f}  CV {v.std()/v.mean():.3f}  "
              f"range {v.min():.3f}-{v.max():.3f}  p95/p5 {v.quantile(.95)/v.quantile(.05):.2f}x")
    print(f"  corr(annvol, praw)={q.annvol.corr(q.praw):.3f}   "
          f"corr(annvol, robust_sd(z))={q.annvol.corr(q.zsd):.3f}   "
          f"corr(annvol, P(|z|>2))={q.annvol.corr(q.pz2):.3f}")
    lowv, hiv = q.annvol.quantile(.1), q.annvol.quantile(.9)
    a, b = q[q.annvol <= lowv], q[q.annvol >= hiv]
    print(f"  lowest-vol decile of names (n={len(a)}): praw {a.praw.mean():.4f}  "
          f"robust_sd(z) {a.zsd.mean():.3f}  P(|z|>2) {a.pz2.mean():.4f}")
    print(f"  highest-vol decile of names (n={len(b)}): praw {b.praw.mean():.4f}  "
          f"robust_sd(z) {b.zsd.mean():.3f}  P(|z|>2) {b.pz2.mean():.4f}")
    return q


# ================================================================== 3. earnings
def earnings_map(close, syms):
    """symbol -> sorted array of integer bar positions of the FIRST bar on/after each report date."""
    cache = Path(__file__).resolve().parents[2] / "gold_pine_script" / "pead_earnings_cache.json"
    ec = json.loads(cache.read_text())
    idx = close.index
    out, meta = {}, []
    for s in syms:
        if s not in ec or not ec[s]:
            continue
        d = pd.to_datetime([x[0] for x in ec[s]])
        pos = idx.searchsorted(d, side="left")
        first = close[s].first_valid_index()
        last = close[s].last_valid_index()
        ok = (pos < len(idx) - 2) & (pos > 1) & (d >= first) & (d <= last)
        pp = np.unique(pos[ok])
        if len(pp) >= 5:
            out[s] = pp
            meta.append((s, len(pp), idx[pp[0]].date(), idx[pp[-1]].date()))
    return out, pd.DataFrame(meta, columns=["sym", "n_events", "first", "last"]), len(ec)


def sec_earn(p, singles):
    close = p["Close"]
    print("\n" + "=" * 100)
    print("SECTION 2  EARNINGS")
    print("=" * 100)
    em, meta, ncache = earnings_map(close, singles)
    print(f"cache ../../gold_pine_script/pead_earnings_cache.json: {ncache} symbols total, "
          f"{len(em)} usable & in-panel (>=5 in-history events), {int(meta.n_events.sum())} events")
    print(f"events/symbol: median {int(meta.n_events.median())}, min {meta.n_events.min()}, "
          f"max {meta.n_events.max()};  window {meta['first'].min()} -> {meta['last'].max()}")
    print(f"coverage: {len(em)}/{len(singles)} = {len(em)/len(singles):.0%} of panel singles")
    print("cache symbols in list:", ", ".join(sorted(em)))

    lr = logret(close)
    # --- which bar carries the reaction?
    print("\nREACTION-BAR PROFILE: mean |log ret| on bar b+k / the symbol's own median |log ret|")
    prof = defaultdict(list)
    for s, pos in em.items():
        base = lr[s].abs().median()
        v = lr[s].values
        for k in range(-3, 5):
            q = pos + k
            q = q[(q >= 0) & (q < len(v))]
            x = v[q]
            x = x[~np.isnan(x)]
            if len(x):
                prof[k].append(np.nanmean(np.abs(x)) / base)
    for k in sorted(prof):
        print(f"   bar b{k:+d}: ratio {np.mean(prof[k]):.2f}x   (n symbols {len(prof[k])})")
    print("   -> bar b+1 is the reaction bar: the cached date is the announcement date and US")
    print("      reports are mostly after the close, so the price reacts the FOLLOWING session.")

    # ---------------------------------------------------------------- flag construction
    def flags(s, h, lo=0, hi=1):
        """Boolean over rows t: True if some reaction bar (b+lo .. b+hi) lands in [t+1, t+h].
        Everything is known ex ante -- earnings dates are published weeks in advance."""
        f = np.zeros(len(close), bool)
        for b0 in em[s]:
            for k in range(lo, hi + 1):
                q = b0 + k
                a, z = max(0, q - h), q - 1
                if z >= a:
                    f[a:z + 1] = True
        return f

    def coverage_mask(s):
        """Restrict to the span where this symbol's earnings dates actually exist, so that
        flagged and unflagged windows are drawn from the SAME calendar period. Without this,
        late-coverage names (ZM, NIO) donate their whole pre-2019 history to 'unflagged'."""
        pos = em[s]
        m = np.zeros(len(close), bool)
        m[max(0, pos[0] - 65):min(len(close), pos[-1] + 65)] = True
        return m

    def rsd(x):
        """Robust scale: 1.4826*MAD. The plain std is meaningless here -- RIOT/BLDP/APPS have
        genuine +200% 21-day windows that dominate a 260k-row std."""
        return float(np.median(np.abs(x - np.median(x))) * 1.4826)

    print("\nPER-SYMBOL EARNINGS MULTIPLIERS (restricted to each symbol's earnings-coverage span;")
    print("reaction bar b+1 or b (BMO/AMC unknown ex ante) must fall inside the window)")
    per = {}
    rows = []
    for h in HORIZONS:
        per[h] = {}
        recs = []
        for s in em:
            fwd = L.forward_simple_return(close[s], h).values
            fl = flags(s, h)
            cm = coverage_mask(s)
            ok = ~np.isnan(fwd) & cm
            a, b = fwd[ok & fl], fwd[ok & ~fl]
            if len(a) < 30 or len(b) < 200:
                continue
            per[h][s] = (a, b)
            recs.append(dict(sym=s, n_flag=len(a), n_un=len(b),
                             rsd_flag=rsd(a), rsd_un=rsd(b),
                             sd_mult=rsd(a) / rsd(b) if rsd(b) > 0 else np.nan,
                             mad_abs_mult=np.mean(np.abs(a)) / np.mean(np.abs(b)),
                             p_flag=np.mean(np.abs(a) > THR), p_un=np.mean(np.abs(b) > THR),
                             p_mult=np.mean(np.abs(a) > THR) / max(np.mean(np.abs(b) > THR), 1e-9)))
        r = pd.DataFrame(recs)
        # symbol-cluster bootstrap on the MEDIAN per-symbol multiplier
        syms = r.sym.values
        bs_s, bs_p, bs_a = [], [], []
        for _ in range(2000):
            pick = RNG.integers(0, len(syms), len(syms))
            bs_s.append(np.median(r.sd_mult.values[pick]))
            bs_p.append(np.median(r.p_mult.values[pick]))
            bs_a.append(np.median(r.mad_abs_mult.values[pick]))
        share = r.n_flag.sum() / (r.n_flag.sum() + r.n_un.sum())
        rows.append(dict(h=h, nsym=len(r), n_flag=int(r.n_flag.sum()), n_un=int(r.n_un.sum()),
                         flag_share=share,
                         med_sd_mult=r.sd_mult.median(),
                         sd_lo=np.percentile(bs_s, 2.5), sd_hi=np.percentile(bs_s, 97.5),
                         med_absret_mult=r.mad_abs_mult.median(),
                         abs_lo=np.percentile(bs_a, 2.5), abs_hi=np.percentile(bs_a, 97.5),
                         med_p_mult=r.p_mult.median(),
                         p_lo=np.percentile(bs_p, 2.5), p_hi=np.percentile(bs_p, 97.5),
                         frac_sym_sd_gt1=(r.sd_mult > 1).mean(),
                         frac_sym_p_gt1=(r.p_mult > 1).mean(),
                         med_p_flag=r.p_flag.median(), med_p_un=r.p_un.median()))
    edf = pd.DataFrame(rows)
    print(edf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n(med_* = median over the 52 symbols of that symbol's own flagged/unflagged ratio;")
    print(" CI = 2000-rep bootstrap resampling SYMBOLS, the natural cluster here.)")

    # pooled version too, but robustly scaled
    print("\nPOOLED (all 52 symbols stacked, coverage-restricted, robust scale):")
    rows = []
    for h in HORIZONS:
        a = np.concatenate([per[h][s][0] for s in per[h]])
        b = np.concatenate([per[h][s][1] for s in per[h]])
        rows.append(dict(h=h, n_flag=len(a), n_un=len(b), rsd_flag=rsd(a), rsd_un=rsd(b),
                         sd_mult=rsd(a) / rsd(b), p_flag=np.mean(np.abs(a) > THR),
                         p_un=np.mean(np.abs(b) > THR),
                         p_mult=np.mean(np.abs(a) > THR) / np.mean(np.abs(b) > THR),
                         mean_ret_flag=a.mean(), mean_ret_un=b.mean(),
                         dn_flag=np.mean(a <= -THR), dn_un=np.mean(b <= -THR),
                         up_flag=np.mean(a >= THR), up_un=np.mean(b >= THR)))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- how much of the effect survives the vol forecast?
    print("\nAFTER CONTROLLING FOR sigma_hat: robust scale of z, flagged vs unflagged")
    print("(if sigma_hat already saw the earnings bump, resid_mult ~ 1 and the flag adds nothing)")
    sh = sigma_hat(p, list(em))
    rows = []
    for h in HORIZONS:
        zf, zu = [], []
        for s in per[h]:
            fwd = L.forward_simple_return(close[s], h)
            z = (fwd / (sh[s] * np.sqrt(h))).replace([np.inf, -np.inf], np.nan).values
            fl, cm = flags(s, h), coverage_mask(s)
            ok = ~np.isnan(z) & cm
            zf.append(z[ok & fl])
            zu.append(z[ok & ~fl])
        a, b = np.concatenate(zf), np.concatenate(zu)
        rows.append(dict(h=h, n_flag=len(a), n_un=len(b), rsd_z_flag=rsd(a), rsd_z_un=rsd(b),
                         resid_sd_mult=rsd(a) / rsd(b),
                         mean_absz_flag=np.abs(a).mean(), mean_absz_un=np.abs(b).mean(),
                         resid_absz_mult=np.abs(a).mean() / np.abs(b).mean()))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- THE SHIP TEST: walk-forward log loss, pooled model with vs without the flag
    print("\nSHIP TEST: walk-forward OOS log loss on the 52-symbol earnings subsample.")
    print("  base  = pooled z model (sigma_pred = k_h*YZ20*sqrt(h), empirical z-CDF, prior years)")
    print("  +earn = same, but sigma_pred multiplied by m_h fitted on PRIOR YEARS ONLY")
    print("  clim  = prior-years bucket frequencies")
    syms = list(em)
    shx = sigma_hat(p, syms)
    idx_map = {b: i for i, b in enumerate(L.BUCKETS)}
    out = []
    for h in HORIZONS:
        recs = []
        for s in syms:
            fwd = L.forward_simple_return(close[s], h)
            den = (shx[s] * np.sqrt(h))
            fl, cm = flags(s, h), coverage_mask(s)
            d = pd.DataFrame({"fwd": fwd.values, "sig": den.values, "fl": fl, "cm": cm},
                             index=close.index).dropna()
            d = d[d.cm]
            d["sym"] = s
            recs.append(d)
        d = pd.concat(recs)
        d["z"] = d.fwd / d.sig
        d["bk"] = L.bucketize(d.fwd, THR).map(idx_map).astype(int)
        d["year"] = d.index.year
        years = sorted(d.year.unique())
        acc = {"base": [], "earn": [], "clim": [], "y": [], "n": 0, "mults": []}
        for y in years:
            tr, te = d[d.year < y], d[d.year == y]
            if len(tr) < 5000 or len(te) == 0 or y - years[0] < 5:
                continue
            # earnings multiplier fitted on train only (robust scale of z, flagged vs not)
            zt = tr.z.values
            m = rsd(zt[tr.fl.values]) / rsd(zt[~tr.fl.values]) if (~tr.fl.values).sum() > 100 else 1.0
            acc["mults"].append((y, m))
            for tag, mult in (("base", 1.0), ("earn", m)):
                # standardise train z by its own robust scale AFTER applying the same multiplier
                adj_tr = np.where(tr.fl.values, mult, 1.0)
                zz = zt / adj_tr
                k = rsd(zz)
                zs = np.sort(zz / k)
                sig = te.sig.values * k * np.where(te.fl.values, mult, 1.0)
                F1 = np.searchsorted(zs, -THR / sig) / len(zs)
                F2 = np.searchsorted(zs, np.zeros(len(sig))) / len(zs)
                F3 = np.searchsorted(zs, THR / sig) / len(zs)
                pr = np.clip(np.column_stack([F1, F2 - F1, F3 - F2, 1 - F3]), 1e-6, 1)
                acc[tag].append(pr / pr.sum(1, keepdims=True))
            acc["clim"].append(np.tile(L.climatology(tr.bk.values), (len(te), 1)))
            acc["y"].append(te.bk.values)
            acc.setdefault("fl", []).append(te.fl.values)
        y = np.concatenate(acc["y"])
        fl = np.concatenate(acc["fl"])
        pb = np.vstack(acc["base"]); pe = np.vstack(acc["earn"]); pc = np.vstack(acc["clim"])
        lb, le, lc = L.log_loss(pb, y), L.log_loss(pe, y), L.log_loss(pc, y)
        hit = np.isin(y, [0, 3]).astype(float)
        mm = np.array([m for _, m in acc["mults"]])
        out.append(dict(h=h, n=len(y), ll_clim=lc, ll_base=lb, ll_earn=le,
                        skill_base=L.skill_score(lb, lc), skill_earn=L.skill_score(le, lc),
                        d_ll=lb - le, brier_base=L.brier_multi(pb, y),
                        brier_earn=L.brier_multi(pe, y),
                        ece_base=L.ece(pb[:, 0] + pb[:, 3], hit),
                        ece_earn=L.ece(pe[:, 0] + pe[:, 3], hit),
                        n_flag=int(fl.sum()),
                        ll_base_FLAG=L.log_loss(pb[fl], y[fl]),
                        ll_earn_FLAG=L.log_loss(pe[fl], y[fl]),
                        ll_clim_FLAG=L.log_loss(pc[fl], y[fl]),
                        skillgain_FLAG=L.skill_score(L.log_loss(pe[fl], y[fl]),
                                                     L.log_loss(pb[fl], y[fl])),
                        ll_base_UNFLAG=L.log_loss(pb[~fl], y[~fl]),
                        ll_earn_UNFLAG=L.log_loss(pe[~fl], y[~fl]),
                        fitted_mult_med=np.median(mm), fitted_mult_min=mm.min(),
                        fitted_mult_max=mm.max()))
    sdf = pd.DataFrame(out)
    print(sdf.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("  skillgain_FLAG = fraction of the base model's log loss removed by the flag, "
          "measured ON FLAGGED WINDOWS ONLY.")

    # calibration ONLY on flagged windows: does the base model underpredict there?
    print("\nCALIBRATION ON FLAGGED WINDOWS ONLY (does the un-adjusted model underpredict?)")
    for h in HORIZONS:
        recs = []
        for s in syms:
            fwd = L.forward_simple_return(close[s], h)
            den = shx[s] * np.sqrt(h)
            fl, cm = flags(s, h), coverage_mask(s)
            dd = pd.DataFrame({"fwd": fwd.values, "sig": den.values, "fl": fl, "cm": cm},
                              index=close.index).dropna()
            recs.append(dd[dd.cm])
        d = pd.concat(recs)
        d["z"] = d.fwd / d.sig
        d["bk"] = L.bucketize(d.fwd, THR).map(idx_map).astype(int)
        d["year"] = d.index.year
        years = sorted(d.year.unique())
        P, Y, FL = [], [], []
        for y in years:
            tr, te = d[d.year < y], d[d.year == y]
            if len(tr) < 5000 or len(te) == 0 or y - years[0] < 5:
                continue
            k = rsd(tr.z.values)
            zs = np.sort(tr.z.values / k)
            sig = te.sig.values * k
            F1 = np.searchsorted(zs, -THR / sig) / len(zs)
            F3 = np.searchsorted(zs, THR / sig) / len(zs)
            P.append(F1 + 1 - F3)
            Y.append(np.isin(te.bk.values, [0, 3]).astype(float))
            FL.append(te.fl.values)
        P, Y, FL = np.concatenate(P), np.concatenate(Y), np.concatenate(FL)
        print(f"  h={h:<3d} FLAGGED   n={FL.sum():>7,.0f} pred {P[FL].mean():.4f} obs "
              f"{Y[FL].mean():.4f} gap {Y[FL].mean()-P[FL].mean():+.4f}")
        print(f"       UNFLAGGED n={(~FL).sum():>7,.0f} pred {P[~FL].mean():.4f} obs "
              f"{Y[~FL].mean():.4f} gap {Y[~FL].mean()-P[~FL].mean():+.4f}")
    return em, edf


# ================================================================== 4. idio vs market
def _nsf(x):
    """Normal survival function 1-Phi(x) via erfc (scipy is not installed in vcp_env)."""
    import math
    f = np.vectorize(lambda v: 0.5 * math.erfc(v / math.sqrt(2.0)))
    return f(np.asarray(x, float))


def sec_idio(p, singles):
    close = p["Close"]
    print("\n" + "=" * 100)
    print("SECTION 3  MARKET vs IDIOSYNCRATIC (h=5)")
    print("=" * 100)
    lr = logret(close)
    mkt = lr["SPY"]
    W = 252
    rows = []
    for s in singles:
        y = lr[s]
        df = pd.concat([y.rename("y"), mkt.rename("m")], axis=1).dropna()
        if len(df) < 3 * W:
            continue
        beta = df.y.rolling(W).cov(df.m) / df.m.rolling(W).var()
        resid = df.y - beta.shift(1) * df.m          # beta known at t-1: no lookahead
        sd_tot = df.y.rolling(W).std(ddof=1)
        sd_res = resid.rolling(W).std(ddof=1)
        sd_mkt = df.m.rolling(W).std(ddof=1)
        r2 = 1 - (sd_res / sd_tot) ** 2
        ok = beta.notna() & sd_res.notna() & sd_tot.notna() & sd_mkt.notna()
        if ok.sum() < 500:
            continue
        rows.append(dict(sym=s, n=int(ok.sum()), beta=beta[ok].median(), r2=r2[ok].median(),
                         sd_tot=sd_tot[ok].median(), sd_res=sd_res[ok].median(),
                         sd_beta_mkt=(beta[ok].abs() * sd_mkt[ok]).median(),
                         sd_mkt=sd_mkt[ok].median()))
    idf = pd.DataFrame(rows)
    k = np.sqrt(5)
    # monotone map, so applying it to the median sigma == median of the mapped series
    idf["p_tot"] = 2 * _nsf(THR / (idf.sd_tot * k))
    idf["p_res"] = 2 * _nsf(THR / (idf.sd_res * k))
    idf["p_mkt"] = 2 * _nsf(THR / (idf.sd_beta_mkt * k))
    idf["share_p_res"] = idf.p_res / idf.p_tot
    idf["share_p_mkt"] = idf.p_mkt / idf.p_tot
    idf["var_share_mkt"] = (idf.sd_beta_mkt ** 2) / (idf.sd_tot ** 2)

    print(f"n symbols with >=500 usable rows: {len(idf)}  (rolling {W}d beta on SPY, beta lagged "
          f"1 day so the residual is out-of-sample-ish; medians per symbol then quantiles below)")
    cols = ["beta", "r2", "var_share_mkt", "sd_tot", "sd_res", "sd_beta_mkt",
            "p_tot", "p_res", "p_mkt", "share_p_res", "share_p_mkt"]
    print(idf[cols].quantile([.1, .25, .5, .75, .9]).to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nVARIANCE decomposition at the median name: market {idf.var_share_mkt.median():.1%}, "
          f"idiosyncratic {1-idf.var_share_mkt.median():.1%}  (R^2 median {idf.r2.median():.3f})")
    print(f"PROBABILITY decomposition, h=5, median name:")
    print(f"  P(|move|>2%) full vol      {idf.p_tot.median():.4f}")
    print(f"  ... residual (idio) only   {idf.p_res.median():.4f}  "
          f"= {idf.share_p_res.median():.1%} of full")
    print(f"  ... beta*market only       {idf.p_mkt.median():.4f}  "
          f"= {idf.share_p_mkt.median():.1%} of full")
    print("  (the two do NOT sum to 1: they are two different marginal probabilities, not a "
          "partition. Read them as 'how much of the move budget each source alone can deliver'.)")
    print(f"  killing the market component costs {1-idf.share_p_res.median():.1%} of P; killing "
          f"the idio component costs {1-idf.share_p_mkt.median():.1%} of P.")

    # by own-vol tercile: does the market share depend on how volatile the name is?
    print("\nBY OWN-VOL TERCILE of the name:")
    idf["vt"] = pd.qcut(idf.sd_tot, 3, labels=["low", "mid", "high"])
    print(idf.groupby("vt", observed=True)[["sd_tot", "beta", "r2", "var_share_mkt",
                                            "p_tot", "share_p_res", "share_p_mkt"]]
          .median().to_string(float_format=lambda x: f"{x:.4f}"))

    # market-state sensitivity of the forecast
    print("\nMARKET-STATE SENSITIVITY: hold the name's beta+residual vol fixed at the panel median,")
    print("move only SPY's 252d vol from its 10th to its 90th percentile:")
    spy_sd = lr["SPY"].rolling(W).std(ddof=1).dropna()
    b_med, sres_med = idf.beta.median(), idf.sd_res.median()
    ref = None
    for lab, q in (("p10", .1), ("p50", .5), ("p90", .9), ("p99", .99)):
        sm = spy_sd.quantile(q)
        st = np.sqrt((b_med * sm) ** 2 + sres_med ** 2)
        pp = float(2 * _nsf(THR / (st * k)))
        ref = ref or pp
        print(f"  SPY vol {lab} = {sm:.4f}/day ({sm*np.sqrt(252):.1%} ann)  -> stock sd "
              f"{st:.4f}  P(|move5|>2%) = {pp:.4f}  ({pp/ref:.2f}x the p10 case)")

    # ---- empirical: does market vol add anything beyond the stock's own vol?
    print("\nEMPIRICAL: regress log(realized fwd 5d vol) on log(own EWMA vol) [+ log(SPY EWMA vol)]")
    print("  (sub-sampled to every 5th DATE so the h=5 windows do not overlap)")
    rv = L.realized_vol_forward(close[singles], 5)
    own = L.vol_ewma(close[singles], 0.94)
    mv = L.vol_ewma(close[["SPY"]], 0.94)["SPY"]
    Y = np.log(rv.where(rv > 0)).stack(dropna=True)
    X1 = np.log(own.where(own > 0)).stack(dropna=True)
    d = pd.DataFrame({"y": Y, "x1": X1}).dropna()
    d["x2"] = np.log(mv).reindex(d.index.get_level_values(0)).values
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    dts = np.array(sorted(d.index.get_level_values(0).unique()))
    d = d[d.index.get_level_values(0).isin(dts[::5])]
    yv = d.y.values

    def fit(cols):
        A = np.column_stack([np.ones(len(d))] + [d[c].values for c in cols])
        b, *_ = np.linalg.lstsq(A, yv, rcond=None)
        r2 = 1 - ((yv - A @ b) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
        return b, r2
    b1, r1 = fit(["x1"])
    b2, r2b = fit(["x1", "x2"])
    b3, r3 = fit(["x2"])
    print(f"  n (non-overlapping) = {len(d):,}  over {d.index.get_level_values(0).nunique()} dates")
    print(f"  market vol only : coef={b3[1]:.3f}                 R2={r3:.4f}")
    print(f"  own vol only    : coef={b1[1]:.3f}                 R2={r1:.4f}")
    print(f"  own + market    : own={b2[1]:.3f} mkt={b2[2]:.3f}   R2={r2b:.4f}  dR2 over own-only "
          f"= +{r2b-r1:.4f}")
    # date-clustered bootstrap on the incremental R2 and the market coefficient
    dd = d.index.get_level_values(0)
    ud = np.array(sorted(dd.unique()))
    gi = {u: np.flatnonzero(dd == u) for u in ud}
    bs = []
    for _ in range(200):
        pick = RNG.integers(0, len(ud), len(ud))
        sel = np.concatenate([gi[ud[i]] for i in pick])
        A = np.column_stack([np.ones(len(sel)), d.x1.values[sel], d.x2.values[sel]])
        yb = yv[sel]
        bb, *_ = np.linalg.lstsq(A, yb, rcond=None)
        bs.append(bb[2])
    print(f"  market-vol coefficient, DATE-clustered bootstrap 95%CI: "
          f"[{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}]")
    return idf


# ================================================================== 5. pooled model calibration
def build_pooled_model(p, singles):
    """Walk-forward pooled model: sigma_pred_h = k_h * sigma_hat(t) * sqrt(h), z-dist = empirical
    pooled z from strictly prior years. Returns long-form frame with predicted bucket probs."""
    close = p["Close"]
    sh = sigma_hat(p, singles)
    out = {}
    for h in HORIZONS:
        fwd = L.forward_simple_return(close[singles], h)
        den = (sh[singles] * np.sqrt(h)).replace(0, np.nan)
        z = (fwd / den).replace([np.inf, -np.inf], np.nan)
        F = fwd.stack(dropna=True)
        Z = z.stack(dropna=True)
        S = den.stack(dropna=True)
        d = pd.DataFrame({"fwd": F, "z": Z, "sig": S}).dropna()
        idx_map = {b: i for i, b in enumerate(L.BUCKETS)}
        d["bk"] = L.bucketize(d.fwd, THR).map(idx_map).astype(int)
        dates = d.index.get_level_values(0)
        d["year"] = dates.year
        preds = np.full((len(d), 4), np.nan)
        clim = np.full((len(d), 4), np.nan)
        years = sorted(d.year.unique())
        for y in years[5:]:
            tr = d.year < y
            te = d.year == y
            if tr.sum() < 5000 or te.sum() == 0:
                continue
            ztr = d.z[tr].values
            k = ztr.std(ddof=1)
            zs = np.sort(ztr / k)                       # unit-variance standardized z, train only
            sig = d.sig[te].values * k
            a = -THR / sig
            b_ = 0.0
            c_ = THR / sig
            F1 = np.searchsorted(zs, a) / len(zs)       # P(z<=-thr/sig)
            F2 = np.searchsorted(zs, b_) / len(zs)
            F3 = np.searchsorted(zs, c_) / len(zs)
            pr = np.column_stack([F1, F2 - F1, F3 - F2, 1 - F3])
            pr = np.clip(pr, 1e-6, 1)
            pr = pr / pr.sum(1, keepdims=True)
            preds[te.values] = pr
            clim[te.values] = L.climatology(d.bk[tr].values)
        d[["p0", "p1", "p2", "p3"]] = preds
        d[["c0", "c1", "c2", "c3"]] = clim
        out[h] = d.dropna(subset=["p0", "c0"])
    return out


def sec_calib(p, singles, models=None):
    close = p["Close"]
    print("\n" + "=" * 100)
    print("SECTION 4  POOLED-MODEL CALIBRATION BY OWN-VOL AND PRICE DECILE (walk-forward, OOS)")
    print("  model: sigma_pred = k_h*YZ20(t)*sqrt(h), empirical pooled z-CDF, both fit on prior "
          "years only")
    print("=" * 100)
    models = models or build_pooled_model(p, singles)
    lr = logret(close)
    long_vol = lr[singles].expanding(250).std(ddof=1).shift(1)      # own long-run vol, no lookahead
    for h in HORIZONS:
        d = models[h]
        pm = d[["p0", "p1", "p2", "p3"]].values
        cm = d[["c0", "c1", "c2", "c3"]].values
        y = d.bk.values
        ll_m, ll_c = L.log_loss(pm, y), L.log_loss(cm, y)
        print(f"\n--- h={h}  n={len(d):,}  OOS test years {d.year.min()}-{d.year.max()} ---")
        print(f"  log loss model {ll_m:.4f}  climatology {ll_c:.4f}  skill "
              f"{L.skill_score(ll_m, ll_c):+.4f}   brier {L.brier_multi(pm,y):.4f} vs "
              f"{L.brier_multi(cm,y):.4f}")
        pbig = pm[:, 0] + pm[:, 3]
        hit = np.isin(y, [0, 3]).astype(float)
        print(f"  P(|move|>2%): mean pred {pbig.mean():.4f} observed {hit.mean():.4f}  "
              f"ECE {L.ece(pbig, hit):.4f}")
        rel = L.reliability(pbig, hit, 10)
        print("  reliability (pooled):")
        print(rel.to_string(float_format=lambda x: f"{x:.4f}"))

        # own-vol decile
        lv = long_vol.stack(dropna=True).reindex(d.index)
        dec = pd.qcut(lv, 10, labels=False, duplicates="drop")
        print("  BY OWN LONG-RUN-VOL DECILE (decile computed from data strictly before t):")
        rr = []
        for k in range(10):
            m = (dec == k).values
            if m.sum() < 500:
                continue
            rr.append(dict(decile=k, n=int(m.sum()), ann_vol=float(lv[m].mean() * np.sqrt(252)),
                           pred=pbig[m].mean(), obs=hit[m].mean(),
                           gap=hit[m].mean() - pbig[m].mean(),
                           ratio=hit[m].mean() / pbig[m].mean(),
                           ece=L.ece(pbig[m], hit[m]),
                           ll=L.log_loss(pm[m], y[m]), ll_clim=L.log_loss(cm[m], y[m]),
                           skill=L.skill_score(L.log_loss(pm[m], y[m]),
                                               L.log_loss(cm[m], y[m])),
                           pred_dn=pm[m, 0].mean(), obs_dn=(y[m] == 0).mean(),
                           pred_up=pm[m, 3].mean(), obs_up=(y[m] == 3).mean()))
        print(pd.DataFrame(rr).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        # price decile
        pr = close[singles].shift(1).stack(dropna=True).reindex(d.index)
        pdec = pd.qcut(pr, 10, labels=False, duplicates="drop")
        rr = []
        for k in range(10):
            m = (pdec == k).values
            if m.sum() < 500:
                continue
            rr.append(dict(price_decile=k, n=int(m.sum()), med_price=float(pr[m].median()),
                           pred=pbig[m].mean(), obs=hit[m].mean(),
                           ratio=hit[m].mean() / pbig[m].mean(), ece=L.ece(pbig[m], hit[m]),
                           skill=L.skill_score(L.log_loss(pm[m], y[m]),
                                               L.log_loss(cm[m], y[m]))))
        print("  BY PRICE DECILE (lagged close):")
        print(pd.DataFrame(rr).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        # WHY the top vol decile's up/down split is wrong: the mu=0 assumption. Show realised drift.
        print("  DRIFT BY OWN-VOL DECILE (the model assumes mu=0; is that true per decile?):")
        rr = []
        for k in range(10):
            m = (dec == k).values
            if m.sum() < 500:
                continue
            f = d.fwd.values[m]
            sg = d.sig.values[m]
            rr.append(dict(decile=k, n=int(m.sum()), ann_vol=float(lv[m].mean() * np.sqrt(252)),
                           mean_fwd=f.mean(), median_fwd=np.median(f),
                           mean_fwd_in_sigma=f.mean() / sg.mean(),
                           median_fwd_in_sigma=np.median(f) / sg.mean(),
                           dn_minus_up_pred=pm[m, 0].mean() - pm[m, 3].mean(),
                           dn_minus_up_obs=(y[m] == 0).mean() - (y[m] == 3).mean(),
                           err_dn=(y[m] == 0).mean() - pm[m, 0].mean(),
                           err_up=(y[m] == 3).mean() - pm[m, 3].mean()))
        print(pd.DataFrame(rr).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return models


# ================================================================== 6. survivorship proxy
def sec_surv(p, singles):
    close = p["Close"]
    print("\n" + "=" * 100)
    print("SECTION 5  SURVIVORSHIP PROXY: full-25y-history names vs late arrivals")
    print("  (all names in the panel are TODAY-LISTED survivors, so this is a LOWER BOUND on the")
    print("   true bias: it can only measure listing-age effects, never actual delistings.)")
    print("=" * 100)
    first = {s: close[s].first_valid_index() for s in singles}
    full = [s for s in singles if first[s] <= pd.Timestamp("2001-12-31")]
    late = [s for s in singles if first[s] > pd.Timestamp("2010-01-01")]
    mid = [s for s in singles if s not in full and s not in late]
    print(f"full-history (first bar <= 2001-12-31): {len(full)} symbols")
    print(f"mid (2002-2009): {len(mid)};  late (>= 2010): {len(late)}")
    lr = logret(close)
    for lab, syms in (("FULL 25y", full), ("MID 2002-09", mid), ("LATE >=2010", late)):
        d1 = lr[syms].stack(dropna=True).values
        print(f"\n{lab}: n daily obs {len(d1):,}  daily sd {d1.std(ddof=1):.4f}  "
              f"skew {pd.Series(d1).skew():.3f}  kurt {pd.Series(d1).kurt():.1f}")
        print(f"   P(day <= -5%) {np.mean(d1<=-0.05):.5f}   P(day <= -10%) "
              f"{np.mean(d1<=-0.10):.5f}   P(day <= -20%) {np.mean(d1<=-0.20):.6f}")
        print(f"   P(day >= +5%) {np.mean(d1>=0.05):.5f}   P(day >= +10%) "
              f"{np.mean(d1>=0.10):.5f}   ratio down/up @5% {np.mean(d1<=-0.05)/max(np.mean(d1>=0.05),1e-9):.3f}")
        for h in (5, 21):
            o = stack_obs(close, syms, h)
            fr = np.bincount(o.bk.values, minlength=4) / len(o)
            v = o.fwd.values
            print(f"   h={h}: n {len(o):,} down_big {fr[0]:.4f} up_big {fr[3]:.4f} "
                  f"P|mv|>2% {fr[0]+fr[3]:.4f}  P(r<=-10%) {np.mean(v<=-0.10):.5f}  "
                  f"P(r<=-20%) {np.mean(v<=-0.20):.5f}  q01 {np.quantile(v,.01):.4f}")

    # ---- how much of the group difference survives conditioning on the name's own vol?
    # This is the question that matters for the model: a scale-and-shape model already SEES that
    # late arrivals are more volatile. Only the STANDARDIZED tail difference is a real bias.
    print("\nVOL-STANDARDIZED left tail (z = r_h / (YZ20(t)*sqrt(h)) -- what the model actually")
    print("gets wrong if the two groups differ AFTER conditioning on their own observed vol):")
    sh = sigma_hat(p, singles)
    for h in (1, 5, 21):
        line = []
        for lab, syms in (("FULL", full), ("MID", mid), ("LATE", late)):
            fwd = L.forward_simple_return(close[syms], h)
            z = (fwd / (sh[syms] * np.sqrt(h))).replace([np.inf, -np.inf], np.nan)
            zv = z.stack(dropna=True).values
            line.append(f"{lab} n={len(zv):>7,d} P(z<-2)={np.mean(zv<-2):.4f} "
                        f"P(z<-3)={np.mean(zv<-3):.4f} P(z<-4)={np.mean(zv<-4):.5f} "
                        f"q01={np.quantile(zv,.01):.2f}")
        print(f"  h={h}:")
        for x in line:
            print(f"     {x}")

    # ---- explicit sensitivity: what would a delisting cohort do to the 21d left tail?
    print("\nDELISTING SENSITIVITY (ARITHMETIC, NOT MEASURED -- the panel has zero delisted names,")
    print("all 47 dropped tickers returned 0 bars from yfinance, so F cannot be estimated here):")
    o = stack_obs(close, singles, 21)
    base_tail = float(np.mean(o.fwd.values <= -0.20))
    for F in (0.01, 0.02, 0.04):
        # F = fraction of name-years that end in a delist-for-cause; each contributes ~3 monthly
        # windows at <= -20%. weight = F * 3/12 of that name-year's 21d windows.
        w = F * 0.25
        newt = base_tail * (1 - w) + 1.0 * w
        print(f"  if {F:.0%} of name-years delisted for cause (terminal quarter all <= -20%): "
              f"P(r21 <= -20%) goes {base_tail:.4f} -> {newt:.4f}  ({newt/base_tail:.2f}x)")
    print("  Read this as an order-of-magnitude bound only. The panel's own numbers are the")
    print("  UNADJUSTED survivor numbers; the index series are the honest benchmark.")

    # same names, early vs late in their own life (age effect, not survivorship)
    print("\nAGE EFFECT within LATE names: their first 3 years vs after")
    rowsA, rowsB = [], []
    for s in late:
        x = lr[s].dropna()
        cut = x.index[0] + pd.DateOffset(years=3)
        rowsA.append(x[x.index <= cut].values)
        rowsB.append(x[x.index > cut].values)
    a, b = np.concatenate(rowsA), np.concatenate(rowsB)
    print(f"  first 3y: n {len(a):,} sd {a.std(ddof=1):.4f} P(day<=-10%) {np.mean(a<=-0.10):.5f}")
    print(f"  after:    n {len(b):,} sd {b.std(ddof=1):.4f} P(day<=-10%) {np.mean(b<=-0.10):.5f}")

    # index (unbiased) comparison of the same tails
    print("\nINDEX (unbiased) tails for reference:")
    for ix in ["SPY", "QQQ", "^GSPC"]:
        x = lr[ix].dropna().values
        print(f"  {ix:7s} n {len(x):,} sd {x.std(ddof=1):.4f} P(day<=-5%) {np.mean(x<=-0.05):.5f} "
              f"P(day<=-10%) {np.mean(x<=-0.10):.5f}")
    # what a missing-delisting correction plausibly costs: how many panel names had a >50% drawdown
    print("\nSurvivor-only diagnostic: distribution of worst 21d return per name")
    w = []
    for s in singles:
        f = L.forward_simple_return(close[s], 21).dropna()
        if len(f) > 500:
            w.append((s, f.min()))
    wd = pd.DataFrame(w, columns=["sym", "worst21"])
    print(f"  n {len(wd)}  median worst-21d {wd.worst21.median():.3f}  "
          f"p10 {wd.worst21.quantile(.1):.3f}  min {wd.worst21.min():.3f}  "
          f"share with worst21 <= -50%: {(wd.worst21<=-0.5).mean():.1%}")


# ================================================================== 7. pooled vs per-asset z-shape
def sec_shape(p, singles, sector):
    """THE decision test: fit the z-CDF pooled, per-sector, or per-symbol (with shrinkage) --
    all on strictly prior years -- and compare OOS log loss. If pooled wins, one shape is enough."""
    close = p["Close"]
    print("\n" + "=" * 100)
    print("SECTION 6  POOLED vs PER-SECTOR vs PER-SYMBOL z-SHAPE (walk-forward OOS)")
    print("  sigma_pred = k*YZ20(t)*sqrt(h) with k and the z-CDF both fit on PRIOR YEARS ONLY.")
    print("  Only the SHAPE source changes between the variants.")
    print("=" * 100)
    sh = sigma_hat(p, singles)
    idx_map = {b: i for i, b in enumerate(L.BUCKETS)}

    def probs_from(zs, sig):
        """Empirical-CDF bucket probabilities. zs must be sorted, unit-ish scale."""
        F1 = np.searchsorted(zs, -THR / sig) / len(zs)
        F2 = np.searchsorted(zs, np.zeros(len(sig))) / len(zs)
        F3 = np.searchsorted(zs, THR / sig) / len(zs)
        pr = np.clip(np.column_stack([F1, F2 - F1, F3 - F2, 1 - F3]), 1e-6, 1)
        return pr / pr.sum(1, keepdims=True)

    for h in HORIZONS:
        fwd = L.forward_simple_return(close[singles], h)
        den = sh[singles] * np.sqrt(h)
        F = fwd.stack(dropna=True)
        S = den.stack(dropna=True)
        d = pd.DataFrame({"fwd": F, "sig": S}).dropna()
        d["z"] = d.fwd / d.sig
        d["bk"] = L.bucketize(d.fwd, THR).map(idx_map).astype(int)
        d["sym"] = d.index.get_level_values(1)
        d["sec"] = d["sym"].map(sector)
        d["year"] = d.index.get_level_values(0).year
        years = sorted(d.year.unique())
        acc = defaultdict(list)
        for y in years[5:]:
            tr, te = d[d.year < y], d[d.year == y]
            if len(tr) < 20000 or len(te) == 0:
                continue
            k = tr.z.std(ddof=1)
            zs_pool = np.sort(tr.z.values / k)
            sig_te = te.sig.values * k
            acc["pooled"].append(probs_from(zs_pool, sig_te))
            # per-sector
            out = np.zeros((len(te), 4))
            for sc, gi in te.groupby("sec", observed=True).indices.items():
                trs = tr[tr.sec == sc]
                zs = np.sort(trs.z.values / k) if len(trs) >= 5000 else zs_pool
                out[gi] = probs_from(zs, sig_te[gi])
            acc["per_sector"].append(out)
            # per-symbol, with a shrinkage blend toward the pooled shape
            for lam, tag in ((0.0, "per_symbol"), (0.5, "per_symbol_shrunk50")):
                out = np.zeros((len(te), 4))
                for sym, gi in te.groupby("sym", observed=True).indices.items():
                    trs = tr[tr.sym == sym]
                    if len(trs) >= 750:
                        zs = np.sort(trs.z.values / k)
                        pr = probs_from(zs, sig_te[gi])
                        pr = (1 - lam) * pr + lam * probs_from(zs_pool, sig_te[gi])
                    else:
                        pr = probs_from(zs_pool, sig_te[gi])
                    out[gi] = pr
                acc[tag].append(out)
            # a normal-shape control
            import math
            nf = np.vectorize(lambda v: 0.5 * math.erfc(v / math.sqrt(2.0)))
            F1 = nf(THR / sig_te)
            F3 = 1 - nf(THR / sig_te)
            pr = np.clip(np.column_stack([F1, 0.5 - F1, F3 - 0.5, 1 - F3]), 1e-6, 1)
            acc["normal"].append(pr / pr.sum(1, keepdims=True))
            acc["clim"].append(np.tile(L.climatology(tr.bk.values), (len(te), 1)))
            acc["y"].append(te.bk.values)
            acc["sec_of"].append(te.sec.values)
        yv = np.concatenate(acc["y"])
        lc = L.log_loss(np.vstack(acc["clim"]), yv)
        print(f"\n--- h={h}  n={len(yv):,}  OOS 2006-2026  climatology log loss {lc:.5f} ---")
        rows = []
        for tag in ("normal", "pooled", "per_sector", "per_symbol_shrunk50", "per_symbol"):
            pm = np.vstack(acc[tag])
            ll = L.log_loss(pm, yv)
            hit = np.isin(yv, [0, 3]).astype(float)
            rows.append(dict(shape=tag, log_loss=ll, skill_vs_clim=L.skill_score(ll, lc),
                             brier=L.brier_multi(pm, yv),
                             ece_pbig=L.ece(pm[:, 0] + pm[:, 3], hit),
                             mean_pred_pbig=(pm[:, 0] + pm[:, 3]).mean(), obs_pbig=hit.mean()))
        r = pd.DataFrame(rows)
        base = float(r.loc[r["shape"] == "pooled", "log_loss"].iloc[0])
        r["d_ll_vs_pooled"] = r.log_loss - base
        print(r.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
        # per-sector breakdown of pooled-vs-sector at this horizon
        secv = np.concatenate(acc["sec_of"])
        pp, ps = np.vstack(acc["pooled"]), np.vstack(acc["per_sector"])
        rows = []
        for sc in sorted(set(secv)):
            m = secv == sc
            rows.append(dict(sector=sc, n=int(m.sum()),
                             ll_pooled=L.log_loss(pp[m], yv[m]),
                             ll_sector=L.log_loss(ps[m], yv[m]),
                             gain=L.log_loss(pp[m], yv[m]) - L.log_loss(ps[m], yv[m]),
                             pred_pooled=(pp[m, 0] + pp[m, 3]).mean(),
                             pred_sector=(ps[m, 0] + ps[m, 3]).mean(),
                             obs=np.isin(yv[m], [0, 3]).mean()))
        print("  per-sector detail (gain>0 means the sector-specific shape helped):")
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.5f}"))


def sec_noise(p, singles):
    """Is the per-symbol dispersion in P(|z|>2) real, or just sampling noise?"""
    close = p["Close"]
    print("\n" + "=" * 100)
    print("SECTION 1c  IS THE PER-SYMBOL z-SHAPE DISPERSION REAL OR SAMPLING NOISE?")
    print("=" * 100)
    sh = sigma_hat(p, singles)
    for h in (1, 5, 21):
        fwd = L.forward_simple_return(close[singles], h)
        z = (fwd / (sh[singles] * np.sqrt(h))).replace([np.inf, -np.inf], np.nan)
        rows = []
        for s in singles:
            zv = z[s].dropna().values
            if len(zv) < 400:
                continue
            rows.append((s, len(zv), np.mean(np.abs(zv) > 2)))
        q = pd.DataFrame(rows, columns=["sym", "n", "p"])
        pbar = q.p.mean()
        # binomial noise inflated by the h-day overlap (effective n = n/h)
        var_noise = float((pbar * (1 - pbar) / (q.n / max(h, 1))).mean())
        var_obs = float(q.p.var(ddof=1))
        real = max(var_obs - var_noise, 0.0)
        print(f"  h={h:<3d} nsym={len(q)}  mean P(|z|>2)={pbar:.4f}  observed sd={np.sqrt(var_obs):.4f}"
              f"  noise sd={np.sqrt(var_noise):.4f}  TRUE sd={np.sqrt(real):.4f}"
              f"  ({real/var_obs:.0%} of the variance is real cross-symbol shape difference)")
        print(f"       => a genuine +-1 sd shape spread of {np.sqrt(real)/pbar:.0%} around "
              f"P(|z|>2)={pbar:.4f}")


# ==================================================================
if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    p, singles, sector, exclusive = setup()
    print(f"panel: {p['Close'].shape[1]} symbols kept (>=500 closes), {len(singles)} singles, "
          f"{p['Close'].index[0].date()} -> {p['Close'].index[-1].date()}")
    models = None
    if what in ("clim", "all"):
        sec_clim(p, singles, sector, exclusive)
    if what in ("z", "all"):
        sec_z(p, singles, sector)
    if what in ("earn", "all"):
        sec_earn(p, singles)
    if what in ("idio", "all"):
        sec_idio(p, singles)
    if what in ("calib", "all"):
        models = sec_calib(p, singles, models)
    if what in ("surv", "all"):
        sec_surv(p, singles)
    if what in ("noise", "all"):
        sec_noise(p, singles)
    if what in ("shape", "all"):
        sec_shape(p, singles, sector)
