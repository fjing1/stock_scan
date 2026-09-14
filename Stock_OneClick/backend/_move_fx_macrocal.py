"""
_move_fx_macrocal.py -- does the MACRO / MARKET CALENDAR improve move_prob.py?

Factor family: things you know about the FUTURE window before it starts -- day of week, holiday
adjacency, turn of month, quarter end, triple witching, the December stretch, and the two big
scheduled macro prints (FOMC decision, CPI release).

The shipped model treats every future h-day window as interchangeable except for the single-name
earnings multiplier. This script asks whether the macro calendar deserves the same treatment,
and it carries every claim through to BSS2 / ECE, not just to log-vol R-squared.

METHOD (identical to _move_validate.py so the deltas are comparable):
  expanding walk-forward by calendar year, >=5 training years, test years 2006-2026;
  every coefficient, the empirical z table and the width kappa refit on strictly prior years;
  BSS2 = Brier skill on P(|move| >= thr) against EACH TICKER'S OWN train-year base rate shrunk
  40 pseudo-counts toward pooled; ECE = 10 equal-count bins on the same probability.
  CIs are a block bootstrap over TRADING DATES with block length > h (overlapping windows and
  ~1-correlated cross-section mean pooled n is not independent n).

Run: ../../vcp_env/bin/python _move_fx_macrocal.py            # full study
     ../../vcp_env/bin/python _move_fx_macrocal.py --quick    # descriptive part only
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _move_data as D
import _move_lib as L
import move_prob as M

MIN_TRAIN_YEARS = 5
INNER_FOLDS = 5
SHRINK = 40.0
THRS = (0.02, 0.05)
MACRO = Path(__file__).with_name("_move_fx_macro_dates.json")
CACHE = Path(__file__).with_name("_move_fx_macrocal_cache.pkl")
t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)


# ===================================================================== calendar construction
def trading_calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Every calendar flag that is derivable from the trading index alone.

    EX-ANTE NOTE. All of these are known before the window starts in production: the NYSE
    publishes its holiday schedule years ahead and expiration/turn-of-month are pure date rules.
    The ONE caveat is that here holidays are INFERRED from realized gaps in the index, so the
    handful of unscheduled closures (Sep 2001, Sandy Oct 2012, Bush funeral Dec 2018,
    Carter funeral Jan 2025) are in-sample knowledge. They are ~6 sessions out of 6,289; the
    `holiday_scheduled_only` column repeats the flag using ONLY rule-based federal holidays so the
    difference can be measured.
    """
    idx = pd.DatetimeIndex(index)
    f = pd.DataFrame(index=idx)
    f["dow"] = idx.dayofweek                                    # 0=Mon .. 4=Fri

    # --- holidays inferred from gaps: count weekdays skipped between consecutive sessions
    prev, nxt = idx[:-1], idx[1:]
    skipped = np.array([len(pd.bdate_range(a + pd.Timedelta(days=1), b - pd.Timedelta(days=1)))
                        for a, b in zip(prev, nxt)])
    pre = np.zeros(len(idx), bool); post = np.zeros(len(idx), bool)
    pre[:-1] = skipped > 0
    post[1:] = skipped > 0
    f["pre_holiday"] = pre
    f["post_holiday"] = post

    # --- turn of month: last session of a month, first 3 sessions of the next
    mo = idx.to_period("M")
    last_of_month = np.r_[mo[:-1] != mo[1:], True]
    rank_in_month = pd.Series(1, index=idx).groupby(mo).cumcount().values
    f["tom"] = last_of_month | (rank_in_month <= 2)
    f["month_last"] = last_of_month
    f["quarter_end"] = last_of_month & np.isin(idx.month, [3, 6, 9, 12])

    # --- triple witching: third Friday of Mar/Jun/Sep/Dec; if not a session, the prior session
    tw = np.zeros(len(idx), bool)
    for y in sorted(set(idx.year)):
        for m in (3, 6, 9, 12):
            fri = [d for d in pd.date_range(f"{y}-{m:02d}-01", periods=31, freq="D")
                   if d.month == m and d.dayofweek == 4]
            if len(fri) < 3:
                continue
            d3 = fri[2]
            pos = idx.searchsorted(d3, side="right") - 1        # that Friday, or the prior session
            if 0 <= pos < len(idx) and abs((idx[pos] - d3).days) <= 3:
                tw[pos] = True
    f["triple_witch"] = tw
    # monthly opex = third Friday of every month (superset of triple witching)
    mw = np.zeros(len(idx), bool)
    for y in sorted(set(idx.year)):
        for m in range(1, 13):
            fri = [d for d in pd.date_range(f"{y}-{m:02d}-01", periods=31, freq="D")
                   if d.month == m and d.dayofweek == 4]
            if len(fri) < 3:
                continue
            pos = idx.searchsorted(fri[2], side="right") - 1
            if 0 <= pos < len(idx) and abs((idx[pos] - fri[2]).days) <= 3:
                mw[pos] = True
    f["opex"] = mw

    f["dec_quiet"] = (idx.month == 12) & (idx.day >= 20)
    f["jan_first"] = (idx.month == 1) & (idx.day <= 3)
    f["summer"] = np.isin(idx.month, [7, 8])
    return f


def macro_flags(index: pd.DatetimeIndex):
    """Master-calendar boolean for 'an FOMC decision / a CPI print lands on this session', plus a
    coverage mask marking the span over which the cached date list is complete.

    FOMC dates that do NOT fall on a trading session are DROPPED, not mapped forward: in this
    sample that filter hits exactly one date, 2020-03-15, the Sunday emergency cut. That was an
    unscheduled action, so a forecaster could not have known it on Friday 2020-03-13 -- keeping it
    would import the single most violent week of the sample as if it had been on the calendar.
    (Cost: the cancelled 2020-03-17/18 scheduled meeting is therefore unflagged.)

    CPI dates ARE mapped forward to the next session: 2017-04-14 and 2020-04-10 were Good Fridays,
    the print still happened, and the market's first chance to react was the following Monday --
    which a forecaster knows in advance from the published holiday schedule.
    """
    j = json.loads(MACRO.read_text())
    idx = pd.DatetimeIndex(index)
    out = {}
    for k, forward in (("fomc", False), ("cpi", True)):
        ev = pd.DatetimeIndex(pd.to_datetime(j[k]))
        ev = ev[(ev >= idx[0]) & (ev <= idx[-1])]
        on = idx.isin(ev)
        off = [x for x in ev if x not in idx]
        if forward and off:
            pos = idx.searchsorted(pd.DatetimeIndex(off), side="left")
            on[pos[pos < len(idx)]] = True
        hit = on
        # coverage: first..last verified event, with the tail pulled back 21 sessions so that a
        # window starting inside coverage cannot reach past the end of the verified list
        lo = idx.searchsorted(ev.min(), side="left")
        hi = max(idx.searchsorted(ev.max(), side="right") - 1 - 21, lo)
        cov = np.zeros(len(idx), bool)
        cov[lo:hi + 1] = True
        out[k], out[k + "_cov"] = hit, cov
        print(f"  {k}: {len(ev)} verified dates in span; {int(hit.sum())} flagged sessions; "
              f"{len(off)} off-calendar {'(mapped forward)' if forward else '(DROPPED)'} "
              f"{[str(x.date()) for x in off]}; coverage "
              f"{idx[lo].date()}..{idx[hi].date()} = {cov.mean():.1%} of sessions")
    return out


# ===================================================================== cache
def build(panel):
    """Per (group, h): X (shipped design), lr, tgt, yr, sid, di (row index into the master
    trading calendar).  Verified row-for-row identical to move_prob.build_cache."""
    C, Hi, Lo = panel["Close"], panel["High"], panel["Low"]
    vix = C["^VIX"] if "^VIX" in C.columns else None
    master = pd.DatetimeIndex(C.index)
    usable = [s for s in C.columns if s != "^VIX" and C[s].notna().sum() >= 500]
    groups = {"index": [s for s in usable if s in M.INDEX_LIKE],
              "single": [s for s in usable if s not in M.INDEX_LIKE]}
    cache = {}
    for g, syms in groups.items():
        cols = M.FEATS_OHLC + (["logvix"] if (g == "index" and vix is not None) else [])
        for h in M.HORIZONS:
            X, lr, tg, yr, sid, di = [], [], [], [], [], []
            for i, s in enumerate(syms):
                c = C[s].dropna()
                if len(c) < 400:
                    continue
                fe = M.build_features(c, Hi[s].reindex(c.index), Lo[s].reindex(c.index), vix)
                Xi = M._design(fe, cols)
                li = np.log(c.shift(-h) / c).values
                ti = M._clip_log(L.realized_vol_forward(c, h)).values
                ok = np.isfinite(Xi).all(axis=1) & np.isfinite(li)
                pos = master.searchsorted(c.index)              # symbol bar -> master bar
                X.append(Xi[ok]); lr.append(li[ok]); tg.append(ti[ok])
                yr.append(c.index.year.values[ok]); sid.append(np.full(int(ok.sum()), i))
                di.append(pos[ok])
            cache[(g, h)] = dict(X=np.vstack(X), lr=np.concatenate(lr), tgt=np.concatenate(tg),
                                 yr=np.concatenate(yr), sid=np.concatenate(sid),
                                 di=np.concatenate(di), cols=cols, symbols=syms)
            log(f"cached {g:<7} h={h:<3} rows={len(cache[(g,h)]['lr']):>9,} "
                f"syms={len(set(cache[(g,h)]['sid']))}")
    cache["_master"] = master
    return cache


def window_flag(di, h, flag_master, master_len, counts=False):
    """flag[t] = (number of) flagged master-calendar sessions in (t, t+h].

    The event day itself counts only if it is STRICTLY AFTER the last observed bar t: an FOMC
    statement at 14:00 or a CPI print at 08:30 on day t is already inside close[t], hence already
    in the model's features, not in the forward window.

    NOTE: this walks the MASTER calendar, not the symbol's own bars. For a symbol with a missing
    bar the two differ by at most the number of missing bars; every symbol here is liquid enough
    that this is rare, and using the master calendar is the ex-ante-correct definition anyway."""
    cs = np.concatenate([[0], np.cumsum(flag_master.astype(np.int32))])
    hi = np.minimum(di + h, master_len - 1)
    n = cs[hi + 1] - cs[di + 1]
    return n.astype(np.float64) if counts else (n > 0).astype(np.float64)


# ===================================================================== date-clustered bootstrap
class DateBlocks:
    """Per-DATE sufficient statistics + a circular block bootstrap over dates.

    Every statistic used here is a ratio of two sums over rows, so collapsing rows to per-date
    sums makes a 300-replicate bootstrap cost O(n_dates) instead of O(n_rows) per replicate.
    Block length must exceed h because forward windows overlap h-fold."""

    def __init__(self, di, h, n_boot=400, seed=7):
        self.uniq, self.inv = np.unique(di, return_inverse=True)
        self.nd = len(self.uniq)
        self.L = max(2 * h, 10)
        self.nblk = int(np.ceil(self.nd / self.L))
        rng = np.random.default_rng(seed)
        self.starts = rng.integers(0, self.nd, size=(n_boot, self.nblk))
        off = np.arange(self.L)
        self.pick = (self.starts[:, :, None] + off[None, None, :]).reshape(n_boot, -1) % self.nd

    def dsum(self, v):
        return np.bincount(self.inv, weights=v, minlength=self.nd)

    def ratio_ci(self, num, den, q=(2.5, 97.5)):
        """CI for sum(num)/sum(den) over resampled dates."""
        N, Dn = self.dsum(num), self.dsum(den)
        pt = N.sum() / Dn.sum()
        r = N[self.pick].sum(axis=1) / Dn[self.pick].sum(axis=1)
        return pt, np.percentile(r, q[0]), np.percentile(r, q[1])

    def skill_delta_ci(self, se_base, se_var, se_ref, q=(2.5, 97.5)):
        """CI for  BSS2(var) - BSS2(base) = (sum se_base - sum se_var) / sum se_ref."""
        B, V, R = self.dsum(se_base), self.dsum(se_var), self.dsum(se_ref)
        pt = (B.sum() - V.sum()) / R.sum()
        d = (B[self.pick].sum(axis=1) - V[self.pick].sum(axis=1)) / R[self.pick].sum(axis=1)
        return pt, np.percentile(d, q[0]), np.percentile(d, q[1]), float((d > 0).mean())


# ===================================================================== walk-forward engine
def ols(X, y):
    keep = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if keep.sum() < 200:
        return None
    b, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
    return b


def kappa_for(X, lr, tgt, h, yr, ty, years):
    ks = []
    fin = np.isfinite(tgt)
    for iy in [u for u in years if u < ty][-INNER_FOLDS:]:
        itr, iva = yr < iy, yr == iy
        if itr.sum() < 5000 or iva.sum() < 50:
            continue
        bi = ols(X[itr & fin], tgt[itr & fin])
        if bi is None:
            continue
        zi = np.sort(lr[itr] / (np.exp(X[itr] @ bi) * np.sqrt(h)))
        if len(zi) < 5000:
            continue
        ks.append(M._calibrate_kappa(zi, np.exp(X[iva] @ bi) * np.sqrt(h), lr[iva]))
    return float(np.median(ks)) if ks else 1.0


def walkforward(X, lr, tgt, yr, sid, h):
    """Return, concatenated over test years: p (n,4), a_te index rows, year, row index."""
    years = sorted(set(yr))
    test_years = years[MIN_TRAIN_YEARS:]
    fin = np.isfinite(tgt)
    P, IDX, YY = [], [], []
    SIG = []
    for ty in test_years:
        tr, te = yr < ty, yr == ty
        if tr.sum() < 2000 or te.sum() < 50:
            continue
        beta = ols(X[tr & fin], tgt[tr & fin])
        if beta is None:
            continue
        sig_tr = np.exp(X[tr] @ beta) * np.sqrt(h)
        sig_te = np.exp(X[te] @ beta) * np.sqrt(h)
        kap = kappa_for(X, lr, tgt, h, yr, ty, years)
        z_tr = np.sort(lr[tr] / sig_tr) * kap
        nz = len(z_tr)

        def F(v):
            return np.searchsorted(z_tr, v, side="right") / nz
        rows = []
        for thr in THRS:
            f_dn = F(np.log(1 - thr) / sig_te)
            f_up = F(np.log(1 + thr) / sig_te)
            f_0 = F(0.0)
            p = np.column_stack([f_dn, f_0 - f_dn, f_up - f_0, 1 - f_up])
            p = np.clip(p, 1e-4, None)
            p /= p.sum(axis=1, keepdims=True)
            rows.append(p)
        P.append(rows)
        SIG.append(sig_te)
        IDX.append(np.where(te)[0]); YY.append(np.full(int(te.sum()), ty))
    out = {thr: np.vstack([r[i] for r in P]) for i, thr in enumerate(THRS)}
    return out, np.concatenate(IDX), np.concatenate(YY), np.concatenate(SIG)


def ref_probs(lr, yr, sid, h, idx, thr):
    """Each ticker's own train-year base rate, shrunk 40 pseudo-counts toward pooled.
    Refit per test year on strictly prior years -- same object _move_validate.py uses."""
    a_all = np.where(lr <= np.log(1 - thr), 0,
                     np.where(lr <= 0, 1, np.where(lr < np.log(1 + thr), 2, 3)))
    years = sorted(set(yr))
    nsym = int(sid.max()) + 1
    out = np.empty((len(idx), 4))
    yy = yr[idx]
    for ty in sorted(set(yy)):
        tr = yr < ty
        a_tr = a_all[tr]
        pooled = np.bincount(a_tr, minlength=4) / len(a_tr)
        cnt = np.zeros((nsym, 4))
        np.add.at(cnt, (sid[tr], a_tr), 1.0)
        psym = (cnt + SHRINK * pooled) / (cnt.sum(axis=1, keepdims=True) + SHRINK)
        m = yy == ty
        out[m] = psym[sid[idx][m]]
    return out, a_all[idx]


def ece(p, hit, n_bins=10):
    return L.ece(np.asarray(p), np.asarray(hit, dtype=bool), n_bins)


def score(pm, pms, big):
    bs = ((pm - big) ** 2).mean()
    bref = ((pms - big) ** 2).mean()
    return 1 - bs / bref


# ===================================================================== PART 1: descriptive
def ratio_geo(bb, v, m1, m0):
    """exp(mean v | m1) / exp(mean v | m0) with a date-block CI. Written as per-date sums so the
    bootstrap costs O(n_dates) per replicate instead of O(n_rows)."""
    S1, S0 = bb.dsum(np.where(m1, v, 0.0)), bb.dsum(np.where(m0, v, 0.0))
    N1, N0 = bb.dsum(m1.astype(float)), bb.dsum(m0.astype(float))
    pt = np.exp(S1.sum() / N1.sum() - S0.sum() / N0.sum())
    r = np.exp(S1[bb.pick].sum(1) / np.maximum(N1[bb.pick].sum(1), 1)
               - S0[bb.pick].sum(1) / np.maximum(N0[bb.pick].sum(1), 1))
    return float(pt), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def part1(cache, master, FLAGS, out):
    """Ratio of realized h-day vol in windows CONTAINING each calendar day vs not, per group.
    RAW = realized/realized. EXCESS = realized divided by the shipped model's own forecast, i.e.
    what a sigma multiplier would have to correct. The EXCESS column uses one full-sample OLS fit
    and is therefore IN-SAMPLE, descriptive only; Part 2 refits everything on prior years."""
    print("\n" + "=" * 122)
    print("PART 1 (descriptive) -- realized forward-vol ratio, windows containing the day vs not."
          "  CI = date-block bootstrap 95%")
    print("=" * 122)
    rows = []
    for g in ("index", "single"):
        for h in (1, 5, 10, 21):
            d = cache[(g, h)]
            X, tgt, di = d["X"], d["tgt"], d["di"]
            fin = np.isfinite(tgt)
            beta = ols(X[fin], tgt[fin])
            resid = tgt - X @ beta            # log(realized) - log(forecast)   IN-SAMPLE
            bb = DateBlocks(di, h)
            print(f"\n  {g}  h={h}   n={len(tgt):,} rows over {bb.nd:,} dates  "
                  f"(block len {bb.L})")
            print(f"    {'flag':<14}{'share':>8}{'n_flag':>10}{'raw ratio':>12}{'95% CI':>18}"
                  f"{'excess(IS)':>12}{'95% CI':>18}")
            for name, fm in FLAGS.items():
                if name.endswith("_cov"):
                    continue
                w = window_flag(di, h, fm, len(master)) > 0
                cov = np.ones(len(w), bool)
                if name in ("fomc", "cpi"):
                    cov = FLAGS[name + "_cov"][di]
                sel = fin & cov
                m1, m0 = w & sel, (~w) & sel
                if m1.sum() < 200 or m0.sum() < 200:
                    continue
                pt1, lo1, hi1 = ratio_geo(bb, tgt, m1, m0)
                pt2, lo2, hi2 = ratio_geo(bb, resid, m1, m0)
                print(f"    {name:<14}{m1.mean():>8.1%}{m1.sum():>10,}{pt1:>12.4f}"
                      f"{f'[{lo1:.3f},{hi1:.3f}]':>18}{pt2:>12.4f}{f'[{lo2:.3f},{hi2:.3f}]':>18}")
                rows.append(dict(group=g, h=h, flag=name, share=float(m1.mean()),
                                 n_flag=int(m1.sum()), n_unflag=int(m0.sum()),
                                 raw=pt1, raw_lo=lo1, raw_hi=hi1,
                                 excess=pt2, excess_lo=lo2, excess_hi=hi2))
    out["part1"] = pd.DataFrame(rows)


# ===================================================================== PART 2: variants
def make_extras(di, h, master_len, FLAGS, cal, which):
    """Extra design columns for a variant. Counts inside (t, t+h], scaled so a coefficient reads
    as the log-vol effect of one full day of exposure."""
    cols, names = [], []

    def add(nm, v):
        cols.append(np.asarray(v, dtype=np.float64)); names.append(nm)

    if "cal" in which:
        for nm in ("pre_holiday", "post_holiday", "tom", "quarter_end", "triple_witch",
                   "dec_quiet"):
            add(nm, window_flag(di, h, FLAGS[nm], master_len, counts=True) / h)
    if "dow" in which:
        dnext = np.minimum(di + 1, master_len - 1)
        dw = cal["dow"].values[dnext]
        for k in (0, 1, 3, 4):                                  # Wednesday is the reference level
            add(f"dow{k}", (dw == k).astype(float))
    for ev in ("fomc", "cpi"):
        if ev in which:
            add(ev, window_flag(di, h, FLAGS[ev], master_len, counts=True))
            add(ev + "_nocov", (~FLAGS[ev + "_cov"][di]).astype(float))
    return (np.column_stack(cols) if cols else np.zeros((len(di), 0))), names


def part2(cache, master, FLAGS, cal, out, variants, groups=("index", "single"),
          horizons=(1, 5, 10, 21)):
    print("\n" + "=" * 122)
    print("PART 2 -- walk-forward: does it move BSS2 / ECE?  expanding, everything refit on "
          "strictly prior years, test 2006-2026")
    print("=" * 122)
    rows, coefs = [], []
    for g in groups:
        for h in horizons:
            d = cache[(g, h)]
            X0, lr, tgt, yr, sid, di = d["X"], d["lr"], d["tgt"], d["yr"], d["sid"], d["di"]
            res = {}
            for vname, which in variants.items():
                if which is None:
                    X, nm = X0, []
                else:
                    E, nm = make_extras(di, h, len(master), FLAGS, cal, which)
                    X = np.hstack([X0, E])
                P, idx, yy, _s = walkforward(X, lr, tgt, yr, sid, h)
                res[vname] = (P, idx, yy)
                if nm:
                    fin = np.isfinite(tgt)
                    b = ols(X[fin], tgt[fin])
                    coefs.append(dict(group=g, h=h, variant=vname,
                                      **{k: round(float(v), 4)
                                         for k, v in zip(nm, b[X0.shape[1]:])}))
            base_idx, yy = res["base"][1], res["base"][2]
            bb = DateBlocks(di[base_idx], h)
            for thr in THRS:
                pref, a = ref_probs(lr, yr, sid, h, base_idx, thr)
                big = np.isin(a, [0, 3]).astype(float)
                pms = pref[:, 0] + pref[:, 3]
                se_ref = (pms - big) ** 2
                pm = {k: v[0][thr][:, 0] + v[0][thr][:, 3] for k, v in res.items()}
                se = {k: (v - big) ** 2 for k, v in pm.items()}
                b0, e0 = score(pm["base"], pms, big), ece(pm["base"], big)
                for vname in variants:
                    if vname == "base":
                        continue
                    bv, ev = score(pm[vname], pms, big), ece(pm[vname], big)
                    pt, lo, hi, pgt = bb.skill_delta_ci(se["base"], se[vname], se_ref)
                    ny = tot = 0
                    for u in np.unique(yy):
                        m = yy == u
                        if m.sum() < 50:
                            continue
                        tot += 1
                        ny += int(se["base"][m].mean() > se[vname][m].mean())
                    rows.append(dict(group=g, h=h, thr=thr, variant=vname, n=len(a),
                                     n_dates=bb.nd, bss2_base=b0, bss2_var=bv, dbss2=bv - b0,
                                     dbss2_lo=lo, dbss2_hi=hi, boot_p_gt0=pgt,
                                     ece_base=e0, ece_var=ev, dece=ev - e0,
                                     yrs_better=f"{ny}/{tot}", obs=float(big.mean()),
                                     pred_base=float(pm["base"].mean()),
                                     pred_var=float(pm[vname].mean())))
            log(f"  scored {g} h={h}")
    out["part2"] = pd.DataFrame(rows)
    out["coefs"] = pd.DataFrame(coefs)
    return out["part2"]


def show2(df, variants):
    for g in sorted(df.group.unique()):
        for thr in sorted(df.thr.unique()):
            s = df[(df.group == g) & (df.thr == thr)]
            if s.empty:
                continue
            print(f"\n  {g}  thr={thr*100:.0f}%")
            print(f"    {'h':>3}{'n':>11}{'dates':>8}{'BSS2 base':>11}{'variant':>14}"
                  f"{'BSS2 var':>10}{'dBSS2':>9}{'95% CI':>20}{'ECE base':>10}{'ECE var':>9}"
                  f"{'dECE':>9}{'yrs+':>7}")
            for h in sorted(s.h.unique()):
                for v in variants:
                    if v == "base":
                        continue
                    r = s[(s.h == h) & (s.variant == v)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    print(f"    {h:>3}{r.n:>11,}{r.n_dates:>8,}{r.bss2_base:>11.4f}{v:>14}"
                          f"{r.bss2_var:>10.4f}{r.dbss2:>+9.4f}"
                          f"{f'[{r.dbss2_lo:+.4f},{r.dbss2_hi:+.4f}]':>20}"
                          f"{r.ece_base:>10.4f}{r.ece_var:>9.4f}{r.dece:>+9.4f}"
                          f"{r.yrs_better:>7}")


# ===================================================================== main
def main(args):
    panel = D.load()
    if CACHE.exists() and not args.rebuild:
        cache = pd.read_pickle(CACHE)
        log(f"loaded cache {CACHE.name}")
    else:
        cache = build(panel)
        pd.to_pickle(cache, CACHE)
    master = cache["_master"]
    cal = trading_calendar(master)
    log("macro date coverage:")
    mac = macro_flags(master)

    # sanity: our cache must equal the shipped build_cache
    if args.verify:
        ref = M.build_cache(panel)
        for k in [("index", 5), ("single", 5)]:
            a, b = cache[k], ref[k]
            assert a["X"].shape == b["X"].shape and np.allclose(a["X"], b["X"], equal_nan=True)
            assert np.allclose(a["lr"], b["lr"], equal_nan=True)
            assert np.allclose(a["tgt"], b["tgt"], equal_nan=True)
        log("VERIFIED: cache identical to move_prob.build_cache")

    FLAGS = {
        "pre_holiday": cal["pre_holiday"].values,
        "post_holiday": cal["post_holiday"].values,
        "tom": cal["tom"].values,
        "quarter_end": cal["quarter_end"].values,
        "triple_witch": cal["triple_witch"].values,
        "opex": cal["opex"].values,
        "dec_quiet": cal["dec_quiet"].values,
        "jan_first": cal["jan_first"].values,
        "fomc": mac["fomc"],
        "fomc_cov": mac["fomc_cov"],
        "cpi": mac["cpi"],
        "cpi_cov": mac["cpi_cov"],
    }
    out = {}
    if not args.skip1:
        part1(cache, master, FLAGS, out)
    if not args.quick:
        variants = {
            "base": None,
            "cal": ("cal",),
            "dow": ("dow",),
            "fomc": ("fomc",),
            "cpi": ("cpi",),
            "all": ("cal", "dow", "fomc", "cpi"),
        }
        df = part2(cache, master, FLAGS, cal, out, variants,
                   groups=tuple(args.groups.split(",")),
                   horizons=tuple(int(x) for x in args.horizons.split(",")))
        show2(df, variants)
        print("\n  fitted extra-column coefficients (full sample, IN-SAMPLE, log-vol units):")
        if not out["coefs"].empty:
            print(out["coefs"].to_string(index=False))
    pd.to_pickle(out, Path(__file__).with_name("_move_fx_macrocal_out.pkl"))
    log("saved _move_fx_macrocal_out.pkl")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--quick", action="store_true", help="descriptive part only")
    ap.add_argument("--skip1", action="store_true", help="skip the descriptive part")
    ap.add_argument("--groups", default="index,single")
    ap.add_argument("--horizons", default="1,5,10,21")
    main(ap.parse_args())
