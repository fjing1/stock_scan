"""
_vix_ma10_bb_research.py — is there a real, tradeable edge in VIX vs its 10-day MA, and in
Bollinger Bands drawn on VIX itself?

The folklore being tested (two separate claims that often get conflated):

  1. "VIX STRETCH" (Larry Connors): when VIX closes some % ABOVE its 10-day moving average, the
     market is short-term oversold — buy stocks. Below its MA10 = complacency = don't. The claim
     is mean-reversion in fear, not a trend signal.

  2. "VIX BOLLINGER BANDS": put BB(n, k) on VIX. A close ABOVE the upper band = panic spike; the
     tradeable trigger is usually stated as the close back INSIDE the band (spike exhausted →
     buy equities). A close BELOW the lower band = complacency → caution. A narrow band
     ("squeeze") = vol about to expand.

Why this repo should not take either on faith: several prior studies here (MTF alignment, RSI-MA
crossover, RSI/OBV divergence, pre-alert escalation) looked good on raw win rate and died once
compared against the unconditional baseline. So every number below is reported as EXCESS over
the same-sample buy-and-hold baseline, and significance uses a rotation test rather than a
t-stat, because forward returns overlap and VIX signals cluster hard inside crises.

Method:
  - ^VIX and ^GSPC daily closes, 1990-01 → today (~9.2k bars; SPY only starts 1993, and the
    index leg needs to cover VIX's whole life). Last bar dropped if today's session is still open.
  - Entry at the NEXT day's close (t+1), not the signal close — VIX prints at 16:15 ET, so a
    same-close fill is not real. Both are computed; t+1 is the one believed.
  - Significance: circular rotation test. Shift the whole signal mask by a random offset 5000x
    and recompute the conditional mean. This preserves BOTH the autocorrelation of returns and
    the clustering of the signal, which a t-test on overlapping windows does not.
  - Stability: four eras (90s, 2000s, 2010s, 2020s). An edge that lives in one crash is not an edge.
  - Walk-forward: threshold picked on data through year Y only, applied in Y+1, chained. Plus a
    deflated Sharpe (Bailey & Lopez de Prado) for the number of configs tried.

Run: ../../vcp_env/bin/python _vix_ma10_bb_research.py
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import yfinance as yf

RNG = np.random.default_rng(20260909)
N_ROT = 5000                      # rotation-test resamples
HORIZONS = [1, 3, 5, 10, 21]
ERAS = [("1990s", 1990, 1999), ("2000s", 2000, 2009),
        ("2010s", 2010, 2019), ("2020s", 2020, 2099)]


# ---------------------------------------------------------------- stats helpers
def _phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _phi_inv(p):
    """Acklam inverse normal CDF (no scipy dependency; same helper as _btc_trend_research.py)."""
    if p <= 0:
        return -math.inf
    if p >= 1:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def deflated_sharpe(sel_pp_sharpe, all_pp_sharpes, n_obs):
    """P(true Sharpe > 0) after deflating for how many configs were tried.
    Bailey & Lopez de Prado (2014), normal-returns simplification."""
    trials = max(len(all_pp_sharpes), 2)
    sd = float(np.std(all_pp_sharpes, ddof=1)) or 1e-9
    emc = 0.5772156649
    e_max = sd * ((1 - emc) * _phi_inv(1 - 1.0 / trials) + emc * _phi_inv(1 - 1.0 / (trials * math.e)))
    denom = math.sqrt(max(n_obs - 1, 1))
    return _phi((sel_pp_sharpe - e_max) * denom)


def rotation_pvalue(mask: np.ndarray, fwd: np.ndarray, observed: float, two_sided=True) -> float:
    """Null: this signal's *shape* (its count and clustering) carries no information about WHEN
    it fires. Rotate the mask circularly by a random offset and recompute. Preserves the
    autocorrelation of `fwd` (overlapping windows) and the burstiness of `mask`, both of which
    make a plain t-test far too generous here."""
    n = len(mask)
    valid = ~np.isnan(fwd)
    null = np.empty(N_ROT)
    offsets = RNG.integers(1, n, size=N_ROT)
    for i, off in enumerate(offsets):
        m = np.roll(mask, off) & valid
        null[i] = fwd[m].mean() if m.sum() else np.nan
    null = null[~np.isnan(null)]
    if not len(null):
        return float("nan")
    if two_sided:
        base = np.nanmean(null)
        return float((np.abs(null - base) >= abs(observed - base)).mean())
    return float((null >= observed).mean())


def _stars(p):
    if np.isnan(p):
        return "   "
    return "***" if p < 0.01 else ("** " if p < 0.05 else ("*  " if p < 0.10 else "   "))


# ---------------------------------------------------------------- data
def load() -> pd.DataFrame:
    def close(t):
        d = yf.download(t, period="max", progress=False, auto_adjust=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        return pd.to_numeric(d["Close"], errors="coerce").dropna()

    vix, spx = close("^VIX"), close("^GSPC")
    df = pd.DataFrame({"vix": vix, "spx": spx}).dropna()
    df.index = pd.to_datetime(df.index)

    # Partial-bar guard. Unclosed bars have silently corrupted three prior analyses in this repo;
    # if the last row is today and US cash equities have not closed yet, drop it.
    now_et = pd.Timestamp.now(tz="America/New_York")
    if df.index[-1].date() == now_et.date() and now_et.time() < pd.Timestamp("16:00").time():
        print(f"⚠️  dropping in-progress bar {df.index[-1].date()} (session still open)")
        df = df.iloc[:-1]
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ma10"] = d.vix.rolling(10).mean()
    d["ema10"] = d.vix.ewm(span=10, adjust=False).mean()
    d["stretch"] = d.vix / d.ma10 - 1.0
    d["stretch_e"] = d.vix / d.ema10 - 1.0
    for n in (10, 20):
        basis = d.vix.rolling(n).mean()
        sd = d.vix.rolling(n).std(ddof=0)
        d[f"bb{n}_basis"] = basis
        d[f"bb{n}_sd"] = sd
        for k in (1.5, 2.0, 2.5):
            tag = f"bb{n}_{k}"
            up, lo = basis + k * sd, basis - k * sd
            d[f"{tag}_up"], d[f"{tag}_lo"] = up, lo
            d[f"{tag}_above"] = d.vix > up
            d[f"{tag}_below"] = d.vix < lo
            # the tradeable trigger: was above the upper band yesterday, back inside today
            d[f"{tag}_reentry"] = d[f"{tag}_above"].shift(1).fillna(False) & ~d[f"{tag}_above"]
            d[f"{tag}_exit_lo"] = d[f"{tag}_below"].shift(1).fillna(False) & ~d[f"{tag}_below"]
        d[f"bb{n}_pctb"] = (d.vix - (basis - 2 * sd)) / (4 * sd)
        d[f"bb{n}_width"] = (4 * sd) / basis          # bandwidth, normalised by the basis

    # forward index returns. t+1 entry is the honest one: VIX settles 16:15 ET, after the cash close.
    for h in HORIZONS:
        d[f"f{h}"] = d.spx.shift(-h) / d.spx - 1.0                 # same-close entry (optimistic)
        d[f"g{h}"] = d.spx.shift(-(h + 1)) / d.spx.shift(-1) - 1.0  # next-close entry (tradeable)
    d["era"] = pd.cut(d.index.year, bins=[0, 1999, 2009, 2019, 9999],
                      labels=[e[0] for e in ERAS])
    return d


# ---------------------------------------------------------------- event study
def event_table(d: pd.DataFrame, rows: list, col_prefix="g", title="", rotation=True):
    print(f"\n{'='*104}\n{title}\n{'='*104}")
    head = f"{'signal':<34}{'n':>6}{'freq':>7}  " + "".join(f"{'D'+str(h):>15}" for h in HORIZONS)
    print(head)
    print(f"{'':<34}{'':>6}{'':>7}  " + "".join(f"{'exc/base  p':>15}" for _ in HORIZONS))
    print("-" * 104)
    out = {}
    for label, mask in rows:
        m = mask.values if isinstance(mask, pd.Series) else mask
        n = int(m.sum())
        if n < 25:
            print(f"{label:<34}{n:>6}  (too few observations, skipped)")
            continue
        cells, rec = [], {}
        for h in HORIZONS:
            fwd = d[f"{col_prefix}{h}"].values
            valid = ~np.isnan(fwd)
            sel = m & valid
            if sel.sum() < 25:
                cells.append(f"{'-':>15}")
                continue
            cond = fwd[sel].mean()
            base = fwd[valid].mean()
            exc = cond - base
            p = rotation_pvalue(m, fwd, cond) if rotation else float("nan")
            rec[h] = {"n": int(sel.sum()), "cond": cond, "base": base, "exc": exc, "p": p,
                      "win": float((fwd[sel] > 0).mean()),
                      "win_base": float((fwd[valid] > 0).mean())}
            cells.append(f"{exc*100:+6.2f}%/{base*100:+5.2f} {_stars(p)}".rjust(15))
        out[label] = rec
        print(f"{label:<34}{n:>6}{n/len(d)*100:>6.1f}%  " + "".join(cells))
    print("\n  exc = conditional mean MINUS unconditional mean over the same sample (this is the")
    print("  only number that matters); base = that unconditional mean. * p<.10 ** p<.05 *** p<.01,")
    print("  circular rotation test, 5000 draws.")
    return out


def era_table(d: pd.DataFrame, label: str, mask: pd.Series, h: int = 5, col_prefix="g"):
    print(f"\n  era stability — {label}, D{h} excess:")
    fwd = d[f"{col_prefix}{h}"]
    ok = ~fwd.isna().values
    parts = []
    for name, y0, y1 in ERAS:
        sub = (d.index.year >= y0) & (d.index.year <= y1)
        m = mask.values & sub & ok
        if m.sum() < 15:
            parts.append(f"{name} n={int(m.sum())} —")
            continue
        b = fwd[sub & ok].mean()
        parts.append(f"{name} n={int(m.sum())} excess {(fwd[m].mean() - b) * 100:+.2f}%")
    print("    " + " | ".join(parts))


# ---------------------------------------------------------------- trading rule
def rule_equity(d: pd.DataFrame, entry: pd.Series, hold: int) -> pd.Series:
    """Long the index for `hold` days after each trigger, cash otherwise. Returns daily strategy
    returns. Entry at t+1 close, so the position starts on the bar after the signal."""
    ret = d.spx.pct_change().shift(-1).fillna(0.0)   # return earned by being long from t close to t+1
    inpos = pd.Series(False, index=d.index)
    idx = np.flatnonzero(entry.values)
    for i in idx:
        inpos.iloc[i + 1: i + 1 + hold] = True
    return ret.where(inpos, 0.0), inpos


def perf(r: pd.Series, exposure: float):
    ann = (1 + r).prod() ** (252 / len(r)) - 1
    vol = r.std() * math.sqrt(252)
    sh = ann / vol if vol else float("nan")
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return {"cagr": ann, "vol": vol, "sharpe": sh, "maxdd": dd, "expo": exposure}


def main():
    d = add_features(load())
    print(f"\nSample: {d.index[0].date()} → {d.index[-1].date()}  ({len(d):,} sessions)")
    print(f"VIX  mean {d.vix.mean():.1f}  median {d.vix.median():.1f}  min {d.vix.min():.1f}  max {d.vix.max():.1f}")
    print(f"SPX  buy-and-hold CAGR {((d.spx.iloc[-1]/d.spx.iloc[0])**(252/len(d))-1)*100:.2f}%")

    # ---------------- PART 1: VIX vs MA10 (the Connors "stretch" claim)
    rows = []
    for thr in (0.05, 0.10, 0.15, 0.20, 0.25):
        rows.append((f"VIX > MA10 +{thr*100:.0f}%", d.stretch >= thr))
    for thr in (-0.05, -0.10, -0.15):
        rows.append((f"VIX < MA10 {thr*100:.0f}%", d.stretch <= thr))
    rows.append(("VIX > MA10 (any)", d.stretch > 0))
    rows.append(("VIX < MA10 (any)", d.stretch < 0))
    r1 = event_table(d, rows, "g", "PART 1 — VIX stretch vs its 10-day SMA  (entry t+1 close)")
    for lbl in ("VIX > MA10 +10%", "VIX > MA10 +20%", "VIX < MA10 (any)"):
        era_table(d, lbl, dict(rows)[lbl])

    # ---------------- PART 2: Bollinger Bands on VIX
    rows = []
    for n in (10, 20):
        for k in (1.5, 2.0, 2.5):
            t = f"bb{n}_{k}"
            rows.append((f"BB({n},{k}) VIX above upper", d[f"{t}_above"]))
            rows.append((f"BB({n},{k}) re-entry from upper", d[f"{t}_reentry"]))
    for n in (10, 20):
        t = f"bb{n}_2.0"
        rows.append((f"BB({n},2.0) VIX below lower", d[f"{t}_below"]))
        rows.append((f"BB({n},2.0) exit up thru lower", d[f"{t}_exit_lo"]))
    r2 = event_table(d, rows, "g", "PART 2 — Bollinger Bands drawn on VIX  (entry t+1 close)")
    for lbl in ("BB(10,2.0) VIX above upper", "BB(10,2.0) re-entry from upper",
                "BB(20,2.0) re-entry from upper", "BB(10,2.0) VIX below lower"):
        if lbl in dict(rows):
            era_table(d, lbl, dict(rows)[lbl])

    # ---------------- PART 3: same-close vs next-close (how much of the edge is unfillable?)
    print(f"\n{'='*104}\nPART 3 — how much of the edge needs a fill you cannot get "
          f"(same-close vs next-close entry)\n{'='*104}")
    print(f"{'signal':<34}{'D5 same-close':>18}{'D5 next-close':>18}{'lost to timing':>18}")
    print("-" * 104)
    for lbl, mask in [("VIX > MA10 +10%", d.stretch >= 0.10),
                      ("VIX > MA10 +20%", d.stretch >= 0.20),
                      ("BB(10,2.0) above upper", d["bb10_2.0_above"]),
                      ("BB(10,2.0) re-entry", d["bb10_2.0_reentry"])]:
        m = mask.values
        cells = []
        for pref in ("f", "g"):
            fwd = d[f"{pref}5"].values
            v = ~np.isnan(fwd)
            cells.append((fwd[m & v].mean() - fwd[v].mean()) * 100)
        print(f"{lbl:<34}{cells[0]:>17.2f}%{cells[1]:>17.2f}%{cells[1]-cells[0]:>17.2f}%")

    # ---------------- PART 4: the squeeze — does a narrow VIX band predict vol expansion?
    print(f"\n{'='*104}\nPART 4 — VIX Bollinger squeeze: does a narrow band precede a vol spike?"
          f"\n{'='*104}")
    w = d["bb10_width"]
    q20 = w.rolling(504).quantile(0.20)
    squeeze = (w <= q20) & w.notna() & q20.notna()
    fwd_vixchg = d.vix.shift(-10) / d.vix - 1.0
    fwd_absret = (d.spx.shift(-10) / d.spx - 1.0).abs()
    for nm, series in (("VIX change over next 10d", fwd_vixchg), ("|SPX move| over next 10d", fwd_absret)):
        v = series.notna().values
        m = squeeze.values & v
        print(f"  {nm:<28} squeeze {series[m].mean()*100:+7.2f}%   "
              f"baseline {series[v].mean()*100:+7.2f}%   "
              f"excess {(series[m].mean()-series[v].mean())*100:+6.2f}%   "
              f"p={rotation_pvalue(squeeze.values, series.values, series[m].mean()):.3f}  n={int(m.sum())}")

    # ---------------- PART 5: walk-forward trading rule
    print(f"\n{'='*104}\nPART 5 — walk-forward: threshold chosen on prior data only, applied out of sample"
          f"\n{'='*104}")
    grid = ([(f"stretch>={t:.2f}", d.stretch >= t) for t in (0.05, 0.10, 0.15, 0.20, 0.25)] +
            [(f"BB({n},{k}) above", d[f"bb{n}_{k}_above"]) for n in (10, 20) for k in (1.5, 2.0, 2.5)] +
            [(f"BB({n},{k}) reentry", d[f"bb{n}_{k}_reentry"]) for n in (10, 20) for k in (1.5, 2.0, 2.5)])
    holds = (3, 5, 10, 21)
    years = sorted({y for y in d.index.year})
    oos = pd.Series(0.0, index=d.index)
    oos_expo = pd.Series(False, index=d.index)
    picks = []
    for y in years:
        if y < years[0] + 8:            # need a warm-up window before selecting anything
            continue
        tr = d.index.year < y
        best, best_s = None, -np.inf
        for lbl, sig in grid:
            for h in holds:
                r, _ = rule_equity(d, sig & tr, h)
                rr = r[tr]
                if rr.abs().sum() == 0:
                    continue
                s = rr.mean() / (rr.std() or 1e-9) * math.sqrt(252)
                if s > best_s:
                    best_s, best = s, (lbl, h, sig)
        if best is None:
            continue
        lbl, h, sig = best
        te = d.index.year == y
        r, inpos = rule_equity(d, sig & te, h)
        oos[te] = r[te]
        oos_expo[te] = inpos[te]
        picks.append((y, lbl, h))
    live = oos.index.year >= years[0] + 8
    o = oos[live]
    bh = d.spx.pct_change().shift(-1).fillna(0.0)[live]
    expo = float(oos_expo[live].mean())
    ps, pb = perf(o, expo), perf(bh, 1.0)
    print(f"  OOS window {o.index[0].date()} → {o.index[-1].date()}   exposure {expo*100:.0f}% of days")
    print(f"  {'':<22}{'CAGR':>10}{'vol':>10}{'Sharpe':>10}{'maxDD':>10}")
    print(f"  {'walk-forward rule':<22}{ps['cagr']*100:>9.2f}%{ps['vol']*100:>9.1f}%{ps['sharpe']:>10.2f}{ps['maxdd']*100:>9.1f}%")
    print(f"  {'buy & hold SPX':<22}{pb['cagr']*100:>9.2f}%{pb['vol']*100:>9.1f}%{pb['sharpe']:>10.2f}{pb['maxdd']*100:>9.1f}%")
    # per-day return while actually in the market — the fair comparison for a part-time rule
    inm = oos_expo[live].values
    bhv = bh.values
    print(f"\n  in-market days only: rule {o[inm].mean()*100:+.4f}%/day vs buy-hold "
          f"{bhv.mean()*100:+.4f}%/day  (rule n={int(inm.sum())})")
    from collections import Counter
    print(f"  configs chosen across {len(picks)} OOS years: "
          f"{Counter((l, h) for _, l, h in picks).most_common(5)}")

    all_sh = []
    for lbl, sig in grid:
        for h in holds:
            r, _ = rule_equity(d, sig, h)
            all_sh.append(r.mean() / (r.std() or 1e-9))
    dsr = deflated_sharpe(o.mean() / (o.std() or 1e-9), all_sh, len(o))
    print(f"  deflated Sharpe P(true Sharpe>0) after {len(all_sh)} trials: {dsr:.3f}")

    # ---------------- PART 6: the honest use case — is it a FILTER, not a trigger?
    print(f"\n{'='*104}\nPART 6 — as a state filter: forward D5 by VIX-vs-MA10 side x band position"
          f"\n{'='*104}")
    above_ma = d.stretch > 0
    hi = d["bb10_2.0_above"]
    lo = d["bb10_2.0_below"]
    mid = ~hi & ~lo
    fwd = d["g5"]
    v = fwd.notna().values
    base = fwd[v].mean()
    print(f"  {'state':<40}{'n':>7}{'mean D5':>11}{'excess':>10}{'win%':>8}")
    for nm, m in [("VIX < MA10, inside bands (calm)", (~above_ma & mid).values),
                  ("VIX > MA10, inside bands (heating)", (above_ma & mid).values),
                  ("VIX above upper band (spike)", hi.values),
                  ("VIX below lower band (complacent)", lo.values)]:
        s = m & v
        print(f"  {nm:<40}{int(s.sum()):>7}{fwd[s].mean()*100:>10.2f}%{(fwd[s].mean()-base)*100:>9.2f}%"
              f"{(fwd[s]>0).mean()*100:>7.1f}%")
    print(f"  {'ALL DAYS (baseline)':<40}{int(v.sum()):>7}{base*100:>10.2f}%{0:>9.2f}%"
          f"{(fwd[v]>0).mean()*100:>7.1f}%")

    # current reading
    last = d.iloc[-1]
    print(f"\n{'='*104}\nCURRENT READING  ({d.index[-1].date()})\n{'='*104}")
    print(f"  VIX {last.vix:.2f} | MA10 {last.ma10:.2f} | stretch {last.stretch*100:+.1f}%")
    print(f"  BB(10,2): {last['bb10_2.0_lo']:.2f} .. {last['bb10_2.0_up']:.2f}  "
          f"%B {last['bb10_pctb']*100:.0f}%  width {last['bb10_width']*100:.0f}% of basis")
    print(f"  BB(20,2): {last['bb20_2.0_lo']:.2f} .. {last['bb20_2.0_up']:.2f}  %B {last['bb20_pctb']*100:.0f}%")


if __name__ == "__main__":
    main()
