"""
_gold_ma_daily.py — the same "which MA length?" question, on DAILY bars.

Why redo it daily:
- Finer grid. A monthly MA of N months is roughly a daily MA of N*21 days, so the monthly sweep
  (3-36 months) was really a coarse sample of the 63-756 day range with big gaps between rungs.
  18 months ~= 378 days; the famous 200-day MA ~= 9.5 months.
- It removes a hidden arbitrary choice. A monthly system samples ONE day per month (the last).
  That choice was never justified either, and test 5 below shows how much it matters.

What daily does NOT buy: more independent information. 26 years is still ~26 independent annual
observations no matter how finely you slice it. A finer grid mostly buys more chances to overfit,
which is why the bootstrap and walk-forward tests matter more here, not less.

Decision cadence is kept SEPARATE from signal resolution — that is the real design question for a
1yr+ holder, since daily decisions mean daily whipsaw and a 28% collectibles tax on every exit.

Run: ../../vcp_env/bin/python _gold_ma_daily.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import gold_system as G

RNG = np.random.default_rng(999)
DAILY_LENGTHS = list(range(40, 521, 10))       # ~2 months to ~2 years
TRADING_DAYS = 252


def load_daily():
    panel = G.load_panel()
    g = panel["GC=F"]["Close"].dropna()
    # drop an in-progress final bar; see partial-bar notes in gold_system.monthly()
    return g


def ann_daily(curve: pd.Series, rf: float = G.RISK_FREE):
    ret = curve.pct_change().dropna()
    yrs = len(ret) / TRADING_DAYS
    cagr = curve.iloc[-1] ** (1 / yrs) - 1
    vol = ret.std(ddof=1) * np.sqrt(TRADING_DAYS)
    mdd = ((curve / curve.cummax()) - 1).min()
    return cagr, vol, mdd, ((cagr - rf) / vol if vol > 0 else np.nan)


def overlay(ret: pd.Series, sig: pd.Series, rf_d: float) -> pd.Series:
    return (1 + ret.where(sig, rf_d)).cumprod()


def main():
    g = load_daily()
    r = g.pct_change()
    idx = r.dropna().index
    r = r.loc[idx]
    rf_d = (1 + G.RISK_FREE) ** (1 / TRADING_DAYS) - 1
    print(f"gold daily {idx[0].date()} -> {idx[-1].date()}  "
          f"({len(idx)} bars, {len(idx)/TRADING_DAYS:.0f} independent years)\n")

    bh = (1 + r).cumprod()
    bc, bv, bd, bs = ann_daily(bh)
    print(f"buy-and-hold gold: CAGR {bc*100:.2f}%  vol {bv*100:.1f}%  "
          f"MaxDD {bd*100:.1f}%  Sharpe {bs:.3f}\n")

    # ---- 1. daily sweep, decide DAILY -------------------------------------- #
    print("=== 1. DAILY MA SWEEP, decision every day (cash at T-bill) ===")
    sigs = {}
    rows = []
    for n in DAILY_LENGTHS:
        s = (g > g.rolling(n).mean()).shift(1).reindex(idx).fillna(False)
        sigs[n] = s
        c, v, d, sh = ann_daily(overlay(r, s, rf_d))
        flips = int((s.astype(int).diff() != 0).sum())
        rows.append({"days": n, "~months": round(n / 21, 1), "cagr": c * 100,
                     "maxdd": d * 100, "sharpe": sh, "inmkt": s.mean() * 100,
                     "flips": flips, "flips_per_yr": flips / (len(idx) / TRADING_DAYS)})
    sw = pd.DataFrame(rows)
    print(sw.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    best = sw.loc[sw.sharpe.idxmax()]
    print(f"\n  best Sharpe: {int(best.days)}d (~{best['~months']}mo) at {best.sharpe:.3f}")
    print(f"  median across lengths {sw.sharpe.median():.3f}   worst {sw.sharpe.min():.3f}")
    print(f"  lengths beating buy-and-hold Sharpe ({bs:.3f}): "
          f"{int((sw.sharpe>bs).sum())} of {len(sw)}")
    for ref, lbl in ((200, "200d (the classic)"), (378, "378d = the 18-month MA"),
                     (252, "252d = 12 months")):
        row = sw.iloc[(sw.days - ref).abs().argmin()]
        rank = int((sw.sharpe > row.sharpe).sum()) + 1
        print(f"  {lbl:<24} -> {int(row.days)}d  Sharpe {row.sharpe:.3f}  "
              f"rank {rank}/{len(sw)}  flips/yr {row.flips_per_yr:.1f}")

    # ---- 2. is the spread real? -------------------------------------------- #
    print("\n=== 2. BLOCK BOOTSTRAP (annual blocks): best vs median length ===")
    med_days = int(sw.loc[(sw.sharpe - sw.sharpe.median()).abs().idxmin(), "days"])
    yrs = sorted({d.year for d in idx})
    diffs = []
    for _ in range(1500):
        pick = RNG.choice(yrs, size=len(yrs), replace=True)
        sel = np.concatenate([np.where(idx.year == y)[0] for y in pick])
        a = r.iloc[sel].where(sigs[int(best.days)].iloc[sel], rf_d)
        b = r.iloc[sel].where(sigs[med_days].iloc[sel], rf_d)
        sa, sb = a.std(ddof=1), b.std(ddof=1)
        if sa > 0 and sb > 0:
            diffs.append(((a.mean() - rf_d) / sa - (b.mean() - rf_d) / sb) * np.sqrt(TRADING_DAYS))
    diffs = np.array(diffs)
    print(f"  best {int(best.days)}d vs median {med_days}d: mean diff {diffs.mean():+.3f}  "
          f"SE {diffs.std(ddof=1):.3f}  95% CI [{np.percentile(diffs,2.5):+.3f}, "
          f"{np.percentile(diffs,97.5):+.3f}]")
    print(f"  P(the 'best' length is actually worse) = {(diffs<0).mean()*100:.1f}%")

    # ---- 3. walk-forward selection ----------------------------------------- #
    print("\n=== 3. WALK-FORWARD: which length would you have picked, prior data only? ===")
    picks = []
    for i in range(TRADING_DAYS * 10, len(idx), TRADING_DAYS):
        hist = idx[:i]
        sc = {}
        for n in DAILY_LENGTHS:
            x = r.loc[hist].where(sigs[n].loc[hist], rf_d)
            sd = x.std(ddof=1)
            if sd > 0:
                sc[n] = (x.mean() - rf_d) / sd
        if sc:
            picks.append({"date": idx[i].date(), "best": max(sc, key=sc.get)})
    pk = pd.DataFrame(picks)
    print(f"  re-selections: {len(pk)}   choices: {pk.best.tolist()}")
    print(f"  distinct {pk.best.nunique()}  range {pk.best.min()}-{pk.best.max()}d  "
          f"std {pk.best.std():.0f}d")

    # ---- 4. decision cadence: the part that actually costs money ----------- #
    print("\n=== 4. SIGNAL DAILY, but DECIDE how often? (200d MA, 20% sleeve in the book) ===")
    m = G.monthly(G.load_panel())
    px = m[list(G.EQUITY_PROXY)].dropna()
    eq_m = sum(px[t].pct_change() * w for t, w in G.EQUITY_PROXY.items())
    gm = m["GC=F"].dropna()
    gr_m = gm.pct_change()
    common = gr_m.dropna().index.intersection(eq_m.dropna().index)

    sig200 = (g > g.rolling(200).mean())
    for lbl, s in (("raw daily signal", sig200),
                   ("+10d confirmation", sig200.rolling(10).min().astype(bool)),
                   ("+21d confirmation", sig200.rolling(21).min().astype(bool)),
                   ("+63d confirmation", sig200.rolling(63).min().astype(bool))):
        flips = int((s.astype(int).diff().abs() == 1).sum())
        # sample the daily reading at month end -> monthly decision cadence
        s_m = s.reindex(gm.index, method="ffill").shift(1).reindex(common).fillna(False)
        flips_m = int((s_m.astype(int).diff().abs() == 1).sum())
        tgt = pd.Series(np.where(s_m, 0.20, 0.0), index=common)
        res = G.simulate(gr_m.loc[common], eq_m.loc[common], tgt, mode="band", band=0.50)
        print(f"  {lbl:<20} daily flips {flips:>4}  monthly-sampled flips {flips_m:>3}  "
              f"Sharpe {res['sharpe']:.3f}  trades {res['trades']:>2}  "
              f"tax/final {res['tax_pct_final']*100:4.1f}%")
    flat = G.simulate(gr_m.loc[common], eq_m.loc[common], 0.20, mode="band", band=0.50)
    print(f"  {'flat 20%, no signal':<20} {'':>28}  Sharpe {flat['sharpe']:.3f}  "
          f"trades {flat['trades']:>2}  tax/final {flat['tax_pct_final']*100:4.1f}%")

    # ---- 5. the hidden arbitrary choice: WHICH DAY of the month? ----------- #
    print("\n=== 5. HOW MUCH DID 'SAMPLE AT MONTH END' MATTER? (18-month MA, by sample day) ===")
    res = []
    for off in range(0, 21):
        # shift the sampling point through the month
        samp = g.iloc[off::21]
        s = (samp > samp.rolling(18).mean()).shift(1)
        s_d = s.reindex(idx, method="ffill").fillna(False)
        c, v, d, sh = ann_daily(overlay(r, s_d, rf_d))
        res.append({"offset_days": off, "cagr": c * 100, "maxdd": d * 100, "sharpe": sh})
    rr = pd.DataFrame(res)
    print(f"  Sharpe across the 21 possible sample days: min {rr.sharpe.min():.3f}  "
          f"median {rr.sharpe.median():.3f}  max {rr.sharpe.max():.3f}  "
          f"spread {rr.sharpe.max()-rr.sharpe.min():.3f}")
    print(f"  CAGR spread: {rr.cagr.min():.2f}% to {rr.cagr.max():.2f}%")
    print("  -> a real signal should not care which day you look. Compare this spread to the")
    print("     spread across MA LENGTHS above; if similar, length choice is the same kind of noise.")

    # ---- 6. ensemble across daily lengths ---------------------------------- #
    print("\n=== 6. ENSEMBLE VOTE across all daily lengths (no length to choose) ===")
    votes = pd.DataFrame({n: sigs[n] for n in DAILY_LENGTHS}).astype(float)
    frac = votes.mean(axis=1)
    for thr in (0.3, 0.5, 0.7):
        s = frac >= thr
        c, v, d, sh = ann_daily(overlay(r, s, rf_d))
        flips = int((s.astype(int).diff().abs() == 1).sum())
        print(f"  vote >= {thr*100:3.0f}%:  CAGR {c*100:5.2f}%  MaxDD {d*100:6.1f}%  "
              f"Sharpe {sh:.3f}  flips {flips} ({flips/(len(idx)/TRADING_DAYS):.1f}/yr)")

    # ---- 7. current reading ------------------------------------------------ #
    print("\n=== 7. CURRENT READING ON DAILY BARS ===")
    px_now = float(g.iloc[-1])
    print(f"  gold {px_now:,.2f}  as of {g.index[-1].date()}")
    for n in (50, 100, 150, 200, 250, 300, 378, 450, 500):
        ma = float(g.rolling(n).mean().iloc[-1])
        print(f"  {n:>4}d MA (~{n/21:4.1f}mo) {ma:>9,.2f}  {(px_now/ma-1)*100:+6.1f}%  "
              f"{'ABOVE' if px_now > ma else 'BELOW'}")
    up = [n for n in DAILY_LENGTHS if bool(sigs[n].iloc[-1])]
    print(f"  ensemble vote: {len(up)}/{len(DAILY_LENGTHS)} = {len(up)/len(DAILY_LENGTHS)*100:.0f}% say uptrend")
    if up:
        print(f"  crossover point: lengths >= {min(up)}d say uptrend, shorter say downtrend")


if __name__ == "__main__":
    main()
