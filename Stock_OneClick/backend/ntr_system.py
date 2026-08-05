"""
ntr_system.py — trading system for Nutrien Ltd (NYSE: NTR), the US listing.

WHY THIS IS BUILT BACKWARDS FROM THE AAPL/GOLD VERSIONS.

NTR only has 8.6 years of price history — Nutrien was created in Jan 2018 by merging PotashCorp
and Agrium, and neither predecessor is retrievable from the data feed. Those 8.6 years contain
essentially ONE fertilizer cycle: a 2018-20 grind, the 2021-22 potash supply shock (peak $100.26
on 2022-04-18), a slide to a -58.2% drawdown bottoming 2024-12-19, and a 2025-26 recovery to ~$67.
Fitting a trend parameter on one cycle is fitting the cycle.

So the parameter is NOT fitted on NTR. It is fitted on the long-history fertilizer/ag complex
(MOS 38y, FMC 46y, CF 21y, ICL 21y, Yara 20y, MOO 19y, plus ADM/AGCO/DE/IPI), pooled, and then
tested OUT-OF-SAMPLE on NTR. That is the honest ordering when the symbol you want to trade is
younger than the effect you are trying to measure. The in-sample-best-on-NTR number is also
reported, purely to show the size of the overfitting gap.

NTR is a COMMODITY CYCLICAL with a ~3.3% dividend, beta ~1.06, vol ~34%. That is a different
animal from AAPL (secular compounder) and gold (monetary asset): cyclicals mean-revert around a
cycle instead of trending secularly, and a -58% drawdown is normal rather than exceptional. Trend
filters have more to work with here than on a stock that only went up.

All series are total-return (auto_adjust=True). A 3.3% yield is a third of NTR's realized CAGR;
using raw closes would understate it badly.

Run: ../../vcp_env/bin/python ntr_system.py --refresh
"""
from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CACHE = Path(__file__).resolve().parent / "_ntr_panel.pkl"
TD = 252
RISK_FREE = 0.0182
LTCG_TAX = 0.20

TARGET = "NTR"
# Long-history fertilizer / ag-input complex. This is the FITTING set — NTR is held out.
FIT_UNIVERSE = ["MOS", "CF", "IPI", "FMC", "ICL", "YARIY", "MOO", "ADM", "AGCO", "DE"]
# Commodity + macro drivers specific to fertilizer economics.
DRIVERS = ["NG=F", "ZC=F", "ZW=F", "DBA", "SPY", "XLB", "DX-Y.NYB", "^VIX"]


def refresh_cache() -> dict:
    import yfinance as yf
    out = {}
    for t in [TARGET] + FIT_UNIVERSE + DRIVERS:
        h = yf.Ticker(t).history(period="max", interval="1d", auto_adjust=True)
        if h.empty:
            print(f"  WARN no data {t}")
            continue
        if getattr(h.index, "tz", None) is not None:
            h.index = h.index.tz_localize(None)
        out[t] = h[["Open", "High", "Low", "Close", "Volume"]]
        print(f"  {t:<9} {h.index[0].date()} -> {h.index[-1].date()}  {len(h)} bars "
              f"({(h.index[-1]-h.index[0]).days/365.25:.1f} yr)")
    with open(CACHE, "wb") as f:
        pickle.dump(out, f)
    return out


def load_panel() -> dict:
    if not CACHE.exists():
        return refresh_cache()
    with open(CACHE, "rb") as f:
        return pickle.load(f)


def banded_state(px: pd.Series, n: int, band: float = 0.02) -> pd.Series:
    """Hysteresis-banded trend state; flips only when price clears the MA by +/-band."""
    ma = px.rolling(n).mean()
    st = pd.Series(np.nan, index=px.index)
    st[px > ma * (1 + band)] = 1.0
    st[px < ma * (1 - band)] = 0.0
    return st.ffill().fillna(1.0).astype(bool)


def perf(r: pd.Series, sig: pd.Series | None = None) -> dict:
    rf_d = (1 + RISK_FREE) ** (1 / TD) - 1
    x = r if sig is None else r.where(sig.astype(bool), rf_d)
    c = (1 + x).cumprod()
    ret = c.pct_change().dropna()
    if len(ret) < TD:
        return {}
    yrs = len(ret) / TD
    cagr = c.iloc[-1] ** (1 / yrs) - 1
    vol = ret.std(ddof=1) * np.sqrt(TD)
    out = {"cagr": cagr * 100, "vol": vol * 100,
           "maxdd": ((c / c.cummax()) - 1).min() * 100,
           "sharpe": (cagr - RISK_FREE) / vol if vol > 0 else np.nan, "years": yrs}
    if sig is not None:
        s = sig.astype(bool)
        out["inmkt"] = s.mean() * 100
        out["flips"] = int((s.astype(int).diff().abs() == 1).sum())
    return out


def bootstrap_diff(r, sig_a, sig_b, n_boot=1500, seed=5):
    """Annual-block paired bootstrap of Sharpe(b) - Sharpe(a)."""
    rng = np.random.default_rng(seed)
    rf_d = (1 + RISK_FREE) ** (1 / TD) - 1
    idx = r.index
    years = sorted({d.year for d in idx})
    out = []
    for _ in range(n_boot):
        pick = rng.choice(years, size=len(years), replace=True)
        sel = np.concatenate([np.where(idx.year == y)[0] for y in pick])
        a = r.iloc[sel].where(sig_a.iloc[sel], rf_d)
        b = r.iloc[sel].where(sig_b.iloc[sel], rf_d)
        sa, sb = a.std(ddof=1), b.std(ddof=1)
        if sa > 0 and sb > 0:
            out.append((((b.mean() - rf_d) / sb) - ((a.mean() - rf_d) / sa)) * np.sqrt(TD))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    panel = refresh_cache() if args.refresh else load_panel()

    ntr = panel[TARGET]["Close"].dropna()
    r_ntr = ntr.pct_change().dropna()
    print(f"\n{TARGET} (NYSE, USD, total-return) {ntr.index[0].date()} -> {ntr.index[-1].date()}"
          f"  {len(ntr)} bars = {len(ntr)/TD:.1f} independent years")
    bh = perf(r_ntr)
    print(f"  buy-and-hold: CAGR {bh['cagr']:.2f}%  vol {bh['vol']:.1f}%  "
          f"MaxDD {bh['maxdd']:.1f}%  Sharpe {bh['sharpe']:.3f}")
    print(f"  ^ NOT a compounder like AAPL. 8.6 yrs, one cycle, Sharpe {bh['sharpe']:.2f}.")

    LENGTHS = list(range(40, 501, 20))

    # ---- 1. fit the length on the LONG-HISTORY complex, NTR held out ------ #
    print(f"\n=== 1. FIT THE MA LENGTH ON THE AG COMPLEX (NTR HELD OUT) ===")
    print(f"  pooling {len(FIT_UNIVERSE)} names; each contributes its own full history")
    pool = {}
    for t in FIT_UNIVERSE:
        if t not in panel:
            continue
        p = panel[t]["Close"].dropna()
        if len(p) < TD * 8:
            continue
        pool[t] = p
    print(f"  usable: {', '.join(f'{t}({len(p)/TD:.0f}y)' for t, p in pool.items())}")

    rows = []
    for n in LENGTHS:
        deltas, cagr_d = [], []
        for t, p in pool.items():
            rr = p.pct_change().dropna()
            s = banded_state(p, n).shift(1).reindex(rr.index).fillna(True)
            a, b = perf(rr), perf(rr, s)
            if a and b:
                deltas.append(b["sharpe"] - a["sharpe"])
                cagr_d.append(b["cagr"] - a["cagr"])
        if deltas:
            d = np.array(deltas)
            rows.append({"n": n, "mean_d": d.mean(), "median_d": np.median(d),
                         "won": int((d > 0).sum()), "of": len(d),
                         "mean_cagr_d": np.mean(cagr_d),
                         "se": d.std(ddof=1) / np.sqrt(len(d))})
    fit = pd.DataFrame(rows)
    print(f"\n  {'n':>5}{'mean dSharpe':>14}{'median':>9}{'won':>7}{'mean dCAGR':>12}{'SE':>7}")
    for _, x in fit.iterrows():
        star = "  <" if x.mean_d == fit.mean_d.max() else ""
        print(f"  {int(x.n):>5}{x.mean_d:>+14.3f}{x.median_d:>+9.3f}"
              f"{int(x.won):>4}/{int(x.of):<3}{x.mean_cagr_d:>+11.2f}%{x.se:>7.3f}{star}")
    bestn = int(fit.loc[fit.mean_d.idxmax(), "n"])
    bestrow = fit.loc[fit.mean_d.idxmax()]
    print(f"\n  POOLED BEST: n={bestn}  mean dSharpe {bestrow.mean_d:+.3f} "
          f"(SE {bestrow.se:.3f}, won {int(bestrow.won)}/{int(bestrow.of)})")
    print(f"  lengths with a POSITIVE pooled mean: "
          f"{int((fit.mean_d > 0).sum())} of {len(fit)}")
    print(f"  lengths where the pooled mean beats CAGR too: "
          f"{int((fit.mean_cagr_d > 0).sum())} of {len(fit)}")

    # ---- 2. out-of-sample test on NTR ------------------------------------- #
    print(f"\n=== 2. OUT-OF-SAMPLE: apply n={bestn} to NTR (never used in the fit) ===")
    s_oos = banded_state(ntr, bestn).shift(1).reindex(r_ntr.index).fillna(True)
    m_oos = perf(r_ntr, s_oos)
    print(f"  buy-and-hold NTR   CAGR {bh['cagr']:>7.2f}%  MaxDD {bh['maxdd']:>7.1f}%  "
          f"Sharpe {bh['sharpe']:>6.3f}")
    print(f"  n={bestn} overlay      CAGR {m_oos['cagr']:>7.2f}%  MaxDD {m_oos['maxdd']:>7.1f}%  "
          f"Sharpe {m_oos['sharpe']:>6.3f}  inMkt {m_oos['inmkt']:.0f}%  flips {m_oos['flips']}")
    print(f"  delta: Sharpe {m_oos['sharpe']-bh['sharpe']:+.3f}   "
          f"CAGR {m_oos['cagr']-bh['cagr']:+.2f}pp   "
          f"drawdown {m_oos['maxdd']-bh['maxdd']:+.1f}pp")
    d = bootstrap_diff(r_ntr, pd.Series(True, index=r_ntr.index), s_oos)
    print(f"  bootstrap vs buy-hold: mean {d.mean():+.3f}  SE {d.std(ddof=1):.3f}  "
          f"95% CI [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}]  "
          f"P(worse) {(d<0).mean()*100:.0f}%")

    # ---- 3. how much would fitting on NTR have flattered us? -------------- #
    print(f"\n=== 3. THE OVERFITTING GAP (what fitting on NTR itself would have claimed) ===")
    insample = []
    for n in LENGTHS:
        s = banded_state(ntr, n).shift(1).reindex(r_ntr.index).fillna(True)
        m = perf(r_ntr, s)
        if m:
            insample.append({"n": n, "sharpe": m["sharpe"], "cagr": m["cagr"],
                             "maxdd": m["maxdd"], "inmkt": m["inmkt"], "flips": m["flips"]})
    ins = pd.DataFrame(insample)
    bi = ins.loc[ins.sharpe.idxmax()]
    print(f"  in-sample best on NTR: n={int(bi.n)}  Sharpe {bi.sharpe:.3f}  CAGR {bi.cagr:.2f}%")
    print(f"  honest OOS choice:     n={bestn}  Sharpe {m_oos['sharpe']:.3f}  CAGR {m_oos['cagr']:.2f}%")
    print(f"  overfitting gap: {bi.sharpe - m_oos['sharpe']:+.3f} Sharpe. "
          f"That gap is what you would have believed and not earned.")
    print(f"  NTR in-sample Sharpe spread across lengths: "
          f"{ins.sharpe.min():.3f} to {ins.sharpe.max():.3f} (range {ins.sharpe.max()-ins.sharpe.min():.3f})")

    # ---- 4. per-name detail at the chosen length -------------------------- #
    print(f"\n=== 4. PER-NAME DETAIL at n={bestn} (the evidence base) ===")
    print(f"  {'ticker':<8}{'yrs':>5}{'BH CAGR':>9}{'ovl CAGR':>10}{'BH Shrp':>9}"
          f"{'ovl Shrp':>10}{'dShrp':>8}{'BH DD':>8}{'ovl DD':>8}")
    for t, p in list(pool.items()) + [(TARGET, ntr)]:
        rr = p.pct_change().dropna()
        s = banded_state(p, bestn).shift(1).reindex(rr.index).fillna(True)
        a, b = perf(rr), perf(rr, s)
        if not a or not b:
            continue
        mark = "  <- OOS target" if t == TARGET else ""
        print(f"  {t:<8}{a['years']:>5.0f}{a['cagr']:>8.2f}%{b['cagr']:>9.2f}%{a['sharpe']:>9.3f}"
              f"{b['sharpe']:>10.3f}{b['sharpe']-a['sharpe']:>+8.3f}{a['maxdd']:>7.1f}%"
              f"{b['maxdd']:>7.1f}%{mark}")

    # ---- 5. current reading ---------------------------------------------- #
    print(f"\n=== 5. CURRENT READING ===")
    now = float(ntr.iloc[-1])
    print(f"  NTR {now:,.2f}  as of {ntr.index[-1].date()}   "
          f"({(now/float(ntr.max())-1)*100:+.1f}% from its {float(ntr.max()):.2f} high)")
    for n in (50, 100, 150, 200, bestn, 300):
        ma = float(ntr.rolling(n).mean().iloc[-1])
        st = bool(banded_state(ntr, n).iloc[-1])
        flip = ma * (0.98 if st else 1.02)
        tag = "  <- system" if n == bestn else ""
        print(f"  {n:>4}d MA {ma:>8,.2f}  {(now/ma-1)*100:+6.1f}%  "
              f"{'UPTREND' if st else 'DOWNTREND':<10} flip {flip:>8,.2f} "
              f"({(flip/now-1)*100:+.1f}%){tag}")


if __name__ == "__main__":
    main()
