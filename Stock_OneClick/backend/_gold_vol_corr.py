"""
_gold_vol_corr.py — the gap in the earlier overlay test: volatility indices and rolling
CORRELATION as signals for the gold sleeve.

The previous sweep (_gold_signal_overlay.py) tested DXY only DIRECTIONALLY (below its 200d, 60d
change < 0) and left out VIX entirely. It never tested the thing gold is actually sold on:
behaviour in volatility/stress regimes, or the correlation structure itself.

Two families here:

  VOL GAUGES   VIX (equity vol), VVIX (vol-of-vol), VIX3M/VIX term structure, GVZ (gold's OWN
               vol index), MOVE (bond vol — arguably the most relevant, gold being a real-rate
               asset), SKEW (tail pricing). Tested as overlay filters on the SMA(500) base.

  CORRELATION  rolling corr(gold, DXY / SPY-book / rates) used two ways:
               (a) as a regime filter, and
               (b) as a SIZING input — the sleeve's whole justification is decorrelation, so
                   sizing up when gold decorrelates and down when it converges is the one
                   theoretically-motivated timing rule in this entire project.
               Also tests whether trailing correlation even PREDICTS forward correlation, since
               a sizing rule built on a non-persistent estimate is noise by construction.

Sample-size warning: VVIX starts 2007, GVZ 2008, VIX3M 2006. Those overlays have ~18-19
independent years, not 26, and correspondingly less power.

Run: ../../vcp_env/bin/python _gold_vol_corr.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import gold_system as G

TD = 252
RNG = np.random.default_rng(8888)


def banded_state(g: pd.Series, n: int = 500, band: float = 0.02) -> pd.Series:
    ma = g.rolling(n).mean()
    st = pd.Series(np.nan, index=g.index)
    st[g > ma * (1 + band)] = 1.0
    st[g < ma * (1 - band)] = 0.0
    return st.ffill().fillna(1.0).astype(bool)


def perf(r, sig, rf_d):
    s = sig.astype(bool)
    c = (1 + r.where(s, rf_d)).cumprod()
    ret = c.pct_change().dropna()
    yrs = len(ret) / TD
    cagr = c.iloc[-1] ** (1 / yrs) - 1
    vol = ret.std(ddof=1) * np.sqrt(TD)
    return {"cagr": cagr * 100, "maxdd": ((c / c.cummax()) - 1).min() * 100,
            "sharpe": (cagr - G.RISK_FREE) / vol, "inmkt": s.mean() * 100,
            "flips": int((s.astype(int).diff().abs() == 1).sum())}


def main():
    p = G.load_panel()
    g = p["GC=F"]["Close"].dropna()
    r = g.pct_change()
    idx = r.dropna().index
    r = r.loc[idx]
    rf_d = (1 + G.RISK_FREE) ** (1 / TD) - 1
    base_sig = banded_state(g).shift(1).reindex(idx).fillna(True)
    base = perf(r, base_sig, rf_d)
    print(f"gold daily {idx[0].date()} -> {idx[-1].date()} ({len(idx)/TD:.0f} yrs)")
    print(f"BASE SMA500+/-2%: CAGR {base['cagr']:.2f}%  MaxDD {base['maxdd']:.1f}%  "
          f"Sharpe {base['sharpe']:.3f}  inMkt {base['inmkt']:.0f}%\n")

    def series(t):
        return p[t]["Close"].reindex(g.index).ffill() if t in p else None

    vix, vvix, vix3m = series("^VIX"), series("^VVIX"), series("^VIX3M")
    gvz, move, skew = series("^GVZ"), series("^MOVE"), series("^SKEW")
    dxy, tnx, spy = series("DX-Y.NYB"), series("^TNX"), series("SPY")

    # ---------------- part 1: vol gauges as overlays ------------------------ #
    print("=== 1. VOLATILITY / STRESS GAUGES as overlay filters on SMA(500) ===")
    cands = {}
    if vix is not None:
        cands["VIX > 20 (stress)"] = vix > 20
        cands["VIX < 20 (calm)"] = vix < 20
        cands["VIX above its 200d"] = vix > vix.rolling(200).mean()
        cands["VIX below its 200d"] = vix < vix.rolling(200).mean()
        cands["VIX > 80th pctile (2y)"] = vix > vix.rolling(500).quantile(0.80)
    if vvix is not None:
        cands["VVIX above its 200d"] = vvix > vvix.rolling(200).mean()
        cands["VVIX > 100"] = vvix > 100
    if vix3m is not None and vix is not None:
        cands["VIX curve inverted (VIX>VIX3M)"] = vix > vix3m
        cands["VIX curve normal (VIX<VIX3M)"] = vix < vix3m
    if gvz is not None:
        cands["GVZ above its 200d (gold vol up)"] = gvz > gvz.rolling(200).mean()
        cands["GVZ below its 200d (gold vol down)"] = gvz < gvz.rolling(200).mean()
    if gvz is not None and vix is not None:
        rel = gvz / vix
        cands["GVZ/VIX above its 200d"] = rel > rel.rolling(200).mean()
    if move is not None:
        cands["MOVE above its 200d (bond vol up)"] = move > move.rolling(200).mean()
        cands["MOVE below its 200d"] = move < move.rolling(200).mean()
    if skew is not None:
        cands["SKEW above its 200d (tail bid)"] = skew > skew.rolling(200).mean()

    rows = []
    print(f"  {'signal':<38}{'CAGR':>7}{'MaxDD':>8}{'Sharpe':>8}{'dSh':>7}{'inMkt':>7}{'N yrs':>7}")
    for name, s in cands.items():
        sv = s.shift(1).reindex(idx)
        valid = sv.notna()
        sig = base_sig & sv.fillna(False).astype(bool)
        m = perf(r.loc[valid], sig.loc[valid], rf_d)
        b2 = perf(r.loc[valid], base_sig.loc[valid], rf_d)   # base on the SAME window
        d = m["sharpe"] - b2["sharpe"]
        rows.append({"name": name, **m, "d": d, "yrs": valid.sum() / TD})
        print(f"  {name:<38}{m['cagr']:>6.2f}%{m['maxdd']:>7.1f}%{m['sharpe']:>8.3f}"
              f"{d:>+7.3f}{m['inmkt']:>6.0f}%{valid.sum()/TD:>7.0f}")
    print("  (dSh compares against the base measured on the SAME shortened window)")

    # ---------------- part 2: does correlation persist? --------------------- #
    print("\n=== 2. DOES TRAILING CORRELATION PREDICT FORWARD CORRELATION? ===")
    print("  (a sizing rule built on a non-persistent estimate is noise by construction)")
    book = None
    if all(x in p for x in ("QQQ", "SMH", "SPY")):
        bpx = pd.DataFrame({t: p[t]["Close"].reindex(g.index).ffill()
                            for t in ("QQQ", "SMH", "SPY")}).pct_change()
        book = bpx["QQQ"] * 0.5 + bpx["SMH"] * 0.3 + bpx["SPY"] * 0.2
    pairs = {"gold~DXY": dxy.pct_change() if dxy is not None else None,
             "gold~book": book,
             "gold~10y yield": tnx.pct_change() if tnx is not None else None}
    for lbl, other in pairs.items():
        if other is None:
            continue
        for win in (126, 252):
            c = r.rolling(win).corr(other.reindex(idx))
            fwd = c.shift(-win)
            ok = c.notna() & fwd.notna()
            if ok.sum() > 100:
                rho = c[ok].corr(fwd[ok])
                print(f"  {lbl:<16} {win:>3}d trailing vs next {win}d:  "
                      f"corr of corrs {rho:+.3f}  (n={int(ok.sum())})")

    # ---------------- part 3: correlation as a filter ---------------------- #
    print("\n=== 3. ROLLING CORRELATION as an overlay filter ===")
    corr_c = {}
    if dxy is not None:
        cd = r.rolling(252).corr(dxy.pct_change().reindex(idx))
        corr_c["corr(gold,DXY) < -0.3 (normal $ regime)"] = cd < -0.3
        corr_c["corr(gold,DXY) > -0.3 (decoupled)"] = cd > -0.3
    if book is not None:
        cb = r.rolling(252).corr(book.reindex(idx))
        corr_c["corr(gold,book) < 0 (diversifying)"] = cb < 0
        corr_c["corr(gold,book) > 0.2 (converging)"] = cb > 0.2
    print(f"  {'signal':<44}{'CAGR':>7}{'MaxDD':>8}{'Sharpe':>8}{'dSh':>7}{'inMkt':>7}")
    for name, s in corr_c.items():
        sv = s.shift(1).reindex(idx)
        valid = sv.notna()
        sig = base_sig & sv.fillna(False).astype(bool)
        m = perf(r.loc[valid], sig.loc[valid], rf_d)
        b2 = perf(r.loc[valid], base_sig.loc[valid], rf_d)
        rows.append({"name": name, **m, "d": m["sharpe"] - b2["sharpe"], "yrs": valid.sum() / TD})
        print(f"  {name:<44}{m['cagr']:>6.2f}%{m['maxdd']:>7.1f}%{m['sharpe']:>8.3f}"
              f"{m['sharpe']-b2['sharpe']:>+7.3f}{m['inmkt']:>6.0f}%")

    # ---------------- part 4: correlation as SIZING ------------------------ #
    print("\n=== 4. CORRELATION-BASED SIZING vs flat 20% (the theoretically motivated rule) ===")
    m_ = G.monthly(p)
    gm = m_["GC=F"].dropna()
    grm = gm.pct_change()
    bpx_m = m_[list(G.EQUITY_PROXY)].dropna()
    eqm = sum(bpx_m[t].pct_change() * w for t, w in G.EQUITY_PROXY.items())
    com = grm.dropna().index.intersection(eqm.dropna().index)
    grm2, eqm2 = grm.loc[com], eqm.loc[com]
    rc = grm2.rolling(36).corr(eqm2)

    flat = G.simulate(grm2, eqm2, 0.20, mode="band", band=0.50)
    print(f"  {'flat 20%':<40} Sharpe {flat['sharpe']:.3f}  CAGR {flat['cagr']:5.2f}%  "
          f"trades {flat['trades']:>2}  tax {flat['tax_pct_final']*100:4.1f}%")
    for lo, hi, lbl in ((0.15, 0.25, "15/25 on corr sign"), (0.10, 0.30, "10/30 on corr sign")):
        tgt = pd.Series(np.where(rc.shift(1).fillna(0) < 0, hi, lo), index=com)
        s = G.simulate(grm2, eqm2, tgt, mode="band", band=0.50)
        print(f"  {lbl:<40} Sharpe {s['sharpe']:.3f}  CAGR {s['cagr']:5.2f}%  "
              f"trades {s['trades']:>2}  tax {s['tax_pct_final']*100:4.1f}%")
    # inverse-correlation continuous sizing
    tgt = (0.20 * (1 - rc.shift(1).fillna(0))).clip(0.10, 0.30)
    s = G.simulate(grm2, eqm2, tgt, mode="band", band=0.50)
    print(f"  {'continuous 20%*(1-corr), clip 10-30%':<40} Sharpe {s['sharpe']:.3f}  "
          f"CAGR {s['cagr']:5.2f}%  trades {s['trades']:>2}  tax {s['tax_pct_final']*100:4.1f}%")

    # ---------------- summary + haircut ------------------------------------ #
    rf_ = pd.DataFrame(rows).sort_values("d", ascending=False)
    print(f"\n=== RANKING ({len(rf_)} candidates) ===")
    print(rf_[["name", "sharpe", "d", "maxdd", "inmkt", "yrs"]].head(6)
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n  beating the base: {int((rf_.d > 0).sum())} of {len(rf_)}   "
          f"by >0.05: {int((rf_.d > 0.05).sum())}")

    top = rf_.iloc[0]
    print(f"\n=== BOOTSTRAP top candidate: {top['name']} ===")
    allc = {**cands, **corr_c}
    sv = allc[top["name"]].shift(1).reindex(idx)
    valid = sv.notna()
    sig = (base_sig & sv.fillna(False).astype(bool)).loc[valid]
    bs = base_sig.loc[valid]
    rr = r.loc[valid]
    yrs = sorted({d.year for d in rr.index})
    diffs = []
    for _ in range(1500):
        pick = RNG.choice(yrs, size=len(yrs), replace=True)
        sel = np.concatenate([np.where(rr.index.year == y)[0] for y in pick])
        a = rr.iloc[sel].where(bs.iloc[sel], rf_d)
        b = rr.iloc[sel].where(sig.iloc[sel], rf_d)
        sa, sb = a.std(ddof=1), b.std(ddof=1)
        if sa > 0 and sb > 0:
            diffs.append((((b.mean() - rf_d) / sb) - ((a.mean() - rf_d) / sa)) * np.sqrt(TD))
    d = np.array(diffs)
    print(f"  advantage: mean {d.mean():+.3f}  SE {d.std(ddof=1):.3f}  "
          f"95% CI [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}]  "
          f"P(worse) {(d<0).mean()*100:.0f}%")

    print("\n=== CURRENT READINGS ===")
    for name, s in {**cands, **corr_c}.items():
        v = s.dropna()
        if len(v):
            print(f"  {name:<44} {'ON' if bool(v.iloc[-1]) else 'off'}")
    for lbl, sr in (("VIX", vix), ("VVIX", vvix), ("GVZ", gvz), ("MOVE", move), ("SKEW", skew)):
        if sr is not None and sr.notna().any():
            print(f"  {lbl:<8} {float(sr.dropna().iloc[-1]):8.2f}   "
                  f"200d avg {float(sr.rolling(200).mean().dropna().iloc[-1]):8.2f}")
    if dxy is not None:
        cd = r.rolling(252).corr(dxy.pct_change().reindex(idx)).dropna()
        print(f"  corr(gold,DXY) 252d  {float(cd.iloc[-1]):+.3f}")
    if book is not None:
        cb = r.rolling(252).corr(book.reindex(idx)).dropna()
        print(f"  corr(gold,book) 252d {float(cb.iloc[-1]):+.3f}")


if __name__ == "__main__":
    main()
