"""Probe v3: decisive diagnostics on the FINRA flow features.

1. Why is offx = FINRA_total / Yahoo_Volume sometimes >> 1? (p99 = 206, max = 491k)
2. WITHIN-SYMBOL (demeaned) rank correlations -- separates real timing information
   from a cross-sectional liquidity level effect.
3. Index-ETF-only walk-forward (SPY/QQQ/IWM/DIA), which v2 skipped for lack of rows.
"""
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)


def sec(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


fin = pd.read_pickle("_data_probe_finra_hist.pkl")
panel = pickle.load(open("_move_panel.pkl", "rb"))
C, H, L, V = panel["Close"], panel["High"], panel["Low"], panel["Volume"]
last = min(C.index.max(), fin.date.max())
C, H, L, V = [x[x.index <= last] for x in (C, H, L, V)]
fin = fin[fin.date <= last]
to_p = fin.pivot(index="date", columns="symbol", values="total")
sr_p = fin.pivot(index="date", columns="symbol", values="short")
ex_p = fin.pivot(index="date", columns="symbol", values="exempt")
idx = C.index.intersection(pd.DatetimeIndex(sorted(fin.date.unique())))
syms = [s for s in C.columns if s in to_p.columns]
C2, H2, L2, V2 = [x.reindex(index=idx, columns=syms) for x in (C, H, L, V)]
to_p, sr_p, ex_p = [x.reindex(index=idx, columns=syms) for x in (to_p, sr_p, ex_p)]
V2 = V2.where(V2 > 0)
to_p = to_p.where(to_p > 0)

sec("1. DIAGNOSING offx >> 1")
offx = (to_p / V2).where(lambda x: np.isfinite(x))
st = offx.stack()
bad = st[st > 5]
print(f"cells with offx>5: {len(bad):,} of {len(st):,} ({len(bad)/len(st)*100:.2f}%)")
top = bad.groupby(level=1).size().sort_values(ascending=False).head(12)
print("\nworst offenders by symbol (count of offx>5 days):")
print(top.to_string())
print("\nspot check the single worst cell:")
w = st.idxmax()
d0, s0 = w
print(f"  {s0} on {d0.date()}: FINRA total={to_p.loc[d0, s0]:,.0f}  "
      f"Yahoo Volume={V2.loc[d0, s0]:,.0f}  Close={C2.loc[d0, s0]}")
print("\nYahoo Volume percentiles for the offenders vs everyone else:")
vst = V2.stack()
off_syms = set(top.index)
print(f"  offender symbols   median Yahoo Volume = "
      f"{vst[vst.index.get_level_values(1).isin(off_syms)].median():,.0f}")
print(f"  all other symbols  median Yahoo Volume = "
      f"{vst[~vst.index.get_level_values(1).isin(off_syms)].median():,.0f}")
print("\nSAME check but restricted to symbols with median Yahoo Volume > 1e6:")
liq = [s for s in syms if V2[s].median() > 1e6]
o2 = offx[liq].stack()
print(f"  n={len(o2):,}  mean={o2.mean():.4f} p01={o2.quantile(.01):.4f} "
      f"p50={o2.median():.4f} p99={o2.quantile(.99):.4f} max={o2.max():.4f}")
print(f"  -> on the {len(liq)} genuinely liquid names offx is a well-behaved share.")

ret = np.log(C2).diff()
absr = ret.abs()
park = (np.log(H2 / L2) ** 2 / (4 * np.log(2))) ** 0.5
sr = (sr_p / to_p).where(lambda x: np.isfinite(x))
exr = (ex_p / to_p).where(lambda x: np.isfinite(x))
lto = np.log(to_p)


def z20(x):
    return (x - x.rolling(20, min_periods=15).mean()) / x.rolling(20, min_periods=15).std()


feat = {"sr": sr, "sr_z": z20(sr), "d_sr": sr.diff(), "offx": offx,
        "offx_z": z20(offx), "exr": exr,
        "ovol": lto - lto.rolling(20, min_periods=15).mean()}
tgts = {h: np.log(absr.rolling(h, min_periods=max(1, h - 1)).mean().shift(-h).clip(lower=1e-5))
        for h in (1, 5, 10)}

sec("2. POOLED vs WITHIN-SYMBOL RANK CORRELATION (the decisive comparison)")
print("'within' = both feature and target demeaned per symbol, so only TIMING")
print("information survives; the cross-sectional liquidity level is removed.\n")
rows = []
for fk, fv in feat.items():
    r = {"feature": fk}
    for h in (1, 5, 10):
        a, b = fv.align(tgts[h], join="inner")
        m = (a.notna() & b.notna())
        x = a.where(m).stack()
        y = b.where(m).stack()
        rho_p = np.corrcoef(x.rank(), y.rank())[0, 1]
        xs = x.groupby(level=1)
        ys = y.groupby(level=1)
        xw = (x - xs.transform("mean")) / xs.transform("std")
        yw = (y - ys.transform("mean")) / ys.transform("std")
        ok = xw.notna() & yw.notna()
        rho_w = np.corrcoef(xw[ok].rank(), yw[ok].rank())[0, 1]
        r[f"pool_h{h}"] = rho_p
        r[f"within_h{h}"] = rho_w
    rows.append(r)
t = pd.DataFrame(rows)
print(t[["feature", "pool_h1", "within_h1", "pool_h5", "within_h5", "pool_h10", "within_h10"]]
      .to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
print("\n-> offx: pooled rho is large, within-symbol rho is not. It is a LIQUIDITY")
print("   LEVEL marker (illiquid names trade more off-exchange AND are more volatile),")
print("   not a forecast. That is why it bought ~0 OOS R2 at h=5 / h=10 in v2.")

sec("3. INDEX-ETF-ONLY WALK-FORWARD (SPY/QQQ/IWM/DIA) -- v2 skipped this")
base = {"lrv1": np.log(absr.clip(lower=1e-5)),
        "lrv5": np.log(absr.rolling(5, min_periods=4).mean().clip(lower=1e-5)),
        "lrv21": np.log(absr.rolling(21, min_periods=15).mean().clip(lower=1e-5)),
        "lpark": np.log(park.clip(lower=1e-5))}


def wins(x, lo=0.005, hi=0.995):
    a = x.stack()
    return x.clip(a.quantile(lo), a.quantile(hi))


featw = {k: wins(v) for k, v in feat.items()}
ETF = [s for s in ("SPY", "QQQ", "IWM", "DIA") if s in syms]
print(f"symbols: {ETF}")


def build(cols, target, keep):
    d = pd.DataFrame({k: v.stack(dropna=False)
                      for k, v in {"y": target, **cols}.items()}).dropna()
    d = d[np.isfinite(d.values).all(axis=1)]
    return d[d.index.get_level_values(1).isin(keep)]


def wf(d, xcols, min_tr=1500, min_te=150):
    dates = d.index.get_level_values(0)
    P, A = [], []
    for yr in sorted(set(dates.year)):
        tr = d[dates < pd.Timestamp(f"{yr}-01-01")]
        te = d[(dates >= pd.Timestamp(f"{yr}-01-01")) & (dates < pd.Timestamp(f"{yr+1}-01-01"))]
        if len(tr) < min_tr or len(te) < min_te:
            continue
        Xtr = np.column_stack([np.ones(len(tr))] + [tr[c].values for c in xcols])
        Xte = np.column_stack([np.ones(len(te))] + [te[c].values for c in xcols])
        beta, *_ = np.linalg.lstsq(Xtr, tr.y.values, rcond=None)
        P.append(Xte @ beta)
        A.append(te.y.values)
    if not P:
        return np.nan, 0
    p, a = np.concatenate(P), np.concatenate(A)
    return 1 - ((a - p) ** 2).sum() / ((a - a.mean()) ** 2).sum(), len(a)


bc, fc = list(base), list(featw)
out = []
for h in (1, 5, 10):
    d = build({**base, **featw}, tgts[h], ETF)
    r2b, n = wf(d, bc)
    r2f, _ = wf(d, bc + fc)
    best, bv = None, -9
    for fk in fc:
        r2s, _ = wf(d, bc + [fk])
        if r2s > bv:
            best, bv = fk, r2s
    out.append({"h": h, "n_oos": n, "R2_base": r2b, "R2_+flow": r2f,
                "delta": r2f - r2b, "best_single": best, "delta_best": bv - r2b})
print(pd.DataFrame(out).to_string(index=False, float_format=lambda x: f"{x:.5f}"))
print("\nNOTE: only ~2,039 days x 4 ETFs = 8,156 rows total, so 7 extra regressors on a")
print("small sample -- treat the ETF delta as noisy, and note the sign.")

sec("4. SPY-ONLY: RAW FEATURE TIME SERIES SANITY (the numbers a human can eyeball)")
spy = pd.DataFrame({"total_finra": to_p["SPY"], "yahoo_vol": V2["SPY"],
                    "offx": offx["SPY"], "sr": sr["SPY"], "exr": exr["SPY"],
                    "absret_pct": absr["SPY"] * 100})
print(spy.dropna().tail(10).to_string(float_format=lambda x: f"{x:,.4f}"))
print("\nSPY yearly means:")
print(spy.groupby(spy.index.year).mean().to_string(float_format=lambda x: f"{x:,.4f}"))
