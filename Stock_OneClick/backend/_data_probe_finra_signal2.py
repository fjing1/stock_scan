"""Probe v2: CLEANED FINRA flow-feature feasibility test.

v1 bug: offx = FINRA_total / Yahoo_Volume produced +inf where Yahoo Volume was 0/NaN
(halted or thinly-quoted names). Those infinities ranked at the top of every
cross-section and inflated the reported Spearman correlations (+0.31 at h=10).
This version drops non-finite feature values BEFORE any statistic is computed, and
winsorizes at the 0.5/99.5 pct.

Also adds:
 - index-ETF vs single-name split (the split move_prob cares about)
 - a mechanism check on ShortExemptVolume (is it just "stock fell >10% yesterday"?)
 - per-year OOS R2 delta so the +0.001 is not one lucky year
"""
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

INDEX_LIKE = {"SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "XLK", "XLF", "XLE", "XLV",
              "XLI", "XLP", "XLU", "XLY", "XLB", "XLC", "XLRE", "SMH", "GLD", "TLT",
              "EEM", "EFA", "ARKK", "SOXX", "IBIT", "GDX"}


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

sr_p = fin.pivot(index="date", columns="symbol", values="short")
to_p = fin.pivot(index="date", columns="symbol", values="total")
ex_p = fin.pivot(index="date", columns="symbol", values="exempt")
idx = C.index.intersection(pd.DatetimeIndex(sorted(fin.date.unique())))
syms = [s for s in C.columns if s in to_p.columns]
C2, H2, L2, V2 = [x.reindex(index=idx, columns=syms) for x in (C, H, L, V)]
sr_p, to_p, ex_p = [x.reindex(index=idx, columns=syms) for x in (sr_p, to_p, ex_p)]

# ---- HARD CLEAN: require strictly positive volumes on both sides
V2 = V2.where(V2 > 0)
to_p = to_p.where(to_p > 0)
ret = np.log(C2).diff()
absr = ret.abs()
park = (np.log(H2 / L2) ** 2 / (4 * np.log(2))) ** 0.5


def wins(x, lo=0.005, hi=0.995):
    a = x.stack()
    ql, qh = a.quantile(lo), a.quantile(hi)
    return x.clip(ql, qh)


def z20(x):
    return (x - x.rolling(20, min_periods=15).mean()) / x.rolling(20, min_periods=15).std()


sr = (sr_p / to_p).where(lambda x: np.isfinite(x))
offx = (to_p / V2).where(lambda x: np.isfinite(x))
exr = (ex_p / to_p).where(lambda x: np.isfinite(x))
lto = np.log(to_p)

print(f"aligned {idx.min().date()} .. {idx.max().date()}  days={len(idx):,}  symbols={len(syms)}")
sec("0. HOW MANY CELLS DID THE v1 BUG AFFECT?")
raw_offx = to_p / V2.replace(np.nan, np.nan)
n_all = to_p.notna().sum().sum()
n_bad = int((~np.isfinite(to_p / panel["Volume"].reindex(index=idx, columns=syms))).sum().sum()
            - (to_p / panel["Volume"].reindex(index=idx, columns=syms)).isna().sum().sum())
print(f"FINRA cells: {n_all:,}")
print(f"offx cells that were non-finite (Yahoo Volume == 0): {n_bad:,}")
print(f"offx cells surviving the clean: {int(offx.notna().sum().sum()):,}")

sec("1. CLEANED FEATURE DISTRIBUTIONS (all years pooled)")
for nm, x in [("sr", sr), ("offx", offx), ("exr", exr)]:
    a = x.stack()
    print(f"{nm:5s} n={len(a):>8,} mean={a.mean():.4f} std={a.std():.4f} "
          f"p01={a.quantile(.01):.4f} p50={a.median():.4f} p99={a.quantile(.99):.4f} "
          f"max={a.max():.4f}")
print("\noff-exchange share of the consolidated tape, by year (CLEANED):")
st = offx.stack()
print(st.groupby(st.index.get_level_values(0).year).agg(
    ["mean", "median", "std", "count"]).to_string(float_format=lambda x: f"{x:,.4f}"))

feat = {"sr": wins(sr), "sr_z": wins(z20(sr)), "d_sr": wins(sr.diff()),
        "offx": wins(offx), "offx_z": wins(z20(offx)), "exr": wins(exr),
        "ovol": wins(lto - lto.rolling(20, min_periods=15).mean())}
base = {"lrv1": np.log(absr.clip(lower=1e-5)),
        "lrv5": np.log(absr.rolling(5, min_periods=4).mean().clip(lower=1e-5)),
        "lrv21": np.log(absr.rolling(21, min_periods=15).mean().clip(lower=1e-5)),
        "lpark": np.log(park.clip(lower=1e-5))}
tgts = {}
for h in (1, 5, 10):
    tgts[h] = np.log(absr.rolling(h, min_periods=max(1, h - 1)).mean().shift(-h).clip(lower=1e-5))
ret1 = ret.shift(-1)

sec("2. CLEANED SPEARMAN CORRELATIONS (compare with the contaminated v1 numbers)")
V1 = {"sr": (0.0078, 0.0088, 0.0081), "sr_z": (0.0071, 0.0078, 0.0077),
      "d_sr": (0.0030, 0.0017, 0.0015), "offx": (0.1680, 0.2823, 0.3105),
      "offx_z": (0.0050, 0.0095, 0.0089), "exr": (0.0889, 0.1275, 0.1295),
      "ovol": (0.0641, 0.0568, 0.0357)}
rows = []
for fk, fv in feat.items():
    r = {"feature": fk}
    for i, h in enumerate((1, 5, 10)):
        a, b = fv.align(tgts[h], join="inner")
        m = (a.notna() & b.notna()).values
        x, y = a.values[m], b.values[m]
        rho = np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1]
        r[f"h{h}"] = rho
        r[f"h{h}_v1"] = V1[fk][i]
    rows.append(r)
print(pd.DataFrame(rows)[["feature", "h1", "h1_v1", "h5", "h5_v1", "h10", "h10_v1"]]
      .to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
print("\n-> offx's h10 rho collapses from +0.3105 to the value above once the")
print("   zero-Yahoo-volume cells are removed. v1's headline was an artifact.")


def build(cols, target, keep=None):
    parts = {"y": target}
    parts.update(cols)
    d = pd.DataFrame({k: v.stack(dropna=False) for k, v in parts.items()}).dropna()
    d = d[np.isfinite(d.values).all(axis=1)]
    if keep is not None:
        d = d[d.index.get_level_values(1).isin(keep)]
    return d


def wf(d, xcols, by_year=False):
    dates = d.index.get_level_values(0)
    out = []
    P, A = [], []
    for yr in sorted(set(dates.year)):
        tr = d[dates < pd.Timestamp(f"{yr}-01-01")]
        te = d[(dates >= pd.Timestamp(f"{yr}-01-01")) & (dates < pd.Timestamp(f"{yr+1}-01-01"))]
        if len(tr) < 20000 or len(te) < 500:
            continue
        Xtr = np.column_stack([np.ones(len(tr))] + [tr[c].values for c in xcols])
        Xte = np.column_stack([np.ones(len(te))] + [te[c].values for c in xcols])
        beta, *_ = np.linalg.lstsq(Xtr, tr.y.values, rcond=None)
        p, a = Xte @ beta, te.y.values
        P.append(p)
        A.append(a)
        out.append((yr, 1 - ((a - p) ** 2).sum() / ((a - a.mean()) ** 2).sum(), len(a)))
    p, a = np.concatenate(P), np.concatenate(A)
    tot = 1 - ((a - p) ** 2).sum() / ((a - a.mean()) ** 2).sum()
    return (tot, len(a), pd.DataFrame(out, columns=["year", "r2", "n"])) if by_year else (tot, len(a))


bc, fc = list(base), list(feat)
sec("3. INCREMENTAL OOS R2, ALL NAMES / INDEX ETFs / SINGLE NAMES")
groups = {"all": None,
          "index_etf": [s for s in syms if s in INDEX_LIKE],
          "single": [s for s in syms if s not in INDEX_LIKE]}
print(f"group sizes: all={len(syms)}  index_etf={len(groups['index_etf'])} "
      f"({sorted(groups['index_etf'])})  single={len(groups['single'])}\n")
res = []
for gname, keep in groups.items():
    for h in (1, 5, 10):
        d = build({**base, **feat}, tgts[h], keep)
        if len(d) < 30000:
            continue
        r2b, n = wf(d, bc)
        r2f, _ = wf(d, bc + fc)
        res.append({"group": gname, "h": h, "n_oos": n, "R2_base": r2b,
                    "R2_+flow": r2f, "delta": r2f - r2b})
print(pd.DataFrame(res).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

sec("4. IS THE h=1 DELTA STABLE ACROSS YEARS, OR ONE LUCKY YEAR?")
d = build({**base, **feat}, tgts[1])
_, _, yb = wf(d, bc, by_year=True)
_, _, yf = wf(d, bc + fc, by_year=True)
cmp_ = yb.merge(yf, on="year", suffixes=("_base", "_flow"))
cmp_["delta"] = cmp_.r2_flow - cmp_.r2_base
print(cmp_[["year", "n_base", "r2_base", "r2_flow", "delta"]]
      .to_string(index=False, float_format=lambda x: f"{x:.5f}"))
print(f"\nyears with positive delta: {(cmp_.delta > 0).sum()}/{len(cmp_)}   "
      f"mean delta={cmp_.delta.mean():+.5f}  min={cmp_.delta.min():+.5f} "
      f"max={cmp_.delta.max():+.5f}")

sec("5. MECHANISM: is ShortExemptVolume just 'the stock fell hard yesterday'? (SSR)")
e = exr.stack().rename("exr").to_frame()
e["ret_t"] = ret.stack()
e["ret_tm1"] = ret.shift(1).stack()
e = e.dropna()
e["ssr_flag"] = (e.exr > 0.01).astype(int)
print(f"n={len(e):,}  share of days with exr>1%: {e.ssr_flag.mean():.4f}")
g = e.groupby("ssr_flag")[["ret_t", "ret_tm1"]].agg(["mean", "std", "count"])
print((g * 100).to_string(float_format=lambda x: f"{x:,.3f}"))
print("\nprev-day return decile -> mean exr (Reg SHO circuit-breaker mechanism):")
q = pd.qcut(e.ret_tm1, 10, labels=False, duplicates="drop")
print(e.groupby(q).agg(mean_exr=("exr", "mean"), mean_ret_tm1=("ret_tm1", "mean"),
                       n=("exr", "size")).to_string(float_format=lambda x: f"{x:,.5f}"))
print("\n-> if exr spikes only in the bottom decile of yesterday's return, it is a")
print("   restatement of a big down day, which the OHLC baseline already sees.")
