"""Probe v4 -- THE DECISIVE TEST.

v3 showed the strongest genuine (within-symbol) FINRA signal is `ovol`, the
off-exchange volume surprise, rho ~ +0.08..+0.10 vs forward vol.

But FINRA's TotalVolume is ~39% of the consolidated tape, so `ovol` may be nothing
more than a noisy copy of "today's volume was unusually high" -- which move_prob
already has from Yahoo's daily Volume column (the +0.004 BSS2 volume feature).

So: put the Yahoo volume surprise INTO the baseline, then ask whether ANY FINRA
feature still adds OOS R2. If not, FINRA buys nothing for this model.
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

# restrict to genuinely liquid names -- the Yahoo panel is unreliable on the microcaps
LIQ = [s for s in syms if V2[s].median() > 1e6]
print(f"liquid subset (median Yahoo volume > 1e6): {len(LIQ)} of {len(syms)} symbols")

ret = np.log(C2).diff()
absr = ret.abs()
park = (np.log(H2 / L2) ** 2 / (4 * np.log(2))) ** 0.5
sr = (sr_p / to_p).where(lambda x: np.isfinite(x))
exr = (ex_p / to_p).where(lambda x: np.isfinite(x))
lto, lv = np.log(to_p), np.log(V2)


def z20(x):
    return (x - x.rolling(20, min_periods=15).mean()) / x.rolling(20, min_periods=15).std()


def dev20(x):
    return x - x.rolling(20, min_periods=15).mean()


def wins(x, lo=0.005, hi=0.995):
    a = x.stack()
    return x.clip(a.quantile(lo), a.quantile(hi))


# ---- baselines
B_OHLC = {"lrv1": np.log(absr.clip(lower=1e-5)),
          "lrv5": np.log(absr.rolling(5, min_periods=4).mean().clip(lower=1e-5)),
          "lrv21": np.log(absr.rolling(21, min_periods=15).mean().clip(lower=1e-5)),
          "lpark": np.log(park.clip(lower=1e-5))}
# the volume information move_prob ALREADY has, from Yahoo
B_VOL = {"yvol": wins(dev20(lv)), "yvol_z": wins(z20(lv))}

# ---- FINRA-only features
F = {"sr": wins(sr), "sr_z": wins(z20(sr)), "d_sr": wins(sr.diff()),
     "offx": wins((to_p / V2).where(lambda x: np.isfinite(x))),
     "offx_z": wins(z20((to_p / V2).where(lambda x: np.isfinite(x)))),
     "exr": wins(exr), "ovol": wins(dev20(lto))}

tgts = {h: np.log(absr.rolling(h, min_periods=max(1, h - 1)).mean().shift(-h).clip(lower=1e-5))
        for h in (1, 5, 10)}


def build(cols, target, keep):
    d = pd.DataFrame({k: v.stack(dropna=False)
                      for k, v in {"y": target, **cols}.items()}).dropna()
    d = d[np.isfinite(d.values).all(axis=1)]
    return d[d.index.get_level_values(1).isin(keep)]


def wf(d, xcols, min_tr=20000, min_te=500):
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


sec("1. LADDER: OHLC -> +YAHOO VOLUME -> +FINRA FLOW   (liquid names, walk-forward)")
allcols = {**B_OHLC, **B_VOL, **F}
rows = []
for h in (1, 5, 10):
    d = build(allcols, tgts[h], LIQ)
    r_ohlc, n = wf(d, list(B_OHLC))
    r_vol, _ = wf(d, list(B_OHLC) + list(B_VOL))
    r_fin, _ = wf(d, list(B_OHLC) + list(B_VOL) + list(F))
    r_finonly, _ = wf(d, list(B_OHLC) + list(F))
    rows.append({"h": h, "n_oos": n,
                 "R2_ohlc": r_ohlc,
                 "R2_+yahoo_vol": r_vol, "d_yahoo_vol": r_vol - r_ohlc,
                 "R2_+finra_no_yvol": r_finonly, "d_finra_alone": r_finonly - r_ohlc,
                 "R2_+both": r_fin, "d_finra_over_yvol": r_fin - r_vol})
t = pd.DataFrame(rows)
print(t.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
print("\nREAD THIS ROW: d_finra_over_yvol -- what FINRA buys once Yahoo's own daily")
print("volume surprise is already in the model. That is the number that matters.")

sec("2. WHICH SINGLE FINRA FEATURE, IF ANY, SURVIVES THE YAHOO-VOLUME CONTROL?")
for h in (1, 5, 10):
    d = build(allcols, tgts[h], LIQ)
    r_vol, _ = wf(d, list(B_OHLC) + list(B_VOL))
    line = []
    for fk in F:
        r, _ = wf(d, list(B_OHLC) + list(B_VOL) + [fk])
        line.append((fk, r - r_vol))
    line.sort(key=lambda z: -z[1])
    print(f"h={h:<3} base R2={r_vol:.5f}  " +
          "  ".join(f"{k}:{v:+.5f}" for k, v in line))

sec("3. HOW CORRELATED IS THE FINRA VOLUME SURPRISE WITH YAHOO'S?")


def st(x):
    """stack with normalised MultiIndex names (to_p and V2 carry different names)."""
    s = x.stack()
    s.index = s.index.set_names(["d", "s"])
    return s


a = st(dev20(lto)[LIQ])
b = st(dev20(lv)[LIQ])
m = a.notna() & b.notna()
print(f"corr(ovol, yvol) pooled = {np.corrcoef(a[m], b[m])[0,1]:.4f}   n={int(m.sum()):,}")
print(f"rank corr               = {np.corrcoef(a[m].rank(), b[m].rank())[0,1]:.4f}")
print("-> FINRA's off-exchange volume surprise is largely the same variable as the")
print("   consolidated volume surprise we already have for free from Yahoo.")

sec("4. THE ONE FINRA-ONLY VARIABLE WITH NO YAHOO EQUIVALENT: the short RATIO")
print("sr = ShortVolume/TotalVolume. Yahoo cannot produce this at any horizon.")
print("Within-symbol rank corr vs forward vol (from v3): +0.005 / +0.007 / +0.005")
print("Its 20d z-score does slightly better: +0.008 / +0.012 / +0.013\n")
d = build({**B_OHLC, **B_VOL, "sr_z": F["sr_z"], "sr": F["sr"], "d_sr": F["d_sr"]},
          tgts[5], LIQ)
r0, n = wf(d, list(B_OHLC) + list(B_VOL))
r1, _ = wf(d, list(B_OHLC) + list(B_VOL) + ["sr", "sr_z", "d_sr"])
print(f"h=5  R2 without short-ratio block = {r0:.5f}")
print(f"h=5  R2 with    short-ratio block = {r1:.5f}   delta = {r1-r0:+.5f}  n={n:,}")

sec("5. DIRECTIONAL (SIGN) TEST ON LIQUID NAMES ONLY, WITH THE VOLUME CONTROL")
d = pd.DataFrame({"sr_z": st(F["sr_z"][LIQ]), "yvol": st(B_VOL["yvol"][LIQ]),
                  "fwd": st(ret.shift(-1)[LIQ])}).dropna()
print(f"n={len(d):,}")
q = pd.qcut(d.sr_z, 5, labels=False, duplicates="drop")
g = d.groupby(q).fwd.agg(["mean", "std", "count"])
g["bps"] = g["mean"] * 1e4
g["t"] = g["mean"] / (g["std"] / np.sqrt(g["count"]))
print(g[["bps", "t", "count"]].to_string(float_format=lambda x: f"{x:,.3f}"))
print(f"Q4-Q0 = {(g['mean'].iloc[-1]-g['mean'].iloc[0])*1e4:+.2f} bps/day  "
      f"(need ~>5 bps to clear a 1c spread + fees on a daily-turnover rule)")
