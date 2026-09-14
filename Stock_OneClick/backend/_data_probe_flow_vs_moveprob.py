"""Probe: honest marginal test against move_prob's REAL baseline.

The previous probe used a crude 4-feature HAR proxy as the baseline, so its deltas were
measured against something weaker than the shipped model. move_prob.build_features()
already contains HAR(rv_d/w/m/q, ewma97) + LEV(sv_dn/sv_up/r5) + VOL(dvol/vsurp/amihud),
and move_prob's docstring records that an overnight/intraday variance split was already
measured and REJECTED -- which is close to the |overnight gap| feature that won the
previous probe.

So this script asks the only question that is still open:

  Over move_prob's ACTUAL feature set, does anything in the free order-flow-proxy family
  still add OOS R2?

  (a) FINRA short-volume ratio block           -- external data, 2018-08 onward
  (b) |overnight gap|                          -- expected ~0, it is the rejected feature
  (c) Corwin-Schultz + Roll effective SPREAD   -- NEW to this repo, cost nothing
"""
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/feijing/github.com/stock_scan/Stock_OneClick/backend")
import move_prob as MP

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)


def sec(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


panel = pickle.load(open("_move_panel.pkl", "rb"))
C, H, L, V, O = panel["Close"], panel["High"], panel["Low"], panel["Volume"], panel["Open"]
V = V.where(V > 0)
LIQ = [s for s in C.columns if V[s].median() and V[s].median() > 1e6]
print(f"move_prob feature set: {MP.FEATS_HAR + MP.FEATS_LEV + MP.FEATS_VOL}")
print(f"liquid subset: {len(LIQ)} symbols")

fin = pd.read_pickle("_data_probe_finra_hist.pkl")
to_p = fin.pivot(index="date", columns="symbol", values="total")
sr_p = fin.pivot(index="date", columns="symbol", values="short")
ex_p = fin.pivot(index="date", columns="symbol", values="exempt")

# ---- build move_prob's real features per symbol
BASE_COLS = MP.FEATS_HAR + MP.FEATS_LEV + MP.FEATS_VOL
bf = {}
for s in LIQ:
    f = MP.build_features(C[s], H[s], L[s], vix=None, volume=V[s])
    bf[s] = f[BASE_COLS]
base = {c: pd.DataFrame({s: bf[s][c] for s in LIQ}) for c in BASE_COLS}
print(f"built move_prob features for {len(LIQ)} symbols; cols={len(BASE_COLS)}")

ret = np.log(C).diff()
absr = ret.abs()
rng = (H - L).where(lambda x: x > 0)

# ---- (b) overnight gap
gap = np.log(O / C.shift(1))
# ---- (c) spread estimators
Ht = np.maximum(H, C.shift(1))
Lt = np.minimum(L, C.shift(1))
b1 = np.log(Ht / Lt) ** 2
beta_ = b1 + b1.shift(-1)
gamma_ = np.log(np.maximum(Ht, Ht.shift(-1)) / np.minimum(Lt, Lt.shift(-1))) ** 2
alpha = (np.sqrt(2 * beta_) - np.sqrt(beta_)) / (3 - 2 * np.sqrt(2)) \
    - np.sqrt(gamma_ / (3 - 2 * np.sqrt(2)))
cs_spr = (2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))).clip(lower=0).shift(1)
roll_spr = 2 * np.sqrt((-ret.rolling(21, min_periods=15).cov(ret.shift(1))).clip(lower=0))
# ---- (a) FINRA
sr = (sr_p / to_p).where(lambda x: np.isfinite(x))
exr = (ex_p / to_p).where(lambda x: np.isfinite(x))


def z20(x):
    return (x - x.rolling(20, min_periods=15).mean()) / x.rolling(20, min_periods=15).std()


def wins(x, lo=.005, hi=.995):
    a = x.stack()
    return x.clip(a.quantile(lo), a.quantile(hi))


BLOCKS = {
    "b_gap": {"agap": wins(gap.abs()), "gap": wins(gap)},
    "c_spread": {"lcs": wins(np.log(cs_spr.clip(lower=1e-6))),
                 "lroll": wins(np.log(roll_spr.clip(lower=1e-6))),
                 "dcs": wins(np.log(cs_spr.clip(lower=1e-6))
                             - np.log(cs_spr.clip(lower=1e-6)).rolling(20, min_periods=15).mean())},
    "a_finra": {"sr": wins(sr), "sr_z": wins(z20(sr)), "d_sr": wins(sr.diff()),
                "exr": wins(exr)},
}
tgts = {h: np.log(absr.rolling(h, min_periods=max(1, h - 1)).mean().shift(-h).clip(lower=1e-5))
        for h in (1, 5, 10)}


def stk(x):
    s = x.stack()
    s.index = s.index.set_names(["d", "s"])
    return s


def build(cols, target, keep, lo, hi):
    d = pd.DataFrame({k: stk(v.reindex(columns=keep).loc[lo:hi])
                      for k, v in {"y": target, **cols}.items()}).dropna()
    return d[np.isfinite(d.values).all(axis=1)]


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
        bb, *_ = np.linalg.lstsq(Xtr, tr.y.values, rcond=None)
        P.append(Xte @ bb)
        A.append(te.y.values)
    p, a = np.concatenate(P), np.concatenate(A)
    return 1 - ((a - p) ** 2).sum() / ((a - a.mean()) ** 2).sum(), len(a)


W0, W1 = pd.Timestamp("2018-08-01"), pd.Timestamp("2026-09-11")
sec("1. FINRA WINDOW (2018-08-01 .. 2026-09-11) vs move_prob's REAL feature set")
allc = {**base}
for b in BLOCKS.values():
    allc.update(b)
rows = []
for h in (1, 5, 10):
    d = build(allc, tgts[h], LIQ, W0, W1)
    r0, n = wf(d, BASE_COLS)
    row = {"h": h, "n_oos": n, "R2_move_prob": r0}
    for bn, bc in BLOCKS.items():
        r, _ = wf(d, BASE_COLS + list(bc))
        row[f"d_{bn}"] = r - r0
    r_all, _ = wf(d, BASE_COLS + [c for b in BLOCKS.values() for c in b])
    row["d_everything"] = r_all - r0
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

sec("2. FULL PANEL (2001-09 .. 2026-09) -- no FINRA, it does not reach back")
rows = []
noext = {**base, **BLOCKS["b_gap"], **BLOCKS["c_spread"]}
for h in (1, 5, 10):
    d = build(noext, tgts[h], LIQ, C.index.min(), W1)
    r0, n = wf(d, BASE_COLS)
    rg, _ = wf(d, BASE_COLS + list(BLOCKS["b_gap"]))
    rs, _ = wf(d, BASE_COLS + list(BLOCKS["c_spread"]))
    rb, _ = wf(d, BASE_COLS + list(BLOCKS["b_gap"]) + list(BLOCKS["c_spread"]))
    rows.append({"h": h, "n_oos": n, "R2_move_prob": r0, "d_gap": rg - r0,
                 "d_spread": rs - r0, "d_both": rb - r0})
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

sec("3. PER-YEAR STABILITY OF THE SPREAD BLOCK (h=5, full panel)")
d = build(noext, tgts[5], LIQ, C.index.min(), W1)
dates = d.index.get_level_values(0)
out = []
for yr in sorted(set(dates.year)):
    tr = d[dates < pd.Timestamp(f"{yr}-01-01")]
    te = d[(dates >= pd.Timestamp(f"{yr}-01-01")) & (dates < pd.Timestamp(f"{yr+1}-01-01"))]
    if len(tr) < 20000 or len(te) < 500:
        continue
    r2 = {}
    for tag, cols in (("base", BASE_COLS),
                      ("spread", BASE_COLS + list(BLOCKS["c_spread"]))):
        Xtr = np.column_stack([np.ones(len(tr))] + [tr[c].values for c in cols])
        Xte = np.column_stack([np.ones(len(te))] + [te[c].values for c in cols])
        bb, *_ = np.linalg.lstsq(Xtr, tr.y.values, rcond=None)
        p, a = Xte @ bb, te.y.values
        r2[tag] = 1 - ((a - p) ** 2).sum() / ((a - a.mean()) ** 2).sum()
    out.append((yr, len(te), r2["base"], r2["spread"], r2["spread"] - r2["base"]))
o = pd.DataFrame(out, columns=["year", "n", "r2_base", "r2_spread", "delta"])
print(o.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
print(f"\npositive years: {(o.delta>0).sum()}/{len(o)}  mean={o.delta.mean():+.5f} "
      f"min={o.delta.min():+.5f} max={o.delta.max():+.5f}")
