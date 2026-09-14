"""Probe: order-flow-like information ALREADY derivable from the daily OHLCV panel.

The point of this probe is honesty about the counterfactual: before paying (in effort or
money) for order flow, measure what the free daily bars already encode.

Quantities built here:
  clv      close location value (2C-H-L)/(H-L) in [-1,1] -- signed intraday pressure
  sgnvol   clv * log volume                              -- signed volume (OBV kernel)
  gap      log(O/C_prev)                                 -- overnight order imbalance
  olv      open location value (O-L)/(H-L)
  amihud   |ret| / dollar volume                         -- price impact per dollar
  cs_spr   CORWIN-SCHULTZ (2012) high-low effective SPREAD estimator
  roll_spr ROLL (1984) serial-covariance effective spread estimator

cs_spr and roll_spr are the interesting ones: they are microstructure quantities --
effective bid-ask spread, i.e. the cost of demanding liquidity -- recovered from daily
bars alone. Order-flow vendors sell the same thing.

Evaluation: same expanding walk-forward OOS R2 ladder used for the FINRA probe, on the
SAME liquid subset and SAME date window, so the numbers are directly comparable.
"""
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)


def sec(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


panel = pickle.load(open("_move_panel.pkl", "rb"))
C, H, L, V = panel["Close"], panel["High"], panel["Low"], panel["Volume"]
O = panel.get("Open")
last = C.index.max()
C, H, L, V, O = [x[x.index <= last] for x in (C, H, L, V, O)]
V = V.where(V > 0)
rng = (H - L).where(lambda x: x > 0)

LIQ = [s for s in C.columns if V[s].median() and V[s].median() > 1e6]
# match the FINRA probe's window exactly so the deltas are comparable
W0, W1 = pd.Timestamp("2018-08-01"), pd.Timestamp("2026-09-11")
print(f"panel: {C.index.min().date()} .. {C.index.max().date()}  cols={len(C.columns)}")
print(f"liquid subset: {len(LIQ)} symbols;  comparison window {W0.date()} .. {W1.date()}")

ret = np.log(C).diff()
absr = ret.abs()
park = (np.log(H / L) ** 2 / (4 * np.log(2))) ** 0.5

clv = (2 * C - H - L) / rng
olv = (O - L) / rng
gap = np.log(O / C.shift(1))
sgnvol = clv * np.log(V)
amihud = absr / (C * V)

# ---- Corwin-Schultz (2012) two-day high-low spread estimator
Ht = np.maximum(H, C.shift(1))          # overnight-adjusted, 1-day
Lt = np.minimum(L, C.shift(1))
b1 = np.log(Ht / Lt) ** 2
H2 = pd.concat([Ht, Ht.shift(-0)], axis=0)  # placeholder, real 2-day below
H2d = np.maximum(Ht, Ht.shift(-1))
L2d = np.minimum(Lt, Lt.shift(-1))
beta = b1 + b1.shift(-1)
gamma = np.log(H2d / L2d) ** 2
k1 = np.sqrt(2) - 1
alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) \
    - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
cs_spr = (2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha)))
cs_spr = cs_spr.clip(lower=0)           # CS: negative estimates set to zero
cs_spr = cs_spr.shift(1)                # value for day t uses days t and t+1 -> lag it

# ---- Roll (1984) spread: 2*sqrt(-cov(r_t, r_{t-1})) over a rolling window
cov21 = ret.rolling(21, min_periods=15).cov(ret.shift(1))
roll_spr = 2 * np.sqrt((-cov21).clip(lower=0))


def dev20(x):
    return x - x.rolling(20, min_periods=15).mean()


def z20(x):
    return (x - x.rolling(20, min_periods=15).mean()) / x.rolling(20, min_periods=15).std()


def wins(x, lo=0.005, hi=0.995):
    a = x.stack()
    return x.clip(a.quantile(lo), a.quantile(hi))


B_OHLC = {"lrv1": np.log(absr.clip(lower=1e-5)),
          "lrv5": np.log(absr.rolling(5, min_periods=4).mean().clip(lower=1e-5)),
          "lrv21": np.log(absr.rolling(21, min_periods=15).mean().clip(lower=1e-5)),
          "lpark": np.log(park.clip(lower=1e-5))}
B_VOL = {"yvol": wins(dev20(np.log(V))), "yvol_z": wins(z20(np.log(V)))}
FLOWISH = {"clv": wins(clv), "aclv": wins(clv.abs()), "sgnvol": wins(sgnvol),
           "gap": wins(gap), "agap": wins(gap.abs()), "olv": wins(olv),
           "lamihud": wins(np.log(amihud.clip(lower=1e-14))),
           "lcs_spr": wins(np.log(cs_spr.clip(lower=1e-6))),
           "lroll_spr": wins(np.log(roll_spr.clip(lower=1e-6)))}
tgts = {h: np.log(absr.rolling(h, min_periods=max(1, h - 1)).mean().shift(-h).clip(lower=1e-5))
        for h in (1, 5, 10)}

sec("1. LEVELS OF THE MICROSTRUCTURE ESTIMATORS (sanity: are these plausible spreads?)")
for nm, x in [("cs_spr", cs_spr), ("roll_spr", roll_spr)]:
    a = x[LIQ].loc[W0:W1].stack()
    print(f"{nm:9s} n={len(a):>8,}  median={a.median()*1e4:>8.2f} bps  "
          f"p05={a.quantile(.05)*1e4:>7.2f}  p95={a.quantile(.95)*1e4:>8.2f}  "
          f"zeros={float((a==0).mean()):.3f}")
print("\nSPY vs a mid-cap, median estimated spread (bps) -- should differ a lot:")
for s in ["SPY", "QQQ", "AAPL", "NVDA", "FUBO", "ROOT"]:
    if s in C.columns:
        cs = cs_spr[s].loc[W0:W1].median() * 1e4
        rl = roll_spr[s].loc[W0:W1].median() * 1e4
        print(f"  {s:6s} corwin-schultz={cs:>8.2f} bps   roll={rl:>8.2f} bps")

sec("2. WITHIN-SYMBOL RANK CORRELATION vs FORWARD VOL (comparable to the FINRA table)")


def stk(x):
    s = x.stack()
    s.index = s.index.set_names(["d", "s"])
    return s


rows = []
for fk, fv in FLOWISH.items():
    r = {"feature": fk}
    for h in (1, 5, 10):
        x = stk(fv[LIQ].loc[W0:W1])
        y = stk(tgts[h][LIQ].loc[W0:W1])
        d = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
        gx = d.groupby(level=1)
        xw = (d.x - gx.x.transform("mean")) / gx.x.transform("std")
        yw = (d.y - gx.y.transform("mean")) / gx.y.transform("std")
        ok = xw.notna() & yw.notna()
        r[f"within_h{h}"] = np.corrcoef(xw[ok].rank(), yw[ok].rank())[0, 1]
    rows.append(r)
tt = pd.DataFrame(rows)
print(tt.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
print("\nFINRA's best, same metric (from _data_probe_finra_signal3.py):")
print("  ovol +0.0817 / +0.0963 / +0.0754     exr +0.0483 / +0.0752 / +0.0768")
print("  sr_z +0.0084 / +0.0117 / +0.0128")


def build(cols, target, keep, lo, hi):
    d = pd.DataFrame({k: stk(v[keep].loc[lo:hi]) for k, v in {"y": target, **cols}.items()}).dropna()
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
        beta_, *_ = np.linalg.lstsq(Xtr, tr.y.values, rcond=None)
        P.append(Xte @ beta_)
        A.append(te.y.values)
    if not P:
        return np.nan, 0
    p, a = np.concatenate(P), np.concatenate(A)
    return 1 - ((a - p) ** 2).sum() / ((a - a.mean()) ** 2).sum(), len(a)


sec("3. OOS R2 LADDER ON THE FINRA WINDOW (apples-to-apples with the FINRA result)")
allc = {**B_OHLC, **B_VOL, **FLOWISH}
rows = []
for h in (1, 5, 10):
    d = build(allc, tgts[h], LIQ, W0, W1)
    r0, n = wf(d, list(B_OHLC) + list(B_VOL))
    r1, _ = wf(d, list(B_OHLC) + list(B_VOL) + list(FLOWISH))
    best, bv = None, -9
    for fk in FLOWISH:
        rr, _ = wf(d, list(B_OHLC) + list(B_VOL) + [fk])
        if rr > bv:
            best, bv = fk, rr
    rows.append({"h": h, "n_oos": n, "R2_ohlc+vol": r0, "R2_+derived_flow": r1,
                 "delta_all": r1 - r0, "best_single": best, "delta_best": bv - r0})
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.5f}"))
print("\nFINRA on the identical window/subset bought: +0.00007 / +0.00005 / -0.00012")

sec("4. AND ON THE FULL 25-YEAR PANEL (derived features need no external data)")
rows = []
for h in (1, 5, 10):
    d = build(allc, tgts[h], LIQ, C.index.min(), W1)
    r0, n = wf(d, list(B_OHLC) + list(B_VOL))
    r1, _ = wf(d, list(B_OHLC) + list(B_VOL) + list(FLOWISH))
    rows.append({"h": h, "n_oos": n, "R2_ohlc+vol": r0,
                 "R2_+derived_flow": r1, "delta": r1 - r0})
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.5f}"))
print("\n-> these features cost ZERO new data and cover the WHOLE panel history,")
print("   unlike FINRA which only starts 2018-08-01 (32% of the panel).")
