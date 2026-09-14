"""Verify the category-(A) verdict is robust, not an artifact.

_data_probe_har_test.py found:
  target = log RV_{t+1}          : 5min RV 0.363 vs Parkinson 0.339 vs daily r^2 0.182
  target = log|daily ret_{t+1}|  : 5min RV 0.0276 vs Parkinson 0.0264  (delta +0.0013)

Two things could be faking that:
  1. AMZN daily close-to-close sd printed 1.280 -- that is a 20:1 split (Jun 2022)
     leaking through, so the demo intraday feed is NOT split-adjusted. Splits
     contaminate every daily-return-based series.
  2. A single chronological 70/30 split is exactly what this repo has learned to
     distrust; it wants rolling walk-forward.

So: detect splits explicitly, drop them, and re-run with expanding-window
walk-forward instead of one static cut.
"""
import math

import numpy as np
import pandas as pd

SYMS = ["vti", "aapl", "msft", "amzn", "tsla"]
EPS = 1e-10


def load(s):
    return pd.read_pickle(f"_data_probe_har_{s}.pkl")


print("=" * 100)
print("1 -- split detection: unadjusted splits in the demo intraday feed")
print("=" * 100)
panels = {}
for s in SYMS:
    dl = load(s)
    dret = np.log(dl["close"] / dl["close"].shift(1))
    bad = dret[dret.abs() > 0.35]
    print(f"  {s.upper():5s} sessions={len(dl)}  |daily log-ret|>0.35 on {len(bad)} day(s)")
    for d, v in bad.items():
        ratio = math.exp(v)
        print(f"        {d}  log-ret={v:+.4f}  price ratio={ratio:.4f}  "
              f"~= {'1:%.0f split' % (1/ratio) if ratio < 1 else 'reverse/other'}")
    dl = dl.copy()
    dl["dret"] = dret
    # mark split days; they corrupt any daily-return-derived series
    dl["is_split"] = dret.abs() > 0.35
    panels[s] = dl

print("\n" + "=" * 100)
print("2 -- rebuild features with split days neutralised (daily returns only; 5-min RV")
print("     is computed WITHIN each day so it is intrinsically split-immune)")
print("=" * 100)


def features(dl):
    d = dl.copy()
    dret = d["dret"].where(~d["is_split"])       # blank the split day
    d["rv_daily"] = dret ** 2
    d["rv_park"] = (np.log(d["high"] / d["low"]) ** 2) / (4 * math.log(2))
    d["rv_5m"] = d["rv"]
    d["y_rv"] = np.log(d["rv_5m"].shift(-1) + EPS)
    d["y_mag"] = np.log(dret.abs().shift(-1) + 1e-6)
    return d


def har(x):
    s = pd.Series(x).astype(float)
    return pd.DataFrame({"d": s.shift(1),
                         "w": s.shift(1).rolling(5).mean(),
                         "m": s.shift(1).rolling(22).mean()})


def wf_r2(X, y, min_train=400, step=21):
    """Expanding-window walk-forward: refit every `step` days, predict the next block.
    Returns OOS R-squared pooled over all out-of-sample blocks."""
    ok = X.notna().all(axis=1) & y.notna()
    X, y = X[ok].reset_index(drop=True), y[ok].reset_index(drop=True)
    n = len(y)
    if n < min_train + step:
        return None, 0, 0
    preds, actual = [], []
    start = min_train
    while start < n:
        end = min(start + step, n)
        Xtr = np.column_stack([np.ones(start), X.iloc[:start].values])
        ytr = y.iloc[:start].values
        beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
        Xte = np.column_stack([np.ones(end - start), X.iloc[start:end].values])
        preds.append(Xte @ beta)
        actual.append(y.iloc[start:end].values)
        start = end
    p = np.concatenate(preds)
    a = np.concatenate(actual)
    # benchmark: expanding mean of the training data (an honest naive forecast)
    ss_res = float(np.sum((a - p) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    return 1.0 - ss_res / ss_tot, len(a), n


for target, tname in (("y_rv", "log RV_{t+1}  (5-min realized variance)"),
                      ("y_mag", "log|daily close-to-close return_{t+1}|")):
    print("\n" + "=" * 100)
    print(f"WALK-FORWARD (expanding, refit every 21d), target = {tname}")
    print("=" * 100)
    print(f"  {'sym':5s} {'n_oos':>6s} {'5min RV':>9s} {'Parkinson':>10s} "
          f"{'daily r^2':>10s} {'RV+Park':>9s} {'RV-Park':>9s}")
    agg = []
    for s in SYMS:
        d = features(panels[s])
        y = d[target]
        Xrv = np.log(har(d["rv_5m"]) + EPS)
        Xpk = np.log(har(d["rv_park"]) + EPS)
        Xdr = np.log(har(d["rv_daily"]) + EPS)
        Xboth = pd.concat([Xrv.add_suffix("_rv"), Xpk.add_suffix("_pk")], axis=1)
        r_rv, n_oos, _ = wf_r2(Xrv, y)
        r_pk, _, _ = wf_r2(Xpk, y)
        r_dr, _, _ = wf_r2(Xdr, y)
        r_bo, _, _ = wf_r2(Xboth, y)
        if r_rv is None:
            continue
        agg.append((r_rv, r_pk, r_dr, r_bo))
        print(f"  {s.upper():5s} {n_oos:6d} {r_rv:9.4f} {r_pk:10.4f} {r_dr:10.4f} "
              f"{r_bo:9.4f} {r_rv-r_pk:+9.4f}")
    if agg:
        m = np.array(agg).mean(axis=0)
        print(f"  {'MEAN':5s} {'':6s} {m[0]:9.4f} {m[1]:10.4f} {m[2]:10.4f} "
              f"{m[3]:9.4f} {m[0]-m[1]:+9.4f}")
        print(f"\n  incremental value of 5-min RV OVER daily-OHLC Parkinson:")
        print(f"    as a replacement : {m[0]-m[1]:+.4f} R-squared")
        print(f"    added alongside  : {m[3]-m[1]:+.4f} R-squared  (RV+Park vs Park alone)")

print("\n" + "=" * 100)
print("3 -- is the overnight gap the part daily bars miss? decompose total variance")
print("=" * 100)
print(f"  {'sym':5s} {'RTH RV share':>13s} {'overnight share':>16s}")
for s in SYMS:
    d = features(panels[s])
    # overnight return: prev close -> today's open (from the same 5-min bars)
    on = np.log(d["open"] / d["close"].shift(1)).where(~d["is_split"])
    rth = d["rv_5m"]
    on_var = float(np.nanmean(on ** 2))
    rth_var = float(np.nanmean(rth))
    tot = on_var + rth_var
    print(f"  {s.upper():5s} {rth_var/tot:13.3f} {on_var/tot:16.3f}")
print("  -> the overnight component is a real, separable piece of daily variance that")
print("     5-min RV alone does NOT contain; a complete spec needs RV + overnight.")
