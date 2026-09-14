"""Does the free intraday data actually support HAR-RV, and does it beat daily OHLC?

This is the payoff test for category (A). Everything here uses the EODHD demo
token (no signup, no key) verified by _data_probe_eodhd*.py:
  5-minute bars, 2020-10-12 .. today, 600-day cap per request, ~78 bars/day.

Build, for each of the 5 demo equities:
  * daily realized variance RV from 5-minute intraday log returns  (the new input)
  * daily OHLC vol proxies from the SAME bars aggregated to daily  (the incumbent)
Then compare out-of-sample R-squared of
  (a) HAR-RV        : log RV_{t+1} ~ log RV_d + log RV_w + log RV_m     [5-min RV]
  (b) HAR-on-daily  : same lag structure but RV built from squared daily returns
  (c) HAR-Parkinson : same lag structure, Parkinson high-low estimator
with a single chronological 70/30 split (no tuning, no peeking).

If (a) does not beat (b)/(c) on this sample, intraday bars are not the unlock.

No scipy/sklearn in the venv -- numpy.linalg.lstsq only.
"""
import datetime as dt
import math
import time

import numpy as np
import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
SYMS = ["VTI.US", "AAPL.US", "MSFT.US", "AMZN.US", "TSLA.US"]
CHUNK_DAYS = 590


def fetch_5m(sym):
    """Walk 590-day chunks back until the vendor floor; return one clean frame."""
    now = int(time.time())
    cursor, frames, nreq, nbytes = now, [], 0, 0
    while True:
        frm = cursor - CHUNK_DAYS * 86400
        r = requests.get(f"https://eodhd.com/api/intraday/{sym}",
                         params={"api_token": "demo", "interval": "5m", "fmt": "json",
                                 "from": frm, "to": cursor},
                         headers=UA, timeout=180)
        nreq += 1
        nbytes += len(r.content)
        if r.status_code != 200:
            break
        js = r.json()
        if not js:
            break
        frames.append(pd.DataFrame(js))
        cursor = frm
        time.sleep(0.4)
    if not frames:
        return None, nreq, nbytes
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["timestamp"])
    df["dtm"] = pd.to_datetime(df["datetime"], utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # drop the synthetic closing-auction stub bar (NaN volume) and any NaN OHLC
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df = df.sort_values("dtm").reset_index(drop=True)
    return df, nreq, nbytes


def daily_from_5m(df):
    """RTH-only daily aggregation + realized variance from 5-min returns."""
    et = df["dtm"].dt.tz_convert("America/New_York")
    mins = et.dt.hour * 60 + et.dt.minute
    rth = df[(mins >= 570) & (mins < 960)].copy()          # 09:30 <= t < 16:00
    rth["d"] = rth["dtm"].dt.tz_convert("America/New_York").dt.date
    g = rth.groupby("d")
    out = pd.DataFrame({
        "n_bars": g.size(),
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
    })
    # realized variance = sum of squared 5-min log returns WITHIN each day
    rth["lr"] = np.log(rth["close"] / rth.groupby("d")["close"].shift(1))
    out["rv"] = rth.groupby("d")["lr"].apply(lambda s: np.nansum(s.values ** 2))
    # keep only full sessions: 78 bars expected; allow 75+ to tolerate feed hiccups
    out = out[out["n_bars"] >= 75]
    # drop today's in-progress session (repo lesson: partial bars corrupt analyses)
    today_et = dt.datetime.now(dt.timezone.utc).astimezone(
        dt.timezone(dt.timedelta(hours=-4))).date()
    out = out[out.index < today_et]
    return out


def har_design(x):
    """HAR lags: previous day, previous week mean, previous month mean (all shifted)."""
    s = pd.Series(x).astype(float)
    d = s.shift(1)
    w = s.shift(1).rolling(5).mean()
    m = s.shift(1).rolling(22).mean()
    return pd.DataFrame({"d": d, "w": w, "m": m})


def oos_r2(X, y, split=0.70):
    """Chronological split, OLS on train, R-squared vs train-mean on test."""
    ok = X.notna().all(axis=1) & y.notna()
    X, y = X[ok], y[ok]
    n = len(y)
    if n < 200:
        return None, n
    k = int(n * split)
    Xtr = np.column_stack([np.ones(k), X.iloc[:k].values])
    Xte = np.column_stack([np.ones(n - k), X.iloc[k:].values])
    ytr, yte = y.iloc[:k].values, y.iloc[k:].values
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    pred = Xte @ beta
    ss_res = float(np.sum((yte - pred) ** 2))
    ss_tot = float(np.sum((yte - ytr.mean()) ** 2))
    return 1.0 - ss_res / ss_tot, n


print("=" * 104)
print("FETCH -- 5-minute bars via the EODHD demo token (no key, no signup)")
print("=" * 104)
panels, tot_req, tot_bytes, t0 = {}, 0, 0, time.time()
for sym in SYMS:
    df, nreq, nb = fetch_5m(sym)
    tot_req += nreq
    tot_bytes += nb
    if df is None or df.empty:
        print(f"  {sym:9s} FAILED")
        continue
    dl = daily_from_5m(df)
    panels[sym] = dl
    print(f"  {sym:9s} 5m bars={len(df):7,d}  requests={nreq}  "
          f"{df['dtm'].min().date()}..{df['dtm'].max().date()}  "
          f"-> clean RTH sessions={len(dl):5d}")
print(f"\n  TOTAL: {tot_req} requests, {tot_bytes/1e6:.1f} MB gzipped-on-wire, "
      f"{time.time()-t0:.1f}s wall for {len(panels)} symbols")

print("\n" + "=" * 104)
print("SANITY -- annualized vol from 5-min RV vs from daily close-to-close")
print("=" * 104)
for sym, dl in panels.items():
    rv_ann = np.sqrt(dl["rv"] * 252)
    dret = np.log(dl["close"] / dl["close"].shift(1))
    dv_ann = dret.std() * math.sqrt(252)
    print(f"  {sym:9s} RV-based mean={rv_ann.mean():.3f} median={rv_ann.median():.3f} "
          f"max={rv_ann.max():.3f} | daily close-to-close sd={dv_ann:.3f} "
          f"| ratio(RV/daily)={rv_ann.mean()/dv_ann:.2f}")
print("  (RV excludes the overnight gap, so RV < close-to-close is EXPECTED and is")
print("   exactly the decomposition daily bars cannot give you.)")

print("\n" + "=" * 104)
print("HAR HORSE RACE -- out-of-sample R-squared, target = log RV_{t+1} (5-min truth)")
print("=" * 104)
print(f"  {'symbol':9s} {'n':>5s} {'(a) HAR-RV 5min':>16s} {'(b) HAR daily r^2':>18s} "
      f"{'(c) HAR Parkinson':>18s} {'a-b':>7s} {'a-c':>7s}")
rows = []
for sym, dl in panels.items():
    dl = dl.copy()
    dret = np.log(dl["close"] / dl["close"].shift(1))
    dl["rv_daily"] = dret ** 2
    dl["rv_park"] = (np.log(dl["high"] / dl["low"]) ** 2) / (4 * math.log(2))
    eps = 1e-10
    y = np.log(dl["rv"].shift(-1) + eps)          # same target for all three
    Xa = np.log(har_design(dl["rv"]) + eps)
    Xb = np.log(har_design(dl["rv_daily"]) + eps)
    Xc = np.log(har_design(dl["rv_park"]) + eps)
    ra, n = oos_r2(Xa, y)
    rb, _ = oos_r2(Xb, y)
    rc, _ = oos_r2(Xc, y)
    if ra is None:
        print(f"  {sym:9s} n={n} too short")
        continue
    rows.append((sym, n, ra, rb, rc))
    print(f"  {sym:9s} {n:5d} {ra:16.4f} {rb:18.4f} {rc:18.4f} "
          f"{ra-rb:+7.4f} {ra-rc:+7.4f}")
if rows:
    a = np.mean([r[2] for r in rows]); b = np.mean([r[3] for r in rows])
    c = np.mean([r[4] for r in rows])
    print(f"  {'MEAN':9s} {'':5s} {a:16.4f} {b:18.4f} {c:18.4f} {a-b:+7.4f} {a-c:+7.4f}")

print("\n" + "=" * 104)
print("ALSO -- can 5-min RV predict the DAILY-return magnitude the live model cares about?")
print("   target = log|daily close-to-close return_{t+1}|")
print("=" * 104)
print(f"  {'symbol':9s} {'n':>5s} {'RV(5min) lags':>15s} {'daily r^2 lags':>16s} "
      f"{'Parkinson lags':>16s} {'delta':>8s}")
d_rows = []
for sym, dl in panels.items():
    dl = dl.copy()
    dret = np.log(dl["close"] / dl["close"].shift(1))
    dl["rv_daily"] = dret ** 2
    dl["rv_park"] = (np.log(dl["high"] / dl["low"]) ** 2) / (4 * math.log(2))
    eps = 1e-10
    y = np.log(dret.abs().shift(-1) + 1e-6)
    ra, n = oos_r2(np.log(har_design(dl["rv"]) + eps), y)
    rb, _ = oos_r2(np.log(har_design(dl["rv_daily"]) + eps), y)
    rc, _ = oos_r2(np.log(har_design(dl["rv_park"]) + eps), y)
    if ra is None:
        continue
    d_rows.append((ra, rb, rc))
    print(f"  {sym:9s} {n:5d} {ra:15.4f} {rb:16.4f} {rc:16.4f} {ra-max(rb,rc):+8.4f}")
if d_rows:
    a = np.mean([r[0] for r in d_rows]); b = np.mean([r[1] for r in d_rows])
    c = np.mean([r[2] for r in d_rows])
    print(f"  {'MEAN':9s} {'':5s} {a:15.4f} {b:16.4f} {c:16.4f} "
          f"{a-max(b,c):+8.4f}   <-- this is the number that matters for move_prob")

for sym, dl in panels.items():
    dl.to_pickle(f"_data_probe_har_{sym.split('.')[0].lower()}.pkl")
print("\nsaved per-symbol daily panels -> _data_probe_har_<sym>.pkl")
