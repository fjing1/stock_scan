"""Probe 3: how far back does the EODHD demo token really go?

_data_probe_eodhd.py saw [] for a 2020-01-02..2020-02-02 window but
_data_probe_eodhd2.py saw 1170 bars for 2021-01. Resolve the true floor by
walking month windows back from 2021-01 through 2014. Reaching 2020-02/03 matters
enormously: that is the COVID vol explosion, the most informative regime a
volatility model can train on.

Also: does the 600-day cap chunk cleanly into a continuous multi-year series?
"""
import datetime as dt
import time

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def fetch(sym, interval, frm, to, timeout=120):
    p = {"api_token": "demo", "interval": interval, "fmt": "json",
         "from": int(frm), "to": int(to)}
    r = requests.get(f"https://eodhd.com/api/intraday/{sym}", params=p,
                     headers=UA, timeout=timeout)
    if r.status_code != 200:
        return r.status_code, None, r.text[:90]
    try:
        return 200, r.json(), ""
    except Exception:  # noqa: BLE001
        return 200, None, "nonjson"


def ts(y, m, d=1):
    return int(dt.datetime(y, m, d, tzinfo=dt.timezone.utc).timestamp())


print("=" * 100)
print("A -- walk BACK from 2021 to 2014, one 26-day window per quarter, AAPL 5m")
print("=" * 100)
found = []
for y in range(2021, 2013, -1):
    for m in (1, 4, 7, 10):
        frm, to = ts(y, m), ts(y, m) + 26 * 86400
        code, js, err = fetch("AAPL.US", "5m", frm, to)
        n = len(js) if js else 0
        first = js[0]["datetime"] if n else "-"
        flag = "DATA" if n else "none"
        print(f"  {y}-{m:02d}  HTTP {code}  n={n:5d}  {flag:4s} first={first} {err}")
        if n:
            found.append((y, m, first, n))
        time.sleep(0.6)

if found:
    oldest = min(found)
    print(f"\n  --> oldest window WITH data: {oldest[0]}-{oldest[1]:02d}, first bar {oldest[2]}")
else:
    print("\n  --> no data older than 2021")

print("\n" + "=" * 100)
print("B -- month-by-month through 2020 (COVID year) to see if the vol spike is covered")
print("=" * 100)
cov = []
for m in range(1, 13):
    frm, to = ts(2020, m), ts(2020, m) + 26 * 86400
    code, js, err = fetch("AAPL.US", "5m", frm, to)
    n = len(js) if js else 0
    first = js[0]["datetime"] if n else "-"
    print(f"  2020-{m:02d}  HTTP {code}  n={n:5d}  first={first} {err}")
    cov.append((m, n))
    time.sleep(0.6)
print(f"  2020 months with data: {[m for m, n in cov if n]}")

print("\n" + "=" * 100)
print("C -- can 600-day chunks be stitched into one continuous multi-year 5m series? (VTI)")
print("=" * 100)
now = int(time.time())
CHUNK = 590 * 86400
frames = []
cursor = now
t0 = time.time()
nreq = 0
for i in range(8):
    frm = cursor - CHUNK
    code, js, err = fetch("VTI.US", "5m", frm, cursor, timeout=180)
    nreq += 1
    n = len(js) if js else 0
    ds = (dt.datetime.fromtimestamp(frm, dt.timezone.utc).date(),
          dt.datetime.fromtimestamp(cursor, dt.timezone.utc).date())
    if n:
        d = pd.DataFrame(js)
        d["dtm"] = pd.to_datetime(d["datetime"], utc=True)
        frames.append(d)
        print(f"  chunk {i}: req {ds[0]}..{ds[1]}  HTTP {code} n={n:6d}  "
              f"got {d['dtm'].min().date()}..{d['dtm'].max().date()}")
    else:
        print(f"  chunk {i}: req {ds[0]}..{ds[1]}  HTTP {code} n=0 {err}  -> floor reached")
        break
    cursor = frm
    time.sleep(0.6)
el = time.time() - t0

if frames:
    allf = pd.concat(frames, ignore_index=True)
    allf = allf.drop_duplicates(subset=["timestamp"]).sort_values("dtm")
    # drop the synthetic closing stub bar (NaN volume) flagged in probe 2
    stub = allf["volume"].isna().sum()
    clean = allf.dropna(subset=["open", "high", "low", "close", "volume"])
    print(f"\n  STITCHED: {len(allf):,} unique bars in {nreq} requests, {el:.1f}s wall")
    print(f"  span    : {allf['dtm'].min()} .. {allf['dtm'].max()}")
    yrs = (allf['dtm'].max() - allf['dtm'].min()).days / 365.25
    print(f"  years   : {yrs:.2f}")
    print(f"  trading days: {allf['dtm'].dt.date.nunique()}")
    print(f"  NaN-volume stub bars: {stub} (~1/day, must be dropped)")
    print(f"  clean bars: {len(clean):,}")
    bpd = clean.groupby(clean["dtm"].dt.date).size()
    print(f"  bars/day: median={bpd.median():.0f} p05={bpd.quantile(.05):.0f} "
          f"p95={bpd.quantile(.95):.0f} min={bpd.min()} max={bpd.max()}")
    days_short = (bpd < 70).sum()
    print(f"  days with <70 bars (half-days/gaps): {days_short}")
    # realized vol from 5m returns, per day, as a sanity check
    clean = clean.copy()
    clean["d"] = clean["dtm"].dt.date
    clean["lr"] = clean.groupby("d")["close"].transform(
        lambda s: pd.Series(s).astype(float).pipe(lambda x: (x / x.shift(1)).apply(
            lambda v: None if v is None or v <= 0 else __import__("math").log(v))))
    rv = clean.groupby("d")["lr"].apply(lambda s: (s.dropna() ** 2).sum())
    rv_ann = (rv * 252) ** 0.5
    print(f"\n  daily RV from 5m returns (annualized): n={len(rv_ann)}")
    print(f"    mean={rv_ann.mean():.3f} median={rv_ann.median():.3f} "
          f"min={rv_ann.min():.3f} max={rv_ann.max():.3f}")
    print(f"    highest-vol days:\n{rv_ann.sort_values(ascending=False).head(5).to_string()}")
    allf.to_pickle("_data_probe_eodhd_vti_stitched.pkl")
    print("\n  saved -> _data_probe_eodhd_vti_stitched.pkl")
