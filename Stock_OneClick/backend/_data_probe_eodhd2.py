"""Probe 2: pin down EODHD demo-token limits precisely.

Established by _data_probe_eodhd.py:
  * 5m bars exist for a 2022-01 window, empty for 2020-01  -> floor is between
  * hard 600-day cap per request (HTTP 422 beyond)
  * AAPL/MSFT/AMZN/TSLA/VTI = 200 ; SPY/QQQ/NVDA/GE = 403
  * X-RateLimit-Limit: 1200

Now measure: exact floor date per interval, the full symbol whitelist, and
whether the bars agree with yfinance on an overlapping window.
"""
import datetime as dt
import json
import time

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
TOKEN = "demo"


def fetch(sym, interval, frm, to, timeout=90):
    p = {"api_token": TOKEN, "interval": interval, "fmt": "json",
         "from": int(frm), "to": int(to)}
    r = requests.get(f"https://eodhd.com/api/intraday/{sym}", params=p,
                     headers=UA, timeout=timeout)
    if r.status_code != 200:
        return r.status_code, None, r.text[:100]
    try:
        js = r.json()
    except Exception:  # noqa: BLE001
        return 200, None, "nonjson"
    return 200, js, ""


def ts(y, m, d):
    return int(dt.datetime(y, m, d, tzinfo=dt.timezone.utc).timestamp())


print("=" * 110)
print("A -- exact history floor: probe one 25-day window per month, 2021-01 .. 2022-06, AAPL 5m")
print("=" * 110)
floor = None
for y in (2021, 2022):
    for m in range(1, 13):
        if y == 2022 and m > 6:
            break
        frm = ts(y, m, 1)
        to = frm + 25 * 24 * 3600
        code, js, err = fetch("AAPL.US", "5m", frm, to)
        n = len(js) if js else 0
        first = js[0]["datetime"] if n else "-"
        print(f"  {y}-{m:02d}  HTTP {code}  n={n:5d}  first={first}  {err}")
        if n and floor is None:
            floor = (y, m, first)
        time.sleep(0.7)
print(f"\n  --> earliest month with 5m data: {floor}")

print("\n" + "=" * 110)
print("B -- floor per interval (1m vs 5m vs 1h), AAPL, using a window at the floor and older")
print("=" * 110)
for interval in ("1m", "5m", "1h"):
    for label, (frm, to) in {
        "2021-06 (25d)": (ts(2021, 6, 1), ts(2021, 6, 1) + 25 * 86400),
        "2022-01 (25d)": (ts(2022, 1, 1), ts(2022, 1, 1) + 25 * 86400),
        "2023-01 (25d)": (ts(2023, 1, 1), ts(2023, 1, 1) + 25 * 86400),
        "2026-06 (25d)": (ts(2026, 6, 1), ts(2026, 6, 1) + 25 * 86400),
    }.items():
        code, js, err = fetch("AAPL.US", interval, frm, to)
        n = len(js) if js else 0
        rng = f"{js[0]['datetime']} .. {js[-1]['datetime']}" if n else "-"
        print(f"  {interval:3s} {label:16s} HTTP {code} n={n:6d}  {rng}  {err}")
        time.sleep(0.7)

print("\n" + "=" * 110)
print("C -- full symbol whitelist for the demo token (5m, recent 20d)")
print("=" * 110)
CANDIDATES = [
    "AAPL.US", "MSFT.US", "AMZN.US", "TSLA.US", "VTI.US", "GOOGL.US", "GOOG.US",
    "META.US", "NVDA.US", "SPY.US", "QQQ.US", "IWM.US", "DIA.US", "VOO.US",
    "BRK-B.US", "JPM.US", "XOM.US", "UNH.US", "V.US", "WMT.US",
    "EURUSD.FOREX", "BTC-USD.CC", "GSPC.INDX", "VIX.INDX", "^GSPC",
]
now = int(time.time())
ok, bad = [], []
for sym in CANDIDATES:
    code, js, err = fetch(sym, "5m", now - 20 * 86400, now)
    n = len(js) if js else 0
    if code == 200 and n:
        ok.append(sym)
        print(f"  OK      {sym:16s} n={n:6d}  {js[0]['datetime']} .. {js[-1]['datetime']}")
    else:
        bad.append((sym, code))
        print(f"  DENIED  {sym:16s} HTTP {code} n={n} {err}")
    time.sleep(0.7)
print(f"\n  allowed: {ok}")
print(f"  denied : {[b[0] for b in bad]}")

print("\n" + "=" * 110)
print("D -- pull ONE full 600-day window and inspect quality + size")
print("=" * 110)
to = now
frm = now - 595 * 86400
t0 = time.time()
code, js, err = fetch("VTI.US", "5m", frm, to, timeout=180)
el = time.time() - t0
print(f"  VTI.US 5m 595d: HTTP {code} n={len(js) if js else 0} in {el:.1f}s")
if js:
    df = pd.DataFrame(js)
    df["dtm"] = pd.to_datetime(df["datetime"], utc=True)
    print(f"  span      : {df['dtm'].min()} .. {df['dtm'].max()}")
    print(f"  trading days: {df['dtm'].dt.date.nunique()}")
    print(f"  columns   : {list(df.columns)}")
    print(f"  bars/day  : median={df.groupby(df['dtm'].dt.date).size().median():.0f}")
    print(f"  nulls     : {df[['open','high','low','close','volume']].isna().sum().to_dict()}")
    print(f"  zero-vol bars: {(df['volume'] == 0).sum()} / {len(df)}")
    print("\n  first 3 rows:")
    print(df[["datetime", "open", "high", "low", "close", "volume"]].head(3).to_string(index=False))
    print("  last 3 rows:")
    print(df[["datetime", "open", "high", "low", "close", "volume"]].tail(3).to_string(index=False))
    raw_bytes = len(json.dumps(js))
    print(f"\n  raw JSON bytes: {raw_bytes:,}  ({raw_bytes/len(js):.1f} B/bar)")
    df.to_pickle("_data_probe_eodhd_vti5m.pkl")
    with open("_data_probe_eodhd_sample.json", "w") as fh:
        json.dump(js[:50], fh, indent=1)
    print("  saved -> _data_probe_eodhd_vti5m.pkl, _data_probe_eodhd_sample.json")

print("\n" + "=" * 110)
print("E -- cross-validate EODHD 5m against yfinance 5m on the overlapping window (AAPL)")
print("=" * 110)
import yfinance as yf  # noqa: E402

yfd = yf.download("AAPL", period="1mo", interval="5m", progress=False,
                  auto_adjust=False, prepost=False, threads=False)
if yfd is not None and len(yfd):
    if isinstance(yfd.columns, pd.MultiIndex):
        yfd.columns = yfd.columns.get_level_values(0)
    code, js, err = fetch("AAPL.US", "5m", now - 32 * 86400, now)
    e = pd.DataFrame(js)
    e["dtm"] = pd.to_datetime(e["datetime"], utc=True)
    e = e.set_index("dtm")[["open", "high", "low", "close", "volume"]]
    y = yfd.copy()
    y.index = pd.to_datetime(y.index, utc=True)
    y = y[["Open", "High", "Low", "Close", "Volume"]]
    y.columns = ["open", "high", "low", "close", "volume"]
    j = e.join(y, how="inner", lsuffix="_eod", rsuffix="_yf")
    print(f"  yfinance bars={len(y)}  eodhd bars={len(e)}  matched timestamps={len(j)}")
    if len(j):
        for c in ("close", "high", "low"):
            d = (j[f"{c}_eod"] - j[f"{c}_yf"]).abs()
            rel = (d / j[f"{c}_yf"]).replace([float("inf")], pd.NA).dropna()
            print(f"  {c:6s} mean|diff|={d.mean():.6f}  max|diff|={d.max():.4f}  "
                  f"median rel={rel.median():.2e}  bars w/ rel>1e-4: {(rel > 1e-4).sum()}")
        dv = (j["volume_eod"] - j["volume_yf"]).abs()
        print(f"  volume mean|diff|={dv.mean():.1f}  median eod={j['volume_eod'].median():.0f} "
              f"median yf={j['volume_yf'].median():.0f}")
        print("\n  sample matched rows:")
        print(j[["close_eod", "close_yf", "volume_eod", "volume_yf"]].head(5).to_string())
