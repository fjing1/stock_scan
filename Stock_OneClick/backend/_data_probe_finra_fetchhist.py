"""Download the FULL free FINRA Reg SHO short-volume history (2018-08-01 .. today)
filtered to the move_prob 284-symbol universe, and cache it.

Output: _data_probe_finra_hist.pkl   (tidy: date, symbol, short, exempt, total)
        _data_probe_finra_hist.log
"""
import concurrent.futures as cf
import pickle
import sys
import time

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) research-probe"}
URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d}.txt"

panel = pickle.load(open("_move_panel.pkl", "rb"))
SYMS = set(panel["Close"].columns.tolist())
SYMS |= {"SPY", "QQQ", "IWM", "DIA", "VTI", "VOO"}
SYMS = {s for s in SYMS if not s.startswith("^")}
print(f"universe: {len(SYMS)} symbols", flush=True)

days = pd.bdate_range("2018-08-01", "2026-09-11").strftime("%Y%m%d").tolist()
print(f"business days to try: {len(days)}", flush=True)

sess_local = {}


def one(d):
    s = sess_local.get("s")
    if s is None:
        s = requests.Session()
        s.headers.update(UA)
        sess_local["s"] = s
    for attempt in range(3):
        try:
            r = s.get(URL.format(d=d), timeout=45)
            break
        except Exception:
            if attempt == 2:
                return d, None, "EXC"
            time.sleep(1.5 * (attempt + 1))
    if r.status_code != 200:
        return d, None, r.status_code
    rows = []
    for ln in r.text.split("\n")[1:]:
        if not ln:
            continue
        p = ln.split("|")
        if len(p) < 6:
            continue
        if p[1] in SYMS:
            try:
                rows.append((p[0], p[1], float(p[2]), float(p[3]), float(p[4])))
            except ValueError:
                continue
    return d, rows, 200


t0 = time.time()
allrows = []
codes = {}
done = 0
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    for d, rows, code in ex.map(one, days):
        codes[code] = codes.get(code, 0) + 1
        if rows:
            allrows.extend(rows)
        done += 1
        if done % 250 == 0:
            print(f"  {done}/{len(days)} days, {len(allrows):,} rows, "
                  f"{time.time()-t0:.0f}s", flush=True)

el = time.time() - t0
print(f"\ndownload+parse wall clock: {el:.1f}s  ({el/60:.1f} min)")
print(f"HTTP code histogram: {codes}")

df = pd.DataFrame(allrows, columns=["date", "symbol", "short", "exempt", "total"])
df["date"] = pd.to_datetime(df.date, format="%Y%m%d")
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
print(f"rows={len(df):,}  symbols={df.symbol.nunique()}  "
      f"span {df.date.min().date()} .. {df.date.max().date()}  "
      f"trading days={df.date.nunique()}")
df.to_pickle("_data_probe_finra_hist.pkl")
print("saved -> _data_probe_finra_hist.pkl")
print(df.head(3).to_string(index=False))
print(f"memory: {df.memory_usage(deep=True).sum()/1e6:.1f} MB")
