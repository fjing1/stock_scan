"""PROBE round 9 -- THE DECISIVE MEASUREMENT.
Take a random sample of Alpha Vantage's 9,470 delisted symbols and measure what fraction
stockanalysis.com's free keyless API actually returns usable daily history for.
Also test the `otc/<SYM>` route discovered via their search API, and the ticker-recycling hazard.
"""
import json
import random
import statistics
import time

import pandas as pd
import requests

BR = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
S = requests.Session()
S.headers.update({"User-Agent": BR})
HIST = "https://stockanalysis.com/api/symbol/s/{s}/history"
SEARCH = "https://stockanalysis.com/api/search"


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def hist(path, rng="10Y"):
    r = S.get(HIST.format(s=path), params={"range": rng, "period": "Daily"}, timeout=45)
    if r.status_code == 200:
        return r.json().get("data") or [], r.status_code, len(r.content)
    return [], r.status_code, len(r.content)


hr("9.1 the otc/<SYM> route from their search API")
for p in ["otc/FRCB", "FRCB", "otc/SBNY", "SBNY", "otc/SIVBQ", "otc/CTXS", "otc/ABMD"]:
    try:
        d, sc, nb = hist(p)
        print(f"  {p:12s} HTTP{sc} rows={len(d):<6} "
              f"{(d[-1]['t'] if d else '-')}..{(d[0]['t'] if d else '-')} "
              f"lastC={(d[0]['c'] if d else '-')}")
    except Exception as e:
        print(f"  {p:12s} EXC {str(e)[:70]}")
    time.sleep(0.4)

hr("9.2 TICKER RECYCLING HAZARD -- does a reused ticker silently return the WRONG company?")
for sym, expect in [("VIAC", "ViacomCBS (delisted 2019-12-04 per AV)"),
                    ("ABMD", "Abiomed (delisted 2023-01-03 per AV)"),
                    ("FRC", "First Republic (failed 2023-05)")]:
    try:
        r = S.get(f"https://stockanalysis.com/stocks/{sym.lower()}/", timeout=45)
        import re
        t = re.search(r"<title>(.*?)</title>", r.text, re.S)
        print(f"  {sym:6s} expected={expect}")
        print(f"         page title -> {(t.group(1)[:70] if t else 'n/a')}")
    except Exception as e:
        print(f"  {sym} EXC {str(e)[:60]}")
    time.sleep(0.5)

hr("9.3 DECISIVE: hit rate over a random sample of AV's 9,470 delisted symbols")
dl = pd.read_csv("/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/"
                 "_data_probe_surv_av_delisted_all.csv")
stocks = dl[dl["assetType"] == "Stock"].copy()
stocks["dy"] = stocks["delistingDate"].str[:4].astype(int)
print(f"  AV delisted STOCKS: {len(stocks):,}")

random.seed(20260914)
SAMPLE_N = 200
samp = stocks.sample(SAMPLE_N, random_state=7)
ok, empty, bad = [], [], []
lat, byt = [], []
by_era = {}
for i, row in enumerate(samp.itertuples(), 1):
    sym = str(row.symbol)
    era = "2010-2015" if row.dy <= 2015 else ("2016-2020" if row.dy <= 2020
                                             else "2021-2026")
    by_era.setdefault(era, [0, 0])
    by_era[era][1] += 1
    try:
        t0 = time.time()
        d, sc, nb = hist(sym)
        lat.append(time.time() - t0)
        if sc == 200 and len(d) >= 60:
            ok.append((sym, len(d), d[-1]["t"], d[0]["t"], row.delistingDate))
            byt.append(nb)
            by_era[era][0] += 1
        elif sc == 200:
            empty.append((sym, len(d)))
        else:
            bad.append((sym, sc))
    except Exception:
        bad.append((sym, "EXC"))
    if i % 50 == 0:
        print(f"    ...{i}/{SAMPLE_N}  ok={len(ok)} thin={len(empty)} fail={len(bad)}")
    time.sleep(0.12)

print(f"\n  RESULT over {SAMPLE_N} random AV-delisted stocks:")
print(f"    usable (>=60 daily bars) : {len(ok):3d}  ({len(ok)/SAMPLE_N:.1%})")
print(f"    HTTP200 but <60 bars     : {len(empty):3d}  ({len(empty)/SAMPLE_N:.1%})")
print(f"    HTTP error / absent      : {len(bad):3d}  ({len(bad)/SAMPLE_N:.1%})")
print(f"    error codes: {pd.Series([b[1] for b in bad]).value_counts().to_dict()}")
print("\n  hit rate by delisting era:")
for era in sorted(by_era):
    g, n = by_era[era]
    print(f"    {era}: {g}/{n} = {g/max(1,n):.1%}")

if ok:
    rows = [x[1] for x in ok]
    print(f"\n  for the hits: median bars={statistics.median(rows):,.0f} "
          f"mean={statistics.mean(rows):,.0f} min={min(rows)} max={max(rows)}")
    print(f"  median KB/symbol={statistics.median(byt)/1024:.0f}  "
          f"median latency={statistics.median(lat):.2f}s")
    print("\n  sample hits (sym, bars, first, last, AV delistingDate):")
    for x in ok[:12]:
        print(f"    {x}")
    # does the API's last bar agree with AV's delisting date?
    import datetime as dtm
    diffs = []
    for sym, n, f, l, dd in ok:
        try:
            diffs.append(abs((dtm.date.fromisoformat(l)
                              - dtm.date.fromisoformat(str(dd))).days))
        except Exception:
            pass
    if diffs:
        print(f"\n  |last bar - AV delistingDate| days: median={statistics.median(diffs):.0f} "
              f"p90={sorted(diffs)[int(0.9*len(diffs))]} max={max(diffs)}")
        print(f"    within 5 days: {sum(1 for d in diffs if d<=5)/len(diffs):.1%}  "
              f"within 30 days: {sum(1 for d in diffs if d<=30)/len(diffs):.1%}")
    pd.DataFrame(ok, columns=["symbol", "bars", "first", "last", "av_delist"]).to_csv(
        "_data_probe_surv_sa_hitrate.csv", index=False)
    print("  saved _data_probe_surv_sa_hitrate.csv")

hr("9.4 projected panel size with the measured hit rate")
if ok:
    hit = len(ok) / SAMPLE_N
    med_bars = statistics.median([x[1] for x in ok])
    med_kb = statistics.median(byt) / 1024
    med_lat = statistics.median(lat)
    n_del = len(stocks)
    n_live = 1283
    print(f"  measured hit rate={hit:.1%}, median {med_bars:,.0f} bars, "
          f"{med_kb:.0f} KB, {med_lat:.2f}s per symbol")
    recovered = int(n_del * hit)
    print(f"  of {n_del:,} AV-delisted stocks -> ~{recovered:,} recoverable with "
          f"up-to-10y daily history")
    tot = recovered + n_live
    print(f"  delisted-inclusive panel: ~{tot:,} names, "
          f"~{tot*med_bars/1e6:.1f}M bars, ~{tot*med_kb/1024:.0f} MB JSON")
    print(f"  wall clock: serial {tot*med_lat/60:.0f} min; "
          f"4 workers {tot*med_lat/60/4:.0f} min")
    print(f"  vs the current 284-name survivor panel (6,289 x 284 = "
          f"{6289*284/1e6:.1f}M bars)")

print("\nDONE")
