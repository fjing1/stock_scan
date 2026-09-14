"""Probe: Alpha Vantage HISTORICAL_OPTIONS on the free key. Budget: <=5 requests."""
import json, os, sys, time, urllib.parse
import requests

ENV = "/Users/feijing/github.com/stock_scan/.env"
key = None
for line in open(ENV):
    if line.startswith("ALPHAVANTAGE_API_KEY="):
        key = line.strip().split("=", 1)[1]
assert key, "no key"
BASE = "https://www.alphavantage.co/query"

def call(params, label):
    params = dict(params); params["apikey"] = key
    t0 = time.time()
    r = requests.get(BASE, params=params, timeout=60)
    dt = time.time() - t0
    print(f"\n=== {label} ===")
    print("URL:", r.url.replace(key, "***"))
    print(f"HTTP {r.status_code}  {len(r.content)} bytes  {dt:.2f}s")
    txt = r.text
    try:
        j = r.json()
    except Exception:
        print("NON-JSON first 500:", txt[:500]); return None
    if isinstance(j, dict):
        print("top-level keys:", list(j.keys())[:10])
        for k in ("Information", "Note", "Error Message", "message"):
            if k in j:
                print(f"  {k}: {j[k]}")
    return j

# 1) exactly the documented call, no date -> most recent trading day
j1 = call({"function": "HISTORICAL_OPTIONS", "symbol": "IBM"}, "1. HISTORICAL_OPTIONS IBM (no date)")
if j1 and "data" in j1:
    d = j1["data"]
    print("rows:", len(d))
    if d:
        print("row[0] keys:", list(d[0].keys()))
        print("row[0]:", json.dumps(d[0], indent=1)[:900])
        dates = sorted({r.get("date","") for r in d})
        exps  = sorted({r.get("expiration","") for r in d})
        print("distinct date(s):", dates[:5], "n=", len(dates))
        print("expiries:", exps[:6], "...", exps[-3:], "n=", len(exps))
        ivs = [r.get("implied_volatility") for r in d[:5]]
        print("sample implied_volatility:", ivs)
        json.dump(j1, open("/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/_data_probe_av_histopt.json","w"))
