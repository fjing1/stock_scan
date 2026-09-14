"""Probe: FINRA OTC Transparency -- per-symbol, per-ATS (dark pool) and per-market-maker
weekly volume. Keyless via api.finra.org.

This is the closest thing to real VENUE-LEVEL order flow that is free: it says how many
shares of each symbol printed at each named ATS (UBS ATS, JPB-X, Sigma X, ...) and at each
non-ATS wholesaler (Citadel, Virtu, ...) each week.

The decisive question is the PUBLICATION LAG. FINRA publishes Tier 1 NMS names 2 weeks
after the week ends and Tier 2 / OTC names 4 weeks after. Measured below.
"""
import io
import json

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) research-probe"}
BASE = "https://api.finra.org/data/group/otcMarket/name/{ds}"


def sec(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def get(ds, params=None, body=None, fmt="csv"):
    url = BASE.format(ds=ds)
    h = dict(UA)
    if body is not None:
        h["Content-Type"] = "application/json"
        h["Accept"] = "text/plain" if fmt == "csv" else "application/json"
        r = requests.post(url, headers=h, data=json.dumps(body), timeout=120)
    else:
        r = requests.get(url, headers=h, params=params or {}, timeout=120)
    return r


sec("1. KEYLESS ACCESS + FIELD LIST")
r = get("weeklySummary", {"limit": 3})
print(f"GET weeklySummary?limit=3 -> HTTP {r.status_code}, {len(r.content):,} bytes, "
      f"content-type={r.headers.get('content-type')}")
df = pd.read_csv(io.StringIO(r.text))
print(f"columns ({len(df.columns)}): {list(df.columns)}")
print(df.to_string(index=False)[:1200])

sec("2. WHAT SUMMARY TYPES EXIST? (ATS vs non-ATS wholesaler)")
r = get("weeklySummary", body={"limit": 20000, "fields": ["summaryTypeCode", "tierDescription",
                                                          "weekStartDate"]})
print(f"POST 20k rows -> HTTP {r.status_code}, {len(r.content):,} bytes")
d = pd.read_csv(io.StringIO(r.text))
print(f"rows={len(d):,}")
print("\nsummaryTypeCode counts:")
print(d.summaryTypeCode.value_counts().to_string())
print("\ntierDescription counts:")
print(d.tierDescription.value_counts().to_string())

sec("3. MEASURED PUBLICATION LAG  (initialPublishedDate - weekStartDate)")
r = get("weeklySummary", body={
    "limit": 40000,
    "fields": ["issueSymbolIdentifier", "MPID", "marketParticipantName", "tierDescription",
               "summaryTypeCode", "weekStartDate", "initialPublishedDate",
               "totalWeeklyShareQuantity", "totalWeeklyTradeCount"],
    "compareFilters": [{"fieldName": "weekStartDate", "compareType": "GTE",
                        "fieldValue": "2026-01-01"}],
})
print(f"POST 2026 rows -> HTTP {r.status_code}, {len(r.content):,} bytes")
d = pd.read_csv(io.StringIO(r.text))
print(f"rows={len(d):,}")
if len(d):
    d["weekStartDate"] = pd.to_datetime(d.weekStartDate)
    d["initialPublishedDate"] = pd.to_datetime(d.initialPublishedDate)
    d["lag_days"] = (d.initialPublishedDate - d.weekStartDate).dt.days
    print(f"\nweekStartDate span: {d.weekStartDate.min().date()} .. {d.weekStartDate.max().date()}")
    print(f"MOST RECENT published week: {d.weekStartDate.max().date()}")
    print(f"\nlag (days from week start to first publication), by tier:")
    print(d.groupby("tierDescription").lag_days.describe()[
        ["count", "min", "25%", "50%", "75%", "max"]].to_string(float_format=lambda x: f"{x:,.0f}"))
    print("\nlag by summaryTypeCode:")
    print(d.groupby("summaryTypeCode").lag_days.agg(["count", "median", "max"])
          .to_string(float_format=lambda x: f"{x:,.0f}"))
    print(f"\ndistinct symbols: {d.issueSymbolIdentifier.nunique():,}  "
          f"distinct MPIDs (venues): {d.MPID.nunique()}")
    print("\ntop venues by total shares in 2026 sample:")
    v = (d.groupby(["MPID", "marketParticipantName"]).totalWeeklyShareQuantity.sum()
         .sort_values(ascending=False).head(12) / 1e9)
    print(v.to_string(float_format=lambda x: f"{x:,.2f} B shares"))
    d.to_csv("_data_probe_finra_otc_ats.csv", index=False)
    print("\nsaved -> _data_probe_finra_otc_ats.csv")

sec("4. TODAY'S DATE vs MOST RECENT DATA -- IS IT USABLE FOR A NEXT-WEEK FORECAST?")
today = pd.Timestamp("today").normalize()
if len(d):
    mr = d.weekStartDate.max()
    print(f"today                       : {today.date()}")
    print(f"most recent published week  : {mr.date()}")
    print(f"staleness                   : {(today - mr).days} days "
          f"({(today - mr).days/7:.1f} weeks)")
    print("\nA feature this stale cannot inform a 1-day or 1-week-ahead volatility")
    print("forecast. It is only usable for slow-moving, month-plus-horizon research.")

sec("5. SINGLE-SYMBOL PULL (AAPL) -- how granular does it get?")
r = get("weeklySummary", body={
    "limit": 5000,
    "compareFilters": [{"fieldName": "weekStartDate", "compareType": "GTE",
                        "fieldValue": "2026-06-01"}],
    "domainFilters": [{"fieldName": "issueSymbolIdentifier", "values": ["AAPL"]}],
})
print(f"POST AAPL -> HTTP {r.status_code}, {len(r.content):,} bytes")
if r.status_code == 200 and r.text.strip():
    a = pd.read_csv(io.StringIO(r.text))
    print(f"rows={len(a):,}")
    if len(a):
        keep = ["weekStartDate", "MPID", "marketParticipantName", "summaryTypeCode",
                "totalWeeklyTradeCount", "totalWeeklyShareQuantity"]
        keep = [c for c in keep if c in a.columns]
        print(a[keep].sort_values("totalWeeklyShareQuantity", ascending=False)
              .head(15).to_string(index=False))
        print(f"\ndistinct weeks: {a.weekStartDate.nunique()}  "
              f"distinct venues: {a.MPID.nunique()}")

sec("6. HISTORY DEPTH")
for yr in ("2015-01-01", "2017-01-01", "2019-01-01", "2021-01-01"):
    r = get("weeklySummary", body={
        "limit": 5,
        "fields": ["weekStartDate", "issueSymbolIdentifier"],
        "compareFilters": [{"fieldName": "weekStartDate", "compareType": "GTE",
                            "fieldValue": yr},
                           {"fieldName": "weekStartDate", "compareType": "LTE",
                            "fieldValue": yr[:4] + "-03-01"}],
    })
    n = 0
    if r.status_code == 200 and r.text.strip():
        try:
            n = len(pd.read_csv(io.StringIO(r.text)))
        except Exception:
            n = 0
    print(f"  weeks starting {yr[:7]}..{yr[:4]}-03 -> HTTP {r.status_code}, rows={n}")
