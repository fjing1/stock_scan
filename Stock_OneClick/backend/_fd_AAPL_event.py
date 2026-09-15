"""EVENT STUDY. Event dates come from the SEC submissions index (8-K Item 2.02 = results of
operations), never from memory. Reaction day = first trading day strictly AFTER the filing date,
because Apple releases results after the close and files the 8-K the same day.

Abnormal return = AAPL return minus benchmark return (SPY, and separately XLK)."""
import json, requests, numpy as np, pandas as pd, time
pd.set_option("display.width", 240); pd.set_option("display.max_rows", 200)
H = {"User-Agent": "stock_scan research fjresearch@gmail.com", "Accept-Encoding": "gzip, deflate"}
j = json.load(open("/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/_fd_AAPL_submissions.json"))
frames = [pd.DataFrame(j["filings"]["recent"])]
for f in j["filings"].get("files", []):
    r = requests.get("https://data.sec.gov/submissions/" + f["name"], headers=H, timeout=30)
    if r.status_code == 200:
        frames.append(pd.DataFrame(r.json())); time.sleep(0.2)
    else:
        print("older file fail", f["name"], r.status_code)
d = pd.concat(frames, ignore_index=True)
d["filingDate"] = pd.to_datetime(d.filingDate)
er = d[(d.form == "8-K") & d["items"].fillna("").str.contains("2.02")].sort_values("filingDate")
er = er[er.filingDate >= "2016-01-01"]
print(f"earnings 8-K (item 2.02) events found: {len(er)}  {er.filingDate.min().date()} .. {er.filingDate.max().date()}")
print(er[["filingDate","reportDate","accessionNumber","items"]].tail(16).to_string(index=False))

r = pd.read_pickle("/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/_fd_AAPL_rets.pkl")
idx = r.index

def study(dates, label, bench=("SPY","XLK")):
    rows = []
    for dt in dates:
        loc = idx.searchsorted(pd.Timestamp(dt), side="right")   # first trading day AFTER filing
        if loc >= len(idx) - 1: continue
        rows.append((idx[loc], dt))
    out = []
    for react, evt in rows:
        i = idx.get_loc(react)
        rec = {"filed": pd.Timestamp(evt).date(), "react_day": react.date()}
        rec["AAPL_d0"] = r["AAPL"].iloc[i]
        for b in bench:
            rec[f"AR_d0_{b}"] = r["AAPL"].iloc[i] - r[b].iloc[i]
        for lo, hi, nm in [(-5, 5, "m5p5"), (-1, 20, "m1p20"), (0, 0, "d0"), (1, 20, "p1p20")]:
            a, z = max(0, i+lo), min(len(idx)-1, i+hi)
            rec[f"raw_{nm}"] = r["AAPL"].iloc[a:z+1].sum()
            rec[f"AR_{nm}_SPY"] = (r["AAPL"] - r["SPY"]).iloc[a:z+1].sum()
            rec[f"AR_{nm}_XLK"] = (r["AAPL"] - r["XLK"]).iloc[a:z+1].sum()
        out.append(rec)
    o = pd.DataFrame(out)
    print(f"\n{'='*130}\n{label}  n={len(o)}\n{'='*130}")
    cols = ["filed","react_day","AAPL_d0","AR_d0_SPY","AR_d0_XLK","AR_m5p5_SPY","AR_m1p20_SPY","AR_p1p20_SPY","AR_m1p20_XLK"]
    show = o[cols].copy()
    for c in cols[2:]:
        show[c] = (show[c]*100).round(2)
    print(show.to_string(index=False))
    print("\n--- summary (log %, mean / median / t-stat / hit rate>0) ---")
    for c in ["AAPL_d0","AR_d0_SPY","AR_d0_XLK","AR_m5p5_SPY","AR_m1p20_SPY","AR_p1p20_SPY","AR_m1p20_XLK"]:
        v = o[c].dropna().values*100
        t = v.mean()/(v.std(ddof=1)/np.sqrt(len(v))) if len(v) > 1 else np.nan
        print(f"  {c:<14} n={len(v):>3}  mean {v.mean():>+7.2f}  median {np.median(v):>+7.2f}  t {t:>+6.2f}  hit {np.mean(v>0)*100:>5.1f}%")
    return o

o = study(er.filingDate.tolist(), "AAPL earnings reaction (8-K item 2.02, primary dates)")
o.to_csv("/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/_fd_AAPL_event_earnings.csv", index=False)

# how much of the last year's return happened on the 4 earnings reaction days?
last = r.loc["2025-09-12":]
er_react = set()
for dt in er.filingDate:
    loc = idx.searchsorted(pd.Timestamp(dt), side="right")
    if loc < len(idx): er_react.add(idx[loc])
ev = last.index.isin(er_react)
print(f"\n{'='*130}\nRETURN DECOMPOSITION, last ~1 year ({last.index.min().date()} .. {last.index.max().date()})\n{'='*130}")
print(f"  total AAPL log return          {last['AAPL'].sum()*100:>+7.2f}%   (simple {np.expm1(last['AAPL'].sum())*100:>+.1f}%)")
print(f"  on {ev.sum()} earnings reaction days  {last['AAPL'][ev].sum()*100:>+7.2f}%   dates {[d.date() for d in last.index[ev]]}")
print(f"  on the other {len(last)-ev.sum()} days       {last['AAPL'][~ev].sum()*100:>+7.2f}%")
print(f"  SPY same window                {last['SPY'].sum()*100:>+7.2f}%     XLK {last['XLK'].sum()*100:>+7.2f}%")
print(f"  AAPL - SPY                     {(last['AAPL']-last['SPY']).sum()*100:>+7.2f}%   of which on earnings days {(last['AAPL']-last['SPY'])[ev].sum()*100:>+7.2f}%")

# alpha with beta estimated OUT OF SAMPLE (2023-09..2025-09) then applied forward
est = r.loc["2023-09-14":"2025-09-11"]
X = np.column_stack([np.ones(len(est)), est["SPY"].values])
b, *_ = np.linalg.lstsq(X, est["AAPL"].values, rcond=None)
print(f"\n  out-of-sample market model estimated 2023-09..2025-09: alpha/day {b[0]*100:.4f}%  beta {b[1]:.3f}")
for lab, w in [("last 1y", r.loc["2025-09-12":]), ("last 3m", r.loc["2026-06-12":])]:
    pred = b[0] + b[1]*w["SPY"].values
    ab = w["AAPL"].values - pred
    print(f"  {lab}: actual {w['AAPL'].sum()*100:>+7.2f}%   predicted-by-beta {pred.sum()*100:>+7.2f}%   ABNORMAL {ab.sum()*100:>+7.2f}%")
