"""MEASURE: of the exact symbols this repo requested but got ZERO bars for, how many does
Alpha Vantage LISTING_STATUS recover? Uses the already-downloaded CSVs -- no new API calls."""
import pickle
import sys
from pathlib import Path

import pandas as pd

HERE = Path("/Users/feijing/github.com/stock_scan/Stock_OneClick/backend")
sys.path.insert(0, "/Users/feijing/github.com/stock_scan")

# --- reproduce _move_data.universe() exactly ---
import stock_symbols_1243 as S

INDICES = ["SPY", "QQQ", "IWM", "DIA", "^GSPC", "^VIX"]
N_PER_SECTOR = 30
buckets = ["TECH_STOCKS", "HEALTHCARE_STOCKS", "FINANCIAL_STOCKS", "CONSUMER_DISCRETIONARY",
           "CONSUMER_STAPLES", "ENERGY_STOCKS", "MATERIALS_INDUSTRIALS", "UTILITIES",
           "REAL_ESTATE_REITS", "COMMUNICATION_SERVICES"]
picked = []
for b in buckets:
    names = [s for s in getattr(S, b, []) if "." not in s]
    stride = max(1, len(names) // N_PER_SECTOR)
    picked += names[::stride][:N_PER_SECTOR]
requested = list(dict.fromkeys(INDICES + picked))

panel = pickle.load(open(HERE / "_move_panel.pkl", "rb"))
got = list(panel["Close"].columns)
missing = [s for s in requested if s not in got]
print(f"requested={len(requested)}  in panel={len(got)}  ZERO-BAR CASUALTIES={len(missing)}")
print(f"missing = {missing}")

act = pd.read_csv(HERE / "_data_probe_surv_av_active_now.csv")
dl = pd.read_csv(HERE / "_data_probe_surv_av_delisted_all.csv")
a15 = pd.read_csv(HERE / "_data_probe_surv_av_active_2015-01-02.csv")
print(f"\nAV active_now={len(act):,}  AV delisted={len(dl):,}  AV active@2015-01-02={len(a15):,}")

dmap = dl.set_index("symbol")[["name", "exchange", "ipoDate", "delistingDate"]].to_dict("index")
a15set = set(a15["symbol"])
actset = set(act["symbol"])

rec_dl, rec_a15, none = [], [], []
print(f"\n{'sym':8s} {'in AV delisted?':16s} {'delistDate':12s} {'active@2015?':13s} name")
for m in sorted(missing):
    ind = dmap.get(m)
    in15 = m in a15set
    if ind:
        rec_dl.append(m)
        print(f"{m:8s} {'YES':16s} {str(ind['delistingDate']):12s} {str(in15):13s} "
              f"{str(ind['name'])[:40]}")
    elif in15:
        rec_a15.append(m)
        print(f"{m:8s} {'no':16s} {'-':12s} {'YES':13s} (only in dated-active roster)")
    else:
        none.append(m)
        print(f"{m:8s} {'no':16s} {'-':12s} {'no':13s} NOT IN AV AT ALL")

print(f"\nRECOVERED via AV delisted roster : {len(rec_dl)}/{len(missing)}  {rec_dl}")
print(f"RECOVERED only via dated-active   : {len(rec_a15)}/{len(missing)}  {rec_a15}")
print(f"NOT IN ALPHA VANTAGE AT ALL       : {len(none)}/{len(missing)}  {none}")
cov = (len(rec_dl) + len(rec_a15)) / max(1, len(missing))
print(f"\nAV roster coverage of this repo's casualties = {cov:.1%}")

# how much of the full 1,283-name repo universe does AV know?
allsyms = set()
for n in dir(S):
    v = getattr(S, n)
    if isinstance(v, (list, tuple)) and v and isinstance(v[0], str):
        allsyms |= set(v)
allsyms = {s for s in allsyms if "." not in s}
print(f"\nrepo universe (dot-free) = {len(allsyms):,}")
print(f"  known to AV active_now  = {len(allsyms & actset):,}")
print(f"  known to AV delisted    = {len(allsyms & set(dl['symbol'])):,}")
print(f"  unknown to AV entirely  = {len(allsyms - actset - set(dl['symbol'])):,}")

# What does the delisted roster look like as a research asset?
dl["dyear"] = dl["delistingDate"].str[:4]
print("\nAV delisted-roster count by delisting year:")
vc = dl["dyear"].value_counts().sort_index()
print(vc.to_string())
print(f"\nassetType breakdown of AV delisted: "
      f"{dl['assetType'].value_counts().to_dict()}")
print(f"exchange breakdown of AV delisted: {dl['exchange'].value_counts().to_dict()}")

# Point-in-time roster arithmetic
print("\n--- point-in-time universe arithmetic ---")
n_now, n_15 = len(actset), len(a15set)
gone = a15set - actset
print(f"active 2026-09 = {n_now:,};  active 2015-01-02 = {n_15:,}")
print(f"names listed in 2015 but NOT listed now = {len(gone):,} "
      f"({len(gone)/n_15:.1%} of the 2015 roster)")
print(f"  of those, AV delisted roster has a delisting date for "
      f"{len(gone & set(dl['symbol'])):,}")
print(f"  sample: {sorted(gone)[:20]}")
