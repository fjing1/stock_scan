"""Extract AAPL revenue BY PRODUCT CATEGORY and BY GEOGRAPHIC SEGMENT from the raw XBRL
instance documents of each 10-Q / 10-K. companyfacts drops dimensional facts, so this is the
only primary route. Saves each instance to disk for audit."""
import json, os, re, sys, time
import requests, pandas as pd
from lxml import etree

HERE = "/Users/feijing/github.com/stock_scan/Stock_OneClick/backend"
UA = "stock_scan research fjresearch@gmail.com"
H = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}
CIK = "320193"

ACCNS = ["0000320193-26-000020","0000320193-26-000013","0000320193-26-000006",
         "0000320193-25-000079","0000320193-25-000073","0000320193-25-000057",
         "0000320193-25-000008","0000320193-24-000123","0000320193-24-000081",
         "0000320193-24-000069","0000320193-24-000006","0000320193-23-000106",
         "0000320193-23-000077","0000320193-22-000108","0000320193-21-000105",
         "0000320193-20-000096","0000320193-19-000119"]

def get(url, binary=False):
    for _ in range(3):
        r = requests.get(url, headers=H, timeout=60)
        if r.status_code == 200:
            return r.content if binary else r.text
        time.sleep(1.0)
    print("FAIL", url, r.status_code); return None

rows = []
for accn in ACCNS:
    nod = accn.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{CIK}/{nod}"
    idx = get(base + "/index.json")
    time.sleep(0.25)
    if idx is None: continue
    names = [i["name"] for i in json.loads(idx)["directory"]["item"]]
    inst = [n for n in names if re.match(r"aapl-\d{8}_htm\.xml$", n)]
    if not inst:
        inst = [n for n in names if n.endswith("_htm.xml")]
    if not inst:
        print(accn, "no instance;", names[:12]); continue
    fn = inst[0]
    local = os.path.join(HERE, f"_fd_AAPL_inst_{accn}_{fn}")
    if os.path.exists(local):
        raw = open(local, "rb").read()
    else:
        raw = get(f"{base}/{fn}", binary=True); time.sleep(0.25)
        if raw is None: continue
        open(local, "wb").write(raw)
    root = etree.fromstring(raw)
    ns = root.nsmap
    # contexts
    ctx = {}
    for c in root.iter("{http://www.xbrl.org/2003/instance}context"):
        cid = c.get("id")
        per = c.find("{http://www.xbrl.org/2003/instance}period")
        sd = per.findtext("{http://www.xbrl.org/2003/instance}startDate")
        ed = per.findtext("{http://www.xbrl.org/2003/instance}endDate")
        inst_d = per.findtext("{http://www.xbrl.org/2003/instance}instant")
        dims = {}
        for m in c.iter("{http://xbrl.org/2006/xbrldi}explicitMember"):
            dims[m.get("dimension").split(":")[-1]] = (m.text or "").strip().split(":")[-1]
        ctx[cid] = {"start": sd, "end": ed or inst_d, "dims": dims}
    TAGS = {"{http://fasb.org/us-gaap/2025}RevenueFromContractWithCustomerExcludingAssessedTax"}
    for el in root.iter():
        tg = el.tag
        if not isinstance(tg, str) or "}" not in tg: continue
        local_name = tg.split("}")[1]
        if local_name != "RevenueFromContractWithCustomerExcludingAssessedTax": continue
        cid = el.get("contextRef")
        if cid not in ctx: continue
        c = ctx[cid]
        if not c["start"]: continue
        try: val = float(el.text)
        except (TypeError, ValueError): continue
        rows.append({"accn": accn, "start": c["start"], "end": c["end"],
                     "dims": json.dumps(c["dims"], sort_keys=True), "val": val,
                     "ndims": len(c["dims"]),
                     "axis": ";".join(sorted(c["dims"].keys())),
                     "member": ";".join(c["dims"][k] for k in sorted(c["dims"]))})
    print(f"{accn} {fn}: {len([r for r in rows if r['accn']==accn])} revenue facts")

d = pd.DataFrame(rows)
d["start"] = pd.to_datetime(d.start); d["end"] = pd.to_datetime(d.end)
d["days"] = (d.end - d.start).dt.days
d.to_pickle(os.path.join(HERE, "_fd_AAPL_segments.pkl"))
print("\naxes seen:"); print(d.axis.value_counts())
print("\nmembers:"); print(d[d.ndims==1].groupby(["axis","member"]).size())
