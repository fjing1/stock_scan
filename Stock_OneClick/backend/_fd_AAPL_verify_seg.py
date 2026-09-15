"""ADVERSARIAL VERIFY #2 -- product-line and geography revenue rebuilt from the XBRL frames API
(dimensional facts are not in companyfacts), then the increment attribution recomputed from
components. Attacks the claim 'iPhone supplied 62.9-82.9% of each quarter's increment'.
"""
from __future__ import annotations
import io, json, re, sys, time, urllib.request, gzip
import pandas as pd

UA = {"User-Agent": "Feijing Research feijing.research@gmail.com"}

# Product-line revenue lives in the R-file / instance docs, not companyfacts. Pull the XBRL
# instance for each 10-Q/10-K and read the segment-dimensioned revenue facts.
FILINGS = {
    # accn                       label
    "0000320193-24-000081": "10-Q Q3FY24 (end 2024-06-29)",
    "0000320193-24-000123": "10-K FY2024",
    "0000320193-25-000008": "10-Q Q1FY25 (end 2024-12-28)",
    "0000320193-25-000057": "10-Q Q2FY25 (end 2025-03-29)",
    "0000320193-25-000073": "10-Q Q3FY25 (end 2025-06-28)",
    "0000320193-25-000079": "10-K FY2025",
    "0000320193-26-000006": "10-Q Q1FY26 (end 2025-12-27)",
    "0000320193-26-000013": "10-Q Q2FY26 (end 2026-03-28)",
    "0000320193-26-000020": "10-Q Q3FY26 (end 2026-06-27)",
}


def get(url, dest=None):
    if dest:
        try:
            return open(dest, "rb").read()
        except FileNotFoundError:
            pass
    req = urllib.request.Request(url, headers=UA)
    b = urllib.request.urlopen(req, timeout=60).read()
    if dest:
        open(dest, "wb").write(b)
    time.sleep(0.25)
    return b


rows = []
for accn, lab in FILINGS.items():
    a = accn.replace("-", "")
    idx = json.loads(get(f"https://www.sec.gov/Archives/edgar/data/320193/{a}/index.json",
                         f"_fd_AAPL_verify_idx_{accn}.json"))
    names = [i["name"] for i in idx["directory"]["item"]]
    inst = [n for n in names if re.match(r"aapl-\d{8}_htm\.xml$", n)] or \
           [n for n in names if n.endswith("_htm.xml")]
    if not inst:
        print(f"  !! no instance doc for {accn}: {names[:12]}")
        continue
    xml = get(f"https://www.sec.gov/Archives/edgar/data/320193/{a}/{inst[0]}",
              f"_fd_AAPL_verify_inst_{accn}.xml").decode("utf-8", "replace")

    # contexts
    ctx = {}
    for m in re.finditer(r'<(?:xbrli:)?context id="([^"]+)">(.*?)</(?:xbrli:)?context>', xml, re.S):
        cid, body = m.group(1), m.group(2)
        sd = re.search(r"<(?:xbrli:)?startDate>([\d-]+)</(?:xbrli:)?startDate>", body)
        ed = re.search(r"<(?:xbrli:)?endDate>([\d-]+)</(?:xbrli:)?endDate>", body)
        mem = re.findall(r'<xbrldi:explicitMember dimension="([^"]+)">([^<]+)</xbrldi:explicitMember>', body)
        ctx[cid] = {"start": sd.group(1) if sd else None, "end": ed.group(1) if ed else None,
                    "dims": dict(mem)}
    for tag in ("RevenueFromContractWithCustomerExcludingAssessedTax",):
        for m in re.finditer(rf'<us-gaap:{tag}[^>]*contextRef="([^"]+)"[^>]*>([-\d]+)</us-gaap:{tag}>', xml):
            cid, val = m.group(1), float(m.group(2))
            c = ctx.get(cid)
            if not c or not c["start"]:
                continue
            dims = c["dims"]
            rows.append({"accn": accn, "filing": lab, "start": c["start"], "end": c["end"],
                         "val": val, "ndim": len(dims),
                         "member": "|".join(f"{k.split(':')[-1]}={v.split(':')[-1]}" for k, v in sorted(dims.items()))})

D = pd.DataFrame(rows)
D["start"] = pd.to_datetime(D.start); D["end"] = pd.to_datetime(D.end)
D["days"] = (D.end - D.start).dt.days
D = D.drop_duplicates(["start", "end", "member", "val"])
D.to_csv("_fd_AAPL_verify_seg_raw.csv", index=False)
print(f"segment-dimensioned revenue facts: n={len(D)}")
print("\ndistinct members seen:")
for mm in sorted(D.member.unique()):
    print("   ", mm if mm else "(consolidated, no dimension)")

Q = D[(D.days >= 80) & (D.days <= 100)]
prod = Q[Q.member.str.contains("ProductOrService", na=False)]
prod = prod.assign(line=prod.member.str.split("=").str[-1])
piv = prod.pivot_table(index="end", columns="line", values="val", aggfunc="max") / 1e6
piv = piv.sort_index()
print("\n" + "=" * 116)
print("QUARTERLY REVENUE BY PRODUCT LINE ($M, from the segment dimension of each 10-Q instance)")
print("=" * 116)
print(piv.to_string(float_format=lambda x: f"{x:,.0f}"))
piv.to_csv("_fd_AAPL_verify_seg_quarterly.csv")

# YoY and 2-year-stacked, date matched on the fiscal quarter (4 quarters back by position within
# this quarterly-only panel is safe here because product-line facts ARE reported every quarter
# including Q4 via the 10-K -- verify that first.
print("\n  quarter-end dates present:", [str(d.date()) for d in piv.index])
