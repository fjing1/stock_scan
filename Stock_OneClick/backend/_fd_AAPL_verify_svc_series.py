"""Build AAPL Services + iPhone quarterly revenue series from PRIMARY 10-Q/10-K XBRL
instances, date-matched YoY. Tests the 'Services is decelerating' sub-claim against
sample size, and checks the Services-share-of-gross-margin arithmetic.
"""
import json, os, time, collections
from datetime import date
from lxml import etree
import urllib.request

UA = "Feijing Research feijing.research@gmail.com"
SUB = json.load(open("_fd_AAPL_verify_submissions.json"))

def get(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req) as r:
        open(path, "wb").write(r.read())
    time.sleep(0.25)
    return path

# collect 10-Q / 10-K accessions (recent + older files)
rows = []
r = SUB["filings"]["recent"]
for f, acc, rd in zip(r["form"], r["accessionNumber"], r["reportDate"]):
    if f in ("10-Q", "10-K"):
        rows.append((f, acc, rd))
for extra in SUB["filings"].get("files", []):
    p = get("https://data.sec.gov/submissions/" + extra["name"],
            "_fd_AAPL_verify_" + extra["name"])
    d = json.load(open(p))
    for f, acc, rd in zip(d["form"], d["accessionNumber"], d["reportDate"]):
        if f in ("10-Q", "10-K"):
            rows.append((f, acc, rd))
rows = sorted(set(rows), key=lambda x: x[2], reverse=True)
rows = [x for x in rows if x[2] >= "2018-01-01"]
print(f"{len(rows)} 10-Q/10-K filings since 2018")

REV = "RevenueFromContractWithCustomerExcludingAssessedTax"
PROD_AXIS = "srt:ProductOrServiceAxis"
MEMBERS = {"us-gaap:ServiceMember": "Services", "us-gaap:ProductMember": "Products",
           "aapl:IPhoneMember": "iPhone", "aapl:MacMember": "Mac",
           "aapl:IPadMember": "iPad",
           "aapl:WearablesHomeandAccessoriesMember": "Wearables"}
# also older axis spelling
ALT_AXES = {PROD_AXIS, "us-gaap:ProductOrServiceAxis"}

# facts[(member, start, end)] -> value ; keep newest filing's value
facts = {}
GM = {}  # gross margin facts by (concept, member, start, end)

for form, acc, rd in rows:
    a = acc.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/320193/{a}"
    stem = f"aapl-{rd.replace('-','')}"
    inst = f"{base}/{stem}_htm.xml"
    path = f"_fd_AAPL_verify_inst_{rd}.xml"
    try:
        get(inst, path)
        tree = etree.parse(path)
    except Exception as e:
        print(f"  skip {form} {rd}: {e}")
        continue
    root = tree.getroot()
    ctx = {}
    for c in root.iter("{http://www.xbrl.org/2003/instance}context"):
        per = c.find("{http://www.xbrl.org/2003/instance}period")
        s = per.find("{http://www.xbrl.org/2003/instance}startDate") if per is not None else None
        e_ = per.find("{http://www.xbrl.org/2003/instance}endDate") if per is not None else None
        dims = {}
        for m in c.iter("{http://xbrl.org/2006/xbrldi}explicitMember"):
            dims[m.get("dimension")] = (m.text or "").strip()
        ctx[c.get("id")] = (s.text if s is not None else None,
                            e_.text if e_ is not None else None, dims)
    for el in root:
        q = etree.QName(el)
        if q.localname not in (REV, "GrossProfit"):
            continue
        cr = el.get("contextRef")
        if cr not in ctx:
            continue
        s, e_, dims = ctx[cr]
        if not s:
            continue
        txt = (el.text or "").strip().replace(",", "")
        try:
            v = float(txt)
        except ValueError:
            continue
        if el.get("sign") == "-":
            v = -v
        ax = set(dims) & ALT_AXES
        if len(dims) == 0:
            mem = "TOTAL"
        elif len(dims) == 1 and ax:
            mem = MEMBERS.get(dims[list(ax)[0]])
            if mem is None:
                continue
        else:
            continue
        key = (q.localname, mem, s, e_)
        facts.setdefault(key, v)

def days(s, e):
    d0 = date(*map(int, s.split("-"))); d1 = date(*map(int, e.split("-")))
    return (d1 - d0).days + 1

# --- assemble quarterly (~91d) series per member ---
print("\n==== Services revenue: date-matched YoY, quarterly (duration 80-100d) ====")
ser = collections.defaultdict(dict)
for (cn, mem, s, e_), v in facts.items():
    if cn != REV:
        continue
    n = days(s, e_)
    if 80 <= n <= 100:
        ser[mem][e_] = (v, s)
    elif 350 <= n <= 380:
        ser[mem + "_FY"][e_] = (v, s)
    elif 260 <= n <= 285:
        ser[mem + "_9M"][e_] = (v, s)

def yoy_table(mem):
    ends = sorted(ser[mem])
    out = []
    for e_ in ends:
        v, s = ser[mem][e_]
        # find prior-year quarter: end date ~365d earlier (+-10d)
        target = date(*map(int, e_.split("-")))
        best = None
        for e2 in ends:
            d2 = date(*map(int, e2.split("-")))
            gap = (target - d2).days
            if 355 <= gap <= 375:
                best = e2
        if best:
            pv = ser[mem][best][0]
            out.append((e_, v, best, pv, v / pv - 1))
    return out

for mem in ("Services", "iPhone", "TOTAL"):
    print(f"\n-- {mem} --")
    print(f"{'qtr end':<12}{'rev $M':>10}{'prior end':<13}{'prior $M':>10}{'YoY':>9}")
    for e_, v, pe, pv, g in yoy_table(mem):
        print(f"{e_:<12}{v/1e6:>10,.0f}{pe:<13}{pv/1e6:>10,.0f}{g*100:>8.1f}%")

# --- FY (annual) Services growth, incl FY2025 ---
print("\n==== Services ANNUAL (FY) revenue, date-matched YoY ====")
ends = sorted(ser["Services_FY"])
for i, e_ in enumerate(ends):
    v = ser["Services_FY"][e_][0]
    prev = None
    for e2 in ends:
        d = (date(*map(int, e_.split("-"))) - date(*map(int, e2.split("-")))).days
        if 355 <= d <= 375:
            prev = e2
    g = (v / ser["Services_FY"][prev][0] - 1) if prev else None
    print(f"  FY end {e_}  Services {v/1e6:>10,.0f}   YoY {('%+.1f%%'%(g*100)) if g else 'n/a'}")

print("\n==== TOTAL ANNUAL ====")
ends = sorted(ser["TOTAL_FY"])
for e_ in ends:
    v = ser["TOTAL_FY"][e_][0]
    prev = None
    for e2 in ends:
        d = (date(*map(int, e_.split("-"))) - date(*map(int, e2.split("-")))).days
        if 355 <= d <= 375:
            prev = e2
    g = (v / ser["TOTAL_FY"][prev][0] - 1) if prev else None
    print(f"  FY end {e_}  Total {v/1e6:>10,.0f}   YoY {('%+.1f%%'%(g*100)) if g else 'n/a'}")

# --- H1 FY26 services check ---
s9 = ser["Services_9M"]
sq = ser["Services"]
print("\n==== H1 derivation check (9M minus Q3) ====")
for e9, (v9, s9s) in sorted(s9.items()):
    if e9 in sq:
        vq = sq[e9][0]
        print(f"  9M end {e9}: 9M={v9/1e6:,.0f}  Q3={vq/1e6:,.0f}  H1={(v9-vq)/1e6:,.0f}")

# --- Services share of gross margin, with/without tariff refund ---
print("\n==== Services share of GROSS MARGIN, Q3FY26 vs Q3FY25 ====")
gm = {(m, s, e_): v for (cn, m, s, e_), v in facts.items() if cn == "GrossProfit"}
for lbl, (s, e_) in (("Q3FY26", ("2026-03-29", "2026-06-27")),
                     ("Q3FY25", ("2025-03-30", "2025-06-28"))):
    tot = gm.get(("TOTAL", s, e_))
    svc = gm.get(("Services", s, e_))
    prd = gm.get(("Products", s, e_))
    print(f"  {lbl}: total GM {tot and tot/1e6:,.0f}  Services GM {svc and svc/1e6:,.0f} "
          f" Products GM {prd and prd/1e6:,.0f}  Services share "
          f"{(svc/tot*100) if (svc and tot) else float('nan'):.2f}%")

# ex-tariff adjustment (press release: ~2pp of company gross margin)
rev26 = 109417e6; gm26 = 54770e6; svc26 = 23245e6
for pp in (1.5, 2.0, 2.5):
    refund = rev26 * pp / 100
    print(f"  ex-refund @{pp}pp (=${refund/1e6:,.0f}M): total GM {(gm26-refund)/1e6:,.0f} "
          f" Services share {svc26/(gm26-refund)*100:.2f}%  vs 47.40% prior "
          f"=> {svc26/(gm26-refund)*100-47.40:+.2f}pp (reported {23245/54770*100-47.40:+.2f}pp)")
