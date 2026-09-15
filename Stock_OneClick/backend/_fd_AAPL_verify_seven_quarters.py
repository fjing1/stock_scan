"""Two remaining checks:
 (1) Q4FY25 'Total net sales' printed in the freshly fetched EX-99.1 (the footnote marker '(1)'
     broke the earlier regex) -- must equal 102,466.
 (2) The claim's stated cross-check: FY2025 iPhone (10-K) minus 9M-FY2025 iPhone (three 10-Qs /
     press releases) == 49,025 as filed in the Q4FY25 release.
 (3) Spot-check the claim's 7-quarter table against freshly fetched EDGAR text for every quarter.
"""
from __future__ import annotations
import os, re, json, requests
import pandas as pd

UA = {"User-Agent": "stock_scan research fj.research.contact@gmail.com",
      "Accept-Encoding": "gzip, deflate"}

ACCS = {  # tag -> accession of the 8-K carrying EX-99.1
    "Q1FY25": "0000320193-25-000007", "Q2FY25": "0000320193-25-000055",
    "Q3FY25": "0000320193-25-000071", "Q4FY25": "0000320193-25-000077",
    "Q1FY26": "0000320193-26-000005", "Q2FY26": "0000320193-26-000011",
    "Q3FY26": "0000320193-26-000018",
}


def get(url, path):
    if os.path.exists(path):
        return open(path, encoding="utf-8", errors="replace").read()
    r = requests.get(url, headers=UA, timeout=90)
    r.raise_for_status()
    open(path, "wb").write(r.content)
    print(f"  GET {os.path.basename(url)} -> {r.status_code}")
    return r.text


def strip(html):
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    t = re.sub(r"(?is)<br[^>]*>|</t[dhr]>|</p>|</div>", " ", t)
    t = re.sub(r"(?s)<[^>]+>", "", t)
    t = t.replace("&nbsp;", " ").replace("&#160;", " ").replace("&amp;", "&")
    return re.sub(r"\s+", " ", t)


def firstnum(body, label):
    """number immediately after `label`, ignoring '(1)'/'(2)' footnote markers"""
    b = body.replace("(1)", " ").replace("(2)", " ").replace("(3)", " ")
    i = b.find(label)
    if i < 0:
        return None
    m = re.search(r"\(?\$?([\d,]+(?:\.\d+)?)\)?", b[i + len(label): i + len(label) + 80])
    return float(m.group(1).replace(",", "")) if m else None


rows = []
for tag, acc in ACCS.items():
    a = acc.replace("-", "")
    idx = json.loads(get(f"https://www.sec.gov/Archives/edgar/data/320193/{a}/index.json",
                         f"_fd_AAPL_verify_idx_{acc}.json"))
    docs = [it["name"] for it in idx["directory"]["item"]
            if it["name"].lower().endswith(".htm") and re.search(r"ex.?99", it["name"], re.I)]
    doc = docs[0]
    txt = strip(get(f"https://www.sec.gov/Archives/edgar/data/320193/{a}/{doc}",
                    f"_fd_AAPL_verify_{tag}_ex991.htm"))
    i = txt.find("CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS")
    body = txt[i:i + 2600]
    sales = body.split("Cost of sales:")[0]
    # product table (Segment/product performance) -- take the first current-quarter column
    prod = {}
    for p in ("iPhone", "Mac", "iPad", "Services"):
        m = re.search(r"\b" + p + r"\b[^\d]{0,40}([\d,]{5,})", txt)
        prod[p] = float(m.group(1).replace(",", "")) if m else None
    rows.append(dict(tag=tag, doc=doc,
                     rev=firstnum(sales, "Total net sales"),
                     eps=firstnum(body[body.find("Shares used") - 400: body.find("Shares used")], "Diluted"),
                     ni=firstnum(body, "Net income"), **prod))

R = pd.DataFrame(rows).set_index("tag")
print("\n" + "=" * 130)
print("FRESHLY FETCHED FROM EDGAR (not the claim's local files)")
print("=" * 130)
print(R.to_string(float_format=lambda x: f"{x:,.2f}"))

loc = pd.read_csv("_fd_AAPL_rev_quarterly.csv").set_index("tag")
print("\n  claim's parsed rev vs fresh EDGAR rev, per quarter:")
for t in R.index:
    a, b = float(loc.loc[t, "rev"]), R.loc[t, "rev"]
    print(f"    {t}  claim {a:>12,.0f}   edgar {b:>12,.0f}   diff {a-b:+,.0f}")
print(f"\n  claim's parsed EPS vs fresh EDGAR EPS:")
for t in R.index:
    a, b = float(loc.loc[t, "eps"]), R.loc[t, "eps"]
    print(f"    {t}  claim {a:>6.2f}   edgar {b if b is None else f'{b:6.2f}'}   diff "
          f"{'n/a' if b is None else f'{a-b:+.2f}'}")

ttm_tags = ["Q4FY25", "Q1FY26", "Q2FY26", "Q3FY26"]
print(f"\n  TTM revenue from freshly fetched EDGAR docs: {R.loc[ttm_tags,'rev'].sum():,.0f} $M"
      f"   (claim 466,823)")
print(f"  TTM diluted EPS from freshly fetched EDGAR docs: {R.loc[ttm_tags,'eps'].sum():.2f}"
      f"   (claim 8.72)  -> P/E {333.08/R.loc[ttm_tags,'eps'].sum():.2f}x")
print(f"  FY2025 = sum of the 4 FY25 quarters: {R.loc[['Q1FY25','Q2FY25','Q3FY25','Q4FY25'],'rev'].sum():,.0f}"
      f"   (10-K FY2025 revenue 416,161)")

print("\n" + "=" * 130)
print("iPhone FY-minus-9M cross-check")
print("=" * 130)
nine = R.loc[["Q1FY25", "Q2FY25", "Q3FY25"], "iPhone"].sum()
print(f"  9M FY2025 iPhone from the three quarterly releases: {nine:,.0f}")
# FY2025 iPhone from the 10-K text on disk
kt = open("_fd_AAPL_10K_FY2025.txt", encoding="utf-8", errors="replace").read().replace("\n", " ")
hits = re.findall(r"iPhone[^\d]{0,60}([\d,]{6,})", kt)[:6]
print(f"  candidate FY2025 iPhone figures in the 10-K text: {hits}")
for h in hits[:6]:
    v = float(h.replace(",", ""))
    if 200000 < v < 300000:
        print(f"  -> FY2025 iPhone {v:,.0f} ; derived Q4FY25 iPhone {v-nine:,.0f} ; "
              f"filed Q4FY25 release {R.loc['Q4FY25','iPhone']:,.0f} ; "
              f"residual {v-nine-R.loc['Q4FY25','iPhone']:+,.0f}")
        break
