"""Independent EDGAR pull of the Q4FY25 and Q3FY26 8-K EX-99.1, to check the claim's local
_fd_AAPL_ir_*.txt files were not mangled/fabricated, and to check the iPhone cross-check.
Also checks dei shares-outstanding vs the weighted-average diluted the claim used for market cap.
"""
from __future__ import annotations
import json, os, re, requests
import pandas as pd

UA = {"User-Agent": "stock_scan research fj.research.contact@gmail.com",
      "Accept-Encoding": "gzip, deflate"}
CIK = "0000320193"


def get(url, path=None, binary=False):
    if path and os.path.exists(path):
        return open(path, "rb").read() if binary else open(path, encoding="utf-8", errors="replace").read()
    r = requests.get(url, headers=UA, timeout=90)
    print(f"  GET {url} -> {r.status_code} {len(r.content)}B")
    r.raise_for_status()
    if path:
        open(path, "wb").write(r.content)
    return r.content if binary else r.text


def strip(html):
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    t = re.sub(r"(?is)<br[^>]*>|</t[dhr]>|</p>|</div>", " ", t)
    t = re.sub(r"(?s)<[^>]+>", "", t)
    t = (t.replace("&nbsp;", " ").replace("&#160;", " ").replace("&amp;", "&")
          .replace("&#8212;", "-").replace("&mdash;", "-").replace("&#8217;", "'"))
    return re.sub(r"[ \t]+", " ", t)


print("=" * 110)
print("A. Confirm the accessions the claim cites are real and are 8-Ks with an EX-99.1")
print("=" * 110)
for acc in ("0000320193-25-000077", "0000320193-26-000018"):
    a = acc.replace("-", "")
    idx = json.loads(get(f"https://www.sec.gov/Archives/edgar/data/320193/{a}/index.json",
                         f"_fd_AAPL_verify_idx_{acc}.json"))
    items = idx["directory"]["item"]
    print(f"\n  {acc}: {len(items)} files")
    for it in items:
        if re.search(r"(ex|a)99|\.htm$", it["name"], re.I) and "R" not in it["name"][:1]:
            print(f"     {it['name']:<42} {it['size']:>10}  {it.get('last-modified','')}")

print("\n" + "=" * 110)
print("B. Re-fetch Q4FY25 EX-99.1 fresh from EDGAR and re-parse revenue / EPS / iPhone")
print("=" * 110)
# find the EX-99.1 doc name from the filing index
for acc, tag in (("0000320193-25-000077", "q4fy25"), ("0000320193-26-000018", "q3fy26")):
    a = acc.replace("-", "")
    idx = json.loads(open(f"_fd_AAPL_verify_idx_{acc}.json").read())
    cands = [it["name"] for it in idx["directory"]["item"]
             if it["name"].lower().endswith(".htm") and re.search(r"ex.?99|a\d{6}exhibit99", it["name"], re.I)]
    if not cands:
        cands = [it["name"] for it in idx["directory"]["item"]
                 if it["name"].lower().endswith(".htm") and int(it["size"]) > 50000]
    print(f"\n  {acc} EX-99.1 candidates: {cands}")
    doc = cands[0]
    html = get(f"https://www.sec.gov/Archives/edgar/data/320193/{a}/{doc}",
               f"_fd_AAPL_verify_{tag}_ex991.htm")
    txt = strip(html)
    open(f"_fd_AAPL_verify_{tag}_ex991.txt", "w").write(txt)
    flat = txt.replace("\n", " ")
    i = flat.find("CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS")
    body = flat[i:i + 3000]
    for label in ("Total net sales", "Diluted", "Net income"):
        m = re.search(re.escape(label) + r"[^\d(]{0,40}(\(?\$?[\d,]+(?:\.\d+)?\)?)", body)
        print(f"     {label:<20} first number after label: {m.group(1) if m else 'NOT FOUND'}")
    # segment / product table
    j = flat.find("Segment Operating Performance") if "Segment Operating Performance" in flat else -1
    for prod in ("iPhone", "Mac", "iPad", "Services"):
        for mm in re.finditer(r"\b" + prod + r"\b[^\d]{0,30}([\d,]{5,})", flat):
            print(f"     {prod:<10} -> {mm.group(1)}")
            break
    # local file comparison
    lp = f"_fd_AAPL_ir_{tag}.txt"
    if os.path.exists(lp):
        loc = open(lp).read().replace("\n", " ")
        for probe in ("Total net sales", "102,466", "109,417", "1.85", "2.02", "49,025"):
            print(f"     local {lp}: contains {probe!r}? {probe in loc}")

print("\n" + "=" * 110)
print("C. iPhone cross-check: FY2025 iPhone minus 9M-FY2025 iPhone == Q4FY25 iPhone (filed)?")
print("=" * 110)
CF = json.load(open("_fd_AAPL_verify_companyfacts.json"))
# product revenue is dimensional in XBRL -> not in companyfacts (undimensioned only). Say so.
print("  NOTE: companyfacts carries only UNDIMENSIONED facts, so iPhone/Mac/Services product")
print("        revenue is NOT retrievable there (it is an axis member). Cross-check must come")
print("        from the filed documents themselves -- which is what part B printed above.")

print("\n" + "=" * 110)
print("D. Shares used for market cap: weighted-average DILUTED vs actual shares OUTSTANDING")
print("=" * 110)
dei = CF["facts"]["dei"]["EntityCommonStockSharesOutstanding"]["units"]["shares"]
d = pd.DataFrame(dei)
d["end"] = pd.to_datetime(d["end"]); d["filed"] = pd.to_datetime(d["filed"])
print(d.sort_values("filed")[["end", "val", "form", "filed", "accn"]].tail(6).to_string(index=False))
wa = CF["facts"]["us-gaap"]["WeightedAverageNumberOfDilutedSharesOutstanding"]["units"]["shares"]
w = pd.DataFrame(wa)
w["start"] = pd.to_datetime(w["start"]); w["end"] = pd.to_datetime(w["end"])
w["days"] = (w.end - w.start).dt.days
wq = w[(w.days > 60) & (w.days < 110)].sort_values("end").drop_duplicates("end", keep="last")
print("\n  WeightedAverageNumberOfDilutedSharesOutstanding, last 5 quarters:")
print(wq[["end", "val", "form", "accn"]].tail(5).to_string(index=False))
wb = pd.DataFrame(CF["facts"]["us-gaap"]["WeightedAverageNumberOfSharesOutstandingBasic"]["units"]["shares"])
wb["start"] = pd.to_datetime(wb["start"]); wb["end"] = pd.to_datetime(wb["end"])
wb["days"] = (wb.end - wb.start).dt.days
wbq = wb[(wb.days > 60) & (wb.days < 110)].sort_values("end").drop_duplicates("end", keep="last")
print("\n  WeightedAverageNumberOfSharesOutstandingBasic, last 3 quarters (what the LAYER picked):")
print(wbq[["end", "val", "form", "accn"]].tail(3).to_string(index=False))

PRICE, NET_CASH, TTM, FY25 = 333.08, 62220.0, 466823.0, 416161.0
sh_out = float(d.sort_values("filed").val.iloc[-1]) / 1e6
sh_dil = float(wq.val.iloc[-1]) / 1e6
sh_bas = float(wbq.val.iloc[-1]) / 1e6
for lab, sh in (("dei shares OUTSTANDING (cover page)", sh_out),
                ("wtd-avg DILUTED  (claim used this)", sh_dil),
                ("wtd-avg BASIC    (layer used this)", sh_bas)):
    ev = PRICE * sh - NET_CASH
    print(f"\n  {lab}: {sh:,.1f}M -> mcap ${PRICE*sh/1e6:.4f}T EV ${ev/1e6:.4f}T"
          f"  EV/FY25 {ev/FY25:.2f}x  EV/TTM {ev/TTM:.2f}x")
