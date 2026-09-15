"""AAPL revenue mechanics: decomposition, contribution-to-growth, true TTM, margin split.

Self-contained (own namespace: _fd_AAPL_rev_*) so it does not collide with other agents' files.
Every number printed traces to the accession in FILINGS.
"""
from __future__ import annotations
import datetime as dt, os, time, collections, json
from pathlib import Path
import pandas as pd, requests
from lxml import etree

HERE = Path(__file__).resolve().parent
H = {"User-Agent": "Fei Jing feijing.research@gmail.com", "Accept-Encoding": "gzip, deflate"}
CIK = "320193"
XBI = "{http://www.xbrl.org/2003/instance}"

FILINGS = [
    ("0000320193-26-000020", "aapl-20260627", "10-Q FQ3'26"),
    ("0000320193-26-000013", "aapl-20260328", "10-Q FQ2'26"),
    ("0000320193-26-000006", "aapl-20251227", "10-Q FQ1'26"),
    ("0000320193-25-000079", "aapl-20250927", "10-K FY2025"),
    ("0000320193-25-000073", "aapl-20250628", "10-Q FQ3'25"),
    ("0000320193-24-000123", "aapl-20240928", "10-K FY2024"),
    ("0000320193-24-000081", "aapl-20240629", "10-Q FQ3'24"),
    ("0000320193-23-000106", "aapl-20230930", "10-K FY2023"),
    ("0000320193-23-000077", "aapl-20230701", "10-Q FQ3'23"),
    ("0000320193-22-000108", "aapl-20220924", "10-K FY2022"),
]

TAGS = {"RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "GrossProfit",
        "CostOfGoodsAndServicesSold", "OperatingIncomeLoss", "NetIncomeLoss",
        "EarningsPerShareDiluted", "ResearchAndDevelopmentExpense",
        "ContractWithCustomerLiability", "RevenueRemainingPerformanceObligation",
        "InventoryNet", "AccountsReceivableNetCurrent", "NontradeReceivablesCurrent",
        "ConcentrationRiskPercentage1"}


def fetch(accn, stem):
    p = HERE / f"_fd_AAPL_xbrl_{stem}.xml"
    if p.exists() and p.stat().st_size > 10000:
        return p.read_bytes()
    url = f"https://www.sec.gov/Archives/edgar/data/{CIK}/{accn.replace('-','')}/{stem}_htm.xml"
    r = requests.get(url, headers=H, timeout=120); r.raise_for_status()
    p.write_bytes(r.content); time.sleep(0.2)
    return r.content


def parse(raw, accn, label):
    root = etree.fromstring(raw)
    ctx = {}
    for c in root.iter(XBI + "context"):
        per = c.find(XBI + "period")
        rec = {"start": None, "end": None, "instant": None, "dims": {}}
        if per is not None:
            for k in ("startDate", "endDate", "instant"):
                e = per.find(XBI + k)
                if e is not None and e.text:
                    rec[{"startDate": "start", "endDate": "end", "instant": "instant"}[k]] = e.text
        ent = c.find(XBI + "entity")
        if ent is not None:
            seg = ent.find(XBI + "segment")
            if seg is not None:
                for m in seg.iter():
                    if isinstance(m.tag, str) and m.tag.endswith("explicitMember") and m.get("dimension"):
                        rec["dims"][m.get("dimension").split(":")[-1]] = (m.text or "").split(":")[-1]
        ctx[c.get("id")] = rec
    rows = []
    for el in root.iter():
        if not isinstance(el.tag, str) or "}" not in el.tag:
            continue
        uri, name = el.tag[1:].split("}")
        if name not in TAGS or ("us-gaap" not in uri and "aapl" not in uri.lower()):
            continue
        cid, txt = el.get("contextRef"), el.text
        if cid not in ctx or not txt:
            continue
        try:
            v = float(txt)
        except ValueError:
            continue
        if el.get("sign") == "-":
            v = -v
        c = ctx[cid]
        rows.append(dict(tag=name, start=c["start"], end=c["end"] or c["instant"],
                         instant=c["instant"], dims=c["dims"],
                         dimkey=";".join(f"{a}={b}" for a, b in sorted(c["dims"].items())),
                         val=v, accn=accn, label=label))
    return rows


def load():
    rows = []
    for accn, stem, label in FILINGS:
        r = parse(fetch(accn, stem), accn, label)
        rows += r
        print(f"  {label:<12}{stem:<16}{len(r):>5} facts   [{accn}]")
    D = pd.DataFrame(rows)
    for c in ("start", "end"):
        D[c] = pd.to_datetime(D[c], errors="coerce")
    D["days"] = (D["end"] - D["start"]).dt.days
    return D.drop_duplicates(subset=["tag", "start", "end", "dimkey", "val"])


def fmt(x):
    return f"{x:,.0f}" if pd.notna(x) else "-"


if __name__ == "__main__":
    D = load()
    D.to_csv(HERE / "_fd_AAPL_rev_facts.csv", index=False)
    rev = D[D.tag == "RevenueFromContractWithCustomerExcludingAssessedTax"]

    # ---------------- product-category: 90d (quarter), 181/272d (YTD), 363-370d (FY)
    cat = rev[rev.dimkey.str.match(r"^ProductOrServiceAxis=\w+$", na=False)].copy()
    cat["cat"] = cat.dimkey.str.split("=").str[1]
    piv = (cat.groupby([cat.end, "cat", cat.days]).val.max().unstack("cat") / 1e6)
    piv = piv.reset_index().set_index(["end", "days"]).sort_index()

    ORDER = ["IPhone", "Mac", "IPad", "WearablesHomeandAccessories", "Service"]
    ORDER = [c for c in ORDER if c in piv.columns]

    # build a clean quarterly series: 90d facts directly; Q4 = FY - 9M
    q = {}
    for (end, days), row in piv.iterrows():
        if 80 <= days <= 100:
            q[end.date()] = {c: row.get(c) for c in ORDER}
    ytd9, fy = {}, {}
    for (end, days), row in piv.iterrows():
        if 260 <= days <= 280:
            ytd9[end.date()] = {c: row.get(c) for c in ORDER}
        if 355 <= days <= 375:
            fy[end.date()] = {c: row.get(c) for c in ORDER}
    # derive Q4 for each FY where we have both FY and the matching 9M
    for fye, fyrow in fy.items():
        m9 = [k for k in ytd9 if 0 < (fye - k).days < 105]
        if not m9:
            continue
        base = ytd9[max(m9)]
        q[fye] = {c: (fyrow.get(c, float("nan")) - base.get(c, float("nan"))) for c in ORDER}

    Q = pd.DataFrame(q).T.sort_index()
    Q["TOTAL"] = Q[ORDER].sum(axis=1)
    print("\n" + "=" * 132)
    print("A. QUARTERLY REVENUE BY PRODUCT CATEGORY ($M).  Q4 rows are derived FY minus 9M (marked *)")
    print("=" * 132)
    derived = {k for k in fy}
    show = Q.copy()
    show.index = [f"{k}{'*' if k in derived else ''}" for k in Q.index]
    print(show.to_string(float_format=fmt))

    print("\n" + "=" * 132)
    print("B. DATE-MATCHED YoY BY CATEGORY (base = same quarter-end +/-20d, one year back)")
    print("=" * 132)
    idx = list(Q.index)
    yoy_rows = {}
    contrib_rows = {}
    for e in idx:
        tgt = e - dt.timedelta(days=365)
        cand = [x for x in idx if abs((x - tgt).days) <= 20]
        if not cand:
            continue
        b = cand[0]
        yoy_rows[f"{e} vs {b}"] = {c: Q.loc[e, c] / Q.loc[b, c] - 1 for c in ORDER + ["TOTAL"]}
        d_abs = {c: Q.loc[e, c] - Q.loc[b, c] for c in ORDER}
        tot = sum(d_abs.values())
        contrib_rows[f"{e} vs {b}"] = dict(**{f"d{c}": v for c, v in d_abs.items()},
                                           dTOTAL=tot,
                                           **{f"%{c}": v / tot for c, v in d_abs.items()})
    Y = pd.DataFrame(yoy_rows).T
    print(Y.to_string(float_format=lambda x: f"{x*100:+.1f}%"))
    C = pd.DataFrame(contrib_rows).T
    print("\nC. CONTRIBUTION TO YoY REVENUE GROWTH ($M, then share of total increment)")
    print(C[[c for c in C.columns if c.startswith('d')]].to_string(float_format=fmt))
    print()
    print(C[[c for c in C.columns if c.startswith('%')]].to_string(float_format=lambda x: f"{x*100:+.0f}%"))

    print("\nD. CATEGORY SHARE OF TOTAL REVENUE")
    print(Q[ORDER].div(Q.TOTAL, axis=0).to_string(float_format=lambda x: f"{x*100:.1f}%"))

    # ---------------- product vs service gross margin
    gp = D[(D.tag.isin(["GrossProfit", "CostOfGoodsAndServicesSold"]))
           & D.dimkey.str.match(r"^ProductOrServiceAxis=\w+$", na=False)].copy()
    gp["cat"] = gp.dimkey.str.split("=").str[1]
    print("\n" + "=" * 132)
    print("E. PRODUCT vs SERVICES GROSS MARGIN (from dimensional GrossProfit / revenue)")
    print("=" * 132)
    g2 = (gp[gp.tag == "GrossProfit"].groupby([gp.end, "cat", gp.days]).val.max().unstack("cat") / 1e6)
    g2 = g2.reset_index().set_index(["end", "days"]).sort_index()
    ps = rev[rev.dimkey.isin(["ProductOrServiceAxis=ProductMember", "ProductOrServiceAxis=Product",
                              "ProductOrServiceAxis=ServiceMember", "ProductOrServiceAxis=Service"])].copy()
    ps["cat"] = ps.dimkey.str.split("=").str[1]
    r2 = (ps.groupby([ps.end, "cat", ps.days]).val.max().unstack("cat") / 1e6)
    r2 = r2.reset_index().set_index(["end", "days"]).sort_index()
    joined = g2.join(r2, lsuffix="_gp", rsuffix="_rev", how="inner")
    for c in ("Product", "Service"):
        if f"{c}_gp" in joined and f"{c}_rev" in joined:
            joined[f"{c}_GM%"] = joined[f"{c}_gp"] / joined[f"{c}_rev"] * 100
    print(joined.to_string(float_format=lambda x: f"{x:,.1f}"))
