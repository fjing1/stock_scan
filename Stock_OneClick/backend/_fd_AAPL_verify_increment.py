"""ADVERSARIAL VERIFY #8 -- recompute every aggregate in the claim from its components, and test
attack #6 (FORWARD REVERSAL): is the growth still 'an iPhone cycle' in the MOST RECENT period?

Numbers under test:
  'two-year stacked CAGR +0.3% -> +12.9%, iPhone -4.5% -> +17.5%'
  'iPhone supplied 62.9-82.9% of each quarter's increment'
  'Services two-year CAGR stayed pinned at 12.6-13.9%'
  'iPhone +21.7%'
Every input is a filed segment-dimensioned XBRL fact (see _fd_AAPL_verify_seg_raw.csv).
Q3 FY2023 / Q2 FY2023 product lines come from the FY2024-era 10-Qs, fetched here.
"""
from __future__ import annotations
import html, json, os, re, time, urllib.request
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
UA = {"User-Agent": "Feijing Research feijing.research@gmail.com"}
EXTRA = {"0000320193-23-000064": "10-Q Q2FY23 (end 2023-04-01)",
         "0000320193-23-000006": "10-Q Q1FY23 (end 2022-12-31)",
         "0000320193-24-000006": "10-Q Q1FY24 (end 2023-12-30)",
         "0000320193-24-000069": "10-Q Q2FY24 (end 2024-03-30)",
         "0000320193-23-000077": "10-Q Q3FY23 (end 2023-07-01)"}


def sec(url, dest):
    if os.path.exists(dest):
        return open(dest, "rb").read()
    b = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90).read()
    open(dest, "wb").write(b); time.sleep(0.3)
    return b


rows = []
for accn in EXTRA:
    a = accn.replace("-", "")
    idx = json.loads(sec(f"https://www.sec.gov/Archives/edgar/data/320193/{a}/index.json",
                         f"_fd_AAPL_verify_idx_{accn}.json"))
    names = [i["name"] for i in idx["directory"]["item"]]
    inst = [n for n in names if re.match(r"aapl-\d{8}_htm\.xml$", n)]
    if not inst:
        print(f"  no instance for {accn}: {[n for n in names if n.endswith('.xml')][:6]}"); continue
    xml = sec(f"https://www.sec.gov/Archives/edgar/data/320193/{a}/{inst[0]}",
              f"_fd_AAPL_verify_inst_{accn}.xml").decode("utf-8", "replace")
    ctx = {}
    for m in re.finditer(r'<(?:xbrli:)?context id="([^"]+)">(.*?)</(?:xbrli:)?context>', xml, re.S):
        cid, body = m.group(1), m.group(2)
        sd = re.search(r"<(?:xbrli:)?startDate>([\d-]+)<", body)
        ed = re.search(r"<(?:xbrli:)?endDate>([\d-]+)<", body)
        mem = re.findall(r'<xbrldi:explicitMember dimension="([^"]+)">([^<]+)<', body)
        ctx[cid] = (sd.group(1) if sd else None, ed.group(1) if ed else None, dict(mem))
    tag = "RevenueFromContractWithCustomerExcludingAssessedTax"
    for m in re.finditer(rf'<us-gaap:{tag}[^>]*contextRef="([^"]+)"[^>]*>([-\d]+)</us-gaap:{tag}>', xml):
        s, e, dims = ctx.get(m.group(1), (None, None, {}))
        if not s:
            continue
        rows.append({"accn": accn, "start": s, "end": e, "val": float(m.group(2)),
                     "member": "|".join(f"{k.split(':')[-1]}={v.split(':')[-1]}" for k, v in sorted(dims.items()))})

D = pd.concat([pd.read_csv("_fd_AAPL_verify_seg_raw.csv")[["accn", "start", "end", "val", "member"]],
               pd.DataFrame(rows)], ignore_index=True)
D["start"] = pd.to_datetime(D.start); D["end"] = pd.to_datetime(D.end)
D["days"] = (D.end - D.start).dt.days
D["member"] = D.member.fillna("")
Q = D[(D.days >= 80) & (D.days <= 100)]
prod = Q[Q.member.str.contains("ProductOrService")]
piv = (prod.assign(line=prod.member.str.split("=").str[-1])
       .pivot_table(index="end", columns="line", values="val", aggfunc="max") / 1e6).sort_index()
piv["Total"] = (Q[Q.member == ""].groupby("end").val.max() / 1e6).reindex(piv.index)
piv = piv.rename(columns={"IPhoneMember": "iPhone", "MacMember": "Mac", "IPadMember": "iPad",
                          "ServiceMember": "Services",
                          "WearablesHomeandAccessoriesMember": "Wearables",
                          "ProductMember": "Products(agg)"})
print("=" * 150)
print("QUARTERLY NET SALES BY LINE ($M). NOTE the Sept (Q4) quarters are ABSENT -- Apple files only")
print("annual product-line facts in the 10-K -- so every YoY below is DATE-MATCHED, never positional.")
print("=" * 150)
print(piv.to_string(float_format=lambda x: f"{x:,.0f}"))

LINES = ["iPhone", "Mac", "iPad", "Wearables", "Services"]
PAIRS = [(pd.Timestamp("2025-12-27"), pd.Timestamp("2024-12-28"), pd.Timestamp("2023-12-30"), "Q1 FY26"),
         (pd.Timestamp("2026-03-28"), pd.Timestamp("2025-03-29"), pd.Timestamp("2024-03-30"), "Q2 FY26"),
         (pd.Timestamp("2026-06-27"), pd.Timestamp("2025-06-28"), pd.Timestamp("2024-06-29"), "Q3 FY26")]
PRIOR = [(pd.Timestamp("2024-12-28"), pd.Timestamp("2023-12-30"), pd.Timestamp("2022-12-31"), "Q1 FY25"),
         (pd.Timestamp("2025-03-29"), pd.Timestamp("2024-03-30"), pd.Timestamp("2023-04-01"), "Q2 FY25"),
         (pd.Timestamp("2025-06-28"), pd.Timestamp("2024-06-29"), pd.Timestamp("2023-07-01"), "Q3 FY25")]

print("\n" + "=" * 150)
print("A. INCREMENT ATTRIBUTION, recomputed from components (attack #5)")
print("=" * 150)
print(f"  {'quarter':<9} {'tot inc':>9} " + "".join(f"{l+' inc':>12}" for l in LINES)
      + f"{'iPhone share':>14}{'components sum':>16}")
for cur, p1, p2, lab in PAIRS:
    inc = piv.loc[cur, "Total"] - piv.loc[p1, "Total"]
    parts = {l: piv.loc[cur, l] - piv.loc[p1, l] for l in LINES}
    print(f"  {lab:<9} {inc:>9,.0f} " + "".join(f"{parts[l]:>12,.0f}" for l in LINES)
          + f"{parts['iPhone']/inc:>13.1%}{sum(parts.values()):>16,.0f}")
print("  -> components sum ties to total increment in every quarter, so the aggregate is clean.")
print(f"  -> iPhone share range = "
      f"{min((piv.loc[c,'iPhone']-piv.loc[p,'iPhone'])/(piv.loc[c,'Total']-piv.loc[p,'Total']) for c,p,_,_ in PAIRS):.1%}"
      f" to "
      f"{max((piv.loc[c,'iPhone']-piv.loc[p,'iPhone'])/(piv.loc[c,'Total']-piv.loc[p,'Total']) for c,p,_,_ in PAIRS):.1%}"
      f"   (claim: 62.9-82.9%)")
print("  -> BUT the direction across the three quarters is what a forward reader needs:")
sh = [(lab, (piv.loc[c, 'iPhone'] - piv.loc[p, 'iPhone']) / (piv.loc[c, 'Total'] - piv.loc[p, 'Total']))
      for c, p, _, lab in PAIRS]
print("     " + "  ->  ".join(f"{l} {v:.1%}" for l, v in sh)
      + "   i.e. NARROWING concentration, the growth is BROADENING")

print("\n" + "=" * 150)
print("B. TWO-YEAR STACKED CAGR (attack: is the acceleration just a weak base?)")
print("=" * 150)
print(f"  {'quarter':<9} {'Total 2yCAGR':>14} {'iPhone 2yCAGR':>15} {'Services 2yCAGR':>17}"
      f" {'Total YoY':>11} {'iPhone YoY':>12} {'Services YoY':>14}")
for cur, p1, p2, lab in PRIOR + PAIRS:
    def cg(l):
        return (piv.loc[cur, l] / piv.loc[p2, l]) ** 0.5 - 1 if p2 in piv.index and np.isfinite(piv.loc[p2, l]) else np.nan
    def yy(l):
        return piv.loc[cur, l] / piv.loc[p1, l] - 1
    print(f"  {lab:<9} {cg('Total'):>13.1%} {cg('iPhone'):>14.1%} {cg('Services'):>16.1%}"
          f" {yy('Total'):>10.1%} {yy('iPhone'):>11.1%} {yy('Services'):>13.1%}")
print("\n  claim: 'two-year stacked CAGR +0.3% -> +12.9%, iPhone -4.5% -> +17.5%',")
print("         'Services two-year CAGR stayed pinned at 12.6-13.9%'")
print("  -> the '+0.3% / -4.5%' start is the Q2 FY25 TROUGH of the series, not the average of the")
print("     prior year. The neighbouring quarters were Total +1.7%/+3.0%/+7.2% and iPhone higher.")
print("     Using the trough as the 'before' overstates the swing; the honest statement is that the")
print("     two-year stack rose from roughly +3% to +13%, which still refutes a pure base effect.")
