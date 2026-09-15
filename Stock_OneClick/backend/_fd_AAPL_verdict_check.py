"""Final verdict sanity check: recompute the load-bearing numbers from a fresh SEC pull.

Every number the verdict leans on gets one independent derivation here so the writeup is not
just relaying the five research lines. Saves _fd_AAPL_verdict_check.json.
"""
import json
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
UA = "Feijing Research feijing.research@gmail.com"
CF = HERE / "_fd_AAPL_verdict_cf.json"


def cf():
    if CF.exists():
        return json.loads(CF.read_text())
    req = urllib.request.Request(
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        headers={"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})
    import gzip
    import io
    raw = urllib.request.urlopen(req, timeout=120).read()
    try:
        raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
    except Exception:
        pass
    CF.write_text(raw.decode())
    return json.loads(raw.decode())


F = cf()
US = F["facts"]["us-gaap"]


def rows(tag, unit="USD"):
    out = []
    for u in US.get(tag, {}).get("units", {}).get(unit, []):
        if "start" not in u:
            continue
        s, e = pd.Timestamp(u["start"]), pd.Timestamp(u["end"])
        out.append({"tag": tag, "start": s, "end": e, "days": (e - s).days,
                    "val": u["val"], "fy": u.get("fy"), "fp": u.get("fp"),
                    "form": u.get("form"), "filed": pd.Timestamp(u["filed"]),
                    "accn": u.get("accn")})
    return pd.DataFrame(out)


rev = pd.concat([rows("RevenueFromContractWithCustomerExcludingAssessedTax")])
rev = rev.sort_values(["end", "filed"])
q = rev[(rev.days >= 80) & (rev.days <= 100)].drop_duplicates("end", keep="first")
a = rev[(rev.days >= 350) & (rev.days <= 380)].drop_duplicates("end", keep="first")
n9 = rev[(rev.days >= 260) & (rev.days <= 285)].drop_duplicates("end", keep="first")

print("=== quarterly revenue facts (first-filed), last 9 ===")
print(q.tail(9)[["end", "days", "val", "form", "filed", "accn"]].to_string(index=False))
print("\n=== annual ===")
print(a.tail(3)[["end", "val", "form", "filed", "accn"]].to_string(index=False))
print("\n=== nine-month ===")
print(n9.tail(3)[["end", "val", "form", "filed", "accn"]].to_string(index=False))

FY25 = float(a[a.end == "2025-09-27"].val.iloc[0])
FY24 = float(a[a.end == "2024-09-28"].val.iloc[0])
M9_26 = float(n9[n9.end == "2026-06-27"].val.iloc[0])
M9_25 = float(n9[n9.end == "2025-06-28"].val.iloc[0])
Q4_25 = FY25 - M9_25
TTM = M9_26 + Q4_25
TTM_prior = FY24 - (FY24 - 0)  # placeholder, computed below
M9_24 = float(n9[n9.end == "2024-06-29"].val.iloc[0])
Q4_24 = FY24 - M9_24
TTM_prior = M9_25 + Q4_24

print(f"\nFY2025 {FY25/1e6:,.0f}M  FY2024 {FY24/1e6:,.0f}M  FY growth {FY25/FY24-1:+.2%}")
print(f"9M FY26 {M9_26/1e6:,.0f}M  9M FY25 {M9_25/1e6:,.0f}M  9M growth {M9_26/M9_25-1:+.2%}")
print(f"derived Q4 FY25 {Q4_25/1e6:,.0f}M   derived Q4 FY24 {Q4_24/1e6:,.0f}M")
print(f"TTM to 2026-06-27 {TTM/1e6:,.0f}M   prior TTM {TTM_prior/1e6:,.0f}M   "
      f"TTM growth {TTM/TTM_prior-1:+.2%}")

# ---- EPS
eps = rows("EarningsPerShareDiluted", unit="USD/shares").sort_values(["end", "filed"])
eq = eps[(eps.days >= 80) & (eps.days <= 100)].drop_duplicates("end", keep="first")
ea = eps[(eps.days >= 350) & (eps.days <= 380)].drop_duplicates("end", keep="first")
e9 = eps[(eps.days >= 260) & (eps.days <= 285)].drop_duplicates("end", keep="first")
print("\n=== quarterly diluted EPS (first-filed), last 8 ===")
print(eq.tail(8)[["end", "val", "form", "filed"]].to_string(index=False))
eps_fy25 = float(ea[ea.end == "2025-09-27"].val.iloc[0])
eps_9m25 = float(e9[e9.end == "2025-06-28"].val.iloc[0])
eps_q4_25 = round(eps_fy25 - eps_9m25, 2)
eps_fy26q = [float(eq[eq.end == d].val.iloc[0]) for d in ("2025-12-27", "2026-03-28", "2026-06-27")]
EPS_TTM = eps_q4_25 + sum(eps_fy26q)
print(f"FY2025 EPS {eps_fy25}  9M FY25 {eps_9m25}  derived Q4FY25 {eps_q4_25}")
print(f"FY26 quarters {eps_fy26q}  ->  TTM diluted EPS {EPS_TTM:.2f}")
naive4 = float(eq.tail(4).val.sum())
print(f"layer's naive last-4-row sum: {naive4:.2f}  (window {list(eq.tail(4).end.dt.date)})")

# ---- net income, gross profit, operating income (TTM)
def ttm_of(tag):
    r = rows(tag).sort_values(["end", "filed"])
    aa = r[(r.days >= 350) & (r.days <= 380)].drop_duplicates("end", keep="first")
    nn = r[(r.days >= 260) & (r.days <= 285)].drop_duplicates("end", keep="first")
    fy = float(aa[aa.end == "2025-09-27"].val.iloc[0])
    m26 = float(nn[nn.end == "2026-06-27"].val.iloc[0])
    m25 = float(nn[nn.end == "2025-06-28"].val.iloc[0])
    return fy - m25 + m26, fy

for tag, name in (("NetIncomeLoss", "net income"), ("GrossProfit", "gross profit"),
                  ("OperatingIncomeLoss", "operating income"),
                  ("ResearchAndDevelopmentExpense", "R&D")):
    t, fy = ttm_of(tag)
    print(f"TTM {name:<17} {t/1e6:>10,.0f}M  ({t/TTM:.2%} of TTM rev)   FY2025 {fy/1e6:>10,.0f}M "
          f"({fy/FY25:.2%})")

# ---- shares, three bases
dei = F["facts"]["dei"]["EntityCommonStockSharesOutstanding"]["units"]["shares"]
dei = sorted(dei, key=lambda x: x["end"])[-1]
sh_cover = dei["val"]
shd = rows("WeightedAverageNumberOfDilutedSharesOutstanding", unit="shares").sort_values(["end", "filed"])
shd_q = shd[(shd.days >= 80) & (shd.days <= 100)].drop_duplicates("end", keep="first")
sh_dil = float(shd_q.val.iloc[-1])
shb = rows("WeightedAverageNumberOfSharesOutstandingBasic", unit="shares").sort_values(["end", "filed"])
shb_q = shb[(shb.days >= 80) & (shb.days <= 100)].drop_duplicates("end", keep="first")
sh_bas = float(shb_q.val.iloc[-1])
print(f"\nshares: dei cover {sh_cover/1e6:,.1f}M (as of {dei['end']})  "
      f"diluted wtd {sh_dil/1e6:,.1f}M   basic wtd {sh_bas/1e6:,.1f}M")

PX = 333.08
NETCASH = 62_173e6
print(f"\n=== multiples at ${PX} ===")
for nm, s in (("dei cover", sh_cover), ("basic wtd (layer)", sh_bas), ("diluted wtd", sh_dil)):
    mcap = PX * s
    ev = mcap - NETCASH
    print(f"  {nm:<18} mcap {mcap/1e9:,.0f}bn  EV {ev/1e9:,.0f}bn  "
          f"EV/TTM-Sales {ev/TTM:.2f}x  EV/FY-Sales {ev/FY25:.2f}x")
print(f"  P/E on TTM EPS {EPS_TTM:.2f}: {PX/EPS_TTM:.2f}x    "
      f"on layer's {naive4:.2f}: {PX/naive4:.2f}x    on FY2025 EPS {eps_fy25}: {PX/eps_fy25:.2f}x")

out = {"ttm_revenue": TTM, "ttm_revenue_prior": TTM_prior, "ttm_growth": TTM / TTM_prior - 1,
       "fy2025_revenue": FY25, "fy_growth": FY25 / FY24 - 1, "q4fy25_revenue": Q4_25,
       "ttm_eps": EPS_TTM, "layer_naive_eps": naive4, "fy2025_eps": eps_fy25,
       "shares_cover": sh_cover, "px": PX}
(HERE / "_fd_AAPL_verdict_check.json").write_text(json.dumps(out, indent=2, default=str))
print("\nwrote _fd_AAPL_verdict_check.json")
