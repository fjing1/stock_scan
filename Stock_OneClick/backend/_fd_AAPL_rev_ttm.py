"""AAPL quarterly P&L parsed straight out of each 8-K EX-99.1 (no hand-typed numbers), then
true TTM, corrected multiples, tariff bridge, product/services margin split, elasticity.

Accessions:
  Q1FY25 0000320193-25-000007   Q2FY25 0000320193-25-000055   Q3FY25 0000320193-25-000071
  Q4FY25 0000320193-25-000077   Q1FY26 0000320193-26-000005   Q2FY26 0000320193-26-000011
  Q3FY26 0000320193-26-000018
"""
from __future__ import annotations
import re
import numpy as np, pandas as pd

pd.set_option("display.width", 260)

FILES = {  # tag -> (txt file, quarter end, accession)
    "Q1FY25": ("_fd_AAPL_ir_q1fy25.txt", "2024-12-28", "0000320193-25-000007"),
    "Q2FY25": ("_fd_AAPL_ir_q2fy25.txt", "2025-03-29", "0000320193-25-000055"),
    "Q3FY25": ("_fd_AAPL_ir_q3fy25.txt", "2025-06-28", "0000320193-25-000071"),
    "Q4FY25": ("_fd_AAPL_ir_q4fy25.txt", "2025-09-27", "0000320193-25-000077"),
    "Q1FY26": ("_fd_AAPL_ir_q1fy26.txt", "2025-12-27", "0000320193-26-000005"),
    "Q2FY26": ("_fd_AAPL_ir_q2fy26.txt", "2026-03-28", "0000320193-26-000011"),
    "Q3FY26": ("_fd_AAPL_ir_q3fy26.txt", "2026-06-27", "0000320193-26-000018"),
}
NUM = r"\$?\(?-?\d[\d,]*(?:\.\d+)?\)?"


def nums(s):
    out = []
    for m in re.findall(NUM, s):
        neg = m.startswith("(") or m.startswith("$(")
        v = float(m.replace("$", "").replace(",", "").replace("(", "").replace(")", ""))
        out.append(-v if neg else v)
    return out


def first_after(body, label, occurrence=0):
    """first number after the `occurrence`-th appearance of `label` = the CURRENT quarter column
    (Apple's tables always put the current three-month column first)."""
    idx, start = -1, 0
    for _ in range(occurrence + 1):
        idx = body.find(label, start)
        if idx < 0:
            raise KeyError(f"{label} #{occurrence}")
        start = idx + len(label)
    v = nums(body[idx + len(label): idx + len(label) + 100])
    if not v:
        raise KeyError(f"no number after {label}")
    return v[0]


rows = []
for tag, (path, qend, acc) in FILES.items():
    t = open(path).read().replace("\n", "")
    i = t.find("CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS")
    body = t[i:i + 2600]
    # strip the footnote markers "(1)"/"(2)": "Total net sales (1)124,300" otherwise parses as -1
    body = body.replace("(1)", "").replace("(2)", "")
    sales, rest = body.split("Cost of sales:", 1)
    cos, opex = rest.split("Gross margin", 1)
    shares_i = body.find("Shares used in computing")
    rows.append(dict(
        tag=tag, qend=qend, acc=acc,
        prod_rev=first_after(sales, "Products"), svc_rev=first_after(sales, "Services"),
        rev=first_after(sales, "Total net sales"),
        prod_cos=first_after(cos, "Products"), svc_cos=first_after(cos, "Services"),
        gp=first_after(opex, ""), rnd=first_after(opex, "Research and development"),
        sga=first_after(opex, "Selling, general and administrative"),
        opinc=first_after(opex, "Operating income"),
        pretax=first_after(opex, "Income before provision for income taxes"),
        tax=first_after(opex, "Provision for income taxes"),
        ni=first_after(opex, "Net income"),
        eps=first_after(opex[:opex.find("Shares used")], "Diluted"),
        sh_dil=first_after(body[shares_i:], "Diluted"),
    ))

Q = pd.DataFrame(rows).set_index("qend")
Q.index = pd.to_datetime(Q.index)
Q = Q.sort_index()

# ---- integrity check: revenue and gross margin must reconcile
chk = (Q.prod_rev + Q.svc_rev - Q.rev).abs().max()
chk2 = (Q.rev - Q.prod_cos - Q.svc_cos - Q.gp).abs().max()
chk3 = (Q.gp - Q.rnd - Q.sga - Q.opinc).abs().max()
print(f"reconciliation residuals (must be 0): rev {chk:.0f}  gross margin {chk2:.0f}  op income {chk3:.0f}")
assert chk == 0 and chk2 == 0 and chk3 == 0, "parse failed"

Q["prod_gm%"] = (Q.prod_rev - Q.prod_cos) / Q.prod_rev * 100
Q["svc_gm%"] = (Q.svc_rev - Q.svc_cos) / Q.svc_rev * 100
Q["gm%"] = Q.gp / Q.rev * 100
Q["op%"] = Q.opinc / Q.rev * 100
Q["rnd%"] = Q.rnd / Q.rev * 100
Q["tax%"] = Q.tax / Q.pretax * 100
Q["svc_rev_share%"] = Q.svc_rev / Q.rev * 100
Q["svc_gp_share%"] = (Q.svc_rev - Q.svc_cos) / Q.gp * 100

print("\n" + "=" * 150)
print("1. QUARTERLY P&L, parsed from the filed EX-99.1 tables  ($M, EPS $)")
print("=" * 150)
show = ["tag", "rev", "prod_rev", "svc_rev", "prod_cos", "svc_cos", "gp", "gm%", "prod_gm%",
        "svc_gm%", "rnd", "rnd%", "opinc", "op%", "tax%", "ni", "eps", "sh_dil",
        "svc_rev_share%", "svc_gp_share%"]
print(Q[show].to_string(float_format=lambda x: f"{x:,.2f}"))

print("\n2. YoY, date-matched (i-4 = same fiscal quarter one year earlier)")
y = pd.DataFrame(index=Q.index[4:])
y["tag"] = Q.tag.values[4:]
for c in ["rev", "prod_rev", "svc_rev", "gp", "rnd", "opinc", "ni", "eps", "sh_dil"]:
    y[c + "_%"] = (Q[c].values[4:] / Q[c].values[:3] - 1) * 100
for c in ["gm%", "prod_gm%", "svc_gm%", "op%", "rnd%", "svc_rev_share%", "svc_gp_share%"]:
    y[c.replace("%", "") + "_ppt"] = Q[c].values[4:] - Q[c].values[:3]
print(y.to_string(float_format=lambda x: f"{x:+.2f}"))

# ---------------------------------------------------------------- TRUE TTM
ttm = Q.iloc[-4:]
FY25 = dict(rev=416161.0, ni=112010.0, eps=7.46, gp=195201.0, opinc=133050.0, rnd=34550.0)
print("\n" + "=" * 150)
print("3. TRUE TTM (4 consecutive filed quarters) vs the FY-2025 fallback the repo layer used")
print("=" * 150)
print(f"  window                      {ttm.index[0].date()} .. {ttm.index[-1].date()}   ({', '.join(ttm.tag)})")
for k, lab in [("rev", "revenue"), ("gp", "gross profit"), ("opinc", "operating income"), ("ni", "net income")]:
    print(f"  TTM {lab:<18}{ttm[k].sum():>12,.0f} $M      FY2025 {FY25[k]:>12,.0f}   "
          f"TTM/FY {ttm[k].sum()/FY25[k]-1:+7.1%}")
print(f"  TTM diluted EPS (sum)     {ttm.eps.sum():>12.2f} $        FY2025 {FY25['eps']:>12.2f}")
print(f"  TTM gross margin          {ttm.gp.sum()/ttm.rev.sum():>12.1%}          FY2025 {FY25['gp']/FY25['rev']:>12.1%}")
print(f"  TTM operating margin      {ttm.opinc.sum()/ttm.rev.sum():>12.1%}          FY2025 {FY25['opinc']/FY25['rev']:>12.1%}")
print(f"  TTM R&D / revenue         {ttm.rnd.sum()/ttm.rev.sum():>12.1%}          FY2025 {FY25['rnd']/FY25['rev']:>12.1%}")

bad_win = Q.loc[["2025-06-28", "2025-12-27", "2026-03-28", "2026-06-27"]]
print(f"\n  the layer's 4-ROW quarterly sum skips Q4FY25 and re-uses Q3FY25:")
print(f"    that EPS sum  {bad_win.eps.sum():.2f}   vs true trailing-4Q EPS {ttm.eps.sum():.2f}"
      f"   ({bad_win.eps.sum()/ttm.eps.sum()-1:+.1%})")

PRICE, NET_CASH = 333.08, 62220.0
SH = float(ttm.sh_dil.iloc[-1]) / 1000.0   # filed in THOUSANDS -> $M-consistent millions
mcap = PRICE * SH
ev = mcap - NET_CASH
print("\n" + "=" * 150)
print(f"4. CORRECTED MULTIPLES   price ${PRICE} (2026-09-14 close), diluted shares {SH:,.1f}M -> mcap ${mcap/1e6:.3f}T,"
      f" net cash ${NET_CASH:,.0f}M, EV ${ev/1e6:.3f}T")
print("=" * 150)
tbl = [
    ("EV/Sales   on FY2025 revenue   <- what the repo layer printed", ev / FY25["rev"]),
    ("EV/Sales   on TRUE TTM revenue", ev / ttm.rev.sum()),
    ("P/E GAAP   on the layer's non-consecutive 4-row EPS sum", PRICE / bad_win.eps.sum()),
    ("P/E GAAP   on TRUE trailing-4-quarter EPS", PRICE / ttm.eps.sum()),
    ("P/E GAAP   on FY2025 EPS", PRICE / FY25["eps"]),
    ("EV/EBIT    on TRUE TTM operating income", ev / ttm.opinc.sum()),
    ("EV/GrossProfit on TRUE TTM", ev / ttm.gp.sum()),
]
for k, v in tbl:
    print(f"  {k:<60}{v:>8.1f}x")

# ---------------------------------------------------------------- TARIFF BRIDGE
print("\n" + "=" * 150)
print("5. TARIFF ROUND-TRIP. Only Q3FY26 quantifies it in a filed document (8-K EX-99.1,")
print("   acc 0000320193-26-000018): 'gross margin ... including a favorable impact of approximately")
print("   2 percentage points from tariff refunds' and 'EPS ... included a favorable impact of $0.11'.")
print("=" * 150)
q, b = Q.loc["2026-06-27"], Q.loc["2025-06-28"]
pre_from_gm = 0.02 * q.rev
aft_from_eps = 0.11 * q.sh_dil / 1000
print(f"  implied PRE-tax benefit from the 2ppt GM statement : {pre_from_gm:>9,.0f} $M")
print(f"  implied AFTER-tax benefit from the $0.11 statement : {aft_from_eps:>9,.0f} $M")
print(f"  Q3FY26 effective tax rate {q['tax%']:.1f}% -> gross-up of the EPS figure: "
      f"{aft_from_eps/(1-q['tax%']/100):>9,.0f} $M   (brackets the 2ppt, so the two disclosures agree)")
eps_ex = q.eps - 0.11
print(f"\n  Q3FY26 EPS reported {q.eps:.2f}  = {q.eps/b.eps-1:+.1%} YoY   (company headline: 'up 29 percent')")
print(f"  Q3FY26 EPS ex-refund {eps_ex:.2f} = {eps_ex/b.eps-1:+.1%} YoY")
print(f"  share of the YoY EPS increment that IS the refund: {0.11/(q.eps-b.eps):.0%}")
print(f"  Q3FY26 company GM {q['gm%']:.1f}% -> ex-refund {q['gm%']-2:.1f}%  vs Q3FY25 {b['gm%']:.1f}%"
      f"  (still {q['gm%']-2-b['gm%']:+.1f}ppt)")
print(f"  Products GM {b['prod_gm%']:.1f}% -> {q['prod_gm%']:.1f}%  ({q['prod_gm%']-b['prod_gm%']:+.1f}ppt);"
      f"  the 2ppt company-level refund is {0.02*q.rev/q.prod_rev*100:.1f}ppt of PRODUCTS margin,")
print(f"    so Products GM ex-refund ~= {q['prod_gm%']-0.02*q.rev/q.prod_rev*100:.1f}%"
      f" ({q['prod_gm%']-0.02*q.rev/q.prod_rev*100-b['prod_gm%']:+.1f}ppt YoY)")
print(f"  Services GM {b['svc_gm%']:.1f}% -> {q['svc_gm%']:.1f}%  ({q['svc_gm%']-b['svc_gm%']:+.1f}ppt) -- FLAT")
print(f"\n  TTM EPS ex that single refund {ttm.eps.sum()-0.11:.2f} -> P/E {PRICE/(ttm.eps.sum()-0.11):.1f}x"
      f"  (vs {PRICE/ttm.eps.sum():.1f}x as reported)")

Q.to_csv("_fd_AAPL_rev_quarterly.csv")
print("\nwrote _fd_AAPL_rev_quarterly.csv")
