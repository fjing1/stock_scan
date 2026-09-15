import importlib.util, json, gzip, datetime as dt
spec=importlib.util.spec_from_file_location("p","_fd_AAPL_parse.py"); P=importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
Q=P.parse("_fd_AAPL_10Q_FY26Q3_inst.xml"); K=P.parse("_fd_AAPL_10K_FY2025_inst.xml")
def val(F,tag,key=None,inst=None,dei=False):
    out=[]
    for f in F:
        if f["tag"]!=tag or f["dims"]: continue
        p=f["per"]
        if key and (p.get("endDate"),p.get("startDate"))!=key: continue
        if inst and p.get("instant")!=inst: continue
        out.append(f["val"])
    return out
Q3=("2026-06-27","2026-03-29"); N26=("2026-06-27","2025-09-28"); N25=("2025-06-28","2024-09-29")
FY25=("2025-09-27","2024-09-29")
ni_q=val(Q,"NetIncomeLoss",Q3)[0]/1e6
ni_9=val(Q,"NetIncomeLoss",N26)[0]/1e6
ni_925=val(Q,"NetIncomeLoss",N25)[0]/1e6
ni_fy25=val(K,"NetIncomeLoss",FY25)[0]/1e6
eps_9=val(Q,"EarningsPerShareDiluted",N26); eps_q=val(Q,"EarningsPerShareDiluted",Q3)
sh_q=val(Q,"WeightedAverageNumberOfDilutedSharesOutstanding",Q3)[0]/1e6
print("FACT net income: Q3FY26",f"{ni_q:,.0f}M", " 9M FY26",f"{ni_9:,.0f}M"," 9M FY25",f"{ni_925:,.0f}M"," FY25",f"{ni_fy25:,.0f}M")
print("   9M YoY net income:", f"{100*(ni_9/ni_925-1):+.1f}%")
print("   EPS diluted Q3FY26",eps_q," 9M",eps_9," diluted shares Q3",f"{sh_q:,.0f}M")
ttm_ni = (ni_fy25 - ni_925) + ni_9
print(f"TTM net income = (FY25 {ni_fy25:,.0f} - 9M FY25 {ni_925:,.0f}) + 9M FY26 {ni_9:,.0f} = {ttm_ni:,.0f}M")
# shares outstanding (dei on cover)
import re
raw=open("_fd_AAPL_10Q_FY26Q3_inst.xml",encoding="utf-8",errors="ignore").read()
m=re.search(r"EntityCommonStockSharesOutstanding[^>]*>([\d,\.]+)<",raw)
print("dei EntityCommonStockSharesOutstanding (10-Q cover):", m.group(1) if m else "n/a")
so=float(m.group(1)) if m else None
PRICE=333.08
if so:
    mc=so*PRICE/1e6
    print(f"\nMarket cap @ {PRICE} x {so/1e6:,.0f}M sh = ${mc/1e3:,.1f}bn")
    # net cash from 10-Q
    def inst(F,tag,d): 
        v=val(F,tag,inst=d); return v[0]/1e6 if v else None
    D="2026-06-27"
    parts={t:inst(Q,t,D) for t in ["CashAndCashEquivalentsAtCarryingValue","MarketableSecuritiesCurrent","MarketableSecuritiesNoncurrent","CommercialPaper","LongTermDebtCurrent","LongTermDebtNoncurrent","OtherLiabilitiesCurrent"]}
    print(" balance-sheet parts:",{k:(f"{v:,.0f}" if v else None) for k,v in parts.items()})
    cash=(parts["CashAndCashEquivalentsAtCarryingValue"] or 0)+(parts["MarketableSecuritiesCurrent"] or 0)+(parts["MarketableSecuritiesNoncurrent"] or 0)
    debt=(parts["CommercialPaper"] or 0)+(parts["LongTermDebtCurrent"] or 0)+(parts["LongTermDebtNoncurrent"] or 0)
    print(f" cash+investments {cash:,.0f}M  total debt {debt:,.0f}M  NET CASH {cash-debt:+,.0f}M")
    ev=mc-(cash-debt)
    ttm_rev=466823.0
    print(f"\nEV = {ev/1e3:,.1f}bn")
    print(f"EV/Sales on TTM revenue {ttm_rev:,.0f}M = {ev/ttm_rev:.2f}x   (on FY2025 416,161 = {ev/416161:.2f}x)")
    print(f"PE on TTM net income {ttm_ni:,.0f}M = {mc/ttm_ni:.1f}x   (on FY2025 NI = {mc/ni_fy25:.1f}x)")
    # ex tariff-refund
    print(f"\nINFERENCE ex-tariff-refund: Q3 EPS 2.02 - 0.11 = 1.91 -> Q3 'clean' NI ~ {ni_q*(1.91/2.02):,.0f}M (vs reported {ni_q:,.0f}M)")
