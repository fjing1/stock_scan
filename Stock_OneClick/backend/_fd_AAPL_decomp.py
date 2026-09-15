import importlib.util
spec=importlib.util.spec_from_file_location("p","_fd_AAPL_parse.py"); P=importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
F=P.parse("_fd_AAPL_10Q_FY26Q3_inst.xml")
K=P.parse("_fd_AAPL_10K_FY2025_inst.xml")
K4=P.parse("_fd_AAPL_10K_FY2024_inst.xml")
RT="RevenueFromContractWithCustomerExcludingAssessedTax"
def g(facts,tag,ax=None,m=None,key=None):
    for f in facts:
        if f["tag"]!=tag: continue
        d=f["dims"]
        if ax is None:
            if d: continue
        else:
            if len(d)!=1 or d.get(ax)!=m: continue
        if (f["per"].get("endDate"),f["per"].get("startDate"))!=key: continue
        return f["val"]/1e6
    return None
Q26=("2026-06-27","2026-03-29"); Q25=("2025-06-28","2025-03-30")
N26=("2026-06-27","2025-09-28"); N25=("2025-06-28","2024-09-29")
cats=["IPhoneMember","MacMember","IPadMember","WearablesHomeandAccessoriesMember","ServiceMember"]
print("="*104)
print("FACT: Q3 FY2026 (2026-03-29..2026-06-27) vs Q3 FY2025 -- source 10-Q acc 0000320193-26-000020")
t26=g(F,RT,key=Q26); t25=g(F,RT,key=Q25)
print(f"Total revenue {t25:,.0f} -> {t26:,.0f}   YoY {100*(t26/t25-1):+.2f}%   delta {t26-t25:+,.0f}M")
print(f"{'category':38s} {'Q3FY25':>10s} {'Q3FY26':>10s} {'YoY':>8s} {'delta':>10s} {'share of growth':>16s}")
tot_d=t26-t25
for c in cats:
    a=g(F,RT,"ProductOrServiceAxis",c,Q25); b=g(F,RT,"ProductOrServiceAxis",c,Q26)
    print(f"{c:38s} {a:10,.0f} {b:10,.0f} {100*(b/a-1):+7.1f}% {b-a:+10,.0f} {100*(b-a)/tot_d:15.1f}%")
print()
print("FACT: 9-month FY2026 vs FY2025 (same filing)")
n26=g(F,RT,key=N26); n25=g(F,RT,key=N25)
print(f"9M total {n25:,.0f} -> {n26:,.0f}  YoY {100*(n26/n25-1):+.2f}%")
for c in cats:
    a=g(F,RT,"ProductOrServiceAxis",c,N25); b=g(F,RT,"ProductOrServiceAxis",c,N26)
    print(f"  {c:38s} {a:10,.0f} {b:10,.0f} {100*(b/a-1):+7.1f}%   share of 9M growth {100*(b-a)/(n26-n25):5.1f}%")
print()
print("="*104)
print("FACT: gross profit split Products vs Services")
def gp(facts,key,label):
    tp=g(facts,RT,"ProductOrServiceAxis","ProductMember",key); ts=g(facts,RT,"ProductOrServiceAxis","ServiceMember",key)
    cp=g(facts,"CostOfGoodsAndServicesSold","ProductOrServiceAxis","ProductMember",key)
    cs=g(facts,"CostOfGoodsAndServicesSold","ProductOrServiceAxis","ServiceMember",key)
    gpp=tp-cp; gps=ts-cs; tot=gpp+gps
    print(f"{label:16s} rev P {tp:9,.0f} S {ts:9,.0f} | GM% P {100*gpp/tp:5.1f} S {100*gps/ts:5.1f} | GP P {gpp:9,.0f} S {gps:9,.0f} | Services = {100*gps/tot:5.1f}% of GP  (on {100*ts/(tp+ts):4.1f}% of revenue)")
    return gps,tot
gp(K4,("2022-09-24","2021-09-26"),"FY2022")
gp(K4,("2023-09-30","2022-09-25"),"FY2023")
gp(K,("2024-09-28","2023-10-01"),"FY2024")
gp(K,("2025-09-27","2024-09-29"),"FY2025")
gp(F,Q25,"Q3FY25")
gps,tot=gp(F,Q26,"Q3FY26")
gp(F,N26,"9M FY2026")
print()
print("="*104)
print("INFERENCE: TTM revenue (Q4FY25 = FY2025 - 9M FY2025)")
fy25=g(K,RT,key=("2025-09-27","2024-09-29"))
# need 9M FY2025 from FY25 Q3 10-Q -> we have N25 in this filing
q4fy25=fy25-n25
ttm=q4fy25+n26
print(f"FY2025 {fy25:,.0f}  9M FY2025 {n25:,.0f}  => Q4FY25 {q4fy25:,.0f}")
print(f"TTM (Q4FY25 + 9M FY2026) = {ttm:,.0f}M   vs FY2025 {fy25:,.0f}  => TTM growth {100*(ttm/fy25-1):+.2f}%")
print(f"NOTE fund_metrics reported revenue_ttm == revenue_fy == {fy25:,.0f} -> it is NOT rolling TTM; real TTM is {ttm:,.0f}")
print(f"=> EV/Sales on true TTM is lower than the reported 11.6x by factor {ttm/fy25:.4f}")
