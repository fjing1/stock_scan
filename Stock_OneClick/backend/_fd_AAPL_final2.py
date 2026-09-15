import importlib.util, re
spec=importlib.util.spec_from_file_location("p","_fd_AAPL_parse.py"); P=importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
F=P.parse("_fd_AAPL_10Q_FY26Q3_inst.xml")
RT="RevenueFromContractWithCustomerExcludingAssessedTax"
segs=["AmericasSegmentMember","EuropeSegmentMember","GreaterChinaSegmentMember","JapanSegmentMember","RestOfAsiaPacificSegmentMember"]
def g(tag,ax,m,key):
    for f in F:
        d=f["dims"]
        if f["tag"]!=tag or len(d)!=1 or d.get(ax)!=m: continue
        if (f["per"].get("endDate"),f["per"].get("startDate"))!=key: continue
        return f["val"]/1e6
Q26=("2026-06-27","2026-03-29"); Q25=("2025-06-28","2025-03-30")
N26=("2026-06-27","2025-09-28"); N25=("2025-06-28","2024-09-29")
print("FACT: segment revenue, 10-Q Q3 FY2026 (acc 0000320193-26-000020)")
print(f"{'segment':32s} {'Q3FY25':>9s} {'Q3FY26':>9s} {'YoY':>8s}   {'9MFY25':>9s} {'9MFY26':>9s} {'YoY':>8s}")
for m in segs:
    a=g(RT,"StatementBusinessSegmentsAxis",m,Q25); b=g(RT,"StatementBusinessSegmentsAxis",m,Q26)
    c=g(RT,"StatementBusinessSegmentsAxis",m,N25); d=g(RT,"StatementBusinessSegmentsAxis",m,N26)
    print(f"{m:32s} {a:9,.0f} {b:9,.0f} {100*(b/a-1):+7.1f}%   {c:9,.0f} {d:9,.0f} {100*(d/c-1):+7.1f}%")
print()
print("FACT: Greater China revenue, full-year history (10-Ks)")
for y,v in [("FY2022",74200),("FY2023",72559),("FY2024",66952),("FY2025",64377)]: print("  ",y,f"{v:,}M")
print()
q=open("_fd_AAPL_10Q_FY26Q3.txt",encoding="utf-8",errors="ignore").read()
k=open("_fd_AAPL_10K_FY2025.txt",encoding="utf-8",errors="ignore").read()
pr=open("_fd_AAPL_8K_ex991_Q3FY26.txt",encoding="utf-8",errors="ignore").read()
print("FACT: AI product naming in primary docs")
for kw in ["Apple Intelligence","Siri AI","Siri","artificial intelligence"]:
    print(f"  '{kw}': 10-K FY2025 {len(re.findall(kw,k))} | 10-Q Q3FY26 {len(re.findall(kw,q))} | PR Q3FY26 {len(re.findall(kw,pr))}")
print()
print("FACT: 9M FY2026 cash flow quality")
ni=101464; ocf=116996; capex=6799
ni_p=84544; ocf_p=81754; capex_p=9473
wc = 8316+5671-5461-16266-5203+10016
wc_p= 5685+13555+1223-6116-18479-15161
print(f"  net income {ni:,} -> OCF {ocf:,}  (OCF/NI {ocf/ni:.2f});  prior yr NI {ni_p:,} OCF {ocf_p:,} (OCF/NI {ocf_p/ni_p:.2f})")
print(f"  capex {capex:,} vs prior {capex_p:,}  ({100*(capex/capex_p-1):+.0f}%)   FCF {ocf-capex:,} vs {ocf_p-capex_p:,} ({100*((ocf-capex)/(ocf_p-capex_p)-1):+.0f}%)")
print(f"  working-capital delta this yr {wc:+,}  prior yr {wc_p:+,}  -> favourable swing {wc-wc_p:+,}M")
print(f"  OCF increase {ocf-ocf_p:+,}M of which WC swing {wc-wc_p:+,}M = {100*(wc-wc_p)/(ocf-ocf_p):.0f}%")
