import urllib.request, urllib.parse, json, os
UA={"User-Agent":"Research Analyst research.analyst.fj@gmail.com"}
def get(u,hdr=None):
    h=dict(UA); h.update(hdr or {})
    return urllib.request.urlopen(urllib.request.Request(u,headers=h),timeout=90).read()
# CourtListener RECAP search for US v Google remedies + liability opinions
q="https://www.courtlistener.com/api/rest/v4/search/?"+urllib.parse.urlencode({
 "q":"Google search distribution Apple Information Services Agreement",
 "type":"r","docket_number":"1:20-cv-03010","order_by":"score desc"})
try:
    r=json.loads(get(q))
    print("count",r.get("count"))
    for x in r.get("results",[])[:10]:
        print(x.get("dateFiled"), x.get("caseName"), x.get("docketNumber"))
        for d in x.get("recap_documents",[])[:5]:
            print("    ", d.get("document_number"), d.get("description","")[:90], d.get("filepath_local"))
except Exception as e:
    print("CL search fail", e)
