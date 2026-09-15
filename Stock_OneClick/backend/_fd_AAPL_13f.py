import urllib.request, json, re, os, time
UA={"User-Agent":"Research Analyst research.analyst.fj@gmail.com"}
def get(u): return urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=120).read()
# Berkshire Hathaway CIK 1067983
sub=json.loads(get("https://data.sec.gov/submissions/CIK0001067983.json"))
rec=sub["filings"]["recent"]
rows=[r for r in zip(rec["form"],rec["filingDate"],rec["reportDate"],rec["accessionNumber"]) if r[0].startswith("13F-HR")]
print("Berkshire 13F-HR filings (recent):", rows[:10])
for form,fd,rd,acc in rows[:9]:
    a=acc.replace("-","")
    try:
        idx=json.loads(get(f"https://www.sec.gov/Archives/edgar/data/1067983/{a}/index.json"))
    except Exception as e:
        print(rd,"idx fail",e); continue
    xml=[i["name"] for i in idx["directory"]["item"] if i["name"].lower().endswith(".xml") and "primary_doc" not in i["name"].lower()]
    got=False
    for x in xml:
        try: b=get(f"https://www.sec.gov/Archives/edgar/data/1067983/{a}/{x}").decode("utf-8","ignore")
        except: continue
        if "APPLE" not in b.upper(): continue
        for m in re.finditer(r"<infoTable>(.*?)</infoTable>", b, re.S):
            blk=m.group(1)
            if "APPLE INC" not in blk.upper(): continue
            val=re.search(r"<value>([\d\.]+)</value>",blk); sh=re.search(r"<sshPrnamt>([\d\.]+)</sshPrnamt>",blk)
            cls=re.search(r"<titleOfClass>([^<]+)</titleOfClass>",blk)
            print(f"  {rd}  filed {fd}  acc {acc}  class={cls.group(1) if cls else '?'}  value=${float(val.group(1))/1e3:,.1f}M(k$)  shares={int(float(sh.group(1))):,}")
            got=True
        if got: break
    if not got: print(f"  {rd} filed {fd} acc {acc}: no APPLE INC line found")
    time.sleep(0.3)
