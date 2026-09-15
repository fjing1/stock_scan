import urllib.request, json, re, time
UA={"User-Agent":"Research Analyst research.analyst.fj@gmail.com"}
def get(u): return urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=120).read()
sub=json.loads(get("https://data.sec.gov/submissions/CIK0001067983.json"))
rec=sub["filings"]["recent"]
rows=[r for r in zip(rec["form"],rec["filingDate"],rec["reportDate"],rec["accessionNumber"]) if r[0]=="13F-HR"]
print("period      filed        accession                AAPL shares      value($k)   implied px   total 13F value($bn)  AAPL % of 13F")
out=[]
for form,fd,rd,acc in rows[:10]:
    a=acc.replace("-","")
    idx=json.loads(get(f"https://www.sec.gov/Archives/edgar/data/1067983/{a}/index.json"))
    names=[i["name"] for i in idx["directory"]["item"] if i["name"].lower().endswith(".xml")]
    body=None
    for x in names:
        b=get(f"https://www.sec.gov/Archives/edgar/data/1067983/{a}/{x}").decode("utf-8","ignore")
        if "infoTable" in b: body=b; break
    if not body: print(rd,"no infotable"); continue
    blocks=re.findall(r"<(?:\w+:)?infoTable>(.*?)</(?:\w+:)?infoTable>", body, re.S)
    tot=0.0; aapl_sh=0; aapl_val=0.0
    for blk in blocks:
        nm=re.search(r"<(?:\w+:)?nameOfIssuer>([^<]+)<",blk)
        v=re.search(r"<(?:\w+:)?value>([\d\.]+)<",blk)
        s=re.search(r"<(?:\w+:)?sshPrnamt>([\d\.]+)<",blk)
        cu=re.search(r"<(?:\w+:)?cusip>([^<]+)<",blk)
        if not (nm and v and s): continue
        tot+=float(v.group(1))
        if cu and cu.group(1).strip().upper().startswith("037833"):
            aapl_sh+=int(float(s.group(1))); aapl_val+=float(v.group(1))
    px=aapl_val*1000/aapl_sh if aapl_sh else 0
    print(f"{rd}  {fd}  {acc}  {aapl_sh:>14,}  {aapl_val:>13,.0f}  {px:9.2f}   {tot/1e6:16,.1f}   {100*aapl_val/tot:11.1f}%   (nblocks {len(blocks)})")
    out.append((rd,aapl_sh))
    time.sleep(0.3)
print("\nFACT: Berkshire AAPL share count by quarter (CUSIP 037833100):")
for rd,sh in out: print("  ",rd, f"{sh:,}")
