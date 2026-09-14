import json, os, time, requests
UA={"User-Agent":"independent research fj@example.com"}
d=json.load(open('_arm_submissions.json'))
r=d['filings']['recent']
rows=list(zip(r['form'],r['filingDate'],r['reportDate'],r['accessionNumber'],r['primaryDocument']))
want=[x for x in rows if x[0]=='6-K']
for form,fd,rd,acc,doc in want:
    a=acc.replace('-','')
    out=f"_arm_6k_{fd}_{rd or 'na'}.htm"
    if os.path.exists(out) and os.path.getsize(out)>1000:
        continue
    url=f"https://www.sec.gov/Archives/edgar/data/1973239/{a}/{doc}"
    resp=requests.get(url,headers=UA,timeout=60)
    print(resp.status_code,len(resp.content),out)
    if resp.status_code==200: open(out,'wb').write(resp.content)
    time.sleep(0.35)
