import json, os, time, requests, re
UA={"User-Agent":"independent research fj@example.com"}
d=json.load(open('_arm_submissions.json'))
r=d['filings']['recent']
rows=list(zip(r['form'],r['filingDate'],r['reportDate'],r['accessionNumber']))
res=[]
for form,fd,rd,acc in rows:
    if form!='6-K': continue
    a=acc.replace('-','')
    j=requests.get(f"https://www.sec.gov/Archives/edgar/data/1973239/{a}/index.json",headers=UA,timeout=60).json()
    names=[i['name'] for i in j['directory']['item']]
    print(fd, rd, names)
    time.sleep(0.3)
