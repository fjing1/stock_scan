import urllib.request, json, os, re, time
UA={"User-Agent":"Research Analyst research.analyst.fj@gmail.com"}
def get(u): return urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=120).read()
sub=json.loads(get("https://data.sec.gov/submissions/CIK0001652044.json"))
rec=sub["filings"]["recent"]
rows=list(zip(rec["form"],rec["filingDate"],rec["reportDate"],rec["accessionNumber"],rec["primaryDocument"]))
tenk=[r for r in rows if r[0]=="10-K"]
print("Alphabet 10-Ks:",tenk[:3])
tenq=[r for r in rows if r[0]=="10-Q"][:2]
print("Alphabet 10-Qs:",tenq)
for form,fd,rd,acc,doc in (tenk[:1]+tenq[:1]):
    a=acc.replace("-","")
    url=f"https://www.sec.gov/Archives/edgar/data/1652044/{a}/{doc}"
    p=f"_fd_AAPL_GOOGL_{form.replace('-','')}_{rd}.htm"
    if not os.path.exists(p):
        open(p,"wb").write(get(url)); time.sleep(0.4)
    raw=open(p,encoding="utf-8",errors="ignore").read()
    t=re.sub(r"<[^>]+>"," ",raw); t=re.sub(r"&nbsp;?"," ",t); t=re.sub(r"&#\d+;"," ",t)
    t=re.sub(r"&amp;","&",t); t=re.sub(r"[ \t]+"," ",t)
    tp=p.replace(".htm",".txt"); open(tp,"w").write(t)
    print("\n"+"="*100); print(form, rd, "filed", fd, "acc", acc, "->", tp)
    for m in re.finditer(r"[Tt]raffic acquisition cost", t):
        s=max(0,m.start()-700); print("   ...", t[s:m.start()+1100].strip()[:1800], "\n   ---")
        break
    # find TAC numbers table
    for m in re.finditer(r"TAC", t):
        s=max(0,m.start()-400); seg=t[s:m.start()+900]
        if re.search(r"\d{2},\d{3}", seg):
            print("   [TAC-with-numbers] ...", seg.strip()[:1400], "\n   ---")
            break
