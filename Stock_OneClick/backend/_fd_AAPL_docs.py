import urllib.request, os, time, re
UA={"User-Agent":"Research Analyst research.analyst.fj@gmail.com"}
def get(u):
    return urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=120).read()
docs={
 "10K_FY2025":"https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm",
 "10Q_FY26Q3":"https://www.sec.gov/Archives/edgar/data/320193/000032019326000020/aapl-20260627.htm",
 "8K_FY26Q3_ER":"https://www.sec.gov/Archives/edgar/data/320193/000032019326000018/aapl-20260730.htm",
}
for k,u in docs.items():
    p=f"_fd_AAPL_{k}.htm"
    if not os.path.exists(p):
        open(p,"wb").write(get(u)); time.sleep(0.4)
    raw=open(p,encoding="utf-8",errors="ignore").read()
    txt=re.sub(r"<[^>]+>"," ",raw); txt=re.sub(r"&nbsp;?"," ",txt); txt=re.sub(r"&amp;","&",txt)
    txt=re.sub(r"&#\d+;"," ",txt); txt=re.sub(r"[ \t]+"," ",txt); txt=re.sub(r"\n\s*\n+","\n",txt)
    open(f"_fd_AAPL_{k}.txt","w").write(txt)
    print(k, len(raw), "->", len(txt))
