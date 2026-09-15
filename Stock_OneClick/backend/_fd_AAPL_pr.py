"""Apple's own quarterly earnings press releases (8-K EX-99.1) -- the document investors actually
read. Extract the CEO/CFO quotes and any forward guidance, and count AI language."""
import requests, re, json, time, os
H={"User-Agent":"stock_scan research fjresearch@gmail.com","Accept-Encoding":"gzip, deflate"}
CIK="320193"
EIGHTKS=[("2026-07-30","0000320193-26-000018"),("2026-04-30","0000320193-26-000011"),
         ("2026-01-29","0000320193-26-000005"),("2025-10-30","0000320193-25-000077"),
         ("2025-07-31","0000320193-25-000071"),("2025-05-01","0000320193-25-000055"),
         ("2025-01-30","0000320193-25-000007"),("2024-10-31","0000320193-24-000120")]
for dt,accn in EIGHTKS:
    nod=accn.replace("-","")
    base=f"https://www.sec.gov/Archives/edgar/data/{CIK}/{nod}"
    idx=requests.get(base+"/index.json",headers=H,timeout=30); time.sleep(0.25)
    names=[i["name"] for i in idx.json()["directory"]["item"]]
    ex=[n for n in names if "ex991" in n.lower() or "exhibit991" in n.lower()]
    if not ex: print(dt,"no ex99.1:",names); continue
    fn=ex[0]
    loc=f"/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/_fd_AAPL_pr_{dt}_{fn}"
    if os.path.exists(loc): raw=open(loc,encoding="utf-8",errors="ignore").read()
    else:
        raw=requests.get(f"{base}/{fn}",headers=H,timeout=60).text; time.sleep(0.25)
        open(loc,"w").write(raw)
    t=re.sub(r"<[^>]+>"," ",raw); t=re.sub(r"&#\d+;|&nbsp;?|&amp;"," ",t); t=re.sub(r"\s+"," ",t)
    print(f"\n{'='*118}\n8-K {dt}  accn {accn}  exhibit {fn}\n{'='*118}")
    print(t[:1900])
    for kw in ["Apple Intelligence","artificial intelligence"," AI ","guidance","expect"]:
        n=len(re.findall(re.escape(kw),t))
        print(f"   [count] {kw!r}: {n}")
