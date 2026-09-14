import re, time, requests, pandas as pd, datetime as dt
H={"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
rows=[]; page=1
pat=re.compile(r'<a href="(/newsroom/\d{4}/\d{2}/[^"]+)"[^>]*aria-label="([^"]*?) - ([A-Z ]+) - ([^"]*)"',re.S)
while page<=120:
    r=requests.get("https://www.apple.com/newsroom/archive/",params={"page":page},headers=H,timeout=30)
    if r.status_code!=200: print("stop http",r.status_code,page); break
    ms=pat.findall(r.text)
    if not ms: print("no matches page",page); break
    for u,d,cat,head in ms:
        rows.append({"page":page,"url":u,"date_raw":d.strip(),"category":cat.strip(),"headline":' '.join(head.split())})
    last=rows[-1]["date_raw"]
    if page%10==0 or page<3: print(page,"->",last,len(rows),flush=True)
    try:
        if dt.datetime.strptime(last,"%B %d, %Y") < dt.datetime(2023,8,1): print("reached",last); break
    except Exception as e: print("parsefail",last,e)
    page+=1; time.sleep(0.35)
df=pd.DataFrame(rows).drop_duplicates(subset="url")
df["date"]=pd.to_datetime(df.date_raw,format="%B %d, %Y",errors="coerce")
df=df.sort_values("date")
df.to_csv("_arm_apple_newsroom_items.csv",index=False)
print("TOTAL",len(df), df.date.min(), df.date.max())
