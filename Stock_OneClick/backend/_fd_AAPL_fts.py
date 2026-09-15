"""EDGAR full-text search, restricted to Apple's own filings: where does the company itself
use AI language, and where does it name product mix? Counts are the deliverable, not vibes."""
import requests, json, time, pandas as pd
H={"User-Agent":"stock_scan research fjresearch@gmail.com","Accept-Encoding":"gzip, deflate"}
def fts(q, dfrom="2001-01-01", dto="2026-09-14", cik="0000320193"):
    out=[]
    for frm in range(0, 200, 10):
        r=requests.get("https://efts.sec.gov/LATEST/search-index",headers=H,
                       params={"q":q,"ciks":cik,"startdt":dfrom,"enddt":dto,"from":frm},timeout=30)
        time.sleep(0.25)
        if r.status_code!=200: break
        j=r.json(); hits=j["hits"]["hits"]
        if not hits: break
        for h in hits:
            s=h["_source"]
            out.append({"q":q,"form":s.get("root_form") or s.get("file_type"),
                        "filed":s.get("file_date"),"accn":h["_id"].split(":")[0],
                        "doc":h["_id"].split(":")[-1]})
        if frm+10 >= j["hits"]["total"]["value"]: break
    return pd.DataFrame(out), (j["hits"]["total"]["value"] if r.status_code==200 else None)

for q in ['"Apple Intelligence"','"artificial intelligence"','"Pro models"','"generative AI"']:
    d, tot = fts(q)
    print(f"\n===== {q}: total hits in AAPL filings = {tot} =====")
    if len(d):
        d["filed"]=pd.to_datetime(d.filed)
        print(d.sort_values("filed")[["form","filed","accn","doc"]].to_string(index=False))
        print("by form:", d.form.value_counts().to_dict())
