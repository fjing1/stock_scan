"""Remaining Part-1 sweep on the hosts this machine can actually reach: GitHub + HuggingFace,
for any BULK historical option-chain / IV dataset. Verify by downloading, not by linking."""
import requests, json, io, time, pandas as pd
H={"User-Agent":"Mozilla/5.0","Accept":"application/vnd.github+json"}
print("="*106); print("GITHUB REPO SEARCH (api.github.com reachable)"); print("="*106)
Q=["historical option chain implied volatility dataset",
   "options eod implied volatility csv dataset",
   "optionsdx data","spx options historical data","option greeks historical dataset",
   "implied volatility surface dataset"]
seen={}
for q in Q:
    r=requests.get("https://api.github.com/search/repositories",
                   params={"q":q,"sort":"stars","per_page":8},headers=H,timeout=40)
    print(f"\n-- '{q}'  HTTP {r.status_code}")
    if r.status_code!=200: print("   ",r.text[:160]); time.sleep(3); continue
    for it in r.json().get("items",[]):
        k=it["full_name"]
        if k in seen: continue
        seen[k]=it
        print(f"   {it['stargazers_count']:>5}* {k:<48} {(it['description'] or '')[:70]}")
    time.sleep(4)
print("\n"+"="*106); print("GITHUB RELEASE ASSETS: do any of those repos ship actual DATA files?"); print("="*106)
for k in list(seen)[:22]:
    try:
        r=requests.get(f"https://api.github.com/repos/{k}/releases",headers=H,timeout=30)
        if r.status_code!=200: continue
        rel=r.json()
        assets=[(a["name"],a["size"]) for x in rel for a in x.get("assets",[])]
        if assets:
            tot=sum(s for _,s in assets)
            print(f"   {k:<48} {len(assets)} assets, {tot/1e6:.1f} MB  e.g. {assets[:3]}")
    except Exception: pass
    time.sleep(0.6)
print("\n"+"="*106); print("HUGGINGFACE: broader dataset sweep for option/IV panels"); print("="*106)
HH={"User-Agent":"Mozilla/5.0"}
cands={}
for q in ["option","options iv","implied vol","volatility","stock options","derivatives",
          "SPY options","greeks","optionmetrics","iv surface","equity options"]:
    r=requests.get("https://huggingface.co/api/datasets",params={"search":q,"limit":40},headers=HH,timeout=40)
    if r.status_code!=200: continue
    for d in r.json():
        i=d["id"].lower()
        if any(w in i for w in ["option","volatil","iv_","_iv","greek","derivativ"]) and \
           not any(w in i for w in ["medqa","usmle","confusing","choice","theorem","xtts","ranking_options"]):
            cands[d["id"]]=d.get("downloads",0)
    time.sleep(0.4)
print(f"candidate finance option datasets: {len(cands)}")
for k,v in sorted(cands.items(),key=lambda x:-x[1]):
    r=requests.get(f"https://huggingface.co/api/datasets/{k}",headers=HH,timeout=30)
    if r.status_code!=200: print(f"   {k:<52} meta {r.status_code}"); continue
    j=r.json(); sib=[s["rfilename"] for s in j.get("siblings",[])]
    data=[f for f in sib if f.endswith((".csv",".parquet",".arrow",".json",".csv.gz"))]
    print(f"   {k:<52} dl={v:<7} files={len(sib):<4} data={data[:3]}")
    time.sleep(0.3)
