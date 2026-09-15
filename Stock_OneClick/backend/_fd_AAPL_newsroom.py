"""Fetch Apple's own newsroom articles for the claimed-catalyst event dates and product PRICES.
Dates are read off the page, never recalled. Files saved for audit."""
import re, os, time, requests, json
H={"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"}
SLUGS=[
 ("2024","06","introducing-apple-intelligence-for-iphone-ipad-and-mac","AI"),
 ("2024","09","apple-debuts-iphone-16-pro-and-iphone-16-pro-max","HW"),
 ("2024","10","apple-intelligence-is-available-today-on-iphone-ipad-and-mac","AI"),
 ("2025","06","apple-intelligence-gets-even-more-powerful-with-new-capabilities-across-apple-devices","AI"),
 ("2025","09","apple-debuts-iphone-17","HW"),
 ("2025","09","apple-unveils-iphone-17-pro-and-iphone-17-pro-max","HW"),
 ("2025","09","introducing-iphone-air-a-powerful-new-iphone-with-a-breakthrough-design","HW"),
 ("2025","09","new-apple-intelligence-features-are-available-today","AI"),
 ("2026","03","apple-introduces-iphone-17e","HW"),
 ("2026","06","apple-unveils-next-generation-of-apple-intelligence-siri-ai-and-more","AI"),
 ("2026","06","apple-introduces-siri-ai-a-profoundly-more-capable-and-personal-assistant","AI"),
 ("2026","06","due-to-dma-siri-ai-delayed-in-eu-for-ios-27-and-ipados-27","AI"),
 ("2026","09","apple-debuts-iphone-18-pro-and-iphone-18-pro-max","HW"),
 ("2026","09","apple-unveils-iphone-duo","HW"),
 ("2026","09","siri-ai-a-profoundly-more-capable-and-personal-assistant-is-here","AI"),
]
out=[]
for y,m,slug,kind in SLUGS:
    url=f"https://www.apple.com/newsroom/{y}/{m}/{slug}/"
    loc=f"/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/_fd_AAPL_news_{y}{m}_{slug[:60]}.html"
    if os.path.exists(loc): raw=open(loc,encoding="utf-8",errors="ignore").read()
    else:
        r=requests.get(url,headers=H,timeout=40); time.sleep(0.4)
        if r.status_code!=200: print("FAIL",r.status_code,url); continue
        raw=r.text; open(loc,"w").write(raw)
    t=re.sub(r"<[^>]+>"," ",raw); t=re.sub(r"&#\d+;|&nbsp;?|&amp;"," ",t); t=re.sub(r"\s+"," ",t)
    # date: prefer the JSON-LD / meta published date, then the visible dateline
    d=None
    for pat in [r'"datePublished"\s*:\s*"([0-9]{4}-[0-9]{2}-[0-9]{2})',
                r'property="article:published_time"\s+content="([0-9]{4}-[0-9]{2}-[0-9]{2})',
                r'name="apple:published"\s+content="([0-9]{4}-[0-9]{2}-[0-9]{2})']:
        mm=re.search(pat,raw)
        if mm: d=mm.group(1); break
    if d is None:
        mm=re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s+(20\d\d)",t)
        d=f"{mm.group(3)}-{mm.group(1)[:3]}-{mm.group(2)}" if mm else "?"
    prices=sorted(set(re.findall(r"\$[0-9],?[0-9]{2,3}(?:\.\d\d)?", t)))
    out.append({"kind":kind,"date":d,"slug":slug,"prices":prices[:14],"headline":t[:150]})
    print(f"{kind}  {d}  {slug}")
    print(f"      prices seen: {prices[:14]}")
json.dump(out, open("/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/_fd_AAPL_news_events.json","w"), indent=1)
