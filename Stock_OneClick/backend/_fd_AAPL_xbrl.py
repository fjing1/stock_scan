import json, urllib.request, os, time
UA = {"User-Agent": "Research Analyst research.analyst.fj@gmail.com"}
def get(url):
    r = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(r, timeout=120).read()
targets = {
 "10K_FY2025": ("0000320193-25-000079", "aapl-20250927"),
 "10Q_FY26Q3": ("0000320193-26-000020", "aapl-20260627"),
 "10K_FY2024": ("0000320193-24-000123", "aapl-20240928"),
}
for label,(acc,stem) in targets.items():
    a = acc.replace("-","")
    base = f"https://www.sec.gov/Archives/edgar/data/320193/{a}"
    out = f"_fd_AAPL_{label}_inst.xml"
    if os.path.exists(out) and os.path.getsize(out)>100000:
        print("have", out); continue
    for cand in [f"{stem}_htm.xml", f"{stem}.xml"]:
        try:
            b = get(f"{base}/{cand}")
            open(out,"wb").write(b)
            print(label, cand, len(b)); break
        except Exception as e:
            print(label, cand, "FAIL", e)
    time.sleep(0.4)
