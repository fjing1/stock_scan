import json, urllib.request, gzip, os
UA = {"User-Agent": "Research Analyst research.analyst.fj@gmail.com"}
def get(url):
    r = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(r, timeout=60).read()
sub = json.loads(get("https://data.sec.gov/submissions/CIK0000320193.json"))
rec = sub["filings"]["recent"]
keys = ["form","filingDate","reportDate","accessionNumber","primaryDocument"]
rows = list(zip(*[rec[k] for k in keys]))
print("form counts (recent):")
from collections import Counter
print(Counter(r[0] for r in rows).most_common(12))
print()
for r in rows:
    if r[0] in ("10-K","10-Q","8-K") and r[1] >= "2025-01-01":
        print(r)
