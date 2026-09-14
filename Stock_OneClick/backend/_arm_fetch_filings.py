"""Fetch ARM primary SEC filings to disk for audit. CIK 0001973239."""
import time
import requests

UA = {"User-Agent": "independent research analysis research@example.com"}
BASE = "https://www.sec.gov/Archives/edgar/data/1973239"

JOBS = [
    # (accession, primary doc, local name)
    ("0001973239-26-000097", "arm-20260331.htm", "_arm_20F_FY2026.htm"),
    ("0001973239-25-000016", "arm-20250331.htm", "_arm_20F_FY2025.htm"),
    ("0001973239-24-000012", "arm-20240331.htm", "_arm_20F_FY2024.htm"),
    ("0001973239-26-000114", "arm-20260630.htm", "_arm_6K_Q1FY27_fin.htm"),
    ("0001973239-26-000113", "arm-20260729.htm", "_arm_6K_20260729.htm"),
    ("0001973239-26-000128", "arm-20260910.htm", "_arm_6K_20260910.htm"),
    ("0001973239-26-000117", "arm-20260810.htm", "_arm_6K_20260810.htm"),
    ("0001973239-26-000062", "arm-20260506.htm", "_arm_6K_20260506.htm"),
    ("0001973239-26-000006", "arm-20251231.htm", "_arm_6K_Q3FY26_fin.htm"),
    ("0001973239-26-000005", "arm-20260204.htm", "_arm_6K_20260204.htm"),
    ("0001973239-25-000043", "arm-20250930.htm", "_arm_6K_Q2FY26_fin.htm"),
]

for acc, doc, out in JOBS:
    url = f"{BASE}/{acc.replace('-', '')}/{doc}"
    r = requests.get(url, headers=UA, timeout=60)
    if r.status_code == 200:
        open(out, "wb").write(r.content)
        print(f"OK   {out:34s} {len(r.content):>9,}  {url}")
    else:
        print(f"FAIL {out:34s} {r.status_code}  {url}")
    time.sleep(0.25)

# also grab the filing index for each so exhibits are discoverable
for acc, doc, out in JOBS:
    url = f"{BASE}/{acc.replace('-', '')}/index.json"
    r = requests.get(url, headers=UA, timeout=60)
    if r.status_code == 200:
        open(out.replace(".htm", "_index.json"), "wb").write(r.content)
    time.sleep(0.25)
print("done")
