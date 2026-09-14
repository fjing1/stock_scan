"""Fetch ARM earnings exhibits (99.x) from the 6-K accessions."""
import json
import time
import requests

UA = {"User-Agent": "independent research analysis research@example.com"}
BASE = "https://www.sec.gov/Archives/edgar/data/1973239"

ACCS = {
    "0001973239-26-000113": "Q1FY27_20260729",
    "0001973239-26-000062": "Q4FY26_20260506",
    "0001973239-26-000005": "Q3FY26_20260204",
    "0001973239-25-000042": "Q2FY26_20251105",
    "0001973239-25-000023": "Q1FY26_20250730",
    "0001973239-25-000010": "Q4FY25_20250507",
    "0001973239-25-000006": "Q3FY25_20250205",
    "0001973239-24-000038": "Q2FY25_20241107",
    "0001973239-24-000017": "Q1FY25_20240731",
    "0001973239-24-000007": "Q4FY24_20240508",
    "0001973239-24-000002": "Q3FY24_20240207",
    "0001973239-23-000009": "Q2FY24_20231108",
}

for acc, label in ACCS.items():
    nod = acc.replace("-", "")
    idx = requests.get(f"{BASE}/{nod}/index.json", headers=UA, timeout=60).json()
    time.sleep(0.2)
    for item in idx["directory"]["item"]:
        n = item["name"]
        if n.startswith("ex") or "ex99" in n or "exhibit" in n.lower():
            if not n.endswith((".htm", ".html", ".txt")):
                continue
            r = requests.get(f"{BASE}/{nod}/{n}", headers=UA, timeout=60)
            out = f"_arm_ex_{label}_{n}"
            open(out, "wb").write(r.content)
            print(f"OK {out:55s} {len(r.content):>9,}")
            time.sleep(0.2)
print("done")
