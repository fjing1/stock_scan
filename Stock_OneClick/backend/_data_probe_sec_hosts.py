"""Measure which SEC hosts actually answer through the shared corporate-proxy egress IP."""
import time

import requests

S = requests.Session()
S.headers.update({"User-Agent": "stock_scan research feijing@users.noreply.github.com"})

TESTS = [
    ("www.sec.gov/files/company_tickers.json",
     "https://www.sec.gov/files/company_tickers.json"),
    ("www.sec.gov/files/company_tickers_exchange.json",
     "https://www.sec.gov/files/company_tickers_exchange.json"),
    ("www.sec.gov Archives full-index",
     "https://www.sec.gov/Archives/edgar/full-index/2015/QTR1/company.idx"),
    ("data.sec.gov submissions AAPL",
     "https://data.sec.gov/submissions/CIK0000320193.json"),
    ("efts.sec.gov full-text-search",
     "https://efts.sec.gov/LATEST/search-index?q=&forms=25-NSE&dateRange=custom"
     "&startdt=2023-01-01&enddt=2023-03-31"),
]

for label, url in TESTS:
    codes = []
    for attempt in range(6):
        try:
            r = S.get(url, timeout=60)
            codes.append((r.status_code, len(r.content)))
            if r.status_code == 200:
                break
        except Exception as e:
            codes.append((type(e).__name__, 0))
        time.sleep(2.0)
    print(f"{label:44s} attempts={codes}")
