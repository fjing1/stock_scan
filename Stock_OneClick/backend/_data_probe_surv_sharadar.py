"""Pin down the ACTUAL current Sharadar SEP price (the repo's flagged 'biggest paid unlock')."""
import json
import re
import time

import requests

B = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
     "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
S = requests.Session()
S.headers.update({"User-Agent": B, "Accept": "application/json,text/html,*/*"})


def show(u, r, n=500):
    print(f"  {u[:100]}\n    -> HTTP{r.status_code} {len(r.content):,}B {r.text[:n]!r}")


print("=" * 78, "\n1. Nasdaq Data Link marketplace/product API candidates\n", "=" * 78)
CANDS = [
    "https://data.nasdaq.com/api/v3/marketplace/products/SFA",
    "https://data.nasdaq.com/api/v3/marketplace/databases/SFA",
    "https://data.nasdaq.com/api/v3/products?database_code=SFA",
    "https://data.nasdaq.com/api/v3/databases/SFA.json",
    "https://data.nasdaq.com/api/v3/datatables/SHARADAR/TICKERS/metadata.json",
    "https://data.nasdaq.com/api/v3/datatables/SHARADAR/ACTIONS/metadata.json",
]
for u in CANDS:
    try:
        r = S.get(u, timeout=45)
        show(u, r, 400)
    except Exception as e:
        print(f"  {u} EXC {e}")
    time.sleep(0.4)

print("\n" + "=" * 78, "\n2. Wayback: prerendered Sharadar product pages with price text\n", "=" * 78)
W = requests.Session()
W.headers.update({"User-Agent": B})
for target in ["data.nasdaq.com/databases/SFA*", "www.quandl.com/databases/SFA*",
               "data.nasdaq.com/databases/SEP*"]:
    try:
        r = W.get("https://web.archive.org/cdx/search/cdx",
                  params={"url": target, "output": "json",
                          "fl": "timestamp,original,statuscode,length",
                          "filter": "statuscode:200", "collapse": "timestamp:6",
                          "limit": "60"}, timeout=180)
        print(f"  CDX {target} HTTP{r.status_code} {len(r.content)}B")
        if r.status_code == 200 and r.text.strip().startswith("["):
            rows = json.loads(r.text)[1:]
            print(f"    {len(rows)} snapshots; last few: "
                  f"{[(x[0][:8], x[1][-28:], x[3]) for x in rows[-6:]]}")
            # pick the biggest (most likely prerendered with content)
            rows.sort(key=lambda x: int(x[3] or 0), reverse=True)
            for ts, orig, sc, ln in rows[:3]:
                u = f"https://web.archive.org/web/{ts}id_/{orig}"
                rr = W.get(u, timeout=180)
                t = re.sub(r"<script.*?</script>", " ", rr.text, flags=re.S | re.I)
                t = re.sub(r"<style.*?</style>", " ", t, flags=re.S | re.I)
                t = re.sub(r"<[^>]+>", " ", t)
                t = re.sub(r"\s+", " ", t)
                hits = re.findall(r".{0,90}\$\s?[\d,]+(?:\.\d\d)?.{0,70}", t)
                print(f"    snap {ts[:8]} len={len(rr.content):,} textlen={len(t):,} "
                      f"dollar_hits={len(hits)}")
                for h in hits[:12]:
                    print(f"       $ {h.strip()[:170]}")
                if hits:
                    break
                time.sleep(1.0)
    except Exception as e:
        print(f"  {target} EXC {type(e).__name__}: {str(e)[:120]}")
    time.sleep(1.0)

print("\n" + "=" * 78, "\n3. Sharadar's own docs/site (via archive if live is blocked)\n", "=" * 78)
for u in ["https://data.nasdaq.com/databases/SFA/documentation",
          "https://web.archive.org/web/2025/https://data.nasdaq.com/databases/SFA"]:
    try:
        r = S.get(u, timeout=90, allow_redirects=True)
        t = re.sub(r"<[^>]+>", " ", r.text)
        t = re.sub(r"\s+", " ", t)
        hits = re.findall(r".{0,80}\$\s?[\d,]+(?:\.\d\d)?.{0,60}", t)
        print(f"  {u[:80]} HTTP{r.status_code} {len(r.content):,}B textlen={len(t):,} "
              f"dollar_hits={len(hits)}")
        for h in hits[:10]:
            print(f"     $ {h.strip()[:170]}")
    except Exception as e:
        print(f"  {u} EXC {type(e).__name__}: {str(e)[:100]}")
    time.sleep(0.8)

print("\nDONE")
