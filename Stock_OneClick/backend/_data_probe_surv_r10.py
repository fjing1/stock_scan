"""Capture Sharadar's own terms: exact price tiers, delisted coverage, bulk download,
free sample -- from sharadar.com (they now sell direct, not only via Nasdaq Data Link)."""
import re
import time
import html as H

import requests

BOT = "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
S = requests.Session()
S.headers.update({"User-Agent": BOT})


def text_of(s):
    t = re.sub(r"<script.*?</script>", " ", s, flags=re.S | re.I)
    t = re.sub(r"<style.*?</style>", " ", t, flags=re.S | re.I)
    return re.sub(r"\s+", " ", H.unescape(re.sub(r"<[^>]+>", " ", t)))


for u in ["https://sharadar.com/subscribe", "https://sharadar.com/sample",
          "https://sharadar.com/prices", "https://sharadar.com/bulk",
          "https://sharadar.com/docs"]:
    print("\n" + "=" * 78)
    print(u)
    print("=" * 78)
    try:
        r = S.get(u, timeout=90, allow_redirects=True)
        t = text_of(r.text)
        print(f"  HTTP{r.status_code} raw={len(r.content):,}B text={len(t):,} final={r.url}")
        print(f"\n  FULL TEXT:\n{t[:4200]}")
    except Exception as e:
        print(f"  EXC {type(e).__name__}: {str(e)[:120]}")
    time.sleep(1.2)
