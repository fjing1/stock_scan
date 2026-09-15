"""ADVERSARIAL VERIFY #4 -- the pricing bound. The claim asserts 'the largest list-price rise in the
generation was +10.0% (Pro tier)'. Test it by reading the PRICE AND BASE STORAGE off Apple's own
dated newsroom pages for EVERY tier in both generations, including the tiers the claim's script did
not fetch (iPhone 16 / 16 Plus, which the iPhone Air replaced).
"""
from __future__ import annotations
import glob, html, os, re, time
import requests

H = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126 Safari/537.36"}
EXTRA = [("2024", "09", "apple-debuts-iphone-16-and-iphone-16-plus"),
         ("2025", "09", "apple-debuts-iphone-17")]
for y, m, slug in EXTRA:
    loc = f"_fd_AAPL_verify_news_{y}{m}_{slug}.html"
    if not os.path.exists(loc):
        r = requests.get(f"https://www.apple.com/newsroom/{y}/{m}/{slug}/", headers=H, timeout=40)
        print(f"  fetch {slug} -> HTTP {r.status_code}")
        if r.status_code == 200:
            open(loc, "w").write(r.text)
        time.sleep(0.5)


def clean(path):
    raw = open(path, encoding="utf-8", errors="ignore").read()
    t = re.sub(r"<[^>]+>", " ", raw)
    t = html.unescape(t)
    t = re.sub(r"\s+", " ", t)
    d = None
    for pat in (r'"datePublished"\s*:\s*"([0-9]{4}-[0-9]{2}-[0-9]{2})',
                r'property="article:published_time"\s+content="([0-9]{4}-[0-9]{2}-[0-9]{2})'):
        mm = re.search(pat, raw)
        if mm:
            d = mm.group(1); break
    return d, t


FILES = sorted(glob.glob("_fd_AAPL_news_2024*.html") + glob.glob("_fd_AAPL_news_2025*.html")
               + glob.glob("_fd_AAPL_news_2026*.html") + glob.glob("_fd_AAPL_verify_news_*.html"))
print("=" * 122)
print("PRICE + STORAGE SENTENCES, quoted verbatim from Apple's own dated newsroom pages")
print("=" * 122)
for f in FILES:
    d, t = clean(f)
    if "iphone" not in f.lower():
        continue
    hits = []
    for pat in (r".{0,170}(?:starting at|start at|starts at|pricing (?:and|&) availability).{0,300}",
                r".{0,120}\$\d,?\d{3}\s*\(?U\.?S\.?\)?.{0,220}",
                r".{0,140}(?:128GB|256GB|512GB|1TB).{0,200}"):
        for m in re.finditer(pat, t, re.I):
            s = m.group(0).strip()
            if "$" in s and s not in hits:
                hits.append(s)
    if not hits:
        continue
    print(f"\n--- {os.path.basename(f)}   datePublished={d}")
    for s in hits[:6]:
        print("    ", s[:430])

print("\n" + "=" * 122)
print("TIER-BY-TIER LADDER (US, base storage) -- assembled ONLY from the sentences printed above")
print("=" * 122)
LADDER = [
    # tier                gen-A (Sept 2024)          gen-B (Sept 2025)
    ("entry",            "iPhone 16      $799",      "iPhone 17      $799"),
    ("second / Air slot", "iPhone 16 Plus $899",     "iPhone Air     $999"),
    ("Pro",              "iPhone 16 Pro  $999",      "iPhone 17 Pro  $1,099"),
    ("Pro Max",          "iPhone 16 ProMax $1,199",  "iPhone 17 ProMax $1,199"),
]
pairs = [(799, 799), (899, 999), (999, 1099), (1199, 1199)]
print(f"  {'tier':<18} {'FY25 gen (iPhone 16)':<26} {'FY26 gen (iPhone 17)':<26} list-price change")
mx = 0.0
for (lab, a, b), (pa, pb) in zip(LADDER, pairs):
    ch = pb / pa - 1
    mx = max(mx, ch)
    print(f"  {lab:<18} {a:<26} {b:<26} {ch:+7.1%}")
print(f"\n  >>> LARGEST list-price rise across the ladder = {mx:+.1%}   (claim says +10.0%)")
g = 0.217
print(f"  >>> revised bound: if every unit were the biggest-riser tier, price explains at most {mx:+.1%}")
print(f"      of iPhone's +{g:.1%}, so units/mix must supply at least {(1+g)/(1+mx)-1:+.1%} "
      f"(claim said {(1+g)/1.10-1:+.1%})")
print("\n  STORAGE CAVEAT: the Pro tier's +10.0% came with base storage doubling 128GB -> 256GB.")
print("  On a $/GB basis that is a PRICE CUT. So the +10.0% is itself a mix effect (more storage")
print("  sold per unit), not a pure price rise -- which means the claim's split between 'price' and")
print("  'mix' is not a clean partition of the +21.7%. The BOUND survives; the LABEL does not.")
