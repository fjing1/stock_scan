"""ADVERSARIAL VERIFY #5 -- the press-release leg of the cross-check. The claim says TTM diluted EPS
$8.72 was 'summed independently from the four press releases' and that derived Q4 FY2025 revenue
'matches the 8-K's own $102.5 billion'. Pull all four Exhibit 99.1s and read the figures.
Also fetch the iPhone 16 / 16 Plus newsroom page (correct slug) for the Plus list price.
"""
from __future__ import annotations
import html, json, os, re, time, urllib.request
import requests

UA = {"User-Agent": "Feijing Research feijing.research@gmail.com"}
BROWSER = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/126 Safari/537.36"}


def sec(url, dest):
    if os.path.exists(dest):
        return open(dest, "rb").read()
    b = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()
    open(dest, "wb").write(b)
    time.sleep(0.3)
    return b


def txt(raw):
    t = re.sub(r"<[^>]+>", " ", raw.decode("utf-8", "replace"))
    t = html.unescape(t)
    return re.sub(r"\s+", " ", t)


EIGHTKS = {"0000320193-25-000077": ("2025-10-30", "Q4 FY2025"),
           "0000320193-26-000005": ("2026-01-29", "Q1 FY2026"),
           "0000320193-26-000011": ("2026-04-30", "Q2 FY2026"),
           "0000320193-26-000018": ("2026-07-30", "Q3 FY2026")}

print("=" * 122)
print("A. THE FOUR EARNINGS 8-Ks -- Exhibit 99.1, revenue and diluted EPS as Apple stated them")
print("=" * 122)
res = {}
for accn, (fdate, lab) in EIGHTKS.items():
    a = accn.replace("-", "")
    idx = json.loads(sec(f"https://www.sec.gov/Archives/edgar/data/320193/{a}/index.json",
                         f"_fd_AAPL_verify_8kidx_{accn}.json"))
    items = [i["name"] for i in idx["directory"]["item"]]
    ex = [n for n in items if re.match(r"a8-k.*ex99.*\.htm$", n, re.I) or re.match(r".*ex-?99.*\.htm$", n, re.I)]
    if not ex:
        ex = [n for n in items if n.endswith(".htm") and "aapl-" not in n]
    print(f"\n--- {lab}  8-K filed {fdate}  accn {accn}")
    print(f"    exhibit files: {ex[:4]}   (all files: {[n for n in items if n.endswith('.htm')][:6]})")
    if not ex:
        continue
    t = txt(sec(f"https://www.sec.gov/Archives/edgar/data/320193/{a}/{ex[0]}",
                f"_fd_AAPL_verify_8kex_{accn}.htm"))
    open(f"_fd_AAPL_verify_8kex_{accn}.txt", "w").write(t)
    # headline sentence
    for pat in (r"[^.]{0,320}(?:revenue of|quarterly revenue).{0,260}",
                r"[^.]{0,200}diluted (?:earnings per share|EPS).{0,220}"):
        for m in list(re.finditer(pat, t, re.I))[:2]:
            print("     >>", m.group(0).strip()[:400])
    # the condensed income statement's diluted EPS row
    m = re.search(r"Earnings per share:.{0,400}?Diluted\s*\$?\s*([\d.]+)\s*\$?\s*([\d.]+)", t)
    if m:
        print(f"     income-statement Diluted EPS row: current {m.group(1)}   year-ago {m.group(2)}")
        res[lab] = float(m.group(1))
    else:
        m2 = re.search(r"Diluted\s*\$\s*([\d.]+)", t)
        if m2:
            print(f"     first 'Diluted $' hit: {m2.group(1)}")
            res[lab] = float(m2.group(1))
    m = re.search(r"Total net sales\s*\$?\s*([\d,]+)", t)
    if m:
        print(f"     Total net sales (current period column): {m.group(1)}")

print("\n" + "=" * 122)
print("B. TTM diluted EPS from the press releases")
print("=" * 122)
for k, v in res.items():
    print(f"    {k:<12} {v:>6.2f}")
if len(res) == 4:
    s = sum(res.values())
    print(f"    SUM = {s:.2f}   (claim says 8.72; XBRL-derived is 8.71)")
    print(f"    P/E at 333.08 = {333.08/s:.2f}x")

print("\n" + "=" * 122)
print("C. iPhone 16 / 16 Plus list price -- correct newsroom slug (the earlier fetch used a slug")
print("   that 404s, so the $899 Plus price was NOT primary-sourced in the claim's work either)")
print("=" * 122)
loc = "_fd_AAPL_verify_news_202409_apple-introduces-iphone-16-and-iphone-16-plus.html"
if not os.path.exists(loc):
    r = requests.get("https://www.apple.com/newsroom/2024/09/apple-introduces-iphone-16-and-iphone-16-plus/",
                     headers=BROWSER, timeout=40)
    print(f"  HTTP {r.status_code}")
    if r.status_code == 200:
        open(loc, "w").write(r.text)
if os.path.exists(loc):
    raw = open(loc, encoding="utf-8", errors="ignore").read()
    d = re.search(r'"datePublished"\s*:\s*"([0-9-]{10})', raw)
    t = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw)))
    print(f"  datePublished = {d.group(1) if d else '?'}")
    for m in re.finditer(r".{0,200}(?:starts at|Pricing and Availability).{0,320}", t):
        s = m.group(0).strip()
        if "$" in s:
            print("   >>", s[:460])
            break
    for m in re.finditer(r".{0,60}(?:128GB|256GB).{0,240}", t):
        print("   >>", m.group(0).strip()[:300]); break
