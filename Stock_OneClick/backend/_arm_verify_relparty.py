"""Adversarial verification of the 'FY2026 growth is 74% related-party' claim.

Re-extracts everything from a FRESH download of the FY2026 20-F
(acc 0001973239-26-000097, arm-20260331.htm) rather than trusting any
prior on-disk artifact or prior script.

Usage: python _arm_verify_relparty.py <search-term> [context_chars]
"""
import re
import sys
import os
import lxml.html

HTM = "/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/_arm_verify_20f_fy2026_fresh.htm"
TXT = "/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/_arm_verify_20f_fy2026_fresh.txt"


def load():
    if not os.path.exists(TXT):
        t = lxml.html.parse(HTM).getroot().text_content()
        t = re.sub(r"[\xa0​]", " ", t)
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n\s*\n+", "\n", t)
        open(TXT, "w").write(t)
    return open(TXT).read()


if __name__ == "__main__":
    txt = load()
    term = sys.argv[1] if len(sys.argv) > 1 else "Consulting Agreement"
    ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 900
    hits = [m.start() for m in re.finditer(re.escape(term), txt)]
    print(f"len(text)={len(txt)}  term={term!r}  hits={len(hits)}")
    for i, h in enumerate(hits):
        print("=" * 78)
        print(f"--- hit {i+1} @ {h} ---")
        print(txt[max(0, h - ctx // 3):h + ctx])
