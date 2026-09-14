"""Already-priced test: was the SoftBank-affiliate consulting revenue disclosed in the
quarterly 6-K shareholder letters DURING FY2026, or only revealed in the May-2026 20-F?

Sources: ARM 6-K exhibits (quarterly shareholder letters / supplemental financials),
CIK 1973239, files already on disk from EDGAR.
"""
import glob
import re

import lxml.html

FILES = {
    "FY26Q1 (Jun-2025) letter": "_arm_sl_fye26q130-junx25.htm",
    "FY26Q2 (Sep-2025) letter": "_arm_sl_fye26q230-sepx25.htm",
    "FY26Q3 (Dec-2025) letter": "_arm_sl_fye26q331-decx25.htm",
    "FY26Q4 (Mar-2026) letter": "_arm_sl_fye26q431-marx26.htm",
    "FY27Q1 (Jun-2026) letter": "_arm_sl_fye27q130-junx26.htm",
    "FY26Q2 6-K financials": "_arm_6K_Q2FY26_fin.htm",
    "FY26Q3 6-K financials": "_arm_6K_Q3FY26_fin.htm",
    "FY27Q1 6-K financials": "_arm_6K_Q1FY27_fin.htm",
}

PATTERNS = [
    r"Revenue from related part\w+",
    r"related part\w+",
    r"Consulting Agreement",
    r"affiliate of SoftBank",
    r"contract asset",
]


def txt(p):
    t = lxml.html.parse(p).getroot().text_content()
    t = re.sub(r"[\xa0​]", " ", t)
    return re.sub(r"\s+", " ", t)


for label, path in FILES.items():
    try:
        t = txt(path)
    except Exception as e:  # noqa
        print(f"{label}: MISSING ({e})")
        continue
    print(f"\n########## {label}  ({path}, {len(t):,} chars)")
    for pat in PATTERNS:
        hits = list(re.finditer(pat, t, re.I))
        print(f"  {pat!r}: {len(hits)} hits")
    # pull the related-party revenue line from the income statement
    for m in re.finditer(r"Revenue from related part\w+(.{0,160})", t, re.I):
        print(f"    -> Revenue from related parties{m.group(1)}")
    for m in re.finditer(r"(.{0,200}Consulting Agreement.{0,600})", t, re.I):
        print(f"    -> CONSULTING: ...{m.group(1)}...")
        break
