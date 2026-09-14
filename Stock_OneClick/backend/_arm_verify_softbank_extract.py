"""Extract, from the FY2026 20-F primary doc, the exact numbers the claim rests on.

Source: SEC EDGAR accession 0001973239-26-000097, arm-20260331.htm (ARM HOLDINGS PLC /UK,
CIK 1973239, 20-F for FY ended 2026-03-31, filed 2026-05-26).
"""
import re
import sys

import lxml.html

SRC = "_arm_verify_20f_fy26.htm"

root = lxml.html.parse(SRC).getroot()

# Build a text version that keeps block structure: each table row / paragraph on its own line.
for br in root.iter("br"):
    br.tail = "\n" + (br.tail or "")

lines = []


def block_text(el):
    t = el.text_content()
    t = re.sub(r"[\xa0​]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# Walk table rows and paragraphs
for el in root.iter():
    if el.tag in ("tr",):
        cells = []
        for td in el.iter("td", "th"):
            c = block_text(td)
            if c:
                cells.append(c)
        if cells:
            lines.append(" | ".join(cells))
    elif el.tag in ("p", "div", "span"):
        # only leaf-ish text blocks that are not inside tables
        if el.find(".//table") is not None:
            continue
        anc = [a.tag for a in el.iterancestors()]
        if "td" in anc or "th" in anc or "tr" in anc:
            continue
        if any(child.tag in ("p", "div") for child in el):
            continue
        t = block_text(el)
        if t:
            lines.append(t)

# de-dup consecutive identical lines (nested span/div produce repeats)
out = []
for ln in lines:
    if not out or out[-1] != ln:
        out.append(ln)

with open("_arm_verify_20f_fy26_lines.txt", "w") as f:
    f.write("\n".join(out))

print(f"wrote {len(out)} lines")

KEYS = sys.argv[1:] if len(sys.argv) > 1 else []
for k in KEYS:
    print(f"\n===== SEARCH: {k!r} =====")
    for i, ln in enumerate(out):
        if k.lower() in ln.lower():
            print(f"[{i}] {ln[:400]}")
