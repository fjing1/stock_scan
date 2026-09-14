#!/usr/bin/env python
"""HTML -> text for Arm filings, preserving block breaks."""
import sys, re, lxml.html

src, dst = sys.argv[1], sys.argv[2]
root = lxml.html.parse(src).getroot()
out = []
BLK = {'p', 'div', 'tr', 'li', 'h1', 'h2', 'h3', 'h4', 'table', 'br'}


def rec(el):
    if el.tag in ('script', 'style'):
        return
    if el.text:
        out.append(el.text)
    for c in el:
        rec(c)
        if c.tag in BLK:
            out.append('\n')
        if c.tag in ('td', 'th'):
            out.append(' | ')
        if c.tail:
            out.append(c.tail)


rec(root)
txt = ''.join(out)
txt = re.sub(r'[ \t\xa0]+', ' ', txt)
txt = re.sub(r' *\n *', '\n', txt)
txt = re.sub(r'\n{3,}', '\n\n', txt)
open(dst, 'w').write(txt)
print(dst, len(txt), txt.count('\n'), 'lines')
