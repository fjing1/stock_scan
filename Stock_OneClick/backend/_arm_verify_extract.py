import re, sys, html
from pathlib import Path

def to_text(path):
    raw = Path(path).read_text(errors='ignore')
    # drop XBRL inline hidden blocks & scripts/styles
    raw = re.sub(r'(?is)<(script|style).*?</\1>', ' ', raw)
    raw = re.sub(r'(?is)<div[^>]*style="[^"]*display:\s*none[^"]*"[^>]*>.*?</div>', ' ', raw)
    # table cells -> pipe, rows/blocks -> newline
    raw = re.sub(r'(?is)</t[dh]>', ' | ', raw)
    raw = re.sub(r'(?is)</tr>', '\n', raw)
    raw = re.sub(r'(?is)</(p|div|table|br|li|h[1-6])>', '\n', raw)
    raw = re.sub(r'(?is)<br[^>]*>', '\n', raw)
    raw = re.sub(r'(?s)<[^>]+>', '', raw)
    raw = html.unescape(raw)
    raw = raw.replace(' ',' ').replace('’',"'").replace('—','-').replace('–','-')
    lines = [re.sub(r'[ \t]+',' ',l).strip() for l in raw.split('\n')]
    lines = [l for l in lines if l and l not in ('|',)]
    return '\n'.join(lines)

for src in sys.argv[1:]:
    out = src.rsplit('.',1)[0] + '_clean.txt'
    Path(out).write_text(to_text(src))
    print(out, len(Path(out).read_text().split('\n')), 'lines')
