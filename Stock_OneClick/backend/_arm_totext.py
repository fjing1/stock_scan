import sys, re
from lxml import html, etree
BLOCK = {'p','div','tr','br','li','h1','h2','h3','h4','h5','h6','table','ul','ol','hr','td'}
for f in sys.argv[1:]:
    root = html.parse(f).getroot()
    for bad in root.xpath('//script|//style'):
        bad.getparent().remove(bad)
    # drop hidden inline-XBRL blobs
    for bad in root.xpath('//*[contains(translate(@style,"DISPLAYNONE","displaynone"),"display:none")]'):
        p = bad.getparent()
        if p is not None: p.remove(bad)
    for el in root.iter():
        tag = (el.tag.split('}')[-1].split(':')[-1].lower() if isinstance(el.tag, str) else '')
        if tag in BLOCK:
            el.tail = ('\n' + (el.tail or ''))
            if tag == 'td': el.tail = (' | ' + (el.tail or ''))
    txt = root.text_content()
    txt = txt.replace('​',' ').replace('\xa0',' ')
    txt = re.sub(r'[ \t]+', ' ', txt)
    txt = re.sub(r'\n[ \t]+', '\n', txt)
    txt = re.sub(r'\n{2,}', '\n', txt)
    out = f.rsplit('.',1)[0] + '.txt'
    open(out,'w').write(txt)
    n=len(txt.split('\n'))
    print(out, len(txt), n, 'lines')
