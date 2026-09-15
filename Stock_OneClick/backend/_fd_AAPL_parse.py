import sys, re, json
from lxml import etree
from collections import defaultdict

def parse(path):
    t = etree.parse(path); root = t.getroot()
    ns = root.nsmap
    # contexts
    ctx = {}
    for c in root.iter():
        if etree.QName(c).localname != "context": continue
        cid = c.get("id"); dims={}; per={}
        for e in c.iter():
            ln = etree.QName(e).localname
            if ln=="explicitMember": dims[e.get("dimension").split(":")[-1]] = (e.text or "").split(":")[-1]
            elif ln in ("startDate","endDate","instant"): per[ln]=e.text
        ctx[cid]={"dims":dims,"per":per}
    facts=[]
    for e in root:
        q = etree.QName(e)
        if q.namespace and ("us-gaap" in q.namespace or "aapl" in (q.namespace or "")):
            cr = e.get("contextRef")
            if cr is None or cr not in ctx: continue
            txt = (e.text or "").strip()
            try: val = float(txt)
            except: continue
            sc = e.get("scale"); sign = e.get("sign")
            facts.append({"tag":q.localname,"ns":"us-gaap" if "us-gaap" in q.namespace else "aapl",
                          "val":val,"dims":ctx[cr]["dims"],"per":ctx[cr]["per"],"ctx":cr})
    return facts

def show(facts, tag, dimfilter=None, period=None):
    out=[]
    for f in facts:
        if f["tag"]!=tag: continue
        out.append(f)
    return out

if __name__=="__main__":
    path=sys.argv[1]
    facts=parse(path)
    print(path, "facts:", len(facts))
    # what dimensions exist
    dd=defaultdict(set)
    for f in facts:
        for k,v in f["dims"].items(): dd[k].add(v)
    for k in sorted(dd): print(" AXIS", k, "->", sorted(dd[k])[:20])
