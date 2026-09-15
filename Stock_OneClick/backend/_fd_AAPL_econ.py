import importlib.util, sys
spec=importlib.util.spec_from_file_location("p","_fd_AAPL_parse.py"); P=importlib.util.module_from_spec(spec); spec.loader.exec_module(P)

def load(path): return P.parse(path)

def pick(facts, tag, axis=None, member=None, dur=None, ns="us-gaap"):
    res=[]
    for f in facts:
        if f["tag"]!=tag: continue
        d=f["dims"]
        if axis is None:
            if d: continue
        else:
            if d.get(axis)!=member: continue
            if len(d)!=1: continue
        p=f["per"]
        if dur and p.get("startDate"):
            if (p["endDate"],p["startDate"]) != dur: continue
        res.append(f)
    return res

def durs(facts, tag):
    s=set()
    for f in facts:
        if f["tag"]==tag and f["per"].get("startDate"):
            s.add((f["per"]["startDate"],f["per"]["endDate"]))
    return sorted(s)

for label,path in [("FY2025 10-K","_fd_AAPL_10K_FY2025_inst.xml"),("FY2024 10-K","_fd_AAPL_10K_FY2024_inst.xml"),("FY26Q3 10-Q","_fd_AAPL_10Q_FY26Q3_inst.xml")]:
    F=load(path)
    print("="*100); print(label, path)
    RT="RevenueFromContractWithCustomerExcludingAssessedTax"
    print("durations for revenue:", durs(F,RT))
    dl = durs(F,RT)
    for dd in dl:
        key=(dd[1],dd[0])
        tot=[f["val"] for f in pick(F,RT,dur=key)]
        if not tot: continue
        print(f"\n-- period {dd[0]} .. {dd[1]}   TOTAL REV {tot[0]/1e6:,.0f}M")
        for ax,mems in [("ProductOrServiceAxis",["IPhoneMember","MacMember","IPadMember","WearablesHomeandAccessoriesMember","ServiceMember","ProductMember"]),
                        ("StatementBusinessSegmentsAxis",["AmericasSegmentMember","EuropeSegmentMember","GreaterChinaSegmentMember","JapanSegmentMember","RestOfAsiaPacificSegmentMember"])]:
            for m in mems:
                v=pick(F,RT,ax,m,dur=key)
                if v: print(f"     {m:42s} {v[0]['val']/1e6:12,.0f}M  {100*v[0]['val']/tot[0]:5.1f}%")
        # gross margin split
        for tag in ["CostOfGoodsAndServicesSold","GrossProfit"]:
            for ax,m in [(None,None),("ProductOrServiceAxis","ProductMember"),("ProductOrServiceAxis","ServiceMember")]:
                v=pick(F,tag,ax,m,dur=key)
                if v: print(f"     {tag} [{m}] {v[0]['val']/1e6:,.0f}M")
        # segment operating income
        for m in ["AmericasSegmentMember","EuropeSegmentMember","GreaterChinaSegmentMember","JapanSegmentMember","RestOfAsiaPacificSegmentMember"]:
            v=[f for f in F if f["tag"]=="OperatingIncomeLoss" and f["dims"].get("StatementBusinessSegmentsAxis")==m and (f["per"].get("endDate"),f["per"].get("startDate"))==key]
            if v: print(f"     OPINC {m:42s} {v[0]['val']/1e6:12,.0f}M")
