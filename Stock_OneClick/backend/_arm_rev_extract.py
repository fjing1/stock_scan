import re,glob,os,pandas as pd
Q={"fye24q230-sepx23":("FY24Q2","2023-09-30"),"fye24q331-decx23":("FY24Q3","2023-12-31"),
   "fye24q431-marx24":("FY24Q4","2024-03-31"),"fye25q130-junx24":("FY25Q1","2024-06-30"),
   "fye25q230-sepx24":("FY25Q2","2024-09-30"),"fye25q331-decx24":("FY25Q3","2024-12-31"),
   "fye25q431-marx25":("FY25Q4","2025-03-31"),"fye26q130-junx25":("FY26Q1","2025-06-30"),
   "fye26q230-sepx25":("FY26Q2","2025-09-30"),"fye26q331-decx25":("FY26Q3","2025-12-31"),
   "fye26q431-marx26":("FY26Q4","2026-03-31"),"fye27q130-junx26":("FY27Q1","2026-06-30")}
rows=[]
for k,(lbl,qend) in Q.items():
    f=f"_arm_sl_{k}.txt"
    if not os.path.exists(f): print("MISSING",f); continue
    t=open(f,encoding='utf-8',errors='ignore').read()
    tt=' '.join(t.split())
    # the financial-overview table: "| Total revenue (2) | $X | $Y | Z% | ..."
    def grab(label):
        m=re.search(re.escape(label)+r"[^|]*\|\s*\$?([\d,]+)\s*\|\s*\$?\(?([\d,]+)\)?\s*\|\s*\(?(-?[\d.]+)\)?%",tt)
        return (int(m.group(1).replace(',','')),int(m.group(2).replace(',','')),float(m.group(3))) if m else (None,None,None)
    tot=grab("Total revenue (2)");  roy=grab("Royalty revenue");  lic=grab("License and other revenue")
    # smartphone narrative intensity: count in the letter body (before the financial statements)
    body=tt[:tt.find("Financial Overview")+20000]
    rows.append({"q":lbl,"q_end":qend,
        "total_rev":tot[0],"total_yoy%":tot[2],
        "royalty":roy[0],"royalty_prior":roy[1],"royalty_yoy%":roy[2],
        "license":lic[0],"license_yoy%":lic[2],
        "n_smartphone":len(re.findall(r"[Ss]martphone",tt)),
        "n_datacenter":len(re.findall(r"[Dd]ata [Cc]enter|[Dd]atacenter",tt)),
        "n_AI":len(re.findall(r"\bAI\b",tt)),
        "n_Apple":len(re.findall(r"\bApple\b",tt)),
        "n_NVIDIA":len(re.findall(r"NVIDIA|Nvidia",tt))})
df=pd.DataFrame(rows)
df["royalty_share%"]=(100*df.royalty/df.total_rev).round(1)
pd.set_option("display.width",250)
print(df.to_string(index=False))
df.to_csv("_arm_quarterly_royalty.csv",index=False)
