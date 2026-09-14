import json,re,html,os,pandas as pd
d=json.load(open('_arm_aapl_submissions.json')); df=pd.DataFrame(d['filings']['recent'])
k=df[df.form.isin(['10-Q','10-K'])].head(14)
rows=[]
for _,x in k.iterrows():
    out=f"_arm_aapl_{x.form.replace('-','')}_{x.reportDate}.htm"
    t=open(out,encoding='utf-8',errors='ignore').read()
    txt=html.unescape(re.sub(r'<[^>]+>','|',t)); txt=re.sub(r'\|+','|',txt); txt=re.sub(r'[ \t\xa0]+',' ',txt)
    m=re.search(r'\|iPhone\s*(?:®|\(1\))?\s*\|(.{0,260}?)\|Mac\s*(?:®)?\s*\|',txt,re.S)
    nums=[int(n.replace(',','')) for n in re.findall(r'\d{1,3}(?:,\d{3})+',m.group(1))] if m else []
    # for 10-Q: [current Q, prior-year Q, YTD, prior YTD] ; for 10-K: [FY, FY-1, FY-2]
    rows.append({"form":x.form,"period_end":x.reportDate,"filed":x.filingDate,"nums":nums})
    print(f"{x.form} {x.reportDate} {nums}")
raw=pd.DataFrame(rows)

# --- build a clean quarterly iPhone net sales series ---
per=pd.to_datetime(raw.period_end)
q={}
for _,r in raw.iterrows():
    n=r.nums; pe=pd.Timestamp(r.period_end)
    if r.form=='10-Q' and len(n)>=2:
        q[pe]=n[0]
        q[pe-pd.DateOffset(years=1)]=n[1]      # prior-year same quarter (approx date; relabelled below)
# 10-K gives full FY; derive FQ4 = FY - sum(FQ1..FQ3)
fy={}
for _,r in raw.iterrows():
    if r.form=='10-K' and len(r.nums)>=1: fy[pd.Timestamp(r.period_end)]=r.nums[0]
print("\nFY iPhone net sales from 10-K:",{str(k.date()):v for k,v in fy.items()})
pd.DataFrame({"period_end":list(q.keys()),"iphone_musd":list(q.values())}).sort_values("period_end").to_csv("_arm_aapl_iphone_q_raw.csv",index=False)
