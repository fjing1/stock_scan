import os, time, requests
UA={"User-Agent":"independent research fj@example.com"}
jobs=[
 ("000197323926000113","exhibit991fye27q130-junx26.htm","_arm_sl_fy27q1.htm"),
 ("000197323926000113","exhibit992fye27q130-junx26.htm","_arm_pr_fy27q1.htm"),
 ("000197323926000062","exhibit991fye26q431-marx26.htm","_arm_sl_fy26q4.htm"),
 ("000197323926000062","exhibit992fye26q431-marx26.htm","_arm_pr_fy26q4.htm"),
 ("000197323926000005","exhibit991fye26q331-decx25.htm","_arm_sl_fy26q3.htm"),
 ("000197323926000005","exhibit993armeverywherepre.htm","_arm_everywhere_presentation.htm"),
 ("000197323925000043","exhibit991fye26q230-sepx25.htm","_arm_sl_fy26q2.htm"),
 ("000197323925000042","exhibit991fye26q230-sepx25.htm","_arm_sl_fy26q2b.htm"),
 ("000197323925000023","exhibit991fye26q130-junx25.htm","_arm_sl_fy26q1.htm"),
 ("000197323925000010","exhibit991fye25q431-marx25.htm","_arm_sl_fy25q4.htm"),
 ("000197323925000006","exhibit991fye25q331-decx24.htm","_arm_sl_fy25q3.htm"),
 ("000197323924000036","exhibit991fye25q230-sepx24.htm","_arm_sl_fy25q2.htm"),
 ("000197323924000017","exhibit991fye25q130-junx24.htm","_arm_sl_fy25q1.htm"),
 ("000197323924000007","exhibit991fye24q431-marx24.htm","_arm_sl_fy24q4.htm"),
 ("000197323924000002","exhibit991fye24q331-decx23.htm","_arm_sl_fy24q3.htm"),
 ("000197323923000009","exhibit991fye24q230-sepx23.htm","_arm_sl_fy24q2.htm"),
 ("000197323926000050","exhibit991-march24finalarm.htm","_arm_pr_2026-03-24.htm"),
]
for a,f,o in jobs:
    if os.path.exists(o) and os.path.getsize(o)>1000: continue
    r=requests.get(f"https://www.sec.gov/Archives/edgar/data/1973239/{a}/{f}",headers=UA,timeout=60)
    print(r.status_code,len(r.content),o)
    if r.status_code==200: open(o,'wb').write(r.content)
    time.sleep(0.35)
