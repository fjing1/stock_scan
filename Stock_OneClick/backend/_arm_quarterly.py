import json, pandas as pd
pd.set_option('display.width',300); pd.set_option('display.max_rows',300)
d=json.load(open('_arm_companyfacts.json')); ug=d['facts']['us-gaap']
def q(tag,unit='USD'):
    df=pd.DataFrame(ug[tag]['units'][unit])
    df['start']=pd.to_datetime(df['start']); df['end']=pd.to_datetime(df['end'])
    df['days']=(df['end']-df['start']).dt.days
    x=df[(df.days>80)&(df.days<100)].sort_values('filed').drop_duplicates(subset=['start','end'],keep='last')
    return x.set_index(x.end.dt.strftime('%Y-%m-%d'))['val'].sort_index()
tags=['RevenueFromContractWithCustomerExcludingAssessedTax','GrossProfit','ResearchAndDevelopmentExpense',
'SellingGeneralAndAdministrativeExpense','OperatingIncomeLoss','NetIncomeLoss','ShareBasedCompensation',
'NetCashProvidedByUsedInOperatingActivities']
o={}
for t in tags:
    try: o[t]=q(t)/1e6
    except Exception as e: print(t,'ERR',e)
Q=pd.DataFrame(o)
Q['SBC_pct_rev']=Q['ShareBasedCompensation']/Q['RevenueFromContractWithCustomerExcludingAssessedTax']*100
Q['GAAP_opm']=Q['OperatingIncomeLoss']/Q['RevenueFromContractWithCustomerExcludingAssessedTax']*100
Q['adj_opm_exSBC']=(Q['OperatingIncomeLoss']+Q['ShareBasedCompensation'])/Q['RevenueFromContractWithCustomerExcludingAssessedTax']*100
Q['rev_yoy']=Q['RevenueFromContractWithCustomerExcludingAssessedTax'].pct_change(4)*100
print('QUARTERLY, USD m  (fiscal Q1=Jun, Q2=Sep, Q3=Dec, Q4=Mar)')
print(Q.round(1).to_string())
for t,u in [('EarningsPerShareDiluted','USD/shares')]:
    print(); print(t); print(q(t,u).round(3).to_string())
