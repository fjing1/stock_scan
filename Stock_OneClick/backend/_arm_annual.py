import json, pandas as pd
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
d = json.load(open('_arm_companyfacts.json'))
ug = d['facts']['us-gaap']

def get(tag, unit='USD'):
    u = ug[tag]['units'][unit]
    df = pd.DataFrame(u)
    return df

def annual(tag, unit='USD'):
    df = get(tag, unit)
    if 'start' not in df.columns:  # instant
        df = df[df.end.str.endswith('-03-31')]
        df = df.sort_values('filed').drop_duplicates(subset=['end'], keep='last')
        return df[['end','val','form','fy','fp','filed']]
    df['start']=pd.to_datetime(df['start']); df['end']=pd.to_datetime(df['end'])
    df['days']=(df['end']-df['start']).dt.days
    a = df[(df.days>350)&(df.days<380)].copy()
    a = a.sort_values('filed').drop_duplicates(subset=['start','end'], keep='last')
    return a[['start','end','val','form','fy','fp','filed']].sort_values('end')

def quarterly(tag, unit='USD'):
    df = get(tag, unit)
    if 'start' not in df.columns: return None
    df['start']=pd.to_datetime(df['start']); df['end']=pd.to_datetime(df['end'])
    df['days']=(df['end']-df['start']).dt.days
    q = df[(df.days>80)&(df.days<100)].copy()
    q = q.sort_values('filed').drop_duplicates(subset=['start','end'], keep='last')
    return q[['start','end','val','form','filed']].sort_values('end')

print('='*30,'ANNUAL (FY ends Mar-31), USD millions')
tags=['RevenueFromContractWithCustomerExcludingAssessedTax','CostOfRevenue','GrossProfit',
      'ResearchAndDevelopmentExpense','SellingGeneralAndAdministrativeExpense','OperatingIncomeLoss',
      'NetIncomeLoss','ShareBasedCompensation','NetCashProvidedByUsedInOperatingActivities',
      'PaymentsToAcquirePropertyPlantAndEquipment','DepreciationDepletionAndAmortization']
out={}
for t in tags:
    a=annual(t)
    if a is None or a.empty: print(t,'-> none'); continue
    s = a.set_index(a.end.dt.strftime('%Y-%m-%d'))['val']/1e6
    out[t]=s
A=pd.DataFrame(out).T
print(A.round(1))
print()
print('='*30,'ANNUAL EPS / shares')
for t,u in [('EarningsPerShareDiluted','USD/shares'),('EarningsPerShareBasic','USD/shares'),
            ('WeightedAverageNumberOfDilutedSharesOutstanding','shares'),
            ('WeightedAverageNumberOfSharesOutstandingBasic','shares')]:
    a=annual(t,u)
    if a is None or a.empty: continue
    s=a.set_index(a.end.dt.strftime('%Y-%m-%d'))['val']
    print(t); print((s/1e6 if 'Number' in t else s).round(3).to_string())
print()
print('='*30,'BALANCE (instant), USD m')
for t in ['CashAndCashEquivalentsAtCarryingValue','Assets','Liabilities','StockholdersEquity']:
    df=get(t); df=df.sort_values('filed').drop_duplicates(subset=['end'],keep='last')
    print(t); print((df.set_index('end')['val']/1e6).round(1).tail(14).to_string())
