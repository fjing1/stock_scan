import json, pandas as pd
d = json.load(open('_arm_companyfacts.json'))
ug = d['facts']['us-gaap']

def series(tag, unit='USD'):
    if tag not in ug: return None
    u = ug[tag]['units']
    if unit not in u: 
        print('units avail for',tag,list(u.keys())); return None
    rows=[]
    for it in u[unit]:
        rows.append(dict(start=it.get('start'), end=it['end'], val=it['val'],
                         fy=it.get('fy'), fp=it.get('fp'), form=it.get('form'),
                         frame=it.get('frame'), filed=it.get('filed')))
    return pd.DataFrame(rows)

tags = ['Revenues','RevenueFromContractWithCustomerExcludingAssessedTax','CostOfRevenue',
        'ResearchAndDevelopmentExpense','SellingGeneralAndAdministrativeExpense',
        'OperatingIncomeLoss','NetIncomeLoss','ShareBasedCompensation',
        'AllocatedShareBasedCompensationExpense','IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
        'GrossProfit','EarningsPerShareDiluted','EarningsPerShareBasic',
        'WeightedAverageNumberOfDilutedSharesOutstanding','WeightedAverageNumberOfSharesOutstandingBasic',
        'CommonStockSharesOutstanding','CashAndCashEquivalentsAtCarryingValue',
        'DepreciationDepletionAndAmortization','NetCashProvidedByUsedInOperatingActivities',
        'PaymentsToAcquirePropertyPlantAndEquipment','Assets','Liabilities','StockholdersEquity']
avail = [t for t in tags if t in ug]
print('AVAILABLE:', avail)
print('MISSING:', [t for t in tags if t not in ug])
