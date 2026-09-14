import yfinance as yf, pandas as pd, json
pd.set_option('display.width',250)
tick=['ARM','NVDA','AVGO','QCOM','CDNS','SNPS','TXN','MRVL']
rows=[]
for t in tick:
    T=yf.Ticker(t); i=T.info
    rows.append(dict(tk=t,
      price=i.get('currentPrice'), mcap=i.get('marketCap'),
      ev=i.get('enterpriseValue'),
      trailPE=i.get('trailingPE'), fwdPE=i.get('forwardPE'),
      ps=i.get('priceToSalesTrailing12Months'),
      evs=(i.get('enterpriseValue')/i.get('totalRevenue')) if i.get('enterpriseValue') and i.get('totalRevenue') else None,
      ev_ebitda=i.get('enterpriseToEbitda'),
      rev_ttm=i.get('totalRevenue'), rev_g=i.get('revenueGrowth'),
      earn_g=i.get('earningsGrowth'), eps_ttm=i.get('trailingEps'), eps_fwd=i.get('forwardEps'),
      gm=i.get('grossMargins'), om=i.get('operatingMargins'), pm=i.get('profitMargins')))
D=pd.DataFrame(rows).set_index('tk')
D['mcap_bn']=D.mcap/1e9; D['rev_bn']=D.rev_ttm/1e9
print(D[['price','mcap_bn','rev_bn','trailPE','fwdPE','ps','evs','ev_ebitda','rev_g','om','pm','eps_ttm','eps_fwd']].round(2).to_string())
D.to_csv('_arm_peers.csv')
