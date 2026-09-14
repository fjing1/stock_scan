import yfinance as yf, pandas as pd
pd.set_option('display.width',250)
P,SH,NETCASH=239.01,1068.0,3888.0
REV_TTM=5156.0
GP_TTM=4799.0-1023.0+1253.0            # 20-F L1090 (4,799) + 6-K L107 (1,253 / 1,023)
print(f"ARM TTM gross profit FROM FILINGS = 4799-1023+1253 = ${GP_TTM:,.0f}m -> GM {GP_TTM/REV_TTM:.1%}")
EV=P*SH-NETCASH
rows=[]
for tk in ['ARM','NVDA','AVGO','QCOM','CDNS','SNPS','TXN','MRVL']:
    i=yf.Ticker(tk).info
    ev=i.get('enterpriseValue'); rev=i.get('totalRevenue'); gm=i.get('grossMargins')
    gp = GP_TTM*1e6 if tk=='ARM' else (rev*gm if (rev and gm) else None)
    rows.append(dict(tk=tk, ev_bn=ev/1e9, GM=gm, EV_S=ev/rev, EV_GP=ev/gp if gp else None,
                     rev_g=i.get('revenueGrowth')))
d=pd.DataFrame(rows).set_index('tk')
print(d.round(3).to_string())
ex=d.drop('ARM')
print(f"\nARM EV/GrossProfit = {d.loc['ARM','EV_GP']:.1f}x   (filings-based GP)")
print("ARM implied price at PEER EV/GrossProfit multiples:")
for lbl,m in [('median peer',ex.EV_GP.median()),('MAX peer',ex.EV_GP.max())]:
    px=(m*GP_TTM+NETCASH)/SH
    print(f"  {lbl} EV/GP {m:5.1f}x -> ARM ${px:6.0f}   vs price 239.01 ({px/239.01-1:+.0%})  vs cost 257 ({px/257-1:+.0%})")
print(f"\n  max-EV/GP peer is {ex.EV_GP.idxmax()} at {ex.EV_GP.max():.1f}x")
print("\nGrowth-adjusted (EV/GP per point of revenue growth):")
d['EV_GP_per_gpt']=(d.EV_GP/(d.rev_g*100)).round(2)
print(d[['EV_GP','rev_g','EV_GP_per_gpt']].round(3).to_string())
