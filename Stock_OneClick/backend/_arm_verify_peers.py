import yfinance as yf, pandas as pd
pd.set_option('display.width',250)
P,SH,NETCASH,REV_TTM = 239.01,1068.0,3888.0,5156.0
EV_filed = P*SH-NETCASH
print(f"ARM EV/Sales derived FROM FILINGS: EV ${EV_filed/1000:.1f}bn / TTM rev ${REV_TTM/1000:.3f}bn = {EV_filed/REV_TTM:.1f}x")
rows=[]
for tk in ['ARM','NVDA','AVGO','QCOM','CDNS','SNPS','TXN','MRVL']:
    try:
        i=yf.Ticker(tk).info
        rows.append(dict(tk=tk, EV_S=i.get('enterpriseToRevenue'), fwdPE=i.get('forwardPE'),
                         rev_g=i.get('revenueGrowth'), ev_bn=(i.get('enterpriseValue') or 0)/1e9,
                         rev_bn=(i.get('totalRevenue') or 0)/1e9))
    except Exception as e: print(tk,'ERR',e)
d=pd.DataFrame(rows).set_index('tk')
d['EV_S_recomputed']=(d.ev_bn/d.rev_bn).round(2)
print(d.round(2).to_string())
print(f"\nyfinance says ARM EV/S = {d.loc['ARM','EV_S']}, using its revenue base ${d.loc['ARM','rev_bn']:.3f}bn")
print(f"  -> that revenue base differs from filed TTM ${REV_TTM/1000:.3f}bn. Filings-derived multiple is {EV_filed/REV_TTM:.1f}x")
ex=d.drop('ARM')
print(f"\nCDNS: EV/S {d.loc['CDNS','EV_S']}x, rev growth {d.loc['CDNS','rev_g']:.1%}  vs ARM growth {d.loc['ARM','rev_g']:.1%}")
print(f"MRVL: EV/S {d.loc['MRVL','EV_S']}x, rev growth {d.loc['MRVL','rev_g']:.1%}")
for lbl,m in [('median peer EV/S',ex.EV_S.median()),('MAX peer EV/S',ex.EV_S.max())]:
    print(f"  {lbl} {m:.1f}x -> ARM ${(m*REV_TTM+NETCASH)/SH:.0f}")
print(f"  MAX peer fwdPE {ex.fwdPE.max():.1f}x -> ARM on FY28 nonGAAP $3.056 = ${ex.fwdPE.max()*3.056:.0f}")
