import yfinance as yf, pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_rows',80)
h=yf.Ticker('ARM').history(period='max', auto_adjust=False)
h.index=h.index.tz_localize(None)
h=h[h.index<pd.Timestamp('2026-09-15')]
print('rows',len(h),'first',h.index[0].date(),'last',h.index[-1].date(), 'lastclose',round(h.Close.iloc[-1],2))
ath=h.Close.idxmax(); print('ATH close',round(h.Close.max(),2),'on',ath.date())
print('ATH intraday high',round(h.High.max(),2),'on',h.High.idxmax().date())
# quarter-end marks vs reported numbers
print()
print('Close on key dates:')
for d in ['2024-07-08','2025-03-31','2025-06-30','2025-09-30','2025-12-31','2026-03-31','2026-05-06','2026-06-30','2026-07-29','2026-08-10','2026-09-01','2026-09-14']:
    ts=pd.Timestamp(d); sub=h[h.index<=ts]
    if len(sub): print(' ',d, round(sub.Close.iloc[-1],2))
h.to_csv('_arm_price_history.csv')
# drawdown decomposition: price vs non-GAAP EPS known at the time
