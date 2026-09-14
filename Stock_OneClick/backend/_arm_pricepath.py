#!/usr/bin/env python
"""ARM price path + biggest daily moves since IPO, and the last 6 months day by day."""
import pandas as pd, numpy as np, yfinance as yf

pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 400)

TICK = ['ARM', 'AAPL', 'NVDA', 'SMH', 'QQQ', 'SPY', 'AVGO', 'QCOM', 'TSM', 'SOXX']
px = yf.download(TICK, start='2023-09-13', end='2026-09-16', auto_adjust=True,
                 progress=False, group_by='column')['Close']
px = px.dropna(how='all')
px.to_pickle('_arm_prices.pkl')
print('rows', len(px), 'range', px.index[0].date(), px.index[-1].date())
print(px.tail(3).round(2).to_string())

r = px.pct_change()
arm = px['ARM']
print('\n=== ARM last close', round(arm.iloc[-1], 2), 'on', arm.index[-1].date())
print('ATH', round(arm.max(), 2), 'on', arm.idxmax().date())

# biggest single-day moves since IPO
top = r['ARM'].abs().sort_values(ascending=False).head(25)
tbl = pd.DataFrame({'ARM': r['ARM'][top.index], 'SMH': r['SMH'][top.index],
                    'AAPL': r['AAPL'][top.index], 'NVDA': r['NVDA'][top.index],
                    'ARM_close': arm[top.index]})
tbl['ARM_ex_SMH'] = tbl['ARM'] - tbl['SMH']
print('\n=== 25 biggest ARM daily moves since IPO (decimal returns)')
print((tbl.sort_index() * 1).round(4).to_string())

# day-by-day last 90 calendar days
w = px.loc['2026-06-01':]
rr = r.loc['2026-06-01':]
out = pd.DataFrame({'ARM': w['ARM'].round(2), 'ARM_r': (rr['ARM'] * 100).round(2),
                    'SMH_r': (rr['SMH'] * 100).round(2), 'exSMH': ((rr['ARM'] - rr['SMH']) * 100).round(2),
                    'NVDA_r': (rr['NVDA'] * 100).round(2), 'AAPL_r': (rr['AAPL'] * 100).round(2)})
print('\n=== daily since 2026-06-01')
print(out.to_string())

# 3-month benchmark
base = px.loc[:'2026-06-14'].iloc[-1]
last = px.iloc[-1]
print('\n=== 3-month total return from', px.loc[:'2026-06-14'].index[-1].date())
print(((last / base - 1) * 100).round(1).to_string())
