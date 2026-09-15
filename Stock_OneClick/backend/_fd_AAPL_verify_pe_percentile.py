"""Part 2: prices, the 1-year return decomposition, and the point-in-time P/E & P/S percentiles.
Tests RAW (split-only) closes vs dividend-adjusted closes, and several window lengths.
"""
import json
import pandas as pd, numpy as np, yfinance as yf

T = pd.read_csv('_fd_AAPL_verify_ttm_pit.csv', parse_dates=['end', 'avail'])
STATE_AID = 10246e6
SA_QEND = pd.Timestamp('2024-09-28')   # charge sits in the quarter ended 2024-09-28

# ---------------------------------------------------------------- prices
tk = yf.Ticker('AAPL')
raw = tk.history(period='max', auto_adjust=False)      # 'Close' = split-adj only
raw.index = pd.to_datetime(raw.index).tz_localize(None)
px = pd.DataFrame({'close_raw': raw['Close'], 'close_adj': raw['Adj Close']}).dropna()
print('prices %s .. %s  n=%d' % (px.index[0].date(), px.index[-1].date(), len(px)))
print('last 3 raw closes:'); print(px.tail(3).round(2).to_string())

LAST = px.index[-1]
P_NOW = px.close_raw.iloc[-1]

# 1-year-ago anchor
target = LAST - pd.Timedelta(days=365)
prior = px.index[px.index.get_indexer([target], method='nearest')][0]
P_THEN = px.close_raw.loc[prior]
ret = P_NOW / P_THEN - 1
print('\n1Y RETURN (raw closes, i.e. price only, no dividends)')
print('  %s  %.2f  ->  %s  %.2f   = %+.2f%%' % (prior.date(), P_THEN, LAST.date(), P_NOW, 100 * ret))
adj_ret = px.close_adj.iloc[-1] / px.close_adj.loc[prior] - 1
print('  total-return (div reinvested) equivalent: %+.2f%%' % (100 * adj_ret))

# ---------------------------------------------------------------- point-in-time TTM as-of series
T = T.sort_values('avail').reset_index(drop=True)
T['eps_norm'] = T.eps_ttm
# normalized variant: add the state-aid charge back to any TTM window that contains 2024-09-28
mask = (T.end >= SA_QEND) & (T.end <= SA_QEND + pd.Timedelta(days=290))
T.loc[mask, 'eps_norm'] = (T.loc[mask, 'ttm'] + STATE_AID) / T.loc[mask, 'sh']
print('\nTTM windows that CONTAIN the state-aid quarter (normalization applies to these only):')
print(T.loc[mask].assign(end=lambda d: d.end.dt.date, avail=lambda d: d.avail.dt.date,
                         eps_ttm=lambda d: d.eps_ttm.round(3), eps_norm=lambda d: d.eps_norm.round(3),
                         sh=lambda d: (d.sh/1e6).round(0))[['end','avail','sh','eps_ttm','eps_norm']].to_string(index=False))


def asof(dates, col):
    """merge_asof: for each price date, the most recent TTM value whose filing was on/before it."""
    left = pd.DataFrame({'d': dates}).sort_values('d')
    right = T[['avail', col, 'ttm_rev', 'sh', 'end']].sort_values('avail')
    m = pd.merge_asof(left, right, left_on='d', right_on='avail', direction='backward')
    return m.set_index('d')


A = asof(px.index, 'eps_ttm')
A['eps_norm'] = asof(px.index, 'eps_norm')['eps_norm'].values
A['close_raw'] = px.close_raw.values
A['close_adj'] = px.close_adj.values
A = A.dropna(subset=['eps_ttm'])
A['pe_raw'] = A.close_raw / A.eps_ttm
A['pe_norm'] = A.close_raw / A.eps_norm
A['pe_divadj'] = A.close_adj / A.eps_ttm
A['ps_raw'] = A.close_raw * A.sh / A.ttm_rev
A['ps_divadj'] = A.close_adj * A.sh / A.ttm_rev
A = A[(A.pe_raw > 0) & (A.pe_raw < 300)]

print('\n=== DECOMPOSITION (as an investor would have computed it on each date) ===')
for lbl, c in [('as-reported GAAP', 'pe_raw'), ('state-aid normalized', 'pe_norm')]:
    e0 = A.eps_ttm.loc[prior] if c == 'pe_raw' else A.eps_norm.loc[prior]
    e1 = A.eps_ttm.iloc[-1] if c == 'pe_raw' else A.eps_norm.iloc[-1]
    m0, m1 = A[c].loc[prior], A[c].iloc[-1]
    ge, gm = e1 / e0 - 1, m1 / m0 - 1
    cross = ge * gm
    tot = (1 + ge) * (1 + gm) - 1
    print('\n %s' % lbl)
    print('   EPS  %.3f -> %.3f  = %+.1f%%' % (e0, e1, 100 * ge))
    print('   P/E  %.2fx -> %.2fx = %+.1f%%' % (m0, m1, 100 * gm))
    print('   product %+.2f%%  (actual price return %+.2f%%)' % (100 * tot, 100 * ret))
    print('   share of return: EPS %.1f%%   multiple %.1f%%   cross %.1f%%   [linear split]'
          % (100 * ge / tot, 100 * gm / tot, 100 * cross / tot))
    le, lm = np.log1p(ge), np.log1p(gm)
    print('   share of return: EPS %.1f%%   multiple %.1f%%              [log split, no residual]'
          % (100 * le / (le + lm), 100 * lm / (le + lm)))

print('\n=== PERCENTILES ===')
print('today  P/E(raw close, GAAP TTM) = %.2fx    P/S = %.2fx    TTM rev $%.0fM  shares %.0fM'
      % (A.pe_raw.iloc[-1], A.ps_raw.iloc[-1], A.ttm_rev.iloc[-1] / 1e6, A.sh.iloc[-1] / 1e6))
for win_label, start in [('full history', A.index[0]),
                         ('10 calendar years', LAST - pd.Timedelta(days=3653)),
                         ('4199 calendar days', LAST - pd.Timedelta(days=4199)),
                         ('4199 trading days', A.index[max(0, len(A) - 4199)]),
                         ('since 2013-01-01', pd.Timestamp('2013-01-01'))]:
    w = A[A.index >= start]
    if len(w) < 50:
        continue
    row = []
    for c, cur in [('pe_raw', A.pe_raw.iloc[-1]), ('pe_divadj', A.pe_divadj.iloc[-1]),
                   ('ps_raw', A.ps_raw.iloc[-1]), ('ps_divadj', A.ps_divadj.iloc[-1])]:
        pct = 100.0 * (w[c] <= cur).mean()
        row.append('%s %.1f%%ile (med %.2fx)' % (c, pct, w[c].median()))
    print('\n %-20s n=%d days, %s..%s' % (win_label, len(w), w.index[0].date(), w.index[-1].date()))
    for r in row:
        print('    ', r)

print('\nby-year medians (raw close / GAAP TTM EPS, first-filed):')
byy = A.groupby(A.index.year).agg(pe_raw=('pe_raw', 'median'), pe_divadj=('pe_divadj', 'median'),
                                  ps_raw=('ps_raw', 'median'), n=('pe_raw', 'size'))
print(byy.round(2).to_string())
A.to_csv('_fd_AAPL_verify_pe_pit.csv')
print('\n-> _fd_AAPL_verify_pe_pit.csv')
