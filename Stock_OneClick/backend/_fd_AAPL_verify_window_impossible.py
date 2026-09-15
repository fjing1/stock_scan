"""Part 5: (a) can ANY grid/window simultaneously give n=4,199 AND median P/S 6.82x?
(b) the 1-year return anchor; (c) the multiple-expansion share under 4 defensible normalizations.
"""
import json
import pandas as pd, numpy as np, yfinance as yf

A = pd.read_csv('_fd_AAPL_verify_pe_epssum.csv', index_col=0, parse_dates=True)
LAST = A.index[-1]
CPE = 333.08 / A.eps.iloc[-1]
CPS = 333.08 * A.cover.iloc[-1] / A.rev.iloc[-1]
print('panel: n=%d trading days %s..%s | today P/E %.3fx  P/S %.3fx'
      % (len(A), A.index[0].date(), LAST.date(), CPE, CPS))
print('  earliest possible point-in-time date is bounded by SEC XBRL start; panel begins %s'
      % A.index[0].date())

print('\n(a) IS n=4,199 COMPATIBLE WITH median P/S = 6.82x ?  scan every window length:')
print('%-9s %-11s %6s %8s %8s %7s %7s' % ('grid', 'start', 'n', 'peMed', 'psMed', 'pe%ile', 'ps%ile'))
grids = {'trading': A,
         'busday': A.reindex(pd.bdate_range(A.index[0], LAST)).ffill().dropna(subset=['eps']),
         'calendar': A.reindex(pd.date_range(A.index[0], LAST, freq='D')).ffill().dropna(subset=['eps'])}
hits = []
for gname, G in grids.items():
    for st in pd.date_range(A.index[0], '2021-06-01', freq='MS'):
        w = G[G.index >= st]
        if len(w) < 150: continue
        hits.append(dict(grid=gname, start=st, n=len(w), peMed=w.pe.median(), psMed=w.ps.median(),
                         pep=100*(w.pe <= CPE).mean(), psp=100*(w.ps <= CPS).mean()))
H = pd.DataFrame(hits)
# rows whose n is within 3% of 4199
near = H[(H.n > 4199*0.97) & (H.n < 4199*1.03)]
print('  windows with n within 3%% of 4,199 (any grid):')
if len(near) == 0:
    print('    NONE. max n by grid: %s' % H.groupby('grid').n.max().to_dict())
else:
    for r in near.itertuples():
        print('    %-9s %s n=%d peMed %.2f psMed %.2f pe%%ile %.1f ps%%ile %.1f'
              % (r.grid, r.start.date(), r.n, r.peMed, r.psMed, r.pep, r.psp))
print('\n  windows whose median P/S is within 0.05 of 6.82x:')
for r in H[(H.psMed - 6.82).abs() < 0.06].itertuples():
    print('    %-9s start %s n=%d  peMed %.2f psMed %.2f  pe%%ile %.1f  ps%%ile %.1f  (=%.1f yr)'
          % (r.grid, r.start.date(), r.n, r.peMed, r.psMed, r.pep, r.psp, (LAST-r.start).days/365.25))
print('\n  windows whose P/E percentile is within 0.15 of 96.8:')
for r in H[(H.pep - 96.8).abs() < 0.16].itertuples():
    print('    %-9s start %s n=%d  peMed %.2f psMed %.2f  pe%%ile %.1f  ps%%ile %.1f  (=%.1f yr)'
          % (r.grid, r.start.date(), r.n, r.peMed, r.psMed, r.pep, r.psp, (LAST-r.start).days/365.25))
print('\n  windows whose P/S percentile is within 0.06 of 99.4:')
for r in H[(H.psp - 99.4).abs() < 0.07].itertuples():
    print('    %-9s start %s n=%d  peMed %.2f psMed %.2f  pe%%ile %.1f  ps%%ile %.1f  (=%.1f yr)'
          % (r.grid, r.start.date(), r.n, r.peMed, r.psMed, r.pep, r.psp, (LAST-r.start).days/365.25))
print('\n  the STRICT 10-calendar-year window the claim names:')
w = A[A.index >= LAST - pd.Timedelta(days=3653)]
print('    trading  start %s n=%d  peMed %.2f psMed %.2f  pe%%ile %.1f  ps%%ile %.1f'
      % (w.index[0].date(), len(w), w.pe.median(), w.ps.median(),
         100*(w.pe <= CPE).mean(), 100*(w.ps <= CPS).mean()))

# ---------------------------------------------------------------- (b) 1-year anchor
raw = yf.Ticker('AAPL').history(period='max', auto_adjust=False)
raw.index = pd.to_datetime(raw.index).tz_localize(None)
px = raw['Close'].dropna()
print('\n(b) 1-YEAR RETURN ANCHOR (raw split-adjusted closes, price only)')
print('   what prior price gives exactly +42.8%%?  333.08/1.428 = %.2f' % (333.08/1.428))
sub = px['2025-08-25':'2025-10-05']
print(sub.round(2).to_string())
