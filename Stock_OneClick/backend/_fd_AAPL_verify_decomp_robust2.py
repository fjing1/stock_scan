"""Part 6b: FIXED. Uses the properly built TTM panel (Q4 derived from FY minus Q1-Q3) instead of
naively summing quarterly EPS facts -- Apple has no Q4 duration fact, which is exactly the trap.
"""
import pandas as pd, numpy as np, yfinance as yf

A = pd.read_csv('_fd_AAPL_verify_pe_epssum.csv', index_col=0, parse_dates=True)
raw = yf.Ticker('AAPL').history(period='max', auto_adjust=False)
raw.index = pd.to_datetime(raw.index).tz_localize(None)
px_raw, px_adj = raw['Close'].dropna(), raw['Adj Close'].dropna()

P_NOW = 333.08
EPS_NOW = A.eps.iloc[-1]
# what TTM EPS was publicly available on each candidate 1-year-ago date
print('TTM EPS available on candidate 1-year-ago dates (point-in-time, from the panel):')
for d in ['2025-09-11', '2025-09-12', '2025-09-15']:
    r = A.loc[pd.Timestamp(d)]
    print('  %s: close %.2f  TTM EPS %.4f (period end %s)  -> P/E %.2fx'
          % (d, r.close, r.eps, r['end'], r.pe))
EPS_PRIOR = A.loc[pd.Timestamp('2025-09-12'), 'eps']
STATE_AID, SH_PRIOR = 10246e6, 15126e6
addback = STATE_AID / SH_PRIOR
EPS_PRIOR_N = EPS_PRIOR + addback
# symmetric median-ETR variant (part 3: prior TTM NI 99,280 -> 109,253 ; current 128,930 -> 131,495)
EPS_PRIOR_S = EPS_PRIOR * 109253/99280
EPS_NOW_S = EPS_NOW * 131495/128930
print('\nprior-year TTM EPS as-reported %.4f ; +state-aid add-back $%.3f -> %.4f ; symmetric-ETR -> %.4f'
      % (EPS_PRIOR, addback, EPS_PRIOR_N, EPS_PRIOR_S))
print('current  TTM EPS as-reported %.4f ; symmetric-ETR -> %.4f' % (EPS_NOW, EPS_NOW_S))
print('  claim states prior normalized EPS 7.27 (mine %.2f, gap %+.1f%%) and current 8.71 (mine %.2f)'
      % (EPS_PRIOR_N, 100*(7.27/EPS_PRIOR_N-1), EPS_NOW))

anchors = {'nearest-365d raw (%s)' % px_raw.index[px_raw.index.get_indexer(
                [px_raw.index[-1]-pd.Timedelta(days=365)], method='nearest')][0].date():
           px_raw.iloc[px_raw.index.get_indexer([px_raw.index[-1]-pd.Timedelta(days=365)], method='nearest')[0]],
           '2025-09-12 raw': px_raw.loc['2025-09-12'],
           '2025-09-15 raw': px_raw.loc['2025-09-15'],
           'implied by claimed +42.8%': P_NOW/1.428}

print('\n=== IS "45%% OF THE RETURN IS MULTIPLE EXPANSION" ROBUST? ===')
print('%-26s %7s %7s | %-24s %7s %7s %8s %8s' % ('price anchor','P_then','1Y ret','normalization','EPS g','P/E g','mult% lin','mult% log'))
for albl, P0 in anchors.items():
    ret = P_NOW/P0-1
    for nlbl, e0, e1 in [('none (as-reported GAAP)', EPS_PRIOR, EPS_NOW),
                         ('state-aid add-back only', EPS_PRIOR_N, EPS_NOW),
                         ('symmetric median-ETR', EPS_PRIOR_S, EPS_NOW_S)]:
        ge, gm = e1/e0-1, (P_NOW/e1)/(P0/e0)-1
        tot = (1+ge)*(1+gm)-1
        le, lm = np.log1p(ge), np.log1p(gm)
        print('%-26s %7.2f %+6.1f%% | %-24s %+6.1f%% %+6.1f%% %8.1f%% %8.1f%%   (P/E %.1fx->%.1fx)'
              % (albl, P0, 100*ret, nlbl, 100*ge, 100*gm, 100*gm/tot, 100*lm/(le+lm), P0/e0, P_NOW/e1))
    print()

print('=== FORWARD CHECK ===')
print('trailing P/E %.2fx on TTM EPS %.3f' % (P_NOW/EPS_NOW, EPS_NOW))
ttm_yoy = EPS_NOW/EPS_PRIOR-1
print('TTM EPS YoY as-reported %+.1f%% ; vs the state-aid-normalized base %+.1f%%'
      % (100*ttm_yoy, 100*(EPS_NOW/EPS_PRIOR_N-1)))
for gr in (0.10,0.15,0.20,0.25):
    print('  NTM EPS at %+.0f%% -> %.2f -> forward P/E %.1fx  (vs the 10-yr MEDIAN trailing P/E of 27.6x)'
          % (100*gr, EPS_NOW*(1+gr), P_NOW/(EPS_NOW*(1+gr))))
print('\nTTM EPS growth path (point-in-time panel, last 8 TTM readings):')
u = A.drop_duplicates(subset=['end']).tail(8)[['end','eps','close','pe','ps']]
print(u.assign(eps=lambda d: d.eps.round(3), pe=lambda d: d.pe.round(1), ps=lambda d: d.ps.round(2)).to_string())
