"""Part 6: robustness of the '45% of the return is multiple expansion' figure.
Vary (i) the 1-year price anchor, (ii) the normalization method (none / state-aid only /
symmetric median-ETR on BOTH ends), (iii) the attribution convention (linear-with-residual vs log).
Then the forward-reversal check: what does the multiple look like on forward earnings?
"""
import json
import pandas as pd, numpy as np, yfinance as yf

A = pd.read_csv('_fd_AAPL_verify_pe_epssum.csv', index_col=0, parse_dates=True)
T = pd.read_csv('_fd_AAPL_verify_ttm_pit.csv', parse_dates=['end', 'avail'])
raw = yf.Ticker('AAPL').history(period='max', auto_adjust=False)
raw.index = pd.to_datetime(raw.index).tz_localize(None)
px_raw, px_adj = raw['Close'].dropna(), raw['Adj Close'].dropna()

P_NOW = 333.08          # the close the claim and the measured context both use (2026-09-14)
EPS_NOW = 8.7100        # sum of 4 first-filed quarterly diluted EPS, verified
# prior-year TTM (available 2025-08-01, so it is what was public all of Sep 2025)
EPS_PRIOR_GAAP = 0.97 + 2.40 + 1.65 + 1.57      # from first-filed quarterly EPS, checked below
q = json.load(open('_fd_AAPL_companyfacts.json'))['facts']['us-gaap']['EarningsPerShareDiluted']['units']['USD/shares']
qq = pd.DataFrame(q); qq['end'] = pd.to_datetime(qq.end); qq['start'] = pd.to_datetime(qq.start)
qq = qq.dropna(subset=['start']); qq['days'] = (qq.end-qq.start).dt.days
qq = qq[(qq.days.between(60,110))].sort_values('filed').groupby('end', as_index=False).first()
want = pd.to_datetime(['2024-09-28','2024-12-28','2025-03-29','2025-06-28'])
print('first-filed quarterly diluted EPS in the prior-year TTM window:')
print(qq[qq.end.isin(want)][['end','val','filed','accn','form']].to_string(index=False))
EPS_PRIOR_GAAP = qq[qq.end.isin(want)].val.sum()
print('  sum = %.2f' % EPS_PRIOR_GAAP)

STATE_AID = 10246e6
SH_PRIOR = 15126e6      # FY-weighted diluted shares available at the time (verified part 1)
addback = STATE_AID / SH_PRIOR
print('  state-aid add-back = $10,246M / %.0fM sh = $%.3f/sh -> normalized prior EPS %.3f'
      % (SH_PRIOR/1e6, addback, EPS_PRIOR_GAAP + addback))

# symmetric median-ETR normalization (from part 3): prior TTM NI 109,253 vs 99,280 ; current 131,495 vs 128,930
EPS_PRIOR_SYM = EPS_PRIOR_GAAP * 109253/99280
EPS_NOW_SYM = EPS_NOW * 131495/128930

anchors = {}
for lbl, d in [('nearest 365 cal days (raw)', px_raw.index[px_raw.index.get_indexer([px_raw.index[-1]-pd.Timedelta(days=365)], method='nearest')][0]),
               ('2025-09-12 raw', pd.Timestamp('2025-09-12')),
               ('2025-09-15 raw', pd.Timestamp('2025-09-15'))]:
    anchors[lbl] = px_raw.loc[d]
anchors['implied by the claimed +42.8%'] = P_NOW/1.428
anchors['2025-09-12 div-adjusted'] = px_adj.loc[pd.Timestamp('2025-09-12')] * (px_raw.iloc[-1]/px_adj.iloc[-1])

print('\n=== SENSITIVITY OF THE "45%% IS MULTIPLE EXPANSION" HEADLINE ===')
print('%-34s %7s %7s | %-26s %7s %7s %7s %7s' % ('price anchor','P_then','1Y ret','normalization','EPSg','P/Eg','mult%L','mult%log'))
for albl, P0 in anchors.items():
    ret = P_NOW/P0 - 1
    for nlbl, e0, e1 in [('none (as-reported GAAP)', EPS_PRIOR_GAAP, EPS_NOW),
                         ('state-aid add-back only', EPS_PRIOR_GAAP+addback, EPS_NOW),
                         ('symmetric median-ETR', EPS_PRIOR_SYM, EPS_NOW_SYM)]:
        m0, m1 = P0/e0, P_NOW/e1
        ge, gm = e1/e0-1, m1/m0-1
        tot = (1+ge)*(1+gm)-1
        le, lm = np.log1p(ge), np.log1p(gm)
        print('%-34s %7.2f %+6.1f%% | %-26s %+6.1f%% %+6.1f%% %6.1f%% %7.1f%%'
              % (albl, P0, 100*ret, nlbl, 100*ge, 100*gm, 100*gm/tot, 100*lm/(le+lm)))
    print()

# ---------------------------------------------------------------- forward reversal
print('=== FORWARD-REVERSAL CHECK: the multiple capitalises FORWARD earnings ===')
print('P/E on trailing TTM EPS %.2f = %.2fx' % (EPS_NOW, P_NOW/EPS_NOW))
# annualise the latest quarter and the latest-4-quarter YoY growth
qni = qq.set_index('end').val
print('quarterly diluted EPS, last 9 quarters (first-filed):')
print(qq.sort_values('end').tail(9)[['end','val','filed']].to_string(index=False))
last4 = qq.sort_values('end').tail(4)
prior4 = qq.sort_values('end').iloc[-8:-4]
g = last4.val.sum()/prior4.val.sum() - 1
print('  TTM EPS YoY as-reported = %+.1f%% ; vs state-aid-normalized base = %+.1f%%'
      % (100*g, 100*(last4.val.sum()/(EPS_PRIOR_GAAP+addback) - 1)))
for gr in (0.10, 0.15, 0.20, 0.25):
    print('  if the next 4 quarters grow EPS %+.0f%% YoY -> NTM EPS %.2f -> forward P/E %.1fx'
          % (100*gr, EPS_NOW*(1+gr), P_NOW/(EPS_NOW*(1+gr))))
print('  latest quarter (2026-06-27) EPS YoY = %+.1f%%  [base quarter 2025-06-28 had a NORMAL 16.4%% ETR,'
      % (100*(qq[qq.end==pd.Timestamp('2026-06-27')].val.iloc[0]/qq[qq.end==pd.Timestamp('2025-06-28')].val.iloc[0]-1)))
print('   so this acceleration is NOT a state-aid base effect]')
