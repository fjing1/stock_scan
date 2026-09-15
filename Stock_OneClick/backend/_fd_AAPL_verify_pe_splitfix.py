"""Part 2b: same as part 2 but with the 7:1 (2014-06-09) and 4:1 (2020-08-31) splits rescaled
onto today's share basis, applied PER-FACT using each fact's own `filed` date -- which is the only
correct way, because a TTM window can straddle a split date.
"""
import json
import pandas as pd, numpy as np, yfinance as yf

cf = json.load(open('_fd_AAPL_companyfacts.json'))
SPLITS = [(pd.Timestamp('2014-06-09'), 7.0), (pd.Timestamp('2020-08-31'), 4.0)]
STATE_AID = 10246e6
SA_QEND = pd.Timestamp('2024-09-28')


def sf(filed):
    f = 1.0
    for d, r in SPLITS:
        if filed < d:
            f *= r
    return f


def facts(ns, tag, unit):
    node = cf['facts'][ns].get(tag)
    if node is None:
        return pd.DataFrame()
    rows = [dict(start=x.get('start'), end=x.get('end'), val=x['val'], filed=x['filed'],
                 accn=x.get('accn'), form=x.get('form'))
            for u, arr in node['units'].items() if u == unit for x in arr]
    d = pd.DataFrame(rows)
    for c in ('start', 'end', 'filed'):
        d[c] = pd.to_datetime(d[c])
    return d


def firstfiled(tag, unit, lo, hi, ns='us-gaap'):
    d = facts(ns, tag, unit).dropna(subset=['start'])
    d['days'] = (d.end - d.start).dt.days
    d = d[(d.days >= lo) & (d.days <= hi)]
    return d.sort_values('filed').groupby(['start', 'end'], as_index=False).first().sort_values('end')


q_ni, a_ni = firstfiled('NetIncomeLoss', 'USD', 60, 110), firstfiled('NetIncomeLoss', 'USD', 330, 400)
REVT = ['RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues', 'SalesRevenueNet']
q_rev = pd.concat([firstfiled(t, 'USD', 60, 110) for t in REVT], ignore_index=True) \
    .sort_values('filed').groupby(['start', 'end'], as_index=False).first().sort_values('end')
a_rev = pd.concat([firstfiled(t, 'USD', 330, 400) for t in REVT], ignore_index=True) \
    .sort_values('filed').groupby(['start', 'end'], as_index=False).first().sort_values('end')
SHT = 'WeightedAverageNumberOfDilutedSharesOutstanding'
q_sh, a_sh = firstfiled(SHT, 'shares', 60, 110), firstfiled(SHT, 'shares', 330, 400)
# rescale share counts onto today's post-split basis using each fact's own filing date
for d in (q_sh, a_sh):
    d['val_adj'] = d.apply(lambda r: r.val * sf(r.filed), axis=1)

print('share-count rescaling spot checks (as-filed -> today basis):')
for e in ['2013-06-29', '2014-06-28', '2020-06-27', '2020-09-26', '2021-06-26', '2026-06-27']:
    for src, nm in ((q_sh, 'Q'), (a_sh, 'A')):
        r = src[src.end == pd.Timestamp(e)]
        if len(r):
            r = r.iloc[0]
            print('  %s %s filed %s  %10.0fM -> %10.0fM  (x%g)'
                  % (nm, e, r.filed.date(), r.val / 1e6, r.val_adj / 1e6, sf(r.filed)))


def quarter_grid(qdf, adf, valcol='val'):
    g = {}
    for r in qdf.itertuples():
        g[r.end.normalize()] = (getattr(r, valcol), r.filed)
    for r in adf.itertuples():
        fe = r.end.normalize()
        inner = [(e, v) for e, (v, f) in g.items() if r.start <= e < fe and (fe - e).days <= 300]
        if len(inner) == 3 and fe not in g:
            g[fe] = (getattr(r, valcol) - sum(v for _, v in inner), r.filed)
    return g


gni, grev = quarter_grid(q_ni, a_ni), quarter_grid(q_rev, a_rev)
gsh = quarter_grid(q_sh, a_sh, 'val_adj')   # note: Q4 shares derived as FY*4 - Q1-Q3 is WRONG for
# an average, so rebuild Q4 shares as the annual average instead
gsh = {}
for r in q_sh.itertuples():
    gsh[r.end.normalize()] = (r.val_adj, r.filed)
for r in a_sh.itertuples():
    fe = r.end.normalize()
    if fe not in gsh:
        gsh[fe] = (r.val_adj, r.filed)     # FY weighted-avg diluted as the Q4 stand-in

rows = []
ends = sorted(set(gni) & set(grev))
for i in range(3, len(ends)):
    w = ends[i - 3:i + 1]
    if not (250 <= (w[-1] - w[0]).days <= 300):
        continue
    if not all(e in gsh for e in w):
        continue
    filed = max(max(gni[e][1] for e in w), max(grev[e][1] for e in w), max(gsh[e][1] for e in w))
    rows.append(dict(end=w[-1], avail=filed,
                     ttm_ni=sum(gni[e][0] for e in w), ttm_rev=sum(grev[e][0] for e in w),
                     sh=np.mean([gsh[e][0] for e in w])))
T = pd.DataFrame(rows).sort_values('avail').reset_index(drop=True)
T['eps'] = T.ttm_ni / T.sh
T['eps_norm'] = np.where((T.end >= SA_QEND) & (T.end <= SA_QEND + pd.Timedelta(days=290)),
                         (T.ttm_ni + STATE_AID) / T.sh, T.eps)
print('\nTTM points n=%d  %s..%s' % (len(T), T.end.min().date(), T.end.max().date()))

# ---------------------------------------------------------------- prices, raw split-adj close
raw = yf.Ticker('AAPL').history(period='max', auto_adjust=False)
raw.index = pd.to_datetime(raw.index).tz_localize(None)
px = pd.DataFrame({'raw': raw['Close'], 'adj': raw['Adj Close']}).dropna()
L = pd.DataFrame({'d': px.index}).sort_values('d')
M = pd.merge_asof(L, T.sort_values('avail'), left_on='d', right_on='avail', direction='backward')
A = M.set_index('d').dropna(subset=['eps'])
A['raw'], A['adj'] = px.raw.reindex(A.index), px.adj.reindex(A.index)
A['pe'] = A.raw / A.eps
A['pe_norm'] = A.raw / A.eps_norm
A['pe_divadj'] = A.adj / A.eps
A['ps'] = A.raw * A.sh / A.ttm_rev
A['ps_divadj'] = A.adj * A.sh / A.ttm_rev
A = A[(A.pe > 0) & (A.pe < 200)]

print('\nSANITY: by-year medians after the split fix (compare to known AAPL history)')
print(A.groupby(A.index.year).agg(pe=('pe','median'), pe_norm=('pe_norm','median'),
                                  pe_divadj=('pe_divadj','median'), ps=('ps','median'),
                                  n=('pe','size')).round(2).to_string())

PN, EN = A.raw.iloc[-1], A.eps.iloc[-1]
print('\nTODAY (%s): close %.2f  TTM EPS %.3f  P/E %.2fx  P/S %.2fx  (shares %.0fM, TTM rev $%.0fM)'
      % (A.index[-1].date(), PN, EN, A.pe.iloc[-1], A.ps.iloc[-1], A.sh.iloc[-1]/1e6, A.ttm_rev.iloc[-1]/1e6))
# also at the stated 2026-09-14 close
for p in (PN, 333.08):
    print('   at close %.2f -> P/E %.2fx   P/S(wtd dil %.0fM) %.2fx   P/S(dei cover 14594M) %.2fx'
          % (p, p/EN, A.sh.iloc[-1]/1e6, p*A.sh.iloc[-1]/A.ttm_rev.iloc[-1],
             p*14594e6/A.ttm_rev.iloc[-1]))

print('\nPERCENTILE of today\'s P/E and P/S across windows (current value vs history):')
cur = {'pe': 333.08/EN, 'pe_norm': 333.08/EN, 'pe_divadj': 333.08/EN,
       'ps': 333.08*A.sh.iloc[-1]/A.ttm_rev.iloc[-1], 'ps_divadj': 333.08*A.sh.iloc[-1]/A.ttm_rev.iloc[-1]}
LAST = A.index[-1]
for lbl, st in [('full (2010-)', A.index[0]),
                ('10 cal yr', LAST - pd.Timedelta(days=3653)),
                ('4199 cal d', LAST - pd.Timedelta(days=4199)),
                ('since 2013', pd.Timestamp('2013-01-01')),
                ('5 cal yr', LAST - pd.Timedelta(days=1826))]:
    w = A[A.index >= st]
    if len(w) < 50: continue
    out = []
    for c in ['pe', 'pe_norm', 'pe_divadj', 'ps', 'ps_divadj']:
        out.append('%s %.1f (med %.1fx)' % (c, 100*(w[c] <= cur[c]).mean(), w[c].median()))
    print(' %-12s n=%4d %s..%s | %s' % (lbl, len(w), w.index[0].date(), w.index[-1].date(), '  '.join(out)))

A.to_csv('_fd_AAPL_verify_pe_pit_splitfix.csv')
print('\n-> _fd_AAPL_verify_pe_pit_splitfix.csv')
