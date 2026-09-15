"""Part 4: reproduce the claim's exact series construction. Build P/E from the sum of first-filed
QUARTERLY diluted EPS (split-rescaled by each fact's own filed date) -- this is the construction
that can reach back to 2008 and therefore plausibly yields n=4,199 days. Also build P/S off the
dei cover share count. Then test: does ANY single window give (96.8, 99.4, psMed 6.82)?
"""
import json
import pandas as pd, numpy as np, yfinance as yf

cf = json.load(open('_fd_AAPL_companyfacts.json'))
SPLITS = [(pd.Timestamp('2014-06-09'), 7.0), (pd.Timestamp('2020-08-31'), 4.0)]
def sf(f):
    x = 1.0
    for d, r in SPLITS:
        if f < d: x *= r
    return x

def ff(ns, tag, unit, lo, hi):
    node = cf['facts'][ns].get(tag)
    if node is None: return pd.DataFrame()
    rows=[dict(start=x.get('start'),end=x.get('end'),val=x['val'],filed=x['filed'],accn=x.get('accn'),form=x.get('form'))
          for u,arr in node['units'].items() if u==unit for x in arr]
    d=pd.DataFrame(rows)
    for c in ('start','end','filed'): d[c]=pd.to_datetime(d[c])
    d=d.dropna(subset=['start']); d['days']=(d.end-d.start).dt.days
    d=d[(d.days>=lo)&(d.days<=hi)]
    return d.sort_values('filed').groupby(['start','end'],as_index=False).first().sort_values('end')

qe, ae = ff('us-gaap','EarningsPerShareDiluted','USD/shares',60,110), ff('us-gaap','EarningsPerShareDiluted','USD/shares',330,400)
qe['adj'] = qe.apply(lambda r: r.val/sf(r.filed), axis=1)
ae['adj'] = ae.apply(lambda r: r.val/sf(r.filed), axis=1)
print('quarterly EPS facts: n=%d  %s..%s' % (len(qe), qe.end.min().date(), qe.end.max().date()))
print('annual    EPS facts: n=%d  %s..%s' % (len(ae), ae.end.min().date(), ae.end.max().date()))
print('  split spot check: %s' % qe[qe.end.isin(pd.to_datetime(['2008-06-28','2013-06-29','2020-06-27','2026-06-27']))]
      [['end','filed','val','adj']].to_string(index=False))

g = {r.end.normalize(): (r.adj, r.filed) for r in qe.itertuples()}
for r in ae.itertuples():
    fe = r.end.normalize()
    inner = [(e, v) for e, (v, f) in g.items() if r.start <= e < fe and (fe-e).days <= 300]
    if len(inner) == 3 and fe not in g:
        g[fe] = (r.adj - sum(v for _, v in inner), r.filed)
ends = sorted(g)
rows=[]
for i in range(3, len(ends)):
    w = ends[i-3:i+1]
    if not (250 <= (w[-1]-w[0]).days <= 300): continue
    rows.append(dict(end=w[-1], avail=max(g[e][1] for e in w), eps=sum(g[e][0] for e in w)))
E = pd.DataFrame(rows).sort_values('avail').reset_index(drop=True)
print('\nEPS-sum TTM series: n=%d  %s..%s  (first available %s)'
      % (len(E), E.end.min().date(), E.end.max().date(), E.avail.min().date()))

# revenue TTM + cover shares for P/S
REVT=['RevenueFromContractWithCustomerExcludingAssessedTax','Revenues','SalesRevenueNet']
qr=pd.concat([ff('us-gaap',t,'USD',60,110) for t in REVT],ignore_index=True).sort_values('filed').groupby(['start','end'],as_index=False).first().sort_values('end')
ar=pd.concat([ff('us-gaap',t,'USD',330,400) for t in REVT],ignore_index=True).sort_values('filed').groupby(['start','end'],as_index=False).first().sort_values('end')
gr={r.end.normalize():(r.val,r.filed) for r in qr.itertuples()}
for r in ar.itertuples():
    fe=r.end.normalize(); inner=[(e,v) for e,(v,f) in gr.items() if r.start<=e<fe and (fe-e).days<=300]
    if len(inner)==3 and fe not in gr: gr[fe]=(r.val-sum(v for _,v in inner), r.filed)
re_=sorted(gr); rrows=[]
for i in range(3,len(re_)):
    w=re_[i-3:i+1]
    if not (250<=(w[-1]-w[0]).days<=300): continue
    rrows.append(dict(end=w[-1],avail=max(gr[e][1] for e in w),rev=sum(gr[e][0] for e in w)))
R=pd.DataFrame(rrows).sort_values('avail').reset_index(drop=True)

cov = ff('dei','EntityCommonStockSharesOutstanding','shares',-10**9,10**9) if False else None
node = cf['facts']['dei']['EntityCommonStockSharesOutstanding']
cr=[dict(end=x.get('end'),val=x['val'],filed=x['filed']) for u,a in node['units'].items() for x in a]
C=pd.DataFrame(cr); C['end']=pd.to_datetime(C.end); C['filed']=pd.to_datetime(C.filed)
C['adj']=C.apply(lambda r: r.val*sf(r.filed),axis=1)
C=C.sort_values('filed').groupby('filed',as_index=False).last()
print('cover-share facts n=%d  %s..%s' % (len(C), C.filed.min().date(), C.filed.max().date()))

raw = yf.Ticker('AAPL').history(period='max', auto_adjust=False)
raw.index = pd.to_datetime(raw.index).tz_localize(None)
px = raw['Close'].dropna()
L = pd.DataFrame({'d': px.index}).sort_values('d')
M = pd.merge_asof(L, E[['avail','eps','end']].sort_values('avail'), left_on='d', right_on='avail', direction='backward')
M = pd.merge_asof(M.sort_values('d'), R[['avail','rev']].rename(columns={'avail':'avail_r'}).sort_values('avail_r'),
                  left_on='d', right_on='avail_r', direction='backward')
M = pd.merge_asof(M.sort_values('d'), C[['filed','adj']].rename(columns={'filed':'avail_c','adj':'cover'}).sort_values('avail_c'),
                  left_on='d', right_on='avail_c', direction='backward')
A = M.set_index('d').dropna(subset=['eps'])
A['close'] = px.reindex(A.index)
A['pe'] = A.close/A.eps
A['ps'] = A.close*A.cover/A.rev
A = A[(A.pe>0)&(A.pe<200)]
print('\nEPS-sum P/E panel: n=%d trading days  %s..%s' % (len(A), A.index[0].date(), A.index[-1].date()))
print('  CLAIM SAYS n=4,199 days.  mine=%d' % len(A))
print('\nby-year medians:')
print(A.groupby(A.index.year).agg(pe=('pe','median'), ps=('ps','median'), n=('pe','size')).round(2).to_string())

CPE = 333.08/A.eps.iloc[-1]
CPS = 333.08*A.cover.iloc[-1]/A.rev.iloc[-1]
print('\ntoday: TTM EPS(sum of 4 first-filed qtrs) %.4f -> P/E %.3fx ; cover %.0fM, TTM rev $%.0fM -> P/S %.3fx'
      % (A.eps.iloc[-1], CPE, A.cover.iloc[-1]/1e6, A.rev.iloc[-1]/1e6, CPS))
print('   CLAIM: 38.24x P/E, 10.41x P/S')
print('\nwindow scan on this construction:')
print('%-12s %5s %7s %7s %8s %8s' % ('start','n','pe%ile','ps%ile','peMed','psMed'))
for st in [A.index[0]] + list(pd.date_range('2011-01-01','2020-01-01',freq='YS')) + [A.index[-1]-pd.Timedelta(days=3653)]:
    w=A[A.index>=st]
    if len(w)<200: continue
    print('%-12s %5d %7.1f %7.1f %8.2f %8.2f' % (pd.Timestamp(st).date(), len(w),
          100*(w.pe<=CPE).mean(), 100*(w.ps<=CPS).mean(), w.pe.median(), w.ps.median()))
# what window has exactly n=4199?
print('\n  n=4,199 would require a series starting %s (%d cal days = %.1f yr before today)'
      % ('~'+str((A.index[-1]-pd.Timedelta(days=int(4199*365.25/252))).date()),
         int(4199*365.25/252), 4199/252))
A.to_csv('_fd_AAPL_verify_pe_epssum.csv')
