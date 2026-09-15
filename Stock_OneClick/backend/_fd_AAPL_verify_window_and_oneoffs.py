"""Part 3: (a) which historical window reproduces the claim's 96.8 / 99.4 / median 6.82x triple;
(b) quarterly YoY earnings path for the forward-reversal test; (c) the claim's own
'independently derived $10,175M from a 50.23% vs 15.87% effective-rate gap' cross-check;
(d) does the CURRENT TTM contain one-offs of its own (asymmetric-normalization test)?
"""
import json
import pandas as pd, numpy as np

A = pd.read_csv('_fd_AAPL_verify_pe_pit_splitfix.csv', index_col=0, parse_dates=True)
EN = A.eps.iloc[-1]
CUR_PE = 333.08 / EN
CUR_PS_cover = 333.08 * 14594e6 / A.ttm_rev.iloc[-1]
CUR_PS_wtd = 333.08 * A.sh.iloc[-1] / A.ttm_rev.iloc[-1]
print('today: EPS %.4f  P/E %.3fx   P/S(cover) %.3fx   P/S(wtd) %.3fx' % (EN, CUR_PE, CUR_PS_cover, CUR_PS_wtd))

print('\n(a) WINDOW SCAN -- what start date reproduces the claim (P/E 96.8%ile, P/S 99.4%ile, P/S median 6.82x)?')
print('%-12s %5s %7s %7s %8s %8s' % ('start', 'n', 'pe%ile', 'ps%ile', 'peMed', 'psMed'))
best = None
for st in pd.date_range('2010-11-01', '2021-01-01', freq='MS'):
    w = A[A.index >= st]
    if len(w) < 200: continue
    pep = 100 * (w.pe <= CUR_PE).mean()
    psp = 100 * (w.ps <= CUR_PS_wtd).mean()
    err = abs(w.ps.median() - 6.82)
    if best is None or err < best[0]: best = (err, st, len(w), pep, psp, w.pe.median(), w.ps.median())
    if st.month in (1, 7):
        print('%-12s %5d %7.1f %7.1f %8.2f %8.2f' % (st.date(), len(w), pep, psp, w.pe.median(), w.ps.median()))
print('\n  closest match to psMed 6.82x -> start %s  n=%d  pe%%ile %.1f  ps%%ile %.1f  peMed %.2f psMed %.2f'
      % (best[1].date(), best[2], best[3], best[4], best[5], best[6]))
w10 = A[A.index >= A.index[-1] - pd.Timedelta(days=3653)]
print('  strict 10 calendar years: n=%d  pe%%ile %.1f  ps%%ile %.1f  psMed %.2f'
      % (len(w10), 100*(w10.pe <= CUR_PE).mean(), 100*(w10.ps <= CUR_PS_wtd).mean(), w10.ps.median()))
print('  full series             : n=%d  pe%%ile %.1f  ps%%ile %.1f  psMed %.2f'
      % (len(A), 100*(A.pe <= CUR_PE).mean(), 100*(A.ps <= CUR_PS_wtd).mean(), A.ps.median()))
print('  NOTE claim states n=4,199 days. My series has %d trading days total (starts %s).'
      % (len(A), A.index[0].date()))

# ---------------------------------------------------------------- (b) quarterly YoY
cf = json.load(open('_fd_AAPL_companyfacts.json'))
def ff(tag, unit, lo, hi):
    node = cf['facts']['us-gaap'].get(tag)
    rows=[dict(start=x.get('start'),end=x.get('end'),val=x['val'],filed=x['filed'],accn=x.get('accn'),form=x.get('form'))
          for u,arr in node['units'].items() if u==unit for x in arr]
    d=pd.DataFrame(rows)
    for c in ('start','end','filed'): d[c]=pd.to_datetime(d[c])
    d=d.dropna(subset=['start']); d['days']=(d.end-d.start).dt.days
    d=d[(d.days>=lo)&(d.days<=hi)]
    return d.sort_values('filed').groupby(['start','end'],as_index=False).first().sort_values('end')

qni, ani = ff('NetIncomeLoss','USD',60,110), ff('NetIncomeLoss','USD',330,400)
qrev = ff('RevenueFromContractWithCustomerExcludingAssessedTax','USD',60,110)
arev = ff('RevenueFromContractWithCustomerExcludingAssessedTax','USD',330,400)
qtax = ff('IncomeTaxExpenseBenefit','USD',60,110); atax = ff('IncomeTaxExpenseBenefit','USD',330,400)
qpre = ff('IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest','USD',60,110)
apre = ff('IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest','USD',330,400)

def grid(q,a):
    g={r.end.normalize():r.val for r in q.itertuples()}
    for r in a.itertuples():
        fe=r.end.normalize()
        inner=[v for e,v in g.items() if r.start<=e<fe and (fe-e).days<=300]
        if len(inner)==3 and fe not in g: g[fe]=r.val-sum(inner)
    return g
gni,grev,gtax,gpre = grid(qni,ani),grid(qrev,arev),grid(qtax,atax),grid(qpre,apre)
ends=sorted(set(gni)&set(grev)&set(gtax)&set(gpre))
print('\n(b) QUARTERLY path, last 10 quarters (Q4 derived from FY minus Q1-Q3):')
print('%-12s %10s %8s %10s %8s %10s %7s' % ('qtr end','rev $M','rev YoY','NI $M','NI YoY','pretax $M','ETR'))
for e in ends[-10:]:
    prev=[x for x in ends if 340<=(e-x).days<=390]
    ry=ny=np.nan
    if prev:
        p=prev[-1]; ry=grev[e]/grev[p]-1; ny=gni[e]/gni[p]-1
    print('%-12s %10.0f %7s %10.0f %7s %10.0f %6.2f%%' % (
        e.date(), grev[e]/1e6, ('%+.1f%%'%(100*ry)) if ry==ry else '-', gni[e]/1e6,
        ('%+.1f%%'%(100*ny)) if ny==ny else '-', gpre[e]/1e6, 100*gtax[e]/gpre[e]))

# (c) the claim's rate-gap derivation
q_sa = pd.Timestamp('2024-09-28')
etr_hist = sorted(100*gtax[e]/gpre[e] for e in ends if e != q_sa)
med = float(np.median(etr_hist))
print('\n(c) rate-gap cross-check for the 2024-09-28 quarter')
print('    quarter ETR = %.2f%%   median ETR of all other %d quarters = %.2f%%'
      % (100*gtax[q_sa]/gpre[q_sa], len(etr_hist), med))
print('    implied one-off = pretax %.0fM x (%.2f%% - %.2f%%) = $%.0fM   (filed line item: $10,246M)'
      % (gpre[q_sa]/1e6, 100*gtax[q_sa]/gpre[q_sa], med, gpre[q_sa]*(gtax[q_sa]/gpre[q_sa]-med/100)/1e6))

# (d) asymmetric-normalization test: is the CURRENT TTM tax rate normal?
print('\n(d) ASYMMETRIC-NORMALIZATION TEST: ETR of the 4 quarters in the CURRENT TTM vs history')
cur4 = ends[-4:]
for e in cur4:
    print('    %s  pretax %9.0fM  tax %8.0fM  ETR %6.2f%%' % (e.date(), gpre[e]/1e6, gtax[e]/1e6, 100*gtax[e]/gpre[e]))
ttm_pre = sum(gpre[e] for e in cur4); ttm_tax = sum(gtax[e] for e in cur4)
print('    CURRENT TTM ETR = %.2f%%   vs median quarter ETR %.2f%%' % (100*ttm_tax/ttm_pre, med))
# what would current TTM net income be at the historical median ETR?
alt = ttm_pre*(1-med/100)
print('    TTM NI as reported $%.0fM ; at median ETR $%.0fM  (diff $%.0fM = %+.1f%%)'
      % (sum(gni[e] for e in cur4)/1e6, alt/1e6, (alt-sum(gni[e] for e in cur4))/1e6,
         100*(alt/sum(gni[e] for e in cur4)-1)))
prior4 = [x for x in ends if x <= pd.Timestamp('2025-06-28')][-4:]
print('    PRIOR-YEAR-ANCHOR TTM (ending 2025-06-28) quarters: %s' % ', '.join(str(x.date()) for x in prior4))
pp = sum(gpre[e] for e in prior4); pt = sum(gtax[e] for e in prior4)
print('    that TTM ETR = %.2f%% ; at median ETR NI would be $%.0fM vs reported $%.0fM'
      % (100*pt/pp, pp*(1-med/100)/1e6, sum(gni[e] for e in prior4)/1e6))
