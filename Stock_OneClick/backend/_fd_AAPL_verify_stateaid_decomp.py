"""Adversarial verification of the AAPL 'state aid normalization / 45% multiple expansion /
96.8th-99.4th percentile' claim.  Independent recomputation from SEC companyfacts + raw prices.

Everything is built from scratch here; none of the prior _fd_AAPL_* scripts are imported.
"""
import json, gzip, sys, os
from datetime import date, datetime, timedelta
import pandas as pd, numpy as np

CF = '_fd_AAPL_companyfacts.json'
cf = json.load(open(CF))
US = cf['facts']['us-gaap']
DEI = cf['facts']['dei']


def facts(ns, tag, unit=None):
    node = (cf['facts'][ns]).get(tag)
    if node is None:
        return pd.DataFrame()
    rows = []
    for u, arr in node['units'].items():
        if unit and u != unit:
            continue
        for f in arr:
            rows.append(dict(tag=tag, unit=u, start=f.get('start'), end=f.get('end'),
                             val=f['val'], filed=f['filed'], accn=f.get('accn'),
                             form=f.get('form'), fy=f.get('fy'), fp=f.get('fp'),
                             frame=f.get('frame')))
    df = pd.DataFrame(rows)
    for c in ('start', 'end', 'filed'):
        if c in df:
            df[c] = pd.to_datetime(df[c])
    return df


def dur(df):
    d = df.dropna(subset=['start']).copy()
    d['days'] = (d['end'] - d['start']).dt.days
    return d


print('=' * 100)
print('STEP 0  data freshness')
eps = dur(facts('us-gaap', 'EarningsPerShareDiluted', 'USD/shares'))
ni = dur(facts('us-gaap', 'NetIncomeLoss', 'USD'))
print('  EPS facts', len(eps), 'max end', eps.end.max().date(), 'max filed', eps.filed.max().date())
print('  NI  facts', len(ni), 'max end', ni.end.max().date(), 'max filed', ni.filed.max().date())

# ---------------------------------------------------------------- revenue ladder
REV_TAGS = ['RevenueFromContractWithCustomerExcludingAssessedTax',
            'RevenueFromContractWithCustomerIncludingAssessedTax',
            'Revenues', 'SalesRevenueNet', 'SalesRevenueGoodsNet']
rev = pd.concat([dur(facts('us-gaap', t, 'USD')) for t in REV_TAGS if t in US], ignore_index=True)

SPLITS = [(pd.Timestamp('2014-06-09'), 7.0), (pd.Timestamp('2020-08-31'), 4.0)]


def split_factor(filed):
    """cumulative split factor to divide an as-filed per-share number by, to put it on
    today's post-split basis."""
    f = 1.0
    for d, r in SPLITS:
        if filed < d:
            f *= r
    return f


# ---------------------------------------------------------------- point in time quarterly EPS
def first_filed(df, lo, hi):
    d = df[(df.days >= lo) & (df.days <= hi)].copy()
    d = d.sort_values('filed').groupby(['start', 'end'], as_index=False).first()
    return d.sort_values('end')


q_eps = first_filed(eps, 60, 110)
a_eps = first_filed(eps, 330, 400)
print('\nSTEP 1  first-filed EPS series (no restatement lookahead)')
print('  quarterly n=%d  %s..%s' % (len(q_eps), q_eps.end.min().date(), q_eps.end.max().date()))
print('  annual    n=%d  %s..%s' % (len(a_eps), a_eps.end.min().date(), a_eps.end.max().date()))

q_ni = first_filed(ni, 60, 110)
a_ni = first_filed(ni, 330, 400)
q_rev = first_filed(rev, 60, 110)
a_rev = first_filed(rev, 330, 400)


# ---------------------------------------------------------------- build TTM in DOLLARS (avoids
# the per-share split problem entirely) then divide by point-in-time diluted shares.
def build_ttm_dollars(qdf, adf, label):
    """Return list of (available_from, period_end, ttm_value, how). Q4 is derived as
    FY - (Q1+Q2+Q3) which is the standard fix for Apple's missing Q4 duration fact."""
    q = {(r.start, r.end): r for r in qdf.itertuples()}
    qs = sorted(q, key=lambda k: k[1])
    out = []
    # quarterly TTM: at each quarter end, sum trailing 4 quarter values (deriving Q4 from FY)
    # first assemble a clean quarter grid
    grid = {}   # period_end -> (val, filed)
    for (s, e), r in q.items():
        grid[e.normalize()] = (r.val, r.filed)
    # derive Q4 from annual
    for r in adf.itertuples():
        fy_end, fy_start = r.end.normalize(), r.start
        # find the 3 quarters inside this FY
        inner = [(e, v) for e, (v, f) in grid.items() if fy_start <= e < fy_end and (fy_end - e).days <= 300]
        if len(inner) == 3:
            q4 = r.val - sum(v for _, v in inner)
            if fy_end not in grid:
                grid[fy_end] = (q4, r.filed)
    ends = sorted(grid)
    for i in range(3, len(ends)):
        window = ends[i - 3:i + 1]
        # require the window to actually span ~1 year
        span = (window[-1] - window[0]).days
        if not (250 <= span <= 300):
            continue
        vals = [grid[e][0] for e in window]
        filed = max(grid[e][1] for e in window)
        out.append(dict(end=window[-1], ttm=sum(vals), avail=filed))
    d = pd.DataFrame(out).sort_values('avail').reset_index(drop=True)
    print('  %s TTM points n=%d  %s..%s' % (label, len(d), d.end.min().date(), d.end.max().date()))
    return d


print('\nSTEP 2  TTM in dollars (Q4 derived from FY minus Q1-Q3)')
ttm_ni = build_ttm_dollars(q_ni, a_ni, 'net income')
ttm_rev = build_ttm_dollars(q_rev, a_rev, 'revenue   ')

# diluted share count, point in time, first-filed quarterly
sh = dur(facts('us-gaap', 'WeightedAverageNumberOfDilutedSharesOutstanding', 'shares'))
q_sh = first_filed(sh, 60, 110)
a_sh = first_filed(sh, 330, 400)
# shares outstanding on the cover (dei) - actual, not weighted
cover = facts('dei', 'EntityCommonStockSharesOutstanding', 'shares')
cover = cover.dropna(subset=['end']).sort_values('filed')

print('\nSTEP 3  share counts')
print('  weighted diluted quarterly n=%d  latest %s = %.0fM (filed %s)' % (
    len(q_sh), q_sh.end.max().date(), q_sh.iloc[-1].val / 1e6, q_sh.iloc[-1].filed.date()))
print('  dei cover shares latest %.0fM  (filed %s, as-of %s)' % (
    cover.iloc[-1].val / 1e6, cover.iloc[-1].filed.date(), cover.iloc[-1].end.date()))

ttm_sh = []
for r in ttm_ni.itertuples():
    w = q_sh[q_sh.end <= r.end].tail(4)
    if len(w) == 4:
        ttm_sh.append(dict(end=r.end, sh=w.val.mean(), sh_last=w.iloc[-1].val))
    else:
        # fall back on the annual weighted diluted count
        aw = a_sh[a_sh.end <= r.end]
        if len(aw):
            ttm_sh.append(dict(end=r.end, sh=aw.iloc[-1].val, sh_last=aw.iloc[-1].val))
ttm_sh = pd.DataFrame(ttm_sh)
T = ttm_ni.merge(ttm_rev[['end', 'ttm']].rename(columns={'ttm': 'ttm_rev'}), on='end', how='left')
T = T.merge(ttm_sh, on='end', how='left')
T['eps_ttm'] = T.ttm / T.sh
T = T.dropna(subset=['eps_ttm']).sort_values('avail').reset_index(drop=True)

print('\n  last 10 TTM points (dollars, millions; EPS on the as-of-then share basis):')
print(T.tail(10).assign(end=lambda d: d.end.dt.date, avail=lambda d: d.avail.dt.date,
                        ttm=lambda d: (d.ttm / 1e6).round(0),
                        ttm_rev=lambda d: (d.ttm_rev / 1e6).round(0),
                        sh=lambda d: (d.sh / 1e6).round(0),
                        eps_ttm=lambda d: d.eps_ttm.round(3))[
    ['end', 'avail', 'ttm', 'ttm_rev', 'sh', 'eps_ttm']].to_string(index=False))
T.to_csv('_fd_AAPL_verify_ttm_pit.csv', index=False)
print('\n  -> _fd_AAPL_verify_ttm_pit.csv')
