"""_vix_wf_decayvsnoise3.py — final verification of the load-bearing numbers."""
import numpy as np, pandas as pd, _vix_data
d=_vix_data.add_features(_vix_data.load()); fwd=d['g5'].values; yrs=d.index.year.values
S={'s10':(d.stretch>=0.10).values,'s20':(d.stretch>=0.20).values,
   'bba':d['bb10_2.0_above'].values.astype(bool),'bbr':d['bb10_2.0_reentry'].values.astype(bool),
   'bbb':d['bb10_2.0_below'].values.astype(bool)}
late=yrs>=2015; early=yrs<=2014; v=~np.isnan(fwd)

print("A) 2020Q1 leverage — signal days in 2020-02-01..2020-06-30 and their g5")
cov=np.asarray((d.index>='2020-02-01')&(d.index<='2020-06-30'))
for k,m in S.items():
    sel=m&cov&v
    if sel.sum()==0: continue
    twenties=(yrs>=2020)
    a=(m&twenties&v); b=(m&twenties&~cov&v)
    print(f"  {k:5} covid n={int(sel.sum()):>3} mean g5 {fwd[sel].mean()*100:+6.2f}% | "
          f"2020s all n={int(a.sum())} cond {fwd[a].mean()*100:+.3f}% -> exCOVID n={int(b.sum())} "
          f"cond {fwd[b].mean()*100:+.3f}%  (worst single g5 in covid block: {fwd[sel].min()*100:+.1f}%)")

print("\nB) overlap of the late-window (2015-2026) signal days")
for k in S:
    for j in S:
        if k<j:
            a,b=S[k]&late&v,S[j]&late&v
            print(f"  {k:5} n={int(a.sum()):>4}  {j:5} n={int(b.sum()):>4}  both={int((a&b).sum()):>4}  "
                  f"jaccard={(a&b).sum()/max((a|b).sum(),1):.2f}")

print("\nC) late-window excesses of all five, side by side (spread of the family)")
base=fwd[late&v].mean()
es={k:fwd[S[k]&late&v].mean()-base for k in S}
for k,e in sorted(es.items(),key=lambda x:-x[1]):
    print(f"  {k:5} n={int((S[k]&late&v).sum()):>4}  exc {e*100:+.3f}pp")
print(f"  spread across the 5: {(max(es.values())-min(es.values()))*100:.3f}pp; "
      f"the 4 bullish span {(max(es[k] for k in ['s10','s20','bba','bbr'])-min(es[k] for k in ['s10','s20','bba','bbr']))*100:.3f}pp")

print("\nD) did BB-above decay BEFORE covid? 2015-2019 only (no covid contamination)")
p1519=(yrs>=2015)&(yrs<=2019); b2=fwd[p1519&v].mean()
for k,m in S.items():
    s=m&p1519&v
    tag="INCONCLUSIVE n<25" if s.sum()<25 else f"exc {(fwd[s].mean()-b2)*100:+.3f}pp"
    print(f"  {k:5} n={int(s.sum()):>4}  {tag}")

print("\nE) BB(10,2) below-lower: fires per year since 2015")
for Y in range(2015,2027):
    n=int((S['bbb']&(yrs==Y)&v).sum())
    print(f"  {Y}: {n:>2}", end="" if Y%6 else "\n")
print()
print(f"  total 2015-2026 = {int((S['bbb']&late&v).sum())} in {late.sum()} sessions "
      f"= 1 every {late.sum()/max(int((S['bbb']&late&v).sum()),1):.0f} sessions")
