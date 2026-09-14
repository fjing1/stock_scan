"""Inner-fold scale calibration: hold out the LAST 2 years of the TRAIN window, fit on the rest,
find the sigma multiplier kappa that makes predicted P(|move|>=thr) match observed on that inner
fold, then apply kappa to the real test year. Uses no test data at any point."""
import numpy as np, pandas as pd, _move_lib as L, move_prob as M, _move_validate as V, _move_data as D

def probs(y_tr, sig_tr, sig_te, thr, kappa=1.0):
    # kappa WIDENS the reference z distribution relative to the evaluation point. Scaling sigma
    # on both sides instead would cancel exactly -- the empirical table absorbs constant scale.
    z = np.sort(y_tr / sig_tr * kappa)
    a_dn, a_up = np.log(1-thr), np.log(1+thr)
    F = lambda v: np.searchsorted(z, v, side="right")/len(z)
    s = sig_te
    f_dn, f0, f_up = F(a_dn/s), F(0.0), F(a_up/s)
    p = np.column_stack([f_dn, f0-f_dn, f_up-f0, 1-f_up])
    p = np.clip(p,1e-4,None); return p/p.sum(axis=1,keepdims=True)

panel=D.load(); cache=V.build_cache(panel)
thr=0.02
print(f"\n{'grp':<8}{'h':>3}{'kappa mean':>12}{'pred raw':>10}{'pred cal':>10}{'obs':>8}"
      f"{'BSS2 raw':>10}{'BSS2 cal':>10}{'ECE raw':>9}{'ECE cal':>9}")
for (g,h),d in cache.items():
    X,y,tgt,yr,sid=d['X'],d['y'],d['tgt'],d['yr'],d['sid']
    years=sorted(set(yr)); fin=np.isfinite(tgt)
    Pr,Pc,A,PS,ks=[],[],[],[],[]
    act_all=V.bucket_idx(y,thr)
    for ty in years[5:]:
        tr,te=yr<ty,yr==ty
        if tr.sum()<2000 or te.sum()<50: continue
        itr,ival = yr<ty-2, (yr>=ty-2)&(yr<ty)
        if itr.sum()<1000 or ival.sum()<50: continue
        bi,_=M._ols(X[itr&fin],tgt[itr&fin])
        if bi is None: continue
        s_itr=np.exp(X[itr]@bi)*np.sqrt(h); s_ival=np.exp(X[ival]@bi)*np.sqrt(h)
        obs_i=np.isin(act_all[ival],[0,3]).mean()
        grid=np.linspace(0.90,1.40,26)
        errs=[abs((probs(y[itr],s_itr,s_ival,thr,k)[:,[0,3]].sum(1)).mean()-obs_i) for k in grid]
        k=float(grid[int(np.argmin(errs))]); ks.append(k)
        b,_=M._ols(X[tr&fin],tgt[tr&fin])
        s_tr=np.exp(X[tr]@b)*np.sqrt(h); s_te=np.exp(X[te]@b)*np.sqrt(h)
        Pr.append(probs(y[tr],s_tr,s_te,thr,1.0)); Pc.append(probs(y[tr],s_tr,s_te,thr,k))
        a_tr=act_all[tr]; pc=np.bincount(a_tr,minlength=4)/len(a_tr)
        ps=np.empty((te.sum(),4)); s_id=sid[te]
        for s in np.unique(s_id):
            m=sid[tr]==s
            cnt=np.bincount(a_tr[m],minlength=4).astype(float) if m.any() else np.zeros(4)
            ps[s_id==s]=(cnt+40*pc)/(cnt.sum()+40)
        PS.append(ps); A.append(act_all[te])
    pr,pcal,ps=np.vstack(Pr),np.vstack(Pc),np.vstack(PS); a=np.concatenate(A)
    big=np.isin(a,[0,3]).astype(float)
    f=lambda P: P[:,0]+P[:,3]
    bs=lambda q: ((q-big)**2).mean()
    print(f"{g:<8}{h:>3}{np.mean(ks):>12.3f}{f(pr).mean():>10.3f}{f(pcal).mean():>10.3f}{big.mean():>8.3f}"
          f"{1-bs(f(pr))/bs(f(ps)):>+10.4f}{1-bs(f(pcal))/bs(f(ps)):>+10.4f}"
          f"{L.ece(f(pr),big.astype(bool)):>9.4f}{L.ece(f(pcal),big.astype(bool)):>9.4f}")
