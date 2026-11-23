"""Robustness of the persistence estimates (v1.3): moments of Proposition 2 recomputed (a) from RUL residuals only
(exact targets T-t), (b) from increment residuals at wear-measurement times only (PHM2010: windows where the
per-pass label changes; NASA milling: runs with a measured VB), (c) on windows thinned by a factor of five
(non-overlapping in time by >= 5 windows). Inner LOTO residuals, averaged over outer folds."""
import pickle, numpy as np, scipy.io as sio, os
from v2 import Nets, tags, D, emask, rul_true
from pipeline import mom
mill=sio.loadmat(os.environ.get('MILL_MAT','data/mill.mat'))['mill'][0]
measured={}
cases={}
for r in mill: cases.setdefault(int(r['case'][0,0]),[]).append(r)
for c,rs in cases.items():
    rs=sorted(rs,key=lambda r: float(r['time'][0,0])); measured[f'M{c:02d}']=np.array([r['VB'].size>0 and np.isfinite(float(r['VB'][0,0])) for r in rs])
def seqs(ds,src,mode):
    N=Nets(ds); out=[]
    for k in tags(ds):
        for j in tags(ds):
            if j==k: continue
            I,R=N.inner(k,j); m=emask(j)
            if src=='rul': e=(R['z']-rul_true(j))[m]; s=R['s'][m]; idx=np.arange(len(e))
            else:
                w=D[j]['wte']-D[j]['wte'][0]; e=I['z']-w; s=I['s']; idx=np.arange(len(e))
                if mode=='meas':
                    if ds=='PHM': idx=np.r_[0,np.where(np.abs(np.diff(D[j]['wte']))>1e-9)[0]+1]
                    elif ds=='MILL': mk=measured[j]; idx=np.where(mk[:len(e)])[0]
                    else: continue
            if mode=='thin': idx=idx[::5]
            if len(idx)>=3: out.append((e[idx],s[idx]))
    return out
res={}
for ds in ['PHM','NUAA','MILL']:
    for src in ['inc','rul']:
        for mode in ['all','meas','thin']:
            if mode=='meas' and (src=='rul' or ds=='NUAA'): continue
            S=seqs(ds,src,mode)
            if S: p=mom(S); res[(ds,src,mode)]=p; print(ds,src,mode,len(S),{k:round(v,3) for k,v in p.items()})
pickle.dump(res,open('out/v6_persist_check.pkl','wb'))
