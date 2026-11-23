"""V* sensitivity (v1.1): evaluate the proposed method and baselines with the networks retrained by vstar_nets.py.
Run from the same work directory and with the same TH_* variables as vstar_nets.py; writes out/vstar_eval.pkl."""
import os, pickle, numpy as np
from multiprocessing import Pool
def fold(a):
    ds,k=a
    from v2 import Nets, run_v2, tags, INSP, ens, emask, rul_true, conformal_Q, cqr_score, cqr_apply, TH
    from metrics import tool_metrics
    T=tags(ds); tr=[t for t in T if t!=k]; N=Nets(ds,qens=True)
    out={}
    mde,sde=ens(N.RO,('rul',k)); out['DE']=dict(R=np.maximum(mde,0),lo=np.maximum(mde-1.645*sde,0),hi=mde+1.645*sde)
    q=N.q(N.RO,('rulq',k)); out['QR']=dict(R=np.maximum(q[:,2],0),lo=q[:,0],hi=q[:,4])
    S,_=N.outer_srcs(k,tr)
    for scen,idt in [('none',None),('insp',INSP[ds])]:
        o=run_v2(k,tr,S,idt); out['Ours_'+scen]=dict(R=o['R'],lo=o['Rq'][:,0],hi=o['Rq'][:,4])
        o=run_v2(k,tr,{},idt); out['Law_'+scen]=dict(R=o['R'],lo=o['Rq'][:,0],hi=o['Rq'][:,4])
    sc=[]
    for j in tr:
        tr2=[x for x in tr if x!=j]; oj=run_v2(j,tr2,N.inner_srcs(k,j,tr),INSP[ds]); m=emask(j)
        sc.append(cqr_score(rul_true(j)[m],oj['Rq'][m,0],oj['Rq'][m,4]))
    Q=conformal_Q(sc); lo,hi=cqr_apply(out['Ours_insp']['lo'],out['Ours_insp']['hi'],Q)
    out['OursCP_insp']=dict(R=out['Ours_insp']['R'],lo=lo,hi=hi)
    return ds,k,{m:tool_metrics(k,p) for m,p in out.items()},TH[ds]
if __name__=='__main__':
    os.environ['NT']='1'
    from lib import tags
    args=[(ds,k) for ds in ['PHM','NUAA','MILL'] for k in tags(ds)]
    with Pool(16) as p: res=p.map(fold,args,chunksize=1)
    R={}
    for ds,k,m,th in res: R.setdefault(ds,{'TH':th,'tools':{}})['tools'][k]=m
    pickle.dump(R,open('out/vstar_eval.pkl','wb'))
    for ds,r in R.items():
        ms=list(next(iter(r['tools'].values())).keys())
        print(ds,'V*=',r['TH'],{m:(round(np.mean([t[m]['nMAE'] for t in r['tools'].values()]),2),round(np.mean([t[m].get('NIS',np.nan) for t in r['tools'].values()]),1)) for m in ms})
