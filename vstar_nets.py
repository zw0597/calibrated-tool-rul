"""V* sensitivity (v1.1): retrain the V*-dependent networks (direct-RUL Gaussian ensemble, outer and inner
quantile ensembles) for the wear limits given by TH_PHM/TH_NUAA/TH_MILL; the wear-increment networks do not
depend on V* and are reused. Usage (from a work directory with data/ and out/):
  TH_PHM=0.12 TH_NUAA=0.155 TH_MILL=0.35 python vstar_nets.py
"""
import os, sys, pickle, glob, time, numpy as np
from multiprocessing import Pool
QS=[0.05,0.25,0.5,0.75,0.95]; QMAX={'PHM':5,'NUAA':5,'MILL':2}
def job(a):
    key,kw=a
    from lib import train, predict, rul_target, tags
    trs=kw.pop('trs'); pt=kw.pop('pt'); net=train(trs,target=rul_target,**kw)
    out={}
    if kw.get('loss')=='q': out[key]=predict(net,pt)
    else:
        out[key]=predict(net,pt)
        if key[0]=='rul' and key[2]==0: out[('rulmc',key[1],0)]=predict(net,pt,mc=30)
    return out
if __name__=='__main__':
    os.environ['NT']='1'
    from lib import tags
    jobs=[]
    for ds in ['PHM','NUAA','MILL']:
        T=tags(ds)
        for k in T:
            tr=[t for t in T if t!=k]
            for s in range(5): jobs.append((('rul',k,s),dict(seed=s,trs=tr,pt=k)))
            for s in range(5): jobs.append((('rulq',k,s),dict(seed=s,loss='q',qs=QS,trs=tr,pt=k)))
            for j in tr:
                tr2=[t for t in tr if t!=j]
                for s in range(QMAX[ds]): jobs.append((('rulq',k,j,s),dict(seed=s,loss='q',qs=QS,trs=tr2,pt=j)))
    jobs.sort(key=lambda a: {'PHM':0,'NUAA':1,'MILL':2}[[d for d in ['PHM','NUAA','MILL'] if a[0][1] in tags(d)][0]])
    t0=time.time(); R={}
    with Pool(int(os.environ.get('NPROC','16'))) as p:
        for i,o in enumerate(p.imap_unordered(job,jobs,chunksize=1)):
            R.update(o)
            if i%50==0: print(i,len(jobs),round(time.time()-t0),flush=True)
    pickle.dump(R,open('out/vstar_rulnets.pkl','wb'))
    # merge into the file layout expected by v2.Nets: inc keys from the original files, RUL keys new
    for ds in ['PHM','NUAA','MILL']:
        T=set(tags(ds))
        O={k:v for k,v in pickle.load(open(f'out/nets_{ds}_outer.pkl','rb')).items() if k[0]=='inc'}
        O.update({k:v for k,v in R.items() if k[1] in T and len(k)==3 and (k[0] in ('rul','rulmc') or (k[0]=='rulq' and k[2]==0))})
        pickle.dump(O,open(f'out/nets_{ds}_outer.pkl','wb'))
        pickle.dump({k:v for k,v in R.items() if k[1] in T and len(k)==3 and k[0]=='rulq' and k[2]>0},open(f'out/nets_{ds}_qens.pkl','wb'))
        pickle.dump({k:v for k,v in R.items() if k[1] in T and len(k)==4},open(f'out/nets_{ds}_qensin.pkl','wb'))
    print('done',round(time.time()-t0))
