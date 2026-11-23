"""Inner networks (trained without the outer test tool k and the inner tool j) evaluated on the outer test tool k,
for the tool-level jackknife+ of the revised manuscript (v1.3). Same seeds and settings as the stored inner networks.
  python v6_jack_nets.py   -> out/v6_jack_nets.pkl   keys ('inc'|'rulq', k, j, s)
"""
import os, pickle, time
from multiprocessing import Pool
NIS={'PHM':3,'NUAA':2,'MILL':2}; QMAX={'PHM':5,'NUAA':5,'MILL':2}; QS=[0.05,0.25,0.5,0.75,0.95]
def job(a):
    kind,ds,k,j,s=a
    import lib
    T=lib.tags(ds); tr2=[t for t in T if t not in (k,j)]
    if kind=='inc': net=lib.train(tr2,lib.inc_target,seed=s); return a,lib.predict(net,k)
    net=lib.train(tr2,lib.rul_target,seed=s,loss='q',qs=QS); return a,lib.predict(net,k)
if __name__=='__main__':
    os.environ['NT']='1'; t0=time.time()
    from lib import tags
    args=[(kind,ds,k,j,s) for ds in ['PHM','NUAA','MILL'] for k in tags(ds) for j in tags(ds) if j!=k
          for kind,ns in [('inc',NIS[ds]),('rulq',QMAX[ds])] for s in range(ns)]
    R=pickle.load(open('out/v6_jack_nets.tmp','rb')) if os.path.exists('out/v6_jack_nets.tmp') else {}
    args=[a for a in args if (a[0],a[2],a[3],a[4]) not in R]
    args.sort(key=lambda a:{'MILL':0,'NUAA':1,'PHM':2}[a[1]]); print('remaining',len(args),flush=True)
    with Pool(int(os.environ.get('NPROC','17'))) as p:
        for i,(a,v) in enumerate(p.imap_unordered(job,args,chunksize=1)):
            R[(a[0],a[2],a[3],a[4])]=v
            if i%50==0: print(i,len(args),round(time.time()-t0),flush=True); pickle.dump(R,open('out/v6_jack_nets.tmp','wb'))
    pickle.dump(R,open('out/v6_jack_nets.pkl','wb')); print('done',round(time.time()-t0))
