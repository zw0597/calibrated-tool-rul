import sys,pickle,numpy as np,time
from v2 import *
from metrics import tool_metrics
ds,bank=sys.argv[1],sys.argv[2]; T=tags(ds); N=Nets(ds,qens=True)
out={}
for k in T:
    tr=[t for t in T if t!=k]; S,_=N.outer_srcs(k,tr)
    for scen,idt in [('none',None),('insp',INSP[ds])]:
        o=run_v2(k,tr,S,idt,bank=bank); out[(k,scen)]=dict(R=o['R'],lo=o['Rq'][:,0],hi=o['Rq'][:,4])
pickle.dump(out,open(f'out/q_{ds}_{bank}.pkl','wb'))
for scen in ['none','insp']:
    per=[tool_metrics(k,out[(k,scen)]) for k in T]
    a={m:np.nanmean([q[m] for q in per]) for m in per[0]}
    print(ds,bank,scen,' '.join(f'{m} {a[m]:.2f}' for m in ['nMAE','MAElate','AL','PICP','NIS']),flush=True)
