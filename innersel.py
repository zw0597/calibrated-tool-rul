import sys,pickle,numpy as np
from v2 import *
from metrics import tool_metrics
ds,bank=sys.argv[1],sys.argv[2]; T=tags(ds); N=Nets(ds,qens=True)
res={}
for k in T:
    tr=[t for t in T if t!=k]
    for j in tr:
        tr2=[x for x in tr if x!=j]; S=N.inner_srcs(k,j,tr)
        for scen,idt in [('none',None),('insp',INSP[ds])]:
            o=run_v2(j,tr2,S,idt,bank=bank)
            res[(k,j,scen)]=tool_metrics(j,dict(R=o['R'],lo=o['Rq'][:,0],hi=o['Rq'][:,4]))
pickle.dump(res,open(f'out/inner_{ds}_{bank}.pkl','wb'))
for scen in ['none','insp']:
    v=[r for (k,j,s),r in res.items() if s==scen]
    print(ds,bank,scen,'INNER',' '.join(f'{m} {np.nanmean([q[m] for q in v]):.2f}' for m in ['nMAE','PICP','NIS']),flush=True)
