import pickle,numpy as np,sys
from v2 import *
from metrics import tool_metrics
ds=sys.argv[1]; T=tags(ds); N=Nets(ds,True)
grid={'PHM':[None,400,200,100,50],'NUAA':[None,600,300,150,75],'MILL':[None,1200,600,300,150]}[ds]
import os
if os.environ.get('GRID'): grid=eval(os.environ['GRID'])
OUT={}
for idt in grid:
    per={'Ours':[],'OursCP':[],'MCprior':[]}
    for k in T:
        tr=[t for t in T if t!=k]; S,_=N.outer_srcs(k,tr)
        o=run_v2(k,tr,S,idt); p=dict(R=o['R'],lo=o['Rq'][:,0],hi=o['Rq'][:,4])
        sc=[]
        for j in tr:
            tr2=[x for x in tr if x!=j]; oj=run_v2(j,tr2,N.inner_srcs(k,j,tr),idt)
            m=emask(j); sc.append(cqr_score(rul_true(j)[m],oj['Rq'][m,0],oj['Rq'][m,4]))
        Q=conformal_Q(sc); lo,hi=cqr_apply(p['lo'],p['hi'],Q)
        per['Ours'].append(tool_metrics(k,p)); per['OursCP'].append(tool_metrics(k,dict(R=p['R'],lo=lo,hi=hi)))
        o=run_v2(k,tr,{},idt); per['MCprior'].append(tool_metrics(k,dict(R=o['R'],lo=o['Rq'][:,0],hi=o['Rq'][:,4])))
    OUT[idt]={m:{key:np.mean([x[key] for x in v]) for key in v[0]} for m,v in per.items()}
    OUT[idt]['per']={m:v for m,v in per.items()}
    print(ds,idt,{m:(round(v['nMAE'],2),round(v['PICP'],1),round(v['NIS'],1)) for m,v in OUT[idt].items() if m!='per'},flush=True)
pickle.dump(OUT,open(f'out/sens2_{ds}'+os.environ.get('SUF','')+'.pkl','wb'))
