import sys,os,pickle,numpy as np,time
from lib import *
ds=sys.argv[1]; part=sys.argv[2]
T=tags(ds); OUT=f'out/nets_{ds}_{part}'+os.environ.get('SUF','')+'.pkl'
FOLDS=os.environ.get('FOLDS')
R=pickle.load(open(OUT,'rb')) if os.path.exists(OUT) else {}
def save(): pickle.dump(R,open(OUT+'.tmp','wb')); os.replace(OUT+'.tmp',OUT)
QS=[0.05,0.25,0.5,0.75,0.95]
t0=time.time()
for k in T:
    if FOLDS and k not in FOLDS.split(','): continue
    tr=[t for t in T if t!=k]
    jobs=[]
    if part=='outer':
        for s in range(5): jobs.append((('inc',k,s),dict(target=inc_target,seed=s)))
        for s in range(5): jobs.append((('rul',k,s),dict(target=rul_target,seed=s)))
        jobs.append((('rulq',k,0),dict(target=rul_target,seed=0,loss='q',qs=QS)))
    if part=='inner':
        for j in tr:
            tr2=[t for t in tr if t!=j]
            for s in range(int(os.environ.get('NIS','3'))): jobs.append((('inc',k,j,s),dict(target=inc_target,seed=s,trs=tr2,pt=j)))
            jobs.append((('rulq',k,j,0),dict(target=rul_target,seed=0,loss='q',qs=QS,trs=tr2,pt=j)))
    if part=='innerde':
        for j in tr:
            tr2=[t for t in tr if t!=j]
            for s in range(int(os.environ.get('NIS','3'))): jobs.append((('rul',k,j,s),dict(target=rul_target,seed=s,trs=tr2,pt=j)))
    if part=='qens':
        for s in range(1,5): jobs.append((('rulq',k,s),dict(target=rul_target,seed=s,loss='q',qs=QS)))
    if part=='qensin':
        for j in tr:
            tr2=[t for t in tr if t!=j]
            for s in range(1,int(os.environ.get('QMAX','2'))): jobs.append((('rulq',k,j,s),dict(target=rul_target,seed=s,loss='q',qs=QS,trs=tr2,pt=j)))
    if part=='arch':
        for name,arch in [('lstm',dict(trunk='lstm')),('tf',dict(trunk='tf')),('nostat',dict(stat=False)),('nocnn',dict(cnn=False)),('nocond',dict(cond=False)),('plain',dict(stat=False,cond=False))]:
            for s in range(int(os.environ.get('NAS','3'))): jobs.append((('inc',name,k,s),dict(target=inc_target,seed=s,arch=arch)))
    for key,kw in jobs:
        if key in R: continue
        trs=kw.pop('trs',tr); pt=kw.pop('pt',k)
        net=train(trs,**kw)
        if kw.get('loss')=='q': R[key]=predict(net,pt)
        else:
            mu,sd=predict(net,pt); R[key]=(mu,sd)
            if part=='outer' and key[2]==0 and key[0]=='rul':
                R[('rulmc',k,0)]=predict(net,pt,mc=30)
        save(); print(key, round(time.time()-t0), flush=True)
print('DONE')
