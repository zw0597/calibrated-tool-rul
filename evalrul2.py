import pickle,numpy as np,sys,time,os
from v2 import *
ds=sys.argv[1]; T=tags(ds)
OLD=pickle.load(open(f'out/rul_{ds}.pkl','rb'))
N=Nets(ds,qens=True); N1=Nets(ds,qens=False)
OUT={}; t0=time.time()
VARS={'Ours':dict(),
      'woShift':dict(shift=False),
      'woBank':dict(bank='none'),
      'woI':dict(drop='inc'),'woR':dict(drop='rul'),
      'woAR':dict(ar_on=False),'woDisc':dict(disc_on=False),
      'singleQ':dict(single=True)}
if ds!='PHM': VARS.update({'condOnly':dict(cond='cond'),'poolOnly':dict(cond='pool')})
def go(k,tr,S,idt,cfg):
    S=dict(S); c=dict(cfg)
    if 'drop' in c: S.pop(c.pop('drop'))
    c.pop('single',None)
    return run_v2(k,tr,S,idt,**c)
FO=os.environ.get('FOLDS')
for k in T:
    if FO and k not in FO.split(','): continue
    tr=[t for t in T if t!=k]
    S,(pI,pR)=N.outer_srcs(k,tr); S1,_=N1.outer_srcs(k,tr)
    res=dict(params=dict(inc=pI,rul=pR))
    for b in ['Pop','DE','MCD']: res[b]=OLD[k][b]
    # QR / CQR on the 5-member quantile ensemble
    q=N.q(N.RO,('rulq',k)); res['CQRraw']=dict(R=np.maximum(q[:,2],0),lo=q[:,0],hi=q[:,4])
    sc=[]
    for j in tr:
        qj=N.q(N.RI,('rulq',k,j)); m=emask(j); sc.append(cqr_score(rul_true(j)[m],qj[m,0],qj[m,4]))
    Q=conformal_Q(sc); lo,hi=cqr_apply(np.maximum(q[:,0],0),np.maximum(q[:,4],0),Q)
    res['CQR']=dict(R=np.maximum(q[:,2],0),lo=lo,hi=hi,Q=Q)
    for scen,idt in [('none',None),('insp',INSP[ds])]:
        r={}
        for b in ['LinEx','Wiener']: r[b]=OLD[k][scen][b]
        o=run_v2(k,tr,{},idt); r['MCprior']=dict(R=o['R'],lo=o['Rq'][:,0],hi=o['Rq'][:,4])
        for nm,cfg in VARS.items():
            o=go(k,tr,S1 if cfg.get('single') else S,idt,cfg)
            r[nm]=dict(R=o['R'],lo=o['Rq'][:,0],hi=o['Rq'][:,4],Rq=o['Rq'],v=o['v'],vq=o['vq'],theta=o['theta'],J=o['J'])
            if nm=='Ours':
                ow=run_v2(k,tr,S,idt,keep_weights=True); r[nm]['W']=np.array(ow['W']); r[nm]['bank']=ow['bank']
        # conformal: inner fusions on held-out training tools
        sc=[]
        for j in tr:
            tr2=[x for x in tr if x!=j]
            oj=run_v2(j,tr2,N.inner_srcs(k,j,tr),idt)
            m=emask(j); sc.append(cqr_score(rul_true(j)[m],oj['Rq'][m,0],oj['Rq'][m,4]))
        Q=conformal_Q(sc); lo,hi=cqr_apply(r['Ours']['lo'],r['Ours']['hi'],Q)
        r['OursCP']=dict(R=r['Ours']['R'],lo=lo,hi=hi,Q=Q,scores=sc)
        res[scen]=r
    OUT[k]=res
    print(k,'done',round(time.time()-t0),flush=True)
    pickle.dump(OUT,open(f'out/rul2_{ds}'+os.environ.get('SUF','')+'.pkl','wb'))
