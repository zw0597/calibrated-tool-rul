import pickle,numpy as np,sys,time
from pipeline import *
ds=sys.argv[1]; uc=(ds=='NUAA')
T=tags(ds)
RO=pickle.load(open(f'out/nets_{ds}_outer.pkl','rb')); import glob
RI={}
for fn in glob.glob(f'out/nets_{ds}_inner*.pkl'): RI.update(pickle.load(open(fn,'rb')))
OUT={}
def inner_srcs(k,j):
    mu,sd=ens(RI,('inc',k,j)); q=RI[('rulq',k,j,0)]; return inc_src(mu,sd), rul_src(q)
def resid_pairs(k,tools):
    pi=[];pr=[]
    for j in tools:
        I,Rr=inner_srcs(k,j); m=emask(j); w=D[j]['wte']-D[j]['wte'][0]
        pi.append((I['z']-w, I['s'])); pr.append(((Rr['z']-rul_true(j))[m], Rr['s'][m]))
    return mom(pi), mom(pr)
t0=time.time()
for k in T:
    tr=[t for t in T if t!=k]
    pI,pR=resid_pairs(k,tr)
    mu,sd=ens(RO,('inc',k)); q=RO[('rulq',k,0)]
    SI=dict(inc_src(mu,sd),**pI,kind='inc'); SR=dict(rul_src(q),**pR,kind='rul')
    res=dict(params=dict(inc=pI,rul=pR))
    # ---------- baselines without inspections dependence ----------
    tt=D[k]['tte']; Tpop=np.mean([tfail(j) for j in tr]); lives=np.array([tfail(j) for j in tr])
    res['Pop']=dict(R=np.maximum(Tpop-tt,0))
    mde,sde=ens(RO,('rul',k)); res['DE']=dict(R=np.maximum(mde,0),lo=np.maximum(mde-1.645*sde,0),hi=mde+1.645*sde)
    mcm,mcs=RO[('rulmc',k,0)]; mm=mcm.mean(0); ms=np.sqrt((mcs**2).mean(0)+mcm.var(0))
    res['MCD']=dict(R=np.maximum(mm,0),lo=np.maximum(mm-1.645*ms,0),hi=mm+1.645*ms)
    res['CQRraw']=dict(R=q[:,2],lo=q[:,0],hi=q[:,4])
    # conformal for CQR from inner rulq
    sc=[]
    for j in tr:
        qj=RI[('rulq',k,j,0)]; m=emask(j); sc.append(cqr_score(rul_true(j)[m],qj[m,0],qj[m,4]))
    Q=conformal_Q(sc); lo,hi=cqr_apply(np.maximum(q[:,0],0),np.maximum(q[:,4],0),Q)
    res['CQR']=dict(R=np.maximum(q[:,2],0),lo=lo,hi=hi,Q=Q)
    for scen,idt in [('none',None),('insp',INSP[ds])]:
        r={}
        # linear extrapolation of sensor wear (anchored at initial inspection)
        ins=make_insp(k,idt,SIG_I); v=ins[0]+mu; Rl=np.zeros(len(tt))
        for i in range(len(tt)):
            a=max(0,i-10); 
            if i-a>=2:
                sl=np.polyfit(tt[a:i+1],v[a:i+1],1)[0]
            else: sl=np.nan
            if not np.isfinite(sl) or sl<=1e-7: sl=1e-7
            Rl[i]=np.clip((TH[ds]-v[i])/sl,0,3*Tpop)
        r['LinEx']=dict(R=Rl)
        # Wiener process with Bayesian drift updated by inspections
        rates=[];sb=[]
        for j in tr:
            m=D[j]['tte']<=tfail(j); x=D[j]['tte'][m]; y=D[j]['wte'][m]
            b=(y[-1]-y[0])/(x[-1]-x[0]); rates.append(b); dy=np.diff(y); dx=np.diff(x); sb.append(np.var(dy-b*dx)/np.mean(dx))
        mu0=np.mean(rates); s_mu=max(np.std(rates,ddof=1) if len(rates)>1 else 0.3*mu0,0.15*mu0); sB2=np.mean(sb)
        rng=np.random.default_rng(0); Rw=np.zeros(len(tt)); Rwq=np.zeros((len(tt),2))
        keys=sorted(ins.keys())
        for i in range(len(tt)):
            kk=[x for x in keys if x<=i]; prec=1/s_mu**2; num=mu0/s_mu**2
            for a_,b_ in zip(kk[:-1],kk[1:]):
                dtt=tt[b_]-tt[a_]; dv=ins[b_]-ins[a_]; var=sB2*dtt+2*SIG_I**2
                prec+=dtt**2/var; num+=dtt*dv/var
            mpost=num/prec; spost=prec**-0.5
            mus=np.maximum(mpost+spost*rng.standard_normal(2000),1e-8)
            last=kk[-1]; el=tt[i]-tt[last]
            vnow=ins[last]+mus*el+np.sqrt(sB2*el+1e-12)*rng.standard_normal(2000)
            dlt=np.maximum(TH[ds]-vnow,0)
            # inverse Gaussian first-passage samples
            lam=dlt**2/sB2; mean=dlt/mus
            y_=rng.standard_normal(2000)**2; xx=mean+mean**2*y_/(2*lam)-mean/(2*lam)*np.sqrt(4*mean*lam*y_+mean**2*y_**2)
            zz=rng.random(2000); smp=np.where(zz<=mean/(mean+xx),xx,mean**2/np.maximum(xx,1e-12)); smp[dlt<=0]=0
            Rw[i]=np.median(smp); Rwq[i]=np.quantile(smp,[0.05,0.95])
        r['Wiener']=dict(R=Rw,lo=Rwq[:,0],hi=Rwq[:,1])
        # master-curve prior + inspections only
        o=run_fusion(k,tr,{},uc,idt); r['MCprior']=dict(R=o['R'],lo=o['Rq'][:,0],hi=o['Rq'][:,4])
        # proposed + ablations (outer)
        variants={'Ours':dict(srcs={'inc':SI,'rul':SR}),
                  'woI':dict(srcs={'rul':SR}),'woR':dict(srcs={'inc':SI}),
                  'woAR':dict(srcs={'inc':SI,'rul':SR},ar_on=False),
                  'woDisc':dict(srcs={'inc':SI,'rul':SR},disc_on=False),
                  'anch':dict(srcs={'inc':SI,'rul':SR},anchored=True)}
        if uc: variants['woCond']=dict(srcs={'inc':SI,'rul':SR},cond=False)
        for nm,cfg in variants.items():
            o=run_fusion(k,tr,cfg['srcs'],cfg.get('cond',uc),idt,disc_on=cfg.get('disc_on',True),ar_on=cfg.get('ar_on',True),anchored=cfg.get('anchored',False))
            r[nm]=dict(R=o['R'],lo=o['Rq'][:,0],hi=o['Rq'][:,4],Rq=o['Rq'],v=o['v'],vq=o['vq'],theta=o['theta'])
        # inner fusions on held-out training tools -> conformal scores for Ours
        sc=[]
        for j in tr:
            tr2=[x for x in tr if x!=j]
            pI2,pR2=resid_pairs(k,tr2) if len(tr2)>=1 else (pI,pR)
            I2,R2=inner_srcs(k,j)
            o=run_fusion(j,tr2,{'inc':dict(I2,**pI2,kind='inc'),'rul':dict(R2,**pR2,kind='rul')},uc,idt)
            m=emask(j); sc.append(cqr_score(rul_true(j)[m],o['Rq'][m,0],o['Rq'][m,4]))
        Q=conformal_Q(sc); lo,hi=cqr_apply(r['Ours']['lo'],r['Ours']['hi'],Q)
        r['OursCP']=dict(R=r['Ours']['R'],lo=lo,hi=hi,Q=Q)
        res[scen]=r
    OUT[k]=res
    print(k,'done',round(time.time()-t0),flush=True)
    pickle.dump(OUT,open(f'out/rul_{ds}.pkl','wb'))
