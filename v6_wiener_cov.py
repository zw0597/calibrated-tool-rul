"""Wiener-process baseline with a condition-dependent drift prior (v1.3): ln(drift) ~ ridge regression on the
standardised log-conditions with the covariate subset chosen by exact LOO (as the extended-Taylor prior)."""
import pickle, numpy as np
from multiprocessing import Pool
from pipeline import D, tags, tfail, TH, INSP, SIG_I
from mc import make_insp, theta_prior
def job(a):
    ds,k=a; T=tags(ds); tr=[t for t in T if t!=k]; tt=D[k]['tte']; out={}
    rates={}; sb=[]
    for j in tr:
        m=D[j]['tte']<=tfail(j); x=D[j]['tte'][m]; y=D[j]['wte'][m]
        b=(y[-1]-y[0])/(x[-1]-x[0]); rates[j]=max(b,1e-9); dy=np.diff(y); dx=np.diff(x); sb.append(np.var(dy-b*dx)/np.mean(dx))
    f,s=theta_prior(tr,rates,ds!='PHM'); mu0=float(np.exp(f(D[k]['c']))); s_mu=mu0*s; sB2=np.mean(sb)
    Tpop=np.mean([tfail(j) for j in tr])
    for scen,idt in [('none',None),('insp',INSP[ds])]:
        ins=make_insp(k,idt,SIG_I); rng=np.random.default_rng(0); Rw=np.zeros(len(tt)); Rq=np.zeros((len(tt),2)); keys=sorted(ins)
        for i in range(len(tt)):
            kk=[x for x in keys if x<=i]; prec=1/s_mu**2; num=mu0/s_mu**2
            for a_,b_ in zip(kk[:-1],kk[1:]):
                dtt=tt[b_]-tt[a_]; dv=ins[b_]-ins[a_]; var=sB2*dtt+2*SIG_I**2; prec+=dtt**2/var; num+=dtt*dv/var
            mp=num/prec; sp=prec**-0.5; mus=np.maximum(mp+sp*rng.standard_normal(2000),1e-8)
            last=kk[-1]; el=tt[i]-tt[last]; vnow=ins[last]+mus*el+np.sqrt(sB2*el+1e-12)*rng.standard_normal(2000)
            dlt=np.maximum(TH[ds]-vnow,0); lam=dlt**2/sB2; mean=dlt/mus
            y_=rng.standard_normal(2000)**2; xx=mean+mean**2*y_/(2*lam)-mean/(2*lam)*np.sqrt(4*mean*lam*y_+mean**2*y_**2)
            zz=rng.random(2000); smp=np.where(zz<=mean/(mean+xx),xx,mean**2/np.maximum(xx,1e-12)); smp[dlt<=0]=0
            Rw[i]=np.median(smp); Rq[i]=np.quantile(smp,[0.05,0.95])
        out[scen]=dict(R=Rw,lo=Rq[:,0],hi=Rq[:,1])
    return ds,k,out
if __name__=='__main__':
    with Pool(16) as p: res=p.map(job,[(ds,k) for ds in ['PHM','NUAA','MILL'] for k in tags(ds)])
    O={}
    for ds,k,o in res: O.setdefault(ds,{})[k]=o
    pickle.dump(O,open('out/v6_wiener_cov.pkl','wb'))
    from metrics import tool_metrics
    for scen in ['insp','none']:
        for ds in ['PHM','NUAA','MILL']:
            per=[tool_metrics(t,O[ds][t][scen]) for t in tags(ds)]
            print('WienerC',scen,ds,{q:round(np.mean([p[q] for p in per]),1) for q in ['nMAE','PICP','NMPIW','NIS']})
