"""Scale-family master-curve prior + multi-source bias-aware Bayesian evidence fusion (grid RB filter)."""
import numpy as np
from mc import Master, register, theta_prior, make_insp
from lib import D, TH, tags, tfail
QS=np.array([0.05,0.25,0.5,0.75,0.95])
def disc_params(M, th, train, stride=3):
    """discrepancy delta(t)=v_true - v_model: sd and per-second AR decay"""
    res=[];ac=[];dts=[]
    for t in train:
        st=D[t].get('stride',stride); tt=D[t]['tte'][::st]; vv=D[t]['wte'][::st]
        vm=M.hinv(M.h(vv[:1])+tt/th[t]); e=vv-vm; res.append(e)
        ac.append(np.corrcoef(e[1:],e[:-1])[0,1]); dts.append(np.median(np.diff(tt)))
    sd=float(np.sqrt(np.mean(np.concatenate(res)**2)))
    r1=float(np.clip(np.nanmean(ac),0.5,0.9999)); ell=float(-np.mean(dts)/np.log(r1))
    return sd, ell
def fuse(t, M, prior, vstar, insp, sig_I, disc, src, ngrid=241, nv0=15, anchored=True, marg=False):
    """src: dict name -> dict(z=obs array, s=sd array, kappa, phi, psi, kind in {'inc','rul'})"""
    d=D[t]; tt=d['tte']; K=len(tt)
    m0,s0=prior; sdl,elld=disc
    lth=np.linspace(m0-4*s0,m0+4*s0,ngrid)
    vI0=insp[0]; v0g=vI0+np.linspace(-3,3,nv0)*sig_I
    LT,V0=np.meshgrid(lth,v0g,indexing='ij'); THm=np.exp(LT); G=LT.size
    logp=(-0.5*((LT-m0)/s0)**2-0.5*((V0-vI0)/sig_I)**2).ravel()
    h0=M.h(V0.ravel()); hstar=float(M.h(np.array([vstar]))[0]); TH_=THm.ravel(); V0r=V0.ravel()
    names=list(src.keys()); n=1+len(names)        # state: [delta, b_1..b_m]
    m=np.zeros((G,n)); P=np.zeros((n,n)); P[0,0]=sdl**2
    for i,nm in enumerate(names):
        s=src[nm]; P[i+1,i+1]=s['psi']*(s['kappa']*s['s'][0])**2
    out=dict(R=np.zeros(K),Rq=np.zeros((K,5)),v=np.zeros(K),theta=np.zeros(K),vq=np.zeros((K,2)),lth=lth,M=[])
    for k in range(K):
        vm=M.hinv(h0+tt[k]/TH_)
        if k>0:
            dt=tt[k]-tt[k-1]; a=np.ones(n); q=np.zeros(n)
            a[0]=np.exp(-dt/elld); q[0]=sdl**2*(1-a[0]**2)
            for i,nm in enumerate(names):
                s=src[nm]; a[i+1]=s['phi']; q[i+1]=s['psi']*(s['kappa']*s['s'][k])**2*(1-s['phi']**2)
            m=m*a[None,:]; P=(a[:,None]*P*a[None,:])+np.diag(q)
            obs=[]
            for i,nm in enumerate(names):
                s=src[nm]; Hr=np.zeros(n); Hr[i+1]=1.0
                if s['kind']=='inc': f=vm-V0r; Hr[0]=1.0
                else: f=TH_*(hstar-h0)-tt[k]
                obs.append((s['z'][k]-f, Hr, (1-s['psi'])*(s['kappa']*s['s'][k])**2+1e-12))
            if k in insp:
                Hr=np.zeros(n); Hr[0]=1.0; obs.append((insp[k]-vm, Hr, sig_I**2))
            for res,Hr,r in obs:
                nu=res-m@Hr; S=float(Hr@P@Hr)+r
                logp+=-0.5*nu**2/S
                Kg=(P@Hr)/S; m=m+nu[:,None]*Kg[None,:]; P=P-np.outer(Kg,Hr@P)
        p=np.exp(logp-logp.max()); p/=p.sum()
        vnow=vm+m[:,0] if anchored else vm
        if anchored:
            R=np.maximum(TH_*(hstar-M.h(np.minimum(vnow,vstar))),0.0)
        else:
            R=np.maximum(TH_*(hstar-h0)-tt[k],0.0)
        i=np.argsort(R); c=np.cumsum(p[i]); c/=c[-1]
        rq=np.interp(QS,c,R[i]); out['Rq'][k]=rq; out['R'][k]=rq[2]
        out['v'][k]=np.sum(p*vnow); out['theta'][k]=np.sum(p*TH_)
        if marg: out['M'].append(p.reshape(LT.shape).sum(1))
        j=np.argsort(vnow); c=np.cumsum(p[j]); c/=c[-1]; out['vq'][k]=np.interp([0.05,0.95],c,vnow[j])
    return out
