"""Hierarchical-reliability grid filter: Bayesian model averaging over a bank of
error-model hypotheses (prior scale, evidence reliability), each an exact
Rao-Blackwellised grid filter; bank weights follow the marginal likelihood."""
import numpy as np, itertools
from lib import D
QS=np.array([0.05,0.25,0.5,0.75,0.95])
RUL_DISC=True    # v1.2: propagate the model discrepancy delta into the failure time (posterior predictive of the first passage of v_bar+delta)
NMC=1000         # Monte Carlo draws per window for RUL_DISC
DELTA0_COMP=False # v1.3 variant: initial discrepancy mean set to -epsilon (decouples the offset prior from sigma_delta)
CORR_IR=0.0       # v1.3 variant: correlation of the innovations of the increment and RUL biases
ERR_RI=False      # v1.3 variant: per-source random intercept + AR(1) + white error model (needs src['psi0'])

def rul_mc(rng, p, TH_, h0, mdel, Pdel, SDL, elld, M, tk, vstar, hstar, nmc=NMC, J=300):
    """RUL quantiles of the first passage of v_bar(t;theta,eps)+delta(t) over vstar.
    p: (B,G) posterior weights; mdel: (B,G) filtered mean of delta_k; Pdel: (B,) its variance; SDL: (B,) stationary s.d.;
    delta follows the OU process of the model (correlation time elld)."""
    B,G=p.shape; idx=rng.choice(B*G,size=nmc,p=(p/p.sum()).ravel()); b=idx//G; g=idx%G
    th=TH_[g]; a0=h0[g]; d=mdel[b,g]+np.sqrt(np.maximum(Pdel[b],0))*rng.standard_normal(nmc)
    Rn=np.maximum(th*(hstar-a0)-tk,0.0)
    H=max(2.0*Rn.max(),3.0*elld,10.0); tau=np.linspace(0,H,J+1); dt=tau[1]
    a=np.exp(-dt/elld); q=SDL[b]*np.sqrt(1-a*a)
    xi=rng.standard_normal((nmc,J)); D_=np.empty((nmc,J+1)); D_[:,0]=d
    for j in range(J): D_[:,j+1]=a*D_[:,j]+q*xi[:,j]
    V=M.hinv((a0[:,None]+(tk+tau[None,:])/th[:,None]).ravel()).reshape(nmc,J+1)+D_
    over=V>=vstar; hit=over.any(1); j1=np.argmax(over,1)
    R=np.full(nmc,H)
    m=hit&(j1==0); R[m]=0.0
    m=hit&(j1>0); jj=j1[m]; r=np.arange(nmc)[m]
    v0=V[r,jj-1]; v1=V[r,jj]; f=np.clip((vstar-v0)/np.maximum(v1-v0,1e-12),0,1)
    R[m]=tau[jj-1]+f*dt
    return np.quantile(R,QS)

def make_bank(priors, src_names, cp=(1.0,2.0), cs=(0.5,1.0,2.0), pm=None, cd=(1.0,), ce=(1.0,)):
    """priors: list of (m0,s0); returns list of hypotheses dict(pi,cp,c={name:mult})"""
    H=[]
    for pi in range(len(priors)):
        for a in cp:
            lists=[cs[nm] if isinstance(cs,dict) else cs for nm in src_names]
            plists=[(pm[nm] if pm and nm in pm else (1.0,)) for nm in src_names]
            for combo in itertools.product(*lists):
                for pc in itertools.product(*plists):
                    for d in cd:
                        for e in ce:
                            H.append(dict(pi=pi, cp=a, c=dict(zip(src_names, combo)), pm=dict(zip(src_names,pc)), cd=d, ce=e))
    return H

def fuse_bank(t, M, priors, vstar, insp, sig_I, disc, src, bank, ngrid=241, nv0=15,
              keep_weights=False, prior_w=None, shift=None, na=21, rgrid=None):
    d=D[t]; tt=d['tte']; K=len(tt); B=len(bank)
    sdl,elld=disc
    names=list(src.keys()); nS=len(names); RI=ERR_RI and all('psi0' in src[nm] for nm in names); n=1+nS*(2 if RI else 1)
    # common log-theta grid covering every hypothesis
    lo=min(priors[h['pi']][0]-4*priors[h['pi']][1]*h['cp'] for h in bank)
    hi=max(priors[h['pi']][0]+4*priors[h['pi']][1]*h['cp'] for h in bank)
    lth=np.linspace(lo,hi,ngrid); vI0=insp[0]; dl=lth[1]-lth[0]
    hstar=float(M.h(np.array([vstar]))[0])
    out_eg=None
    if shift is None:
        v0g=vI0+np.linspace(-3,3,nv0)*sig_I
        LT,V0=np.meshgrid(lth,v0g,indexing='ij'); TH_=np.exp(LT).ravel(); V0r=V0.ravel(); G=TH_.size
        h0=M.h(V0r); lpa=-0.5*((V0r-vI0)/sig_I)**2
    else:
        ebar,se=shift[:2]
        if len(shift)>2 and shift[2]=='v':     # effective-age deviation expressed in wear units
            cemax=max(h.get('ce',1.0) for h in bank); nae=na if cemax==1.0 else 2*na-1
            eg=ebar+np.linspace(-3.5,3.5,nae)*se*cemax; out_eg=eg
            LT,EG=np.meshgrid(lth,eg,indexing='ij'); TH_=np.exp(LT).ravel(); h0=M.h(np.maximum(vI0+EG.ravel(),M.k[0])); G=TH_.size
            V0r=np.full(G,vI0); EGr=EG.ravel(); lpa=None
        else:
            ac=float(M.h(np.array([vI0]))[0])+ebar
            ag=ac+np.linspace(-3.5,3.5,na)*se
            LT,AG=np.meshgrid(lth,ag,indexing='ij'); TH_=np.exp(LT).ravel(); h0=AG.ravel(); G=TH_.size
            V0r=np.full(G,vI0); lpa=-0.5*((h0-ac)/se)**2
    lw=np.log(np.full(B,1.0/B) if prior_w is None else np.asarray(prior_w))
    logp=np.zeros((B,G))
    for b,h in enumerate(bank):
        m0,s0=priors[h['pi']]; s=s0*h['cp']
        if lpa is None:
            sc=se*h.get('ce',1.0); la=-0.5*((EGr-ebar)/sc)**2-np.log(sc)
        else: la=lpa
        logp[b]=(-0.5*((LT.ravel()-m0)/s)**2-np.log(s)+la)+np.log(dl)-0.5*np.log(2*np.pi)
        logp[b]+=lw[b]
    # per-hypothesis effective source parameters
    kap=np.array([[src[nm]['kappa']*(h['c'][nm] if h['c'][nm]>0 else 1.0) for nm in names] for h in bank]).reshape(B,len(names))
    OFF=np.array([[h['c'][nm]==0 for nm in names] for h in bank]).reshape(B,len(names))
    TH0=np.array([np.exp(priors[h['pi']][0]) for h in bank])
    PSI=np.array([[src[nm]['psi']*h.get('pm',{}).get(nm,1.0) for nm in names] for h in bank]).reshape(B,len(names))
    CD=np.array([h.get('cd',1.0) for h in bank]); SDL=sdl*CD
    m=np.zeros((B,G,n)); P=np.zeros((B,n,n)); P[:,0,0]=SDL**2
    for i,nm in enumerate(names):
        s=src[nm]; P[:,i+1,i+1]=PSI[:,i]*(kap[:,i]*s['s'][0])**2
        if RI: P[:,1+nS+i,1+nS+i]=src[nm]['psi0']          # intercept in normalised units
    if CORR_IR and 'inc' in names and 'rul' in names:
        iI,iR=1+names.index('inc'),1+names.index('rul'); c0=CORR_IR*np.sqrt(P[:,iI,iI]*P[:,iR,iR]); P[:,iI,iR]=c0; P[:,iR,iI]=c0
    PSI0=np.array([[src[nm]['psi0'] if RI else 0.0 for nm in names]]*B).reshape(B,nS)
    out=dict(R=np.zeros(K),Rq=np.zeros((K,5)),v=np.zeros(K),theta=np.zeros(K),vq=np.zeros((K,2)),W=[],lz=np.zeros(K))
    if rgrid is not None: out['cdf']=np.zeros((K,len(rgrid)))
    cum=0.0
    rng=np.random.default_rng(20261122)
    for k in range(K):
        vm=M.hinv(h0+tt[k]/TH_)
        if k==0 and shift is not None and not DELTA0_COMP:
            nu=insp[0]-vm[None,:]-m[:,:,0]; S=P[:,0,0]+sig_I**2
            logp+=-0.5*nu**2/S[:,None]-0.5*np.log(2*np.pi*S)[:,None]
            Kg=P[:,:,0]/S[:,None]; PH=P[:,:,0].copy()
            m=m+nu[:,:,None]*Kg[:,None,:]; P=P-Kg[:,:,None]*PH[:,None,:]
        if k==0 and shift is not None and DELTA0_COMP:   # variant: delta_0 has mean -epsilon, the mount record carries no information on epsilon
            m[:,:,0]=(vI0-vm)[None,:]
        if k==0:   # condition on the mount record: evidence is counted from k>=1
            mx0=logp.max(); logp-=mx0+np.log(np.exp(logp-mx0).sum())
        if k>0:
            dt=tt[k]-tt[k-1]
            a=np.ones((B,n)); q=np.zeros((B,n))
            a[:,0]=np.exp(-dt/elld); q[:,0]=SDL**2*(1-a[0,0]**2)
            for i,nm in enumerate(names):
                s=src[nm]; a[:,i+1]=s['phi']; q[:,i+1]=PSI[:,i]*(kap[:,i]*s['s'][k])**2*(1-s['phi']**2)
            m=m*a[:,None,:]; P=a[:,:,None]*P*a[:,None,:]+np.einsum('bi,ij->bij',q,np.eye(n))
            if CORR_IR and 'inc' in names and 'rul' in names:
                iI,iR=1+names.index('inc'),1+names.index('rul'); cq=CORR_IR*np.sqrt(q[:,iI]*q[:,iR]); P[:,iI,iR]+=cq; P[:,iR,iI]+=cq
            obs=[]
            for i,nm in enumerate(names):
                s=src[nm]; Hr=np.zeros((B,n)); Hr[:,i+1]=1.0
                if RI: Hr[:,1+nS+i]=kap[:,i]*s['s'][k]
                if s['kind']=='inc':
                    f=np.broadcast_to(vm-V0r,(B,G)).copy(); Hr[:,0]=1.0
                    if OFF[:,i].any():
                        hb=float(M.h(np.array([max(vI0+(ebar if (shift is not None and len(shift)>2) else 0.0),M.k[0])]))[0])
                        for b in np.where(OFF[:,i])[0]:
                            f[b]=float(M.hinv(np.array([hb+tt[k]/TH0[b]]))[0])-vI0; Hr[b,0]=0.0
                else:
                    f=np.broadcast_to(TH_*(hstar-h0)-tt[k],(B,G)).copy()
                    if OFF[:,i].any():
                        hb=float(M.h(np.array([max(vI0+(ebar if (shift is not None and len(shift)>2) else 0.0),M.k[0])]))[0])
                        for b in np.where(OFF[:,i])[0]: f[b]=TH0[b]*(hstar-hb)-tt[k]
                obs.append((s['z'][k]-f, Hr, np.maximum(1-PSI[:,i]-PSI0[:,i],1e-3)*(kap[:,i]*s['s'][k])**2+1e-12))
            if k in insp:
                Hr=np.zeros((B,n)); Hr[:,0]=1.0; obs.append((np.broadcast_to(insp[k]-vm,(B,G)), Hr, np.full(B,sig_I**2)))
            for res,Hr,r in obs:
                nu=res-np.einsum('bgn,bn->bg',m,Hr)        # (B,G)
                PH=np.einsum('bij,bj->bi',P,Hr)            # (B,n)
                S=np.einsum('bi,bi->b',PH,Hr)+r            # (B,)
                logp+=-0.5*nu**2/S[:,None]-0.5*np.log(2*np.pi*S)[:,None]
                Kg=PH/S[:,None]
                m=m+nu[:,:,None]*Kg[:,None,:]
                P=P-Kg[:,:,None]*PH[:,None,:]
        mx=logp.max(); p=np.exp(logp-mx); Z=p.sum(); p/=Z
        out['lz'][k]=cum+mx+np.log(Z)
        R=np.maximum(TH_*(hstar-h0)-tt[k],0.0)
        pg=p.sum(0)                                        # marginal over hypotheses
        i=np.argsort(R); c=np.cumsum(pg[i]); c/=c[-1]
        rq=np.interp(QS,c,R[i])
        if RUL_DISC: rq=rul_mc(rng,p,TH_,h0,m[:,:,0],P[:,0,0],SDL,elld,M,tt[k],vstar,hstar)
        out['Rq'][k]=rq; out['R'][k]=rq[2]
        if rgrid is not None: out['cdf'][k]=np.interp(rgrid,R[i],c,left=0.0,right=1.0)
        vnow=vm[None,:]+m[:,:,0]
        out['v'][k]=np.sum(p*vm[None,:]); out['theta'][k]=np.sum(pg*TH_)
        j=np.argsort(vm); c=np.cumsum(pg[j]); c/=c[-1]; out['vq'][k]=np.interp([0.05,0.95],c,vm[j])
        if keep_weights:
            out['W'].append(p.sum(1)); out.setdefault('MT',[]).append(pg.reshape(LT.shape).sum(1)); out.setdefault('ME',[]).append(pg.reshape(LT.shape).sum(0))
            out['lth']=lth; out['eg']=out_eg if shift is not None else None
        # keep logp bounded
        logp-=mx; cum+=mx
    return out

def mix_structures(outs, rgrid, prior=None):
    """Bayesian model averaging over law structures using conditional evidence lz"""
    S=len(outs); K=len(outs[0]['R'])
    lw=np.log(np.full(S,1.0/S) if prior is None else np.asarray(prior))
    LZ=np.array([o['lz'] for o in outs])+lw[:,None]; W=np.exp(LZ-LZ.max(0)); W/=W.sum(0)
    cdf=np.einsum('sk,skr->kr',W,np.array([o['cdf'] for o in outs]))
    Rq=np.zeros((K,5))
    for k in range(K):
        c=np.maximum.accumulate(cdf[k]); Rq[k]=np.interp(QS,c+np.arange(len(c))*1e-12,rgrid)
    v=(W*np.array([o['v'] for o in outs])).sum(0); th=(W*np.array([o['theta'] for o in outs])).sum(0)
    return dict(R=Rq[:,2],Rq=Rq,v=v,theta=th,W=W,vq=np.einsum('sk,skj->kj',W,np.array([o['vq'] for o in outs])))
