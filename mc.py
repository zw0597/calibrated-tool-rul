"""Condition-scaled master-curve (time-scale family) prognostics with bias-aware Bayesian fusion."""
import numpy as np
from scipy.optimize import nnls
from lib import D, TH, tags, tfail
class Master:
    """h(v): normalised age at which wear v is reached; piecewise-linear monotone (NNLS increments)"""
    def __init__(s, knots): s.k=np.asarray(knots); s.c=None
    def basis(s,v):
        k=s.k; v=np.atleast_1d(v)
        B=np.clip((v[:,None]-k[None,:-1])/(k[1:]-k[:-1])[None,:],0,None)
        B[:,:-1]=np.minimum(B[:,:-1],1.0)   # last segment extrapolates linearly
        return B
    def h(s,v): return s.basis(v)@s.c
    def hinv(s,x):
        vv=np.linspace(s.k[0],s.k[-1]+0.5*(s.k[-1]-s.k[0]),4000); hh=s.h(vv)
        return np.interp(x,hh,vv)
def register(train, nseg=10, lam=1.0, iters=30, stride=3):
    vs=np.concatenate([D[t]['wte'] for t in train])
    knots=np.linspace(vs.min()-1e-3, vs.max()+1e-3, nseg+1)
    M=Master(knots)
    data={t:(D[t]['tte'][::D[t].get('stride',stride)], D[t]['wte'][::D[t].get('stride',stride)]) for t in train}
    th={t:1.0 for t in train}
    # init theta by duration ratio
    for t in train: th[t]=D[t]['tte'][-1]/np.mean([D[u]['tte'][-1] for u in train])
    nseg=len(knots)-1
    Dm=np.diff(np.eye(nseg),2,axis=0)
    for it in range(iters):
        A=[];b=[]
        for t,(tt,vv) in data.items():
            Bk=M.basis(vv)-M.basis(vv[:1]); A.append(th[t]*Bk/1000.0); b.append(tt/1000.0)
        A=np.vstack(A+[np.sqrt(lam)*Dm]); b=np.concatenate(b+[np.zeros(len(Dm))])
        M.c,_=nnls(A,b,maxiter=5000); M.c*=1.0
        for t,(tt,vv) in data.items():
            g=M.h(vv)-M.h(vv[:1]); th[t]=float(max((g@tt)/(g@g+1e-12),1e-3))
        gm=np.exp(np.mean(np.log(list(th.values()))))
        for t in th: th[t]/=gm
        M.c*=gm
    # residual (wear units): model path vs truth
    res=[]
    for t,(tt,vv) in data.items():
        vm=M.hinv(M.h(vv[:1])+tt/th[t]); res.append(vv-vm)
    sdelta=float(np.sqrt(np.mean(np.concatenate(res)**2)))
    return M, th, sdelta
from scipy.stats import t as _tdist
PRED=False
def tfac(dof):
    """ratio of Student-t to Gaussian 97.5% quantiles: matches the central 95% interval of the predictive t"""
    return float(_tdist.ppf(0.975,max(dof,1))/1.959964)
SUBSETS=[(),(0,),(1,),(2,),(0,1),(0,2),(1,2),(0,1,2)]
def _ridge(X,y,lam):
    C=np.c_[np.ones(len(y)),X]; Pn=lam*np.eye(C.shape[1]); Pn[0,0]=0
    A=C.T@C+Pn; w=np.linalg.solve(A,C.T@y); H=C@np.linalg.solve(A,C.T); return w,H
def theta_prior(train, th, use_cond, s_floor=0.15, lam_cond=0.1, return_info=False):
    """extended-Taylor prior ln(theta)=w0+sum_j w_j z_j(ln c); covariate subset chosen by exact ridge LOO on training tools"""
    lt=np.log([th[t] for t in train])
    info={}
    if use_cond and len(train)>=4:
        Xall=np.array([np.log(D[t]['c']) for t in train]); mx=Xall.mean(0); sx=Xall.std(0)+1e-12; Z=(Xall-mx)/sx
        best=None
        for S in SUBSETS:
            w,H=_ridge(Z[:,list(S)],lt,lam_cond); r=lt-H@lt; loo=np.mean((r/(1-np.diag(H)))**2)
            if best is None or loo<best[0]: best=(loo,S,w,H)
        loo,S,w,H=best; r=lt-H@lt; dof=len(lt)-np.trace(H)
        s=float(np.sqrt(max(loo,np.sum(r**2)/max(dof,1.0))))
        if PRED: s*=tfac(len(lt)-len(S)-1)
        f=lambda c,w=w,S=S,mx=mx,sx=sx: float(np.r_[1,((np.log(c)-mx)/sx)[list(S)]]@w)
        info=dict(subset=S,w=w,loo=loo)
    else:
        m=float(lt.mean()); s=float(lt.std(ddof=1)) if len(lt)>1 else 0.3
        if PRED and len(lt)>1: s*=np.sqrt(1+1/len(lt))*tfac(len(lt)-1)
        f=lambda c,m=m: m
    s=max(s,s_floor)
    return (f,s,info) if return_info else (f,s)
def posterior_grid(t, M, prior, sens, vstar, insp, sig_I, sdelta, kappa=1.0, phi=0.97, psi=0.8,
                   use_sensor=True, ar=True, ngrid=301, nv0=21):
    """exact grid posterior over (log theta, v0); returns per-step RUL quantiles and wear estimate"""
    d=D[t]; tt=d['tte']; K=len(tt)
    m0,s0=prior
    lth=np.linspace(m0-4*s0,m0+4*s0,ngrid)
    vI0=insp[0]; sv0=np.sqrt(sig_I**2)
    v0g=vI0+np.linspace(-3,3,nv0)*sv0
    LT,V0=np.meshgrid(lth,v0g,indexing='ij'); TH_=np.exp(LT)
    logp=-0.5*((LT-m0)/s0)**2-0.5*((V0-vI0)/sv0)**2
    h0=M.h(V0.ravel()).reshape(V0.shape); hstar=float(M.h(np.array([vstar]))[0])
    mu,sd=sens
    S=(kappa*sd)**2+sdelta**2
    ps=psi if ar else 0.0
    # AR(1) bias b ~ var ps*S + sdelta^2 (discrepancy fully persistent-ish) ; white (1-ps)*S
    mb=np.zeros_like(LT); Pb=ps*(kappa*sd[0])**2+sdelta**2
    out=dict(R=np.zeros(K),Rq=np.zeros((K,5)),v=np.zeros(K),theta=np.zeros(K))
    qs=np.array([0.05,0.25,0.5,0.75,0.95])
    for k in range(K):
        vm=M.hinv(h0.ravel()+tt[k]/TH_.ravel()).reshape(LT.shape)
        if k>0:
            Pb=phi**2*Pb+(ps*(kappa*sd[k])**2+sdelta**2)*(1-phi**2); mb=phi*mb
            if use_sensor:
                r2=(1-ps)*(kappa*sd[k])**2+1e-10
                nu=mu[k]-(vm-V0)-mb; Sx=Pb+r2
                logp+=-0.5*nu**2/Sx-0.5*np.log(Sx)
                G=Pb/Sx; mb=mb+G*nu; Pb=(1-G)*Pb
            if k in insp:
                logp+=-0.5*(insp[k]-vm)**2/(sig_I**2+sdelta**2)
        p=np.exp(logp-logp.max()); p/=p.sum()
        R=np.maximum(TH_*(hstar-h0)-tt[k],0.0)
        i=np.argsort(R.ravel()); c=np.cumsum(p.ravel()[i]); c/=c[-1]
        rq=np.interp(qs,c,R.ravel()[i])
        out['Rq'][k]=rq; out['R'][k]=rq[2]; out['v'][k]=np.sum(p*vm); out['theta'][k]=np.sum(p*TH_)
    return out
def make_insp(t,insp_dt,sig_I=0.005):
    d=D[t]; tt=d['tte']; v=d['wte']; rng=np.random.default_rng(int(sum(map(ord,t)))*7+1)
    insp={0:v[0]+sig_I*rng.standard_normal()}
    if insp_dt:
        nxt=insp_dt
        for k in range(1,len(tt)):
            if tt[k]>=nxt: insp[k]=v[k]+sig_I*rng.standard_normal(); nxt+=insp_dt
    return insp

def register_shift(train, nseg=3, lam=1.0, iters=40, stride=3, lam_e=0.0):
    """mixed-effects AFT registration: h(v_i(t)) = a_i + t/theta_i  (scale theta_i and effective-age shift a_i)
    returns master curve, theta, shift a, and effective-age deviation e_i = a_i - h(v_i(0))"""
    vs=np.concatenate([D[t]['wte'] for t in train])
    knots=np.linspace(vs.min()-1e-3, vs.max()+1e-3, nseg+1); M=Master(knots)
    data={t:(D[t]['tte'][::D[t].get('stride',stride)], D[t]['wte'][::D[t].get('stride',stride)]) for t in train}
    th={t:D[t]['tte'][-1]/np.mean([D[u]['tte'][-1] for u in train]) for t in train}
    a={t:0.0 for t in train}
    Dm=np.diff(np.eye(nseg),2,axis=0)
    M.c=None
    for it in range(iters):
        # master-curve step (NNLS): basis(v) c = a_i + t/theta_i  -> with a_i known
        A=[];b=[]
        for t,(tt,vv) in data.items():
            A.append(M.basis(vv)/1.0); b.append(a[t]+tt/th[t]/1000.0*1000.0/1000.0*1.0)
        A2=[];b2=[]
        for t,(tt,vv) in data.items():
            A2.append(th[t]*M.basis(vv)/1000.0); b2.append((th[t]*a[t]+tt)/1000.0)
        A2=np.vstack(A2+[np.sqrt(lam)*Dm]); b2=np.concatenate(b2+[np.zeros(len(Dm))])
        M.c,_=nnls(A2,b2,maxiter=5000)
        # per-tool step: t = theta*h - theta*a  (LS in (theta, theta*a))
        eb=np.mean([a[t]-float(M.h(D[t]['wte'][:1])[0]) for t in train]) if it>0 else 0.0
        for t,(tt,vv) in data.items():
            g=M.h(vv)
            if lam_e<=0 or it==0:
                X=np.c_[g,-np.ones_like(g)]
                sol,*_=np.linalg.lstsq(X,tt,rcond=None); th_t=max(sol[0],1e-3); th[t]=float(th_t); a[t]=float(sol[1]/th_t)
            else:
                c=float(M.h(vv[:1])[0])+eb
                for _ in range(5):
                    u=tt/th[t]-g; a[t]=float((-u.mean()+lam_e*c)/(1+lam_e))
                    x=g-a[t]; th[t]=float(max((tt@x)/(x@x+1e-12),1e-3))
        gm=np.exp(np.mean(np.log(list(th.values()))))
        for t in th: th[t]/=gm
        M.c*=gm
        for t in a: a[t]*=gm
    e={t:a[t]-float(M.h(D[t]['wte'][:1])[0]) for t in train}
    return M, th, a, e
