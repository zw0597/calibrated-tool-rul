import pickle,numpy as np,sys,os,json
from lib import D, TH, tags, tfail
from mc import register, theta_prior, make_insp
from fuse import fuse, disc_params, QS
ALPHA=0.1; TAU=30.0
INSP={'PHM':200.0,'NUAA':300.0,'MILL':600.0}; SIG_I=0.005
def rul_true(t): return tfail(t)-D[t]['tte']
def emask(t):
    tt=D[t]['tte']; m=tt<tfail(t); m[:D[t].get('nb',5)]=False; return m
def ens(R,key):
    mus=[];sds=[]
    for s in range(5):
        kk=key+(s,)
        if kk in R: mus.append(R[kk][0]); sds.append(R[kk][1])
    mus=np.array(mus); sds=np.array(sds)
    return mus.mean(0), np.sqrt((sds**2).mean(0)+mus.var(0))
def mom(pairs):
    """method-of-moments AR(1)+white error model from normalised residual sequences"""
    z2=[];a1=[];a2=[]
    for e,s in pairs:
        if len(e)<3: continue   # lag-2 moment undefined (only with very short records, e.g. the V* study)
        z=e/np.maximum(s,1e-9); v=np.mean(z*z); z2.append(v)
        a1.append(np.mean(z[1:]*z[:-1])/v); a2.append(np.mean(z[2:]*z[:-2])/v)
    kappa=float(np.sqrt(np.mean(z2))); r1=float(np.mean(a1)); r2=float(np.mean(a2))
    phi=float(np.clip(r2/max(r1,1e-3),0.5,0.995)); psi=float(np.clip(r1/phi,0.05,0.98))
    return dict(kappa=kappa,phi=phi,psi=psi)
def inc_src(mu,sd): return dict(z=mu,s=np.maximum(sd,1e-4))
def rul_src(q): return dict(z=q[:,2],s=np.maximum((q[:,4]-q[:,0])/3.29,5.0))
NSEG_GRID=(1,2,3,4,6)
_sel_cache={}
def select_nseg(train,use_cond):
    """choose master-curve resolution by inner leave-one-tool-out prior RUL error (training tools only)"""
    key=(tuple(sorted(train)),use_cond)
    if key in _sel_cache: return _sel_cache[key]
    if len(train)<3: _sel_cache[key]=3; return 3
    best=None
    for ns in NSEG_GRID:
        err=[]
        for j in train:
            tr2=[x for x in train if x!=j]
            M,th,_=register(tr2,nseg=ns); f,s=theta_prior(tr2,th,use_cond)
            hs=M.h(np.array([TH[D[j]['ds']]]))[0]; v0=D[j]['wte'][0]
            R=np.maximum(np.exp(f(D[j]['c']))*(hs-M.h(np.array([v0]))[0])-D[j]['tte'],0)
            m=emask(j); err.append(np.abs(R[m]-rul_true(j)[m]).mean()/tfail(j))
        e=np.mean(err)
        if best is None or e<best[0]: best=(e,ns)
    _sel_cache[key]=best[1]; return best[1]
def run_fusion(t, train, srcs, use_cond, insp_dt, disc_on=True, ar_on=True, anchored=False, marg=False):
    ns=select_nseg(train,use_cond)
    M,th,_=register(train,nseg=ns); f,s=theta_prior(train,th,use_cond)
    disc=disc_params(M,th,train) if disc_on else (1e-6,1.0)
    if not ar_on:
        srcs={k:dict(v,psi=0.05) for k,v in srcs.items()}
    return fuse(t,M,(f(D[t]['c']),s),TH[D[t]['ds']],make_insp(t,insp_dt,SIG_I),SIG_I,disc,srcs,anchored=anchored,marg=marg)
def wq(x,w,q):
    i=np.argsort(x); c=np.cumsum(w[i]); c=c/c[-1]; return x[i][min(np.searchsorted(c,q),len(x)-1)]
def conformal_Q(scores):
    """two-layer CDF pooling: each calibration tool gets total weight 1"""
    x=np.concatenate([s for s in scores]); w=np.concatenate([np.full(len(s),1.0/len(s)) for s in scores])
    return max(float(wq(x,w,1-ALPHA)),0.0)   # one-sided: conformal step may only enlarge
def cqr_score(R,lo,hi):
    lo=np.maximum(lo,0); hi=np.maximum(hi,0)
    l=np.log1p(R/TAU); L=np.log1p(lo/TAU); U=np.log1p(hi/TAU); return np.maximum(L-l,l-U)
def cqr_apply(lo,hi,Q):
    return TAU*np.expm1(np.maximum(np.log1p(lo/TAU)-Q,0)), TAU*np.expm1(np.log1p(hi/TAU)+Q)
def mom_ri(pairs, L=10):
    """v1.3 variant: random intercept + AR(1) + white noise, rho_l = psi0 + psi1*phi**l, fitted by least squares to the
    averaged autocorrelation of the normalised residuals at lags 1..L (lags available in at least three sequences)."""
    from scipy.optimize import least_squares
    z2=[]; A=[]
    for e,s in pairs:
        z=e/np.maximum(s,1e-9); v=np.mean(z*z); z2.append(v)
        A.append([np.mean(z[l:]*z[:-l])/v if len(z)>l+1 else np.nan for l in range(1,L+1)])
    A=np.array(A); cnt=np.sum(np.isfinite(A),0); rho=np.nanmean(A,0); ok=cnt>=min(3,len(A)); lags=np.arange(1,L+1)[ok]; rho=rho[ok]
    def res(x):
        p0,p1,ph=x; return p0+p1*ph**lags-rho+10*max(0,p0+p1-0.99)
    best=None
    for x0 in [(0.3,0.5,0.9),(0.6,0.3,0.8),(0.1,0.8,0.97)]:
        r=least_squares(res,x0,bounds=([0,0,0.5],[0.98,0.98,0.995]))
        if best is None or r.cost<best.cost: best=r
    p0,p1,ph=best.x
    return dict(kappa=float(np.sqrt(np.mean(z2))),psi0=float(p0),psi=float(p1),phi=float(ph))
