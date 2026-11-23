"""Version-2 framework: mixed-effects (scale + effective-age) wear law, hierarchical
evidence-reliability averaging, condition/pooled prior averaging, grid fusion."""
import numpy as np, pickle, glob
from pipeline import *
from mc import register_shift, register, theta_prior, tfac
from fuse2 import fuse_bank, make_bank
from fuse import disc_params

def disc_shift(M,th,a,train,stride=3):
    res=[];ac=[];dts=[]
    for t in train:
        st=D[t].get('stride',stride); tt=D[t]['tte'][::st]; vv=D[t]['wte'][::st]; e=vv-M.hinv(a[t]+tt/th[t]); res.append(e)
        ac.append(np.corrcoef(e[1:],e[:-1])[0,1]); dts.append(np.median(np.diff(tt)))
    sd=float(np.sqrt(np.mean(np.concatenate(res)**2))); r1=float(np.clip(np.nanmean(ac),0.5,0.9999))
    return sd,float(-np.mean(dts)/np.log(r1))

SHIFT_UNITS='v'
PREDSCALE=True   # small-sample predictive scale of the offset distribution (Eq. 7); False only for the ablation
def shift_stats(M,e,ds,a=None):
    if SHIFT_UNITS=='v':
        ev=np.array([float(M.hinv(np.array([a[t]]))[0])-D[t]['wte'][0] for t in a]); ebar=float(ev.mean())
        floor=0.005; se=max(ev.std(ddof=1) if len(ev)>=2 else floor, floor)   # population s.d., not below inspection noise
        if len(ev)>=2 and PREDSCALE: se*=np.sqrt(1+1/len(ev))*tfac(len(ev)-1)             # predictive (Student-t matched) for a new tool
        return ebar, float(se), 'v'
    ev=np.array(list(e.values())); ebar=float(ev.mean())
    floor=0.05*float(M.h(np.array([TH[ds]]))[0])
    se=ev.std(ddof=1) if len(ev)>2 else (abs(np.diff(ev)).max()/np.sqrt(2) if len(ev)==2 else floor)
    return ebar, float(max(se,floor))

_law_cache={}
def law(train, ds, shift=True, nseg=None):
    key=(tuple(sorted(train)),shift,nseg)
    if key in _law_cache: return _law_cache[key]
    if nseg is None: nseg=select_J(train,ds,shift)
    if shift:
        M,th,a,e=register_shift(train,nseg=nseg); out=dict(M=M,th=th,a=a,e=e,shift=shift_stats(M,e,ds,a),disc=disc_shift(M,th,a,train),J=nseg)
    else:
        M,th,_=register(train,nseg=nseg); out=dict(M=M,th=th,shift=None,disc=disc_params(M,th,train),J=nseg)
    _law_cache[key]=out; return out

_J={}
def select_J(train, ds, shift, grid=(1,2,3,4,6)):
    """law resolution J by inner leave-one-tool-out error of the prior RUL (training tools only)"""
    key=(tuple(sorted(train)),shift)
    if key in _J: return _J[key]
    if len(train)<3: _J[key]=3; return 3
    uc=(ds!='PHM'); best=None
    for J in grid:
        err=[]
        for j in train:
            tr2=[x for x in train if x!=j]; L=law(tr2,ds,shift,nseg=J); f,s=theta_prior(tr2,L['th'],uc)
            M=L['M']; hs=float(M.h(np.array([TH[ds]]))[0]); v0=D[j]['wte'][0]
            a0=(float(M.h(np.array([v0+L['shift'][0]]))[0]) if L['shift'][2:]==('v',) else float(M.h(np.array([v0]))[0])+L['shift'][0]) if shift else float(M.h(np.array([v0]))[0])
            R=np.maximum(np.exp(f(D[j]['c']))*(hs-a0)-D[j]['tte'],0); m=emask(j)
            err.append(np.abs(R[m]-rul_true(j)[m]).mean()/tfail(j))
        e=np.mean(err)
        if best is None or e<best[0]: best=(e,J)
    _J[key]=best[1]; return best[1]

BANKDEF='verif'
PBMA=True
PBMA_MODE='bb'   # 'bb': Bayesian-bootstrap pseudo-BMA (default); 'plain': pseudo-BMA without bootstrap; 'uniform': equal weights (ablations)
BANKS={'none':dict(cp=(1.0,),cs=(1.0,)),
       'verif':dict(cp=(1.0,2.0),cs={'inc':(0.5,1.0,2.0),'rul':(1.0,2.0)}),
       'verif4':dict(cp=(1.0,2.0),cs={'inc':(0.5,1.0,2.0,4.0),'rul':(1.0,2.0,4.0)}),
       'verifo':dict(cp=(1.0,2.0),cs={'inc':(0.5,1.0,2.0,0.0),'rul':(1.0,2.0,0.0)}),
       'verifoi':dict(cp=(1.0,2.0),cs={'inc':(0.5,1.0,2.0,0.0),'rul':(1.0,2.0)}),
       'verifw':dict(cp=(1.0,2.0,4.0),cs={'inc':(0.5,1.0,2.0),'rul':(1.0,2.0)}),
       'full':dict(cp=(1.0,2.0),cs=(0.5,1.0,2.0)),
       'verife':dict(cp=(1.0,2.0),cs={'inc':(0.5,1.0,2.0),'rul':(1.0,2.0)},ce=(1.0,2.0,4.0)),
       'verifd':dict(cp=(1.0,2.0),cs={'inc':(0.5,1.0,2.0),'rul':(1.0,2.0)},cd=(1.0,2.0)),
       'verifp':dict(cp=(1.0,2.0),cs={'inc':(0.5,1.0,2.0),'rul':(1.0,2.0)},pm={'inc':(1.0,0.5),'rul':(1.0,0.5)})}

def run_v2(t, train, srcs, insp_dt, shift=True, bank=BANKDEF, cond='avg', disc_on=True, ar_on=True, keep_weights=False, **kw):
    ds=D[t]['ds']; uc=(ds!='PHM'); L=law(train,ds,shift)
    pri=[]
    if uc and cond in ('avg','cond'): pri.append(theta_prior(train,L['th'],True))
    if (not uc) or cond in ('avg','pool'): pri.append(theta_prior(train,L['th'],False))
    pri=[(f(D[t]['c']),s) for f,s in pri]
    # hyper-prior over prior structures: pseudo-BMA weights from leave-one-tool-out predictive densities
    wp=np.ones(len(pri))/len(pri)
    if len(pri)>1 and PBMA:
        structs=[True,False]; LP=np.zeros((len(train),2))
        for jj,j in enumerate(train):
            tr2=[x for x in train if x!=j]
            for q,uc_ in enumerate(structs):
                f,s_=theta_prior(tr2,L['th'],uc_); LP[jj,q]=-0.5*((np.log(L['th'][j])-f(D[j]['c']))/s_)**2-np.log(s_)
        # Bayesian-bootstrap pseudo-BMA (Yao et al. 2018): average of exp-normalised weights over Dirichlet resamples
        if PBMA_MODE=='bb':
            rng=np.random.default_rng(0); U=rng.dirichlet(np.ones(len(train)),size=1000)
            Z=len(train)*U@LP; Z-=Z.max(1,keepdims=True); Wb=np.exp(Z); Wb/=Wb.sum(1,keepdims=True); wp=Wb.mean(0)
        elif PBMA_MODE=='plain':
            z=LP.sum(0); z-=z.max(); wp=np.exp(z)/np.exp(z).sum()
    if not ar_on: srcs={k:dict(v,psi=0.05) for k,v in srcs.items()}
    disc=L['disc'] if disc_on else (1e-6,1.0)
    names=list(srcs.keys()); B=BANKS[bank]
    cs=B['cs'] if not isinstance(B['cs'],dict) else {n:B['cs'][n] for n in names}
    H=make_bank(pri,names,cp=B['cp'],cs=cs,pm=B.get('pm'),cd=B.get('cd',(1.0,)),ce=B.get('ce',(1.0,))) if names else [dict(pi=i,cp=a,c={},cd=d,ce=e) for i in range(len(pri)) for a in B['cp'] for d in B.get('cd',(1.0,)) for e in B.get('ce',(1.0,))]
    cnt=np.bincount([h['pi'] for h in H],minlength=len(pri)); pw=np.array([wp[h['pi']]/cnt[h['pi']] for h in H])
    o=fuse_bank(t,L['M'],pri,TH[ds],make_insp(t,insp_dt,SIG_I),SIG_I,disc,srcs,H,shift=L['shift'],keep_weights=keep_weights,prior_w=pw,**kw)
    o['wp']=wp
    o['bank']=H; o['J']=L['J']; return o

# ---------------- evidence sources ----------------
class Nets:
    def __init__(s,ds,qens=True):
        s.ds=ds; s.RO=pickle.load(open(f'out/nets_{ds}_outer.pkl','rb')); s.RI={}
        for fn in glob.glob(f'out/nets_{ds}_inner*.pkl')+glob.glob(f'out/nets_{ds}_qens*.pkl'): s.RI.update(pickle.load(open(fn,'rb')))
        for fn in glob.glob(f'out/nets_{ds}_qens.pkl'): s.RO.update(pickle.load(open(fn,'rb')))
        s.qens=qens
    def q(s,R,key):
        ks=[key+(i,) for i in range(5) if key+(i,) in R] if s.qens else [key+(0,)]
        return np.mean([R[k] for k in ks],0)          # quantile averaging (Vincentisation)
    def outer(s,k):
        mu,sd=ens(s.RO,('inc',k)); return inc_src(mu,sd), rul_src(s.q(s.RO,('rulq',k)))
    def inner(s,k,j):
        mu,sd=ens(s.RI,('inc',k,j)); return inc_src(mu,sd), rul_src(s.q(s.RI,('rulq',k,j)))
    def params(s,k,tools):
        pi=[];pr=[]
        for j in tools:
            I,Rr=s.inner(k,j); m=emask(j); w=D[j]['wte']-D[j]['wte'][0]
            pi.append((I['z']-w,I['s'])); pr.append(((Rr['z']-rul_true(j))[m],Rr['s'][m]))
        return mom(pi),mom(pr)
    def outer_srcs(s,k,tr):
        pI,pR=s.params(k,tr); I,R=s.outer(k)
        return {'inc':dict(I,**pI,kind='inc'),'rul':dict(R,**pR,kind='rul')},(pI,pR)
    def inner_srcs(s,k,j,tr):
        tr2=[x for x in tr if x!=j]; pI,pR=s.params(k,tr2); I,R=s.inner(k,j)
        return {'inc':dict(I,**pI,kind='inc'),'rul':dict(R,**pR,kind='rul')}
