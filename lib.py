import numpy as np, torch, torch.nn as nn, pickle
import os
torch.set_num_threads(int(os.environ.get('NT','2')))
D=pickle.load(open('data/win.pkl','rb'))
TH={'PHM':0.15,'NUAA':0.16,'MILL':0.45,'MATWI':0.09}
# optional override of the wear limit (V* sensitivity study, v1.1); defaults reproduce the paper
for _ds in ('PHM','NUAA','MILL','MATWI'):
    if os.environ.get('TH_'+_ds): TH[_ds]=float(os.environ['TH_'+_ds])
if os.environ.get('TH_MILL'):
    for _d in D.values():
        if _d['ds']=='MILL':
            _t,_w,_V=_d['tte'],_d['wte'],TH['MILL']; _i=int(np.argmax(_w>=_V))
            _d['T']=float(_t[_i] if _i==0 else _t[_i-1]+(_V-_w[_i-1])/(_w[_i]-_w[_i-1]+1e-12)*(_t[_i]-_t[_i-1]))
NB=5  # baseline windows
def tags(ds): return [k for k in D if D[k]['ds']==ds]
def stats(X):
    m=X.mean(1); s=X.std(1)+1e-6; a=np.abs(X-X.mean(1,keepdims=True)).mean(1)+1e-6
    return np.concatenate([m,np.log(s),np.log(a)],1).astype(np.float32)
def prep_tool(t):
    d=D[t]; b=stats(d['Xte'][:d.get('nb',NB)]).mean(0)
    return dict(Str=stats(d['Xtr'])-b, Ste=stats(d['Xte'])-b)
P={t:prep_tool(t) for t in D}
AUX=None   # v1.2: None (paper networks) or ('none',None)/('insp',dt): add inspection features [z0, z_last-z0, t-t_last] to the condition input
def aux_feats(t,split,aux):
    from mc import make_insp
    d=D[t]; ins=make_insp(t,aux[1] if aux[0]=='insp' else None)
    ks=sorted(ins); tau=d['tte'][ks]; z=np.array([ins[k] for k in ks]); tt=d['t'+split]
    j=np.clip(np.searchsorted(tau,tt,side='right')-1,0,len(ks)-1)
    return np.stack([np.full(len(tt),z[0]),z[j]-z[0],tt-tau[j]],1)
class Norm:
    def __init__(s,tr):
        s.aux=AUX
        if s.aux:
            A=np.concatenate([aux_feats(t,'tr',s.aux) for t in tr]); s.amu=A.mean(0); s.asd=A.std(0)+1e-6
        X=np.concatenate([D[t]['Xtr'] for t in tr]); s.mu=X.mean((0,1)); s.sd=X.std((0,1))+1e-6
        S=np.concatenate([P[t]['Str'] for t in tr]); s.smu=S.mean(0); s.ssd=S.std(0)+1e-6
        C=np.stack([D[t]['c'] for t in tr]); s.cmu=C.mean(0); s.csd=C.std(0)+1e-6
    def __call__(s,t,split):
        X=(D[t]['X'+split]-s.mu)/s.sd; S=(P[t]['S'+split]-s.smu)/s.ssd
        c=(D[t]['c']-s.cmu)/s.csd; C=np.tile(c,(len(X),1))
        if getattr(s,'aux',None): C=np.concatenate([C,(aux_feats(t,split,s.aux)-s.amu)/s.asd],1)
        return [torch.tensor(np.nan_to_num(a),dtype=torch.float32) for a in (X,S,C)]
class Net(nn.Module):
    """1D-CNN trunk + baseline-referenced statistics + condition embedding"""
    def __init__(s,ch,ns,nout=2,p=0.1,cnn=True,stat=True,cond=True,trunk='cnn',nc=3):
        super().__init__(); s.cnn,s.stat,s.cond,s.trunk=cnn,stat,cond,trunk
        h=0
        if cnn and trunk=='lstm':
            s.f=nn.LSTM(ch,48,batch_first=True,bidirectional=True); h+=96
        elif cnn and trunk=='tf':
            s.inp=nn.Conv1d(ch,48,5,stride=5); s.pos=nn.Parameter(torch.zeros(1,20,48))
            s.f=nn.TransformerEncoder(nn.TransformerEncoderLayer(48,4,96,0.1,batch_first=True),2); h+=48
        elif cnn:
            s.f=nn.Sequential(nn.Conv1d(ch,32,7,padding=3),nn.GELU(),nn.MaxPool1d(2),
                              nn.Conv1d(32,48,5,padding=2),nn.GELU(),nn.MaxPool1d(2),
                              nn.Conv1d(48,64,3,padding=1),nn.GELU(),nn.AdaptiveAvgPool1d(1)); h+=64
        if stat: s.g=nn.Sequential(nn.Linear(ns,64),nn.GELU()); h+=64
        if cond: s.e=nn.Linear(nc,16); h+=16
        s.h=nn.Sequential(nn.Dropout(p),nn.Linear(h,64),nn.GELU(),nn.Dropout(p),nn.Linear(64,nout))
    def forward(s,X,S,C):
        z=[]
        if s.cnn and s.trunk=='lstm':
            o,_=s.f(X[:,::2]); z.append(o.mean(1))
        elif s.cnn and s.trunk=='tf':
            u=s.inp(X.transpose(1,2)).transpose(1,2)+s.pos; z.append(s.f(u).mean(1))
        elif s.cnn: z.append(s.f(X.transpose(1,2)).squeeze(-1))
        if s.stat: z.append(s.g(S))
        if s.cond: z.append(torch.tanh(s.e(C)))
        return s.h(torch.cat(z,1))
def gnll(o,y):
    mu,ls=o[:,0],o[:,1].clamp(-8,4)
    return (0.5*((y-mu)**2*torch.exp(-2*ls))+ls).mean()
def train(tr,target,seed=0,ep=25,loss='gnll',arch={},qs=None,lr=2e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    nm=Norm(tr); Xs,Ss,Cs,Ys=[],[],[],[]
    for t in tr:
        X,S,C=nm(t,'tr'); y=target(t)
        if isinstance(y,tuple): y,mk=y; mk=torch.tensor(mk); X,S,C=X[mk],S[mk],C[mk]; y=y[mk]
        Xs.append(X);Ss.append(S);Cs.append(C); Ys.append(torch.tensor(y,dtype=torch.float32))
    X,S,C,Y=map(torch.cat,(Xs,Ss,Cs,Ys))
    ysc=float(Y.std())+1e-9; ym=float(Y.mean())
    nout=2 if loss=='gnll' else (len(qs) if loss=='q' else 1)
    net=Net(X.shape[2],S.shape[1],nout,nc=C.shape[1],**arch); opt=torch.optim.AdamW(net.parameters(),lr,weight_decay=1e-4)
    n=len(X); sch=torch.optim.lr_scheduler.OneCycleLR(opt,lr,total_steps=ep*((n+127)//128))
    Yn=(Y-ym)/ysc
    for e in range(ep):
        net.train(); perm=torch.randperm(n)
        for i in range(0,n,128):
            b=perm[i:i+128]; o=net(X[b],S[b],C[b])
            if loss=='gnll': l=gnll(o,Yn[b]) if e>=3 else ((o[:,0]-Yn[b])**2).mean()
            elif loss=='q':
                qq=torch.tensor(qs); r=Yn[b][:,None]-o; l=torch.max(qq*r,(qq-1)*r).mean()
            else: l=((o[:,0]-Yn[b])**2).mean()
            opt.zero_grad(); l.backward(); opt.step(); sch.step()
    net.eval(); net.nm=nm; net.ym=ym; net.ysc=ysc; net.loss=loss
    return net
def predict(net,t,split='te',mc=0):
    X,S,C=net.nm(t,split)
    with torch.no_grad():
        if mc:
            net.train(); outs=torch.stack([net(X,S,C) for _ in range(mc)]); net.eval()
            o=outs.numpy()
        else: o=net(X,S,C).numpy()
    if net.loss=='gnll':
        mu=o[...,0]*net.ysc+net.ym; sd=np.exp(np.clip(o[...,1],-8,4))*net.ysc; return mu,sd
    return o*net.ysc+net.ym

def tfail(t,split='te'):
    d=D[t]
    if 'T' in d: return d['T']
    w=d['w'+split]; tt=d['t'+split]; th=TH[d['ds']]
    i=np.argmax(w>=th); return tt[i]
def inc_target(t): return D[t]['wtr']-D[t]['wte'][0]
def rul_target(t):
    tf=tfail(t); tt=D[t]['ttr']; return (tf-tt), tt<=tf
