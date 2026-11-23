import scipy.io as sio, numpy as np, pickle
d=sio.loadmat(__import__('os').environ.get('MILL_MAT','data/mill.mat'))['mill'][0]
CH=['smcAC','smcDC','vib_table','vib_spindle','AE_table','AE_spindle']
VSTAR=0.45; L=100
cases={}
for r in d:
    c=int(r['case'][0,0]); cases.setdefault(c,[]).append(r)
D=pickle.load(open('data/win.pkl','rb'))
D={k:v for k,v in D.items() if v['ds']!='MILL'}
for c,rs in sorted(cases.items()):
    if len(rs)<3: continue
    rs=sorted(rs,key=lambda r: float(r['time'][0,0]))
    t=np.array([float(r['time'][0,0]) for r in rs])*60.0
    vb=np.array([float(r['VB'][0,0]) if r['VB'].size else np.nan for r in rs])
    ok=np.isfinite(vb); w=np.interp(t,t[ok],vb[ok]); w=np.maximum.accumulate(w)
    if w.max()<VSTAR: print('skip',c); continue
    i=np.argmax(w>=VSTAR); T=t[i] if i==0 else t[i-1]+(VSTAR-w[i-1])/(w[i]-w[i-1]+1e-12)*(t[i]-t[i-1])
    Xtr=[];ttr=[];wtr=[];Xte=[]
    for k,r in enumerate(rs):
        S=np.stack([r[ch][:,0].astype(np.float64) for ch in CH],1)
        S=S[:9000//10*10].reshape(-1,10,6)
        env=np.sqrt((S**2).mean(1))          # 25-Hz RMS envelope
        env[:,1]=S[:,:,1].mean(1)            # DC current: block mean
        for s in range(100,800-L+1,50):
            Xtr.append(env[s:s+L]); ttr.append(t[k]); wtr.append(w[k])
        Xte.append(env[350:350+L])
    c0=rs[0]; cond=np.array([float(c0['feed'][0,0]),float(c0['DOC'][0,0]),float(c0['material'][0,0])],np.float32)
    tag=f'M{c:02d}'
    D[tag]=dict(ds='MILL',Xtr=np.array(Xtr,np.float32),ttr=np.array(ttr),wtr=np.array(wtr),Xte=np.array(Xte,np.float32),tte=t,wte=w,c=cond,fs=25.,T=float(T),nb=1,stride=1)
    print(tag,cond,'runs',len(rs),'T',round(T),'v0',round(w[0],3),'neval',int(((t<T)&(np.arange(len(t))>=1)).sum()))
pickle.dump(D,open('data/win.pkl','wb'))
