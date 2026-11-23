import pickle, numpy as np, pipeline
from metrics import tool_metrics
from metrics2 import agg
from pipeline import rul_true, emask, tags, D, tfail, conformal_Q, cqr_score, cqr_apply
from scipy.stats import wilcoxon
DSS=['PHM','NUAA','MILL']
I=pickle.load(open('out/v4_inner_insp.pkl','rb'))
print('=== T0')
T0R={}
for tau in [10.0,30.0,100.0]:
    pipeline.TAU=tau
    for ds in DSS:
        O=pickle.load(open(f'out/rul2_{ds}.pkl','rb')); per=[]; dq=[]
        for k in tags(ds):
            sc=[cqr_score(rul_true(j)[emask(j)],I[ds][k][j][emask(j),0],I[ds][k][j][emask(j),4]) for j in I[ds][k]]
            Q=conformal_Q(sc); dq.append(Q-O[k]['insp']['OursCP']['Q'])
            o=O[k]['insp']['Ours']; lo,hi=cqr_apply(o['lo'],o['hi'],Q)
            per.append(tool_metrics(k,dict(R=o['R'],lo=lo,hi=hi)))
        a={kk:np.mean([p[kk] for p in per]) for kk in per[0]}; T0R[(tau,ds)]=a
        print(f'T0={tau:5.0f} {ds:5s} PICP {a["PICP"]:5.1f} NMPIW {a["NMPIW"]:6.1f} NIS {a["NIS"]:6.1f}   maxdQ(vs stored, only meaningful at 30) {np.max(np.abs(dq)):.3f}')
pipeline.TAU=30.0
pickle.dump(T0R,open('out/v4_t0.pkl','wb'))
print('=== pooled significance (Proposed - baseline), nMAE')
H=pickle.load(open('out/v4_hybrid.pkl','rb'))
rng=np.random.default_rng(0)
def boot(d,B=10000):
    d=np.asarray(d); idx=rng.integers(0,len(d),(B,len(d))); m=d[idx].mean(1); return d.mean(),np.percentile(m,2.5),np.percentile(m,97.5)
SIG={}
for scen in ['insp','none']:
    P={ds:agg(ds,scen,'Ours')[1] for ds in DSS}
    for b in ['DE','MCD','CQRraw','Wiener','MCprior','WienerNetK']:
        for grp,dss in [('all27',DSS),('multi24',['NUAA','MILL'])]:
            d=[];
            for ds in dss:
                if b.startswith('WienerNet'): B_=[tool_metrics(t,H[ds][t][scen][b]) for t in tags(ds)]
                else: B_=agg(ds,scen,b)[1]
                d+= [p['nMAE']-q['nMAE'] for p,q in zip(P[ds],B_)]
            mu,lo,hi=boot(d); pw=wilcoxon(d).pvalue; nwin=int(np.sum(np.array(d)<0))
            SIG[(scen,b,grp)]=(mu,lo,hi,pw,nwin,len(d))
            print(f'{scen:5s} {b:10s} {grp:8s} diff {mu:7.2f} [{lo:7.2f},{hi:7.2f}] p={pw:.2g} wins {nwin}/{len(d)}')
    # per-dataset for WienerNetK
    for ds in DSS:
        B_=[tool_metrics(t,H[ds][t][scen]['WienerNetK']) for t in tags(ds)]
        for met in ['nMAE','AL']:
            d=[p[met]-q[met] for p,q in zip(P[ds],B_)]; mu,lo,hi=boot(d)
            pw=wilcoxon(d).pvalue if np.any(np.array(d)!=0) else 1.0
            SIG[(scen,'WienerNetK',ds,met)]=(mu,lo,hi,pw)
            print(f'  {scen} {ds} WienerNetK {met} {mu:7.2f} [{lo:7.2f},{hi:7.2f}] p={pw:.3f}')
pickle.dump(SIG,open('out/v4_sig.pkl','wb'))
print('=== NASA increment MAE (used network, per member)')
RO=pickle.load(open('out/nets_MILL_outer.pkl','rb')); per=[]
for t in tags('MILL'):
    w=D[t]['wte']-D[t]['wte'][0]; mus=[RO[('inc',t,s)][0] for s in range(5) if ('inc',t,s) in RO]
    per.append(np.mean([np.abs(m-w).mean()*1e3 for m in mus]))
print('mean %.1f sd %.1f median %.1f'%(np.mean(per),np.std(per),np.median(per)), dict(zip(tags('MILL'),np.round(per))))
print('=== best-tool counts (insp) incl WienerNetK')
from collections import Counter
for incl in [False,True]:
    c=Counter()
    for ds in DSS:
        meths=['Pop','DE','MCD','CQRraw','LinEx','Wiener','MCprior','Ours']
        per={m:agg(ds,'insp',m)[1] for m in meths}
        if incl: per['WienerNetK']=[tool_metrics(t,H[ds][t]['insp']['WienerNetK']) for t in tags(ds)]
        for i,t in enumerate(tags(ds)):
            best=min(per,key=lambda m: per[m][i]['nMAE']); c[best]+=1
    print(incl,c)
