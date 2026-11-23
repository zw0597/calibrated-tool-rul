import numpy as np, pickle
from metrics2 import agg
from scipy.stats import wilcoxon
rng=np.random.default_rng(0)
def boot(d,B=10000):
    d=np.asarray(d); idx=rng.integers(0,len(d),(B,len(d))); m=d[idx].mean(1)
    return d.mean(),np.percentile(m,2.5),np.percentile(m,97.5)
R={}
for ds in ['PHM','NUAA','MILL']:
  for scen in ['insp','none']:
    _,P=agg(ds,scen,'Ours')
    for b in ['DE','QR' if False else 'CQRraw','MCD','Wiener','MCprior']:
        try: _,B_=agg(ds,scen,b)
        except Exception as e: continue
        for met in ['nMAE','AL']:
            d=[p[met]-q[met] for p,q in zip(P,B_)]
            mu,lo,hi=boot(d)
            try: pw=wilcoxon(d).pvalue if np.any(np.array(d)!=0) else 1.0
            except Exception: pw=np.nan
            R[(ds,scen,b,met)]=(mu,lo,hi,pw)
            print(f'{ds:5s}{scen:5s}{b:8s}{met:5s} diff {mu:7.2f} CI [{lo:7.2f},{hi:7.2f}] p={pw:.3f} n={len(d)}')
pickle.dump(R,open('out/sigtests.pkl','wb'))
