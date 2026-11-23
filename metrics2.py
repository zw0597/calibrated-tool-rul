import pickle,numpy as np,sys
from metrics import tool_metrics
def get(O,t,scen,m):
    r=O[t]; return r[scen][m] if m in r[scen] else r[m]
def agg(ds,scen,m,f='rul2'):
    O=pickle.load(open(f'out/{f}_{ds}.pkl','rb')); per=[tool_metrics(t,get(O,t,scen,m)) for t in O]
    return {k:np.nanmean([q[k] for q in per]) for k in per[0]}, per
if __name__=='__main__':
    ds=sys.argv[1]; O=pickle.load(open(f'out/rul2_{ds}.pkl','rb')); t0=list(O)[0]
    for scen in ['none','insp']:
        print('==',ds,scen)
        ms=['Pop','DE','MCD','CQRraw','CQR']+list(O[t0][scen].keys())
        for m in ms:
            a,per=agg(ds,scen,m)
            s=f'{m:9s} nMAE {a["nMAE"]:6.2f} MAE {a["MAE"]:6.1f} late {a["MAElate"]:6.1f} AL {a["AL"]:.2f}'
            if 'PICP' in a: s+=f' PICP {a["PICP"]:5.1f} W {a["NMPIW"]:6.1f} NIS {a["NIS"]:6.1f}'
            print(s,' per:',','.join(f'{q["MAE"]:.0f}' for q in per))

def fs_conformal(ds,scen='insp',B=400,alpha=0.1,seed=0):
    """finite-sample (subsampled) two-layer conformal of Proposition 5: one random window per calibration tool"""
    import math
    from pipeline import cqr_apply
    O=pickle.load(open(f'out/rul2_{ds}.pkl','rb')); rng=np.random.default_rng(seed); per=[]
    for t,r in O.items():
        sc=r[scen]['OursCP']['scores']; m=len(sc); idx=math.ceil((1-alpha)*(m+1))
        if idx>m: return None
        Ms=[]
        for b in range(B):
            draw=np.sort([s[rng.integers(len(s))] for s in sc]); Q=max(draw[idx-1],0.0)
            lo,hi=cqr_apply(r[scen]['Ours']['lo'],r[scen]['Ours']['hi'],Q)
            Ms.append(tool_metrics(t,dict(R=r[scen]['Ours']['R'],lo=lo,hi=hi)))
        per.append({k:np.mean([x[k] for x in Ms]) for k in Ms[0]})
    return {k:np.mean([p[k] for p in per]) for k in per[0]}, per
