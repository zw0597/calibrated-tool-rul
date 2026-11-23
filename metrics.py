import pickle,numpy as np,sys
from pipeline import rul_true, emask, D, tfail, ALPHA
def tool_metrics(t,p):
    m=emask(t); y=rul_true(t)[m]; Tf=tfail(t); R=p['R'][m]; e=R-y
    out=dict(MAE=np.abs(e).mean(), RMSE=np.sqrt((e**2).mean()), nMAE=np.abs(e).mean()/Tf*100)
    tt=D[t]['tte'][m]; frac=tt/Tf
    late=frac>=0.5; out['MAElate']=np.abs(e[late]).mean() if late.any() else np.nan
    # alpha-lambda accuracy (alpha=0.3) at lambda = 0.25,0.5,0.75
    al=[]
    for lam in (0.25,0.5,0.75):
        i=np.argmin(np.abs(frac-lam)); al.append(float(abs(e[i])<=0.3*y[i]))
    out['AL']=np.mean(al)
    if 'lo' in p:
        lo=np.maximum(p['lo'][m],0); hi=p['hi'][m]
        cov=(y>=lo)&(y<=hi); out['PICP']=cov.mean()*100; out['NMPIW']=np.mean(hi-lo)/Tf*100
        IS=(hi-lo)+2/ALPHA*(lo-y)*(y<lo)+2/ALPHA*(y-hi)*(y>hi); out['NIS']=np.mean(IS)/Tf*100
    return out
def table(ds,scen,methods):
    O=pickle.load(open(f'out/rul_{ds}.pkl','rb'))
    rows={}
    for meth in methods:
        per=[]
        for t,res in O.items():
            p=res[scen][meth] if meth in res[scen] else res[meth]
            per.append(tool_metrics(t,p))
        keys=per[0].keys(); rows[meth]={k:(np.nanmean([q[k] for q in per]),np.nanstd([q[k] for q in per])) for k in keys}
        rows[meth]['per']=per
    return rows
if __name__=='__main__':
    ds=sys.argv[1]
    meths=['Pop','DE','MCD','CQRraw','CQR','LinEx','Wiener','MCprior','woI','woR','woAR','woDisc','anch','Ours','OursCP']
    if ds=='NUAA': meths.insert(-2,'woCond')
    for scen in ['none','insp']:
        print('==',ds,scen)
        rows=table(ds,scen,meths)
        for m,r in rows.items():
            s=f'{m:8s} MAE {r["MAE"][0]:7.1f} RMSE {r["RMSE"][0]:7.1f} nMAE {r["nMAE"][0]:5.1f} late {r["MAElate"][0]:6.1f} AL {r["AL"][0]:.2f}'
            if 'PICP' in r: s+=f' PICP {r["PICP"][0]:5.1f} NMPIW {r["NMPIW"][0]:5.1f} NIS {r["NIS"][0]:6.1f}'
            s+='  per:'+','.join(f'{q["MAE"]:.0f}' for q in r['per'])
            print(s)
