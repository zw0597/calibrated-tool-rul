import pickle, numpy as np, pipeline
from metrics import tool_metrics
from metrics2 import agg
from pipeline import rul_true, emask, tags, D, tfail
DSS=['PHM','NUAA','MILL']
def summ(per): return {k:np.nanmean([q[k] for q in per]) for k in per[0]}
def show(lbl,a): print(f'{lbl:22s} nMAE {a["nMAE"]:6.2f} RMSE {a["RMSE"]:6.0f} late {a["MAElate"]:6.0f} AL {a["AL"]:.2f}'+(f' PICP {a["PICP"]:5.1f} W {a["NMPIW"]:6.1f} NIS {a["NIS"]:6.1f}' if 'PICP' in a else ''))
H=pickle.load(open('out/v4_hybrid.pkl','rb'))
print('=== hybrid')
for scen in ['none','insp']:
  for ds in DSS:
    for m in ['WienerNet','WienerNetK']:
        show(f'{ds} {scen} {m}',summ([tool_metrics(t,H[ds][t][scen][m]) for t in tags(ds)]))
    show(f'{ds} {scen} Ours(table)',agg(ds,scen,'Ours')[0])
A=pickle.load(open('out/v4_ablate.pkl','rb'))
print('=== ablate')
for scen in ['insp','none']:
  for ds in DSS:
    for n in A:
        if ds in A[n]: show(f'{ds} {scen} {n}',summ([tool_metrics(t,A[n][ds][scen][t]) for t in tags(ds)]))
S=pickle.load(open('out/v4_sens.pkl','rb'))
print('=== sens')
for ds in DSS:
    for n in S: show(f'{ds} {n}',summ([tool_metrics(t,S[n][ds]['insp'][t]) for t in tags(ds)]))
