"""Supplementary Table: observation-model variants (raw intervals)."""
import pickle, numpy as np
from metrics import tool_metrics
from pipeline import tags
V=pickle.load(open('out/v6_variants.pkl','rb')); DSS=['PHM','NUAA','MILL']
def agg(src,ds,scen):
    per=[tool_metrics(t,src(ds,t,scen)) for t in tags(ds)]; return {k:np.mean([p[k] for p in per]) for k in per[0]}
O={ds:pickle.load(open(f'out/rul2_{ds}.pkl','rb')) for ds in DSS}
rows=[('main',lambda ds,t,sc:O[ds][t][sc]['Ours'],'Main model'),('d0comp',None,r'$\delta_0$ with mean $-\epsilon$ (offset prior decoupled)'),
      ('corr',None,'correlated increment and RUL bias innovations'),('ri',None,'tool-level random intercept + AR(1) + white error'),('both',None,'random intercept and decoupled $\\delta_0$')]
L=[r'\begin{table}[!htbp]',r'\centering',r'\caption{Observation-model variants (raw posterior intervals before the conformal step). Correlated innovations: correlation estimated per fold as the median correlation of the normalised inner residuals of the two sources. Random intercept: $\rho_l=\psi^0_s+\psi_s\phi_s^l$ fitted by least squares to the residual autocorrelation at lags 1--10.}',r'\label{tab:Svariants}',
   r'\footnotesize\setlength{\tabcolsep}{2.5pt}',r'\begin{tabular}{llccccccccc}',r'\toprule',r' & & \multicolumn{3}{c}{PHM2010} & \multicolumn{3}{c}{NUAA} & \multicolumn{3}{c}{NASA milling}\\',
   r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}',r'Scenario & Variant & $n$MAE & PICP & NIS & $n$MAE & PICP & NIS & $n$MAE & PICP & NIS\\',r'\midrule']
for i,(scen,sl) in enumerate([('insp','periodic'),('none','mount-only')]):
    for j,(key,src,lab) in enumerate(rows):
        f=src if src else (lambda ds,t,sc,key=key: V[key][ds][t][sc])
        r=[sl if j==0 else '',lab]
        for ds in DSS:
            a=agg(f,ds,scen); r+=[f"{a['nMAE']:.1f}",f"{a['PICP']:.1f}",f"{a['NIS']:.1f}"]
        L.append(' & '.join(r)+r'\\')
    if i==0: L.append(r'\midrule')
L+=[r'\bottomrule',r'\end{tabular}',r'\end{table}']
open('tables/tab_Svariants.tex','w').write('\n'.join(L)+'\n'); print('\n'.join(L[10:]))
