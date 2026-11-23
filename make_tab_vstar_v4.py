"""Supplementary Table S2: sensitivity to the wear limit V* (networks retrained for every V*; see vstar_nets.py)."""
import pickle,numpy as np
R={S:pickle.load(open(f'out/vstar_eval_{S}.pkl','rb')) for S in ['A','ref','B']}
DN={'PHM':'PHM2010','NUAA':'NUAA','MILL':'NASA milling'}
def m(r,meth,k): return np.mean([t[meth][k] for t in r['tools'].values()])
L=[r'\begin{table}[!htbp]',r'\centering',
   r'\caption{Sensitivity to the wear limit $V^\ast$ (periodic inspections). For every $V^\ast$ the direct-RUL networks and the quantile ensembles (outer and inner) were retrained and the whole pipeline, including the error model and the conformal step, was re-run; the wear-increment networks do not depend on $V^\ast$. The rows marked $^\ast$ use the wear limits of the main text and reproduce Table~\\ref{tab:main_insp} up to retraining noise. Raw/cal.: before/after the conformal step.}',
   r'\label{tab:Svstar}',r'\footnotesize\setlength{\tabcolsep}{3.2pt}',r'\begin{tabular}{llccccccccc}',r'\toprule',
   r' & & \multicolumn{4}{c}{$n$MAE (\%)} & PICP (\%) & \multicolumn{4}{c}{NIS (\%)}\\',r'\cmidrule(lr){3-6}\cmidrule(lr){7-7}\cmidrule(lr){8-11}',
   r'Dataset & $V^\ast$ ($\mu$m) & Proposed & DE & QR & Law & Proposed (cal.) & Proposed (raw) & Proposed (cal.) & DE & Law\\',r'\midrule']
OUT={}
for ds in ['PHM','NUAA','MILL']:
    rows=sorted([(R[S][ds]['TH'],S) for S in R])
    for i,(th,S) in enumerate(rows):
        r=R[S][ds]; v=dict(P=m(r,'Ours_insp','nMAE'),DE=m(r,'DE','nMAE'),QR=m(r,'QR','nMAE'),Law=m(r,'Law_insp','nMAE'),
            cov=m(r,'OursCP_insp','PICP'),nr=m(r,'Ours_insp','NIS'),nc=m(r,'OursCP_insp','NIS'),nde=m(r,'DE','NIS'),nlaw=m(r,'Law_insp','NIS'),
            P0=m(r,'Ours_none','nMAE'),Law0=m(r,'Law_none','nMAE'),nqr=m(r,'QR','NIS'))
        OUT[(ds,th)]=v
        lab=f'{th*1e3:.0f}'+('$^\\ast$' if S=='ref' else '')
        L.append(' & '.join([DN[ds] if i==0 else '',lab]+[f'{v[k]:.1f}' for k in ['P','DE','QR','Law','cov','nr','nc','nde','nlaw']])+r'\\')
    if ds!='MILL': L.append(r'\midrule')
L+=[r'\bottomrule',r'\end{tabular}',r'\end{table}']
open('tables/tab_Svstar.tex','w').write('\n'.join(L)+'\n'); pickle.dump(OUT,open('out/v4_vstar_summary.pkl','wb'))
for k,v in OUT.items(): print(k,{a:round(b,1) for a,b in v.items()})
