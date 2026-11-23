"""Supplementary Table S1: sensitivity to fixed constants (periodic inspections)."""
import pickle,numpy as np
from metrics import tool_metrics
from pipeline import tags
S=pickle.load(open('out/v4_sens.pkl','rb')); T0=pickle.load(open('out/v4_t0.pkl','rb'))
DSS=['PHM','NUAA','MILL']
def a(n,ds): per=[tool_metrics(t,S[n][ds]['insp'][t]) for t in tags(ds)]; return {k:np.mean([p[k] for p in per]) for k in per[0]}
rows=[('ref','Paper setting (raw intervals)'),('sfl010',r'floor of $s_\theta$: 0.10 (paper 0.15)'),('sfl025',r'floor of $s_\theta$: 0.25'),
      ('rfl2',r'floor of $\hat s^R_k$: 2 s (paper 5 s)'),('rfl10',r'floor of $\hat s^R_k$: 10 s'),
      ('psi095',r'upper clip of $\psi_s$: 0.95 (paper 0.98)'),('psi0995',r'upper clip of $\psi_s$: 0.995'),
      ('sig25',r'inspection noise $\sigma_I$: 2.5 $\mu$m (paper 5 $\mu$m)'),('sig10',r'inspection noise $\sigma_I$: 10 $\mu$m')]
L=[r'\begin{table}[!htbp]',r'\centering',r'\caption{Sensitivity of the proposed method to its fixed constants (periodic inspections; one constant changed at a time, all other settings as in the main text). Upper block: raw posterior intervals. Lower block: calibrated intervals for different time constants $T_0$ of the log-time transform of the conformal score (the point estimate does not depend on $T_0$). For $\sigma_I$ both the simulated inspection noise and the value assumed by the filter are changed.}',r'\label{tab:Ssens}',r'\footnotesize\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{lccccccccc}',r'\toprule',
   r' & \multicolumn{3}{c}{PHM2010} & \multicolumn{3}{c}{NUAA} & \multicolumn{3}{c}{NASA milling}\\',r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}',
   r'Setting & $n$MAE & PICP & NIS & $n$MAE & PICP & NIS & $n$MAE & PICP & NIS\\',r'\midrule']
for n,lab in rows:
    L.append(' & '.join([lab]+[f'{a(n,ds)[k]:.1f}' for ds in DSS for k in ['nMAE','PICP','NIS']])+r'\\')
L.append(r'\midrule')
for tau in [10.0,30.0,100.0]:
    lab=f'calibrated, $T_0={tau:.0f}$ s'+(' (paper)' if tau==30 else '')
    L.append(' & '.join([lab]+[('--' if k=='nMAE' else f'{T0[(tau,ds)][k]:.1f}') for ds in DSS for k in ['nMAE','PICP','NIS']])+r'\\')
L+=[r'\bottomrule',r'\end{tabular}',r'\end{table}']
open('tables/tab_Ssens.tex','w').write('\n'.join(L)+'\n')
