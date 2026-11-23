import pickle, numpy as np
from metrics import tool_metrics
from metrics2 import agg
DSS=['PHM','NUAA','MILL']; DN={'PHM':'PHM2010','NUAA':'NUAA','MILL':'NASA milling'}
J={ds:pickle.load(open(f'out/v6_jackplus_{ds}.pkl','rb')) for ds in DSS}
L=[r'\begin{table}[!htbp]',r'\centering',r'\caption{Tool-level jackknife+ (Eq.~\eqref{eq:jplus}) compared with the two-layer conformal step of the main text. Pooled: all windows of every calibration tool, weight 1 per tool (no guarantee); subsampled: one random window per calibration tool, averaged over 400 draws, approximate guarantee $1-2\alpha=80\%$, defined only for $m\ge9$. $<$80\%: tools covered on fewer than 80\% of their windows.}',
   r'\label{tab:Sjack}',r'\footnotesize\setlength{\tabcolsep}{3.5pt}',r'\begin{tabular}{llcccc}',r'\toprule',r'Scenario & Method & PHM2010 & NUAA & NASA milling & \\',r'\midrule']
L=L[:-2]+[r'\begin{tabular}{llccc}',r'\toprule',r'Scenario & Calibration & PHM2010 & NUAA & NASA milling\\',r'\midrule'] if False else L
rows=[]
out=[r'\begin{table}[!htbp]',r'\centering',L[2],r'\label{tab:Sjack}',r'\footnotesize\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{llccccccccc}',r'\toprule',
     r' & & \multicolumn{3}{c}{PHM2010} & \multicolumn{3}{c}{NUAA} & \multicolumn{3}{c}{NASA milling}\\',r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}',
     r'Scenario & Calibration & PICP & NIS & $<$80\% & PICP & NIS & $<$80\% & PICP & NIS & $<$80\%\\',r'\midrule']
for i,(scen,sl) in enumerate([('insp','periodic'),('none','mount-only')]):
    for j,(m,lab) in enumerate([('2L','two-layer (main text)'),('J+pooled','jackknife+, pooled'),('J+sub','jackknife+, subsampled')]):
        r=[sl if j==0 else '',lab]
        for ds in DSS:
            if m=='2L': per=agg(ds,scen,'OursCP')[1]
            else:
                per=[J[ds][k][scen][m] for k in J[ds] if m in J[ds][k][scen]]
            if not per: r+=['--']*3; continue
            r+=[f"{np.mean([p['PICP'] for p in per]):.1f}",f"{np.mean([p['NIS'] for p in per]):.1f}",str(sum(p['PICP']<80 for p in per))]
        out.append(' & '.join(r)+r'\\')
    if i==0: out.append(r'\midrule')
out+=[r'\bottomrule',r'\end{tabular}',r'\end{table}']
open('tables/tab_Sjack.tex','w').write('\n'.join(out)+'\n'); print('\n'.join(out[10:]))
