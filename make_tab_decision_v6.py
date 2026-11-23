"""Table: replacement-policy study (periodic inspections)."""
import pickle, numpy as np
R=pickle.load(open('out/v6_decision.pkl','rb'))
M=[('DE','DE'),('DE+','DE+'),('CQR','CQR'),('CQR+','CQR+'),('Wiener','Wiener'),('WienerC','Wiener-C'),('PF','PF'),('PFC','PF-C'),('MCprior','Law'),('Ours','Proposed-raw'),('OursCP','Proposed')]
leads=[0.05,0.10,0.20]
L=[r'\begin{table}[!tbp]',r'\centering',
   r'\caption{Replacement decisions (periodic inspections). The tool is replaced at the first evaluation window at which the lower bound of the 90\% interval falls below a lead time $L$, a fraction of the mean life of the training tools; a failure is a tool that reaches $V^\ast$ before replacement, and the wasted life is the remaining life at replacement, in \% of the tool life, averaged over the 27 tools. Failures are counted over all 27 tools.}',
   r'\label{tab:decision}',r'\footnotesize\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lcccccc}',r'\toprule',
   r' & \multicolumn{2}{c}{$L=5\%$} & \multicolumn{2}{c}{$L=10\%$} & \multicolumn{2}{c}{$L=20\%$}\\',r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
   r'Method & failures & wasted (\%) & failures & wasted (\%) & failures & wasted (\%)\\',r'\midrule']
for m,nm in M:
    row=[nm]
    for l in leads:
        per=R[(l,m)]; row+=[str(int(sum(p['fail'] for p in per))), f"{np.mean([p['waste'] for p in per]):.1f}"]
    L.append(' & '.join(row)+r'\\')
L+=[r'\bottomrule',r'\end{tabular}',r'\end{table}']
open('tables/tab_decision.tex','w').write('\n'.join(L)+'\n')
