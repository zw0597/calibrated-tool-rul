"""v1.4 (review): replacement decisions per dataset. python make_tab_decds_v7.py -> tables/tab_Sdecds.tex"""
import pickle, numpy as np
R = pickle.load(open('out/v7_decision.pkl', 'rb')); DSS = ['PHM', 'NUAA', 'MILL', 'MATWI']
MS = [('DE+', 'DE+'), ('DE+CP', 'DE+CP'), ('CQR+', 'CQR+'), ('WienerC', 'Wiener-C'), ('MCprior', 'Law'), ('OursCP', 'Framework alone'), ('StackCP', 'Proposed')]
L = [r'\begin{table}[!htbp]', r'\centering', r'\caption{Replacement decisions of Table~\ref{tab:decision} per dataset: number of failures / wasted life (\% of the tool life, mean over the tools of the dataset) for lead times $L=5\%$ and $10\%$.}',
     r'\label{tab:Sdecds}', r'\footnotesize\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{lcccccccc}', r'\toprule',
     r' & \multicolumn{4}{c}{$L=5\%$} & \multicolumn{4}{c}{$L=10\%$}\\', r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}', r'Method & PHM2010 & NUAA & NASA & MATWI & PHM2010 & NUAA & NASA & MATWI\\', r'\midrule']
for m, nm in MS:
    row = [nm]
    for lead in (0.05, 0.10):
        per = R[(lead, m)]
        for ds in DSS:
            pp = [p for p in per if p['ds'] == ds]; row.append(f"{int(sum(p['fail'] for p in pp))} / {np.mean([p['waste'] for p in pp]):.0f}")
    L.append(' & '.join(row) + r'\\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
open('tables/tab_Sdecds.tex', 'w').write('\n'.join(L) + '\n')
