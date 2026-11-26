"""v1.4 (review): calibrated ablation. python make_tab_ablcp_v7.py -> tables/tab_Sablcp.tex"""
import pickle, numpy as np
from metrics import tool_metrics
from lib import tags
from v7_tests import preds
A = pickle.load(open('out/v7_ablate_cp.pkl', 'rb'))
V = [('OursCP', 'Framework alone (full)'), ('woShiftCP', r'w/o effective-age offset'), ('woBankCP', 'w/o hypothesis averaging'), ('woICP', 'w/o wear-increment evidence'),
     ('woRCP', 'w/o direct-RUL evidence'), ('woARCP', r'w/o persistent bias ($\psi_s\to0.05$)'), ('woDiscCP', r'w/o model discrepancy $\delta$'), ('singleQCP', 'single quantile network')]
L = [r'\begin{table}[!htbp]', r'\centering', r'\caption{Calibrated ablation (periodic inspections): each variant of the framework alone with its own two-layer conformal step, the scores computed from inner fusions of the same variant. PICP and NIS in \%; compare the raw intervals of Table~\ref{tab:ablation}.}',
     r'\label{tab:Sablcp}', r'\footnotesize\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{lcccccccc}', r'\toprule',
     r' & \multicolumn{2}{c}{PHM2010} & \multicolumn{2}{c}{NUAA} & \multicolumn{2}{c}{NASA milling} & \multicolumn{2}{c}{MATWI}\\', r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}', r'Variant & PICP & NIS & PICP & NIS & PICP & NIS & PICP & NIS\\', r'\midrule']
for v, nm in V:
    row = [nm]
    for ds in ['PHM', 'NUAA', 'MILL', 'MATWI']:
        per = [tool_metrics(t, preds(ds, t, 'insp', 'OursCP') if v == 'OursCP' else A[ds][t][v]) for t in tags(ds)]
        row += [f"{np.nanmean([q['PICP'] for q in per]):.1f}", f"{np.nanmean([q['NIS'] for q in per]):.1f}"]
    L.append(' & '.join(row) + r'\\')
    if v == 'OursCP': L.append(r'\midrule')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
open('tables/tab_Sablcp.tex', 'w').write('\n'.join(L) + '\n')
