"""v1.4 (review): causal inspections. python make_tab_causal_v7.py -> tables/tab_Scausal.tex"""
import pickle, numpy as np
from metrics import tool_metrics
from lib import tags
from v7_tests import preds
C = pickle.load(open('out/v7_causal.pkl', 'rb')); DSS = [('PHM', 'PHM2010'), ('NUAA', 'NUAA'), ('MILL', 'NASA milling'), ('MATWI', 'MATWI')]
L = [r'\begin{table}[!htbp]', r'\centering',
     r'\caption{Causal inspections (periodic scenario): every inspection reports the most recent actual wear measurement at or before its time (no interpolation towards later measurements, no smoothing; raw expert labels on MATWI, raw VB on NASA milling), plus the same instrument noise; the conformal step is re-estimated with causal inspections in the inner folds. Interpolated: inspections as in the main text. Framework alone (calibrated) and Law; $n$MAE, PICP and NIS in \%.}',
     r'\label{tab:Scausal}', r'\footnotesize\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{llcccccc}', r'\toprule',
     r' & & \multicolumn{3}{c}{Framework alone} & \multicolumn{3}{c}{Law}\\', r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}', r'Dataset & Inspections & $n$MAE & PICP & NIS & $n$MAE & PICP & NIS\\', r'\midrule']
for ds, dn in DSS:
    for i, src in enumerate(['interpolated', 'causal']):
        row = [dn if i == 0 else '', src]
        for m in ['OursCP', 'MCprior']:
            per = [tool_metrics(t, preds(ds, t, 'insp', m) if src == 'interpolated' else C[ds][t][m]) for t in tags(ds)]
            row += [f"{np.nanmean([q[c] for q in per]):.1f}" for c in ['nMAE', 'PICP', 'NIS']]
        L.append(' & '.join(row) + r'\\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
open('tables/tab_Scausal.tex', 'w').write('\n'.join(L) + '\n')
