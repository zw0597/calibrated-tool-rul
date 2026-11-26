"""Supplementary table: history-aware sequence baselines (nested-tuned). python make_tab_seq_v7.py -> tables/tab_Sseq.tex"""
import pickle, numpy as np
from metrics import tool_metrics
from lib import tags
from v7_tests import preds
Z = pickle.load(open('out/v7_seqtune.pkl', 'rb'))['OUT']
DN = [('PHM', 'PHM2010'), ('NUAA', 'NUAA'), ('MILL', 'NASA milling'), ('MATWI', 'MATWI')]
def m3(ds, get):
    per = [tool_metrics(t, get(t)) for t in tags(ds)]
    return [np.nanmean([q[c] for q in per]) for c in ['nMAE', 'PICP', 'NIS']]
f = lambda v: f'{v:.0f}' if v >= 1000 else f'{v:.1f}'
L = [r'\begin{table}[!htbp]', r'\centering',
     r'\caption{History-aware sequence baselines (periodic inspections), each with the two-layer conformal step: SEQ (causal GRU over window statistics, condition, inspection features and elapsed time) and SEQ-ev (the same, with the deep evidence of the framework as additional inputs, i.e.\ a learned fusion), with configurations selected per outer fold by the inner interval score from 16 candidates. $n$MAE, PICP and NIS in \%.}',
     r'\label{tab:Sseq}', r'\footnotesize\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{lccccccccc}', r'\toprule',
     r' & \multicolumn{3}{c}{SEQ} & \multicolumn{3}{c}{SEQ-ev} & \multicolumn{3}{c}{Proposed}\\', r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}',
     r'Dataset & $n$MAE & PICP & NIS & $n$MAE & PICP & NIS & $n$MAE & PICP & NIS\\', r'\midrule']
for ds, dn in DN:
    v = m3(ds, lambda t: Z[ds][t]['SEQoptCP']) + m3(ds, lambda t: Z[ds][t]['SEQevoptCP']) + m3(ds, lambda t: preds(ds, t, 'insp', 'StackCP'))
    L.append(dn + ' & ' + ' & '.join(f(x) for x in v) + r'\\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
open('tables/tab_Sseq.tex', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L[-8:-3]))
