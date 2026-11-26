"""v1.4 (review): robustness of the paired comparisons of Table 7 (periodic inspections): Hodges-Lehmann estimate and
median of the per-tool differences, van Elteren p without tools that have fewer than 5 evaluation windows, and the mean
difference weighted by the number of evaluation windows.  python v7_tests_robust.py -> tables/tab_Srobust.tex"""
import numpy as np, re
from itertools import combinations_with_replacement
from metrics import tool_metrics
from lib import tags, tfail
from v2 import emask
from v7_tests import preds
from v7_tests2 import van_elteren
DSS = ['PHM', 'NUAA', 'MILL', 'MATWI']
ALT = [('DE+', 'DE+'), ('DE+CP', 'DE+CP'), ('QR+', 'QR+'), ('CQR+', 'CQR+'), ('WienerC', 'Wiener-C'), ('WienerARCP', 'Wiener-AR'), ('MCprior', 'Law')]
def hl(d):
    d = np.asarray(d); w = [(d[i] + d[j]) / 2 for i in range(len(d)) for j in range(i, len(d))]; return np.median(w)
def diffs(b, key):
    G = []; K = []
    for ds in DSS:
        g = []; kk = []
        for t in tags(ds):
            a_ = tool_metrics(t, preds(ds, t, 'insp', 'StackCP')); b_ = tool_metrics(t, preds(ds, t, 'insp', b))
            g.append((a_['MAElate'] - b_['MAElate']) / tfail(t) * 100 if key == 'late' else a_[key] - b_[key]); kk.append(int(emask(t).sum()))
        G.append(np.array(g)); K.append(np.array(kk))
    return G, K
L = [r'\begin{table}[!htbp]', r'\centering',
     r'\caption{Robustness of the comparisons of Table~\ref{tab:signif} (proposed method $-$ alternative, periodic inspections, 40 tools): Hodges--Lehmann estimate (HL) and median of the per-tool differences, mean weighted by the number of evaluation windows ($w$-mean), and unadjusted van Elteren $p$-value without the 6 tools with fewer than five evaluation windows ($p_{\ge5}$). Negative values favour the proposed method.}',
     r'\label{tab:Srobust}', r'\scriptsize\setlength{\tabcolsep}{2.5pt}', r'\begin{tabular}{lcccccccccccc}', r'\toprule',
     r' & \multicolumn{4}{c}{$\Delta$NIS (pp)} & \multicolumn{4}{c}{$\Delta n$MAE (pp)} & \multicolumn{4}{c}{$\Delta$ late-life error (pp)}\\',
     r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}', r'Alternative & HL & median & $w$-mean & $p_{\ge5}$ & HL & median & $w$-mean & $p_{\ge5}$ & HL & median & $w$-mean & $p_{\ge5}$\\', r'\midrule']
nshort = None
for b, nm in ALT:
    row = [nm]
    for key in ['NIS', 'nMAE', 'late']:
        G, K = diffs(b, key); d = np.concatenate(G); k = np.concatenate(K)
        nshort = int((k < 5).sum()); p5 = van_elteren([g[kk >= 5] for g, kk in zip(G, K)])
        f = lambda v: f'{v:.0f}' if abs(v) >= 100 else f'{v:.1f}'
        row += [f(hl(d)), f(np.median(d)), f(np.average(d, weights=k)), ('$<$0.001' if p5 < 0.001 else f'{p5:.3f}')]
    L.append(' & '.join(row) + r'\\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
L = [re.sub(r'(?<![\w$])-(\d)', r'$-$\1', l) if '&' in l and 'Alternative' not in l else l for l in L]
L[2] = L[2].replace('the 6 tools', f'the {nshort} tools')
open('tables/tab_Srobust.tex', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L[2:3] + L[10:-3]))
