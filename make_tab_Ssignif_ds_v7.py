"""v1.4 Supplementary table: per-dataset paired tests of the stacked proposed method (periodic inspections), no multiplicity correction.
  python make_tab_Ssignif_ds_v7.py -> tables/tab_Ssignif_ds.tex"""
import numpy as np
from scipy.stats import wilcoxon
from metrics import tool_metrics
from lib import tags, tfail
from v7_tests import preds
DSS = [('PHM', 'PHM2010'), ('NUAA', 'NUAA'), ('MILL', 'NASA milling'), ('MATWI', 'MATWI')]
BASE = [('OursCP', 'Framework alone'), ('DE', 'DE'), ('DE+', 'DE+'), ('DE+CP', 'DE+CP'), ('QR+', 'QR+'), ('CQR', 'CQR'), ('CQR+', 'CQR+'), ('WienerC', 'Wiener-C'), ('PFC', 'PF-C'), ('MCprior', 'Law')]
def fp(p): return '--' if not np.isfinite(p) else ('$<$0.001' if p < 0.001 else f'{p:.3f}')
L = [r'\begin{table}[!tbp]', r'\centering',
     r'\caption{Paired comparison of the proposed (stacked, calibrated) method with selected alternatives within each dataset (periodic inspections): mean difference Proposed $-$ alternative in percentage points, two-sided Wilcoxon signed-rank $p$-value without multiplicity correction (with three tools the smallest attainable value is 0.25), and number of tools on which the proposed method is better. Negative differences favour the proposed method.}',
     r'\label{tab:Ssignif_ds}', r'\footnotesize\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{llccccccccc}', r'\toprule',
     r' & & \multicolumn{3}{c}{$\Delta$NIS (pp)} & \multicolumn{3}{c}{$\Delta n$MAE (pp)} & \multicolumn{3}{c}{$\Delta$ late-life error (pp)}\\',
     r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}', r'Dataset & Alternative & mean & $p$ & wins & mean & $p$ & wins & mean & $p$ & wins\\', r'\midrule']
for ds, dn in DSS:
    T = tags(ds)
    for k, (b, bn) in enumerate(BASE):
        cells = []
        for key in ['NIS', 'nMAE', 'late']:
            d = []
            for t in T:
                a_ = tool_metrics(t, preds(ds, t, 'insp', 'StackCP')); b_ = tool_metrics(t, preds(ds, t, 'insp', b))
                d.append((a_['MAElate'] - b_['MAElate']) / tfail(t) * 100 if key == 'late' else a_[key] - b_[key])
            d = np.array(d)
            try: p = wilcoxon(d).pvalue
            except ValueError: p = np.nan
            cells += [f'{d.mean():.1f}', fp(p), f'{int((d < 0).sum())}/{len(d)}']
        L.append((f'{dn} ($n={len(T)}$)' if k == 0 else '') + f' & {bn} & ' + ' & '.join(cells) + r'\\')
    if ds != 'MATWI': L.append(r'\midrule')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
import re
L = [re.sub(r'(?<![\w$/])-(\d)', r'$-$\1', l) if '&' in l and 'Dataset' not in l else l for l in L]
open('tables/tab_Ssignif_ds.tex', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
