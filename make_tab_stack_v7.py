"""v1.4 (review): stacking weights chosen in the inner folds. python make_tab_stack_v7.py -> tables/tab_Sstack.tex"""
import pickle, numpy as np
from lib import tags
ST = pickle.load(open('out/v7_stack.pkl', 'rb'))
DN = {'PHM': 'PHM2010', 'NUAA': 'NUAA', 'MILL': 'NASA milling', 'MATWI': 'MATWI'}
L = [r'\begin{table}[!htbp]', r'\centering',
     r'\caption{Weight $1-a$ of DE+ in the stacked predictor (Eq.~\eqref{eq:stack}), chosen in the inner folds of every outer fold by the smallest mean inner $n$MAE: mean, range and the values per test tool.}',
     r'\label{tab:Sstack}', r'\footnotesize\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{llccp{7.2cm}}', r'\toprule', r'Dataset & Scenario & mean & range & per tool\\', r'\midrule']
for ds in ['PHM', 'NUAA', 'MILL', 'MATWI']:
    for i, sc in enumerate(['insp', 'none']):
        w = [round(1 - ST[ds][t][sc]['a'], 1) for t in tags(ds)]
        per = ', '.join(f'{t} {x:.1f}' for t, x in zip(tags(ds), w))
        L.append((DN[ds] if i == 0 else '') + f" & {'periodic' if sc=='insp' else 'mount-only'} & {np.mean(w):.2f} & {min(w):.1f}--{max(w):.1f} & {per}\\\\")
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
open('tables/tab_Sstack.tex', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
