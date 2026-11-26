"""v1.4 (review): out-of-sample accuracy of coverage predictors for independent-error averaging (n>=2).
python make_tab_law_v7.py -> tables/tab_Slaw.tex"""
import pickle, numpy as np
O = pickle.load(open('out/v7_law_ri.pkl', 'rb'))
rows = {}
for (ds, u, src, n), v in O.items(): rows.setdefault((ds, src, n), []).append(v)
DN = [('PHM', 'PHM2010'), ('NUAA', 'NUAA'), ('MILL', 'NASA milling'), ('MATWI', 'MATWI'), ('SCIDATA', 'Independent tools'), ('CMAPSS', 'C-MAPSS FD001')]
M = [('AR', 'AR(1) (Cor.~1)'), ('RI', 'intercept + AR(1)'), ('ACF', 'empirical ACF'), ('IND', 'independence')]
S = {}
for ds, _ in DN:
    srcs = {b for a, b, c in rows if a == ds}
    for src in srcs:
        mx = max(len(v) for (a, b, c), v in rows.items() if a == ds and b == src)
        for (a, b, n), v in rows.items():
            if a != ds or b != src or n < 2 or len(v) < max(3, 0.5 * mx): continue
            e = np.mean([x['emp'] for x in v])
            for m, _ in M: S.setdefault(ds, {}).setdefault(m, []).append((np.mean([x[m] for x in v]) - e))
L = [r'\begin{table}[!htbp]', r'\centering',
     r'\caption{Out-of-sample prediction of the coverage of the nominal 90\% interval of averages of $n\ge2$ consecutive recalibrated residuals (Section~\ref{sec:res_law}): mean absolute difference between predicted and observed coverage over the dataset--source--$n$ combinations of Fig.~\ref{fig:law} (number in parentheses), and mean signed difference (predicted $-$ observed) in brackets. All predictors are fitted on the inner folds of each outer unit: AR(1)-plus-white (Corollary~1, moments at lags 1--2), random intercept plus AR(1) (least squares, lags 1--10), the empirical autocorrelation of the inner residuals plugged into $\nu_n$, and independent errors.}',
     r'\label{tab:Slaw}', r'\footnotesize\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{l' + 'c' * len(M) + '}', r'\toprule', 'Dataset & ' + ' & '.join(n for _, n in M) + r'\\', r'\midrule']
allv = {m: [] for m, _ in M}
for ds, dn in DN:
    if ds not in S: continue
    row = [f'{dn} ({len(S[ds]["AR"])})']
    for m, _ in M:
        v = np.array(S[ds][m]); allv[m] += list(v); row.append(f'{np.mean(np.abs(v)):.3f} [{np.mean(v):+.2f}]')
    L.append(' & '.join(row) + r'\\')
L.append(r'\midrule'); L.append(f'All ({len(allv["AR"])}) & ' + ' & '.join(f'{np.mean(np.abs(allv[m])):.3f} [{np.mean(allv[m]):+.2f}]' for m, _ in M) + r'\\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
L = [l.replace('[-', '[$-$').replace('[+', '[$+$') for l in L]
open('tables/tab_Slaw.tex', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L[8:-3]))
