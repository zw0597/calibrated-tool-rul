import pickle
R=pickle.load(open('out/v6_persist_check.pkl','rb'))
DN={'PHM':'PHM2010','NUAA':'NUAA','MILL':'NASA milling'}; MN={'all':'all evaluation windows','meas':'wear-measurement times only','thin':'every fifth window'}
L=[r'\begin{table}[!htbp]',r'\centering',r'\caption{Robustness of the persistence estimates of Proposition~2 (inner leave-one-tool-out residuals pooled over the outer folds). RUL residuals have exact targets $T-t$; wear-measurement times: windows at which the per-pass label changes (PHM2010) or runs with a measured flank wear (NASA milling); NUAA labels are continuous in time, so only thinning is possible. $n$: number of residual sequences with at least three points.}',
   r'\label{tab:Spersist}',r'\footnotesize\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lllrccc}',r'\toprule',r'Dataset & Evidence & Residuals & $n$ & $\kappa_s$ & $\phi_s$ & $\psi_s$\\',r'\midrule']
import collections
for ds in ['PHM','NUAA','MILL']:
    for src in ['inc','rul']:
        for mode in ['all','meas','thin']:
            if (ds,src,mode) not in R: continue
            p=R[(ds,src,mode)]
            L.append(f"{DN[ds]} & {'increment' if src=='inc' else 'RUL'} & {MN[mode]} & -- & {p['kappa']:.2f} & {p['phi']:.2f} & {p['psi']:.2f}\\\\")
    if ds!='MILL': L.append(r'\midrule')
L+=[r'\bottomrule',r'\end{tabular}',r'\end{table}']
s='\n'.join(L).replace(r'\begin{tabular}{lllrccc}',r'\begin{tabular}{lllccc}').replace(r'Dataset & Evidence & Residuals & $n$ & $\kappa_s$',r'Dataset & Evidence & Residuals & $\kappa_s$').replace(' & -- & ',' & ').replace(r' $n$: number of residual sequences with at least three points.','')
open('tables/tab_Spersist.tex','w').write(s+'\n'); print(s[-1500:])
