import pickle,numpy as np
R=pickle.load(open('out/sigtests.pkl','rb'))
DN={'PHM':'PHM2010 ($n=3$)','NUAA':'NUAA ($n=9$)','MILL':'NASA milling ($n=15$)'}
BN=[('DE','DE'),('MCD','MCD'),('CQRraw','QR/CQR'),('Wiener','Wiener'),('MCprior','Law')]
def fp(p): return '$<$0.001' if p<0.001 else f'{p:.3f}'
L=[r'\begin{table}[!tbp]',r'\centering',
r'\caption{Paired comparison of the proposed method with each baseline over test tools (periodic inspections): mean difference Proposed $-$ baseline, 95\% bootstrap confidence interval over tools (10\,000 resamples) and two-sided Wilcoxon signed-rank $p$-value (with three tools the smallest attainable value is 0.25). Negative $\Delta n$MAE and positive $\Delta(\alpha$--$\lambda)$ favour the proposed method.}',
r'\label{tab:signif}',r'\footnotesize\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{llcccc}',r'\toprule',
r'Dataset & Baseline & $\Delta n$MAE (pp) [95\% CI] & $p$ & $\Delta(\alpha$--$\lambda)$ [95\% CI] & $p$\\',r'\midrule']
for i,ds in enumerate(['PHM','NUAA','MILL']):
    for j,(b,bn) in enumerate(BN):
        m=R[(ds,'insp',b,'nMAE')]; a=R[(ds,'insp',b,'AL')]
        L.append((DN[ds] if j==0 else '')+f' & {bn} & {m[0]:.2f} [{m[1]:.2f}, {m[2]:.2f}] & {fp(m[3])} & {a[0]:.2f} [{a[1]:.2f}, {a[2]:.2f}] & {fp(a[3])}\\\\')
    if i<2: L.append(r'\midrule')
L+=[r'\bottomrule',r'\end{tabular}',r'\end{table}']
import re
L=[re.sub(r'(?<![\w$])-(\d)',r'$-$\1',l) if ('&' in l and 'Dataset' not in l) else l for l in L]
open('tables/tab_signif.tex','w').write('\n'.join(L)+'\n')
