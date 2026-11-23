"""Table 8 of the revised manuscript: per-dataset paired tests (as before) plus the Wiener-Net baseline
and tests pooled over all 27 tools and over the 24 multi-condition tools."""
import pickle,numpy as np,re
from scipy.stats import wilcoxon
from metrics import tool_metrics
from metrics2 import agg
from pipeline import tags
HY=pickle.load(open('out/v4_hybrid.pkl','rb'))
from make_tables_v5 import agg as agg5
DN={'PHM':'PHM2010 ($n=3$)','NUAA':'NUAA ($n=9$)','MILL':'NASA milling ($n=15$)'}
BN=[('DE','DE'),('DE+','DE+'),('MCD','MCD'),('CQRraw','QR/CQR'),('Wiener','Wiener'),('WienerNetK','Wiener-Net'),('PF','PF'),('MCprior','Law')]
rng=np.random.default_rng(1)
def boot(d,B=10000):
    d=np.asarray(d); idx=rng.integers(0,len(d),(B,len(d))); m=d[idx].mean(1); return d.mean(),np.percentile(m,2.5),np.percentile(m,97.5)
def per(ds,scen,m): return agg5(ds,scen,m)[1]
def test(dss,b,met,scen='insp'):
    d=[]
    for ds in dss: d+=[p[met]-q[met] for p,q in zip(per(ds,scen,'Ours'),per(ds,scen,b))]
    mu,lo,hi=boot(d); pw=wilcoxon(d).pvalue if np.any(np.array(d)!=0) else 1.0
    return mu,lo,hi,pw
def fp(p): return '$<$0.001' if p<0.001 else f'{p:.3f}'
L=[r'\begin{table}[!tbp]',r'\centering',
r'\caption{Paired comparison of the proposed method with each baseline over test tools (periodic inspections): mean difference Proposed $-$ baseline, 95\% bootstrap confidence interval over tools (10\,000 resamples) and two-sided Wilcoxon signed-rank $p$-value (with three tools the smallest attainable value is 0.25). The last two blocks pool the tools of all datasets and of the two multi-condition datasets; differences in $n$MAE are comparable across datasets because the error is normalised by the tool life. Negative $\Delta n$MAE and positive $\Delta(\alpha$--$\lambda)$ favour the proposed method.}',
r'\label{tab:signif}',r'\footnotesize\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{llcccc}',r'\toprule',
r'Dataset & Baseline & $\Delta n$MAE (pp) [95\% CI] & $p$ & $\Delta(\alpha$--$\lambda)$ [95\% CI] & $p$\\',r'\midrule']
OUT={}
blocks=[(DN[ds],[ds]) for ds in ['PHM','NUAA','MILL']]+[('All tools ($n=27$)',['PHM','NUAA','MILL']),('Multi-condition ($n=24$)',['NUAA','MILL'])]
for i,(lab,dss) in enumerate(blocks):
    for j,(b,bn) in enumerate(BN):
        m=test(dss,b,'nMAE'); a=test(dss,b,'AL')
        OUT[(lab,b)]=(m,a)
        L.append((lab if j==0 else '')+f' & {bn} & {m[0]:.2f} [{m[1]:.2f}, {m[2]:.2f}] & {fp(m[3])} & {a[0]:.2f} [{a[1]:.2f}, {a[2]:.2f}] & {fp(a[3])}\\\\')
    if i<len(blocks)-1: L.append(r'\midrule')
L+=[r'\bottomrule',r'\end{tabular}',r'\end{table}']
L=[re.sub(r'(?<![\w$])-(\d)',r'$-$\1',l) if ('&' in l and 'Dataset' not in l) else l for l in L]
open('tables/tab_signif.tex','w').write('\n'.join(L)+'\n')
# mount-only pooled tests quoted in the text
for lab,dss in blocks[3:]:
    for b in ['DE','DE+','MCprior','WienerNetK','PF']:
        m=test(dss,b,'nMAE','none'); OUT[(lab,b,'none')]=m; print('mount-only',lab,b,np.round(m,3))
pickle.dump(OUT,open('out/v5_sigtab.pkl','wb'))
