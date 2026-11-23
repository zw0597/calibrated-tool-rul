"""Table 8 (v1.3): pooled paired tests over the 27 tools with Holm correction for three endpoints; periodic
inspections in the main text (tab_signif), mount-only in the supplement (tab_Ssignif_none)."""
import pickle, re, numpy as np
BN={'DE':'DE','DE+':'DE+','MCD':'MCD','CQRraw':'QR','CQR':'CQR','CQR+':'CQR+','Wiener':'Wiener','WienerC':'Wiener-C','WienerNetK':'Wiener-Net','PF':'PF','PFC':'PF-C','MCprior':'Law'}
def fp(p): return '$<$0.001' if p<0.001 else f'{p:.3f}'
def cell(r): return f'{r[1]:.1f} [{r[2]:.1f}, {r[3]:.1f}] & {fp(r[6])}' if r else '-- & --'
def build(scen,label,cap):
    T=pickle.load(open(f'out/v6_tests_{scen}.pkl','rb')); D={k:{r[0]:r for r in v} for k,v in T.items()}
    L=[r'\begin{table}[!tbp]',r'\centering',r'\caption{'+cap+'}',r'\label{'+label+'}',r'\scriptsize\setlength{\tabcolsep}{2.6pt}',r'\begin{tabular}{lcccccc}',r'\toprule',
       r' & \multicolumn{2}{c}{$\Delta$NIS (pp, primary)} & \multicolumn{2}{c}{$\Delta n$MAE (pp)} & \multicolumn{2}{c}{$\Delta$ late-life error (pp)}\\',
       r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}',r'Baseline & mean [95\% CI] & $p_{\rm Holm}$ & mean [95\% CI] & $p_{\rm Holm}$ & mean [95\% CI] & $p_{\rm Holm}$\\',r'\midrule']
    for b,nm in BN.items():
        L.append(nm+' & '+cell(D['NIS'].get(b))+' & '+cell(D['nMAE'].get(b))+' & '+cell(D['late'].get(b))+r'\\')
    L+=[r'\bottomrule',r'\end{tabular}',r'\end{table}']
    L=[re.sub(r'(?<![\w$])-(\d)',r'$-$\1',l) if '&' in l and 'Baseline' not in l else l for l in L]
    open('tables/tab_'+label.split(':')[1]+'.tex','w').write('\n'.join(L)+'\n')
cap=r'Paired comparison of the proposed method with each baseline, pooled over the 27 test tools (%s): mean difference Proposed $-$ baseline in percentage points, 95\%% bootstrap confidence interval over tools (10\,000 resamples) and two-sided Wilcoxon signed-rank $p$-value with Holm correction over the baselines of each endpoint. NIS: normalised interval score of the calibrated intervals of the proposed method against the intervals of the baseline (QR has no calibrated version; its NIS row is omitted); late-life error: MAE over the second half of life normalised by the tool life. Errors are normalised by the tool life, so that tools of different datasets are comparable. Negative values favour the proposed method.'
build('insp','tab:signif',cap%'periodic inspections')
build('none','tab:Ssignif_none',cap%'mounting record only')
print(open('tables/tab_signif.tex').read())
