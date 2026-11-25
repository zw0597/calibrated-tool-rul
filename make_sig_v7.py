"""v1.4 Table: pooled paired tests over the 40 tools (four tool datasets) with Holm correction; target = stacked proposed method."""
import pickle, re
BN = {'OursCP': 'Framework alone', 'DE': 'DE', 'DE+': 'DE+', 'MCD': 'MCD', 'CQR': 'CQR', 'CQR+': 'CQR+', 'Wiener': 'Wiener', 'WienerC': 'Wiener-C',
      'WienerNetK': 'Wiener-Net', 'PF': 'PF', 'PFC': 'PF-C', 'MCprior': 'Law'}
def fp(p): return '$<$0.001' if p < 0.001 else f'{p:.3f}'
def cell(r): return f'{r[1]:.1f} [{r[2]:.1f}, {r[3]:.1f}] & {fp(r[7])} & {r[5]}' if r else '-- & -- & --'
def build(scen, label, cap):
    T = pickle.load(open(f'out/v7_tests_StackCP_{scen}.pkl', 'rb')); D = {k: {r[0]: r for r in v} for k, v in T.items()}
    L = [r'\begin{table*}[!tbp]', r'\centering', r'\caption{' + cap + '}', r'\label{' + label + '}', r'\footnotesize\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{lccccccccc}', r'\toprule',
         r' & \multicolumn{3}{c}{$\Delta$NIS (pp, primary)} & \multicolumn{3}{c}{$\Delta n$MAE (pp)} & \multicolumn{3}{c}{$\Delta$ late-life error (pp)}\\',
         r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}', r'Baseline & mean [95\% CI] & $p_{\rm Holm}$ & wins & mean [95\% CI] & $p_{\rm Holm}$ & wins & mean [95\% CI] & $p_{\rm Holm}$ & wins\\', r'\midrule']
    for b, nm in BN.items():
        L.append(nm + ' & ' + cell(D['NIS'].get(b)) + ' & ' + cell(D['nMAE'].get(b)) + ' & ' + cell(D['late'].get(b)) + r'\\')
        if b == 'OursCP': L.append(r'\midrule')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    L = [re.sub(r'(?<![\w$])-(\d)', r'$-$\1', l) if '&' in l and 'Baseline' not in l else l for l in L]
    open('tables/tab_' + label.split(':')[1] + '.tex', 'w').write('\n'.join(L) + '\n')
cap = (r'Paired comparison of the proposed (stacked, calibrated) method with each alternative, pooled over the 40 test tools of the four tool datasets (%s): '
       r'mean difference Proposed $-$ alternative in percentage points, 95\%% bootstrap confidence interval over tools (10\,000 resamples), two-sided Wilcoxon signed-rank $p$-value with Holm correction '
       r'over the alternatives of each endpoint, and number of tools on which the proposed method is better. NIS: interval score of the calibrated proposed intervals against the intervals of the alternative; '
       r'late-life error: MAE over the second half of life normalised by the tool life. Negative values favour the proposed method. The MATWI tools (13 of 40) were evaluated once under a published protocol.')
build('insp', 'tab:signif', cap % 'periodic inspections')
build('none', 'tab:Ssignif_none', cap % 'mounting record only')
print(open('tables/tab_signif.tex').read()[:1500])
