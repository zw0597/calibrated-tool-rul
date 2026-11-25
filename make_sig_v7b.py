"""v1.4 (review) Table: stratified paired tests (van Elteren, Holm) over the 40 tools; target = stacked proposed method.
  python make_sig_v7b.py -> tables/tab_signif.tex, tables/tab_Ssignif_none.tex"""
import pickle, re, numpy as np
BN = [('OursCP', r'Framework alone$^\ast$'), ('DE', 'DE'), ('DE+', 'DE+'), ('DE+CP', 'DE+CP'), ('MCD', 'MCD'), ('CQRraw', 'QR'), ('QR+', 'QR+'), ('CQR', 'CQR'), ('CQR+', 'CQR+'),
      ('Wiener', 'Wiener'), ('WienerC', 'Wiener-C'), ('WienerNetK', 'Wiener-Net'), ('WienerARCP', 'Wiener-AR'), ('PF', 'PF'), ('PFC', 'PF-C'), ('PFNet', 'PF-Net'), ('MCprior', 'Law')]
def fp(p): return '$<$0.001' if p < 0.001 else f'{p:.3f}'
def num(v): return f'{v:.0f}' if abs(v) >= 100 else f'{v:.1f}'
def cell(r):
    if r is None: return '-- & -- & --'
    p = r[4] if r[0] == 'OursCP' else r[8]
    return f'{num(r[1])} [{num(r[2])}, {num(r[3])}] & {fp(p)} & {r[5]}'
def build(scen, label, cap):
    T = pickle.load(open(f'out/v7_tests2_StackCP_{scen}.pkl', 'rb')); D = {k: {r[0]: r for r in v} for k, v in T.items()}
    L = [r'\begin{table*}[!tbp]', r'\centering', r'\caption{' + cap + '}', r'\label{' + label + '}', r'\scriptsize\setlength{\tabcolsep}{2.6pt}', r'\begin{tabular}{lccccccccc}', r'\toprule',
         r' & \multicolumn{3}{c}{$\Delta$NIS (pp, primary)} & \multicolumn{3}{c}{$\Delta n$MAE (pp)} & \multicolumn{3}{c}{$\Delta$ late-life error (pp)}\\',
         r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}', r'Alternative & mean [95\% CI] & $p_{\rm Holm}$ & wins & mean [95\% CI] & $p_{\rm Holm}$ & wins & mean [95\% CI] & $p_{\rm Holm}$ & wins\\', r'\midrule']
    for b, nm in BN:
        L.append(nm + ' & ' + cell(D['NIS'].get(b)) + ' & ' + cell(D['nMAE'].get(b)) + ' & ' + cell(D['late'].get(b)) + r'\\')
        if b == 'OursCP': L.append(r'\midrule')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    L = [re.sub(r'(?<![\w$])-(\d)', r'$-$\1', l) if '&' in l and 'Alternative' not in l else l for l in L]
    open('tables/tab_' + label.split(':')[1] + '.tex', 'w').write('\n'.join(L) + '\n')
cap = (r'Paired comparison of the proposed (stacked, calibrated) method with each alternative over the 40 test tools of the four tool datasets (%s): '
       r'mean difference Proposed $-$ alternative in percentage points with a dataset-stratified bootstrap 95\%% confidence interval (10\,000 resamples); '
       r'$p$-value of the van Elteren test (signed-rank test stratified by dataset) with Holm correction over the 16 alternatives of each endpoint; number of tools on which the proposed method is better. '
       r'NIS: interval score (raw intervals for DE, DE+, MCD, QR and QR+; DE+CP, CQR, CQR+ and Wiener-AR use the two-layer conformal step); late-life error: MAE over the second half of life normalised by the tool life. Negative values favour the proposed method. '
       r'$^\ast$Ablation, not part of the Holm family (unadjusted $p$). The 27 tools of the development datasets were used to design the method; the $p$-values are descriptive.')
build('insp', 'tab:signif', cap % 'periodic inspections')
build('none', 'tab:Ssignif_none', cap % 'mounting record only')
print(open('tables/tab_signif.tex').read())
