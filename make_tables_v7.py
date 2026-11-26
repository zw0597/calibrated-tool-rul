"""v1.4 tables: main results with the stacked predictor and MATWI, ablation and parameter tables with MATWI.
  python make_tables_v7.py -> tables/main_insp_a.tex, main_insp_b.tex, tab_Smain_none_a.tex, tab_Smain_none_b.tex, tab_ablation.tex, tab_params.tex
"""
import pickle, numpy as np
import make_tables_v6 as M
from metrics import tool_metrics
from pipeline import tags
ST = pickle.load(open('out/v7_stack.pkl', 'rb'))
DSS = ['PHM', 'NUAA', 'MILL', 'MATWI']
DNAME = dict(M.DNAME); DNAME['MATWI'] = 'MATWI, frozen evaluation (13 tools, 10 conditions, 2 materials)'
_agg6 = M.agg
def agg(ds, scen, m):
    if m in ('Stack', 'StackCP'):
        per = [tool_metrics(t, ST[ds][t][scen][m]) for t in tags(ds)]
        return {k: np.nanmean([q[k] for q in per]) for k in per[0]}, per
    if m == 'StackCtrlCP':
        SC = pickle.load(open('out/v7_stack_ctrl.pkl', 'rb'))
        per = [tool_metrics(t, SC[ds][t][scen]['StackCtrlCP']) for t in tags(ds)]
        return {k: np.nanmean([q[k] for q in per]) for k in per[0]}, per
    if m in ('DE+CP', 'WienerAR', 'WienerARCP'):
        import v6_analysis as V
        per = [tool_metrics(t, V.preds(ds, t, scen, m)) for t in tags(ds)]
        return {k: np.nanmean([q[k] for q in per]) for k in per[0]}, per
    if m in ('Plus', 'PlusCP') and (M.PE is None or ds not in M.PE): return {}, []
    return _agg6(ds, scen, m)
M.agg = agg
NAMES = [('Pop', 'Pop'), ('DE', 'DE'), ('DE+', 'DE+'), ('DE+CP', 'DE+CP'), ('MCD', 'MCD'), ('CQRraw', 'QR'), ('QR+', 'QR+'), ('CQR', 'CQR'), ('CQR+', 'CQR+'),
         ('LinEx', 'LinEx'), ('Wiener', 'Wiener'), ('WienerC', 'Wiener-C'), ('WienerNetK', 'Wiener-Net'), ('WienerARCP', 'Wiener-AR'), ('PF', 'PF'), ('PFC', 'PF-C'), ('PFNet', 'PF-Net'),
         ('MCprior', 'Law'), ('StackCtrlCP', r'QR+ stacked with DE+$^\S$'), ('Ours', 'Framework alone, raw'), ('OursCP', 'Framework alone'), ('PlusCP', r'Framework, QR+ evidence$^\ddagger$'),
         ('Stack', 'Proposed, raw'), ('StackCP', r'\textbf{Proposed}')]
fmt = M.fmt
def main_table(scen, label, caption, dss, fs=True):
    cols = [('nMAE', 2, 'min'), ('RMSE', 0, 'min'), ('MAElate', 0, 'min'), ('AL', 2, 'max'), ('PICP', 1, None), ('NMPIW', 1, None), ('NIS', 1, 'min')]
    L = [r'\begin{table}[!tbp]', r'\centering', r'\caption{' + caption + '}', r'\label{' + label + '}',
         r'\scriptsize\setlength{\tabcolsep}{4.5pt}\renewcommand{\arraystretch}{0.93}', r'\begin{tabular}{lrrrrrrr}', r'\toprule',
         r'Method & $n$MAE (\%) & RMSE (s) & MAE$_{\rm late}$ (s) & $\alpha$--$\lambda$ & PICP (\%) & NMPIW (\%) & NIS (\%)\\']
    R = {}
    for ds in dss:
        R[ds] = {m: agg(ds, scen, m)[0] for m, _ in NAMES}
        best = {}
        for c, d, how in cols:
            if how is None: continue
            vals = [float(fmt(R[ds][m].get(c, np.nan), d)) for m, _ in NAMES if m != 'LinEx' and np.isfinite(R[ds][m].get(c, np.nan))]
            best[c] = min(vals) if how == 'min' else max(vals)
        L.append(r'\midrule'); L.append(r'\multicolumn{8}{@{}l}{\emph{' + DNAME[ds] + r'}}\\')
        for m, nm in NAMES:
            if not R[ds][m]: continue
            row = [nm]
            for c, d, how in cols:
                v = R[ds][m].get(c, np.nan); s_ = fmt(v, d)
                if how and np.isfinite(v) and abs(float(s_) - best[c]) < 1e-9 and m != 'LinEx': s_ = r'\textbf{' + s_ + '}'
                row.append(s_)
            L.append(' & '.join(row) + r'\\')
        if fs and ds in ('MILL', 'MATWI'):
            from metrics2 import fs_conformal
            fsr = fs_conformal(ds, scen)[0]; R[ds]['OursFS'] = fsr
            L.append(' & '.join([r'Framework alone, subsampled CP$^\dagger$'] + [fmt(fsr[c], d) for c, d, how in cols]) + r'\\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    open(f'tables/{label.split(":")[1]}.tex', 'w').write('\n'.join(L)); return R
NOTE = (r'Metrics are averaged over test tools; bold: best value per dataset and column over all rows, the framework-alone ablation included (PICP and NMPIW excluded, since neither is a score on its own; LinEx excluded because of its divergent extrapolation). '
        r'Proposed: framework stacked with DE+ (Section~\ref{sec:stack}) and calibrated; Framework alone: the unstacked framework ($a=1$). '
        r'DE+, QR+: deep regressors that also receive the mounting wear and the inspections; DE+CP, CQR+: DE+ and QR+ with the two-layer conformal step; Wiener-AR: Wiener-Net with the AR(1) bias state of the error model and the two-layer conformal step; Wiener-C, PF-C: condition-dependent priors; '
        r'Wiener-Net, PF-Net: network wear estimates treated as independent measurements; PF: particle filter with inspections only (Section~\ref{sec:baselines}). '
        r'$^\dagger$Subsampled conformal step of Section~\ref{sec:conformal} (one random window per calibration tool, 400 draws), possible only for $m\ge9$ calibration tools. '
        r'$^\ddagger$Framework with the QR+ ensemble as direct-RUL evidence (counts the inspections twice; not computed for MATWI). $^\S$Stacking control: QR+ in place of the framework in the nested stacking with DE+, with the same conformal step.')
def ablation():
    V = [('Stack', 'Proposed: stacked with DE+, raw'), ('Ours', 'Framework alone ($a=1$), raw'),
         ('woShift', 'w/o effective-age offset ($\\epsilon\\equiv0$)'), ('woBank', 'w/o hypothesis averaging (single filter)'),
         ('condOnly', 'extended-Taylor prior only'), ('poolOnly', 'pooled prior only'),
         ('woI', 'w/o wear-increment evidence'), ('woR', 'w/o direct-RUL evidence'), ('woAR', 'w/o persistent bias ($\\psi_s\\to0.05$)'), ('woDisc', 'w/o model discrepancy $\\delta$'),
         ('singleQ', 'single quantile network instead of ensemble'), ('v4nom', 'nominal-path RUL (discrepancy not propagated)'),
         ('v4:woPred', 'w/o small-sample predictive scale (Eq.~\\eqref{eq:predscale})'), ('v4:plainP', 'plain pseudo-BMA (no Bayesian bootstrap)'), ('v4:uniP', 'uniform weights over prior structures'),
         ('MCprior', 'law + inspections only (Law)')]
    L = [r'\begin{table*}[!tbp]', r'\centering', r'\caption{Ablation study (periodic inspections; raw intervals, before conformal calibration). Rows below the second are variants of the framework alone. --: not computed.}', r'\label{tab:ablation}',
         r'\scriptsize\setlength{\tabcolsep}{2.6pt}', r'\begin{tabular}{lcccccccccccc}', r'\toprule',
         r' & \multicolumn{3}{c}{PHM2010} & \multicolumn{3}{c}{NUAA} & \multicolumn{3}{c}{NASA milling} & \multicolumn{3}{c}{MATWI}\\',
         r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}',
         r'Variant & $n$MAE & PICP & NIS & $n$MAE & PICP & NIS & $n$MAE & PICP & NIS & $n$MAE & PICP & NIS\\', r'\midrule']
    out = {}
    for m, nm in V:
        row = [nm]
        for ds in DSS:
            O = pickle.load(open(f'out/rul2_{ds}.pkl', 'rb'))
            if m in ('Stack', 'v4nom'):
                if m == 'v4nom' and ds == 'MATWI': row += ['--'] * 3; continue
            elif m.startswith('v4:'):
                if ds not in M.AB[m[3:]]: row += ['--'] * 3; continue
            elif m not in O[list(O)[0]]['insp']: row += ['--'] * 3; continue
            a = agg(ds, 'insp', m)[0]; out[(ds, m)] = a; row += [fmt(a['nMAE']), fmt(a['PICP']), fmt(a['NIS'])]
        L.append(' & '.join(row) + r'\\')
        if m == 'Ours': L.append(r'\midrule')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    open('tables/tab_ablation.tex', 'w').write('\n'.join(L)); return out
if __name__ == '__main__':
    M.DSS = DSS
    Ra = main_table('insp', 'tab:main_insp_a', 'RUL prediction with periodic inspections (every 200 s for PHM2010 and 300 s for NUAA). ' + NOTE, ['PHM', 'NUAA'])
    Rb = main_table('insp', 'tab:main_insp_b', r'RUL prediction with periodic inspections (every 600 s for NASA milling and 160 s for MATWI); notation as in Table~\ref{tab:main_insp_a}. MATWI was evaluated once, under a protocol published before any RUL result (Section~\ref{sec:res_matwi}).', ['MILL', 'MATWI'])
    Na = main_table('none', 'tab:Smain_none_a', r'RUL prediction with the mount-only inspection, PHM2010 and NUAA; notation as in Table~\ref{tab:main_insp_a} of the paper.', ['PHM', 'NUAA'])
    Nb = main_table('none', 'tab:Smain_none_b', r'RUL prediction with the mount-only inspection, NASA milling and MATWI; notation as in Table~\ref{tab:main_insp_a} of the paper.', ['MILL', 'MATWI'])
    A = ablation(); P = M.params_table()
    t = open('tables/tab_params.tex').read().replace(r'\begin{tabular}{lccc}', r'\begin{tabular}{lcccc}').replace(r'Quantity & PHM2010 & NUAA & NASA milling\\', r'Quantity & PHM2010 & NUAA & NASA milling & MATWI\\')
    t = t.replace(r'\small\setlength{\tabcolsep}{4pt}', r'\footnotesize\setlength{\tabcolsep}{3pt}'); open('tables/tab_params.tex', 'w').write(t)
    pickle.dump((Ra, Rb, Na, Nb, A), open('out/main_tables_v7.pkl', 'wb'))
    for ds, R in list(Ra.items()) + list(Rb.items()):
        print(ds, {m: (round(R[m].get('nMAE', np.nan), 1), round(R[m].get('NIS', np.nan), 1), round(R[m].get('PICP', np.nan), 1)) for m in ['DE+', 'CQR+', 'MCprior', 'OursCP', 'StackCP']})
