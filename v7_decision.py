"""v1.4: replacement-policy and cost-rate studies over the 40 tools of the four tool datasets, with the stacked method.
  python v7_decision.py -> out/v7_decision.pkl, out/v7_costrate.pkl, tables/tab_decision.tex, tables/tab_cost.tex"""
import pickle, numpy as np
import v6_analysis as V
ST = pickle.load(open('out/v7_stack.pkl', 'rb'))
_p = V.preds
def preds(ds, t, scen, m):
    if m in ('Stack', 'StackCP'): return ST[ds][t][scen][m]
    return _p(ds, t, scen, m)
V.preds = preds
V.DSS = ['PHM', 'NUAA', 'MILL', 'MATWI']
NT = sum(len(V.tags(d)) for d in V.DSS)
MS = ('Pop', 'DE', 'DE+', 'DE+CP', 'CQR', 'CQR+', 'Wiener', 'WienerC', 'WienerARCP', 'PF', 'PFC', 'MCprior', 'OursCP', 'StackCP')
NAMES = [('Pop', 'Pop (time-based)'), ('DE', 'DE'), ('DE+', 'DE+'), ('DE+CP', 'DE+CP'), ('CQR', 'CQR'), ('CQR+', 'CQR+'), ('Wiener', 'Wiener'), ('WienerC', 'Wiener-C'), ('WienerARCP', 'Wiener-AR'), ('PF', 'PF'), ('PFC', 'PF-C'), ('MCprior', 'Law'), ('OursCP', 'Framework alone'), ('StackCP', 'Proposed')]
import builtins
R = V.decision(methods=MS)
# cost rate (as v6_analysis.costrate, with the number of tools computed)
leads = (0.025, 0.05, 0.10, 0.15, 0.20, 0.30); ratios = (5, 10, 20); C = {}
for m in MS:
    rec = {}
    for lead in leads:
        cost = {r: 0.0 for r in ratios}; used = 0.0; ideal = 0.0
        for ds in V.DSS:
            T_ = V.tags(ds)
            for t in T_:
                L = lead * np.mean([V.tfail(j) for j in T_ if j != t]); p = preds(ds, t, 'insp', m); mk = V.emask(t)
                tt = V.D[t]['tte'][mk]; lo = np.maximum(p.get('lo', p['R'])[mk], 0); Tf = V.tfail(t); hit = np.where(lo <= L)[0]
                fail = (len(hit) == 0) or tt[hit[0]] > Tf; u = Tf if fail else tt[hit[0]]
                used += u; ideal += Tf
                for r in ratios: cost[r] += r if fail else 1.0
        for r in ratios: rec[(lead, r)] = (cost[r] / used) / (NT / ideal)
    C[m] = rec
pickle.dump(R, open('out/v7_decision.pkl', 'wb')); pickle.dump(C, open('out/v7_costrate.pkl', 'wb'))
L = [r'\begin{table}[!tbp]', r'\centering',
     r'\caption{Replacement decisions (periodic inspections, 40 tools of the four tool datasets). The tool is replaced at the first evaluation window at which the lower bound of the 90\% interval falls below a lead time $L$, a fraction of the mean life of the training tools; a failure is a tool that reaches $V^\ast$ before replacement; wasted life: remaining life at replacement in \% of the tool life, averaged over the 40 tools.}',
     r'\label{tab:decision}', r'\footnotesize\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{lcccccc}', r'\toprule',
     r' & \multicolumn{2}{c}{$L=5\%$} & \multicolumn{2}{c}{$L=10\%$} & \multicolumn{2}{c}{$L=20\%$}\\', r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
     r'Method & failures & wasted (\%) & failures & wasted (\%) & failures & wasted (\%)\\', r'\midrule']
for m, nm in NAMES:
    row = [nm]
    for l in (0.05, 0.10, 0.20):
        per = R[(l, m)]; row += [str(int(sum(p['fail'] for p in per))), f"{np.mean([p['waste'] for p in per]):.1f}"]
    L.append(' & '.join(row) + r'\\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
open('tables/tab_decision.tex', 'w').write('\n'.join(L) + '\n')
L = [r'\begin{table}[!tbp]', r'\centering', r'\caption{Long-run cost rate of the replacement rule of Table~\ref{tab:decision} (renewal-reward: planned replacement costs 1, a failure costs $r$; total cost divided by total used life over the 40 tools, relative to replacing every tool exactly at failure, so that 1 is ideal). Best: lowest cost rate over $L\in\{2.5,5,10,15,20,30\}\%$ (chosen in hindsight, equally for all methods), with the corresponding $L$.}',
     r'\label{tab:cost}', r'\footnotesize\setlength{\tabcolsep}{3.5pt}', r'\begin{tabular}{lcccccc}', r'\toprule', r' & \multicolumn{3}{c}{$r=10$} & \multicolumn{3}{c}{$r=20$}\\', r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
     r'Method & $L=5\%$ & $L=10\%$ & best ($L$) & $L=5\%$ & $L=10\%$ & best ($L$)\\', r'\midrule']
for m, nm in NAMES:
    row = [nm]
    for r in (10, 20):
        v = {l: C[m][(l, r)] for l in leads}; lb = min(v, key=v.get)
        row += [f'{v[0.05]:.2f}', f'{v[0.10]:.2f}', f'{v[lb]:.2f} ({100*lb:g}\\%)']
    L.append(' & '.join(row) + r'\\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
open('tables/tab_cost.tex', 'w').write('\n'.join(L) + '\n')
print(open('tables/tab_decision.tex').read().split('midrule')[1]); print(open('tables/tab_cost.tex').read().split('midrule')[1])
