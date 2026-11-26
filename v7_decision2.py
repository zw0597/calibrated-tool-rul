"""v1.4 (review): cost-rate study with (i) an optimal age-replacement benchmark fitted on the training tools' lives,
(ii) a cost c_I per inspection performed before replacement, (iii) a dataset-stratified bootstrap interval of the cost
ratio of the proposed method to each alternative at L = 10 %, and coverage/width over the last 30 % of life.
  python v7_decision2.py -> out/v7_decision2.pkl, tables/tab_cost.tex, tables/tab_Slate.tex"""
import pickle, numpy as np
import v6_analysis as V
from v7_decision import preds, MS, NAMES, leads
from pipeline import INSP
DSS = ['PHM', 'NUAA', 'MILL', 'MATWI']; RS = (10, 20); CI = 0.05
USES_INSP = {'DE+', 'DE+CP', 'CQR+', 'Wiener', 'WienerC', 'WienerARCP', 'PF', 'PFC', 'MCprior', 'OursCP', 'StackCP'}
def ninsp(ds, t, trep):
    dt = INSP[ds]; return int(np.floor((trep - 1e-9) / dt)) if trep > 0 else 0
def tool_outcome(ds, t, m, lead):
    T_ = V.tags(ds); Tf = V.tfail(t)
    if m == 'AgeRep':                                    # lead used as the r of the age policy (see below)
        lives = np.array([V.tfail(j) for j in T_ if j != t]); r = lead; best = None
        for tau in np.unique(np.r_[lives, np.linspace(lives.min() * 0.2, lives.max(), 200)]):
            F = np.mean(lives <= tau); ml = np.mean(np.minimum(lives, tau)); cr = (r * F + (1 - F)) / ml
            if best is None or cr < best[0]: best = (cr, tau)
        tau = best[1]; fail = Tf <= tau; return fail, (Tf if fail else tau), 0
    L = lead * np.mean([V.tfail(j) for j in T_ if j != t]); p = preds(ds, t, 'insp', m); mk = V.emask(t)
    tt = V.D[t]['tte'][mk]; lo = np.maximum(p.get('lo', p['R'])[mk], 0); hit = np.where(lo <= L)[0]
    fail = (len(hit) == 0) or tt[hit[0]] > Tf; u = Tf if fail else tt[hit[0]]
    return fail, u, (ninsp(ds, t, u) if m in USES_INSP else 0)
def rate(rows, r, ci):
    used = sum(u for _, _, u, _, _ in rows); ideal = sum(T for _, _, _, _, T in rows); n = len(rows)
    cost = sum((r if f else 1.0) + ci * k for _, f, _, k, _ in rows); return (cost / used) / (n / ideal)
tools = [(ds, t) for ds in DSS for t in V.tags(ds)]
OUT = {}
for m in list(MS) + ['AgeRep']:
    for lead in (leads if m != 'AgeRep' else RS):
        OUT[(m, lead)] = [(ds, *tool_outcome(ds, t, m, lead), V.tfail(t)) for ds, t in tools]
def cr(m, lead, r, ci=0.0): return rate(OUT[(m, r if m == 'AgeRep' else lead)], r, ci)
rng = np.random.default_rng(3); idx = {ds: [i for i, (d, t) in enumerate(tools) if d == ds] for ds in DSS}
BOOT = [np.concatenate([rng.choice(idx[ds], len(idx[ds])) for ds in DSS]) for _ in range(2000)]
def ratio_ci(m, r):
    a = OUT[('StackCP', 0.10)]; b = OUT[(m, r if m == 'AgeRep' else 0.10)]
    v = [rate([a[i] for i in bi], r, 0) / rate([b[i] for i in bi], r, 0) for bi in BOOT]; return np.percentile(v, [2.5, 97.5])
NM = [('AgeRep', 'Age replacement ($r$-optimal)')] + list(NAMES)
L = [r'\begin{table}[!tbp]', r'\centering',
     r'\caption{Long-run cost rate of the replacement rule of Table~\ref{tab:decision} (renewal-reward: planned replacement costs 1, a failure costs $r$; total cost divided by total used life over the 40 tools, relative to replacing every tool exactly at failure, so that 1 is ideal). Best: lowest cost rate over $L\in\{2.5,5,10,15,20,30\}\%$ (chosen in hindsight, equally for all methods). $c_I$: additional cost of 0.05 per inspection performed before replacement. Ratio: cost rate of the proposed method divided by that of the alternative at $L=10\%$, $r=10$, with a dataset-stratified bootstrap 95\% interval over tools. Age replacement: replacement at the age that minimises the empirical cost rate of the training tools\' lives (no condition monitoring).}',
     r'\label{tab:cost}', r'\scriptsize\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{lcccccccc}', r'\toprule',
     r' & \multicolumn{4}{c}{$r=10$} & \multicolumn{3}{c}{$r=20$} & \\', r'\cmidrule(lr){2-5}\cmidrule(lr){6-8}',
     r'Method & $L=5\%$ & $L=10\%$ & best & $L=10\%$, $c_I$ & $L=5\%$ & $L=10\%$ & best & ratio [95\% CI]\\', r'\midrule']
RES = {}
for m, nm in NM:
    row = [nm]
    for r in RS:
        if m == 'AgeRep':
            v = cr(m, None, r); row += ['--', '--', f'{v:.2f}'] + (['--'] if r == 10 else [])
        else:
            vs = {l: cr(m, l, r) for l in leads}; lb = min(vs, key=vs.get)
            row += [f'{vs[0.05]:.2f}', f'{vs[0.10]:.2f}', f'{vs[lb]:.2f}'] + ([f'{cr(m, 0.10, r, CI):.2f}'] if r == 10 else [])
    if m == 'StackCP': row.append('1')
    else: lo_, hi_ = ratio_ci(m, 10); row.append(f'{cr("StackCP",0.10,10)/cr(m,0.10,10) if m!="AgeRep" else cr("StackCP",0.10,10)/cr(m,None,10):.2f} [{lo_:.2f}, {hi_:.2f}]')
    L.append(' & '.join(row) + r'\\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
open('tables/tab_cost.tex', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L[10:]))
ag = {r: OUT[('AgeRep', r)] for r in RS}
for r in RS: print('AgeRep r', r, 'failures', sum(f for _, f, _, _, _ in ag[r]), 'wasted %.1f' % np.mean([100 * (T - u) / T for _, f, u, _, T in ag[r]]))
# late-life coverage / width (last 30 % of life)
from metrics import tool_metrics
MSL = [('DE+', 'DE+'), ('DE+CP', 'DE+CP'), ('CQR+', 'CQR+'), ('WienerC', 'Wiener-C'), ('MCprior', 'Law'), ('OursCP', 'Framework alone'), ('StackCP', 'Proposed')]
L = [r'\begin{table}[!htbp]', r'\centering', r'\caption{Interval quality over the last 30\% of life (periodic inspections): coverage (PICP, \%) and mean width (NMPIW, \% of life) of the 90\% intervals, averaged over the tools of each dataset.}',
     r'\label{tab:Slate}', r'\footnotesize\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{lcccccccc}', r'\toprule',
     r' & \multicolumn{2}{c}{PHM2010} & \multicolumn{2}{c}{NUAA} & \multicolumn{2}{c}{NASA milling} & \multicolumn{2}{c}{MATWI}\\', r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}', r'Method & PICP & NMPIW & PICP & NMPIW & PICP & NMPIW & PICP & NMPIW\\', r'\midrule']
for m, nm in MSL:
    row = [nm]
    for ds in DSS:
        cv = []; wd = []
        for t in V.tags(ds):
            p = preds(ds, t, 'insp', m); mk = V.emask(t); fr = V.D[t]['tte'] / V.tfail(t); s = mk & (fr >= 0.7)
            if not s.any(): continue
            R_ = V.rul_true(t)[s]; lo = np.maximum(p['lo'][s], 0); hi = p['hi'][s]
            cv.append(100 * np.mean((R_ >= lo) & (R_ <= hi))); wd.append(100 * np.mean(hi - lo) / V.tfail(t))
        row += [f'{np.mean(cv):.1f}', f'{np.mean(wd):.1f}']
    L.append(' & '.join(row) + r'\\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
open('tables/tab_Slate.tex', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L[11:-3]))
pickle.dump(OUT, open('out/v7_decision2.pkl', 'wb'))
