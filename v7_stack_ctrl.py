"""v1.4 (review): stacking control. The same nested stacking and two-layer conformal step as v7_stack.py, but with the
QR+ quantile ensemble in place of the framework: Stack(QR+, DE+)+CP. Tests whether the stacked gains come from combining
two diverse predictors rather than from the framework.  python v7_stack_ctrl.py -> out/v7_stack_ctrl.pkl"""
import pickle, numpy as np, os
from pipeline import rul_true, emask, tfail, conformal_Q, cqr_score, cqr_apply
from lib import tags
from v7_stack import DI, AUX, GRID, stack, nmae
QI = pickle.load(open('out/v5_plus_inner.pkl', 'rb')); QI.update(pickle.load(open('out/v7_matwi_plus_inner.pkl', 'rb')))
RAW = pickle.load(open('out/v5_deepaux_raw.pkl', 'rb'))
if os.path.exists('out/v7_matwi_deepaux_raw.pkl'): RAW.update(pickle.load(open('out/v7_matwi_deepaux_raw.pkl', 'rb')))
def qdict(q): return dict(R=np.maximum(q[:, 2], 0), Rq=np.maximum(q, 0))
OUT = {}
for ds in ['PHM', 'NUAA', 'MILL', 'MATWI']:
    T = tags(ds); ns = 2
    for k in T:
        tr = [t for t in T if t != k]
        for scen in ['none', 'insp']:
            inner = {j: (qdict(np.mean([QI[(ds, scen, k, j, s)] for s in range(ns)], 0)), DI[(ds, k, j, scen)]) for j in tr}
            err = [np.mean([nmae(j, stack(F, Dp, a)) for j, (F, Dp) in inner.items()]) for a in GRID]
            a = float(GRID[int(np.argmin(err))]); sc = []
            for j, (F, Dp) in inner.items():
                p = stack(F, Dp, a); m = emask(j); sc.append(cqr_score(rul_true(j)[m], p['lo'][m], p['hi'][m]))
            Q = conformal_Q(sc); F = qdict(np.mean([RAW[(ds, k, scen, 'qr', s)] for s in range(5)], 0))
            p = stack(F, AUX[ds][k][scen]['DE+'], a); lo, hi = cqr_apply(np.maximum(p['lo'], 0), np.maximum(p['hi'], 0), Q)
            OUT.setdefault(ds, {}).setdefault(k, {})[scen] = {'StackCtrlCP': dict(R=p['R'], lo=lo, hi=hi, Q=Q), 'a': a}
pickle.dump(OUT, open('out/v7_stack_ctrl.pkl', 'wb'))
from metrics import tool_metrics
for ds in OUT:
    per = [tool_metrics(t, OUT[ds][t]['insp']['StackCtrlCP']) for t in tags(ds)]
    print(ds, {c: round(float(np.nanmean([q[c] for q in per])), 1) for c in ['nMAE', 'PICP', 'NMPIW', 'NIS', 'MAElate']}, 'a', [OUT[ds][t]['insp']['a'] for t in tags(ds)][:6])
