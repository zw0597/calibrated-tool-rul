"""v1.4 (review): DE+ with the same tool-level two-layer conformal step (DE+CP), scores from the nested inner DE+
ensembles used for stacking (5 members; 2 on MATWI).  python v7_deplus_cp.py -> out/v7_deplus_cp.pkl"""
import pickle, numpy as np
from pipeline import rul_true, emask, conformal_Q, cqr_score, cqr_apply
from lib import tags
import v7_stack as S
OUT = {}
for ds in ['PHM', 'NUAA', 'MILL', 'MATWI']:
    T = tags(ds)
    for k in T:
        tr = [t for t in T if t != k]
        for scen in ['none', 'insp']:
            sc = []
            for j in tr:
                p = S.DI[(ds, k, j, scen)]; m = emask(j); sc.append(cqr_score(rul_true(j)[m], p['lo'][m], p['hi'][m]))
            Q = conformal_Q(sc); p = S.AUX[ds][k][scen]['DE+']
            lo, hi = cqr_apply(np.maximum(p['lo'], 0), np.maximum(p['hi'], 0), Q)
            OUT.setdefault(ds, {}).setdefault(k, {})[scen] = dict(R=np.maximum(p['R'], 0), lo=lo, hi=hi, Q=Q)
pickle.dump(OUT, open('out/v7_deplus_cp.pkl', 'wb'))
from metrics import tool_metrics
for ds in OUT:
    for scen in ['insp', 'none']:
        per = [tool_metrics(t, OUT[ds][t][scen]) for t in tags(ds)]
        print(ds, scen, {c: round(float(np.nanmean([q[c] for q in per])), 1) for c in ['nMAE', 'PICP', 'NMPIW', 'NIS']})
