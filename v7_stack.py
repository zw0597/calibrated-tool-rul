"""v1.4: nested predictive stacking of the framework with the inspection-aware deep ensemble (DE+), followed by the
same tool-level two-layer conformal step.

Inside each outer fold k, the weight a of the framework is chosen on the inner LOTO folds (held-out training tools j,
framework and DE+ both fitted without k and j) by the smallest mean inner nMAE over the grid a = 0, 0.1, ..., 1;
the stacked quantiles are a*q_F + (1-a)*q_DE+ (quantile averaging); the conformal margin is the two-layer quantile of
the inner stacked scores. The outer test tool is used only for evaluation.
  python v7_stack.py -> out/v7_stack.pkl  (ds -> tool -> scen -> {'Stack':..., 'StackCP':..., 'a': ...})
"""
import pickle, numpy as np, sys, os
from pipeline import rul_true, emask, tfail, conformal_Q, cqr_score, cqr_apply
from lib import tags
GRID = np.round(np.linspace(0, 1, 11), 2)
FI = pickle.load(open('out/v7_inner_fuse.pkl', 'rb'))
def _assemble():
    import os
    R = pickle.load(open('out/v7_inner_de_raw.pkl', 'rb')); out = {}
    if os.path.exists('out/v7_inner_de_raw_matwi.pkl'): R.update(pickle.load(open('out/v7_inner_de_raw_matwi.pkl', 'rb')))
    for key in {a[:4] for a in R}:
        ns = 2 if key[0] == 'MATWI' else 5   # MATWI: 2 inner DE+ members (protocol amendment 2)
        if not all(key + (s,) in R for s in range(ns)): continue
        mus = np.array([R[key + (s,)][0] for s in range(ns)]); sds = np.array([R[key + (s,)][1] for s in range(ns)])
        mu = mus.mean(0); sd = np.sqrt((sds ** 2).mean(0) + mus.var(0))
        out[key] = dict(R=np.maximum(mu, 0), lo=np.maximum(mu - 1.645 * sd, 0), hi=mu + 1.645 * sd)
    return out
DI = _assemble()
AUX = pickle.load(open('out/v5_deepaux.pkl', 'rb'))
if os.path.exists('out/v7_matwi_deepaux.pkl'): AUX.update(pickle.load(open('out/v7_matwi_deepaux.pkl', 'rb')))
DSS = sys.argv[1:] or ['PHM', 'NUAA', 'MILL']


def stack(F, Dp, a):
    lo = a * F['Rq'][:, 0] + (1 - a) * np.maximum(Dp['lo'], 0)
    hi = a * F['Rq'][:, 4] + (1 - a) * np.maximum(Dp['hi'], 0)
    return dict(R=a * F['R'] + (1 - a) * np.maximum(Dp['R'], 0), lo=lo, hi=hi)


def nmae(t, p):
    m = emask(t); return np.abs(p['R'][m] - rul_true(t)[m]).mean() / tfail(t)


def main():
    OUT = {}
    for ds in DSS:
        O = pickle.load(open(f'out/rul2_{ds}.pkl', 'rb')); T = tags(ds)
        for k in T:
            tr = [t for t in T if t != k]
            for scen in ['none', 'insp']:
                inner = {j: (FI[(ds, k, j, scen)], DI[(ds, k, j, scen)]) for j in tr}
                err = [np.mean([nmae(j, stack(F, Dp, a)) for j, (F, Dp) in inner.items()]) for a in GRID]
                a = float(GRID[int(np.argmin(err))])
                sc = []
                for j, (F, Dp) in inner.items():
                    p = stack(F, Dp, a); m = emask(j); sc.append(cqr_score(rul_true(j)[m], p['lo'][m], p['hi'][m]))
                Q = conformal_Q(sc)
                Fo = O[k][scen]['Ours']; Fo = dict(R=Fo['R'], Rq=Fo['Rq'])
                p = stack(Fo, AUX[ds][k][scen]['DE+'], a)
                lo, hi = cqr_apply(np.maximum(p['lo'], 0), np.maximum(p['hi'], 0), Q)
                OUT.setdefault(ds, {}).setdefault(k, {})[scen] = {'Stack': p, 'StackCP': dict(R=p['R'], lo=lo, hi=hi, Q=Q), 'a': a, 'inner_err': err}
            print(ds, k, {s: OUT[ds][k][s]['a'] for s in ['none', 'insp']}, flush=True)
    import os
    f = 'out/v7_stack.pkl'; old = pickle.load(open(f, 'rb')) if os.path.exists(f) else {}
    old.update(OUT); pickle.dump(old, open(f, 'wb'))


if __name__ == '__main__':
    main()
