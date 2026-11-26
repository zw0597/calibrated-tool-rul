"""v1.4 (review): calibrated ablations. For each framework variant, inner fusions on the held-out training tools give
two-layer conformal scores (exactly as for the full model in evalrul2.py); the margin is applied to the variant's outer
raw intervals (periodic inspections).  NPROC=8 python v7_ablate_cp.py -> out/v7_ablate_cp.pkl"""
import os, sys, pickle, time, numpy as np
from multiprocessing import Pool
VARS = {'woShift': dict(shift=False), 'woBank': dict(bank='none'), 'woI': dict(drop='inc'), 'woR': dict(drop='rul'),
        'woAR': dict(ar_on=False), 'woDisc': dict(disc_on=False), 'singleQ': dict(single=True)}
_N = {}
def job(a):
    ds, k, j, V = a
    from v2 import Nets, tags, run_v2, INSP, emask, rul_true
    from pipeline import cqr_score
    key = (ds, V == 'singleQ')
    if key not in _N: _N[key] = Nets(ds, qens=not key[1])
    N = _N[key]; T = tags(ds); tr = [t for t in T if t != k]; tr2 = [x for x in tr if x != j]
    S = dict(N.inner_srcs(k, j, tr)); c = dict(VARS[V])
    if 'drop' in c: S.pop(c.pop('drop'))
    c.pop('single', None)
    o = run_v2(j, tr2, S, INSP[ds], **c); m = emask(j)
    return a, cqr_score(rul_true(j)[m], o['Rq'][m, 0], o['Rq'][m, 4])
if __name__ == '__main__':
    os.environ['NT'] = '1'
    from v2 import tags
    f = 'out/v7_ablate_cp_scores.pkl'; R = pickle.load(open(f, 'rb')) if os.path.exists(f) else {}
    args = [(ds, k, j, V) for ds in ['PHM', 'NUAA', 'MILL', 'MATWI'] for V in VARS for k in tags(ds) for j in tags(ds) if j != k]
    args = [a for a in args if a not in R]; print('todo', len(args), flush=True); t0 = time.time()
    with Pool(int(os.environ.get('NPROC', '8'))) as p:
        for i, (a, sc) in enumerate(p.imap_unordered(job, args, chunksize=1)):
            R[a] = sc
            if i % 50 == 0: pickle.dump(R, open(f, 'wb')); print(i, round(time.time() - t0), flush=True)
    pickle.dump(R, open(f, 'wb'))
    from pipeline import conformal_Q, cqr_apply
    OUT = {}
    for ds in ['PHM', 'NUAA', 'MILL', 'MATWI']:
        O = pickle.load(open(f'out/rul2_{ds}.pkl', 'rb')); T = tags(ds)
        for V in VARS:
            for k in T:
                if V not in O[k]['insp']: continue
                Q = conformal_Q([R[(ds, k, j, V)] for j in T if j != k]); r = O[k]['insp'][V]
                lo, hi = cqr_apply(np.maximum(r['lo'], 0), np.maximum(r['hi'], 0), Q)
                OUT.setdefault(ds, {}).setdefault(k, {})[V + 'CP'] = dict(R=r['R'], lo=lo, hi=hi, Q=Q)
    pickle.dump(OUT, open('out/v7_ablate_cp.pkl', 'wb')); print('done', round(time.time() - t0), flush=True)
