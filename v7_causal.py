"""v1.4 (review): framework and Law with causal inspections (causal_insp.py), periodic scenario, with the two-layer
conformal step re-estimated from inner fusions that also use causal inspections.
  NPROC=6 python v7_causal.py -> out/v7_causal.pkl  (ds -> tool -> {'Ours','OursCP','MCprior'})"""
import os, pickle, time, numpy as np
from multiprocessing import Pool
C = pickle.load(open('data/causal_insp.pkl', 'rb'))
def causal_make_insp(t, insp_dt, sig_I=0.005):
    import mc
    d = mc.D[t]; tt = d['tte']; tm, vm = C[t]; rng = np.random.default_rng(int(sum(map(ord, t))) * 7 + 1)
    def val(x):
        i = np.searchsorted(tm, x + 1e-9) - 1; return vm[max(i, 0)]
    insp = {0: val(tt[0]) + sig_I * rng.standard_normal()}
    if insp_dt:
        nxt = insp_dt
        for k in range(1, len(tt)):
            if tt[k] >= nxt: insp[k] = val(tt[k]) + sig_I * rng.standard_normal(); nxt += insp_dt
    return insp
def patch():
    import v2, mc
    v2.make_insp = causal_make_insp; mc.make_insp = causal_make_insp
_N = {}
def job(a):
    ds, k, j = a
    patch()
    from v2 import Nets, tags, run_v2, INSP, emask, rul_true
    from pipeline import cqr_score
    if ds not in _N: _N[ds] = Nets(ds)
    N = _N[ds]; T = tags(ds); tr = [t for t in T if t != k]
    if j is None:
        S, _ = N.outer_srcs(k, tr); o = run_v2(k, tr, S, INSP[ds]); l = run_v2(k, tr, {}, INSP[ds])
        return a, {'Ours': dict(R=o['R'], lo=o['Rq'][:, 0], hi=o['Rq'][:, 4], Rq=o['Rq']), 'MCprior': dict(R=l['R'], lo=l['Rq'][:, 0], hi=l['Rq'][:, 4])}
    tr2 = [x for x in tr if x != j]; o = run_v2(j, tr2, N.inner_srcs(k, j, tr), INSP[ds]); m = emask(j)
    return a, dict(score=cqr_score(rul_true(j)[m], o['Rq'][m, 0], o['Rq'][m, 4]), R=o['R'], Rq=o['Rq'])
if __name__ == '__main__':
    os.environ['NT'] = '1'
    from v2 import tags
    from pipeline import conformal_Q, cqr_apply
    f = 'out/v7_causal_raw.pkl'; R = pickle.load(open(f, 'rb')) if os.path.exists(f) else {}
    DSS = ['PHM', 'NUAA', 'MILL', 'MATWI']
    args = [(ds, k, None) for ds in DSS for k in tags(ds)] + [(ds, k, j) for ds in DSS for k in tags(ds) for j in tags(ds) if j != k]
    args = [a for a in args if a not in R]; print('todo', len(args), flush=True); t0 = time.time()
    with Pool(int(os.environ.get('NPROC', '6'))) as p:
        for i, (a, o) in enumerate(p.imap_unordered(job, args, chunksize=1)):
            R[a] = o
            if i % 40 == 0: pickle.dump(R, open(f, 'wb')); print(i, round(time.time() - t0), flush=True)
    pickle.dump(R, open(f, 'wb')); OUT = {}
    for ds in DSS:
        T = tags(ds)
        for k in T:
            r = dict(R[(ds, k, None)]); Q = conformal_Q([R[(ds, k, j)]['score'] for j in T if j != k])
            lo, hi = cqr_apply(np.maximum(r['Ours']['lo'], 0), np.maximum(r['Ours']['hi'], 0), Q)
            r['OursCP'] = dict(R=r['Ours']['R'], lo=lo, hi=hi, Q=Q); OUT.setdefault(ds, {})[k] = r
    pickle.dump(OUT, open('out/v7_causal.pkl', 'wb')); print('done', round(time.time() - t0), flush=True)
