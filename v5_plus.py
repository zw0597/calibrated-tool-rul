"""Framework with an inspection-aware direct-RUL network as evidence (v1.2, exploratory).

The direct-RUL evidence of the proposed method is replaced by the QR+ quantile ensemble of v5_baselines.py
(inputs include the mounting record and, with periodic inspections, the latest inspection); the error model and
the conformal scores use inner QR+ ensembles trained in the nested leave-one-tool-out splits.
  python v5_plus.py nets   -> out/v5_plus_inner.pkl
  python v5_plus.py eval   -> out/v5_plus_eval.pkl
"""
import os, sys, pickle, time, numpy as np
from multiprocessing import Pool
QS = [0.05, 0.25, 0.5, 0.75, 0.95]; DSS = ['PHM', 'NUAA', 'MILL']; NIN = 2


def job_net(a):
    ds, scen, k, j, s = a
    import lib
    from pipeline import INSP
    lib.AUX = ('insp', INSP[ds]) if scen == 'insp' else ('none', None)
    T = lib.tags(ds); tr2 = [t for t in T if t not in (k, j)]
    net = lib.train(tr2, lib.rul_target, seed=s, loss='q', qs=QS)
    return a, lib.predict(net, j)


def job_eval(a):
    ds, k = a
    from v2 import Nets, run_v2, tags, INSP, emask, rul_true, rul_src, mom, conformal_Q, cqr_score, cqr_apply
    from metrics import tool_metrics
    RAW = pickle.load(open('out/v5_deepaux_raw.pkl', 'rb')); RI = pickle.load(open('out/v5_plus_inner.pkl', 'rb'))
    T = tags(ds); tr = [t for t in T if t != k]; N = Nets(ds); out = {}
    for scen in ['none', 'insp']:
        idt = INSP[ds] if scen == 'insp' else None
        def qin(kk, jj): return np.mean([RI[(ds, scen, kk, jj, s)] for s in range(NIN)], 0)
        def params(kk, tools):
            pr = []
            for jj in tools:
                R = rul_src(qin(kk, jj)); m = emask(jj); pr.append(((R['z'] - rul_true(jj))[m], R['s'][m]))
            return mom(pr)
        S, _ = N.outer_srcs(k, tr)
        q = np.mean([RAW[(ds, k, scen, 'qr', s)] for s in range(5)], 0)
        S['rul'] = dict(rul_src(q), **params(k, tr), kind='rul')
        o = run_v2(k, tr, S, idt); p = dict(R=o['R'], lo=o['Rq'][:, 0], hi=o['Rq'][:, 4])
        sc = []
        for j in tr:
            tr2 = [x for x in tr if x != j]; Sj = N.inner_srcs(k, j, tr)
            Sj['rul'] = dict(rul_src(qin(k, j)), **params(k, tr2), kind='rul')
            oj = run_v2(j, tr2, Sj, idt); m = emask(j); sc.append(cqr_score(rul_true(j)[m], oj['Rq'][m, 0], oj['Rq'][m, 4]))
        Q = conformal_Q(sc); lo, hi = cqr_apply(p['lo'], p['hi'], Q)
        out[scen] = {'Plus': tool_metrics(k, p), 'PlusCP': tool_metrics(k, dict(R=p['R'], lo=lo, hi=hi)), 'pred': p, 'Q': Q}
    return ds, k, out


if __name__ == '__main__':
    os.environ['NT'] = '1'; what = sys.argv[1]; t0 = time.time()
    from lib import tags
    if what == 'nets':
        args = [(ds, sc, k, j, s) for ds in DSS for sc in ['none', 'insp'] for k in tags(ds) for j in tags(ds) if j != k for s in range(NIN)]
        R = {}
        with Pool(int(os.environ.get('NPROC', '17'))) as p:
            for i, (key, val) in enumerate(p.imap_unordered(job_net, args, chunksize=1)):
                R[key] = val
                if i % 50 == 0: print(i, len(args), round(time.time() - t0), flush=True)
        pickle.dump(R, open('out/v5_plus_inner.pkl', 'wb'))
    else:
        args = [(ds, k) for ds in DSS for k in tags(ds)]
        with Pool(16) as p: res = p.map(job_eval, args, chunksize=1)
        E = {}
        for ds, k, o in res: E.setdefault(ds, {})[k] = o
        pickle.dump(E, open('out/v5_plus_eval.pkl', 'wb'))
        for ds in DSS:
            for sc in ['none', 'insp']:
                for m in ['Plus', 'PlusCP']:
                    print(ds, sc, m, {kk: round(np.mean([E[ds][t][sc][m][kk] for t in E[ds]]), 1) for kk in ['nMAE', 'RMSE', 'MAElate', 'PICP', 'NIS']})
    print(what, 'done', round(time.time() - t0))
