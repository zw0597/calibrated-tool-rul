"""Tool-level jackknife+ for the proposed method (v1.3).

For an outer test tool k and every training tool j, the inner pipeline that produced the calibration score of j
(networks, wear law, priors and error model fitted without k and j) is also applied to k. With E_j the scores of j:
  * J+ (subsampled): one random window per calibration tool; lower bound = floor(alpha(m+1))-th smallest of
    {l(lo_j(k)) - E_j}, upper bound = ceil((1-alpha)(m+1))-th smallest of {l(hi_j(k)) + E_j}. For exchangeable tools
    this has coverage >= 1-2 alpha for a random window of k (Barber et al. 2021); it can be formed only when m >= 9.
  * J+ (pooled): the same with all windows, each tool weighted 1 (no finite-sample guarantee).
  python v6_jackplus.py [DS ...]  -> out/v6_jackplus_<DS>.pkl
"""
import os, sys, math, pickle, numpy as np
from multiprocessing import Pool
NIS = {'PHM': 3, 'NUAA': 2, 'MILL': 2}; QMAX = {'PHM': 5, 'NUAA': 5, 'MILL': 2}; ALPHA = 0.1; TAU = 30.0
JN = None


def lfun(r): return np.log1p(np.maximum(r, 0) / TAU)


def job(a):
    ds, k, scen = a
    global JN
    if JN is None: JN = pickle.load(open('out/v6_jack_nets.pkl' if os.path.exists('out/v6_jack_nets.pkl') else 'out/v6_jack_nets.tmp', 'rb'))
    from v2 import Nets, run_v2, tags, INSP, inc_src, rul_src
    T = tags(ds); tr = [t for t in T if t != k]; N = Nets(ds); idt = INSP[ds] if scen == 'insp' else None; out = {}
    for j in tr:
        tr2 = [x for x in tr if x != j]
        mus = np.array([JN[('inc', k, j, s)][0] for s in range(NIS[ds])]); sds = np.array([JN[('inc', k, j, s)][1] for s in range(NIS[ds])])
        I = inc_src(mus.mean(0), np.sqrt((sds ** 2).mean(0) + mus.var(0)))
        q = np.mean([JN[('rulq', k, j, s)] for s in range(QMAX[ds])], 0); R = rul_src(q)
        pI, pR = N.params(k, tr2)
        o = run_v2(k, tr2, {'inc': dict(I, **pI, kind='inc'), 'rul': dict(R, **pR, kind='rul')}, idt)
        out[j] = (o['Rq'][:, 0], o['Rq'][:, 4])
    return ds, k, scen, out


def wq(x, w, q):
    o = np.argsort(x); c = np.cumsum(w[o]); c /= c[-1]; return x[o][min(np.searchsorted(c, q), len(x) - 1)]


def intervals(ds, k, scen, inner, scores, B=400, seed=0):
    tr = list(inner.keys()); m = len(tr); rng = np.random.default_rng(seed)
    Lo = np.array([lfun(inner[j][0]) for j in tr]); Hi = np.array([lfun(inner[j][1]) for j in tr])   # (m, K)
    K = Lo.shape[1]
    # pooled
    lo_p = np.zeros(K); hi_p = np.zeros(K)
    w = np.concatenate([np.full(len(s), 1 / len(s)) for s in scores])
    for i in range(K):
        a = np.concatenate([Lo[jj, i] - scores[jj] for jj in range(m)]); b = np.concatenate([Hi[jj, i] + scores[jj] for jj in range(m)])
        lo_p[i] = wq(a, w, ALPHA); hi_p[i] = wq(b, w, 1 - ALPHA)
    pooled = (TAU * np.expm1(np.maximum(lo_p, 0)), TAU * np.expm1(hi_p))
    # subsampled
    iu = math.ceil((1 - ALPHA) * (m + 1)); il = math.floor(ALPHA * (m + 1))
    sub = None
    if iu <= m:
        draws = []
        for _ in range(B):
            e = np.array([s[rng.integers(len(s))] for s in scores])
            lo_s = np.sort(Lo - e[:, None], 0)[il - 1] if il >= 1 else np.full(K, -np.inf)
            hi_s = np.sort(Hi + e[:, None], 0)[iu - 1]
            draws.append((TAU * np.expm1(np.maximum(lo_s, 0)), TAU * np.expm1(hi_s)))
        sub = draws
    return pooled, sub


if __name__ == '__main__':
    os.environ['NT'] = '1'
    fn = 'out/v6_jack_nets.pkl' if os.path.exists('out/v6_jack_nets.pkl') else 'out/v6_jack_nets.tmp'
    JN = pickle.load(open(fn, 'rb'))
    from pipeline import tags
    from metrics import tool_metrics
    for ds in (sys.argv[1:] or ['MILL', 'NUAA', 'PHM']):
        args = [(ds, k, sc) for k in tags(ds) for sc in ['insp', 'none']]
        with Pool(16) as p: res = p.map(job, args, chunksize=1)
        O = pickle.load(open(f'out/rul2_{ds}.pkl', 'rb')); out = {}
        for _, k, scen, inner in res:
            sc = O[k][scen]['OursCP']['scores']; Rm = O[k][scen]['Ours']['R']
            pooled, sub = intervals(ds, k, scen, inner, sc)
            r = {'J+pooled': tool_metrics(k, dict(R=Rm, lo=pooled[0], hi=pooled[1]))}
            if sub is not None:
                ms = [tool_metrics(k, dict(R=Rm, lo=a, hi=b)) for a, b in sub]
                r['J+sub'] = {q: float(np.mean([x[q] for x in ms])) for q in ms[0]}
            out.setdefault(k, {})[scen] = r
        pickle.dump(out, open(f'out/v6_jackplus_{ds}.pkl', 'wb'))
        for scen in ['insp', 'none']:
            for meth in ['J+pooled', 'J+sub']:
                per = [out[k][scen][meth] for k in out if meth in out[k][scen]]
                if per: print(ds, scen, meth, {q: round(np.mean([p[q] for p in per]), 1) for q in ['PICP', 'NMPIW', 'NIS']},
                              'cov<80:', sum(p['PICP'] < 80 for p in per), flush=True)
