"""Model variants of the revised manuscript (v1.3), outer test tools, raw intervals:
  d0comp : initial discrepancy mean -epsilon (offset prior decoupled from sigma_delta)
  ri     : random intercept + AR(1) + white error model, moments fitted to lags 1..10
  both   : both changes
  python v6_variants.py -> out/v6_variants.pkl"""
import pickle, numpy as np
from multiprocessing import Pool
def job(a):
    var, ds, k = a
    import fuse2, v2, pipeline
    fuse2.DELTA0_COMP = var in ('d0comp', 'both'); fuse2.ERR_RI = var in ('ri', 'both')
    if var == 'corr':
        from v2 import Nets as _N, tags as _t, D as _D, emask as _e, rul_true as _r
        N0 = _N(ds); T0 = _t(ds); cc = []
        for j in T0:
            if j == k: continue
            I, R = N0.inner(k, j); m = _e(j); w = _D[j]['wte'] - _D[j]['wte'][0]
            zi = ((I['z'] - w) / I['s'])[m]; zr = ((R['z'] - _r(j)) / R['s'])[m]
            if len(zi) > 3: cc.append(np.corrcoef(zi, zr)[0, 1])
        fuse2.CORR_IR = float(np.clip(np.median(cc), -0.95, 0.95)) if cc else 0.0
    if fuse2.ERR_RI: v2.mom = pipeline.mom_ri
    from v2 import Nets, run_v2, tags, INSP
    from metrics import tool_metrics
    T = tags(ds); tr = [t for t in T if t != k]; N = Nets(ds); S, par = N.outer_srcs(k, tr); out = {'par': par}
    for scen, idt in [('none', None), ('insp', INSP[ds])]:
        o = run_v2(k, tr, S, idt); out[scen] = dict(R=o['R'], lo=o['Rq'][:, 0], hi=o['Rq'][:, 4])
    return var, ds, k, out
if __name__ == '__main__':
    import sys
    from pipeline import tags
    VARS = sys.argv[1].split(',') if len(sys.argv) > 1 else ['d0comp', 'ri', 'both', 'corr']
    args = [(v, ds, k) for v in VARS for ds in ['PHM', 'NUAA', 'MILL'] for k in tags(ds)]
    with Pool(16) as p: res = p.map(job, args, chunksize=1)
    O = {}
    for v, ds, k, o in res: O.setdefault(v, {}).setdefault(ds, {})[k] = o
    old = pickle.load(open('out/v6_variants.pkl', 'rb')) if __import__('os').path.exists('out/v6_variants.pkl') else {}
    old.update(O); O = old; pickle.dump(O, open('out/v6_variants.pkl', 'wb'))
    from metrics import tool_metrics
    for v in O:
        for scen in ['insp', 'none']:
            for ds in ['PHM', 'NUAA', 'MILL']:
                per = [tool_metrics(t, O[v][ds][t][scen]) for t in tags(ds)]
                print(v, scen, ds, {q: round(np.mean([p[q] for p in per]), 1) for q in ['nMAE', 'PICP', 'NMPIW', 'NIS']})
        if v == 'ri':
            for ds in ['PHM', 'NUAA', 'MILL']:
                ps = [O[v][ds][t]['par'] for t in tags(ds)]
                print('  RI params', ds, {src: {q: round(float(np.mean([p[i][q] for p in ps])), 3) for q in ['kappa', 'psi0', 'psi', 'phi']} for i, src in enumerate(['inc', 'rul'])})
