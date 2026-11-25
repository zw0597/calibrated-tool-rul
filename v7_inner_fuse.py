"""v1.4: per-window predictions of the framework on the inner LOTO folds (the same inner fusions whose scores
calibrate the conformal step in evalrul2.py), stored for the stacking weight selection.
  NPROC=4 python v7_inner_fuse.py -> out/v7_inner_fuse.pkl  (key (ds,k,j,scen) -> dict(R, Rq))
"""
import os, pickle, time, numpy as np
from multiprocessing import Pool
DSS = os.environ.get('DSS', 'PHM,NUAA,MILL').split(',')
_N = {}
def job(a):
    ds, k, j, scen = a
    from v2 import Nets, tags, run_v2, INSP
    if ds not in _N: _N[ds] = Nets(ds)
    N = _N[ds]; T = tags(ds); tr = [t for t in T if t != k]; tr2 = [x for x in tr if x != j]
    o = run_v2(j, tr2, N.inner_srcs(k, j, tr), INSP[ds] if scen == 'insp' else None)
    return a, dict(R=o['R'], Rq=o['Rq'])
if __name__ == '__main__':
    os.environ['NT'] = '1'
    from v2 import tags
    f = os.environ.get('OUTF', 'out/v7_inner_fuse.pkl')
    R = pickle.load(open(f, 'rb')) if os.path.exists(f) else {}
    args = [(ds, k, j, scen) for ds in DSS for k in tags(ds) for j in tags(ds) if j != k for scen in ['none', 'insp']]
    args = [a for a in args if a not in R]; print('todo', len(args), flush=True); t0 = time.time()
    with Pool(int(os.environ.get('NPROC', '4'))) as p:
        for i, (a, o) in enumerate(p.imap_unordered(job, args, chunksize=1)):
            R[a] = o
            if i % 20 == 0: pickle.dump(R, open(f, 'wb')); print(i, round(time.time() - t0), flush=True)
    pickle.dump(R, open(f, 'wb')); print('done', round(time.time() - t0), flush=True)
