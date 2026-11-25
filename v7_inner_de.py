"""v1.4: inner-LOTO DE+ networks (inspection-aware deep ensemble trained without the outer test tool k and the inner held-out tool j),
used only to choose the stacking weight of the framework and DE+ inside each outer fold and to calibrate the stacked intervals.
  NPROC=16 python v7_inner_de.py        -> out/v7_inner_de.pkl  (key (ds,k,j,scen) -> dict(R,lo,hi) of the 5-member ensemble)
"""
import os, sys, pickle, time, numpy as np
from multiprocessing import Pool
DSS = os.environ.get('DSS', 'PHM,NUAA,MILL').split(',')
def job(a):
    ds, k, j, scen, seed = a
    import lib
    from pipeline import INSP
    lib.AUX = ('insp', INSP[ds]) if scen == 'insp' else ('none', None)
    T = lib.tags(ds); tr = [t for t in T if t not in (k, j)]
    net = lib.train(tr, lib.rul_target, seed=seed)
    return a, lib.predict(net, j)
if __name__ == '__main__':
    os.environ['NT'] = '1'
    import lib
    args = [(ds, k, j, scen, s) for ds in DSS for k in lib.tags(ds) for j in lib.tags(ds) if j != k
            for scen in ['none', 'insp'] for s in range(5)]
    f = 'out/v7_inner_de_raw.pkl'
    R = pickle.load(open(f, 'rb')) if os.path.exists(f) else {}
    args = [a for a in args if a not in R]
    print('todo', len(args), flush=True); t0 = time.time()
    with Pool(int(os.environ.get('NPROC', '16'))) as p:
        for i, (a, pr) in enumerate(p.imap_unordered(job, args, chunksize=1)):
            R[a] = pr
            if i % 50 == 0:
                pickle.dump(R, open(f, 'wb')); print(i, round(time.time() - t0), flush=True)
    pickle.dump(R, open(f, 'wb'))
    out = {}
    for (ds, k, j, scen) in sorted({a[:4] for a in R}):
        mus = np.array([R[(ds, k, j, scen, s)][0] for s in range(5)]); sds = np.array([R[(ds, k, j, scen, s)][1] for s in range(5)])
        mu = mus.mean(0); sd = np.sqrt((sds ** 2).mean(0) + mus.var(0))
        out[(ds, k, j, scen)] = dict(R=np.maximum(mu, 0), lo=np.maximum(mu - 1.645 * sd, 0), hi=mu + 1.645 * sd)
    pickle.dump(out, open('out/v7_inner_de.pkl', 'wb')); print('done', round(time.time() - t0), flush=True)
