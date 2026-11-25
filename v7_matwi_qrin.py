"""v1.4: inner-LOTO QR+ ensembles on MATWI (same job as v5_plus.py nets), for the conformal step of CQR+.
  NPROC=6 python v7_matwi_qrin.py -> out/v7_matwi_plus_inner.pkl  (key (ds, scen, k, j, seed))
"""
import os, pickle, time
from multiprocessing import Pool
import v5_plus as vp
if __name__ == '__main__':
    os.environ['NT'] = '1'
    from lib import tags
    f = 'out/v7_matwi_plus_inner.pkl'
    R = pickle.load(open(f, 'rb')) if os.path.exists(f) else {}
    args = [('MATWI', sc, k, j, s) for sc in ['none', 'insp'] for k in tags('MATWI') for j in tags('MATWI') if j != k for s in range(vp.NIN)]
    args = [a for a in args if a not in R]; print('todo', len(args), flush=True); t0 = time.time()
    with Pool(int(os.environ.get('NPROC', '6'))) as p:
        for i, (key, val) in enumerate(p.imap_unordered(vp.job_net, args, chunksize=1)):
            R[key] = val
            if i % 20 == 0: pickle.dump(R, open(f, 'wb')); print(i, len(args), round(time.time() - t0), flush=True)
    pickle.dump(R, open(f, 'wb')); print('done', round(time.time() - t0), flush=True)
