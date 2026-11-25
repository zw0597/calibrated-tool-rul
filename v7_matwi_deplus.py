"""v1.4: DE+ and QR+ networks on MATWI (same job as v5_baselines.py nets), written to a separate file.
  NPROC=8 python v7_matwi_deplus.py -> out/v7_matwi_deepaux_raw.pkl, out/v7_matwi_deepaux.pkl
"""
import os, sys, pickle, time
from multiprocessing import Pool
import v5_baselines as vb
vb.DSS = ['MATWI']
if __name__ == '__main__':
    os.environ['NT'] = '1'
    from lib import tags
    f = 'out/v7_matwi_deepaux_raw.pkl'
    R = pickle.load(open(f, 'rb')) if os.path.exists(f) else {}
    args = [('MATWI', k, sc, kind, s) for k in tags('MATWI') for sc in ['none', 'insp'] for kind in ['de', 'qr'] for s in range(5)]
    args = [a for a in args if a not in R]; print('todo', len(args), flush=True); t0 = time.time()
    with Pool(int(os.environ.get('NPROC', '8'))) as p:
        for i, (key, val) in enumerate(p.imap_unordered(vb.job_net, args, chunksize=1)):
            R[key] = val
            if i % 20 == 0: pickle.dump(R, open(f, 'wb')); print(i, len(args), round(time.time() - t0), flush=True)
    pickle.dump(R, open(f, 'wb')); pickle.dump(vb.assemble_nets(R), open('out/v7_matwi_deepaux.pkl', 'wb')); print('done', round(time.time() - t0), flush=True)
