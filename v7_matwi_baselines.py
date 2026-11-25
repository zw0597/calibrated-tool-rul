"""v1.4: model-based baselines on MATWI with the unchanged v1.2-v1.3 job functions.
  python v7_matwi_baselines.py -> out/v7_matwi_base.pkl  {'WienerC':{k:{scen:..}}, 'GP':{k:..}, 'GPC':{k:..}, 'HY':{k:..}}
"""
import os, pickle, time
from multiprocessing import Pool
os.environ['NT'] = '1'
def jw(k):
    import v6_wiener_cov as w; return w.job(('MATWI', k))[2]
def jg(k):
    import v5_baselines as vb; vb.COV = False; return vb.job_genpath(('MATWI', k))[2]
def jgc(k):
    import v5_baselines as vb; vb.COV = True; return vb.job_genpath(('MATWI', k))[2]
def jh(k):
    import v4_extra as v; return v.job_hybrid(('MATWI', k))[2]
if __name__ == '__main__':
    from lib import tags
    T = tags('MATWI'); t0 = time.time(); out = {}
    with Pool(int(os.environ.get('NPROC', '8'))) as p:
        for nm, fn in [('WienerC', jw), ('GP', jg), ('GPC', jgc), ('HY', jh)]:
            out[nm] = dict(zip(T, p.map(fn, T, chunksize=1))); print(nm, round(time.time() - t0), flush=True)
            pickle.dump(out, open('out/v7_matwi_base.pkl', 'wb'))
    print('done', flush=True)
