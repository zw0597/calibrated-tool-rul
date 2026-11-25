"""v1.4: over-confidence law. For each outer test tool, the error model (kappa, psi, phi) identified on the inner folds
(without the test tool) predicts the coverage of the nominal interval of independent-error averaging of n windows:
   cov_n = 2 Phi( z / (kappa sqrt(rho_n)) ) - 1,  rho_n = (1-psi) + psi g_n(phi),  g_n = 1 + 2 sum_{h<n} (1-h/n) phi^h.
The prediction is compared with the empirical coverage on the outer tool's residuals (sliding blocks of n windows).
  python v7_law.py -> out/v7_law.pkl
"""
import pickle, numpy as np, sys
from scipy.stats import norm
from v2 import Nets, tags, D, emask, rul_true
Z = norm.ppf(0.95)
NS = [1, 2, 3, 5, 10, 20, 50, 100, 200]
def g(n, phi):
    h = np.arange(1, n); return 1 + 2 * np.sum((1 - h / n) * phi ** h)
def pred_cov(n, p, recal=False):
    rho = (1 - p['psi']) + p['psi'] * g(n, p['phi']); return 2 * norm.cdf(Z / ((1.0 if recal else p['kappa']) * np.sqrt(rho))) - 1
def emp(r, n, scale=1.0):
    if len(r) < n: return None
    c = np.convolve(r / scale, np.ones(n) / n, 'valid'); return np.abs(c) <= Z / np.sqrt(n)
if __name__ == '__main__':
    OUT = {}
    for ds in sys.argv[1:] or ['PHM', 'NUAA', 'MILL']:
        N = Nets(ds); T = tags(ds)
        for k in T:
            tr = [t for t in T if t != k]; pI, pR = N.params(k, tr); I, R = N.outer(k)
            w = D[k]['wte'] - D[k]['wte'][0]; m = emask(k)
            res = {'inc': ((I['z'] - w) / I['s'], pI), 'rul': (((R['z'] - rul_true(k)) / R['s'])[m], pR)}
            for src, (r, p) in res.items():
                for n in NS:
                    e = emp(r, n)
                    if e is None: continue
                    er = emp(r, n, p['kappa'])
                    OUT[(ds, k, src, n)] = dict(pred=pred_cov(n, p), emp=e.mean(), pred_rc=pred_cov(n, p, True), emp_rc=er.mean(), cnt=len(e), p=p)
        for src in ['inc', 'rul']:
            for n in NS:
                v = [OUT[x] for x in OUT if x[0] == ds and x[2] == src and x[3] == n]
                if v: print(ds, src, n, 'raw pred %.3f emp %.3f | recal pred %.3f emp %.3f (tools %d)' % (np.mean([a['pred'] for a in v]), np.mean([a['emp'] for a in v]), np.mean([a['pred_rc'] for a in v]), np.mean([a['emp_rc'] for a in v]), len(v)), flush=True)
    import os
    old = pickle.load(open('out/v7_law.pkl', 'rb')) if os.path.exists('out/v7_law.pkl') else {}
    old.update(OUT); pickle.dump(old, open('out/v7_law.pkl', 'wb'))
