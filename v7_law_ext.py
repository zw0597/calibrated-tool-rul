"""v1.4: over-confidence law on C-MAPSS FD001 and the independent tool data (out-of-sample: error model from the
inner folds, coverage on the outer units).  python v7_law_ext.py -> out/v7_law_ext.pkl"""
import pickle, numpy as np
from v7_law import pred_cov, emp, NS
OUT = {}
import cmapss_study as C
R = pickle.load(open('out/cmapss_nets.pkl', 'rb'))
for f, te in enumerate(C.OUTER):
    tr = [u for u in C.UNITS if u not in te]; inner = [sorted(tr[i::4]) for i in range(4)]
    pairs = []
    for g, te2 in enumerate(inner):
        for u in te2:
            q = C.ens_qr(R, ('inner', f, g), u); _, cyc, Rt = C.windows(u, 0, 1); s = np.maximum((q[:, 4] - q[:, 0]) / 3.29, 1.0); pairs.append((q[:, 2] - Rt, s))
    p = C.mom(pairs)
    for u in te:
        q = C.ens_qr(R, ('outer', f), u); _, cyc, Rt = C.windows(u, 0, 1); s = np.maximum((q[:, 4] - q[:, 0]) / 3.29, 1.0); r = (q[:, 2] - Rt) / s
        for n in NS:
            e = emp(r, n)
            if e is None: continue
            OUT[('CMAPSS', u, 'rul', n)] = dict(pred=pred_cov(n, p), emp=e.mean(), pred_rc=pred_cov(n, p, True), emp_rc=emp(r, n, p['kappa']).mean(), cnt=len(e), p=p)
import scidata_study as S
R = pickle.load(open('out/scidata_nets.pkl', 'rb'))
for k in S.UNITS:
    tr = [u for u in S.UNITS if u != k]
    qens = lambda kp, u: np.mean([R[kp + ('qr', s)][u] for s in range(5)], 0)
    pairs = []
    for j in tr:
        q = qens(('inner', k, j), j); s = np.maximum((q[:, 4] - q[:, 0]) / 3.29, 0.5); pairs.append((q[:, 2] - S.U[j]['R'], s))
    p = C.mom(pairs); q = qens(('outer', k), k); s = np.maximum((q[:, 4] - q[:, 0]) / 3.29, 0.5); r = (q[:, 2] - S.U[k]['R']) / s
    for n in NS:
        e = emp(r, n)
        if e is None: continue
        OUT[('SCIDATA', k, 'rul', n)] = dict(pred=pred_cov(n, p), emp=e.mean(), pred_rc=pred_cov(n, p, True), emp_rc=emp(r, n, p['kappa']).mean(), cnt=len(e), p=p)
for ds in ['CMAPSS', 'SCIDATA']:
    for n in NS:
        v = [OUT[x] for x in OUT if x[0] == ds and x[3] == n]
        if v: print(ds, n, 'raw pred %.3f emp %.3f | recal pred %.3f emp %.3f (units %d)' % tuple([np.mean([a[c] for a in v]) for c in ['pred', 'emp', 'pred_rc', 'emp_rc']] + [len(v)]))
pickle.dump(OUT, open('out/v7_law_ext.pkl', 'wb'))
