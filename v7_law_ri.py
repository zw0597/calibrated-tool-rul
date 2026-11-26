"""v1.4 (review): alternative predictors of the coverage of independent-error averaging, all fitted on the inner folds
of each outer unit and evaluated on the outer unit's recalibrated residuals (same protocol as v7_law.py):
  AR   : Corollary 1, nu_n = (1-psi) + psi g_n(phi)              (moments at lags 1-2)
  RI   : random intercept + AR(1), nu_n = (1-psi0-psi1) + psi0 n + psi1 g_n(phi)   (least squares, lags 1-10)
  ACF  : non-parametric plug-in of the averaged inner autocorrelation, nu_n = 1 + 2 sum_{h<n} (1-h/n) rho_h (lags with >=3 sequences)
  IND  : independence, nu_n = 1
  python v7_law_ri.py -> out/v7_law_ri.pkl"""
import pickle, numpy as np
from scipy.stats import norm
from scipy.optimize import least_squares
from v7_law import emp, NS, g, Z
def acf_avg(pairs, L):
    A = []
    for e, s in pairs:
        z = e / np.maximum(s, 1e-9); v = np.mean(z * z)
        A.append([np.mean(z[l:] * z[:-l]) / v if len(z) > l + 1 else np.nan for l in range(1, L + 1)])
    A = np.array(A); cnt = np.sum(np.isfinite(A), 0); rho = np.nanmean(A, 0); rho[cnt < min(3, len(A))] = np.nan; return rho
def fit_ri(pairs):
    rho = acf_avg(pairs, 10); lags = np.arange(1, 11); ok = np.isfinite(rho)
    def res(x): p0, p1, ph = x; return p0 + p1 * ph ** lags[ok] - rho[ok] + 10 * max(0, p0 + p1 - 0.99)
    best = None
    for x0 in [(0.3, 0.5, 0.9), (0.6, 0.3, 0.8), (0.1, 0.8, 0.97)]:
        r = least_squares(res, x0, bounds=([0, 0, 0.5], [0.98, 0.98, 0.995]))
        if best is None or r.cost < best.cost: best = r
    return best.x
def preds(pairs, p):
    p0, p1, ph = fit_ri(pairs); rho = acf_avg(pairs, max(NS))
    out = {}
    for n in NS:
        nu_ar = (1 - p['psi']) + p['psi'] * g(n, p['phi'])
        nu_ri = (1 - p0 - p1) + p0 * n + p1 * g(n, ph)
        h = np.arange(1, n); r = rho[:n - 1] if n > 1 else np.array([])
        r = np.where(np.isfinite(r), r, 0.0) if n > 1 else r          # unavailable long lags -> 0 (optimistic)
        nu_acf = 1 + 2 * np.sum((1 - h / n) * r) if n > 1 else 1.0
        out[n] = {m: 2 * norm.cdf(Z / np.sqrt(max(nu, 1e-6))) - 1 for m, nu in [('AR', nu_ar), ('RI', nu_ri), ('ACF', nu_acf), ('IND', 1.0)]}
    return out, (p0, p1, ph)
OUT = {}
from v2 import Nets, tags, D, emask, rul_true
from pipeline import mom
for ds in ['PHM', 'NUAA', 'MILL', 'MATWI']:
    N = Nets(ds); T = tags(ds)
    for k in T:
        tr = [t for t in T if t != k]; pi = []; pr = []
        for j in tr:
            I, Rr = N.inner(k, j); m = emask(j); w = D[j]['wte'] - D[j]['wte'][0]
            pi.append((I['z'] - w, I['s'])); pr.append(((Rr['z'] - rul_true(j))[m], Rr['s'][m]))
        I, R = N.outer(k); w = D[k]['wte'] - D[k]['wte'][0]; m = emask(k)
        for src, pairs, r in [('inc', pi, (I['z'] - w) / I['s']), ('rul', pr, ((R['z'] - rul_true(k)) / R['s'])[m])]:
            p = mom(pairs); pc, ri = preds(pairs, p)
            for n in NS:
                e = emp(r, n, p['kappa'])
                if e is None: continue
                OUT[(ds, k, src, n)] = dict(emp=e.mean(), ri=ri, **pc[n])
import cmapss_study as C
R = pickle.load(open('out/cmapss_nets.pkl', 'rb'))
for f, te in enumerate(C.OUTER):
    tr = [u for u in C.UNITS if u not in te]; inner = [sorted(tr[i::4]) for i in range(4)]; pairs = []
    for gg, te2 in enumerate(inner):
        for u in te2:
            q = C.ens_qr(R, ('inner', f, gg), u); _, cyc, Rt = C.windows(u, 0, 1); s = np.maximum((q[:, 4] - q[:, 0]) / 3.29, 1.0); pairs.append((q[:, 2] - Rt, s))
    p = C.mom(pairs); pc, ri = preds(pairs, p)
    for u in te:
        q = C.ens_qr(R, ('outer', f), u); _, cyc, Rt = C.windows(u, 0, 1); s = np.maximum((q[:, 4] - q[:, 0]) / 3.29, 1.0); r = (q[:, 2] - Rt) / s
        for n in NS:
            e = emp(r, n, p['kappa'])
            if e is not None: OUT[('CMAPSS', u, 'rul', n)] = dict(emp=e.mean(), ri=ri, **pc[n])
import scidata_study as S
R = pickle.load(open('out/scidata_nets.pkl', 'rb'))
for k in S.UNITS:
    tr = [u for u in S.UNITS if u != k]; qens = lambda kp, u: np.mean([R[kp + ('qr', s)][u] for s in range(5)], 0); pairs = []
    for j in tr:
        q = qens(('inner', k, j), j); s = np.maximum((q[:, 4] - q[:, 0]) / 3.29, 0.5); pairs.append((q[:, 2] - S.U[j]['R'], s))
    p = C.mom(pairs); pc, ri = preds(pairs, p); q = qens(('outer', k), k); s = np.maximum((q[:, 4] - q[:, 0]) / 3.29, 0.5); r = (q[:, 2] - S.U[k]['R']) / s
    for n in NS:
        e = emp(r, n, p['kappa'])
        if e is not None: OUT[('SCIDATA', k, 'rul', n)] = dict(emp=e.mean(), ri=ri, **pc[n])
pickle.dump(OUT, open('out/v7_law_ri.pkl', 'wb'))
# summary over dataset-source-n combinations with n>=2 available for >= half the units (as Fig. 9)
rows = {}
for (ds, u, src, n), v in OUT.items(): rows.setdefault((ds, src, n), []).append(v)
S_ = {}
for (ds, src), _ in {(a, b): 0 for a, b, c in rows}.items():
    mx = max(len(v) for (a, b, c), v in rows.items() if a == ds and b == src)
    for (a, b, n), v in rows.items():
        if a != ds or b != src or n < 2 or len(v) < max(3, 0.5 * mx): continue
        e = np.mean([x['emp'] for x in v])
        for mth in ['AR', 'RI', 'ACF', 'IND']: S_.setdefault(ds, {}).setdefault(mth, []).append(abs(np.mean([x[mth] for x in v]) - e))
allm = {m: np.concatenate([S_[d][m] for d in S_]) for m in ['AR', 'RI', 'ACF', 'IND']}
print('ALL', {m: round(float(np.mean(v)), 3) for m, v in allm.items()}, 'n', len(allm['AR']))
for d in S_: print(d, {m: round(float(np.mean(v)), 3) for m, v in S_[d].items()}, len(S_[d]['AR']))
