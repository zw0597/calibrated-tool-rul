"""Particle-filter baseline with condition-dependent time scale (v1.3): the training wear paths are expressed on the
time axis t/theta_i with theta_i = T_i / mean(T); the path family is chosen and fitted on that axis; ln(theta) of the test
tool has the extended-Taylor (ridge, exact-LOO subset) prior on the cutting conditions and is a particle dimension."""
import pickle, numpy as np
from multiprocessing import Pool
from scipy.optimize import curve_fit
from v5_baselines import FAM, QS
def fit_scaled(t, fam, th):
    from pipeline import D, tfail
    f, _, bd, p0 = FAM[fam]; d = D[t]; m = (d['tte'] <= tfail(t)) & (d['tte'] > 0)
    x = d['tte'][m] / th; y = np.maximum(d['wte'][m] - d['wte'][0], 0)
    p, _ = curve_fit(f, x, y, p0=p0(x, y), bounds=bd, maxfev=20000); return p
def job(a):
    ds, k = a
    from v2 import Nets, tags, D, TH, INSP, SIG_I, tfail
    from mc import make_insp, theta_prior
    from pipeline import rul_true, emask
    T = tags(ds); tr = [t for t in T if t != k]; N = Nets(ds); pI, _ = N.params(k, tr); I, _ = N.outer(k); kap = pI['kappa']
    Tm = np.mean([tfail(t) for t in tr]); th = {t: tfail(t) / Tm for t in tr}
    fth, sth = theta_prior(tr, th, ds != 'PHM'); mth = fth(D[k]['c'])
    best = None                                                     # path family by LOO prior-RUL error on the training tools
    for fam in FAM:
        err = []
        for j in tr:
            tr2 = [t for t in tr if t != j]; Tm2 = np.mean([tfail(t) for t in tr2]); th2 = {t: tfail(t) / Tm2 for t in tr2}
            P = np.array([fit_scaled(t, fam, th2[t]) for t in tr2]); f2, _ = theta_prior(tr2, th2, ds != 'PHM')
            Tf = np.exp(f2(D[j]['c'])) * FAM[fam][1](max(TH[ds] - D[j]['wte'][0], 0), *P.mean(0)); m = emask(j)
            err.append(np.abs(np.maximum(Tf - D[j]['tte'], 0)[m] - rul_true(j)[m]).mean() / tfail(j))
        if best is None or np.mean(err) < best[0]: best = (np.mean(err), fam)
    fam = best[1]; f, finv, bd, _ = FAM[fam]
    P = np.array([fit_scaled(t, fam, th[t]) for t in tr]); mu = P.mean(0)
    C = (np.cov(P.T, ddof=1) if len(P) > 1 else np.zeros((2, 2))) + np.diag([0.1 ** 2, 0.05 ** 2])
    n = 20000; tt = D[k]['tte']; Tmax = 10 * max(tfail(j) for j in tr); out = {}
    for scen, idt, use_net in [('none', None, True), ('insp', INSP[ds], True), ('none', None, False), ('insp', INSP[ds], False)]:
        ins = make_insp(k, idt, SIG_I); rng = np.random.default_rng(0)
        X = np.c_[ins[0] + SIG_I * rng.standard_normal(n), rng.multivariate_normal(mu, C, n), mth + sth * rng.standard_normal(n)]
        clip = lambda X: np.c_[X[:, 0], np.clip(X[:, 1], bd[0][0], bd[1][0]), np.clip(X[:, 2], bd[0][1], bd[1][1]), X[:, 3]]
        X = clip(X); aLW = (3 * 0.99 - 1) / (2 * 0.99); h2 = 1 - aLW ** 2
        lw = np.zeros(n); R = np.zeros(len(tt)); lo = np.zeros(len(tt)); hi = np.zeros(len(tt))
        for i in range(len(tt)):
            sc = np.exp(X[:, 3]); vp = X[:, 0] + f(tt[i] / sc, X[:, 1], X[:, 2])
            if i > 0:
                if use_net: lw += -0.5 * (ins[0] + I['z'][i] - vp) ** 2 / (kap * I['s'][i]) ** 2
                if i in ins: lw += -0.5 * (ins[i] - vp) ** 2 / SIG_I ** 2
            lw = np.nan_to_num(lw, nan=-1e300, neginf=-1e300); lw -= lw.max(); w = np.exp(lw); w /= w.sum()
            Tf = np.nan_to_num(sc * finv(np.maximum(TH[ds] - X[:, 0], 0), X[:, 1], X[:, 2]), nan=Tmax, posinf=Tmax)
            Rs = np.clip(Tf - tt[i], 0, Tmax); o = np.argsort(Rs); c = np.cumsum(w[o])
            q = np.interp(QS, c, Rs[o]); R[i], lo[i], hi[i] = q[2], q[0], q[4]
            if 1.0 / np.sum(w ** 2) < n / 2:
                mbar = w @ X; V = np.cov(X.T, aweights=w) if np.sum(w > 0) > 4 else np.zeros((4, 4))
                if not np.all(np.isfinite(V)): V = np.zeros((4, 4))
                idx = rng.choice(n, n, p=w); M_ = aLW * X[idx] + (1 - aLW) * mbar
                V = 0.5 * (V + V.T); ev, U = np.linalg.eigh(V); L = U * np.sqrt(np.maximum(ev, 0) * h2)
                X = clip(M_ + rng.standard_normal((n, 4)) @ L.T); lw = np.zeros(n)
        out[scen if use_net else scen + '_insponly'] = dict(R=R, lo=lo, hi=hi, fam=fam)
    return ds, k, out
if __name__ == '__main__':
    from pipeline import tags
    with Pool(16) as p: res = p.map(job, [(ds, k) for ds in ['PHM', 'NUAA', 'MILL'] for k in tags(ds)], chunksize=1)
    O = {}
    for ds, k, o in res: O.setdefault(ds, {})[k] = o
    pickle.dump(O, open('out/v6_pf_cov.pkl', 'wb'))
    from metrics import tool_metrics
    for scen in ['insp_insponly', 'insp', 'none_insponly', 'none']:
        for ds in ['PHM', 'NUAA', 'MILL']:
            per = [tool_metrics(t, O[ds][t][scen]) for t in tags(ds)]
            print('PFC', scen, ds, {q: round(np.mean([p[q] for p in per]), 1) for q in ['nMAE', 'PICP', 'NMPIW', 'NIS']})
