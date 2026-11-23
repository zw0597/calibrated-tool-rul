"""Additional baselines of the revised manuscript (v1.2).

  python v5_baselines.py nets      # DE+ and QR+: deep regressors that also receive the mounting wear and the inspections
  python v5_baselines.py genpath   # PF-Net: power-law general-path model, Liu-West particle filter with network evidence
Outputs: out/v5_deepaux.pkl, out/v5_genpath.pkl
"""
import os, sys, pickle, time, numpy as np
from multiprocessing import Pool

QS = [0.05, 0.25, 0.5, 0.75, 0.95]
COV = os.environ.get('PFCOV', '0') == '1'
DSS = ['PHM', 'NUAA', 'MILL']


# ------------------------------------------------------------------ DE+ / QR+
def job_net(a):
    ds, k, scen, kind, seed = a
    import lib
    from pipeline import INSP
    lib.AUX = ('insp', INSP[ds]) if scen == 'insp' else ('none', None)
    T = lib.tags(ds); tr = [t for t in T if t != k]
    if kind == 'de':
        net = lib.train(tr, lib.rul_target, seed=seed); return (ds, k, scen, kind, seed), lib.predict(net, k)
    net = lib.train(tr, lib.rul_target, seed=seed, loss='q', qs=QS); return (ds, k, scen, kind, seed), lib.predict(net, k)


def assemble_nets(R):
    from pipeline import tags
    out = {}
    for ds in DSS:
        for k in tags(ds):
            for scen in ['none', 'insp']:
                mus = np.array([R[(ds, k, scen, 'de', s)][0] for s in range(5)]); sds = np.array([R[(ds, k, scen, 'de', s)][1] for s in range(5)])
                mu = mus.mean(0); sd = np.sqrt((sds ** 2).mean(0) + mus.var(0))
                q = np.mean([R[(ds, k, scen, 'qr', s)] for s in range(5)], 0)
                out.setdefault(ds, {}).setdefault(k, {})[scen] = {
                    'DE+': dict(R=np.maximum(mu, 0), lo=np.maximum(mu - 1.645 * sd, 0), hi=mu + 1.645 * sd),
                    'QR+': dict(R=np.maximum(q[:, 2], 0), lo=q[:, 0], hi=q[:, 4])}
    return out


# ------------------------------------------------------------------ GenPath-Net
FAM = {  # increment path Delta(t; p1, p2), its inverse t(Delta), parameter bounds, start values
    'pow':  (lambda x, p1, p2: np.exp(p1) * x ** p2,
             lambda dv, p1, p2: (dv / np.exp(p1)) ** (1 / p2), ([-40, 0.2], [5, 5]),
             lambda x, y: [np.log(max(y[-1], 1e-4) / x[-1]), 1.0]),
    'exp':  (lambda x, p1, p2: np.exp(p1) * np.expm1(np.exp(p2) * x),
             lambda dv, p1, p2: np.log1p(dv / np.exp(p1)) / np.exp(p2), ([-40, -30], [5, 5]),
             lambda x, y: [np.log(max(y[-1], 1e-4)), np.log(1 / x[-1])]),
    'quad': (lambda x, p1, p2: np.exp(p1) * x + np.exp(p2) * x ** 2,
             lambda dv, p1, p2: (-np.exp(p1) + np.sqrt(np.exp(2 * p1) + 4 * np.exp(p2) * dv)) / (2 * np.exp(p2)), ([-40, -60], [5, 5]),
             lambda x, y: [np.log(max(y[-1], 1e-4) / x[-1] / 2), np.log(max(y[-1], 1e-4) / x[-1] ** 2 / 2)])}


def fit_path(t, fam='pow'):
    from scipy.optimize import curve_fit
    from pipeline import D, tfail
    f, _, bd, p0 = FAM[fam]
    d = D[t]; m = (d['tte'] <= tfail(t)) & (d['tte'] > 0); x = d['tte'][m]; y = np.maximum(d['wte'][m] - d['wte'][0], 0)
    p, _ = curve_fit(f, x, y, p0=p0(x, y), bounds=bd, maxfev=20000)
    return p


def select_family(tr, ds):
    """path family with the smallest leave-one-tool-out error of the prior-mean RUL on the training tools"""
    from pipeline import D, TH, tfail, rul_true, emask
    best = None
    for fam in FAM:
        err = []
        for j in tr:
            P = np.array([fit_path(t, fam) for t in tr if t != j]); p = P.mean(0)
            Tf = FAM[fam][1](max(TH[ds] - D[j]['wte'][0], 0), *p); m = emask(j)
            err.append(np.abs(np.maximum(Tf - D[j]['tte'], 0)[m] - rul_true(j)[m]).mean() / tfail(j))
        e = np.mean(err)
        if best is None or e < best[0]: best = (e, fam)
    return best[1]


def job_genpath(a):
    ds, k = a
    from v2 import Nets, tags, D, TH, INSP, SIG_I, tfail
    from mc import make_insp
    T = tags(ds); tr = [t for t in T if t != k]; N = Nets(ds)
    pI, _ = N.params(k, tr); I, _ = N.outer(k); kap = pI['kappa']
    fam = select_family(tr, ds); f, finv, bd, _ = FAM[fam]
    P = np.array([fit_path(t, fam) for t in tr]); mu = P.mean(0)
    C = (np.cov(P.T, ddof=1) if len(P) > 1 else np.zeros((2, 2))) + np.diag([0.1 ** 2, 0.05 ** 2])
    if COV and ds != 'PHM':   # v1.3: condition-dependent prior on the scale parameter (same ridge model as the extended-Taylor prior)
        from mc import theta_prior
        fp, _ = theta_prior(tr, {t: float(np.exp(p[0])) for t, p in zip(tr, P)}, True)
        r = np.array([p[0] - fp(D[t]['c']) for t, p in zip(tr, P)])
        mu = np.array([fp(D[k]['c']), mu[1]]); C = np.cov(np.c_[r, P[:, 1]].T, ddof=1) + np.diag([0.1 ** 2, 0.05 ** 2])
    n = 20000
    tt = D[k]['tte']; Tmax = 10 * max(tfail(j) for j in tr); out = {}
    for scen, idt, use_net in [('none', None, True), ('insp', INSP[ds], True), ('none', None, False), ('insp', INSP[ds], False)]:
        # particle filter with Liu-West kernel rejuvenation of the static parameters (discount 0.99)
        ins = make_insp(k, idt, SIG_I); rng = np.random.default_rng(0)
        X = np.c_[ins[0] + SIG_I * rng.standard_normal(n), rng.multivariate_normal(mu, C, n)]   # (v0, ln a, b)
        clip = lambda X: np.c_[X[:, 0], np.clip(X[:, 1], bd[0][0], bd[1][0]), np.clip(X[:, 2], bd[0][1], bd[1][1])]
        X = clip(X)
        aLW = (3 * 0.99 - 1) / (2 * 0.99); h2 = 1 - aLW ** 2
        lw = np.zeros(n); R = np.zeros(len(tt)); lo = np.zeros(len(tt)); hi = np.zeros(len(tt))
        for i in range(len(tt)):
            vp = X[:, 0] + f(tt[i], X[:, 1], X[:, 2])
            if i > 0:
                if use_net: r2 = (kap * I['s'][i]) ** 2; lw += -0.5 * (ins[0] + I['z'][i] - vp) ** 2 / r2
                if i in ins: lw += -0.5 * (ins[i] - vp) ** 2 / SIG_I ** 2
            lw = np.nan_to_num(lw, nan=-1e300, neginf=-1e300); lw -= lw.max(); w = np.exp(lw); w /= w.sum()
            Tf = np.nan_to_num(finv(np.maximum(TH[ds] - X[:, 0], 0), X[:, 1], X[:, 2]), nan=Tmax, posinf=Tmax)
            Rs = np.clip(Tf - tt[i], 0, Tmax); o = np.argsort(Rs); c = np.cumsum(w[o])
            q = np.interp(QS, c, Rs[o]); R[i], lo[i], hi[i] = q[2], q[0], q[4]
            if 1.0 / np.sum(w ** 2) < n / 2:                       # resample and rejuvenate
                mbar = w @ X; V = np.cov(X.T, aweights=w) if np.sum(w > 0) > 3 else np.zeros((3, 3))
                if not np.all(np.isfinite(V)): V = np.zeros((3, 3))
                idx = rng.choice(n, n, p=w); M_ = aLW * X[idx] + (1 - aLW) * mbar
                V = 0.5 * (V + V.T); ev, U = np.linalg.eigh(V); L = U * np.sqrt(np.maximum(ev, 0) * h2)
                X = clip(M_ + rng.standard_normal((n, 3)) @ L.T); lw = np.zeros(n)
        out[scen if use_net else scen + '_insponly'] = dict(R=R, lo=lo, hi=hi, fam=fam)
    return ds, k, out


if __name__ == '__main__':
    os.environ['NT'] = '1'
    what = sys.argv[1]; t0 = time.time()
    from lib import tags
    if what == 'nets':
        args = [(ds, k, sc, kind, s) for ds in DSS for k in tags(ds) for sc in ['none', 'insp'] for kind in ['de', 'qr'] for s in range(5)]
        R = {}
        with Pool(int(os.environ.get('NPROC', '17'))) as p:
            for i, (key, val) in enumerate(p.imap_unordered(job_net, args, chunksize=1)):
                R[key] = val
                if i % 50 == 0: print(i, len(args), round(time.time() - t0), flush=True)
        pickle.dump(R, open('out/v5_deepaux_raw.pkl', 'wb')); pickle.dump(assemble_nets(R), open('out/v5_deepaux.pkl', 'wb'))
    elif what == 'genpath':
        args = [(ds, k) for ds in DSS for k in tags(ds)]
        with Pool(16) as p: res = p.map(job_genpath, args, chunksize=1)
        out = {}
        for ds, k, o in res: out.setdefault(ds, {})[k] = o
        pickle.dump(out, open('out/v6_genpath_cov.pkl' if COV else 'out/v5_genpath.pkl', 'wb'))
    print(what, 'done', round(time.time() - t0))
