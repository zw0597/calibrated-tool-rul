"""v1.4 (review): Wiener-AR, the minimal bias-aware hybrid. Wiener-Net (Kalman filter on wear level and drift with the
network wear estimate as measurement) augmented with the AR(1) bias state of the error model (Bryson-Henrikson state
augmentation), with (kappa_I, phi_I, psi_I) from the nested inner folds, followed by the same two-layer conformal step
(inner scores from inner Wiener-AR runs on held-out training tools).
  NPROC=6 python v7_wiener_ar.py -> out/v7_wiener_ar.pkl"""
import os, sys, pickle, numpy as np
from multiprocessing import Pool
from v2 import *
from v4_extra import wiener_prior, ig_samples
from pipeline import conformal_Q, cqr_score, cqr_apply


def wiener_ar(k, tr, ds, insp_dt, I, p, ns=2000):
    mu0, s_mu, sB2 = wiener_prior(tr, ds); ins = make_insp(k, insp_dt, SIG_I)
    kap, phi, psi = p['kappa'], p['phi'], p['psi']
    tt = D[k]['tte']; K = len(tt); Vs = TH[ds]; rng = np.random.default_rng(0)
    x = np.array([ins[0], mu0, 0.0]); P = np.diag([SIG_I ** 2, s_mu ** 2, kap ** 2 * psi])
    R = np.zeros(K); lo = np.zeros(K); hi = np.zeros(K)
    for i in range(K):
        if i > 0:
            dt = tt[i] - tt[i - 1]; F = np.array([[1, dt, 0], [0, 1, 0], [0, 0, phi]])
            x = F @ x; P = F @ P @ F.T + np.diag([sB2 * dt, 0.0, kap ** 2 * psi * (1 - phi ** 2)])
            s = I['s'][i]
            obs = [(ins[0] + I['z'][i], np.array([1.0, 0.0, s]), kap ** 2 * (1 - psi) * s ** 2)]
            if i in ins: obs.append((ins[i], np.array([1.0, 0.0, 0.0]), SIG_I ** 2))
            for z, Hh, r in obs:
                S = Hh @ P @ Hh + r; Kg = P @ Hh / S
                x = x + Kg * (z - Hh @ x); P = P - np.outer(Kg, Hh @ P); P = (P + P.T) / 2
        smp = rng.multivariate_normal(x[:2], P[:2, :2], ns); v = smp[:, 0]; mus = np.maximum(smp[:, 1], 1e-8)
        f = ig_samples(np.maximum(Vs - v, 0), mus, sB2, rng)
        R[i] = np.median(f); lo[i], hi[i] = np.quantile(f, [0.05, 0.95])
    return dict(R=R, lo=lo, hi=hi)


def job(arg):
    ds, k = arg; T = tags(ds); tr = [t for t in T if t != k]; N = Nets(ds, qens=True)
    pI, _ = N.params(k, tr); I, _ = N.outer(k); out = {}
    for scen, idt in [('none', None), ('insp', INSP[ds])]:
        o = wiener_ar(k, tr, ds, idt, I, pI); sc = []
        for j in tr:
            tr2 = [x for x in tr if x != j]; pIj, _ = N.params(k, tr2); Ij, _ = N.inner(k, j)
            oj = wiener_ar(j, tr2, ds, idt, Ij, pIj); m = emask(j)
            sc.append(cqr_score(rul_true(j)[m], oj['lo'][m], oj['hi'][m]))
        Q = conformal_Q(sc); lo, hi = cqr_apply(np.maximum(o['lo'], 0), o['hi'], Q)
        out[scen] = {'WienerAR': o, 'WienerARCP': dict(R=o['R'], lo=lo, hi=hi, Q=Q)}
    return ds, k, out


if __name__ == '__main__':
    DSS = sys.argv[1:] or ['PHM', 'NUAA', 'MILL', 'MATWI']
    args = [(ds, k) for ds in DSS for k in tags(ds)]; OUT = {}
    with Pool(int(os.environ.get('NPROC', '6'))) as p:
        for ds, k, o in p.imap_unordered(job, args):
            OUT.setdefault(ds, {})[k] = o; print(ds, k, flush=True)
    f = 'out/v7_wiener_ar.pkl'; old = pickle.load(open(f, 'rb')) if os.path.exists(f) else {}
    for ds in OUT: old[ds] = OUT[ds]
    pickle.dump(old, open(f, 'wb'))
    from metrics import tool_metrics
    for ds in OUT:
        for m in ['WienerAR', 'WienerARCP']:
            per = [tool_metrics(t, OUT[ds][t]['insp'][m]) for t in tags(ds)]
            print(ds, m, {c: round(float(np.nanmean([q[c] for q in per])), 1) for c in ['nMAE', 'PICP', 'NMPIW', 'NIS']})
