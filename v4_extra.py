"""Additional analyses of the revised manuscript (v4).

  python v4_extra.py hybrid      # Wiener-Net hybrid baseline (network wear evidence as independent measurements)
  python v4_extra.py ablate      # extra ablations: predictive scale, pseudo-BMA variants (periodic inspections, raw)
  python v4_extra.py sens        # sensitivity to fixed constants (s_theta floor, RUL-scale floor, psi clip, sigma_I)
  python v4_extra.py t0          # sensitivity of the conformal step to T0 (needs inner fusions)
All outputs go to out/v4_*.pkl; tables are written by make_tables_v4.py.
"""
import sys, pickle, time, itertools, numpy as np
from multiprocessing import Pool
import v2, pipeline
from v2 import *

DSS = ['PHM', 'NUAA', 'MILL']


# ---------------------------------------------------------------- Wiener-Net hybrid baseline
def wiener_prior(tr, ds):
    """drift prior and diffusion of a Wiener wear model, estimated from the training tools (as in the Wiener baseline)"""
    rates = []; sb = []
    for j in tr:
        m = D[j]['tte'] <= tfail(j); x = D[j]['tte'][m]; y = D[j]['wte'][m]
        b = (y[-1] - y[0]) / (x[-1] - x[0]); rates.append(b)
        dy = np.diff(y); dx = np.diff(x); sb.append(np.var(dy - b * dx) / np.mean(dx))
    mu0 = np.mean(rates); s_mu = max(np.std(rates, ddof=1) if len(rates) > 1 else 0.3 * mu0, 0.15 * mu0)
    return mu0, s_mu, np.mean(sb)


def ig_samples(dlt, mus, sB2, rng):
    """first-passage (inverse-Gaussian) samples for distance dlt and drift mus"""
    n = len(mus); lam = dlt ** 2 / sB2; mean = dlt / mus
    y_ = rng.standard_normal(n) ** 2
    xx = mean + mean ** 2 * y_ / (2 * lam) - mean / (2 * lam) * np.sqrt(4 * mean * lam * y_ + mean ** 2 * y_ ** 2)
    zz = rng.random(n); smp = np.where(zz <= mean / (mean + xx), xx, mean ** 2 / np.maximum(xx, 1e-12))
    smp[dlt <= 0] = 0
    return smp


def wiener_net(k, tr, ds, insp_dt, I, kappa, ns=2000):
    """Kalman filter on (wear, drift) of a Wiener model; observations: inspections (sigma_I) and the network
    wear estimate z0+y^I_k, treated as independent measurements with variance (kappa*s_k)^2."""
    mu0, s_mu, sB2 = wiener_prior(tr, ds); ins = make_insp(k, insp_dt, SIG_I)
    tt = D[k]['tte']; K = len(tt); Vs = TH[ds]; rng = np.random.default_rng(0)
    x = np.array([ins[0], mu0]); P = np.diag([SIG_I ** 2, s_mu ** 2])
    R = np.zeros(K); lo = np.zeros(K); hi = np.zeros(K)
    for i in range(K):
        if i > 0:
            dt = tt[i] - tt[i - 1]; F = np.array([[1, dt], [0, 1]])
            x = F @ x; P = F @ P @ F.T + np.diag([sB2 * dt, 0.0])
            obs = [(ins[0] + I['z'][i], (kappa * I['s'][i]) ** 2)]
            if i in ins: obs.append((ins[i], SIG_I ** 2))
            for z, r in obs:
                Hh = np.array([1.0, 0.0]); S = Hh @ P @ Hh + r; Kg = P @ Hh / S
                x = x + Kg * (z - x[0]); P = P - np.outer(Kg, Hh @ P)
        smp = rng.multivariate_normal(x, P, ns); v = smp[:, 0]; mus = np.maximum(smp[:, 1], 1e-8)
        f = ig_samples(np.maximum(Vs - v, 0), mus, sB2, rng)
        R[i] = np.median(f); lo[i], hi[i] = np.quantile(f, [0.05, 0.95])
    return dict(R=R, lo=lo, hi=hi)


def job_hybrid(arg):
    ds, k = arg; T = tags(ds); tr = [t for t in T if t != k]; N = Nets(ds, qens=True)
    pI, pR = N.params(k, tr); I, _ = N.outer(k); out = {}
    for scen, idt in [('none', None), ('insp', INSP[ds])]:
        out[scen] = {'WienerNet': wiener_net(k, tr, ds, idt, I, 1.0),
                     'WienerNetK': wiener_net(k, tr, ds, idt, I, pI['kappa'])}
    return ds, k, out


# ---------------------------------------------------------------- framework variants
def set_cfg(cfg):
    """apply a configuration to the framework modules (defaults restore the paper settings)"""
    v2._law_cache.clear(); v2._J.clear()
    v2.PREDSCALE = cfg.get('predscale', True); v2.PBMA_MODE = cfg.get('pbma', 'bb')
    sf = cfg.get('s_floor', 0.15)
    v2.theta_prior = (lambda train, th, uc, s_floor=sf, **kw: theta_prior_orig(train, th, uc, s_floor=s_floor, **kw))
    rf = cfg.get('rfloor', 5.0)
    v2.rul_src = lambda q, rf=rf: dict(z=q[:, 2], s=np.maximum((q[:, 4] - q[:, 0]) / 3.29, rf))
    pc = cfg.get('psimax', 0.98)
    v2.mom = lambda pairs, pc=pc: mom_clip(pairs, pc)
    v2.SIG_I = cfg.get('sigI', 0.005)


from mc import theta_prior as theta_prior_orig


def mom_clip(pairs, psimax):
    z2 = []; a1 = []; a2 = []
    for e, s in pairs:
        if len(e) < 3: continue
        z = e / np.maximum(s, 1e-9); v = np.mean(z * z); z2.append(v)
        a1.append(np.mean(z[1:] * z[:-1]) / v); a2.append(np.mean(z[2:] * z[:-2]) / v)
    kappa = float(np.sqrt(np.mean(z2))); r1 = float(np.mean(a1)); r2 = float(np.mean(a2))
    phi = float(np.clip(r2 / max(r1, 1e-3), 0.5, 0.995)); psi = float(np.clip(r1 / phi, 0.05, psimax))
    return dict(kappa=kappa, phi=phi, psi=psi)


ABL = {'ref': {}, 'woPred': dict(predscale=False), 'plainP': dict(pbma='plain'), 'uniP': dict(pbma='uniform')}
SENS = {'ref': {},
        'sfl010': dict(s_floor=0.10), 'sfl025': dict(s_floor=0.25),
        'rfl2': dict(rfloor=2.0), 'rfl10': dict(rfloor=10.0),
        'psi095': dict(psimax=0.95), 'psi0995': dict(psimax=0.995),
        'sig25': dict(sigI=0.0025), 'sig10': dict(sigI=0.010)}


def job_variant(arg):
    name, cfg, ds, k, scen = arg; set_cfg(cfg)
    T = tags(ds); tr = [t for t in T if t != k]; N = Nets(ds, qens=True)
    S, _ = N.outer_srcs(k, tr); idt = INSP[ds] if scen == 'insp' else None
    o = run_v2(k, tr, S, idt)
    return name, ds, k, scen, dict(R=o['R'], lo=o['Rq'][:, 0], hi=o['Rq'][:, 4])


def job_inner(arg):
    ds, k, j = arg; set_cfg({})
    T = tags(ds); tr = [t for t in T if t != k]; tr2 = [x for x in tr if x != j]; N = Nets(ds, qens=True)
    oj = run_v2(j, tr2, N.inner_srcs(k, j, tr), INSP[ds])
    return ds, k, j, oj['Rq']


def pool_run(fn, args, nproc=16):
    with Pool(nproc) as p:
        return p.map(fn, args, chunksize=1)


if __name__ == '__main__':
    what = sys.argv[1]; t0 = time.time()
    if what == 'hybrid':
        args = [(ds, k) for ds in DSS for k in tags(ds)]
        res = {}
        for ds, k, o in pool_run(job_hybrid, args): res.setdefault(ds, {})[k] = o
        pickle.dump(res, open('out/v4_hybrid.pkl', 'wb'))
    elif what in ('ablate', 'sens'):
        V = ABL if what == 'ablate' else SENS
        scens = ['insp'] if what == 'sens' else ['insp', 'none']
        args = [(n, c, ds, k, sc) for n, c in V.items() for ds in DSS for k in tags(ds) for sc in scens
                if not (c.get('pbma') and ds == 'PHM')]
        res = {}
        for n, ds, k, sc, o in pool_run(job_variant, args): res.setdefault(n, {}).setdefault(ds, {}).setdefault(sc, {})[k] = o
        pickle.dump(res, open(f'out/v4_{what}.pkl', 'wb'))
    elif what == 't0':
        args = [(ds, k, j) for ds in DSS for k in tags(ds) for j in tags(ds) if j != k]
        res = {}
        for ds, k, j, Rq in pool_run(job_inner, args): res.setdefault(ds, {}).setdefault(k, {})[j] = Rq
        pickle.dump(res, open('out/v4_inner_insp.pkl', 'wb'))
    print(what, 'done in', round(time.time() - t0), 's')
