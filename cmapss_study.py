"""Generality check on a non-tool benchmark (v1.2): NASA C-MAPSS FD001 turbofan engines.

Question: are deep RUL-network errors on unseen units persistent, and does the persistent-bias treatment with
unit-level conformal calibration (Sections 4.3-4.6 of the paper) improve calibration over independent-error
fusion and window-level conformal prediction?

Protocol: 100 run-to-failure training engines, 5-fold cross-validation by engine (outer), 4-fold nested CV
inside every outer training set (inner) for the error model and the conformal scores. Networks: five-member
Gaussian ensemble (DE) and five-member quantile ensemble (QR) of a small 1D-CNN on 30-cycle windows of
14 sensors; target = true (uncapped) RUL.
  python cmapss_study.py nets     # trains all networks              -> out/cmapss_nets.pkl
  python cmapss_study.py eval     # fusion, calibration and metrics  -> out/cmapss_eval.pkl
"""
import os, sys, pickle, time, numpy as np
from multiprocessing import Pool

DATA = os.environ.get('CMAPSS_DIR', 'data/cmapss')
SENS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]          # informative sensors of FD001
L = 30; QS = [0.05, 0.25, 0.5, 0.75, 0.95]; ALPHA = 0.1; TAU = 10.0  # TAU: time constant of the log score (cycles)


def load():
    a = np.loadtxt(f'{DATA}/train_FD001.txt'); U = {}
    for u in np.unique(a[:, 0]).astype(int):
        r = a[a[:, 0] == u]; X = r[:, [4 + s for s in SENS]]; U[u] = dict(X=X, T=len(r))   # sensor s -> column 4+s
    return U


U = load(); UNITS = sorted(U)
rng0 = np.random.default_rng(0); perm = rng0.permutation(UNITS)
OUTER = [sorted(perm[i::5].tolist()) for i in range(5)]


def windows(u, mu, sd):
    X = (U[u]['X'] - mu) / sd; T = U[u]['T']
    idx = np.arange(L, T + 1)                                   # window ends at cycle idx (1-based)
    W = np.stack([X[i - L:i] for i in idx]).astype(np.float32)
    return W, idx.astype(float), (T - idx).astype(float)       # windows, cycle, true RUL


def train_net(tr, kind, seed, ep=30):
    import torch, torch.nn as nn
    torch.set_num_threads(1); torch.manual_seed(seed); np.random.seed(seed)
    Xa = np.concatenate([U[u]['X'] for u in tr]); mu = Xa.mean(0); sd = Xa.std(0) + 1e-6
    Ws, Ys = zip(*[(w, y) for w, _, y in (windows(u, mu, sd) for u in tr)])
    X = torch.tensor(np.concatenate(Ws)); Y = torch.tensor(np.concatenate(Ys), dtype=torch.float32)
    ym, ys = float(Y.mean()), float(Y.std()); Yn = (Y - ym) / ys
    nout = 2 if kind == 'de' else len(QS)
    net = nn.Sequential(nn.Conv1d(len(SENS), 32, 5, padding=2), nn.GELU(), nn.Conv1d(32, 32, 5, padding=2), nn.GELU(),
                        nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.1), nn.Linear(32, 32), nn.GELU(), nn.Linear(32, nout))
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4); n = len(X)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=ep * ((n + 255) // 256)); qq = torch.tensor(QS)
    for e in range(ep):
        net.train(); p = torch.randperm(n)
        for i in range(0, n, 256):
            b = p[i:i + 256]; o = net(X[b].transpose(1, 2))
            if kind == 'de':
                ls = o[:, 1].clamp(-6, 3); l = ((Yn[b] - o[:, 0]) ** 2).mean() if e < 3 else (0.5 * (Yn[b] - o[:, 0]) ** 2 * torch.exp(-2 * ls) + ls).mean()
            else:
                r = Yn[b][:, None] - o; l = torch.max(qq * r, (qq - 1) * r).mean()
            opt.zero_grad(); l.backward(); opt.step(); sch.step()
    net.eval(); return net, mu, sd, ym, ys


def job(a):
    import torch
    key, tr, te, kind, seed = a
    net, mu, sd, ym, ys = train_net(tr, kind, seed); out = {}
    with torch.no_grad():
        for u in te:
            W, _, _ = windows(u, mu, sd); o = net(torch.tensor(W).transpose(1, 2)).numpy()
            out[u] = (o[:, 0] * ys + ym, np.exp(np.clip(o[:, 1], -6, 3)) * ys) if kind == 'de' else o * ys + ym
    return key, out


def jobs():
    J = []
    for f, te in enumerate(OUTER):
        tr = [u for u in UNITS if u not in te]
        for kind in ['de', 'qr']:
            for s in range(5): J.append((('outer', f, kind, s), tr, te, kind, s))
        inner = [sorted(tr[i::4]) for i in range(4)]
        for g, te2 in enumerate(inner):
            tr2 = [u for u in tr if u not in te2]
            for kind in ['de', 'qr']:
                for s in range(5): J.append((('inner', f, g, kind, s), tr2, te2, kind, s))
    return J


# ------------------------------------------------------------------ evaluation
def ens_de(R, key_prefix, u):
    mus = np.array([R[key_prefix + ('de', s)][u][0] for s in range(5)]); sds = np.array([R[key_prefix + ('de', s)][u][1] for s in range(5)])
    return mus.mean(0), np.sqrt((sds ** 2).mean(0) + mus.var(0))


def ens_qr(R, key_prefix, u):
    return np.mean([R[key_prefix + ('qr', s)][u] for s in range(5)], 0)


def mom(pairs):
    z2 = []; a1 = []; a2 = []
    for e, s in pairs:
        if len(e) < 3: continue
        z = e / np.maximum(s, 1e-9); v = np.mean(z * z); z2.append(v); a1.append(np.mean(z[1:] * z[:-1]) / v); a2.append(np.mean(z[2:] * z[:-2]) / v)
    kappa = float(np.sqrt(np.mean(z2))); r1 = float(np.mean(a1)); r2 = float(np.mean(a2))
    phi = float(np.clip(r2 / max(r1, 1e-3), 0.5, 0.995)); psi = float(np.clip(r1 / phi, 0.05, 0.98))
    return dict(kappa=kappa, phi=phi, psi=psi, r1=r1, r2=r2)


def fuse(y, s, cyc, prior_m, prior_s, par, persistent, G=801):
    """grid over the unit life L (RUL_k = L - cycle_k) with an exact Kalman filter for the persistent bias b_k"""
    kap, phi, psi = par['kappa'], par['phi'], par['psi']
    if not persistent: psi = 1e-6                                   # independent errors with the same (recalibrated) variance
    Lg = np.linspace(max(cyc[0], prior_m - 5 * prior_s), max(prior_m + 5 * prior_s, cyc[0] + 1), G)
    lp = -0.5 * ((Lg - prior_m) / prior_s) ** 2; m = np.zeros(G); P = psi * (kap * s[0]) ** 2
    Rq = np.zeros((len(y), 5))
    for k in range(len(y)):
        if k > 0:
            m = phi * m; P = phi ** 2 * P + psi * (kap * s[k]) ** 2 * (1 - phi ** 2)
        r = (1 - psi) * (kap * s[k]) ** 2; nu = y[k] - (Lg - cyc[k]) - m; S = P + r
        lp += -0.5 * nu ** 2 / S; K = P / S; m = m + K * nu; P = (1 - K) * P
        alive = Lg >= cyc[k]                                        # the unit is known to be running at cycle k (causal)
        lq = np.where(alive, lp, -np.inf)
        if not alive.any(): lq = lp.copy()
        w = np.exp(lq - lq.max()); w /= w.sum(); Rg = np.maximum(Lg - cyc[k], 0); o = np.argsort(Rg); c = np.cumsum(w[o])
        Rq[k] = np.interp(QS, c, Rg[o])
    return Rq


def score(R, lo, hi):
    l = np.log1p(R / TAU); return np.maximum(np.log1p(np.maximum(lo, 0) / TAU) - l, l - np.log1p(np.maximum(hi, 0) / TAU))


def apply_Q(lo, hi, Q):
    return TAU * np.expm1(np.maximum(np.log1p(np.maximum(lo, 0) / TAU) - Q, 0)), TAU * np.expm1(np.log1p(np.maximum(hi, 0) / TAU) + Q)


def q_unit(scores):   # two-layer CDF pooling, every unit weight 1
    x = np.concatenate(scores); w = np.concatenate([np.full(len(s), 1 / len(s)) for s in scores]); o = np.argsort(x); c = np.cumsum(w[o]) / len(scores)
    return max(float(x[o][min(np.searchsorted(c, 1 - ALPHA), len(x) - 1)]), 0.0)


def q_window(scores):  # standard split conformal over pooled windows (treats windows as exchangeable)
    x = np.sort(np.concatenate(scores)); n = len(x); return max(float(x[min(int(np.ceil((1 - ALPHA) * (n + 1))) - 1, n - 1)]), 0.0)


def metrics(u, p):
    T = U[u]['T']; _, cyc, R = windows(u, 0, 1); e = p['R'] - R
    out = dict(nMAE=np.abs(e).mean() / T * 100, RMSE=np.sqrt((e ** 2).mean()))
    lo = np.maximum(p['lo'], 0); hi = p['hi']; cov = (R >= lo) & (R <= hi)
    out.update(PICP=cov.mean() * 100, NMPIW=np.mean(hi - lo) / T * 100,
               NIS=np.mean((hi - lo) + 2 / ALPHA * (lo - R) * (R < lo) + 2 / ALPHA * (R - hi) * (R > hi)) / T * 100)
    return out


def evaluate(R):
    res = {}; params = []; acf = []
    for f, te in enumerate(OUTER):
        tr = [u for u in UNITS if u not in te]; inner = [sorted(tr[i::4]) for i in range(4)]
        lives = np.array([U[u]['T'] for u in tr]); pm, ps = lives.mean(), lives.std(ddof=1)
        # nested residuals of the quantile ensemble (median, scale from the 90% band) on held-out inner units
        pairs = []; inner_out = {}
        for g, te2 in enumerate(inner):
            for u in te2:
                q = ens_qr(R, ('inner', f, g), u); _, cyc, Rt = windows(u, 0, 1); s = np.maximum((q[:, 4] - q[:, 0]) / 3.29, 1.0)
                pairs.append((q[:, 2] - Rt, s)); inner_out[u] = (q, cyc, Rt, s, g)
        par = mom(pairs); params.append(par)
        # autocorrelation of the normalised residuals (median over units), lags 0..40
        for e, s in pairs:
            z = e / s; z = z - 0; v = np.mean(z * z)
            acf.append([np.mean(z[l:] * z[:len(z) - l]) / v if len(z) > l + 2 else np.nan for l in range(41)])
        # inner priors per inner fold (lives of the inner training units)
        sc = {'iid': [], 'pers': [], 'qr': []}
        for u, (q, cyc, Rt, s, g) in inner_out.items():
            l2 = np.array([U[v]['T'] for v in tr if v not in inner[g]])
            pairs2 = [p_ for v, p_ in zip([w for w in inner_out], pairs) if inner_out[v][4] != g]
            par2 = mom(pairs2)
            for name, pers in [('iid', False), ('pers', True)]:
                Rq = fuse(q[:, 2], s, cyc, l2.mean(), l2.std(ddof=1), par2, pers); sc[name].append(score(Rt, Rq[:, 0], Rq[:, 4]))
            sc['qr'].append(score(Rt, q[:, 0], q[:, 4]))
        Qu = {k: q_unit(v) for k, v in sc.items()}; Qw = {k: q_window(v) for k, v in sc.items()}
        for u in te:
            _, cyc, Rt = windows(u, 0, 1)
            mu, sd = ens_de(R, ('outer', f), u); q = ens_qr(R, ('outer', f), u); s = np.maximum((q[:, 4] - q[:, 0]) / 3.29, 1.0)
            P = {'DE': dict(R=np.maximum(mu, 0), lo=mu - 1.645 * sd, hi=mu + 1.645 * sd), 'QR': dict(R=np.maximum(q[:, 2], 0), lo=q[:, 0], hi=q[:, 4])}
            lo, hi = apply_Q(q[:, 0], q[:, 4], Qw['qr']); P['QR+CP(window)'] = dict(R=P['QR']['R'], lo=lo, hi=hi)
            lo, hi = apply_Q(q[:, 0], q[:, 4], Qu['qr']); P['QR+CP(unit)'] = dict(R=P['QR']['R'], lo=lo, hi=hi)
            for name, pers in [('Fusion-iid', False), ('Fusion-persistent', True)]:
                Rq = fuse(q[:, 2], s, cyc, pm, ps, par, pers); key = 'iid' if not pers else 'pers'
                P[name] = dict(R=Rq[:, 2], lo=Rq[:, 0], hi=Rq[:, 4])
                lo, hi = apply_Q(Rq[:, 0], Rq[:, 4], Qw[key]); P[name + '+CP(window)'] = dict(R=Rq[:, 2], lo=lo, hi=hi)
                lo, hi = apply_Q(Rq[:, 0], Rq[:, 4], Qu[key]); P[name + '+CP(unit)'] = dict(R=Rq[:, 2], lo=lo, hi=hi)
            res[u] = {k: metrics(u, v) for k, v in P.items()}
    return dict(res=res, params=params, acf=np.nanmedian(np.array(acf), 0))


if __name__ == '__main__':
    what = sys.argv[1]; t0 = time.time()
    if what == 'nets':
        J = jobs(); R = {}
        with Pool(int(os.environ.get('NPROC', '17'))) as p:
            for i, (k, v) in enumerate(p.imap_unordered(job, J, chunksize=1)):
                R[k] = v
                if i % 25 == 0: print(i, len(J), round(time.time() - t0), flush=True)
        pickle.dump(R, open('out/cmapss_nets.pkl', 'wb'))
    else:
        R = pickle.load(open('out/cmapss_nets.pkl', 'rb')); E = evaluate(R); pickle.dump(E, open('out/cmapss_eval.pkl', 'wb'))
        ms = list(next(iter(E['res'].values())).keys())
        for m in ms: print(f'{m:28s}', {k: round(np.mean([r[m][k] for r in E['res'].values()]), 1) for k in ['nMAE', 'RMSE', 'PICP', 'NMPIW', 'NIS']},
                           'units cov<80%:', sum(r[m]['PICP'] < 80 for r in E['res'].values()))
        print('params', [{k: round(v, 3) for k, v in p.items()} for p in E['params']]); print('acf', np.round(E['acf'][[1, 5, 10, 20, 40]], 2))
    print('done', round(time.time() - t0))
