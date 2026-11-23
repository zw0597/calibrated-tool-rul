"""Independent tool data (v1.3): end-milling tool-life data set of Scientific Data (2025), 13 run-to-failure tools
with per-cycle sensor features and the cycles to failure (no wear measurements, so the wear law cannot be used).
Same analysis as the C-MAPSS study: persistence of network errors on unseen tools, and life-grid fusion with
independent vs persistent-bias errors, with window- and tool-level conformal calibration; nested LOTO.
  SCIDATA=<path to FeatureAndMetadata_Milling.csv> python scidata_study.py   -> out/scidata_eval.pkl
"""
import os, pickle, time, numpy as np, pandas as pd
from multiprocessing import Pool
import cmapss_study as C

QS = [0.05, 0.25, 0.5, 0.75, 0.95]; C.TAU = 3.0      # time constant of the log score (cycles)
META = ['MillingToolType', 'ADOC', 'RDOC', 'HardnessMean', 'ToolHolderLength']


def load():
    d = pd.read_csv(os.environ.get('SCIDATA', 'data/scidata/FeatureAndMetadata_Milling.csv'), sep=';', skiprows=1, decimal=',')
    feat = [c for c in d.columns if c not in ['FileName', 'NumberOfCycle', 'SampleIndex', 'TollIndex', 'CycleToFailure', 'CycleToFailureNormalized'] + META]
    U = {}
    for t, g in d.groupby('TollIndex'):
        g = g.sort_values('NumberOfCycle')
        if len(g) < 10: continue
        F = g[feat].values.astype(float); F = np.sign(F) * np.log1p(np.abs(F))     # signed log: features span several orders of magnitude
        X = np.c_[F, g[META].values.astype(float)]
        cyc = g.NumberOfCycle.values.astype(float); R = g.CycleToFailure.values.astype(float)
        U[int(t)] = dict(X=X, cyc=cyc, R=R, T=float(cyc[0] + R[0]))
    return U


U = load(); UNITS = sorted(U)


def train_net(tr, kind, seed, ep=300):
    import torch, torch.nn as nn
    torch.set_num_threads(1); torch.manual_seed(seed); np.random.seed(seed)
    Xa = np.concatenate([U[u]['X'] for u in tr]); mu = Xa.mean(0); sd = Xa.std(0) + 1e-6
    X = torch.tensor(np.clip((Xa - mu) / sd, -5, 5), dtype=torch.float32); Y = torch.tensor(np.concatenate([U[u]['R'] for u in tr]), dtype=torch.float32)
    ym, ys = float(Y.mean()), float(Y.std()); Yn = (Y - ym) / ys; nout = 2 if kind == 'de' else len(QS)
    net = nn.Sequential(nn.Linear(X.shape[1], 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, nout))
    opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=1e-3); qq = torch.tensor(QS)
    for e in range(ep):
        net.train(); o = net(X)
        if kind == 'de':
            ls = o[:, 1].clamp(-6, 3); l = ((Yn - o[:, 0]) ** 2).mean() if e < 30 else (0.5 * (Yn - o[:, 0]) ** 2 * torch.exp(-2 * ls) + ls).mean()
        else:
            r = Yn[:, None] - o; l = torch.max(qq * r, (qq - 1) * r).mean()
        opt.zero_grad(); l.backward(); opt.step()
    net.eval(); return net, mu, sd, ym, ys


def job(a):
    import torch
    key, tr, te, kind, seed = a
    net, mu, sd, ym, ys = train_net(tr, kind, seed); out = {}
    with torch.no_grad():
        for u in te:
            o = net(torch.tensor(np.clip((U[u]['X'] - mu) / sd, -5, 5), dtype=torch.float32)).numpy()
            out[u] = (o[:, 0] * ys + ym, np.exp(np.clip(o[:, 1], -6, 3)) * ys) if kind == 'de' else o * ys + ym
    return key, out


def metrics(u, p):
    T = U[u]['T']; R = U[u]['R']; e = p['R'] - R
    lo = np.maximum(p['lo'], 0); hi = p['hi']; cov = (R >= lo) & (R <= hi)
    return dict(nMAE=np.abs(e).mean() / T * 100, PICP=cov.mean() * 100, NMPIW=np.mean(hi - lo) / T * 100,
                NIS=np.mean((hi - lo) + 2 / C.ALPHA * (lo - R) * (R < lo) + 2 / C.ALPHA * (R - hi) * (R > hi)) / T * 100)


def evaluate(R):
    res = {}; params = []; acf = []
    for k in UNITS:
        tr = [u for u in UNITS if u != k]; lives = np.array([U[u]['T'] for u in tr])
        def qens(key_prefix, u): return np.mean([R[key_prefix + ('qr', s)][u] for s in range(5)], 0)
        pairs = {}; inner = {}
        for j in tr:
            q = qens(('inner', k, j), j); s = np.maximum((q[:, 4] - q[:, 0]) / 3.29, 0.5)
            pairs[j] = (q[:, 2] - U[j]['R'], s); inner[j] = (q, s)
        par = C.mom(list(pairs.values())); params.append(par)
        for e, s in pairs.values():
            z = e / s; v = np.mean(z * z); acf.append([np.mean(z[l:] * z[:len(z) - l]) / v if len(z) > l + 2 else np.nan for l in range(21)])
        sc = {'iid': [], 'pers': [], 'qr': []}
        for j, (q, s) in inner.items():
            tr2 = [u for u in tr if u != j]; l2 = np.array([U[u]['T'] for u in tr2]); par2 = C.mom([pairs[u] for u in tr2])
            for name, pers in [('iid', False), ('pers', True)]:
                Rq = C.fuse(q[:, 2], s, U[j]['cyc'], l2.mean(), l2.std(ddof=1), par2, pers); sc[name].append(C.score(U[j]['R'], Rq[:, 0], Rq[:, 4]))
            sc['qr'].append(C.score(U[j]['R'], q[:, 0], q[:, 4]))
        Qu = {a: C.q_unit(v) for a, v in sc.items()}; Qw = {a: C.q_window(v) for a, v in sc.items()}
        mus = np.array([R[('outer', k, 'de', s)][k][0] for s in range(5)]); sds = np.array([R[('outer', k, 'de', s)][k][1] for s in range(5)])
        mu = mus.mean(0); sd = np.sqrt((sds ** 2).mean(0) + mus.var(0)); q = qens(('outer', k), k); s = np.maximum((q[:, 4] - q[:, 0]) / 3.29, 0.5)
        P = {'DE': dict(R=np.maximum(mu, 0), lo=mu - 1.645 * sd, hi=mu + 1.645 * sd), 'QR': dict(R=np.maximum(q[:, 2], 0), lo=q[:, 0], hi=q[:, 4])}
        for nm, key in [('QR+CP(window)', 'w'), ('QR+CP(unit)', 'u')]:
            lo, hi = C.apply_Q(q[:, 0], q[:, 4], (Qw if key == 'w' else Qu)['qr']); P[nm] = dict(R=P['QR']['R'], lo=lo, hi=hi)
        for name, pers in [('Fusion-iid', False), ('Fusion-persistent', True)]:
            Rq = C.fuse(q[:, 2], s, U[k]['cyc'], lives.mean(), lives.std(ddof=1), par, pers); key = 'pers' if pers else 'iid'
            P[name] = dict(R=Rq[:, 2], lo=Rq[:, 0], hi=Rq[:, 4])
            for suf, QQ in [('+CP(window)', Qw), ('+CP(unit)', Qu)]:
                lo, hi = C.apply_Q(Rq[:, 0], Rq[:, 4], QQ[key]); P[name + suf] = dict(R=Rq[:, 2], lo=lo, hi=hi)
        res[k] = {m: metrics(k, p) for m, p in P.items()}
    return dict(res=res, params=params, acf=np.nanmedian(np.array(acf), 0))


if __name__ == '__main__':
    os.environ['NT'] = '1'; t0 = time.time()
    J = []
    for k in UNITS:
        tr = [u for u in UNITS if u != k]
        for kind in ['de', 'qr']:
            for s in range(5): J.append((('outer', k, kind, s), tr, [k], kind, s))
        for j in tr:
            tr2 = [u for u in tr if u != j]
            for s in range(5): J.append((('inner', k, j, 'qr', s), tr2, [j], 'qr', s))
    R = {}
    with Pool(int(os.environ.get('NPROC', '8'))) as p:
        for kk, v in p.imap_unordered(job, J, chunksize=4): R[kk] = v
    pickle.dump(R, open('out/scidata_nets.pkl', 'wb'))
    E = evaluate(R); pickle.dump(E, open('out/scidata_eval.pkl', 'wb'))
    print('tools', len(UNITS), 'lives', [int(U[u]['T']) for u in UNITS])
    for m in E['res'][UNITS[0]]:
        print(f'{m:28s}', {q: round(np.mean([r[m][q] for r in E['res'].values()]), 1) for q in ['nMAE', 'PICP', 'NMPIW', 'NIS']},
              'tools cov<80%:', sum(r[m]['PICP'] < 80 for r in E['res'].values()))
    print('params', {q: (round(min(p[q] for p in E['params']), 3), round(max(p[q] for p in E['params']), 3)) for q in ['kappa', 'phi', 'psi']})
    print('acf', np.round(E['acf'][[1, 2, 5, 10, 20]], 2), 'done', round(time.time() - t0))
