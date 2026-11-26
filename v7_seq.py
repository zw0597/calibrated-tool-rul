"""v1.6 (revision material): history-aware deep baseline SEQ+. A causal GRU reads, for every evaluation window k of a tool
from mounting onwards, [reference-relative window statistics, cutting condition, inspection features (z0, z_last-z0,
t-t_last), elapsed time] and outputs a Gaussian RUL at every k; 5-member ensemble, trained on the evaluation-window
sequences of the training tools (loss on windows before failure). SEQ+CP: the same two-layer conformal step as the other
methods, scores from nested inner ensembles (2 members).
  NPROC=12 python v7_seq.py  -> out/v7_seq_raw.pkl, out/v7_seq.pkl"""
import os, sys, pickle, time, numpy as np
from multiprocessing import Pool
os.environ.setdefault('NT', '1')
import torch, torch.nn as nn
torch.set_num_threads(1)
from lib import D, P, tags, tfail, aux_feats
from pipeline import INSP, rul_true, emask
DSS = os.environ.get('DSS', 'PHM,NUAA,MILL,MATWI').split(',')
NOUT, NIN = 5, 2
def feats(t, scen, ds):
    S = P[t]['Ste']; c = np.tile(D[t]['c'], (len(S), 1)); A = aux_feats(t, 'te', (scen, INSP[ds] if scen == 'insp' else None))
    return np.nan_to_num(np.concatenate([S, c, A, D[t]["tte"][:, None]], 1).astype(np.float64))
class GRUNet(nn.Module):
    def __init__(s, nin, h=48):
        super().__init__(); s.inp = nn.Sequential(nn.Linear(nin, h), nn.GELU()); s.gru = nn.GRU(h, h, batch_first=True); s.out = nn.Linear(h, 2)
    def forward(s, x): z, _ = s.gru(s.inp(x)); return s.out(z)
def train(tr, ds, scen, seed, ep=400):
    torch.manual_seed(seed); np.random.seed(seed)
    F = [feats(t, scen, ds) for t in tr]; A = np.concatenate(F); mu = A.mean(0); sd = A.std(0) + 1e-6
    Tbar = np.mean([tfail(t) for t in tr])
    X = [torch.tensor((f - mu) / sd, dtype=torch.float32) for f in F]
    Y = [torch.tensor(rul_true(t) / Tbar, dtype=torch.float32) for t in tr]
    M = [torch.tensor(emask(t) | ((D[t]['tte'] < tfail(t)) & (np.arange(len(D[t]['tte'])) < D[t].get('nb', 5))), dtype=torch.bool) for t in tr]
    L = max(len(x) for x in X); n = len(X)
    Xp = torch.zeros(n, L, X[0].shape[1]); Yp = torch.zeros(n, L); Mp = torch.zeros(n, L, dtype=torch.bool)
    for i in range(n): Xp[i, :len(X[i])] = X[i]; Yp[i, :len(Y[i])] = Y[i]; Mp[i, :len(M[i])] = M[i]
    net = GRUNet(Xp.shape[2]); opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=1e-3)
    for e in range(ep):
        net.train(); o = net(Xp + 0.05 * torch.randn_like(Xp)); mu_, ls = o[..., 0], o[..., 1].clamp(-6, 3)
        l = ((mu_ - Yp) ** 2) if e < 100 else (0.5 * (Yp - mu_) ** 2 * torch.exp(-2 * ls) + ls)
        l = l[Mp].mean(); opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    net.eval(); return net, mu, sd, Tbar
def predict(model, t, ds, scen):
    net, mu, sd, Tbar = model
    with torch.no_grad(): o = net(torch.tensor((feats(t, scen, ds) - mu) / sd, dtype=torch.float32)[None])[0].numpy()
    return o[:, 0] * Tbar, np.exp(np.clip(o[:, 1], -6, 3)) * Tbar
def job(a):
    ds, scen, k, j, s = a; T = tags(ds); tr = [t for t in T if t not in (k, j)]; pt = k if j is None else j
    return a, predict(train(tr, ds, scen, s), pt, ds, scen)
def ensemble(R, key, ns):
    mus = np.array([R[key + (s,)][0] for s in range(ns)]); sds = np.array([R[key + (s,)][1] for s in range(ns)])
    m = mus.mean(0); sd = np.sqrt((sds ** 2).mean(0) + mus.var(0)); return dict(R=np.maximum(m, 0), lo=np.maximum(m - 1.645 * sd, 0), hi=np.maximum(m + 1.645 * sd, 0))
if __name__ == '__main__':
    f = 'out/v7_seq_raw.pkl'; R = pickle.load(open(f, 'rb')) if os.path.exists(f) else {}
    args = []
    for ds in DSS:
        T = tags(ds)
        for scen in ['insp', 'none']:
            args += [(ds, scen, k, None, s) for k in T for s in range(NOUT)]
            args += [(ds, scen, k, j, s) for k in T for j in T if j != k for s in range(NIN)]
    args = [a for a in args if a not in R]; print('todo', len(args), flush=True); t0 = time.time()
    with Pool(int(os.environ.get('NPROC', '12'))) as p:
        for i, (a, o) in enumerate(p.imap_unordered(job, args, chunksize=4)):
            R[a] = o
            if i % 200 == 0: pickle.dump(R, open(f, 'wb')); print(i, len(args), round(time.time() - t0), flush=True)
    pickle.dump(R, open(f, 'wb'))
    from pipeline import conformal_Q, cqr_score, cqr_apply
    OUT = {}
    for ds in DSS:
        T = tags(ds)
        for k in T:
            for scen in ['insp', 'none']:
                o = ensemble(R, (ds, scen, k, None), NOUT); sc = []
                for j in T:
                    if j == k: continue
                    oj = ensemble(R, (ds, scen, k, j), NIN); m = emask(j); sc.append(cqr_score(rul_true(j)[m], oj['lo'][m], oj['hi'][m]))
                Q = conformal_Q(sc); lo, hi = cqr_apply(o['lo'], o['hi'], Q)
                OUT.setdefault(ds, {}).setdefault(k, {})[scen] = {'SEQ+': o, 'SEQ+CP': dict(R=o['R'], lo=lo, hi=hi, Q=Q)}
    pickle.dump(OUT, open('out/v7_seq.pkl', 'wb')); print('done', round(time.time() - t0), flush=True)
