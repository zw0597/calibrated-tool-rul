"""v1.6 (revision material): nested tuning of the history-aware deep baseline.
Two families of causal GRU sequence models over a tool's evaluation windows (periodic inspections):
  A: window statistics + condition + inspection features + elapsed time (as v7_seq.py);
  B: as A plus the deep evidence of the framework (wear-increment estimate and scale, direct-RUL median and scale), i.e. a
     learned fusion of the same evidence. Evidence of training tools comes from the nested inner networks (out-of-sample).
     In the inner folds the training tools' evidence comes from networks that did see the inner held-out tool (no third
     nesting level is available); the test-side evidence is always out-of-sample.
A random sample of configurations (hidden size, layers, dropout, input noise, learning rate, epochs, linear/log target) is
trained in every inner fold (2 members); for each outer test tool the configuration with the smallest mean inner interval
score (raw Gaussian interval, the primary endpoint) is selected, its 5-member outer ensemble is used, and its inner
ensembles give the two-layer conformal scores. The outer test tool never enters the selection.
  NPROC=14 python v7_seqtune.py -> out/v7_seqtune_raw.pkl, out/v7_seqtune.pkl"""
import os, sys, pickle, time, numpy as np
from multiprocessing import Pool
os.environ.setdefault('NT', '1')
import torch, torch.nn as nn
torch.set_num_threads(1)
from lib import D, P, tags, tfail, aux_feats
from pipeline import INSP, rul_true, emask
DSS = os.environ.get('DSS', 'PHM,NUAA,MILL,MATWI').split(',')
NOUT, NIN, SCEN = 5, 2, 'insp'
rng = np.random.default_rng(2026)
CFG = []
for fam in ['A', 'B']:
    for i in range(8):
        CFG.append(dict(fam=fam, h=int(rng.choice([32, 64, 128])), nl=int(rng.choice([1, 2])), drop=float(rng.choice([0.0, 0.2])),
                        noise=float(rng.choice([0.02, 0.1, 0.3])), lr=float(rng.choice([1e-3, 3e-3])), ep=int(rng.choice([300, 600])),
                        tgt=str(rng.choice(['lin', 'log']))))
_N = {}
def evid(ds, k, t):
    from v2 import Nets
    if ds not in _N: _N[ds] = Nets(ds)
    I, R = _N[ds].outer(k) if t == k else _N[ds].inner(k, t)
    return np.stack([I['z'], np.log(I['s']), R['z'], np.log(R['s'])], 1)
def feats(t, ds, cfg, k):
    S = P[t]['Ste']; c = np.tile(D[t]['c'], (len(S), 1)); A = aux_feats(t, 'te', (SCEN, INSP[ds]))
    F = [S, c, A, D[t]['tte'][:, None]]
    if cfg['fam'] == 'B': F.append(evid(ds, k, t))
    return np.nan_to_num(np.concatenate(F, 1).astype(np.float64))
class GRUNet(nn.Module):
    def __init__(s, nin, h, nl, drop):
        super().__init__(); s.inp = nn.Sequential(nn.Linear(nin, h), nn.GELU(), nn.Dropout(drop))
        s.gru = nn.GRU(h, h, num_layers=nl, batch_first=True, dropout=drop if nl > 1 else 0.0); s.out = nn.Sequential(nn.Dropout(drop), nn.Linear(h, 2))
    def forward(s, x): z, _ = s.gru(s.inp(x)); return s.out(z)
EPS = 0.05
def to_y(r, Tbar, tgt): return r / Tbar if tgt == 'lin' else np.log(np.maximum(r, 0) / Tbar + EPS)
def train(tr, ds, cfg, seed, k):
    torch.manual_seed(seed); np.random.seed(seed)
    F = [feats(t, ds, cfg, k) for t in tr]; A = np.concatenate(F); mu = A.mean(0); sd = A.std(0) + 1e-6
    Tbar = np.mean([tfail(t) for t in tr])
    X = [torch.tensor((f - mu) / sd, dtype=torch.float32) for f in F]
    Y = [torch.tensor(to_y(rul_true(t), Tbar, cfg['tgt']), dtype=torch.float32) for t in tr]
    M = [torch.tensor(D[t]['tte'] < tfail(t), dtype=torch.bool) for t in tr]
    L = max(len(x) for x in X); n = len(X)
    Xp = torch.zeros(n, L, X[0].shape[1]); Yp = torch.zeros(n, L); Mp = torch.zeros(n, L, dtype=torch.bool)
    for i in range(n): Xp[i, :len(X[i])] = X[i]; Yp[i, :len(Y[i])] = Y[i]; Mp[i, :len(M[i])] = M[i]
    net = GRUNet(Xp.shape[2], cfg['h'], cfg['nl'], cfg['drop']); opt = torch.optim.AdamW(net.parameters(), cfg['lr'], weight_decay=1e-3)
    ep = cfg['ep']; warm = ep // 4
    for e in range(ep):
        net.train(); o = net(Xp + cfg['noise'] * torch.randn_like(Xp)); m_, ls = o[..., 0], o[..., 1].clamp(-6, 3)
        l = ((m_ - Yp) ** 2) if e < warm else (0.5 * (Yp - m_) ** 2 * torch.exp(-2 * ls) + ls)
        l = l[Mp].mean(); opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    net.eval(); return net, mu, sd, Tbar
def predict(model, t, ds, cfg, k):
    net, mu, sd, Tbar = model
    with torch.no_grad(): o = net(torch.tensor((feats(t, ds, cfg, k) - mu) / sd, dtype=torch.float32)[None])[0].numpy()
    return o[:, 0], np.exp(np.clip(o[:, 1], -6, 3)), Tbar
def job(a):
    ci, ds, k, j, s = a; cfg = CFG[ci]; T = tags(ds); tr = [t for t in T if t not in (k, j)]; pt = k if j is None else j
    return a, predict(train(tr, ds, cfg, s, k), pt, ds, cfg, k)
def ensemble(R, key, ns, tgt):
    mus = np.array([R[key + (s,)][0] for s in range(ns)]); sds = np.array([R[key + (s,)][1] for s in range(ns)]); Tbar = R[key + (0,)][2]
    m = mus.mean(0); sd = np.sqrt((sds ** 2).mean(0) + mus.var(0))
    f = (lambda y: Tbar * y) if tgt == 'lin' else (lambda y: Tbar * (np.exp(y) - EPS))
    return dict(R=np.maximum(f(m), 0), lo=np.maximum(f(m - 1.645 * sd), 0), hi=np.maximum(f(m + 1.645 * sd), 0))
if __name__ == '__main__':
    f = 'out/v7_seqtune_raw.pkl'; R = pickle.load(open(f, 'rb')) if os.path.exists(f) else {}
    args = []
    for ds in DSS:
        T = tags(ds)
        for ci in range(len(CFG)):
            args += [(ci, ds, k, j, s) for k in T for j in T if j != k for s in range(NIN)]
            args += [(ci, ds, k, None, s) for k in T for s in range(NOUT)]
    args = [a for a in args if a not in R]; print('todo', len(args), flush=True); t0 = time.time()
    with Pool(int(os.environ.get('NPROC', '14'))) as p:
        for i, (a, o) in enumerate(p.imap_unordered(job, args, chunksize=4)):
            R[a] = o
            if i % 500 == 0: pickle.dump(R, open(f, 'wb')); print(i, len(args), round(time.time() - t0), flush=True)
    pickle.dump(R, open(f, 'wb'))
    from pipeline import conformal_Q, cqr_score, cqr_apply
    from metrics import tool_metrics
    OUT = {}; SEL = {}
    for ds in DSS:
        T = tags(ds)
        for fams, nm in [(('A',), 'SEQopt'), (('A', 'B'), 'SEQevopt')]:
            for k in T:
                best = None
                for ci, cfg in enumerate(CFG):
                    if cfg['fam'] not in fams: continue
                    sc = np.mean([tool_metrics(j, ensemble(R, (ci, ds, k, j), NIN, cfg['tgt']))['NIS'] for j in T if j != k])
                    if best is None or sc < best[0]: best = (sc, ci)
                ci = best[1]; cfg = CFG[ci]; o = ensemble(R, (ci, ds, k, None), NOUT, cfg['tgt']); sc = []
                for j in T:
                    if j == k: continue
                    oj = ensemble(R, (ci, ds, k, j), NIN, cfg['tgt']); m = emask(j); sc.append(cqr_score(rul_true(j)[m], oj['lo'][m], oj['hi'][m]))
                Q = conformal_Q(sc); lo, hi = cqr_apply(o['lo'], o['hi'], Q)
                OUT.setdefault(ds, {}).setdefault(k, {})[nm] = o; OUT[ds][k][nm + 'CP'] = dict(R=o['R'], lo=lo, hi=hi, Q=Q); SEL[(ds, k, nm)] = ci
    pickle.dump({'OUT': OUT, 'SEL': SEL, 'CFG': CFG}, open('out/v7_seqtune.pkl', 'wb')); print('done', round(time.time() - t0), flush=True)
    for ds in DSS:
        for nm in ['SEQopt', 'SEQoptCP', 'SEQevopt', 'SEQevoptCP']:
            per = [tool_metrics(t, OUT[ds][t][nm]) for t in tags(ds)]
            print(ds, nm, {c: round(float(np.nanmean([q[c] for q in per])), 1) for c in ['nMAE', 'PICP', 'NIS', 'MAElate']}, 'cfg', sorted({SEL[(ds, t, nm.replace('CP', ''))] for t in tags(ds)}))
