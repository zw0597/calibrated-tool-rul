"""v1.4 (review): paired comparisons over the 40 tools of the four tool datasets, stratified by dataset.
Test: van Elteren stratified signed-rank test (Wilcoxon signed-rank within each dataset, strata weighted by 1/(n_s+1),
normal approximation); Holm correction over the alternatives of each endpoint. CI: dataset-stratified bootstrap of the
mean difference (10 000 resamples). Also the unstratified pooled Wilcoxon p for reference.
  python v7_tests2.py <target> <scen> -> out/v7_tests2_<target>_<scen>.pkl"""
import pickle, numpy as np, sys
from scipy.stats import wilcoxon, norm, rankdata
import v6_analysis as V
from metrics import tool_metrics
from lib import tags, tfail
from v7_tests import preds as _p0
def preds(ds, t, scen, m): return _p0(ds, t, scen, m)
DSS = ['PHM', 'NUAA', 'MILL', 'MATWI']
BASE = ['OursCP', 'StackCP', 'DE', 'DE+', 'DE+CP', 'MCD', 'CQRraw', 'QR+', 'CQR', 'CQR+', 'Wiener', 'WienerC', 'WienerNetK', 'WienerARCP', 'PF', 'PFC', 'PFNet', 'MCprior']
def van_elteren(groups):
    num = 0.0; var = 0.0
    for d in groups:
        d = np.asarray(d); d = d[d != 0]; n = len(d)
        if n == 0: continue
        r = rankdata(np.abs(d)); Tp = r[d > 0].sum(); w = 1.0 / (n + 1)
        num += w * (Tp - n * (n + 1) / 4); var += w ** 2 * n * (n + 1) * (2 * n + 1) / 24
    z = num / np.sqrt(var); return 2 * norm.sf(abs(z))
def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for i, j in enumerate(o): run = max(run, (m - i) * ps[j]); adj[j] = min(run, 1)
    return adj
def run(target, scen):
    rng = np.random.default_rng(5); out = {}; base = [b for b in BASE if b != target]
    M = {}
    for m in [target] + base:
        for ds in DSS:
            for t in tags(ds): M[(m, t)] = tool_metrics(t, preds(ds, t, scen, m))
    for key in ['NIS', 'nMAE', 'late']:
        rows = []
        for b in base:
            G = []
            for ds in DSS:
                g = []
                for t in tags(ds):
                    a_, b_ = M[(target, t)], M[(b, t)]
                    if key == 'late': g.append((a_['MAElate'] - b_['MAElate']) / tfail(t) * 100)
                    else: g.append(a_.get(key, np.nan) - b_.get(key, np.nan))
                G.append(np.array(g))
            if any(np.isnan(g).any() for g in G): continue
            d = np.concatenate(G)
            bs = np.array([np.concatenate([g[rng.integers(0, len(g), len(g))] for g in G]).mean() for _ in range(10000)])
            rows.append([b, d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), van_elteren(G), int((d < 0).sum()), len(d), wilcoxon(d).pvalue])
        fam = [r for r in rows if r[0] not in ('OursCP', 'StackCP')]   # ablation rows are not part of the Holm family
        adj = holm([r[4] for r in fam])
        for r, a in zip(fam, adj): r.append(a)
        for r in rows:
            if len(r) == 8: r.append(np.nan)   # r = [b, mean, lo, hi, p_vE, wins, n, p_pooled, p_vE_Holm]
        out[key] = rows
    pickle.dump(out, open(f'out/v7_tests2_{target}_{scen}.pkl', 'wb')); return out
if __name__ == '__main__':
    o = run(sys.argv[1], sys.argv[2])
    for k, rows in o.items():
        print(k)
        for r in rows: print(f'  {r[0]:11s} {r[1]:9.1f} [{r[2]:9.1f},{r[3]:9.1f}] pvE={r[4]:.4f} pHolm={r[8]:.4f} pPool={r[7]:.4f} wins {r[5]}/{r[6]}')
