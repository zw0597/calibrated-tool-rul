"""v1.4: pooled paired tests (Wilcoxon signed-rank, bootstrap CI, Holm) for a target method against all baselines.
  DSS=PHM,NUAA,MILL[,MATWI] python v7_tests.py <target: StackCP|OursCP> <scen> -> out/v7_tests_<target>_<scen>.pkl"""
import pickle, numpy as np, sys, os
from scipy.stats import wilcoxon
import v6_analysis as V
from metrics import tool_metrics
from lib import tags, tfail
DSS = os.environ.get('DSS', 'PHM,NUAA,MILL').split(',')
ST = pickle.load(open('out/v7_stack.pkl', 'rb'))
BASE = ['OursCP', 'DE', 'DE+', 'MCD', 'CQR', 'CQR+', 'Wiener', 'WienerC', 'WienerNetK', 'PF', 'PFC', 'MCprior']
def preds(ds, t, scen, m):
    if m in ('StackCP', 'Stack'): return ST[ds][t][scen][m]
    return V.preds(ds, t, scen, m)
def run(target, scen):
    rng = np.random.default_rng(5); out = {}; base = [b for b in BASE if b != target]
    for key in ['nMAE', 'NIS', 'late']:
        rows = []
        for b in base:
            d = []
            for ds in DSS:
                for t in tags(ds):
                    a_ = tool_metrics(t, preds(ds, t, scen, target)); b_ = tool_metrics(t, preds(ds, t, scen, b))
                    if key == 'late': d.append((a_['MAElate'] - b_['MAElate']) / tfail(t) * 100)
                    elif key == 'NIS':
                        if 'NIS' not in b_: break
                        d.append(a_['NIS'] - b_['NIS'])
                    else: d.append(a_[key] - b_[key])
            if len(d) == 0: continue
            d = np.array(d); i = rng.integers(0, len(d), (10000, len(d))); bm = d[i].mean(1)
            rows.append([b, d.mean(), np.percentile(bm, 2.5), np.percentile(bm, 97.5), wilcoxon(d).pvalue, int((d < 0).sum()), len(d)])
        adj = V.holm(np.array([r[4] for r in rows]))
        for r, a in zip(rows, adj): r.append(a)
        out[key] = rows; print('==', target, scen, key)
        for r in rows: print(f'  {r[0]:10s} {r[1]:8.2f} [{r[2]:8.2f},{r[3]:8.2f}] p={r[4]:.4f} holm={r[7]:.4f} wins {r[5]}/{r[6]}')
    pickle.dump(out, open(f'out/v7_tests_{target}_{scen}.pkl', 'wb'))
if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2])
