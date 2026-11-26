"""Compare tuned sequence baselines with the proposed method (periodic inspections, 40 tools) -> v7/seqtune_report.txt"""
import pickle, numpy as np
from metrics import tool_metrics
from lib import tags, tfail
from v7_tests import preds
from v7_tests2 import van_elteren
Z = pickle.load(open('out/v7_seqtune.pkl', 'rb')); OUT, SEL, CFG = Z['OUT'], Z['SEL'], Z['CFG']
S0 = pickle.load(open('out/v7_seq.pkl', 'rb'))
DSS = ['PHM', 'NUAA', 'MILL', 'MATWI']; L = []
def get(ds, t, m):
    if m in ('SEQ+', 'SEQ+CP'): return S0[ds][t]['insp'][m]
    if m in ('StackCP', 'DE+', 'DE+CP', 'CQR+'): return preds(ds, t, 'insp', m)
    return OUT[ds][t][m]
MS = ['SEQ+', 'SEQ+CP', 'SEQopt', 'SEQoptCP', 'SEQevopt', 'SEQevoptCP', 'DE+', 'DE+CP', 'CQR+', 'StackCP']
for ds in DSS:
    for m in MS:
        per = [tool_metrics(t, get(ds, t, m)) for t in tags(ds)]
        L.append(f"{ds:6s} {m:11s} " + ' '.join(f"{c}={np.nanmean([q[c] for q in per]):.1f}" for c in ['nMAE', 'PICP', 'NMPIW', 'NIS', 'MAElate']))
    for nm in ['SEQopt', 'SEQevopt']: L.append(f"{ds} {nm} selected configs: " + str(sorted({SEL[(ds, t, nm)] for t in tags(ds)})))
L.append('')
for m in ['SEQoptCP', 'SEQevoptCP', 'SEQopt', 'SEQevopt']:
    for key in ['NIS', 'nMAE', 'late']:
        G = []
        for ds in DSS:
            g = []
            for t in tags(ds):
                a = tool_metrics(t, get(ds, t, 'StackCP')); b = tool_metrics(t, get(ds, t, m))
                g.append((a['MAElate'] - b['MAElate']) / tfail(t) * 100 if key == 'late' else a[key] - b[key])
            G.append(np.array(g))
        d = np.concatenate(G)
        L.append(f"Proposed - {m:10s} {key:5s} mean {d.mean():9.1f} median {np.median(d):8.1f} p_vE {van_elteren(G):.5f} wins {(d < 0).sum()}/40")
L.append(''); L += [f"cfg {i}: {c}" for i, c in enumerate(CFG)]
open('v7/seqtune_report.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
