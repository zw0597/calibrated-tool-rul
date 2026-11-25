import pickle, numpy as np, sys
from metrics import tool_metrics
S = pickle.load(open('out/v7_stack.pkl', 'rb')); AUX = pickle.load(open('out/v5_deepaux.pkl', 'rb')); CQ = pickle.load(open('out/v6_cqrplus.pkl', 'rb'))
for ds in S:
    O = pickle.load(open(f'out/rul2_{ds}.pkl', 'rb'))
    for scen in ['insp', 'none']:
        M = {'Proposed': lambda t: O[t][scen]['OursCP'], 'DE+': lambda t: AUX[ds][t][scen]['DE+'], 'CQR+': lambda t: CQ[ds][t][scen]['CQR+'],
             'Stack(raw)': lambda t: S[ds][t][scen]['Stack'], 'StackCP': lambda t: S[ds][t][scen]['StackCP']}
        print('==', ds, scen, 'weights', [S[ds][t][scen]['a'] for t in S[ds]])
        for m, f in M.items():
            per = [tool_metrics(t, f(t)) for t in S[ds]]
            g = lambda k: np.nanmean([q.get(k, np.nan) for q in per])
            print('  %-10s nMAE %6.2f late %6.1f PICP %5.1f NIS %6.1f' % (m, g('nMAE'), g('MAElate'), g('PICP'), g('NIS')))
