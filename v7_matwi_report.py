"""v1.4: merge the per-fold MATWI outputs and print the frozen-evaluation summary.
  python v7_matwi_report.py [merge]
"""
import pickle, glob, os, sys, numpy as np
from metrics import tool_metrics
from pipeline import tags
T = tags('MATWI')
if 'merge' in sys.argv:
    O = {}
    for f in sorted(glob.glob('out/matwi_parts/rul2_MATWI_X*.pkl')): O.update(pickle.load(open(f, 'rb')))
    assert set(O) == set(T), sorted(O); pickle.dump(O, open('out/rul2_MATWI.pkl', 'wb'))
O = pickle.load(open('out/rul2_MATWI.pkl', 'rb'))
B = pickle.load(open('out/v7_matwi_base.pkl', 'rb'))
DA = pickle.load(open('out/v7_matwi_deepaux.pkl', 'rb'))['MATWI'] if os.path.exists('out/v7_matwi_deepaux.pkl') else None
CQ = pickle.load(open('out/v6_cqrplus.pkl', 'rb')).get('MATWI')
ST = pickle.load(open('out/v7_stack.pkl', 'rb')).get('MATWI')
def src(m, t, scen):
    if m in ('Pop', 'DE', 'MCD', 'CQRraw', 'CQR'): return O[t][m]
    if m == 'WienerC': return B['WienerC'][t][scen]
    if m == 'PF': return B['GP'][t][scen + '_insponly']
    if m == 'PFC': return B['GPC'][t][scen + '_insponly']
    if m == 'PFNet': return B['GP'][t][scen]
    if m == 'WienerNetK': return B['HY'][t][scen][m]
    if m in ('DE+', 'QR+'): return DA[t][scen][m]
    if m == 'CQR+': return CQ[t][scen]['CQR+']
    if m in ('Stack', 'StackCP'): return ST[t][scen][m]
    return O[t][scen][m]
MS = ['Pop', 'DE', 'DE+', 'MCD', 'CQRraw', 'QR+', 'CQR', 'CQR+', 'Wiener', 'WienerC', 'WienerNetK', 'PF', 'PFC', 'PFNet', 'MCprior',
      'woAR', 'woDisc', 'woBank', 'woI', 'woR', 'woShift', 'Ours', 'OursCP', 'Stack', 'StackCP']
for scen in ['insp', 'none']:
    print('== MATWI', scen)
    for m in MS:
        try: per = [tool_metrics(t, src(m, t, scen)) for t in T]
        except Exception: continue
        g = lambda k: np.nanmean([q.get(k, np.nan) for q in per])
        print('  %-11s nMAE %6.2f late %7.1f PICP %5.1f NMPIW %7.1f NIS %7.1f  per-tool:' % (m, g('nMAE'), g('MAElate'), g('PICP'), g('NMPIW'), g('NIS')),
              ' '.join('%.0f' % q['nMAE'] for q in per))
