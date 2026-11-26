"""v1.4 (review): causal inspections. An inspection at time tau reports the most recent actual wear measurement made at
or before tau (no interpolation towards later measurements, no smoothing), plus the same instrument noise as before.
  PHM2010: per-pass wear values; the value of a pass is measured at its end, so at tau the last completed pass is used.
  NUAA: the bundle interpolates linearly between measurements; the knots of the piecewise-linear signal are the measurements.
  NASA milling: raw VB of the latest measured run (no running maximum).  MATWI: raw expert label of the run (no smoothing).
  python causal_insp.py -> data/causal_insp.pkl  (tool -> (times, values) of the measurements)"""
import os, json, pickle, numpy as np, pandas as pd, scipy.io as sio
from lib import D, tags
OUT = {}
U = 'data/uniwear/'
for f, ds in [('phm2010_bundle_high_resolution.csv', 'PHM'), ('nuaa_orthogonal_bundle_high_resolution.csv', 'NUAA')]:
    d = pd.read_csv(U + f, usecols=['timestamp', 'tool_wear', 'experiment_tag'])
    for tag in tags(ds):
        g = d[d.experiment_tag == tag]; t = g.timestamp.values - g.timestamp.values[0]; w = g.tool_wear.values
        if ds == 'PHM':                       # step per pass: value measured at the end of the pass
            ch = np.where(np.diff(w) != 0)[0]          # last sample of each pass
            tm = np.r_[t[ch], t[-1]]; vm = np.r_[w[ch], w[-1]]
        else:                                 # interpolated label: slope-change points (1-s grid, clustered 5 s) = measurements
            grid = np.arange(t[0], t[-1], 1.0); wg = np.interp(grid, t, w); sl = np.diff(wg)
            rel = np.abs(np.diff(sl)) / (np.abs(sl[:-1]) + np.abs(sl[1:]) + 1e-12); kn = np.where(rel > 0.02)[0] + 1
            sel = [0.0]
            for i in kn:
                if grid[i] - sel[-1] > 5.0: sel.append(grid[i])
            sel.append(t[-1]); tm = np.array(sel); vm = np.interp(tm, t, w)
        OUT[tag] = (np.asarray(tm, float), np.asarray(vm, float))
m = sio.loadmat('data/mill.mat')['mill'][0]; cases = {}
for r in m: cases.setdefault(int(r['case'][0, 0]), []).append(r)
for tag in tags('MILL'):
    rs = sorted(cases[int(tag[1:])], key=lambda r: float(r['time'][0, 0]))
    t = np.array([float(r['time'][0, 0]) for r in rs]) * 60.0
    vb = np.array([float(r['VB'][0, 0]) if r['VB'].size else np.nan for r in rs]); ok = np.isfinite(vb)
    assert np.allclose(t, D[tag]['tte']); OUT[tag] = (t[ok], vb[ok])
lab = json.load(open('data/matwi/labels_clean.json'))
for name, r in lab.items():
    tag = 'X' + name[1:]
    if tag not in D or D[tag]['ds'] != 'MATWI': continue
    raw = np.array(r['raw'], float) / 1000.0; t = D[tag]['tte']; assert len(raw) == len(t), (tag, len(raw), len(t))
    OUT[tag] = (t, raw)
pickle.dump(OUT, open('data/causal_insp.pkl', 'wb'))
for tag, (tm, vm) in OUT.items(): print(tag, len(tm), 'median gap %.1f s' % np.median(np.diff(tm)) if len(tm) > 1 else '')
