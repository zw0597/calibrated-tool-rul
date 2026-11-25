"""MATWI windowed data (rules of PROTOCOL_MATWI.md). Appends tools 'Xnn' with ds='MATWI' to data/win.pkl.
  MATWI_EXT=<dir with SetN/sensordata/*.csv> python prep_matwi.py
Requires data/matwi/labels_clean.json (v7_matwi_labels.py) and data/matwi/sets.csv.
"""
import os, json, pickle, numpy as np, pandas as pd
EXT = os.environ.get('MATWI_EXT', os.path.expanduser('~/matwi_ext'))
VSTAR = float(os.environ.get("TH_MATWI", "0.09")); L = 100; BLK = 65; STRIDE = 200
lab = json.load(open('data/matwi/labels_clean.json'))
sets = pd.read_csv('data/matwi/sets.csv', index_col=0)


def cond(s):
    r = sets.loc[f'Set {s}']
    mat = 2.0 if 'RVS' in str(r['material']) else 1.0
    if str(r['Vc']).strip() == '?':
        ck = sets[(sets.material == 'CK45') & (sets.Vc.astype(str) != '?')]
        return np.array([np.median(ck.Vc.astype(float)), np.median(ck.fz.astype(float)), mat], np.float32)
    return np.array([float(r['Vc']), float(r['fz']), mat], np.float32)


def run_env(path):
    d = pd.read_csv(path, header=None).values[:, :5].astype(np.float64)
    d = d - np.median(d, 0); n = len(d) // BLK * BLK
    env = np.sqrt((d[:n].reshape(-1, BLK, 5) ** 2).mean(1))
    F = np.sqrt(env[:, 2] ** 2 + env[:, 3] ** 2); on = F > 0.2 * np.percentile(F, 95)
    best = (0, 0); s = None
    for i, x in enumerate(np.r_[on, False]):
        if x and s is None: s = i
        if not x and s is not None:
            if i - s > best[1] - best[0]: best = (s, i)
            s = None
    return env.astype(np.float32), best


D = pickle.load(open('data/win.pkl', 'rb'))
D = {k: v for k, v in D.items() if v['ds'] != 'MATWI'}
for name, r in lab.items():
    w = np.array(r['wear']) / 1000.0
    if not (w[0] < VSTAR <= w.max()): print('excluded', name, 'start', w[0], 'end', w.max()); continue
    envs, segs = [], []
    for f in r['files']:
        p = os.path.join(EXT, f.split('MATWI/', 1)[1])
        e, sg = run_env(p); envs.append(e); segs.append(sg)
    cut = np.array([(b - a) / (1636.0 / BLK) for a, b in segs])          # cutting seconds per run
    t = np.cumsum(cut)
    i = int(np.argmax(w >= VSTAR)); T = t[i] if i == 0 else t[i - 1] + (VSTAR - w[i - 1]) / (w[i] - w[i - 1] + 1e-12) * (t[i] - t[i - 1])
    Xtr, ttr, wtr, Xte = [], [], [], []
    for k, (e, (a, b)) in enumerate(zip(envs, segs)):
        for s in range(a, b - L + 1, STRIDE): Xtr.append(e[s:s + L]); ttr.append(t[k]); wtr.append(w[k])
        c0 = max(a, (a + b) // 2 - L // 2); Xte.append(e[c0:c0 + L] if c0 + L <= len(e) else e[-L:])
    tag = 'X' + name[1:]
    D[tag] = dict(ds='MATWI', Xtr=np.array(Xtr, np.float32), ttr=np.array(ttr), wtr=np.array(wtr), Xte=np.array(Xte, np.float32),
                  tte=t, wte=w, c=cond(r['set']), fs=25., T=float(T), nb=1, stride=1)
    print(tag, D[tag]['c'], 'runs', len(t), 'cut s/run %.1f' % np.median(cut), 'T %.0f' % T, 'v0 %.3f' % w[0],
          'neval', int(((t < T) & (np.arange(len(t)) >= 1)).sum()), 'Xtr', len(Xtr), flush=True)
pickle.dump(D, open('data/win.pkl', 'wb'))
