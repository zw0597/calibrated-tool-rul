"""Analyses of the revised manuscript (v1.3) that use stored predictions only.

  python v6_analysis.py cqrplus     # CQR+: QR+ with the two-layer conformal step (nested QR+ scores) -> out/v6_cqrplus.pkl
  python v6_analysis.py decision    # replacement-policy study                                       -> out/v6_decision.pkl
  python v6_analysis.py tests       # paired tests with Holm correction, nMAE and NIS                -> out/v6_tests.pkl
"""
import sys, pickle, numpy as np
from scipy.stats import wilcoxon
from pipeline import tags, D, tfail, rul_true, emask, conformal_Q, cqr_score, cqr_apply, INSP
from metrics import tool_metrics
DSS = ['PHM', 'NUAA', 'MILL']


def cqrplus():
    RAW = pickle.load(open('out/v5_deepaux_raw.pkl', 'rb')); RI = pickle.load(open('out/v5_plus_inner.pkl', 'rb')); out = {}
    for ds in DSS:
        T = tags(ds)
        for k in T:
            tr = [t for t in T if t != k]
            for scen in ['none', 'insp']:
                q = np.mean([RAW[(ds, k, scen, 'qr', s)] for s in range(5)], 0)
                sc = []
                for j in tr:
                    qj = np.mean([RI[(ds, scen, k, j, s)] for s in range(2)], 0); m = emask(j)
                    sc.append(cqr_score(rul_true(j)[m], qj[m, 0], qj[m, 4]))
                Q = conformal_Q(sc); lo, hi = cqr_apply(np.maximum(q[:, 0], 0), np.maximum(q[:, 4], 0), Q)
                out.setdefault(ds, {}).setdefault(k, {})[scen] = {'CQR+': dict(R=np.maximum(q[:, 2], 0), lo=lo, hi=hi, Q=Q)}
    pickle.dump(out, open('out/v6_cqrplus.pkl', 'wb')); return out


def preds(ds, t, scen, m):
    """predictive (R, lo, hi) of method m for tool t"""
    from make_tables_v5 import agg  # noqa (ensures loaders)
    import make_tables_v5 as M
    if m in ('DE+', 'QR+'): return M.DA[ds][t][scen][m]
    if m == 'CQR+': return pickle.load(open('out/v6_cqrplus.pkl', 'rb'))[ds][t][scen]['CQR+']
    if m == 'WienerC': return pickle.load(open('out/v6_wiener_cov.pkl', 'rb'))[ds][t][scen]
    if m == 'PFC': return pickle.load(open('out/v6_pf_cov.pkl', 'rb'))[ds][t][scen + '_insponly']
    if m == 'PF': return M.GP[ds][t][scen + '_insponly']
    if m == 'PFNet': return M.GP[ds][t][scen]
    if m == 'WienerNetK': return M.HY[ds][t][scen][m]
    O = pickle.load(open(f'out/rul2_{ds}.pkl', 'rb'))[t]
    return O[scen][m] if m in O[scen] else O[m]


def decision(methods=('DE', 'DE+', 'CQR', 'CQR+', 'Wiener', 'WienerC', 'PF', 'PFC', 'MCprior', 'Ours', 'OursCP'), leads=(0.05, 0.10, 0.20), scen='insp'):
    """Replace the tool at the first evaluation window at which the lower 5% RUL quantile falls below a lead time
    L = lead * mean training life. Failure if no replacement before T. Wasted life = (T - t_rep)/T for timely replacements."""
    res = {}
    for lead in leads:
        for m in methods:
            per = []
            for ds in DSS:
                T_ = tags(ds)
                for t in T_:
                    L = lead * np.mean([tfail(j) for j in T_ if j != t])
                    p = preds(ds, t, scen, m); mk = emask(t); tt = D[t]['tte'][mk]; lo = np.maximum(p['lo'][mk], 0)
                    hit = np.where(lo <= L)[0]; Tf = tfail(t)
                    if len(hit): trep = tt[hit[0]]; fail = trep > Tf; waste = max(Tf - trep, 0) / Tf
                    else: fail = True; waste = 0.0
                    per.append(dict(ds=ds, t=t, fail=float(fail), waste=waste * 100))
            res[(lead, m)] = per
    pickle.dump(res, open('out/v6_decision.pkl', 'wb'))
    for lead in leads:
        print('lead', lead)
        for m in methods:
            per = res[(lead, m)]
            print(f'  {m:8s} failures {int(sum(p["fail"] for p in per)):2d}/27  wasted life {np.mean([p["waste"] for p in per]):5.1f}% '
                  + ' '.join(f'{ds}:{int(sum(p["fail"] for p in per if p["ds"]==ds))}/{np.mean([p["waste"] for p in per if p["ds"]==ds]):.0f}%' for ds in DSS))
    return res


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * ps[i]); adj[i] = min(run, 1)
    return adj


def tests(baselines=('DE', 'DE+', 'MCD', 'CQRraw', 'CQR', 'CQR+', 'Wiener', 'WienerC', 'WienerNetK', 'PF', 'PFC', 'MCprior'), scen='insp'):
    rng = np.random.default_rng(5); out = {}
    def met(ds, t, m, key):
        if m == 'Ours' and key == 'NIS': m = 'OursCP'
        return tool_metrics(t, preds(ds, t, scen, m))[key]
    for key in ['nMAE', 'NIS', 'late']:
        rows = []
        for b in baselines:
            if key == 'NIS' and b in ('CQRraw',): continue
            d = []
            for ds in DSS:
                for t in tags(ds):
                    if key == 'late':
                        a_ = tool_metrics(t, preds(ds, t, scen, 'Ours'))['MAElate']; b_ = tool_metrics(t, preds(ds, t, scen, b))['MAElate']; d.append((a_ - b_) / tfail(t) * 100)
                    else: d.append(met(ds, t, 'Ours', key) - met(ds, t, b, key))
            d = np.array(d); i = rng.integers(0, len(d), (10000, len(d))); bm = d[i].mean(1)
            rows.append([b, d.mean(), np.percentile(bm, 2.5), np.percentile(bm, 97.5), wilcoxon(d).pvalue, int((d < 0).sum())])
        adj = holm(np.array([r[4] for r in rows]))
        for r, a in zip(rows, adj): r.append(a)
        out[key] = rows
        print('==', key)
        for r in rows: print(f'  {r[0]:10s} {r[1]:8.2f} [{r[2]:8.2f},{r[3]:8.2f}] p={r[4]:.4f} holm={r[6]:.4f} wins {r[5]}/27')
    pickle.dump(out, open(f'out/v6_tests_{scen}.pkl', 'wb')); return out


if __name__ == '__main__':
    w = sys.argv[1]
    if w == 'cqrplus':
        o = cqrplus()
        for scen in ['insp', 'none']:
            for ds in DSS:
                per = [tool_metrics(t, o[ds][t][scen]['CQR+']) for t in tags(ds)]
                print(scen, ds, {k: round(np.mean([p[k] for p in per]), 1) for k in ['nMAE', 'PICP', 'NMPIW', 'NIS']}, 'cov<80:', sum(p['PICP'] < 80 for p in per))
    elif w == 'decision': decision()
    elif w == 'tests': tests(); tests(scen='none')
    elif w == 'cost': costrate()


def costrate(methods=('DE', 'DE+', 'CQR', 'CQR+', 'Wiener', 'WienerC', 'PF', 'PFC', 'MCprior', 'Ours', 'OursCP'),
             leads=(0.025, 0.05, 0.10, 0.15, 0.20, 0.30), ratios=(5, 10, 20), scen='insp'):
    """Renewal-reward cost rate of the lower-bound replacement rule: planned replacement costs 1, failure costs r;
    cost rate = sum(costs)/sum(used life), relative to the ideal policy (replacement exactly at failure, cost 1)."""
    out = {}
    for m in methods:
        rec = {}
        for lead in leads:
            cost = {r: 0.0 for r in ratios}; used = 0.0; ideal = 0.0
            for ds in DSS:
                T_ = tags(ds)
                for t in T_:
                    L = lead * np.mean([tfail(j) for j in T_ if j != t]); p = preds(ds, t, scen, m); mk = emask(t)
                    tt = D[t]['tte'][mk]; lo = np.maximum(p['lo'][mk], 0); Tf = tfail(t); hit = np.where(lo <= L)[0]
                    fail = (len(hit) == 0) or tt[hit[0]] > Tf; u = Tf if fail else tt[hit[0]]
                    used += u; ideal += Tf
                    for r in ratios: cost[r] += r if fail else 1.0
            n = 27
            for r in ratios: rec[(lead, r)] = (cost[r] / used) / (n / ideal)
        out[m] = rec
    pickle.dump(out, open('out/v6_costrate.pkl', 'wb'))
    for r in ratios:
        print('cost ratio', r)
        for m in methods:
            v = {l: out[m][(l, r)] for l in leads}; lb = min(v, key=v.get)
            print(f'  {m:8s} best L={lb:.3f} rel.cost {v[lb]:.2f}   at L=0.05: {v[0.05]:.2f}  L=0.10: {v[0.10]:.2f}')
    return out
