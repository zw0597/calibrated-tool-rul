"""v1.4 Figs. 6 and 7 with the four tool datasets (incl. MATWI) and the stacked proposed method.
  python f67_v7.py -> figures/fig6_tools.pdf/.png, figures/fig7_calib.pdf/.png"""
import pickle, numpy as np, functools
from style2 import *
from v2 import *
from metrics import tool_metrics
import v6_analysis as V
import make_tables_v7 as T
import matplotlib.gridspec as gs
from matplotlib.colors import LogNorm
from pipeline import tags
DSC['MATWI'] = '#3d7a3a'; DNAME['MATWI'] = 'MATWI'
C['Framework'] = '#7fb0d8'; C['Framework alone'] = '#7fb0d8'; C['DE+CP'] = '#6b3a0a'
DS = ['PHM', 'NUAA', 'MILL', 'MATWI']
ST = pickle.load(open('out/v7_stack.pkl', 'rb'))
@functools.lru_cache(None)
def get(ds, t, m):
    if m in ('Stack', 'StackCP'): return ST[ds][t]['insp'][m]
    return V.preds(ds, t, 'insp', m)
tools = [(ds, t) for ds in DS for t in tags(ds)]
nd = {ds: len(tags(ds)) for ds in DS}; edges = np.cumsum([nd[d] for d in DS])
# ---------------- Fig. 6
MP = [('Pop', 'Pop'), ('DE', 'DE'), ('DE+', 'DE+'), ('MCD', 'MCD'), ('CQRraw', 'QR'), ('QR+', 'QR+'), ('Wiener', 'Wie-\nner'), ('WienerC', 'Wie-\nner-C'),
      ('PFC', 'PF-C'), ('MCprior', 'Law'), ('OursCP', 'Frame-\nwork'), ('StackCP', 'Prop.')]
fig = plt.figure(figsize=(W2, 150 * MM)); G = gs.GridSpec(4, 2, figure=fig, width_ratios=[1.0, 1.0], wspace=0.34, hspace=0.62)
ax = fig.add_subplot(G[:, 0])
A = np.array([[tool_metrics(t, get(ds, t, m))['nMAE'] for m, _ in MP] for ds, t in tools])
norm = LogNorm(2, 300); ax.imshow(A, cmap='Blues', norm=norm, aspect='auto')
cmp_cols = [j for j, (m, _) in enumerate(MP) if m != 'OursCP']           # best among methods, framework alone shown for reference
for i in range(A.shape[0]):
    b = min(A[i, j] for j in cmp_cols)
    for j in range(A.shape[1]):
        v = A[i, j]; ax.text(j, i, f'{v:.1f}' if v < 100 else f'{v:.0f}', ha='center', va='center', fontsize=4.4,
                             color='white' if norm(v) > 0.62 else INK, fontweight='bold' if (v == b and j in cmp_cols) else 'normal')
ax.set_xticks(range(len(MP))); ax.set_xticklabels([n for _, n in MP], fontsize=5.0); ax.xaxis.tick_top()
ax.set_yticks(range(len(tools))); ax.set_yticklabels([t for _, t in tools], fontsize=5.0); ax.tick_params(length=0)
[sp.set_visible(False) for sp in ax.spines.values()]
for b in edges[:-1]: ax.axhline(b - 0.5, color='white', lw=2.2)
ax.axvline(len(MP) - 2.5, color='white', lw=1.5)
y0 = 0
for ds in DS:
    ax.text(-2.0, y0 + (nd[ds] - 1) / 2, DNAME[ds], rotation=90, ha='center', va='center', fontsize=6, color=DSC[ds]); y0 += nd[ds]
ax.set_xlabel(r'$n$MAE (%) per test tool, periodic inspections (bold: best method per tool)', fontsize=6.0, labelpad=4)
panel(ax, 'a', x=-0.3, y=1.03)
win = {}
for i in range(A.shape[0]):
    j = min(cmp_cols, key=lambda j: A[i, j]); win[MP[j][0]] = win.get(MP[j][0], 0) + 1
print('fig6 best counts', sorted(win.items(), key=lambda x: -x[1]))
for j, ds in enumerate(DS):
    ax = fig.add_subplot(G[j, 1]); nb = 5 if ds == 'MILL' else 10; bb = np.linspace(0, 1, nb + 1); cc = (bb[1:] + bb[:-1]) / 2
    for m, nm, col, ls in [('DE', 'DE', C['DE'], '-'), ('DE+', 'DE+', C['DE+'], (0, (4, 1, 1, 1))), ('CQRraw', 'QR', C['QR'], '-'), ('Wiener', 'Wiener', C['Wiener'], (0, (3, 1.5))),
                           ('MCprior', 'Law', C['Law'], (0, (1, 1))), ('StackCP', 'Proposed', C['Proposed'], '-')]:
        E = np.full((nd[ds], nb), np.nan)
        for i, t in enumerate(tags(ds)):
            mk = emask(t); fr = D[t]['tte'][mk] / tfail(t); e = np.abs(get(ds, t, m)['R'][mk] - rul_true(t)[mk]) / tfail(t) * 100
            for b in range(nb):
                s = (fr >= bb[b]) & (fr < bb[b + 1])
                if s.any(): E[i, b] = e[s].mean()
        mu = np.nanmean(E, 0); ax.plot(cc, mu, color=col, ls=ls, lw=1.4 if nm == 'Proposed' else 0.9, marker='o' if nm == 'Proposed' else None, ms=2.4, label=nm)
        if nm == 'Proposed': ax.fill_between(cc, np.nanmin(E, 0), np.nanmax(E, 0), color=C['Proposed'], alpha=0.12, lw=0)
        if nm in ('Proposed', 'DE+'): print(ds, nm, 'bins', np.round(mu, 0))
    ax.set_xlim(0, 1); ax.set_ylim(0, {'PHM': 40, 'NUAA': 90, 'MILL': 80, 'MATWI': 300}[ds]); ax.set_ylabel('Abs. error (% of life)', fontsize=6)
    if j == 3: ax.set_xlabel(r'Life fraction $t/T$')
    ax.set_title(f'{DNAME[ds]}: mean error along the life', loc='left', fontsize=6.2)
    if j == 0: ax.legend(loc='lower left', bbox_to_anchor=(0, 1.2), ncol=6, fontsize=5.3, columnspacing=0.7, handlelength=1.3)
    panel(ax, 'bcde'[j], x=-0.17)
save(fig, 'fig6_tools')
# ---------------- Fig. 7
MS = [('DE', 'DE', 'o', True), ('DE+', 'DE+', 'o', False), ('DE+CP', 'DE+CP', 'p', True), ('MCD', 'MCD', 'v', True), ('CQRraw', 'QR', 'D', False), ('CQR', 'CQR', 'D', True), ('CQR+', 'CQR+', 'h', True),
      ('Wiener', 'Wiener', 's', True), ('MCprior', 'Law', 'P', True), ('OursCP', 'Framework alone', '*', False), ('Stack', 'Proposed-raw', '*', False), ('StackCP', 'Proposed', '*', True)]
fig = plt.figure(figsize=(W2, 120 * MM)); G = gs.GridSpec(2, 4, figure=fig, wspace=0.36, hspace=0.5, top=0.92)
XL = {'PHM': (5, 120, [5, 10, 20, 50, 100]), 'NUAA': (40, 700, [50, 100, 200, 500]), 'MILL': (40, 700, [50, 100, 200, 500]), 'MATWI': (100, 4000, [100, 300, 1000, 3000])}
for j, ds in enumerate(DS):
    ax = fig.add_subplot(G[0, j]); ax.axhline(90, color=MID, lw=0.6, ls=(0, (3, 2)))
    lo, hi, tk = XL[ds]
    for m, nm, mk, fill in MS:
        a = T.agg(ds, 'insp', m)[0]; col = C[nm]; x = min(max(a['NMPIW'], lo), hi)
        ax.scatter(x, a['PICP'], s=40 if mk == '*' else 18, marker=mk, facecolors=col if fill else 'white', edgecolors=col, linewidths=0.8, zorder=3, label=nm, clip_on=False)
        if a['NMPIW'] > hi: ax.annotate(f'{a["NMPIW"]:.0f}→', (hi, a['PICP']), xytext=(-4, -8), textcoords='offset points', fontsize=4.8, color=col, ha='right')
    ax.set_xscale('log'); ax.set_ylim(0, 105); ax.set_xlim(lo, hi); ax.set_xticks(tk); ax.set_xticklabels([str(v) for v in tk]); ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel('NMPIW (% of life)', fontsize=6)
    if j == 0: ax.set_ylabel('Coverage PICP (%)')
    ax.set_title(DNAME[ds], loc='left', fontsize=6.2); panel(ax, 'abcd'[j], x=-0.25)
    if j == 0: fig.legend(*ax.get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(0.5, 0.945), ncol=12, fontsize=4.8, handletextpad=0.15, columnspacing=0.7)
ax = fig.add_subplot(G[1, :])
for k, (m, nm, mk, fill) in enumerate([('DE', 'DE', 'o', True), ('DE+', 'DE+', 'o', False), ('CQR+', 'CQR+', 'h', True), ('Wiener', 'Wiener', 's', True), ('StackCP', 'Proposed', '*', True)]):
    v = np.array([tool_metrics(t, get(ds, t, m))['PICP'] for ds, t in tools]); x = np.arange(len(tools)) + (k - 2) * 0.15
    ax.scatter(x, v, s=22 if mk == '*' else 9, marker=mk, facecolors=C[nm] if fill else 'white', edgecolors=C[nm] if not fill else 'white', linewidths=0.6 if not fill else 0.3, label=nm, zorder=3)
    print('fig7 tools<80', nm, int((v < 80).sum()), 'zero', int((v == 0).sum()), '>=97', int((v >= 97).sum()))
ax.axhline(90, color=MID, lw=0.6, ls=(0, (3, 2)))
x0 = 0
for ds in DS:
    if x0: ax.axvline(x0 - 0.5, color=LIGHT, lw=0.6)
    ax.text(x0 + (nd[ds] - 1) / 2, -0.2, DNAME[ds], ha='center', fontsize=6, color=DSC[ds], clip_on=False, transform=ax.get_xaxis_transform()); x0 += nd[ds]
ax.set_xticks(range(len(tools))); ax.set_xticklabels([t for _, t in tools], fontsize=4.6, rotation=90); ax.set_ylim(-3, 103); ax.set_xlim(-0.6, len(tools) - 0.4)
ax.set_ylabel('Per-tool coverage (%)')
ax.legend(loc='lower left', ncol=5, fontsize=5.6, columnspacing=0.8, handletextpad=0.1, bbox_to_anchor=(0.0, 1.0), borderaxespad=0.1); panel(ax, 'e', x=-0.06)
save(fig, 'fig7_calib')
