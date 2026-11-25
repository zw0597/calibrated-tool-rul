"""Fig. 9 (v1.4): over-confidence law. Coverage of the nominal 90% interval of independent-error averaging of n windows,
after window-level recalibration (error scale kappa from the inner folds): prediction of Proposition 6 (lines) and
outer-unit empirical coverage (markers), per dataset.  python f9_law.py -> figures/fig9_law.pdf/.png"""
import pickle, numpy as np, os
from style2 import plt, panel, W2, MM, INK, MID, LIGHT
L = pickle.load(open('out/v7_law.pkl', 'rb'))
for f in ['out/v7_law_ext.pkl', 'out/v7_law_matwi.pkl']:
    if os.path.exists(f): L.update(pickle.load(open(f, 'rb')))
DS = [('PHM', 'PHM2010'), ('NUAA', 'NUAA'), ('MILL', 'NASA milling'), ('MATWI', 'MATWI'), ('SCIDATA', 'Independent tools'), ('CMAPSS', 'C-MAPSS FD001')]
DS = [d for d in DS if any(k[0] == d[0] for k in L)]
SRC = {'rul': ('#1b5e8c', 'RUL evidence'), 'inc': ('#b2442c', 'Wear-increment evidence')}
fig, axs = plt.subplots(1, len(DS), figsize=(W2, 44 * MM), sharey=True)
for i, (ax, (ds, nm)) in enumerate(zip(axs, DS)):
    ax.axhline(0.9, color=LIGHT, lw=0.6, ls=(0, (3, 2)), zorder=0)
    for src, (col, lab) in SRC.items():
        rows = {}
        for k, v in L.items():
            if k[0] == ds and k[2] == src: rows.setdefault(k[3], []).append(v)
        ns = sorted(n for n, v in rows.items() if len(v) >= max(3, 0.5 * max(len(x) for x in rows.values())))
        if not ns: continue
        pr = [np.mean([a['pred_rc'] for a in rows[n]]) for n in ns]; em = [np.mean([a['emp_rc'] for a in rows[n]]) for n in ns]
        ax.plot(ns, pr, color=col, lw=1.0, label=lab + ', predicted')
        ax.plot(ns, em, 'o', color=col, ms=3.2, mec='white', mew=0.5, label=lab + ', observed')
    ax.set_xscale('log'); ax.set_xlim(0.8, 260); ax.set_xticks([1, 10, 100]); ax.set_xticklabels(['1', '10', '100']); ax.xaxis.set_minor_formatter(plt.NullFormatter()); ax.set_ylim(0, 1); ax.set_xlabel('Windows averaged, $n$')
    ax.set_title(nm + f" ({len({k[1] for k in L if k[0]==ds})})", color=INK)
    panel(ax, 'abcdefgh'[i], x=-0.2 if i == 0 else -0.1)
axs[0].set_ylabel('Coverage of nominal 90% interval')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.12))
fig.tight_layout(w_pad=0.6)
fig.savefig('figures/fig9_law.pdf'); fig.savefig('figures/fig9_law.png', dpi=300)
