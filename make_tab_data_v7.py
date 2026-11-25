"""v1.4: dataset table with MATWI rows. python make_tab_data_v7.py <v6 tab_data.tex> -> tables/tab_data.tex"""
import sys, re, pickle, numpy as np
src = open(sys.argv[1]).read()
body = src.split(r'\midrule')[1].split(r'\bottomrule')[0].strip().split('\n')
rows = []
for l in body:
    c = [x.strip() for x in l.rstrip('\\').split('&')]
    for h in (c[:7], c[7:]):
        if h and h[0]: rows.append(h)
order = {'PHM': 0, 'NUAA': 1, 'Mill': 2}
rows.sort(key=lambda r: (order[r[0]], r[1]))
D = pickle.load(open('data/win.pkl', 'rb'))
for k, v in D.items():
    if v['ds'] != 'MATWI': continue
    t = v['tte']; ne = int(((t < v['T']) & (np.arange(len(t)) >= 1)).sum()); c = v['c']
    mat = 'CK' if c[2] == 1 else 'SS'; star = '$^\\ast$' if k == 'X01' else ''
    rows.append(['MATWI', k, f'{mat}/{c[0]:.0f}/{c[1]:.3f}'.replace('/0.', '/.') + star, f'{v["wte"][0]*1000:.0f}', f'{v["T"]:.0f}', f'{t[-1]:.0f}', str(ne)])
h = (len(rows) + 1) // 2
L = src.split(r'\midrule')[0] + '\\midrule\n'
for i in range(h):
    a = rows[i]; b = rows[i + h] if i + h < len(rows) else [''] * 7
    L += ' & '.join(a + b) + '\\\\\n'
L += '\\bottomrule' + src.split(r'\bottomrule')[1]
L = L.replace(r"and depth of cut $a_p$ (mm).", r"and depth of cut $a_p$ (mm); MATWI, workpiece (CK: CK45 steel, SS: AISI 304 stainless steel), cutting speed $V_c$ (m/min) and feed per tooth $f_z$ (mm); $^\ast$conditions not recorded, imputed by the median of the other CK45 sets.")
open('tables/tab_data.tex', 'w').write(L)
print(L)
