"""Table: generality checks on an independent tool data set (SciData 2025, 13 tools) and on C-MAPSS FD001 (100 engines)."""
import pickle, numpy as np
rows=[('DE','Deep ensemble (DE)'),('QR','Quantile ensemble (QR)'),('QR+CP(window)','QR + window-level split conformal'),('QR+CP(unit)','QR + unit-level two-layer conformal'),
      ('Fusion-iid','Fusion, independent errors'),('Fusion-iid+CP(unit)','\\quad + unit-level conformal'),
      ('Fusion-persistent','Fusion, persistent-bias errors'),('Fusion-persistent+CP(window)','\\quad + window-level split conformal'),('Fusion-persistent+CP(unit)','\\quad + unit-level conformal')]
E={'sci':pickle.load(open('out/scidata_eval.pkl','rb')),'cm':pickle.load(open('out/cmapss_eval.pkl','rb'))}
def rng(P,k):
    a,b=min(p[k] for p in P),max(p[k] for p in P); return f"{a:.2f}" if f"{a:.2f}"==f"{b:.2f}" else f"{a:.2f}$--${b:.2f}"
cap=(r'Generality checks (nominal level 90\%). Left: end-milling tool-life data set~\cite{scidata25} (13 run-to-failure tools, leave-one-tool-out with nested leave-one-tool-out; RUL in cycles). Right: turbofan engines (C-MAPSS FD001~\cite{saxena08}, 100 engines, five-fold cross-validation by engine with nested four-fold cross-validation). '
     r'Neither data set has wear measurements, so the wear law is replaced by the population life distribution; fusion: grid filter over the unit life with the quantile-ensemble median as evidence. Error model identified with Proposition~2: tools $\kappa='+rng(E['sci']['params'],'kappa')+r'$, $\phi='+rng(E['sci']['params'],'phi')+r'$, $\psi='+rng(E['sci']['params'],'psi')+r'$; engines $\kappa='+rng(E['cm']['params'],'kappa')+r'$, $\phi='+rng(E['cm']['params'],'phi')+r'$, $\psi='+rng(E['cm']['params'],'psi')+r'$ (upper clip). $<$80\%: units covered on fewer than 80\% of their windows.')
L=[r'\begin{table*}[!tbp]',r'\centering',r'\caption{'+cap+'}',r'\label{tab:cmapss}',r'\footnotesize\setlength{\tabcolsep}{3.2pt}',r'\begin{tabular}{lrrrrrrrrr}',r'\toprule',
   r' & \multicolumn{4}{c}{Milling tools (13)} & & \multicolumn{4}{c}{Turbofan engines (100)}\\',r'\cmidrule(lr){2-5}\cmidrule(lr){7-10}',
   r'Method & $n$MAE & PICP & NIS & $<$80\% & & $n$MAE & PICP & NIS & $<$80\%\\',r'\midrule']
for m,lab in rows:
    row=[lab]
    for key in ['sci','cm']:
        R=E[key]['res']; a={q:np.mean([r[m][q] for r in R.values()]) for q in ['nMAE','PICP','NIS']}; n80=sum(r[m]['PICP']<80 for r in R.values())
        row+=[f"{a['nMAE']:.1f}",f"{a['PICP']:.1f}",f"{a['NIS']:.1f}",str(n80)]
        if key=='sci': row.append('')
    L.append(' & '.join(row)+r'\\')
L+=[r'\bottomrule',r'\end{tabular}',r'\end{table*}']
open('tables/tab_cmapss.tex','w').write('\n'.join(L)+'\n'); print('\n'.join(L[-12:]))
