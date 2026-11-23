"""Table: generality check on C-MAPSS FD001 (100 engines, 5-fold engine-level CV with nested 4-fold CV)."""
import pickle,numpy as np
E=pickle.load(open('out/cmapss_eval.pkl','rb')); R=E['res']
rows=[('DE','Deep ensemble (DE)'),('QR','Quantile ensemble (QR)'),('QR+CP(window)','QR + window-level split conformal'),('QR+CP(unit)','QR + unit-level two-layer conformal'),
      ('Fusion-iid','Fusion, independent errors'),('Fusion-iid+CP(unit)','\\quad + unit-level conformal'),
      ('Fusion-persistent','Fusion, persistent-bias errors'),('Fusion-persistent+CP(window)','\\quad + window-level split conformal'),('Fusion-persistent+CP(unit)','\\quad + unit-level conformal')]
P=E["params"]
def rng(k):
    a,b=min(p[k] for p in P),max(p[k] for p in P)
    return f"{a:.2f}" if f"{a:.2f}"==f"{b:.2f}" else f"{a:.2f}$--${b:.2f}"
L=[r'\begin{table}[!tbp]',r'\centering',
   r'\caption{Generality check on 100 run-to-failure turbofan engines (C-MAPSS FD001; five-fold cross-validation by engine, nested four-fold cross-validation for the error model and the conformal scores; nominal level 90\%). Fusion: grid filter over the engine life with the quantile-ensemble median as evidence and the population life distribution as prior. Error model identified with Proposition~2: $\kappa='+rng('kappa')+r'$, $\phi='+rng('phi')+r'$, $\psi='+rng('psi')+r'$ (upper clip); median residual autocorrelation 0.82 at lag 10 and 0.56 at lag 40 cycles. Last column: engines covered on fewer than 80\% of their windows.}',
   r'\label{tab:cmapss}',r'\footnotesize\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lrrrrrr}',r'\toprule',
   r'Method & $n$MAE (\%) & RMSE (cycles) & PICP (\%) & NMPIW (\%) & NIS (\%) & $<$80\%\\',r'\midrule']
OUT={}
for m,lab in rows:
    a={k:np.mean([r[m][k] for r in R.values()]) for k in ['nMAE','RMSE','PICP','NMPIW','NIS']}; n80=sum(r[m]['PICP']<80 for r in R.values()); OUT[m]=dict(a,n80=n80)
    L.append(f"{lab} & {a['nMAE']:.1f} & {a['RMSE']:.1f} & {a['PICP']:.1f} & {a['NMPIW']:.1f} & {a['NIS']:.1f} & {n80}\\\\")
L+=[r'\bottomrule',r'\end{tabular}',r'\end{table}']
open('tables/tab_cmapss.tex','w').write('\n'.join(L)+'\n'); pickle.dump(OUT,open('out/v5_cmapss_summary.pkl','wb'))
print('\n'.join(L[-12:]))
