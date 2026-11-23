import pickle
O=pickle.load(open('out/v6_costrate.pkl','rb'))
M=[('DE','DE'),('DE+','DE+'),('CQR','CQR'),('CQR+','CQR+'),('Wiener','Wiener'),('WienerC','Wiener-C'),('PF','PF'),('PFC','PF-C'),('MCprior','Law'),('Ours','Proposed-raw'),('OursCP','Proposed')]
leads=(0.025,0.05,0.10,0.15,0.20,0.30)
L=[r'\begin{table}[!tbp]',r'\centering',r'\caption{Long-run cost rate of the lower-bound replacement rule of Table~\ref{tab:decision} (renewal-reward: planned replacement costs 1, a failure costs $r$; total cost divided by total used life over the 27 tools, relative to replacing every tool exactly at failure, so that 1 is ideal). Best: lowest cost rate over the lead times $L\in\{2.5,5,10,15,20,30\}\%$ (chosen in hindsight, equally for all methods), with the corresponding $L$.}',
   r'\label{tab:cost}',r'\footnotesize\setlength{\tabcolsep}{3.5pt}',r'\begin{tabular}{lcccccc}',r'\toprule',r' & \multicolumn{3}{c}{$r=10$} & \multicolumn{3}{c}{$r=20$}\\',r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
   r'Method & $L=5\%$ & $L=10\%$ & best ($L$) & $L=5\%$ & $L=10\%$ & best ($L$)\\',r'\midrule']
for m,nm in M:
    row=[nm]
    for r in (10,20):
        v={l:O[m][(l,r)] for l in leads}; lb=min(v,key=v.get)
        row+=[f'{v[0.05]:.2f}',f'{v[0.10]:.2f}',f'{v[lb]:.2f} ({100*lb:g}\\%)']
    L.append(' & '.join(row)+r'\\')
L+=[r'\bottomrule',r'\end{tabular}',r'\end{table}']
open('tables/tab_cost.tex','w').write('\n'.join(L)+'\n')
