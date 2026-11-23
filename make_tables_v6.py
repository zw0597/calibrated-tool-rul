import pickle,numpy as np,sys
from v2 import *
from metrics import tool_metrics
from metrics2 import agg as _agg
HY=pickle.load(open('out/v4_hybrid.pkl','rb')); AB=pickle.load(open('out/v4_ablate.pkl','rb'))
import os
DA=pickle.load(open('out/v5_deepaux.pkl','rb')) if os.path.exists('out/v5_deepaux.pkl') else None
GP=pickle.load(open('out/v5_genpath.pkl','rb'))
def _nom(ds): return pickle.load(open(f'out/v4_backup/rul2_{ds}.pkl','rb'))
PE=pickle.load(open('out/v5_plus_eval.pkl','rb')) if os.path.exists('out/v5_plus_eval.pkl') else None
CQP=pickle.load(open('out/v6_cqrplus.pkl','rb')); WC=pickle.load(open('out/v6_wiener_cov.pkl','rb')); PFC_=pickle.load(open('out/v6_pf_cov.pkl','rb'))
def agg(ds,scen,m):
    if m in ('CQR+','WienerC','PFC'):
        src={'CQR+':lambda t: CQP[ds][t][scen]['CQR+'],'WienerC':lambda t: WC[ds][t][scen],'PFC':lambda t: PFC_[ds][t][scen+'_insponly']}[m]
        per=[tool_metrics(t,src(t)) for t in tags(ds)]
        return {k:np.nanmean([q[k] for q in per]) for k in per[0]}, per
    if m in ('Plus','PlusCP'):
        per=[PE[ds][t][scen][m] for t in tags(ds)]
        return {k:np.nanmean([q[k] for q in per]) for k in per[0]}, per
    if m in ('DE+','QR+'):
        per=[tool_metrics(t,DA[ds][t][scen][m]) for t in tags(ds)]
        return {k:np.nanmean([q[k] for q in per]) for k in per[0]}, per
    if m in ('PF','PFNet'):
        per=[tool_metrics(t,GP[ds][t][scen if m=='PFNet' else scen+'_insponly']) for t in tags(ds)]
        return {k:np.nanmean([q[k] for q in per]) for k in per[0]}, per
    if m=='v4nom':
        O=_nom(ds); per=[tool_metrics(t,O[t][scen]['Ours']) for t in tags(ds)]
        return {k:np.nanmean([q[k] for q in per]) for k in per[0]}, per
    if m=='WienerNetK':
        per=[tool_metrics(t,HY[ds][t][scen][m]) for t in tags(ds)]
    elif m.startswith('v4:'):
        per=[tool_metrics(t,AB[m[3:]][ds][scen][t]) for t in tags(ds)]
    else: return _agg(ds,scen,m)
    return {k:np.nanmean([q[k] for q in per]) for k in per[0]}, per
NAMES=[('Pop','Pop'),('DE','DE'),('DE+','DE+'),('MCD','MCD'),('CQRraw','QR'),('QR+','QR+'),('CQR','CQR'),('CQR+','CQR+'),('LinEx','LinEx'),('Wiener','Wiener'),('WienerC','Wiener-C'),('WienerNetK','Wiener-Net'),('PF','PF'),('PFC','PF-C'),('PFNet','PF-Net'),('MCprior','Law'),('Ours','Proposed-raw'),('OursCP','Proposed'),('PlusCP','Proposed, QR+ evidence$^\ddagger$')]
def fmt(v,d=1): return '--' if v is None or not np.isfinite(v) else f'{v:.{d}f}'
DSS=['PHM','NUAA','MILL']; DNAME={'PHM':'PHM2010 (3 tools, 1 condition)','NUAA':'NUAA (9 tools, 9 conditions)','MILL':'NASA milling (15 tools, 8 conditions)'}
def main_table(scen,label,caption):
    cols=[('nMAE',2,'min'),('RMSE',0,'min'),('MAElate',0,'min'),('AL',2,'max'),('PICP',1,None),('NMPIW',1,None),('NIS',1,'min')]
    L=[r'\begin{table}[!tbp]',r'\centering',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\scriptsize\setlength{\tabcolsep}{4.5pt}\renewcommand{\arraystretch}{0.93}',
       r'\begin{tabular}{lrrrrrrr}',r'\toprule',
       r'Method & $n$MAE (\%) & RMSE (s) & MAE$_{\rm late}$ (s) & $\alpha$--$\lambda$ & PICP (\%) & NMPIW (\%) & NIS (\%)\\']
    R={}
    for ds in DSS:
        R[ds]={m:agg(ds,scen,m)[0] for m,_ in NAMES}
        best={}
        for c,d,how in cols:
            if how is None: continue
            vals=[float(fmt(R[ds][m].get(c,np.nan),d)) for m,_ in NAMES if m!='LinEx' and np.isfinite(R[ds][m].get(c,np.nan))]
            best[c]=min(vals) if how=='min' else max(vals)
        L.append(r'\midrule'); L.append(r'\multicolumn{8}{@{}l}{\emph{'+DNAME[ds]+r'}}\\')
        for m,nm in NAMES:
            row=[nm]
            for c,d,how in cols:
                v=R[ds][m].get(c,np.nan); s_=fmt(v,d)
                if how and np.isfinite(v) and abs(float(s_)-best[c])<1e-9 and m!='LinEx': s_=r'\textbf{'+s_+'}'
                row.append(s_)
            L.append(' & '.join(row)+r'\\')
        if ds=='MILL':
            from metrics2 import fs_conformal
            fsr=fs_conformal(ds,scen)[0]; R[ds]['OursFS']=fsr
            L.append(' & '.join([r'Proposed (subsampled CP)$^\dagger$']+[fmt(fsr[c],d) for c,d,how in cols])+r'\\')
    L+= [r'\bottomrule',r'\multicolumn{8}{@{}p{0.95\linewidth}@{}}{\scriptsize $^\dagger$Conservative subsampled conformal step of Section~\ref{sec:conformal} (one random window per calibration tool, $Q$ = largest score for $m=14$, averaged over 400 draws); it can be formed only when $m\ge9$ and is therefore reported for NASA milling only. Its finite-sample guarantee (Proposition~5) holds for calibration tools held out from training; in the nested protocol used here its coverage is an empirical property. $^\ddagger$Calibrated framework with the QR+ ensemble as direct-RUL evidence (Section~\ref{sec:res_rul}).}\\',r'\end{tabular}',r'\end{table}']
    open(f'tables/{label.split(":")[1]}.tex','w').write('\n'.join(L)); return R
def ablation():
    V=[('Ours','Proposed (full, raw intervals)'),('woShift','w/o effective-age offset ($\\epsilon\\equiv0$)'),('woBank','w/o hypothesis averaging (single filter)'),
       ('condOnly','extended-Taylor prior only'),('poolOnly','pooled prior only'),
       ('woI','w/o wear-increment evidence'),('woR','w/o direct-RUL evidence'),('woAR','w/o persistent bias ($\\psi_s\\to0.05$)'),('woDisc','w/o model discrepancy $\\delta$'),
       ('singleQ','single quantile network instead of ensemble'),
       ('v4nom','nominal-path RUL (discrepancy not propagated)'),
       ('v4:woPred','w/o small-sample predictive scale (Eq.~\\eqref{eq:predscale})'),('v4:plainP','plain pseudo-BMA (no Bayesian bootstrap)'),('v4:uniP','uniform weights over prior structures'),('MCprior','law + inspections only (Law)')]
    L=[r'\begin{table*}[!tbp]',r'\centering',r'\caption{Ablation study (periodic inspections; raw posterior intervals, before conformal calibration).}',r'\label{tab:ablation}',r'\footnotesize\setlength{\tabcolsep}{3.6pt}',
       r'\begin{tabular}{lccccccccc}',r'\toprule',r' & \multicolumn{3}{c}{PHM2010} & \multicolumn{3}{c}{NUAA} & \multicolumn{3}{c}{NASA milling}\\',r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}',
       r'Variant & $n$MAE & PICP & NIS & $n$MAE & PICP & NIS & $n$MAE & PICP & NIS\\',r' & (\%) & (\%) & (\%) & (\%) & (\%) & (\%) & (\%) & (\%) & (\%)\\',r'\midrule']
    out={}
    for m,nm in V:
        row=[nm]
        for ds in DSS:
            O=pickle.load(open(f'out/rul2_{ds}.pkl','rb'))
            if m=='v4nom': pass
            elif m.startswith('v4:'):
                if ds not in AB[m[3:]]: row+=['--']*3; continue
            elif m not in O[list(O)[0]]['insp']: row+=['--']*3; continue
            a=agg(ds,'insp',m)[0]; out[(ds,m)]=a; row+=[fmt(a['nMAE']),fmt(a['PICP']),fmt(a['NIS'])]
        L.append(' & '.join(row)+r'\\')
    L+=[r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    open('tables/tab_ablation.tex','w').write('\n'.join(L)); return out
def params_table():
    L=[r'\begin{table}[!tbp]',r'\centering',r'\caption{Quantities estimated from the training tools by nested LOTO (mean over outer folds; range in brackets).}',r'\label{tab:params}',r'\small\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lccc}',r'\toprule',r'Quantity & PHM2010 & NUAA & NASA milling\\',r'\midrule']
    rows={}
    for ds in DSS:
        O=pickle.load(open(f'out/rul2_{ds}.pkl','rb')); T=tags(ds)
        for src in ['inc','rul']:
            for p in ['kappa','phi','psi']:
                rows.setdefault((src,p),{})[ds]=[O[t]['params'][src][p] for t in O]
        rows.setdefault(('J',),{})[ds]=[O[t]['insp']['Ours']['J'] for t in O]
        rows.setdefault(('Q',),{})[ds]=[O[t]['insp']['OursCP']['Q'] for t in O]
        sd=[];ell=[];eb=[];se=[]
        for t in T:
            tr=[x for x in T if x!=t]; Lw=law(tr,ds,True); a,b=Lw['disc']; sd.append(a*1e3); ell.append(b); eb.append(Lw['shift'][0]*1e3); se.append(Lw['shift'][1]*1e3)
        rows.setdefault(('sd',),{})[ds]=sd; rows.setdefault(('ell',),{})[ds]=ell; rows.setdefault(('eb',),{})[ds]=eb; rows.setdefault(('se',),{})[ds]=se
    lab={('inc','kappa'):r'$\kappa_I$',('inc','phi'):r'$\phi_I$',('inc','psi'):r'$\psi_I$',('rul','kappa'):r'$\kappa_R$',('rul','phi'):r'$\phi_R$',('rul','psi'):r'$\psi_R$',('sd',):r'$\sigma_\delta$ ($\mu$m)',('ell',):r'$\ell_\delta$ (s)',('eb',):r'$\bar\epsilon$ ($\mu$m)',('se',):r'$\sigma_\epsilon$ ($\mu$m)',('J',):r'segments $J$',('Q',):r'conformal margin $Q$ (periodic)'}
    for k,nm in lab.items():
        row=[nm]
        for ds in DSS:
            v=np.array(rows[k][ds],float)
            d=0 if k in [('ell',),('J',)] else (1 if k in [('sd',),('eb',),('se',)] else 2)
            row.append(f'{v.mean():.{d}f} [{v.min():.{d}f}, {v.max():.{d}f}]'.replace('-','$-$'))
        L.append(' & '.join(row)+r'\\')
    L+=[r'\bottomrule',r'\end{tabular}',r'\end{table}']
    open('tables/tab_params.tex','w').write('\n'.join(L)); return rows
if __name__=='__main__':
    R1=main_table('none','tab:main_none','RUL prediction with the mount-only inspection. Metrics are averaged over test tools; bold marks the best value per dataset and column (PICP and NMPIW excluded, since neither is a score on its own; LinEx excluded because of its divergent extrapolation). DE+, QR+: deep regressors that also receive the mounting wear and the inspections; CQR+: QR+ with the two-layer conformal step; Wiener-C, PF-C: Wiener and particle-filter models with condition-dependent priors; Wiener-Net, PF-Net: Wiener and general-path particle-filter models that treat the network wear estimates as independent measurements; PF: the particle filter with inspections only (Section~\\ref{sec:baselines}).')
    R2=main_table('insp','tab:main_insp','RUL prediction with periodic inspections (every 200 s for PHM2010, 300 s for NUAA and 600 s for NASA milling); notation as in Table~\\ref{tab:main_none}.')
    A=ablation(); P=params_table()
    pickle.dump((R1,R2,A),open('out/main_tables_v6.pkl','wb'))
