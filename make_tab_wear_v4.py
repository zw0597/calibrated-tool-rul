import pickle,numpy as np
from pipeline import *

PER_MILL={}
def wear_table():
    names=[('full','1D-CNN + ref.\\ stats + condition (used)'),('lstm','BiLSTM + ref.\\ stats + condition'),('tf','Transformer + ref.\\ stats + condition'),('nocnn','ref.\\ stats + condition (no window branch)'),('nostat','1D-CNN + condition (no ref.\\ stats)'),('nocond','1D-CNN + ref.\\ stats (no condition)'),('plain','1D-CNN only')]
    L=[r'\begin{table}[!tbp]',r'\centering',r'\caption{Wear-increment estimation error (MAE, $\mu$m; mean $\pm$ s.d. over LOTO test tools) for alternative evidence networks. The network used in the framework is evaluated per member of its five-member ensemble; the alternatives are averaged over three (PHM2010) or one (NUAA) independently trained models.}',r'\label{tab:wear}',r'\small',r'\begin{tabular}{lccc}',r'\toprule',r'Network & PHM2010 & NUAA & NASA milling$^\ast$\\',r'\midrule']
    out={}
    for key,lab in names:
        row=[lab]
        for ds in ['PHM','NUAA','MILL']:
            if ds=='MILL' and key!='full': row.append('--'); continue
            RO=pickle.load(open(f'out/nets_{ds}_outer.pkl','rb')); RA=pickle.load(open(f'out/nets_{ds}_arch.pkl','rb')) if ds!='MILL' else {}; per=[]
            for t in tags(ds):
                w=D[t]['wte']-D[t]['wte'][0]
                mus=[RO[('inc',t,s)][0] for s in range(5)] if key=='full' else [RA[k][0] for k in RA if k[1]==key and k[2]==t]
                if mus: per.append(np.mean([np.abs(m-w).mean()*1e3 for m in mus]))
                if ds=='MILL': PER_MILL[t]=per[-1]
            out[(ds,key)]=(np.mean(per),np.std(per)); row.append(f'{np.mean(per):.1f} $\\pm$ {np.std(per):.1f}')
        L.append(' & '.join(row)+r'\\')
    pm=PER_MILL; med=np.median(list(pm.values())); wo=[v for k,v in pm.items() if k!='M12']
    L+=[r'\bottomrule',r'\multicolumn{4}{@{}p{0.97\linewidth}@{}}{\scriptsize $^\ast$Only the network used in the framework was trained for NASA milling. The mean is dominated by tool M12, whose increment estimate jumps to about 3 mm from its second run on (MAE %.0f $\mu$m); median %.0f $\mu$m, %.0f $\pm$ %.0f $\mu$m without M12.}\\'%(pm['M12'],med,np.mean(wo),np.std(wo)),r'\end{tabular}',r'\end{table}']
    open('tables/tab_wear.tex','w').write('\n'.join(L)); return out

if __name__=='__main__':
    wear_table()
