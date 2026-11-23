import sys,pickle,numpy as np
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from style2 import *
from v2 import *
from metrics import tool_metrics
import matplotlib.gridspec as gs
from matplotlib.colors import LogNorm
DS=['PHM','NUAA','MILL']
O={ds:pickle.load(open(f'out/rul2_{ds}.pkl','rb')) for ds in DS}
HY=pickle.load(open('out/v4_hybrid.pkl','rb'))
DA=pickle.load(open('out/v5_deepaux.pkl','rb')); GP=pickle.load(open('out/v5_genpath.pkl','rb'))
def get(ds,t,m):
    if m=='WienerNetK': return HY[ds][t]['insp'][m]
    if m=='DE+': return DA[ds][t]['insp'][m]
    if m=='PF': return GP[ds][t]['insp_insponly']
    r=O[ds][t]; return r['insp'][m] if m in r['insp'] else r[m]
tools=[(ds,t) for ds in DS for t in O[ds]]
MP=[('Pop','Pop'),('DE','DE'),('DE+','DE+'),('MCD','MCD'),('CQRraw','QR'),('Wiener','Wiener'),('WienerNetK','Wiener-\nNet'),('PF','PF'),('MCprior','Law'),('Ours','Prop.')]
fig=plt.figure(figsize=(W2,112*MM)); G=gs.GridSpec(3,2,figure=fig,width_ratios=[1.0,1.05],wspace=0.32,hspace=0.75)
ax=fig.add_subplot(G[:,0])
A=np.array([[tool_metrics(t,get(ds,t,m))['nMAE'] for m,_ in MP] for ds,t in tools])
norm=LogNorm(2,200); ax.imshow(A,cmap='Blues',norm=norm,aspect='auto')
best=A.min(1)
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        v=A[i,j]; ax.text(j,i,f'{v:.1f}' if v<100 else f'{v:.0f}',ha='center',va='center',fontsize=4.9,color='white' if norm(v)>0.62 else INK,fontweight='bold' if v==best[i] else 'normal')
ax.set_xticks(range(len(MP))); ax.set_xticklabels([n for _,n in MP],fontsize=5.6)
ax.xaxis.tick_top(); ax.xaxis.set_label_position('top')
ax.set_yticks(range(len(tools))); ax.set_yticklabels([t for _,t in tools],fontsize=5.6); ax.tick_params(length=0)
[sp.set_visible(False) for sp in ax.spines.values()]
for b in [2.5,11.5]: ax.axhline(b,color='white',lw=2.2)
for y0,y1,ds in [(0,2,'PHM'),(3,11,'NUAA'),(12,26,'MILL')]:
    ax.text(-1.75,(y0+y1)/2,DNAME[ds],rotation=90,ha='center',va='center',fontsize=6,color=DSC[ds],transform=ax.transData)
wins=[(np.argmin(A[i])==len(MP)-1) for i in range(len(tools))]
ax.set_xlabel(r'$n$MAE (%) per test tool, periodic inspections (bold: best per tool)',fontsize=6.2,labelpad=4)
panel(ax,'a',x=-0.36,y=1.04)
bins=np.linspace(0,1,11); cen=(bins[1:]+bins[:-1])/2
for j,ds in enumerate(DS):
    ax=fig.add_subplot(G[j,1])
    for m,nm,col,ls in [('DE','DE',C['DE'],'-'),('CQRraw','QR',C['QR'],'-'),('Wiener','Wiener',C['Wiener'],(0,(3,1.5))),('MCprior','Law',C['Law'],(0,(1,1))),('Ours','Proposed',C['Proposed'],'-'),('DE+','DE+',C['DE+'],(0,(4,1,1,1)))]:
        nb=10 if ds!='MILL' else 5
        bb=np.linspace(0,1,nb+1); cc=(bb[1:]+bb[:-1])/2
        E=np.full((len(O[ds]),nb),np.nan)
        for i,t in enumerate(O[ds]):
            mk=emask(t); fr=D[t]['tte'][mk]/tfail(t); e=np.abs(get(ds,t,m)['R'][mk]-rul_true(t)[mk])/tfail(t)*100
            for b in range(nb):
                s=(fr>=bb[b])&(fr<bb[b+1])
                if s.any(): E[i,b]=e[s].mean()
        mu=np.nanmean(E,0); ax.plot(cc,mu,color=col,ls=ls,lw=1.4 if nm=='Proposed' else 0.9,marker='o' if nm=='Proposed' else None,ms=2.4,label=nm)
        if nm=='Proposed': ax.fill_between(cc,np.nanmin(E,0),np.nanmax(E,0),color=C['Proposed'],alpha=0.12,lw=0)
    ax.set_xlim(0,1); ax.set_ylim(0,{'PHM':40,'NUAA':90,'MILL':80}[ds])
    ax.set_ylabel('Abs. error (% of life)')
    if j==2: ax.set_xlabel(r'Life fraction $t/T$')
    ax.set_title(f'{DNAME[ds]}: mean error along the life',loc='left',fontsize=6.2)
    if j==0: ax.legend(loc='lower left',bbox_to_anchor=(0,1.17),ncol=5,fontsize=5.6,columnspacing=0.8,handlelength=1.4)
    panel(ax,'bcd'[j],x=-0.17)
save(fig,'fig6_tools')
print('Proposed best on',sum(wins),'of',len(tools))
