import sys,pickle,numpy as np
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from style2 import *
from v2 import *
from metrics import tool_metrics
from make_tables_v6 import agg
import matplotlib.gridspec as gs
DS=['PHM','NUAA','MILL']
O={ds:pickle.load(open(f'out/rul2_{ds}.pkl','rb')) for ds in DS}
DA=pickle.load(open('out/v5_deepaux.pkl','rb'))
CQP=pickle.load(open('out/v6_cqrplus.pkl','rb'))
def get(ds,t,m):
    if m=='DE+': return DA[ds][t]['insp'][m]
    if m=='CQR+': return CQP[ds][t]['insp']['CQR+']
    r=O[ds][t]; return r['insp'][m] if m in r['insp'] else r[m]
MS=[('DE','DE','o',True),('DE+','DE+','o',False),('MCD','MCD','v',True),('CQRraw','QR','D',False),('CQR','CQR','D',True),('CQR+','CQR+','h',True),('Wiener','Wiener','s',True),('MCprior','Law','P',True),('Ours','Proposed-raw','*',False),('OursCP','Proposed','*',True)]
fig=plt.figure(figsize=(W2,118*MM)); G=gs.GridSpec(2,3,figure=fig,wspace=0.32,hspace=0.45,top=0.93)
for j,ds in enumerate(DS):
    ax=fig.add_subplot(G[0,j]); ax.axhline(90,color=MID,lw=0.6,ls=(0,(3,2))); ax.text(0.02 if ds=='NUAA' else 0.98,88.5,'nominal 90%',transform=ax.get_yaxis_transform(),fontsize=5.6,color=MID,va='top',ha='left' if ds=='NUAA' else 'right')
    for m,nm,mk,fill in MS:
        a=agg(ds,'insp',m)[0]; col=C[nm]
        x=min(a['NMPIW'],560) if ds=='MILL' else a['NMPIW']
        ax.scatter(x,a['PICP'],s=46 if mk=='*' else 20,marker=mk,facecolors=col if fill else 'white',edgecolors=col,linewidths=0.8,zorder=3,label=nm,clip_on=False)
        if ds=='MILL' and a['NMPIW']>560 and nm=='Wiener': ax.annotate(f'{a["NMPIW"]:.0f} \u2192',(560,a['PICP']),xytext=(-6,-9),textcoords='offset points',fontsize=5.2,color=col,ha='right')
    ax.set_xscale('log'); ax.set_ylim(0,105); ax.set_xlabel('Mean interval width NMPIW (% of life)'); 
    if j==0: ax.set_ylabel('Coverage PICP (%)')
    ax.set_title(DNAME[ds],loc='left',fontsize=6.2); panel(ax,'abc'[j],x=-0.2)
    if j==0: fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(0.5,0.955),ncol=10,fontsize=5.5,handletextpad=0.2,columnspacing=1.0)
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    if ds=='PHM': ax.set_xlim(5,120); ax.set_xticks([5,10,20,50,100]); ax.set_xticklabels(['5','10','20','50','100'])
    elif ds=='NUAA': ax.set_xlim(70,700); ax.set_xticks([100,200,400]); ax.set_xticklabels(['100','200','400'])
    else: ax.set_xlim(40,600); ax.set_xticks([50,100,200,500]); ax.set_xticklabels(['50','100','200','500'])
# (c) per-tool coverage
ax=fig.add_subplot(G[1,:]); tools=[(ds,t) for ds in DS for t in O[ds]]
for k,(m,nm,mk,fill) in enumerate([('DE','DE','o',True),('DE+','DE+','o',False),('CQR+','CQR+','h',True),('Wiener','Wiener','s',True),('OursCP','Proposed','*',True)]):
    v=[tool_metrics(t,get(ds,t,m))['PICP'] for ds,t in tools]
    x=np.arange(len(tools))+(k-2)*0.15
    ax.scatter(x,v,s=26 if mk=='*' else 11,marker=mk,facecolors=C[nm] if fill else 'white',edgecolors=C[nm] if not fill else 'white',linewidths=0.6 if not fill else 0.3,label=nm,zorder=3)
ax.axhline(90,color=MID,lw=0.6,ls=(0,(3,2))); ax.axvline(2.5,color=LIGHT,lw=0.6); ax.axvline(11.5,color=LIGHT,lw=0.6)
for x0,ds in [(1,'PHM'),(7,'NUAA'),(19,'MILL')]: ax.text(x0,-0.19,DNAME[ds],ha='center',fontsize=6,color=DSC[ds],clip_on=False,transform=ax.get_xaxis_transform())
ax.set_xticks(range(len(tools))); ax.set_xticklabels([t for _,t in tools],fontsize=5.6); ax.set_ylim(-3,103)
ax.set_ylabel('Per-tool coverage (%)')
ax.legend(loc='lower left',ncol=5,fontsize=5.6,columnspacing=0.8,handletextpad=0.1,bbox_to_anchor=(0.0,1.0),borderaxespad=0.1); panel(ax,'d',x=-0.06); ax.set_xlim(-0.6,len(tools)-0.4)
save(fig,'fig7_calib')
