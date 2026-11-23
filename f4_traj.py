import sys,pickle,numpy as np
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from style2 import *
from v2 import *
import matplotlib.gridspec as gs
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
O={ds:pickle.load(open(f'out/rul2_{ds}.pkl','rb')) for ds in ['PHM','NUAA','MILL']}
tools=[('PHM','c1'),('PHM','c6'),('NUAA','W2'),('NUAA','W3'),('NUAA','W6'),('NUAA','W8'),('MILL','M03'),('MILL','M11'),('MILL','M04')]
fig=plt.figure(figsize=(W2,135*MM)); G=gs.GridSpec(3,3,figure=fig,wspace=0.36,hspace=0.55,top=0.91)
for i,(ds,t) in enumerate(tools):
    ax=fig.add_subplot(G[i//3,i%3]); r=O[ds][t]; m=emask(t); tt=D[t]['tte'][m]; y=rul_true(t)[m]
    p=r['insp']['Ours']; cp=r['insp']['OursCP']; Rq=p['Rq'][m]
    ax.fill_between(tt,np.maximum(cp['lo'][m],0),cp['hi'][m],color=BLUES[0],lw=0,label='90% conformal')
    ax.fill_between(tt,Rq[:,0],Rq[:,4],color=BLUES[1],lw=0,label='90% posterior')
    ax.fill_between(tt,Rq[:,1],Rq[:,3],color=BLUES[2],lw=0,alpha=0.9,label='50% posterior')
    ax.plot(tt,r['DE']['R'][m],color=C['DE'],lw=0.7,alpha=0.9,label='DE')
    ax.plot(tt,r['insp']['Wiener']['R'][m],color=C['Wiener'],lw=0.8,ls=(0,(3,1.5)),label='Wiener')
    mk_='o' if ds=='MILL' else None
    ax.plot(tt,Rq[:,2],color=BLUES[4],lw=1.2,marker=mk_,ms=2.2,label='Proposed (median)')
    ax.plot(tt,y,color=INK,lw=1.0,ls=(0,(1,0.8)),label='true RUL')
    ins=make_insp(t,INSP[ds],SIG_I)
    for k in ins:
        if 0<k and D[t]['tte'][k]<tfail(t): ax.plot(D[t]['tte'][k],0,marker='^',ms=3.6,color=MID,clip_on=False,zorder=6,mec='white',mew=0.3)
    top=max(y.max()*1.55,np.percentile(Rq[:,2],98)*1.2)
    ax.set_ylim(0,top); ax.set_xlim(0,tfail(t))
    a=tool_metrics_s=None
    from metrics import tool_metrics
    mo=tool_metrics(t,dict(R=p['R'],lo=cp['lo'],hi=cp['hi'])); md=tool_metrics(t,r['DE'])
    ax.set_title(f'{t}: $n$MAE {mo["nMAE"]:.1f}% (DE {md["nMAE"]:.1f}%), PICP {mo["PICP"]:.0f}%',loc='left',fontsize=5.7,color=INK)
    if i%3==0: ax.set_ylabel('RUL (s)')
    if i//3==2: ax.set_xlabel('Cutting time $t$ (s)')
    panel(ax,'abcdefghi'[i],x=-0.2 if i%3==0 else -0.15,y=1.04)
h=[Patch(fc=BLUES[2],label='50% posterior'),Patch(fc=BLUES[1],label='90% posterior'),Patch(fc=BLUES[0],label='90% conformal'),
   Line2D([],[],color=BLUES[4],lw=1.2,label='Proposed (median)'),Line2D([],[],color=INK,lw=1.0,ls=(0,(1,0.8)),label='true RUL'),
   Line2D([],[],color=C['DE'],lw=0.8,label='DE'),Line2D([],[],color=C['Wiener'],lw=0.8,ls=(0,(3,1.5)),label='Wiener'),
   Line2D([],[],color=MID,marker='^',ls='none',ms=3.6,label='inspection')]
fig.legend(handles=h,loc='lower center',bbox_to_anchor=(0.5,0.945),ncol=8,fontsize=6.2,handlelength=1.5,columnspacing=1.1)
save(fig,'fig4_traj')
