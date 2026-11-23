import numpy as np, sys
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from style2 import *
from lib import D, TH, tags, tfail
from mc import register_shift, theta_prior
from v2 import law
import matplotlib.gridspec as gs
from matplotlib.lines import Line2D
fig=plt.figure(figsize=(W2,148*MM))
G=gs.GridSpec(3,3,figure=fig,wspace=0.34,hspace=0.6)
FZ={0.045:'#9ecae1',0.05:'#4292c6',0.055:'#08306b'}; NLS={1750:'-',1800:(0,(4,1.5)),1850:(0,(1,1))}
PC={'c1':'#b2442c','c4':'#e08b5b','c6':'#6b1f12'}
MC={(1.0,):'#8c6d1f',(2.0,):'#d9b65a'}   # material: cast iron, steel
MLS={0.25:'-',0.5:(0,(4,1.5))}
def mcol(t): return '#6b4f10' if D[t]['c'][2]==1 else '#d9a441'
def mls(t): return '-' if D[t]['c'][1]==0.75 else (0,(4,1.5))
# (a) PHM
ax=fig.add_subplot(G[0,0])
for t in tags('PHM'):
    d=D[t]; ax.plot(d['tte'],d['wte']*1e3,color=PC[t],lw=1.0)
    ax.plot(tfail(t),150,'o',ms=3.2,mfc='white',mec=PC[t],mew=0.8,zorder=4)
    ax.text(d['tte'][-1]+20,d['wte'][-1]*1e3,t,fontsize=6,va='center',color=PC[t])
ax.axhline(150,color=MID,lw=0.5,ls=(0,(3,2)),zorder=0); ax.text(25,154,r'$V^{*}$ = 150 $\mu$m',fontsize=5.8,color=MID,va='bottom')
ax.set_xlim(0,1520); ax.set_ylim(20,250); ax.set_xlabel('Cutting time $t$ (s)'); ax.set_ylabel(r'Flank wear $v$ ($\mu$m)')
ax.set_title('PHM2010: HRC52 steel, one condition',loc='left',fontsize=6.3,color=MID); panel(ax,'a',x=-0.2)
# (b) NUAA
ax=fig.add_subplot(G[0,1])
for t in tags('NUAA'):
    d=D[t]; c=d['c']; col=FZ[round(float(c[0]),3)]; m=d['tte']<=1800
    ax.plot(d['tte'][m],d['wte'][m]*1e3,color=col,ls=NLS[int(c[1])],lw=1.0)
    ax.plot(tfail(t),160,'o',ms=2.8,mfc='white',mec=col,mew=0.7,zorder=4)
ax.axhline(160,color=MID,lw=0.5,ls=(0,(3,2)),zorder=0); ax.text(2440,164,r'$V^{*}$ = 160 $\mu$m',fontsize=5.8,color=MID,va='bottom',ha='right')
h1=[Line2D([],[],color=v,lw=1.3,label=f'{k:g}') for k,v in FZ.items()]; h2=[Line2D([],[],color=MID,ls=v,lw=1.0,label=f'{k}') for k,v in NLS.items()]
l1=ax.legend(handles=h1,loc='upper right',title=r'$f_z$ (mm)',title_fontsize=5.6,fontsize=5.5,alignment='left',handlelength=1.4); ax.add_artist(l1)
ax.legend(handles=h2,loc='lower right',title=r'$n$ (r/min)',title_fontsize=5.6,fontsize=5.5,alignment='left',handlelength=1.4)
ax.set_xlim(0,2450); ax.set_xticks([0,500,1000,1500]); ax.set_ylim(40,330); ax.set_xlabel('Cutting time $t$ (s)')
ax.set_title(r'NUAA: TC4, 3$\times$3 design (first 1800 s)',loc='left',fontsize=6.3,color=MID); panel(ax,'b',x=-0.17)
# (c) MILL
ax=fig.add_subplot(G[0,2])
for t in tags('MILL'):
    d=D[t]; ax.plot(d['tte'],d['wte']*1e3,color=mcol(t),ls=mls(t),lw=0.9,marker='.',ms=2.2)
    ax.plot(tfail(t),450,'o',ms=2.8,mfc='white',mec=mcol(t),mew=0.7,zorder=4)
ax.axhline(450,color=MID,lw=0.5,ls=(0,(3,2)),zorder=0)
hh=[Line2D([],[],color='#6b4f10',lw=1.2,label='cast iron'),Line2D([],[],color='#d9a441',lw=1.2,label='steel'),Line2D([],[],color=MID,lw=1,ls='-',label=r'$a_p$ = 0.75 mm'),Line2D([],[],color=MID,lw=1,ls=(0,(4,1.5)),label=r'$a_p$ = 1.5 mm'),Line2D([],[],color=MID,lw=0.5,ls=(0,(3,2)),label=r'$V^{*}$ = 450 $\mu$m')]
ax.legend(handles=hh,loc='upper right',fontsize=5.5,handlelength=1.6)
ax.set_xlim(0,6400); ax.set_ylim(0,1600); ax.set_xlabel('Cutting time $t$ (s)')
ax.set_title('NASA milling: 2 materials, 2 feeds, 2 depths',loc='left',fontsize=6.3,color=MID); panel(ax,'c',x=-0.2)
# (d-f) collapse
EPS={}
for j,ds in enumerate(['PHM','NUAA','MILL']):
    ax=fig.add_subplot(G[1,j]); T=tags(ds); M,th,a,e=register_shift(T,nseg=3)
    for t in T:
        d=D[t]; om=a[t]+d['tte']/th[t]
        col=PC[t] if ds=='PHM' else (FZ[round(float(d['c'][0]),3)] if ds=='NUAA' else mcol(t))
        ax.plot(om/1000,d['wte']*1e3,color=col,lw=0.7,alpha=0.85)
        EPS[t]=(float(M.hinv(np.array([a[t]]))[0])-d['wte'][0], np.log(th[t]), ds)
    vmax={'PHM':0.25,'NUAA':0.32,'MILL':0.9}[ds]
    vv=np.linspace(min(D[t]['wte'].min() for t in T),vmax,300)
    ax.plot(M.h(vv)/1000,vv*1e3,color=INK,lw=1.5,zorder=5,label=r'learned $h^{-1}(\omega)$')
    for kn in M.k[1:-1]: ax.axhline(kn*1e3,color=FAINT,lw=0.6,zorder=0)
    ax.axhline(TH[ds]*1e3,color=MID,lw=0.5,ls=(0,(3,2)),zorder=0)
    ax.set_xlabel(r'Normalised age $\omega=a_i+t/\theta_i$ (ks)')
    if j==0: ax.set_ylabel(r'$v$ ($\mu$m)')
    ax.legend(loc='upper left'); ax.set_ylim({'PHM':(20,250),'NUAA':(40,330),'MILL':(0,950)}[ds])
    res=np.concatenate([D[t]['wte']-M.hinv(a[t]+D[t]['tte']/th[t]) for t in T])*1e3
    ax.text(0.97,0.05,f'{DNAME[ds]}\n$J$ = 3, rms {np.sqrt(np.mean(res**2)):.1f} $\\mu$m',transform=ax.transAxes,ha='right',va='bottom',fontsize=5.6,color=MID)
    panel(ax,'def'[j],x=-0.2 if j!=1 else -0.17)
# (g) random effects
ax=fig.add_subplot(G[2,0])
for ib,(ds,mk) in enumerate([('PHM','o'),('NUAA','s'),('MILL','^')]):
    T=tags(ds); x=np.array([EPS[t][0]*1e3 for t in T]); y=np.array([EPS[t][1] for t in T])
    ax.scatter(x,y,s=15,marker=mk,c=DSC[ds],edgecolors='white',linewidths=0.4,zorder=3,label=DNAME[ds])
    ax.errorbar(x.mean(),-1.62-0.17*ib,xerr=x.std(ddof=1),fmt='|',color=DSC[ds],ms=5,elinewidth=0.9,capsize=1.5)
ax.axvline(0,color=FAINT,lw=0.6,zorder=0)
ax.set_xlabel(r'Effective-age offset $\epsilon_i$ ($\mu$m)'); ax.set_ylabel(r'Time scale $\ln\theta_i$')
ax.legend(loc='lower left',bbox_to_anchor=(0,1.0),ncol=3,handletextpad=0.1,columnspacing=0.6,fontsize=5.4,borderaxespad=0.1); panel(ax,'g',x=-0.2,y=1.1)
ax.set_ylim(-2.1,1.25)
# (h),(i) Taylor LOTO
for j,ds in enumerate(['NUAA','MILL']):
    ax=fig.add_subplot(G[2,1+j]); T=tags(ds); obs=[];pred=[];sd=[];pool=[]
    Lall=law(T,ds,True,nseg=3)
    for k in T:
        tr=[t for t in T if t!=k]; L=law(tr,ds,True,nseg=3); f,s=theta_prior(tr,L['th'],True); g,s2=theta_prior(tr,L['th'],False)
        obs.append(np.log(Lall['th'][k])); pred.append(f(D[k]['c'])); sd.append(s); pool.append(g(D[k]['c']))
    obs=np.array(obs); pred=np.array(pred); pool=np.array(pool)
    lim=[min(obs.min(),pred.min())-0.3,max(obs.max(),pred.max())+0.3]
    ax.plot(lim,lim,color=MID,lw=0.6,ls=(0,(3,2)))
    ax.errorbar(pred,obs,xerr=sd,fmt='none',ecolor=LIGHT,elinewidth=0.6,capsize=0,zorder=1)
    ax.scatter(pred,obs,s=14,marker='s' if ds=='NUAA' else '^',c=DSC[ds],edgecolors='white',linewidths=0.4,zorder=3,label='extended Taylor')
    ax.scatter(pool,obs,s=11,marker='x',c=MID,linewidths=0.7,zorder=2,label='pooled prior')
    if ds=='NUAA':
        for x,y,l in zip(pred,obs,T): ax.text(x+0.07,y,l,fontsize=5.2,va='center')
    r=np.corrcoef(pred,obs)[0,1]; e1=np.sqrt(np.mean((pred-obs)**2)); e2=np.sqrt(np.mean((pool-obs)**2))
    ax.set_title(f'{DNAME[ds]}: LOTO $r$ = {r:.2f}\nRMSE {e1:.2f} (pooled prior: {e2:.2f})',fontsize=6,color=MID,loc='left')
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel(r'Prior mean of $\ln\theta$ (LOTO)')
    if j==0: ax.set_ylabel(r'Registered $\ln\theta_i$')
    ax.legend(loc='lower right',bbox_to_anchor=(1.0,0.1 if ds=='NUAA' else 0.0),handletextpad=0.2,fontsize=5.5); panel(ax,'hi'[j],x=-0.17)
    print(ds,'r',r,e1,e2)
save(fig,'fig2_data')
