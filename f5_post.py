import sys,pickle,numpy as np
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from style2 import *
from v2 import *
from metrics import tool_metrics
import matplotlib.gridspec as gs
ds='NUAA'; t=sys.argv[1] if len(sys.argv)>1 else 'W6'; T=tags(ds); tr=[x for x in T if x!=t]
N=Nets(ds,True); S,_=N.outer_srcs(t,tr)
o=run_v2(t,tr,S,INSP[ds],keep_weights=True)
oN=run_v2(t,tr,{},INSP[ds],keep_weights=True)
oI=run_v2(t,tr,S,None,keep_weights=True)
tt=D[t]['tte']; m=tt<tfail(t); ins=make_insp(t,INSP[ds],SIG_I)
fig=plt.figure(figsize=(W2,125*MM)); G=gs.GridSpec(2,3,figure=fig,wspace=0.32,hspace=0.45,height_ratios=[1,1.45])
def rng(O):
    MT=np.array(O['MT'])[m]; c=np.cumsum(MT,1)/MT.sum(1,keepdims=True); l=O['lth']
    return l[np.searchsorted(c.min(0),0.002)], l[min(np.searchsorted(c.max(0),0.998),len(l)-1)]
lo=min(rng(O)[0] for O in [o,oN,oI]); hi=max(rng(O)[1] for O in [o,oN,oI])
def heat(ax,O,title,lab,showins=True,key='MT',yv=None):
    MT=np.array(O[key])[m]; MT=MT/MT.max(1,keepdims=True)
    y=O['lth'] if key=='MT' else O['eg']*1e3
    ax.pcolormesh(tt[m],y,MT.T,cmap='Blues',shading='nearest',rasterized=True,vmin=0,vmax=1)
    ax.set_xlim(0,tfail(t))
    if showins:
        for k in ins:
            if 0<k and tt[k]<tfail(t): ax.axvline(tt[k],color=C['DE'],lw=0.6,ls=(0,(2,1.5)))
    ax.set_title(title,loc='left',fontsize=6.2,color=INK); ax.set_xlabel('Cutting time $t$ (s)')
    panel(ax,lab,x=-0.24,y=1.03)
    mt=tool_metrics(t,dict(R=O['R'])); 
    if key=='MT': ax.text(0.97,0.95,f'$n$MAE {mt["nMAE"]:.1f}%',transform=ax.transAxes,ha='right',va='top',fontsize=5.8,bbox=dict(fc='white',ec='none',alpha=0.85,pad=1))
for i,(O,ti,si) in enumerate([(oN,'law + inspections',True),(oI,'law + evidence (mount record only)',False),(o,'full: law + evidence + inspections',True)]):
    ax=fig.add_subplot(G[0,i]); heat(ax,O,ti,'abc'[i],si); ax.set_ylim(lo,hi)
    if i==0: ax.set_ylabel(r'Time scale $\ln\theta$')
ax=fig.add_subplot(G[1,0])
cols=[(r'$c_I$=½',lambda h:h['c']['inc']==0.5),(r'$c_I$=1',lambda h:h['c']['inc']==1.0),(r'$c_I$=2',lambda h:h['c']['inc']==2.0),
      (r'$c_R$=2',lambda h:h['c']['rul']==2.0),(r'$c_\theta$=2',lambda h:h['cp']==2.0),('Taylor',lambda h:h['pi']==0 and len(set(x['pi'] for x in Hb))>1)]
rows=[];names=[]
for dd in ['PHM','NUAA','MILL']:
    OO=pickle.load(open(f'out/rul2_{dd}.pkl','rb'))
    for tl,rr in OO.items():
        Hb=rr['insp']['Ours']['bank']; Wl=rr['insp']['Ours']['W'][np.where(emask(tl))[0][-1]]
        rows.append([Wl[[i for i,h in enumerate(Hb) if f(h)]].sum() if (j<5 or dd!='PHM') else np.nan for j,(_,f) in enumerate(cols)]); names.append(tl)
A=np.array(rows,float)
im=ax.imshow(np.ma.masked_invalid(A),cmap='Blues',vmin=0,vmax=1,aspect='auto')
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        if np.isfinite(A[i,j]): ax.text(j,i,('1' if round(A[i,j],2)>=1 else ('0' if round(A[i,j],2)==0 else f'{A[i,j]:.2f}'.lstrip('0'))),ha='center',va='center',fontsize=4.3,color='white' if A[i,j]>0.6 else INK)
        else: ax.text(j,i,'n/a',ha='center',va='center',fontsize=4.5,color=MID)
ax.set_xticks(range(len(cols))); ax.set_xticklabels([c for c,_ in cols],fontsize=5.6,rotation=0)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names,fontsize=5.0); ax.tick_params(length=0)
for sp in ax.spines.values(): sp.set_visible(False)
ax.axhline(2.5,color='white',lw=1.5); ax.axhline(11.5,color='white',lw=1.5)
ax.set_title('end-of-life hypothesis weights, all tools',loc='left',fontsize=6.2); panel(ax,'d',x=-0.24,y=1.03)
# (e) hypothesis weights
ax=fig.add_subplot(G[1,1]); W=np.array(o['W'])[m]; H=o['bank']
for lab,f,col,ls in [(r'$c_I=\frac{1}{2}$',lambda h:h['c']['inc']==0.5,BLUES[4],'-'),(r'$c_I=1$',lambda h:h['c']['inc']==1.0,BLUES[2],'-'),(r'$c_I=2$',lambda h:h['c']['inc']==2.0,BLUES[1],'-'),
                     (r'$c_R=2$',lambda h:h['c']['rul']==2.0,C['DE'],(0,(3,1.5))),('Taylor prior',lambda h:h['pi']==0,INK,(0,(1,1))),(r'$c_\theta=2$',lambda h:h['cp']==2.0,MID,(0,(4,1,1,1)))]:
    w=W[:,[i for i,h in enumerate(H) if f(h)]].sum(1); ax.plot(tt[m],w,color=col,lw=1.1,ls=ls,label=lab)
ax.set_ylim(-0.02,1.02); ax.set_xlim(0,tfail(t)); ax.set_xlabel('Cutting time $t$ (s)'); ax.set_ylabel('Posterior probability')
ax.set_title('marginal hypothesis weights $w_k$ (full)',loc='left',fontsize=6.2)
ax.legend(loc='lower right',bbox_to_anchor=(1.0,0.07),fontsize=5.6,ncol=2,columnspacing=0.8); panel(ax,'e',x=-0.24,y=1.03)
# (f) medians
ax=fig.add_subplot(G[1,2])
ax.plot(tt[m],rul_true(t)[m],color=INK,lw=1.0,ls=(0,(1,0.8)),label='true RUL')
for O,lab,col,ls in [(oN,'law + insp.',MID,(0,(3,1.5))),(oI,'law + evidence',BLUES[2],'-'),(o,'full',BLUES[4],'-')]:
    ax.plot(tt[m],O['R'][m],color=col,lw=1.1,ls=ls,label=lab)
for k in ins:
    if 0<k and tt[k]<tfail(t): ax.plot(tt[k],0,marker='^',ms=3.6,color=MID,clip_on=False,mec='white',mew=0.3)
ax.set_ylim(0,None); ax.set_xlim(0,tfail(t)); ax.set_xlabel('Cutting time $t$ (s)'); ax.set_ylabel('RUL (s)')
ax.set_title('posterior medians',loc='left',fontsize=6.2); ax.legend(loc='upper right',fontsize=5.6); panel(ax,'f',x=-0.24,y=1.03)
save(fig,'fig5_posterior')
print(t,[tool_metrics(t,dict(R=O['R']))['nMAE'] for O in [oN,oI,o]], H[np.argmax(W[-1])])
