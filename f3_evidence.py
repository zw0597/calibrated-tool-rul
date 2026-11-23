import sys,pickle,glob,numpy as np
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from style2 import *
from v2 import *
import matplotlib.gridspec as gs
fig=plt.figure(figsize=(W2,62*MM))
G=gs.GridSpec(1,4,figure=fig,wspace=0.5,width_ratios=[1.35,1.0,1.0,1.0])
# (a) increment MAE per architecture, dot plot with per-tool spread
ax=fig.add_subplot(G[0])
names=[('full','1D-CNN + stat. + cond.'),('lstm','BiLSTM + stat. + cond.'),('tf','Transformer + stat. + cond.'),('nocnn','stat. + cond.'),('nostat','1D-CNN + cond.'),('nocond','1D-CNN + stat.'),('plain','1D-CNN only')]
for j,ds in enumerate(['PHM','NUAA']):
    RO=pickle.load(open(f'out/nets_{ds}_outer.pkl','rb')); RA=pickle.load(open(f'out/nets_{ds}_arch.pkl','rb'))
    for i,(key,lab) in enumerate(names):
        per=[]
        for t in tags(ds):
            w=D[t]['wte']-D[t]['wte'][0]
            mus=[RO[('inc',t,s)][0] for s in range(5)] if key=='full' else [RA[kk][0] for kk in RA if kk[1]==key and kk[2]==t]
            if mus: per.append(np.mean([np.abs(m-w).mean()*1e3 for m in mus]))
        y=i+(j-0.5)*0.32; per=np.array(per)
        ax.plot([per.min(),per.max()],[y,y],color=DSC[ds],lw=0.6,alpha=0.5)
        ax.plot(per,np.full(len(per),y),'|',color=DSC[ds],ms=3,mew=0.6,alpha=0.7)
        ax.plot(per.mean(),y,'o',ms=3.6,color=DSC[ds],mec='white',mew=0.4,label=DNAME[ds] if i==0 else None,zorder=3)
ax.set_yticks(range(len(names))); ax.set_yticklabels([l for _,l in names],fontsize=6); ax.invert_yaxis()
ax.set_xscale('log'); ax.set_xticks([10,20,50,100]); ax.set_xticklabels(['10','20','50','100'])
ax.set_xlim(7,150); ax.set_xlabel(r'Increment MAE ($\mu$m), per tool'); ax.legend(loc='lower right',bbox_to_anchor=(1.0,1.0),ncol=2,handletextpad=0.1)
ax.axhspan(-0.5,0.5,color=FAINT,zorder=0,lw=0)
panel(ax,'a',x=-0.78)
# (b),(c) ACF
N=Nets('PHM',True); NN=Nets('NUAA',True); NS={'PHM':N,'NUAA':NN,'MILL':Nets('MILL',True)}
def zseq(ds,src):
    n=NS[ds]; out=[]
    for k in tags(ds):
        for j in tags(ds):
            if j==k: continue
            I,R=n.inner(k,j)
            if src=='inc': z=(I['z']-(D[j]['wte']-D[j]['wte'][0]))/I['s']
            else: m=emask(j); z=((R['z']-rul_true(j))/R['s'])[m]
            out.append((k,j,z))
    return out
Z={(ds,s):zseq(ds,s) for ds in ['PHM','NUAA'] for s in ['inc','rul']}
lags=np.arange(31)
for pi,(src,lab) in enumerate([('inc','wear-increment evidence'),('rul','direct-RUL evidence')]):
    ax=fig.add_subplot(G[1+pi])
    for ds in ['PHM','NUAA']:
        A=[]
        for k,j,z in Z[(ds,src)]:
            if len(z)<35: continue
            v=np.mean(z*z); A.append([np.mean(z[l:]*z[:len(z)-l])/v if l>0 else 1 for l in lags])
        A=np.array(A)
        ax.fill_between(lags,np.percentile(A,25,0),np.percentile(A,75,0),color=DSC[ds],alpha=0.15,lw=0)
        ax.plot(lags,np.median(A,0),color=DSC[ds],lw=1.1,label=f'{DNAME[ds]} (median, IQR)')
        # fitted AR(1)+white model from Proposition 2
        P=[];
        for k in tags(ds):
            tr=[t for t in tags(ds) if t!=k]; pI,pR=NS[ds].params(k,tr); P.append(pI if src=='inc' else pR)
        psi=np.mean([p['psi'] for p in P]); phi=np.mean([p['phi'] for p in P])
        ax.plot(lags,np.where(lags==0,1,psi*phi**lags),color=DSC[ds],lw=0.7,ls=(0,(2,1.2)))
    ax.plot(lags,(lags==0).astype(float),color=INK,lw=0.7,ls=(0,(1,1)),label='i.i.d. assumption')
    ax.plot([],[],color=MID,lw=0.7,ls=(0,(2,1.2)),label=r'fitted $\psi_s\phi_s^{\,l}$')
    ax.set_xlim(0,30); ax.set_ylim(-0.62,1.05); ax.set_yticks([-0.4,0,0.4,0.8]); ax.set_xlabel('Lag $l$ (windows)'); ax.axhline(0,color=FAINT,lw=0.6,zorder=0)
    if pi==0: ax.set_ylabel(r'Autocorrelation of $e_k/\hat s_k$')
    ax.set_title(lab,fontsize=6.5,color=MID,loc='left')
    if pi==0: ax.legend(loc='lower left',bbox_to_anchor=(0.04,0.0),fontsize=5.6)
    panel(ax,'bc'[pi],x=-0.25)
# (d) kappa vs psi
ax=fig.add_subplot(G[3])
for ds,mk in [('PHM','o'),('NUAA','s'),('MILL','^')]:
    for src,fill in [('inc',True),('rul',False)]:
        P=[]
        for k in tags(ds):
            tr=[t for t in tags(ds) if t!=k]; pI,pR=NS[ds].params(k,tr); P.append(pI if src=='inc' else pR)
        x=[p['kappa'] for p in P]; y=[p['psi'] for p in P]
        ax.scatter(x,y,s=14,marker=mk,facecolors=DSC[ds] if fill else 'white',edgecolors=DSC[ds],linewidths=0.7,label=DNAME[ds] if src=='inc' else None,zorder=3)
ax.axvline(1,color=MID,lw=0.6,ls=(0,(3,2)))
ax.set_xscale('log'); ax.set_xticks([1,2,4,8]); ax.set_xticklabels(['1','2','4','8']); ax.set_xlim(0.8,10); ax.set_ylim(0.3,1.0); ax.set_title('filled: increment, open: RUL',loc='left',fontsize=6.0,color=MID)
ax.set_xlabel(r'Error scale $\kappa_s$'); ax.set_ylabel(r'Persistent fraction $\psi_s$'); ax.legend(loc='lower left',fontsize=5.5,handletextpad=0.1)
panel(ax,'d',x=-0.3)
save(fig,'fig3_evidence')
