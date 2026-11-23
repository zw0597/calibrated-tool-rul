import sys,pickle,numpy as np
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from style2 import *
from v2 import *
from metrics import tool_metrics
from metrics2 import agg
import matplotlib.gridspec as gs
fig=plt.figure(figsize=(W2,62*MM)); G=gs.GridSpec(1,3,figure=fig,wspace=0.62,width_ratios=[1,1,1.25])
S={ds:pickle.load(open(f'out/sens2_{ds}.pkl','rb')) for ds in ['PHM','NUAA','MILL']}
ORD={'PHM':[None,400,200,100,50],'NUAA':[None,600,300,150,75],'MILL':[None,1200,600,300,150]}
for j,(key,lab) in enumerate([('nMAE',r'$n$MAE (%)'),('NIS','NIS (%)')]):
    ax=fig.add_subplot(G[j])
    for ds,mk in [('PHM','o'),('NUAA','s'),('MILL','^')]:
        x=[]; y=[]; yl=[]; yc=[]
        for d in ORD[ds]:
            L=np.mean([tfail(t) for t in tags(ds)]); n=0 if d is None else L/d
            x.append(n); y.append(S[ds][d]['Ours'][key]); yl.append(S[ds][d]['MCprior'][key]); yc.append(S[ds][d]['OursCP'][key])
        ax.plot(x,y,marker=mk,color=DSC[ds],lw=1.2,ms=3.5,label=f'{DNAME[ds]}: Proposed-raw')
        if key=='NIS': ax.plot(x,yc,marker=mk,color=DSC[ds],lw=0.9,ms=3.2,mfc='white',ls=(0,(3,1.5)),label=f'{DNAME[ds]}: Proposed')
        ax.plot(x,yl,marker=mk,color=DSC[ds],lw=0.8,ms=3,alpha=0.45,ls=(0,(1,1)),label=f'{DNAME[ds]}: Law')
    ax.set_yscale('log'); ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ticks=[5,10,20,50,100] if key=='nMAE' else [50,100,200,500,1000]
    ax.set_yticks(ticks); ax.set_yticklabels([str(t) for t in ticks])
    ax.set_xlabel('Mean number of inspections per tool life'); ax.set_ylabel(lab)
    if j==0:
        from matplotlib.lines import Line2D
        hh=[Line2D([],[],color=DSC[d],lw=1.2,marker=m,ms=3,label=DNAME[d]) for d,m in [('PHM','o'),('NUAA','s'),('MILL','^')]]
        ax.legend(handles=hh,loc='upper right',fontsize=5.5); ax.set_ylim(4,130)
    else:
        from matplotlib.lines import Line2D
        hh=[Line2D([],[],color=MID,lw=1.2,marker='o',ms=3,label='Proposed-raw'),Line2D([],[],color=MID,lw=0.9,ls=(0,(3,1.5)),marker='o',mfc='white',ms=3,label='Proposed (conformal)'),Line2D([],[],color=MID,alpha=0.5,lw=0.8,ls=(0,(1,1)),marker='o',ms=3,label='Law')]
        ax.legend(handles=hh,loc='upper right',fontsize=5.4); ax.set_ylim(35,2500)
    panel(ax,'ab'[j],x=-0.22)
# (c) ablation: change relative to full
_AB=pickle.load(open('out/v4_ablate.pkl','rb')); _agg0=agg
def agg(ds,scen,m):
    if not m.startswith('v4:'): return _agg0(ds,scen,m)
    per=[tool_metrics(t,_AB[m[3:]][ds][scen][t]) for t in tags(ds)]   # KeyError if the variant does not exist
    return {k:np.nanmean([q[k] for q in per]) for k in per[0]}, per
ax=fig.add_subplot(G[2])
V=[('woShift',r'no offset $\epsilon$'),('woBank','no averaging'),('condOnly','Taylor only'),('poolOnly','pooled only'),('singleQ','single QR net'),('woI','no increment'),('woR','no direct RUL'),('woAR','no persistence'),('woDisc','no discrepancy'),('v4:woPred','no pred. scale'),('v4:plainP','plain p-BMA'),('v4:uniP','equal weights')]
y=np.arange(len(V)); OVF={}
for k,(ds,met,mk,fill) in enumerate([('PHM','nMAE','o',True),('NUAA','nMAE','s',True),('MILL','nMAE','^',True),('PHM','NIS','o',False),('NUAA','NIS','s',False),('MILL','NIS','^',False)]):
    full=agg(ds,'insp','Ours')[0][met]
    vals=[]
    for m,_ in V:
        try: vals.append(100*(agg(ds,'insp',m)[0][met]/full-1))
        except KeyError: vals.append(np.nan)
    yy=y+(k-2.5)*0.13
    va=np.array(vals,float); ok=~(va>150)
    ax.scatter(np.where(ok,va,np.nan),yy,marker=mk,s=14,facecolors=DSC[ds] if fill else 'white',edgecolors=DSC[ds],linewidths=0.8,zorder=3,label=f'{DNAME[ds]} {met}')
    for v,yv in zip(vals,yy):
        if np.isfinite(v) and v>150:
            row=int(round(yv)); xs=150+30*OVF.get(row,0); OVF[row]=OVF.get(row,0)+1
            ax.scatter([xs],[row-0.18],marker=mk,s=14,facecolors=DSC[ds] if fill else 'white',edgecolors=DSC[ds],linewidths=0.8,zorder=3)
            ax.text(xs,row+0.22,f'{v:.0f}',fontsize=4.8,va='center',ha='center',color=DSC[ds])
ax.axvline(0,color=INK,lw=0.6); ax.axvline(137,color=MID,lw=0.5,ls=(0,(1,1.5))); ax.text(141,-0.5,'>150%',fontsize=4.8,color=MID,ha='left',va='top'); ax.set_xlim(-42,222); ax.set_xticks([-40,0,50,100])
ax.set_yticks(y); ax.set_yticklabels([l for _,l in V],fontsize=6)
for i in range(len(V)):
    if i%2==0: ax.axhspan(i-0.5,i+0.5,color=FAINT,alpha=0.6,lw=0,zorder=0)
ax.set_xlabel('Change vs. full model (%)')
ax.legend(loc='lower left',bbox_to_anchor=(-0.02,1.0),fontsize=5.2,ncol=2,columnspacing=0.5,handletextpad=0.05,borderaxespad=0.1); panel(ax,'c',x=-0.5)
ax.text(-40,len(V)-0.3,'← better',fontsize=5.6,color=MID,ha='left',va='top'); ax.text(220,len(V)-0.3,'worse →',fontsize=5.6,color=MID,ha='right',va='top'); ax.set_ylim(len(V)+0.1,-0.6)
save(fig,'fig8_sens')
