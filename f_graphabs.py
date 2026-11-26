import sys,os,pickle,numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from style2 import *
from v2 import *
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.size':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8})
fig=plt.figure(figsize=(13.28,5.31))
ax0=fig.add_axes([0,0,1,1]); ax0.set_xlim(0,13.28); ax0.set_ylim(0,5.31); ax0.axis('off')
def box(x,y,w,h,title,body,fc,ec):
    ax0.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.08',fc=fc,ec=ec,lw=1.0))
    ax0.text(x+0.15,y+h-0.17,title,fontsize=10,fontweight='bold',va='top',color=INK)
    ax0.text(x+0.15,y+h-0.52,body,fontsize=8.6,va='top',color='#333333',linespacing=1.35)
def arrow(x0,y0,x1,y1):
    ax0.annotate('',xy=(x1,y1),xytext=(x0,y0),arrowprops=dict(arrowstyle='-|>',lw=1.1,color=MID,mutation_scale=11))
ax0.text(0.3,5.05,'Tool RUL prediction under persistent network errors',fontsize=13,fontweight='bold',va='top',color=INK)
# inputs
box(0.3,3.2,2.6,1.1,'Offline','run-to-failure records,\nnested leave-one-tool-out splits','#f4f4f4','#9a9a9a')
box(0.3,1.85,2.6,1.1,'Online','signal windows, cutting condition,\nsparse wear inspections','#f4f4f4','#9a9a9a')
# method
box(3.4,3.2,3.7,1.1,'Mixed-effects wear law',r'$h(v)=a_i+t/\theta_i$: closed-form RUL,'+'\nrandom time scale and age offset','#e8f0f7','#1b5e8c')
box(3.4,1.85,3.7,1.1,'Persistent-bias deep evidence','increment and RUL networks with\nAR biases identified by moments','#e8f0f7','#1b5e8c')
box(3.4,0.5,3.7,1.1,'Hypothesis-averaged grid filter','Bayesian model averaging, stacking\n+ tool-level conformal calibration','#e8f0f7','#1b5e8c')
arrow(2.9,3.75,3.4,3.75); arrow(2.9,2.4,3.4,2.4)
arrow(5.25,3.2,5.25,2.95); arrow(5.25,1.85,5.25,1.6)
arrow(7.1,1.05,7.85,1.05)
# right: trajectory
O=pickle.load(open('out/rul2_MILL.pkl','rb')); t='M11'; r=O[t]; m=emask(t); tt=D[t]['tte'][m]; y=rul_true(t)[m]
p=r['insp']['Ours']; cp=r['insp']['OursCP']; Rq=p['Rq'][m]
ax=fig.add_axes([8.45/13.28,0.62/5.31,2.45/13.28,3.55/5.31])
ax.fill_between(tt,np.maximum(cp['lo'][m],0),cp['hi'][m],color=BLUES[1],lw=0,label='90% calibrated interval')
ax.plot(tt,r['DE']['R'][m],color=C['DE'],lw=1.1,marker='s',ms=2.5,label='deep ensemble')
ax.plot(tt,Rq[:,2],color=BLUES[4],lw=1.6,marker='o',ms=3,label='framework (median)')
ax.plot(tt,y,color=INK,lw=1.2,ls=(0,(1,0.8)),label='true RUL')
ax.set_xlim(0,tfail(t)); ax.set_ylim(0,cp['hi'][m].max()*1.04)
ax.set_xlabel('cutting time (s)'); ax.set_ylabel('RUL (s)')
ax.set_title(f'NASA milling tool {t} (unseen, longest life)',fontsize=9,loc='left',color=INK)
ax.legend(loc='upper right',bbox_to_anchor=(1.02,1.0),frameon=False,fontsize=7.6,handlelength=1.5)
# key results
ax0.text(11.05,4.15,'40 tools, 4 datasets\n(1 held out, published protocol)',fontsize=9.5,fontweight='bold',va='top',color=INK)
ax0.text(11.05,3.45,'two-thirds or more of the\nnetwork error variance persists;\nvariance inflation predicts\ncoverage loss out of sample\n\ncalibrated coverage 89–94%\n\nvs 16 alternatives (stratified tests):\nlower late-life error than all,\nlower interval score than 14,\nlowest replacement cost',fontsize=8.6,va='top',color='#333333',linespacing=1.35)
fig.savefig('figures/graphical_abstract.png',dpi=200); fig.savefig('figures/graphical_abstract.pdf')
