import sys,pickle,numpy as np
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from style2 import *
from v2 import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'xtick.labelsize':4.8,'ytick.labelsize':4.8,'axes.labelsize':5.2,'axes.linewidth':0.45,
                     'xtick.major.size':1.5,'ytick.major.size':1.5,'xtick.major.width':0.4,'ytick.major.width':0.4})
fig=plt.figure(figsize=(W2,84*MM))
def box(x,y,w,h,title,fc,ec,tag):
    fig.patches.append(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.004,rounding_size=0.012',transform=fig.transFigure,fc=fc,ec=ec,lw=0.7,zorder=-5))
    fig.text(x+0.008,y+h-0.012,tag,fontsize=7,fontweight='bold',color=ec,va='top')
    fig.text(x+0.032,y+h-0.012,title,fontsize=6.4,fontweight='bold',color=INK,va='top')
def arrow(p,q,col=INK,ls='-',cs='arc3'):
    fig.patches.append(FancyArrowPatch(p,q,transform=fig.transFigure,arrowstyle='-|>',mutation_scale=7,lw=0.8,color=col,ls=ls,connectionstyle=cs,zorder=5))
def mini(x,y,w,h):
    ax=fig.add_axes([x,y,w,h],zorder=3); ax.set_facecolor('none'); return ax
# region labels
fig.text(0.005,0.985,'Offline: training tools, nested leave-one-tool-out',fontsize=6.2,color=MID,va='top',style='italic')
fig.text(0.415,0.985,'Online: new tool $i^\\star$, window $k=1,2,\\dots$',fontsize=6.2,color=MID,va='top',style='italic')
cA='#b2442c'; cB='#d08a1f'; cC='#3c8c6e'; cD='#1b5e8c'; cE='#7b3f80'
# ---------- A: law
box(0.005,0.655,0.385,0.295,'Mixed-effects wear law',  '#fbf1ee',cA,'A')
ax=mini(0.058,0.725,0.13,0.155)
T=tags('NUAA'); Lw=law(T,'NUAA',True,nseg=3); M=Lw['M']
for t in T:
    d=D[t]; ax.plot((Lw['a'][t]+d['tte']/Lw['th'][t])/1000,d['wte']*1e3,lw=0.5,color='#e3a99a')
vv=np.linspace(0.05,0.32,200); ax.plot(M.h(vv)/1000,vv*1e3,color=cA,lw=1.2)
ax.set_xlabel(r'$\omega=a_i+t/\theta_i$ (ks)',labelpad=1); ax.set_ylabel(r'$v$ ($\mu$m)',labelpad=1)
fig.text(0.205,0.885,r'$h(v_i(t))=a_i+t/\theta_i+\eta_i(t)$',fontsize=6.0,va='top')
fig.text(0.205,0.845,r'$a_i=h(v_i(0)+\epsilon_i)$',fontsize=6.0,va='top')
fig.text(0.205,0.805,r'$R=\theta\,[h(V^{*})-a]-t$  (closed form)',fontsize=6.0,va='top')
fig.text(0.205,0.765,r'monotone $h$ by NNLS; $\ln\theta\,|\,\mathbf{c}$:',fontsize=5.6,va='top',color=MID)
fig.text(0.205,0.735,r'extended-Taylor or pooled prior',fontsize=5.6,va='top',color=MID)
fig.text(0.205,0.705,r'offset moments $(\bar\epsilon,\sigma_\epsilon)$',fontsize=5.6,va='top',color=MID)
# ---------- B: networks
box(0.005,0.33,0.385,0.295,'Deep evidence streams','#fdf6ea',cB,'B')
ax=mini(0.058,0.40,0.13,0.155)
O=pickle.load(open('out/rul2_NUAA.pkl','rb')); N=Nets('NUAA',True)
t='W3'; m=emask(t); _,Rs=N.outer(t)
ax.plot(D[t]['tte'][m],rul_true(t)[m],color=INK,lw=0.7,ls=(0,(1,0.8)))
ax.plot(D[t]['tte'][m],Rs['z'][m],color=cB,lw=0.6)
ax.fill_between(D[t]['tte'][m],Rs['z'][m]-1.645*Rs['s'][m],Rs['z'][m]+1.645*Rs['s'][m],color=cB,alpha=0.2,lw=0)
ax.set_xlabel('$t$ (s)',labelpad=1); ax.set_ylabel('RUL (s)',labelpad=1); ax.set_ylim(0,None)
fig.text(0.205,0.56,r'1D-CNN + reference statistics + $\mathbf{c}$',fontsize=5.8,va='top')
fig.text(0.205,0.52,r'increment ens.: $(y^I_k,\hat s^I_k)$',fontsize=6.0,va='top')
fig.text(0.205,0.48,r'quantile ens.: $(y^R_k,\hat s^R_k)$',fontsize=6.0,va='top')
fig.text(0.205,0.44,'mini-plot: unseen tool W3,',fontsize=5.5,va='top',color=MID)
fig.text(0.205,0.41,'nominal 90% band misses the',fontsize=5.5,va='top',color=MID)
fig.text(0.205,0.38,'truth (dotted) for long stretches',fontsize=5.5,va='top',color=MID)
# ---------- C: error model
box(0.005,0.005,0.385,0.295,'Persistent-bias error model','#eef6f2',cC,'C')
ax=mini(0.058,0.075,0.13,0.155)
lags=np.arange(31); A=[]
for k in T:
    for j in T:
        if j==k: continue
        I,R=N.inner(k,j); z=(I['z']-(D[j]['wte']-D[j]['wte'][0]))/I['s']
        if len(z)>35: v=np.mean(z*z); A.append([np.mean(z[l:]*z[:len(z)-l])/v if l else 1 for l in lags])
A=np.array(A); ax.fill_between(lags,np.percentile(A,25,0),np.percentile(A,75,0),color=cC,alpha=0.25,lw=0); ax.plot(lags,np.median(A,0),color=cC,lw=1.0)
ax.plot(lags,(lags==0).astype(float),color=INK,lw=0.6,ls=(0,(1,1))); ax.set_ylim(-0.1,1.05)
ax.set_xlabel('lag (windows)',labelpad=1); ax.set_ylabel('ACF of $e_k/\\hat s_k$',labelpad=1)
fig.text(0.205,0.235,r'$e^s_k=b^s_k+w^s_k$,  $b^s_k=\phi_s b^s_{k-1}+\dots$',fontsize=5.9,va='top')
fig.text(0.205,0.195,r'$\phi_s=\rho_2/\rho_1,\ \psi_s=\rho_1^2/\rho_2,\ \kappa_s^2=\mathbb{E}\varrho^2$',fontsize=5.9,va='top')
fig.text(0.205,0.155,r'discrepancy $\delta$: OU$(\sigma_\delta,\ell_\delta)$',fontsize=5.9,va='top')
fig.text(0.205,0.115,'moments of inner-LOTO residuals;',fontsize=5.5,va='top',color=MID)
fig.text(0.205,0.085,'dotted: i.i.d. assumption',fontsize=5.5,va='top',color=MID)
# ---------- D: bank of grid filters
box(0.415,0.17,0.315,0.78,'Hypothesis-averaged grid filter','#edf3f9',cD,'D')
o=run_v2('W6',[x for x in T if x!='W6'],N.outer_srcs('W6',[x for x in T if x!='W6'])[0],INSP['NUAA'],keep_weights=True)
tt=D['W6']['tte']; mm=tt<tfail('W6')
ax=mini(0.462,0.565,0.25,0.225)
MT=np.array(o['MT'])[mm]; MT=MT/MT.max(1,keepdims=True)
ax.pcolormesh(tt[mm],o['lth'],MT.T,cmap='Blues',shading='nearest',rasterized=True)
ax.set_ylim(-1,2); ax.set_xlabel('$t$ (s)',labelpad=1); ax.set_ylabel(r'$\ln\theta$',labelpad=1); ax.set_title(r'posterior $p(\ln\theta\,|\,\mathcal{D}_{1:k})$, tool W6',fontsize=5.4,loc='left',pad=2)
for k in make_insp('W6',300,SIG_I):
    if 0<k and tt[k]<tfail('W6'): ax.axvline(tt[k],color=cE,lw=0.6,ls=(0,(2,1)))
ax.text(0.97,0.06,'inspections',transform=ax.transAxes,fontsize=4.8,color=cE,ha='right')
fig.text(0.43,0.895,r'grid over $\vartheta=(\ln\theta,\epsilon)$; Kalman on $(\delta,b^I,b^R)$',fontsize=5.9,va='top')
fig.text(0.43,0.862,r'shared covariance $\mathbf{P}$ $\Rightarrow$ $O(G)$ per window',fontsize=5.9,va='top')
ax=mini(0.462,0.255,0.25,0.165)
W=np.array(o['W'])[mm]; H=o['bank']
for lab,f,col in [(r'$c_I=\frac{1}{2}$',lambda h:h['c']['inc']==0.5,cD),(r'$c_I=1$',lambda h:h['c']['inc']==1.0,'#5f97c5'),(r'$c_R=2$',lambda h:h['c']['rul']==2.0,cB),('Taylor',lambda h:h['pi']==0,INK)]:
    ax.plot(tt[mm],W[:,[i for i,h in enumerate(H) if f(h)]].sum(1),color=col,lw=0.9,label=lab)
ax.set_ylim(-0.03,1.03); ax.set_xlabel('$t$ (s)',labelpad=1); ax.set_ylabel('$w_k(H)$',labelpad=1)
ax.legend(loc='lower right',bbox_to_anchor=(1.0,0.1),fontsize=4.6,handlelength=1.0,labelspacing=0.1,ncol=2,columnspacing=0.6)
fig.text(0.43,0.47,r'bank $H=(p,c_\theta,c_I,c_R)$ weighted by model evidence',fontsize=5.9,va='top')
# ---------- E: conformal
box(0.755,0.17,0.24,0.78,'Stacking + tool-level conformal','#f5eef6',cE,'E')
ax=mini(0.80,0.565,0.17,0.245)
r=O['W6']['insp']; m=emask('W6'); x=D['W6']['tte'][m]
ax.fill_between(x,np.maximum(r['OursCP']['lo'][m],0),r['OursCP']['hi'][m],color='#dfcbe2',lw=0)
ax.fill_between(x,r['Ours']['lo'][m],r['Ours']['hi'][m],color=BLUES[1],lw=0)
ax.plot(x,r['Ours']['R'][m],color=cD,lw=0.9); ax.plot(x,rul_true('W6')[m],color=INK,lw=0.7,ls=(0,(1,0.8)))
ax.set_ylim(0,1100); ax.set_xlabel('$t$ (s)',labelpad=1); ax.set_ylabel('RUL (s)',labelpad=1)
fig.text(0.77,0.89,r'score $E=\max\{\ell(\hat q_{lo})-\ell(R),$',fontsize=5.8,va='top')
fig.text(0.80,0.857,r'$\ell(R)-\ell(\hat q_{hi})\}$',fontsize=5.8,va='top')
fig.text(0.77,0.475,r'stack with DE+: $q^S=a\,q^F+(1-a)\,q^D$',fontsize=5.8,va='top')
fig.text(0.77,0.44,'conformal: CDF pooling, weight $1/K_j$,',fontsize=5.8,va='top')
fig.text(0.77,0.405,r'$Q=\max\{0,\hat F^{-1}(1-\alpha)\}$',fontsize=5.8,va='top')
fig.text(0.77,0.35,'output: median $\\hat R_k$ and',fontsize=5.8,va='top')
fig.text(0.77,0.315,r'calibrated interval $C_k$',fontsize=5.8,va='top')
fig.text(0.77,0.26,'purple: conformal band, blue: raw band',fontsize=5.4,va='top',color=MID)
fig.text(0.77,0.23,'(framework alone, $a=1$)',fontsize=5.4,va='top',color=MID)
# inputs
fig.text(0.43,0.10,'inputs for tool $i^\\star$: signal windows $\\mathbf{x}_k$, condition $\\mathbf{c}$,',fontsize=5.9,va='top')
fig.text(0.43,0.065,'mounting record $z_0$ and sparse wear inspections $z_j$',fontsize=5.9,va='top')
# arrows
arrow((0.39,0.80),(0.415,0.72),cA); arrow((0.39,0.48),(0.415,0.52),cB); arrow((0.39,0.15),(0.415,0.30),cC)
arrow((0.73,0.56),(0.755,0.56),cD)
arrow((0.56,0.11),(0.56,0.17),INK)
arrow((0.39,0.03),(0.875,0.165),cE,ls=(0,(3,2)),cs='angle,angleA=0,angleB=90,rad=0')
fig.text(0.882,0.045,'inner-LOTO scores\nand stacking weight $a$',fontsize=5.4,color=cE,ha='left',va='bottom')
save(fig,'fig1_framework')
