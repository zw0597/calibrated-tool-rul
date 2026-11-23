"""Figure style v2: Arial-metric sans (Liberation Sans, TrueType), 7 pt, open axes, outward ticks,
restrained palette; panels labelled with bold lower-case letters."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import os
base=os.environ.get('LIBERATION_DIR','/usr/share/fonts/truetype/liberation/')
_HAS_LIB=all(os.path.exists(base+f) for f in ['LiberationSans-Regular.ttf','LiberationSans-Italic.ttf','LiberationSans-Bold.ttf','LiberationSans-BoldItalic.ttf'])
if _HAS_LIB:
    for f in ['LiberationSans-Regular.ttf','LiberationSans-Italic.ttf','LiberationSans-Bold.ttf','LiberationSans-BoldItalic.ttf']:
        fm.fontManager.addfont(base+f)
MM=1/25.4; W2=180*MM; W15=140*MM; W1=88*MM
plt.rcParams.update({
 'font.family':'Liberation Sans' if _HAS_LIB else 'sans-serif','mathtext.fontset':'custom','mathtext.rm':'Liberation Sans','mathtext.it':'Liberation Sans:italic',
 'mathtext.bf':'Liberation Sans:bold','mathtext.sf':'Liberation Sans','mathtext.cal':'Liberation Sans:italic',
 'font.size':7,'axes.labelsize':7,'axes.titlesize':7,'xtick.labelsize':6.3,'ytick.labelsize':6.3,
 'legend.fontsize':6.2,'legend.frameon':False,'legend.handlelength':1.6,'legend.borderaxespad':0.2,'legend.labelspacing':0.25,
 'legend.handletextpad':0.5,'legend.columnspacing':1.0,
 'axes.linewidth':0.6,'axes.edgecolor':'#262626','axes.labelcolor':'#1a1a1a','xtick.color':'#262626','ytick.color':'#262626',
 'axes.spines.top':False,'axes.spines.right':False,'xtick.direction':'out','ytick.direction':'out',
 'xtick.major.width':0.6,'ytick.major.width':0.6,'xtick.major.size':2.5,'ytick.major.size':2.5,
 'xtick.minor.width':0.45,'ytick.minor.width':0.45,'xtick.minor.size':1.4,'ytick.minor.size':1.4,
 'axes.titlepad':3,'axes.labelpad':2,
 'lines.linewidth':1.0,'lines.markersize':3.2,'patch.linewidth':0.5,'hatch.linewidth':0.5,
 'savefig.dpi':600,'savefig.bbox':'tight','savefig.pad_inches':0.02,'figure.dpi':150,'axes.unicode_minus':True,
 'pdf.fonttype':42})
if not _HAS_LIB: plt.rcParams['mathtext.fontset']='dejavusans'

INK='#1a1a1a'; MID='#6e6e6e'; LIGHT='#bdbdbd'; FAINT='#e6e6e6'
# method colours (fixed per method across all figures)
C={'CQR+':'#c2569a','DE+':'#9a5b13','Proposed':'#1b5e8c','Proposed-raw':'#1b5e8c','DE':'#d08a1f','MCD':'#e3b765','QR':'#a46da6','CQR':'#7b3f80',
   'Wiener':'#3c8c6e','Law':'#8c8c8c','Pop':'#3a3a3a','LinEx':'#b5b5b5'}
DSC={'PHM':'#b2442c','NUAA':'#1b5e8c','MILL':'#8c6d1f'}
DNAME={'PHM':'PHM2010','NUAA':'NUAA','MILL':'NASA milling'}
BLUES=['#dbe8f3','#a9c8e2','#5f97c5','#1b5e8c','#0d3554']
def panel(ax,s,x=-0.16,y=1.02,fig=None):
    ax.text(x,y,s,transform=ax.transAxes,fontsize=8,fontweight='bold',va='bottom',ha='left',color=INK)
def despine(ax,left=False,bottom=False):
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    if left: ax.spines['left'].set_visible(False)
    if bottom: ax.spines['bottom'].set_visible(False)
def save(fig,name):
    fig.savefig(f'figures/{name}.pdf'); fig.savefig(f'figures/{name}.png'); plt.close(fig)
