"""MATWI label cleaning (rules fixed before any model is run on MATWI; see PROTOCOL_MATWI.md).
 1. order the rows of a set by SensorID (capture order); keep rows with a sensor file and a wear label;
 2. split a set where the running median (width 5) of the label drops by >=60 um and stays below the pre-drop level for >=5 rows (insert change) -> segments a, b; segments with <10 rows are dropped;
 3. drop trailing rows after the last such drop if the trailing segment has <10 rows (re-mounted / new insert images);
 4. robust path: running median (width 5) followed by isotonic (non-decreasing) regression.
"""
import pandas as pd, numpy as np, json, sys
from scipy.ndimage import median_filter
def iso(y):
    # pool-adjacent-violators, non-decreasing
    v=list(map(float,y)); w=[1.0]*len(v); idx=[[i] for i in range(len(v))]
    i=0
    while i<len(v)-1:
        if v[i]>v[i+1]:
            nv=(v[i]*w[i]+v[i+1]*w[i+1])/(w[i]+w[i+1]); v[i]=nv; w[i]+=w[i+1]; idx[i]+=idx[i+1]
            del v[i+1],w[i+1],idx[i+1]; i=max(i-1,0)
        else: i+=1
    out=np.empty(len(y))
    for val,ii in zip(v,idx): out[ii]=val
    return out
def segments(g):
    y=median_filter(g.wear.values.astype(float),size=5,mode='nearest'); cut=[]
    for i in range(5,len(y)-4):
        pre=np.max(y[max(0,i-10):i])
        if pre-y[i]>=60 and np.all(y[i:i+5]<=pre-30): cut.append(i)
    # keep first cut position of each cluster
    cuts=[c for n,c in enumerate(cut) if n==0 or c-cut[n-1]>10]
    bounds=[0]+cuts+[len(y)]; segs=[(bounds[i],bounds[i+1]) for i in range(len(bounds)-1)]
    return [s for s in segs if s[1]-s[0]>=10]
d=pd.read_csv(sys.argv[1] if len(sys.argv)>1 else 'data/matwi/labels.csv')
d=d[d.SensorFile.notna()&d.wear.notna()].sort_values(['Set','SensorID'])
out={}
for s,g in d.groupby('Set'):
    g=g.reset_index(drop=True)
    segs=segments(g)
    for n,(a,b) in enumerate(segs):
        name=f'S{s:02d}'+('' if len(segs)==1 else 'abcd'[n])
        gg=g.iloc[a:b]; y=gg.wear.values.astype(float)
        sm=iso(median_filter(y,size=5,mode='nearest'))
        out[name]=dict(set=int(s),files=list(gg.SensorFile.values),raw=y.tolist(),wear=sm.tolist())
        print(name,len(gg),'raw end',y[-3:],'clean start %.0f end %.0f'%(sm[0],sm[-1]))
json.dump(out,open('data/matwi/labels_clean.json','w'))
