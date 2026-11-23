import pandas as pd, numpy as np, pickle
import os
D=os.environ.get('UNIWEAR_DIR','data/uniwear/')+'/'
L=100
def wins(sig, t, w, stride):
    idx=np.arange(0,len(sig)-L+1,stride)
    X=np.stack([sig[i:i+L] for i in idx]).astype(np.float32)
    tt=t[idx+L-1]; ww=w[idx+L-1]
    return X,tt.astype(np.float64),ww.astype(np.float64)
out={}
p=pd.read_csv(D+'phm2010_bundle_high_resolution.csv')
pc=['force_x','force_y','force_z','vibration_x','vibration_y','vibration_z','acoustic_emission_rms']
for tag,g in p.groupby('experiment_tag'):
    s=g[pc].values; t=g.timestamp.values-g.timestamp.values[0]; w=g.tool_wear.values
    Xtr,ttr,wtr=wins(s,t,w,20); Xte,tte,wte=wins(s,t,w,100)
    out[tag]=dict(ds='PHM',Xtr=Xtr,ttr=ttr,wtr=wtr,Xte=Xte,tte=tte,wte=wte,c=np.zeros(3,np.float32),fs=25.)
n=pd.read_csv(D+'nuaa_orthogonal_bundle_high_resolution.csv')
nc=['axial_force','bending_moment_x','bending_moment_y','torsion','vibration1','vibration2','spindle_power','spindle_current']
for tag,g in n.groupby('experiment_tag'):
    g=g.sort_values('timestamp')
    s=g[nc].values; t=g.timestamp.values-g.timestamp.values[0]; w=g.tool_wear.values
    Xtr,ttr,wtr=wins(s,t,w,40); Xte,tte,wte=wins(s,t,w,150)
    c=np.array([g.feed_per_tooth.iloc[0],g.spindle_speed.iloc[0],g.axial_cutting_depth.iloc[0]],np.float32)
    out[tag]=dict(ds='NUAA',Xtr=Xtr,ttr=ttr,wtr=wtr,Xte=Xte,tte=tte,wte=wte,c=c,fs=15.)
for k,v in out.items(): print(k,v['Xtr'].shape,v['Xte'].shape, v['tte'][1]-v['tte'][0])
pickle.dump(out,open('data/win.pkl','wb'))
