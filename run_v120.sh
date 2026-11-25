#!/bin/zsh
# MATWI V*=120 um sensitivity (protocol Sect. 2.2): full framework pipeline + baselines in ~/toolrul_v120
source ~/.venvs/toolrul/bin/activate
cd ~/toolrul_v120
export TH_MATWI=0.12 NIS=2 QMAX=2 NT=1
T=(X01 X02 X03 X04 X05 X08 X11 X12 X13 X14 X16 X17)
( NPROC=4 python v7_matwi_deplus.py > v7/deplus.log 2>&1 ) &
for k in $T; do
  ( for part in outer qens inner qensin; do SUF=_$k FOLDS=$k python run_nets.py MATWI $part > v7/nets_${k}_$part.log 2>&1; done ) &
done
wait
python - <<'PY'
import pickle,glob
for part in ['outer','qens','inner','qensin']:
    R={}
    for f in sorted(glob.glob(f'out/nets_MATWI_{part}_X*.pkl')): R.update(pickle.load(open(f,'rb')))
    pickle.dump(R,open(f'out/nets_MATWI_{part}.pkl','wb')); print(part,len(R))
import os
for f in glob.glob('out/nets_MATWI_*_X*.pkl'): os.rename(f, f.replace('out/','v7/'))
PY
python baselines.py MATWI > v7/baselines.log 2>&1
for k in $T; do ( SUF=_$k FOLDS=$k python evalrul2.py MATWI > v7/eval_$k.log 2>&1 ) & done
wait
python - <<'PY'
import pickle,glob
O={}
for f in sorted(glob.glob('out/rul2_MATWI_X*.pkl')): O.update(pickle.load(open(f,'rb')))
pickle.dump(O,open('out/rul2_MATWI.pkl','wb')); print('rul2',len(O))
PY
echo ALLDONE > v7/ALLDONE
