#!/bin/zsh
# V*=120 um MATWI: DE+/QR+ (outer), inner DE+ (2 members), inner fusions, stacking -> out/v7_stack.pkl
source ~/.venvs/toolrul/bin/activate
cd ~/toolrul_v120
export TH_MATWI=0.12 NIS=2 QMAX=2 NT=1
NPROC=14 python v7_matwi_deplus.py > v7/deplus2.log 2>&1
DSS=MATWI NPROC=16 python v7_inner_de_matwi.py > v7/inner_de.log 2>&1
DSS=MATWI NPROC=16 python v7_inner_fuse.py > v7/inner_fuse.log 2>&1
python v7_stack.py MATWI > v7/stack.log 2>&1
echo DONE > v7/ALLDONE_B
