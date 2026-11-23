#!/usr/bin/env bash
# Full pipeline (Option B of the README). Run from the repository root.
# Expect one to two days on a few CPU cores; long steps can be split with FOLDS=... SUF=_a / _b
# and the partial result files merged afterwards, as done for the paper.
set -euo pipefail
export NT=${NT:-2}

UNIWEAR_DIR=${UNIWEAR_DIR:-data/uniwear} python prep.py
MILL_MAT=${MILL_MAT:-data/mill.mat} python prep_mill.py

for DS in PHM NUAA MILL; do
  python run_nets.py $DS outer
  python run_nets.py $DS qens
done
python run_nets.py PHM inner;  python run_nets.py PHM innerde;  QMAX=5 python run_nets.py PHM qensin
NIS=2 python run_nets.py NUAA inner; python run_nets.py NUAA innerde; QMAX=5 python run_nets.py NUAA qensin
NIS=2 python run_nets.py MILL inner; QMAX=2 python run_nets.py MILL qensin
python run_nets.py PHM arch; NAS=1 python run_nets.py NUAA arch

python evalrul.py PHM; python evalrul.py NUAA; python baselines.py MILL
for DS in PHM NUAA MILL; do python evalrul2.py $DS; python sens2.py $DS; done
python sigtest.py
for DS in PHM NUAA MILL; do for B in verif verif4 verifw verifo verifoi; do python qtest.py $DS $B; done; done
for DS in PHM NUAA MILL; do for B in verif verifoi; do python innersel.py $DS $B; done; done
python innersel.py PHM verifo

python make_tables2.py; python make_tab_wear.py; python make_sig.py
for f in f1_framework f2_data f3_evidence f4_traj f5_post f6_tools f7_calib f8_sens; do python $f.py; done

# ---- additions of the revised manuscript (v1.1) ----
python v4_extra.py hybrid && python v4_extra.py ablate && python v4_extra.py sens && python v4_extra.py t0
python v4_report.py && python v4_report2.py
python make_tables_v4.py && python make_tab_wear_v4.py && python make_sig_v4.py && python make_tab_sens_v4.py
python f1_framework.py && python f6_tools.py && python f8_sens.py && python f_graphabs.py
# V* sensitivity (Supplementary Section S6): see README section 4b; then
python make_tab_vstar_v4.py
# ---- version 1.2 ----
python v5_baselines.py nets && python v5_baselines.py genpath
python v5_plus.py nets && python v5_plus.py eval
python cmapss_study.py nets && python cmapss_study.py eval
python make_tables_v5.py && python make_sig_v5.py && python make_tab_cmapss_v5.py
# ---- version 1.3 ----
python v6_analysis.py cqrplus && python v6_wiener_cov.py && python v6_pf_cov.py
python v6_analysis.py decision && python v6_analysis.py tests && python v6_variants.py
python v6_jack_nets.py && python v6_jackplus.py
python scidata_study.py
python make_tables_v6.py && python make_sig_v6.py && python make_tab_decision_v6.py && python make_tab_generality_v6.py && python make_tab_variants_v6.py
