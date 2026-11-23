# Calibrated remaining-useful-life prediction of cutting tools under persistent deep-network errors

Code and processed results for the manuscript

> Y. Liu, Z. Zeng, Y. Chen, P. Zhang, S. Du, Z. Tian, W. Zeng, *Calibrated remaining-useful-life prediction of cutting tools under persistent deep-network errors*, submitted to *Reliability Engineering & System Safety*.

The repository contains everything needed to reproduce every table and figure of the paper:

- **Option A: rebuild tables and figures from the stored predictions.** The network predictions and evaluation results are included in `out/`, so this takes minutes on a laptop CPU.
- **Option B: rerun the whole pipeline.** This retrains all networks and re-evaluates all methods, and takes about one to two days on a few CPU cores.

Both options need the public datasets (not redistributed here; download them from the sources in Section 2), because every script reads the windowed data file `data/win.pkl`. That file is built from the raw data and is not redistributed here.

## 1. Requirements

Tested with Python 3.13 and the package versions below. Earlier recent versions are expected to work but have not been tested.

```
numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, matplotlib 3.11.2, torch 2.15 (CPU is sufficient)
```

```bash
pip install -r requirements.txt
```

The figures use the Liberation Sans font if it is installed; otherwise matplotlib falls back to its default sans-serif font.

## 2. Data

| Dataset | Source | File expected by the code |
|---|---|---|
| PHM2010 (tools c1, c4, c6) | PHM Society 2010 Data Challenge, in the down-sampled form of the Katulu Uniwear bundle: https://github.com/katulu-io/uniwear-dataset | `data/uniwear/phm2010_bundle_high_resolution.csv` |
| NUAA (tools W1–W9) | NUAA_Ideahouse tool-wear dataset (IEEE DataPort), Uniwear bundle as above | `data/uniwear/nuaa_orthogonal_bundle_high_resolution.csv` |
| NASA milling (cases 1–16, case 6 dropped) | A. Agogino, K. Goebel, Milling Data Set, NASA Prognostics Data Repository | `data/mill.mat` |
| Milling tool life (independent tools, no wear data) | M. Piecuch, T. Żabiński, Scientific Data 12, 650 (2025) | `data/scidata/FeatureAndMetadata_Milling.csv` |
| C-MAPSS FD001 (turbofan engines) | A. Saxena et al., NASA Prognostics Data Repository | `data/cmapss/train_FD001.txt` |

Build the windowed data. Each step writes `data/win.pkl`, and the second step appends to the file written by the first:

```bash
UNIWEAR_DIR=data/uniwear python prep.py        # PHM2010 + NUAA
MILL_MAT=data/mill.mat python prep_mill.py      # NASA milling
```

## 3. Option A: tables and figures from the stored results

```bash
python make_tables2.py     # Tables 4–7: error-model parameters, main results, ablation   -> tables/
python make_tab_wear.py    # Table 3: evidence-network architecture study                -> tables/
python make_sig.py         # Table 8: paired significance tests                          -> tables/
python f1_framework.py; python f2_data.py; python f3_evidence.py; python f4_traj.py
python f5_post.py;      python f6_tools.py; python f7_calib.py;  python f8_sens.py      # -> figures/
python f_graphabs.py      # graphical abstract -> figures/
```

We checked that these commands regenerate the tables of the manuscript character for character and the figures pixel for pixel. Reference copies of all outputs are already in `tables/` and `figures/`.

## 4. Option B: full pipeline

All scripts are run from the repository root. `NT` sets the number of PyTorch threads, and `FOLDS`/`SUF` can be used to split long jobs over several processes (see `scripts/reproduce_all.sh`).

| Step | Command | Output |
|---|---|---|
| Outer LOTO evidence networks | `python run_nets.py <DS> outer` | `out/nets_<DS>_outer.pkl` |
| Outer quantile ensembles | `python run_nets.py <DS> qens` | `out/nets_<DS>_qens.pkl` |
| Nested (inner LOTO) networks | `python run_nets.py <DS> inner`, `innerde`, `qensin` | `out/nets_<DS>_inner*.pkl`, `..._qensin*.pkl` |
| Architecture study (Table 3) | `python run_nets.py <DS> arch` | `out/nets_<DS>_arch.pkl` |
| Baselines | `python evalrul.py PHM`, `python evalrul.py NUAA`, `python baselines.py MILL` | `out/rul_<DS>.pkl` |
| Proposed method and ablations | `python evalrul2.py <DS>` | `out/rul2_<DS>.pkl` |
| Inspection-budget study | `python sens2.py <DS>` | `out/sens2_<DS>.pkl` |
| Significance tests | `python sigtest.py` | `out/sigtests.pkl` |
| Bank-design robustness checks | `python qtest.py <DS> <bank>` with bank in `verif`, `verif4`, `verifw`, `verifo`, `verifoi` | `out/q_<DS>_<bank>.pkl` |
| Inner-fold selection of the source-off variant | `python innersel.py <DS> <bank>` | `out/inner_<DS>_<bank>.pkl` |

`<DS>` is one of `PHM`, `NUAA`, `MILL`. Then run the commands of Section 3.

Settings used in the paper: the inner increment ensembles have three members for PHM2010 and two for NUAA and NASA milling (`NIS=2`), and the inner quantile ensembles have five members (`QMAX=5`), except for NASA milling, where they have two (`QMAX=2`). Network training uses fixed seeds. Results obtained on other hardware or with other library versions may still differ slightly in the last digit, because floating-point operations are not bit-identical across platforms.


## 4a. Additions of the revised manuscript (version 1.1)

These steps use only the stored network predictions in `out/` and `data/win.pkl`; together they take a few minutes on a laptop.

```bash
python v4_extra.py hybrid    # Wiener-Net hybrid baseline                       -> out/v4_hybrid.pkl
python v4_extra.py ablate    # ablations: predictive scale, pseudo-BMA variants   -> out/v4_ablate.pkl
python v4_extra.py sens      # sensitivity to fixed constants                     -> out/v4_sens.pkl
python v4_extra.py t0        # inner fusions for the T0 sensitivity               -> out/v4_inner_insp.pkl
python v4_report.py; python v4_report2.py                   # printed summaries, T0 sensitivity (out/v4_t0.pkl)
python make_tables_v4.py      # Tables 4-7 with Wiener-Net and the new ablation rows -> tables/
python make_tab_wear_v4.py    # Table 3 with the NASA milling column              -> tables/
python make_sig_v4.py         # Table 8 with Wiener-Net and pooled tests          -> tables/
python make_tab_sens_v4.py    # Supplementary Table S1                           -> tables/
python f1_framework.py; python f6_tools.py; python f8_sens.py; python f_graphabs.py   # updated figures
```

`v2.py` gained two switches used only by the ablations (`PREDSCALE`, `PBMA_MODE`); their defaults reproduce the paper. The earlier scripts `make_tables2.py`, `make_tab_wear.py` and `make_sig.py` still reproduce the tables of version 1.0.

## 4b. Sensitivity to the wear limit V* (version 1.1, Supplementary Section S6)

This study retrains the V*-dependent networks (about 1,100 networks per setting, roughly 40 minutes on 17 CPU cores). The wear limits are set with the environment variables `TH_PHM`, `TH_NUAA` and `TH_MILL` (read by `lib.py`; without them the limits of the paper are used). Run each setting in its own work directory containing `data/`, symbolic links to the scripts and an `out/` folder with the wear-increment networks (`nets_<DS>_outer.pkl` and `nets_<DS>_inner.pkl`, increment keys only):

```bash
TH_PHM=0.12 TH_NUAA=0.155 TH_MILL=0.35 python vstar_nets.py && TH_PHM=0.12 TH_NUAA=0.155 TH_MILL=0.35 python vstar_eval.py   # -> out/vstar_eval.pkl
python vstar_nets.py && python vstar_eval.py                                                                                 # limits of the paper
TH_PHM=0.17 TH_NUAA=0.165 TH_MILL=0.40 python vstar_nets.py && TH_PHM=0.17 TH_NUAA=0.165 TH_MILL=0.40 python vstar_eval.py
```

Copy the three `vstar_eval.pkl` files to `out/vstar_eval_A.pkl`, `out/vstar_eval_ref.pkl` and `out/vstar_eval_B.pkl` (included) and run `python make_tab_vstar_v4.py` for Supplementary Table S2. In version 1.1 the method-of-moments estimator (`pipeline.mom`) skips residual sequences shorter than three windows, for which the lag-two moment is undefined; this does not change any result of the paper.

## 4c. Version 1.2 (fifth manuscript version)

* `fuse2.py`: `RUL_DISC=True` (default) computes the RUL as the first passage of the wear path including the model discrepancy (Monte Carlo, 1000 draws, 300 steps); `RUL_DISC=False` gives the nominal-path RUL of version 1.1. All results of the paper were recomputed with the default (`evalrul2.py`, `sens2.py`, `v4_extra.py`, `vstar_eval.py`; the version-1.1 results are kept in `out/v4_backup/`).
* `lib.py`: `AUX` adds the mounting wear and inspection features to the network inputs (DE+, QR+ baselines).
* `v5_baselines.py nets|genpath`: DE+/QR+ and the particle-filter baselines PF/PF-Net.
* `v5_plus.py nets|eval`: the framework with QR+ as direct-RUL evidence.
* `cmapss_study.py nets|eval`: error model and fusion on C-MAPSS FD001 (data in `data/cmapss/`, from the NASA Prognostics Data Repository).
* Tables: `make_tables_v5.py`, `make_sig_v5.py`, `make_tab_cmapss_v5.py` (plus the v4 scripts for Tables 3, S1, S2).

## 4d. Version 1.3 (sixth manuscript version)

* `v6_analysis.py cqrplus|decision|tests`: CQR+ baseline, replacement-policy study, pooled Holm-corrected tests (NIS primary).
* `v6_wiener_cov.py`, `v6_pf_cov.py`: Wiener and particle-filter baselines with condition-dependent priors (Wiener-C, PF-C).
* `v6_variants.py`: observation-model variants (`fuse2.DELTA0_COMP`, `fuse2.CORR_IR`, `fuse2.ERR_RI` with `pipeline.mom_ri`); defaults reproduce the paper.
* `v6_jack_nets.py` + `v6_jackplus.py`: tool-level jackknife+ (inner networks re-applied to the outer test tool).
* `scidata_study.py`: independent tool data (Piecuch and Zabinski, Sci. Data 12, 650, 2025; place `FeatureAndMetadata_Milling.csv` in `data/scidata/`).
* Tables: `make_tables_v6.py`, `make_sig_v6.py`, `make_tab_decision_v6.py`, `make_tab_generality_v6.py`, `make_tab_variants_v6.py`.

## 5. Code map

| File | Content |
|---|---|
| `lib.py` | data access, evidence networks (1D-CNN with reference statistics and condition input), training and prediction |
| `mc.py` | mixed-effects registration of the scale-family wear law, extended-Taylor and pooled priors, small-sample predictive scale |
| `pipeline.py`, `fuse.py` | evidence sources, model discrepancy, method-of-moments identification of the persistent-bias error model, conformal utilities |
| `fuse2.py` | Rao–Blackwellised grid filter with shared covariance and exact Bayesian averaging over the hypothesis bank (`fuse_bank`) |
| `v2.py` | complete framework (`run_v2`): hypothesis banks, pseudo-BMA hyper-weights, offset statistics, network access (`Nets`) |
| `metrics.py`, `metrics2.py` | RUL metrics (nMAE, RMSE, late-life MAE, α–λ, PICP, NMPIW, NIS), two-layer and finite-sample conformal calibration |
| `evalrul.py`, `baselines.py`, `evalrul2.py`, `sens2.py` | experiments |
| `make_*.py`, `f*_*.py`, `style2.py` | tables and figures |
| `vstar_nets.py`, `vstar_eval.py`, `make_tab_vstar_v4.py` | sensitivity to the wear limit (Supplementary Section S6) |
| `v4_extra.py`, `v4_report.py`, `v4_report2.py` | Wiener-Net baseline, additional ablations, sensitivity analyses and pooled tests of the revised manuscript |

The default hypothesis bank of the paper is `BANKDEF='verif'` in `v2.py`. The banks `verif4`, `verifw`, `verifo` and `verifoi` are the robustness variants reported in Section 6.5 and Supplementary Section S4 of the paper. A reliability multiplier of `0.0` in a bank denotes a switched-off source, which is then explained by the population-average wear path.

## 6. Licence and citation

The code is released under the MIT licence (see `LICENSE`). The datasets remain subject to the terms of their original providers. If you use this code, please cite the paper (see `CITATION.cff`).
