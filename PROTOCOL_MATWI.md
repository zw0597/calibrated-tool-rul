# Frozen evaluation protocol: MATWI tool-wear data

Written on 24 November 2026, before any model of the framework or any baseline was trained or evaluated on MATWI.
At the time of writing, only the label files (`labels.csv`, `sets.csv`, `README.md`) and the sensor files of Set 12
(used solely to determine the file format, the sampling rate and the cutting-segment rule below) had been inspected.

Data: L. De Pauw, T. Jacobs, T. Goedemé, "Replication Data for: MATWI: A Multimodal Automatic Tool Wear Inspection dataset",
KU Leuven RDR, V1, 2025, doi:10.48804/GK6LHH. 17 carbide end-mill sets, CK45 and AISI 304, varying cutting speed, feed and depth.

## 1. What is frozen

Everything in the code release v1.3.0 (https://github.com/zw0597/calibrated-tool-rul, tag `v1.3.0`) is used unchanged:
network architecture and training settings, ensemble sizes (inner increment ensembles NIS=2, inner quantile ensembles QMAX=2,
as for NASA milling), the hypothesis bank `verif`, all clipping ranges and floors, T0, the discrepancy and offset estimators,
the Monte Carlo settings of the first-passage RUL, the two-layer conformal step, all baselines and all metrics.
The interval score (NIS) under periodic inspections is the primary endpoint; nMAE and late-life MAE are secondary.
No setting is changed after the first MATWI result is seen. Any deviation will be reported as such.

## 2. Dataset-specific choices (fixed by rules, not by results)

1. **Labels** (`v7_matwi_labels.py`): rows ordered by SensorID, rows without sensor file or wear label dropped;
   insert changes detected as a drop of the running median (width 5) by at least 60 um that persists for 5 rows
   (none detected); wear path = isotonic (non-decreasing) regression of the running median of the expert labels.
2. **Wear limit**: the value on the 15-um label grid that maximises the number of tools whose mounting wear is below
   and whose final wear is at or above it. This gives **V\* = 90 um** with 13 usable tools
   (S01–S06, S08, S09, S11–S14, S17; excluded: S07 and S10 never reach 90 um, S15 and S16 are mounted at or above 90 um).
   Sensitivity: V\* = 120 um (12 tools), reported in the Supplementary Material.
3. **Time axis**: cumulative cutting time. Within each run, the cutting segment is the longest contiguous stretch of
   25-Hz frames whose RMS in-plane force sqrt(Fx^2+Fy^2) (after removing the run median) exceeds 20% of its 95th percentile;
   the cutting time of a run is the length of this segment.
4. **Signals**: five channels (accelerometer, acoustic emission, Fx, Fy, Fz), median-removed, RMS envelope over blocks of
   65 samples (about 25 Hz, as for the other datasets); windows of L=100 frames; training windows with stride 50 inside the
   cutting segment, one test window per run at the centre of the cutting segment; one baseline run (nb=1), as for NASA milling.
5. **Cutting condition** c = (cutting speed Vc [m/min], feed per tooth fz [mm], material indicator 1 = CK45, 2 = AISI 304).
   The unknown conditions of Set 1 are imputed by the median of the other CK45 sets.
6. **Inspection interval**: one fifth of the median life (cutting time to V\*) of the 13 tools, rounded to 10 s.
7. **Protocol**: leave-one-tool-out with nested inner splits, exactly as for the other datasets.

## 3. What will be reported

All results on MATWI are reported whatever they are, in the main text, with the same tables and tests as for the other
datasets, including any baseline that outperforms the framework.

## 4. Values fixed after preprocessing (24 November 2026, before any network was trained on MATWI)

- 13 tools as listed in Section 2.2; lives (cutting seconds to V\* = 90 um): 133–3542 s, median 805 s.
- Inspection interval: 805/5 = 161 s, rounded to **160 s**; inspection noise SIG_I = 5 um as for the other datasets.
- Code changes: `lib.TH['MATWI']=0.09`, `pipeline.INSP['MATWI']=160.0`. Nothing else.
- Networks: `run_nets.py MATWI outer|qens|inner|qensin` with NIS=2, QMAX=2 (as for NASA milling); baselines with `baselines.py MATWI`
  (the script used for NASA milling); proposed method with `evalrul2.py MATWI`; DE+/QR+, CQR+, Wiener-C, PF/PF-C, Wiener-Net
  with the v1.2–v1.3 scripts.
- The nested stacking of the framework with DE+ (`v7_stack.py`, version 1.4) was specified and run on PHM2010 and NUAA before
  this point; it is applied to MATWI unchanged (weight grid 0, 0.1, ..., 1; inner-fold nMAE criterion).

## 5. Amendment (24 November 2026, 18:20, before any MATWI network finished training and before any MATWI result)

Training-window stride changed from 50 to 200 frames (non-overlapping 4-s windows; about 450 training windows per tool
instead of about 1,600), because with stride 50 one network took more than 15 minutes on the shared machine and the
819 MATWI networks would not finish in reasonable time. Test windows, labels, V\*, the time axis and all other settings are unchanged.

## 6. Amendment 2 (25 November 2026, 00:15, before any MATWI RUL result of the framework or of any baseline was seen)

The inner-fold DE+ ensembles used only for choosing the stacking weight and for the conformal scores of the stacked predictor
are trained with 2 members on MATWI instead of 5 (624 instead of 1,560 networks), as all other inner ensembles on MATWI and
NASA milling (NIS=2, QMAX=2, inner QR+ with 2 members). Reason: on the shared machine one network took about 10 minutes and
the 1,560 networks would have taken more than a day. The outer DE+/QR+ baselines keep 5 members. At this time only the
over-confidence-law check of the evidence residuals (Section 6.2 of the manuscript) had been computed on MATWI.
