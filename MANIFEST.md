# Where each result comes from

Every table, figure and annex in this paper can be regenerated with the code in the accompanying repository.
This annex lists, for each one, the script that produces it, what that script reads, and the settings that matter.
Scripts are in `Scripts/bp_lgbm/` unless a folder is given. Experiment scripts write their output to `results/`
(the folder name is set at the top of each script).

## The pipeline in one line

PulseDB subsets → **1** extract PPG features → **2** score and filter signal quality → **3** train and evaluate models
→ **4** compare models with bootstrap statistics → **5** draw figures.

## Settings shared by all experiments

- **Data.** PulseDB VitalDB supplementary subsets (see `dataset.txt`): training 465,480 segments from 1,293 subjects; testing 57,600 segments from 144 subjects; 28 PPG features per segment.
- **Seed.** 42 everywhere (data splits, LightGBM, bootstrap).
- **Two LightGBM configurations.** *Generalisable*: `results/grid_search/Full_Grid_Randomized_search_3targets.json`. *Fit-biased*: `stress_test_params` in `config.py`. Both are listed in Annex 2.
- **Bootstrap.** 1,000 resamples of test subjects (with replacement), the same resamples for every model in a comparison; 95% percentile intervals.
- **Equivalence tests.** Margin of ±2 percentage points of R², 90% intervals (`bootstrap_stats.py`).
- **Models with demographics.** Columns are ordered Age, Gender, Height, Weight, BMI, then the 28 PPG features.

## Run order

1. Download the PulseDB VitalDB subsets and set `PULSE_DB_SUP_DIR` in `Scripts/bp_lgbm/local_paths.py` (and `DATA_DIR` in `Scripts/fiducial_quality_filtering/local_paths.py`).
2. **Features:** `Scripts/new_extraction_loop.py` (training set), `Scripts/new_extraction_fiducials.py` (test set).
3. **Quality filtering** (`Scripts/fiducial_quality_filtering/`): `build_cache.py`, `sweep_thresholds.py`, `sweep_alpha_beta.py`, `rebuild_clean_features.py`, `distribution_shift.py`.
4. **Grid search:** `Experiment_GridSearch.py` (the resulting configuration is already provided in `results/grid_search/`).
5. **Main models:** `Experiment_eval.py` (run once with `DATASET = "original"` and once with `"clean_90"`), `Stress_test.py`, `Experiment_PPG_Demographics.py`, `Demographic_Baseline_Model.py`, `LGBM_Demographic_Baseline_Model.py`.
6. **Bootstraps and sweeps:** `Experiment_Bootstrap*.py`, `Experiment_Bootstrap_CI*.py`, `Experiment_MDE_Equivalence_*.py`, `Sweep_Threshold_ML.py`, `Sweep_AlphaBeta_ML.py`.
7. **Further analyses:** `Experiment_Demographic_Strata.py`, `Abblation_SampleSize_Uncertainty.py`, `Abblation_SubjectSize_Uncertainty.py`, `Abblation_TestBootstrap_CI.py`, `Experiment_SHAPAnalysis.py`, `Experiment_Demographic_Additive_Ablation.py`.
8. **Figures:** the notebooks in `Scripts/bp_lgbm/figures/` (and `Shap_results_visualization.py` for Figure 12).

## Files produced along the way

| File | Made by | From |
|---|---|---|
| `Features_*.h5`, `Fiducial_Points_*.h5` | `Scripts/new_extraction_loop.py`, `new_extraction_fiducials.py` | PulseDB subsets |
| `RawMetricsCache_*.npz` (per-beat quality measures) | `fiducial_quality_filtering/build_cache.py` | Fiducial points, features |
| Dropped-segment lists for each threshold and weighting | `fiducial_quality_filtering/sweep_thresholds.py`, `sweep_alpha_beta.py` | `RawMetricsCache_*.npz` |
| `Clean_Features_*_80.h5`, `_90.h5` | `fiducial_quality_filtering/rebuild_clean_features.py` | Dropped-segment lists, features |
| `shift_pooled_*.csv` (distribution shift) | `fiducial_quality_filtering/distribution_shift.py` | Features, clean features |

## Tables

| Table | Script | Reads | Settings |
|---|---|---|---|
| 1. Dataset summary | `Demographics_Analysis.py` | Original features | Unfiltered training and test sets |
| 2. PPG-only model | `Experiment_eval.py`; intervals from `Experiment_Bootstrap.py` and `Experiment_Bootstrap_CI.py` | Original features, generalisable configuration | `DATASET = "original"`; trained on 90% of the training segments (10% held out), tested on all 144 test subjects |
| 3. Filtering at four thresholds | `fiducial_quality_filtering/sweep_thresholds.py` | Quality cache | Thresholds 80, 85, 90, 95%; weights 0.25 consistency, 0.75 alignment |
| 4. Outlier overlap after filtering | `fiducial_quality_filtering/distribution_shift.py` | Original and 90% filtered features | IQR outlier rule |
| 5. Four reference models | `Experiment_eval.py` (PPG only), `Experiment_PPG_Demographics.py`, `LGBM_Demographic_Baseline_Model.py`, `Demographic_Baseline_Model.py` (linear) | Original features | Training metrics on the full fitting set |
| 6. Stable level, segments per subject | `Abblation_SampleSize_Uncertainty.py` | Original features | Levels 5–100% of segments; 10 random draws per level; stable level read from the Figure 9 results using the rule in the table caption |
| 7. Stable level, subject count | `Abblation_SubjectSize_Uncertainty.py` | Original features | 25–1,293 subjects, balanced over 12 MAP bins; 10 draws; rule as in Table 6 |
| 8. Comparison with literature | Literature values; "this study" row from Table 2 | – | – |

## Figures

Notebook file names carry earlier figure numbers; the table gives the mapping.

| Figure | Script or notebook | Reads | Settings |
|---|---|---|---|
| 1, 2. Illustrations | Not generated by code | – | – |
| 3. Feature distribution shift | `figures/Fig1_DistributionalShifts.ipynb` | `shift_pooled_*_90.csv` | KS statistic and IQR-normalised Wasserstein distance, 90% filter |
| 4. Original vs filtered | `figures/Fig1_R2_OriginalVsFiltered.ipynb` | `Experiment_eval.py` (both `DATASET` settings) for training bars; `Sweep_Threshold_ML.py` (threshold 90) for test bars | Both models tested on the same filtered test set |
| 5. PPG vs demographics (paired) | `figures/bootstrap_paired_plots.ipynb` | `Experiment_Bootstrap.py`, `Experiment_Bootstrap_CI.py` | Shared bootstrap resamples |
| 6. Demographic strata | `figures/Fig_Demog_Strata.ipynb` | `Experiment_Demographic_Strata.py` | Age <40 / 40–60 / ≥60, BMI 25, sex |
| 7. Strata composition | `figures/Fig5_Dataset_Demographics.ipynb` | `Demographics_Analysis.py` | Same thresholds as Figure 6 |
| 8. Generalisable vs fit-biased | `figures/Fig6_R2_GeneralisableVsFitBiased.ipynb` | `Experiment_eval.py`, `Stress_test.py`, `Experiment_Bootstrap_CI_StressTest_Paired.py` | Paired bootstrap of the test difference |
| 9. Segments per subject | `figures/Fig_Ablation_SampleSize.ipynb` | `Abblation_SampleSize_Uncertainty.py` | As Table 6; mean ± 1 SD over draws |
| 10. Subject count | `figures/Fig_Ablation_SubjectSize.ipynb` | `Abblation_SubjectSize_Uncertainty.py` | As Table 7 |
| 11. Age ≥60 | `figures/Fig_Bootstrap_Age60_Comparison_Paired.ipynb` | `Experiment_Bootstrap_AgeStratified_Paired.py`, `Experiment_Bootstrap_CI_AgeStratified_Paired.py`, `Experiment_MDE_Equivalence_AgeStratified_Paired.py` | Both models tested on the same 76 subjects |
| 12. SHAP direction | `Experiment_SHAPAnalysis.py`, then `Shap_results_visualization.py` | Original features, generalisable configuration | 30 subject-disjoint training splits, SBP |

## Annexes

| Annex | Script | Reads |
|---|---|---|
| 1. Feature descriptions | Literature table | – |
| 2. Hyperparameters | `Experiment_GridSearch.py`, `config.py` | Original features |
| 3. Shift in targets and demographics | `fiducial_quality_filtering/distribution_shift.py` | Original and 90% filtered features |
| 4. Subjects retained after filtering | `fiducial_quality_filtering/run_pipeline.py` (per-subject summary) | Quality scores |
| 5. MAE and SDE before and after filtering | `Experiment_eval.py`; per-subject averages from `eval.compute_metrics_macro_by_subject` | Original and filtered features |
| 6. Threshold sensitivity | `Sweep_Threshold_ML.py` | Dropped-segment lists |
| 7. Weighting sensitivity | `Sweep_AlphaBeta_ML.py` | Dropped-segment lists |
| 8. Demographic strata | `Experiment_Demographic_Strata.py` | Original features |
| 9. Generalisable vs fit-biased errors | `Experiment_eval.py`, `Stress_test.py` | Original features |
| 10. Segments-per-subject ablation | `Abblation_SampleSize_Uncertainty.py`, `Abblation_TestBootstrap_CI.py` | Original features |
| 11. Subject-count ablation | `Abblation_SubjectSize_Uncertainty.py`, `Abblation_TestBootstrap_CI.py` | Original features |
| 12. SHAP, DBP and MAP | `Experiment_SHAPAnalysis.py`, `Shap_results_visualization.py` | Original features |
| 13. SHAP rank stability | `Experiment_SHAPAnalysis.py`, `Shap_results_visualization.py` | Original features |
| 14. Incremental demographic ablation | `Experiment_Demographic_Additive_Ablation.py` (writes to `results/demographic_additive_ablation/`) | Original features, generalisable configuration |
