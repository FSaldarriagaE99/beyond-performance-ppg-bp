# Beyond Performance: Demographic and Fiducial-Quality Effects in PPG-Based Blood Pressure Estimation

Code accompanying the paper above: non-invasive blood pressure (BP) estimation from PPG signals using LightGBM, with SHAP-based feature importance and demographic stratification analysis, on the PulseDB/VitalDB dataset. Targets: **SBP** (systolic), **DBP** (diastolic), **MAP** (mean arterial pressure).

This repository accompanies the paper's Code Availability statement: preprocessing and feature extraction settings, the quality-control implementation, model configurations and hyperparameter search, and the bootstrap and equivalence testing procedures used throughout the paper and its annex.

## Repository layout

```
Scripts/
  lib_changes/                    pyPPG overrides (modified fiducial-point and biomarker
                                   extraction; this is how the 28 PPG features are computed)
  fearture_extraction.py          Feature_Extraction class - pyPPG wrapper
  other_functions_PPG.py          Others class (imported by fearture_extraction.py)
  new_extraction.py               Feature-extraction driver
  new_extraction_fiducials.py     Fiducial-only extraction driver
  new_extraction_loop.py          Batched extraction driver over PulseDB supplementary subsets

  fiducial_quality_filtering/     Signal quality-control pipeline: per-signal metrics,
                                   pass/fail scoring (checker.py/cleaner.py), the main
                                   run_pipeline.py entrypoint, threshold and alpha/beta
                                   sensitivity sweeps, and the distributional-shift analysis

  bp_lgbm/                        ML pipeline
    data.py, preprocessing.py, config.py, models.py, gs.py, cv.py, eval.py,
    shap_analysis.py, demo_strata_utils.py, plots.py, bootstrap_stats.py   Core modules
    Experiment_*.py / Abblation_*.py / Sweep_*.py                          Runnable entrypoints -
                                   each reproduces a specific figure/table in the paper or
                                   annex; see the docstring at the top of each file
    figures/                      Jupyter notebooks that render the paper's figures from
                                   each experiment's saved results
```

Data flows as: `data.py -> preprocessing.py -> [config.py + models.py] -> gs.py / cv.py -> eval.py -> shap_analysis.py`.

## Environment setup

Python 3.10.

```
pip install -r requirements.txt
```

### Configuring paths

`Scripts/bp_lgbm/local_paths.py` is included as the main machine-local configuration file. Edit `PULSE_DB_SUP_DIR` there to point to the PulseDB/VitalDB supplementary subset on your machine (see `dataset.txt`). The `fiducial_quality_filtering` layer has its own gitignored `local_paths.py`; copy `Scripts/fiducial_quality_filtering/local_paths.py.example` there and fill in `DATA_DIR` if you run that layer. Every other path - cached artifacts, sweep results, SHAP/grid-search outputs, figures - resolves automatically to `results/` at the repo root via `Scripts/bp_lgbm/repo_paths.py`, so nothing else needs configuring.

`Scripts/new_extraction*.py` (feature extraction, upstream of both layers above) is the one exception: it does not read `local_paths.py`. Edit the `DATA_DIR` / `OUTPUT_ROOT` constants near the top of each script directly - see each script's module docstring.

## Data

The PulseDB/VitalDB-derived `.h5` feature and label files are not distributed in this repository (size and data-use terms). See the paper's Methods for dataset provenance, ethics approval, and the PulseDB/VitalDB access process.

## Running the pipeline

All scripts use relative imports; run each from inside the folder that contains it:

```
cd Scripts/fiducial_quality_filtering
python run_pipeline.py

cd ../bp_lgbm
python Experiment_GridSearch.py
python Experiment_eval.py
python Experiment_SHAPAnalysis.py
```

`Experiment_*.py`, `Abblation_*.py`, and `Sweep_*.py` are the runnable entrypoints; each has a docstring describing the analysis it performs.

## Reproducibility notes

- All experiments use random seed 42.
- LightGBM runs with `n_jobs=-1`. Parallel floating-point aggregation is not guaranteed bit-for-bit identical across runs; the practical effect is under 0.001 mmHg, invisible at the precision reported in the paper.
- Trained models are not persisted to disk - each script retrains from its saved grid-search config. Re-running is fast and deterministic given the seed above.

## License

MIT - see [LICENSE](LICENSE).
