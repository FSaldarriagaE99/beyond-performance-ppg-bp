FIDUCIAL QUALITY FILTERING PIPELINE
====================================

Covers the quality-check / signal-discarding stage of the project - it
assumes pyPPG fiducial points have already been extracted (that happens
upstream, in Scripts/new_extraction_fiducials.py, and is NOT part of this
folder).

Where this fits in the bigger picture:

    [pyPPG fiducial extraction]  (Scripts/new_extraction_fiducials.py, upstream, not here)
                |
                v
    Fiducial_Points_<subset>.h5  +  Features_<subset>.h5
                |
                v
    === everything in this folder ===
                |
                v
    Clean_Features_<subset>_<threshold>.h5   -->  feeds Layer 2 (Scripts/bp_lgbm/)


FILES
-----

metrics.py                 Per-signal quality metrics (SignalMetrics).
checker.py                 Runs metrics.py over every signal and scores/
                            thresholds the result (QualityChecker).
cleaner.py                 Drops flagged signals from a dataset, given a
                            saved quality report (SignalCleaner).
run_pipeline.py             Main entrypoint - run this to produce a
                            cleaned feature set for one or more subsets.
raw_metrics_cache.py        Vectorized/cacheable version of the expensive
                            per-signal metrics loop, used by the sweeps.
build_cache.py               Builds the cache raw_metrics_cache.py needs.
sweep_common.py             Shared helpers for the two sweep scripts.
sweep_alpha_beta.py         Sensitivity sweep over the criterion-4
                            weights (alpha/beta).
sweep_thresholds.py         Sensitivity sweep over the discard thresholds.
rebuild_clean_features.py   Rebuilds a Clean_Features_*.h5 straight from
                            an existing sweep JSON, without re-running the
                            checker.
distribution_shift.py       Compares feature/demographic/label
                            distributions before vs. after filtering.


INPUTS (per subset, from PulseDB SupplementarySubsets folder)
---------------------------------------------------------------

Fiducial_Points_<subset>.h5
    Group "PPG_fiducial_points", dataset "Fiducials", shape
    (n_windows_per_signal * 16, n_signals). Each column is one signal;
    every 16 consecutive rows are one window's fiducial indices, in the
    order defined by metrics.FIDUCIAL_ORDER.

Features_<subset>.h5
    Flat datasets, each shaped (1, n_signals) except PPG_Features which is
    (28, n_signals): Age, BMI, DBP, Gender, Height, MAP, PPG_Features, SBP,
    SF, Subject, Weight. n_signals and column order must match the
    fiducials file exactly (they are joined purely by column position).


OUTPUTS (paths come from local_paths.py - copy local_paths.py.example and
fill in DATA_DIR)
---------------------------------------------------------------------

Written to DATA_DIR (data-sized, kept alongside the raw PulseDB data, not
in the repo - see dataset.txt):

metrics_<subset>.h5
    The full quality report: one row per signal, columns = checkHR,
    checkSP, numberProperFiducials, combinedScore, ppg/d1/d2/d3 (percent of
    each derivative's fiducials properly ordered), report (1 = discard).
    This is the input cleaner.py reads.

Clean_Features_<subset>_<threshold>.h5
    Same shape/format as Features_<subset>.h5, with flagged signals
    dropped. threshold in the filename is DISCARD_THRESHOLDS["thres_
    fiducials"] from run_pipeline.py's config (90 by default).

Written to QUALITY_FILTERING_REPORTS_DIR (repo-relative,
results/quality_filtering):

Problems_Fiducials_<subset>.xlsx
    Mean percent of windows flagged as problematic, per fiducial point
    (on, sp, dn, ... p2), across the whole subset. Descriptive, pre-drop.

criteria_report_<subset>.csv
    THE per-criterion percentages/scores artifact, computed before
    anything is dropped: % failing HR check, % failing peak-count check,
    mean/std % fiducials detected, mean/std combined score, % flagged for
    removal, and (once detect() has run) the count actually removed.

Demographic_Info_<subset>.xlsx
    Per-subject total/removed signal counts and percentage removed.
    Descriptive, computed AFTER dropping (this one necessarily reflects the
    drop, since it's reporting on the drop itself).


SWEEP OUTPUTS (written to FILTERING_SENSITIVITY_DIR - local_paths.py,
repo-relative, results/filtering_sensitivity - one subfolder per subset)
---------------------------------------------------------------------------

<subset>/alpha_beta_sweep_summary.csv
<subset>/threshold_sweep_summary.csv
    One row per grid point (alpha value, or threshold value): the swept
    parameter(s), samples dropped (raw count + % of the full subset),
    subjects with >=1 dropped sample (raw count + %), and subjects
    ENTIRELY dropped - 100% of their samples gone (raw count + %). Both
    subject-drop definitions are reported side by side, since they can
    diverge a lot.

<subset>/dropped_signals/alpha_beta/alpha_<value>.json
<subset>/dropped_signals/threshold/thres_<value>.json
    One JSON file per grid point: which exact signals were dropped at that
    parameter value. Schema:
        {
          "subset": "...",
          "parameters": {"w_consistency": 0.3, "w_alignment": 0.7,
                          "thres_fiducials": 90, "thres_score": 90},
          "n_total_signals": 465480,
          "n_dropped": 276707,
          "dropped": {"p000001_1": [3, 7, 12, ...], "p000003_1": [0, 1, ...]}
        }
    "dropped" maps subject id -> list of dropped LOCAL sample indices (the
    signal's position within that subject's own block of signals, in
    Features_<subset>.h5 file order - not a global index). This is exactly
    what you need to build a per-subject keep/drop boolean mask when
    loading the full dataframe for training, without re-running any of
    this pipeline.


DISTRIBUTION SHIFT OUTPUTS (written to DISTRIBUTION_ANALYSIS_DIR -
local_paths.py, repo-relative, results/distribution_analysis)
---------------------------------------------------------------------------

shift_pooled_<subset>_<threshold>.csv
    One row per column (28 PPG features + Age/BMI/Height/Weight/SBP/DBP/
    MAP - 35 rows total), comparing the full original dataset against
    Clean_Features_<subset>_<threshold>.h5: reference_q1/q3/iqr,
    wasserstein, normalized_wasserstein (wasserstein / reference_iqr),
    ks_statistic_D, ks_pvalue, pct_change_{mean,median,q1,q3,iqr,mad},
    and IQR/MAD outlier counts/percentages before+after plus their
    Jaccard overlap. See distribution_shift.py's docstring for exact
    formulas and the positional-index caveat on the Jaccard numbers.


HOW TO RUN
----------

First copy local_paths.py.example to local_paths.py and fill in DATA_DIR
for your machine. Edit the CONFIGURATION block at the top of
run_pipeline.py if you want different alpha/beta/thresholds.

From inside this folder, using the project's Python 3.10 virtual
environment:

    ..\..\.venv310\Scripts\python.exe run_pipeline.py

For a sweep, run once (expensive - ~60-90 min for the Train subset,
~465k signals):

    ..\..\.venv310\Scripts\python.exe build_cache.py

then as many times as you like (cheap, seconds, since it only loads the
cache and does vectorized numpy - no per-signal loop):

    ..\..\.venv310\Scripts\python.exe sweep_alpha_beta.py
    ..\..\.venv310\Scripts\python.exe sweep_thresholds.py

Edit ALPHA_GRID / THRESHOLD_GRID / FIXED_THRESHOLDS / FIXED_WEIGHTS at the
top of those two scripts to change the grid or the held-fixed parameter.

To rebuild Clean_Features_*.h5 from an existing threshold-sweep JSON
(e.g. after a config fix, without re-running the checker):

    ..\..\.venv310\Scripts\python.exe rebuild_clean_features.py

To (re)generate the distribution-shift CSVs (fast - loads Features_*.h5 +
Clean_Features_*.h5 directly, no cache/checker involved):

    ..\..\.venv310\Scripts\python.exe distribution_shift.py
