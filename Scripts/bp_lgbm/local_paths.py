from pathlib import Path

# Folder containing the PulseDB/VitalDB supplementary subset .h5 files.
# Not self-contained in the repo - depends on where you downloaded
# PulseDB to. See dataset.txt for the download source and partition names.
PULSE_DB_SUP_DIR = Path(r"<path to PulseDB supplementary subsets>")

# --- Result-saving folders, one per experiment family -----------------
# Repo-relative (under results/) via repo_paths.py - self-contained,
# no machine-specific setup needed.

from repo_paths import (
    PERFORMANCE_RESULTS_PAPER,
    GS_RESULT_PAPER,
    SHAP_RESULTS_PAPER,
    DEMOG_RESULTS_PAPER,
    ABBLATION_RESULTS_PAPER,
    FIGURES_PAPER,
    DISTRIBUTION_ANALYSIS_DIR,
    THRESHOLD_SWEEP_ML_RESULTS,
    ALPHA_BETA_SWEEP_ML_RESULTS,
    FILTERING_SENSITIVITY_DIR,
)
