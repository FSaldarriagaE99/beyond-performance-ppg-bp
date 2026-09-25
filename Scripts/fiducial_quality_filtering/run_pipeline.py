"""
Entry point for the fiducial quality filtering pipeline. For each
configured subset, it:
  1. Loads the fiducial points + PPG features from HDF5.
  2. Runs QualityChecker over every signal (the expensive step).
  3. Applies the alpha/beta weights + thresholds to get a pass/fail report
     (the cheap step - see checker.py's module docstring).
  4. Saves the quality report (.h5), a fiducial-problem summary (.xlsx),
     and the full per-criterion percentage report (.csv, csv_report -
     computed BEFORE dropping anything).
  5. Drops the flagged signals and saves the cleaned feature set (.h5),
     plus a per-subject demographic removal summary (.xlsx).

Run from inside this folder:
    ..\..\.venv310\Scripts\python.exe run_pipeline.py

Edit SUBSETS / THRESHOLDS below to point at your data and to change the
alpha/beta weights or discard thresholds. This script deliberately keeps
the "expensive metrics" and "cheap scoring" calls separate (steps 2 and 3
above) so a sweep script can import QualityChecker directly, call
compute_raw_metrics() once, and loop score_and_report() cheaply - see
sweep_alpha_beta.py / sweep_thresholds.py.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from checker import QualityChecker, FIDUCIAL_ORDER
from cleaner import SignalCleaner
from local_paths import DATA_DIR, QUALITY_FILTERING_REPORTS_DIR

# --------------------------------------------------------------------------
# Configuration - edit these for your experiment. Paths come from
# local_paths.py (copy local_paths.py.example and fill in DATA_DIR).
# --------------------------------------------------------------------------

# Folder this script writes its large, data-sized outputs into (per-signal
# quality reports, Clean_Features_*.h5 - the filtered feature sets). Kept
# alongside the raw PulseDB data rather than in the repo (see dataset.txt).
DATA_OUTPUT_DIR = DATA_DIR

# Folder this script writes its small descriptive reports into (xlsx/csv
# summaries) - repo-relative, under results/quality_filtering.
REPORT_OUTPUT_DIR = QUALITY_FILTERING_REPORTS_DIR

# One entry per subset you want to process.
SUBSETS = {
    "VitalDB_Train_Subset": {
        "fiducials": DATA_DIR / "Fiducial_Points_VitalDB_Train_Subset.h5",
        "features": DATA_DIR / "Features_VitalDB_Train_Subset.h5",
    },
    "VitalDB_CalFree_Test_Subset": {
        "fiducials": DATA_DIR / "Fiducial_Points_VitalDB_CalFree_Test_Subset.h5",
        "features": DATA_DIR / "Features_VitalDB_CalFree_Test_Subset.h5",
    },
}

# Checks not affected by alpha/beta or the discard thresholds.
CHECK_THRESHOLDS = {
    "sp_limit": 2,
    "bmin": 40,
    "bmax": 200,
}

# Criterion 4 weights (alpha = w_consistency, beta = w_alignment) and the
# discard thresholds for criteria 3 (numberProperFiducials) and 4 (combinedScore).
SCORE_WEIGHTS = {"w_consistency": 0.25, "w_alignment": 0.75}
DISCARD_THRESHOLDS = {"thres_fiducials": 90, "thres_score": 90}


# --------------------------------------------------------------------------
# Loading helpers
# --------------------------------------------------------------------------

def load_fiducials(path: Path) -> np.ndarray:
    """
    Reads the flat PulseDB-supplement fiducial file. Returns the raw
    (n_windows*16, n_signals) array under "PPG_fiducial_points/Fiducials".
    """
    with h5py.File(path, "r") as f:
        return f["PPG_fiducial_points"]["Fiducials"][()]


def load_features(path: Path) -> dict:
    """
    Reads a flat PulseDB-supplement features/labels file into a dict of
    pandas objects, one entry per top-level HDF5 dataset, each transposed
    so rows = signals (matching the fiducial signal ordering).
    """
    data = {}
    with h5py.File(path, "r") as f:
        for name in f:
            values = f[name][()]
            if name == "PPG_Features":
                data[name] = pd.DataFrame(values.T)
            else:
                # Row vectors (Age, SBP, Subject, ...): keep as a flat array.
                data[name] = values[0]
    return data


def build_checker_inputs(fiducials: np.ndarray, sampling_freq: float, n_samples: int):
    """
    Wraps a single subset's fiducial array into the dict-of-groups shape
    QualityChecker expects. There is only one "group" (called "Full_set")
    per subset here, because the PulseDB supplementary format is flat
    (no real per-patient grouping) - unlike the patient-grouped H5 format
    used elsewhere in this project (see CLAUDE.md).
    """
    n_signals = fiducials.shape[1]
    signal_ids = np.arange(n_signals)
    return (
        {"Full_set": fiducials},
        {"Full_set": {"SamplingFrequency": sampling_freq}},
        {"Full_set": n_samples},
        {"Full_set": signal_ids},
    )


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def run_subset(subset_name: str, fiducials_path: Path, features_path: Path,
                data_output_dir: Path, report_output_dir: Path,
                sampling_freq: float = 125, n_samples: int = 1250,
                max_signals: int = None):
    """
    max_signals : if set, only the first N signals (columns) are processed.
        This is a development/testing knob only - compute_raw_metrics() is a
        plain per-signal Python loop, and a full subset can have several
        hundred thousand signals, so a full run takes a long time (see
        README.txt). Use max_signals for a quick smoke test; leave it as
        None for a real, complete run.
    """
    data_output_dir.mkdir(parents=True, exist_ok=True)
    report_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {subset_name} ===")

    print("Loading fiducials:", fiducials_path)
    fiducials = load_fiducials(fiducials_path)
    if max_signals is not None:
        fiducials = fiducials[:, :max_signals]
        print(f"  (dev mode: truncated to first {max_signals} signals)")
    print("  shape:", fiducials.shape)

    fiducial_groups, demo_info, n_samples_by_group, ids_by_group = build_checker_inputs(
        fiducials, sampling_freq, n_samples
    )

    checker = QualityChecker(fiducial_groups, demo_info, n_samples_by_group, ids_by_group,
                              thresholds=CHECK_THRESHOLDS)

    print("Computing raw per-signal metrics (expensive step)...")
    checker.compute_raw_metrics("Full_set")

    print("Scoring + thresholding (cheap step)...")
    checker.score_and_report(
        "Full_set",
        w_consistency=SCORE_WEIGHTS["w_consistency"],
        w_alignment=SCORE_WEIGHTS["w_alignment"],
        thres_fiducials=DISCARD_THRESHOLDS["thres_fiducials"],
        thres_score=DISCARD_THRESHOLDS["thres_score"],
    )

    # --- Fiducial problem summary (pre-drop, descriptive only) ---
    problems = checker.problematic_fiducials_summary("Full_set")
    problems_path = report_output_dir / f"Problems_Fiducials_{subset_name}.xlsx"
    problems.to_frame().T.to_excel(problems_path, sheet_name="Full_set")
    print("Saved:", problems_path)

    # --- Full per-criterion percentage report (pre-drop) ---
    report_h5_path = data_output_dir / f"metrics_{subset_name}.h5"
    checker.save_report_h5(str(report_h5_path))
    print("Saved:", report_h5_path)

    cleaner = SignalCleaner(str(report_h5_path))
    remove = cleaner.detect()
    n_removed = len(remove["Full_set"])
    print(f"Flagged for removal: {n_removed} / {fiducials.shape[1]} signals "
          f"({100 * n_removed / fiducials.shape[1]:.2f}%)")

    csv_report_path = report_output_dir / f"criteria_report_{subset_name}.csv"
    cleaner.csv_report(str(csv_report_path))
    print("Saved:", csv_report_path)

    # --- Clean the feature set ---
    print("Loading features:", features_path)
    features = load_features(features_path)
    if max_signals is not None:
        features = {name: (values.iloc[:max_signals] if hasattr(values, "iloc") else values[:max_signals])
                    for name, values in features.items()}
    features_df = {"Full_set": {"PPG_Features": features["PPG_Features"]}}
    features_df["Full_set"]["PPG_Features"].index = np.arange(len(features_df["Full_set"]["PPG_Features"]))

    clean_features = cleaner.clean(features_df)["Full_set"]["PPG_Features"]

    clean_path = data_output_dir / f"Clean_Features_{subset_name}_{DISCARD_THRESHOLDS['thres_fiducials']}.h5"
    with h5py.File(clean_path, "w") as f:
        keep_mask = np.isin(np.arange(fiducials.shape[1]), remove["Full_set"], invert=True)
        for name, values in features.items():
            if name == "PPG_Features":
                continue
            f.create_dataset(name, data=values[keep_mask][np.newaxis, :])
        f.create_dataset("PPG_Features", data=clean_features.to_numpy(dtype=np.float64).T)
    print("Saved:", clean_path)

    # --- Demographic removal summary (post-drop, descriptive) ---
    subjects = np.array([s.decode() if isinstance(s, bytes) else s for s in features["Subject"]])
    demo_rows = []
    for subject_id in np.unique(subjects):
        subject_mask = subjects == subject_id
        total = int(subject_mask.sum())
        removed = int(subject_mask.sum() - (subject_mask & keep_mask).sum())
        idx = np.where(subject_mask)[0][0]
        demo_rows.append({
            "Subject": subject_id,
            "Age": features["Age"][idx],
            "Height": features["Height"][idx],
            "Weight": features["Weight"][idx],
            "Total signals": total,
            "Removed signals": removed,
            "Percentage removed (%)": 100 * removed / total,
        })
    demo_df = pd.DataFrame(demo_rows).set_index("Subject")
    demo_path = report_output_dir / f"Demographic_Info_{subset_name}.xlsx"
    demo_df.to_excel(demo_path)
    print("Saved:", demo_path)

    return {
        "checker": checker,
        "cleaner": cleaner,
        "keep_mask": keep_mask,
        "clean_path": clean_path,
    }


if __name__ == "__main__":
    for subset_name, paths in SUBSETS.items():
        if not paths["fiducials"].exists():
            print(f"Skipping {subset_name}: missing {paths['fiducials']}")
            continue
        if not paths["features"].exists():
            print(f"Skipping {subset_name}: missing {paths['features']}")
            continue
        run_subset(subset_name, paths["fiducials"], paths["features"],
                   DATA_OUTPUT_DIR, REPORT_OUTPUT_DIR)
