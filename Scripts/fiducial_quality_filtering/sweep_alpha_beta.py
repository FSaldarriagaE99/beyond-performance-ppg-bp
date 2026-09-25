"""
First-pass sensitivity sweep over alpha (w_consistency) / beta
(w_alignment), the criterion-4 weights. beta = 1 - alpha throughout, so
this is a 1D sweep, NOT a full alpha x beta grid. Thresholds are held
fixed at the pipeline defaults (thres_fiducials=90, thres_score=90) - see
sweep_thresholds.py for the threshold sweep.

For each subset and each alpha value, reports:
  - samples (signals) dropped: raw count + % of the full subset
  - subjects with >=1 dropped sample: raw count + % of all subjects
  - subjects entirely dropped (100% of their samples gone): raw count + %

...and saves the exact set of dropped signals as JSON (subject -> list of
dropped local sample indices), so this can be reused later to filter the
full dataframe live at training time without re-running any of this.

Requires build_cache.py to have been run first (see README.txt) - this
script only does the cheap, vectorized scoring step.

Run from inside this folder:
    ..\..\.venv310\Scripts\python.exe sweep_alpha_beta.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from run_pipeline import SUBSETS
from build_cache import cache_path_for
from raw_metrics_cache import load_cache, vectorized_report
from sweep_common import load_subject_ids, summarize_drop, dropped_signals_payload, save_json
from local_paths import FILTERING_SENSITIVITY_DIR

# Where results are written: one subfolder per subset.
OUTPUT_ROOT = FILTERING_SENSITIVITY_DIR

ALPHA_GRID = np.round(np.arange(0.0, 1.0001, 0.1), 2)  # beta = 1 - alpha
ALPHA_GRID = np.unique(np.concatenate([ALPHA_GRID, [0.25, 0.75]]))  # + the pipeline's own 25/75 and 75/25 split
FIXED_THRESHOLDS = {"thres_fiducials": 90, "thres_score": 90}


def run_subset(subset_name: str, features_path: Path, output_dir: Path):
    cache_file = cache_path_for(subset_name)
    if not cache_file.exists():
        print(f"Skipping {subset_name}: no cache at {cache_file} (run build_cache.py first)")
        return

    print(f"\n=== {subset_name} ===")
    cache = load_cache(cache_file)
    subjects = load_subject_ids(features_path)

    json_dir = output_dir / "dropped_signals" / "alpha_beta"
    rows = []
    for alpha in ALPHA_GRID:
        beta = round(1 - alpha, 2)
        discard = vectorized_report(cache, w_consistency=alpha, w_alignment=beta,
                                     thres_fiducials=FIXED_THRESHOLDS["thres_fiducials"],
                                     thres_score=FIXED_THRESHOLDS["thres_score"])
        stats = summarize_drop(discard, subjects)
        stats_row = {"w_consistency_alpha": alpha, "w_alignment_beta": beta, **FIXED_THRESHOLDS, **stats}
        rows.append(stats_row)
        print(f"  alpha={alpha:.2f} beta={beta:.2f}: "
              f"{stats['n_dropped_signals']}/{stats['n_total_signals']} signals dropped "
              f"({stats['pct_dropped_signals']:.2f}%), "
              f"{stats['n_subjects_any_dropped']}/{stats['n_subjects_total']} subjects touched "
              f"({stats['pct_subjects_any_dropped']:.2f}%), "
              f"{stats['n_subjects_fully_dropped']} subjects fully dropped "
              f"({stats['pct_subjects_fully_dropped']:.2f}%)")

        payload = dropped_signals_payload(
            subset_name,
            {"w_consistency": alpha, "w_alignment": beta, **FIXED_THRESHOLDS},
            discard, subjects,
        )
        save_json(payload, json_dir / f"alpha_{alpha:.2f}.json")

    summary = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "alpha_beta_sweep_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("Saved:", summary_path)


if __name__ == "__main__":
    for subset_name, paths in SUBSETS.items():
        if not paths["features"].exists():
            print(f"Skipping {subset_name}: missing {paths['features']}")
            continue
        run_subset(subset_name, paths["features"], OUTPUT_ROOT / subset_name)
