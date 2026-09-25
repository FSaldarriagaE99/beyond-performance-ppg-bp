"""
First-pass sensitivity sweep over the discard thresholds. thres_fiducials
and thres_score are moved TOGETHER (thres_fiducials = thres_score at every
grid point) - this is a 1D sweep, NOT a 2D grid over the two thresholds
independently. Criterion-4 weights are held fixed at the pipeline defaults
(w_consistency=0.25, w_alignment=0.75) - see sweep_alpha_beta.py for that
sweep.

Same reporting + output shape as sweep_alpha_beta.py - see that script's
docstring and README.txt for details.

Requires build_cache.py to have been run first.

Run from inside this folder:
    ..\..\.venv310\Scripts\python.exe sweep_thresholds.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from run_pipeline import SUBSETS
from build_cache import cache_path_for
from raw_metrics_cache import load_cache, vectorized_report
from sweep_common import load_subject_ids, summarize_drop, dropped_signals_payload, save_json
from local_paths import FILTERING_SENSITIVITY_DIR

OUTPUT_ROOT = FILTERING_SENSITIVITY_DIR

THRESHOLD_GRID = np.arange(70, 96, 5)  # 70, 75, 80, 85, 90, 95
FIXED_WEIGHTS = {"w_consistency": 0.25, "w_alignment": 0.75}


def run_subset(subset_name: str, features_path: Path, output_dir: Path):
    cache_file = cache_path_for(subset_name)
    if not cache_file.exists():
        print(f"Skipping {subset_name}: no cache at {cache_file} (run build_cache.py first)")
        return

    print(f"\n=== {subset_name} ===")
    cache = load_cache(cache_file)
    subjects = load_subject_ids(features_path)

    json_dir = output_dir / "dropped_signals" / "threshold"
    rows = []
    for thres in THRESHOLD_GRID:
        thres = int(thres)
        discard = vectorized_report(cache, w_consistency=FIXED_WEIGHTS["w_consistency"],
                                     w_alignment=FIXED_WEIGHTS["w_alignment"],
                                     thres_fiducials=thres, thres_score=thres)
        stats = summarize_drop(discard, subjects)
        stats_row = {"thres_fiducials": thres, "thres_score": thres, **FIXED_WEIGHTS, **stats}
        rows.append(stats_row)
        print(f"  thres={thres}: "
              f"{stats['n_dropped_signals']}/{stats['n_total_signals']} signals dropped "
              f"({stats['pct_dropped_signals']:.2f}%), "
              f"{stats['n_subjects_any_dropped']}/{stats['n_subjects_total']} subjects touched "
              f"({stats['pct_subjects_any_dropped']:.2f}%), "
              f"{stats['n_subjects_fully_dropped']} subjects fully dropped "
              f"({stats['pct_subjects_fully_dropped']:.2f}%)")

        payload = dropped_signals_payload(
            subset_name,
            {"thres_fiducials": thres, "thres_score": thres, **FIXED_WEIGHTS},
            discard, subjects,
        )
        save_json(payload, json_dir / f"thres_{thres}.json")

    summary = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "threshold_sweep_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("Saved:", summary_path)


if __name__ == "__main__":
    for subset_name, paths in SUBSETS.items():
        if not paths["features"].exists():
            print(f"Skipping {subset_name}: missing {paths['features']}")
            continue
        run_subset(subset_name, paths["features"], OUTPUT_ROOT / subset_name)
