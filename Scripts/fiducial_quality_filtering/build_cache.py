"""
Runs the expensive per-signal metrics loop once per subset and saves the
result as a raw-metrics cache (.npz) next to the subset's source data.

This only needs to be re-run if the fiducial data itself changes (e.g. a
subset is re-extracted). sweep_alpha_beta.py and sweep_thresholds.py both
just load these caches instead of recomputing.

Run from inside this folder:
    ..\..\.venv310\Scripts\python.exe build_cache.py
"""

from pathlib import Path

import h5py

from raw_metrics_cache import compute_raw_metrics_cache, save_cache
from run_pipeline import DATA_DIR, SUBSETS, CHECK_THRESHOLDS

CACHE_DIR = DATA_DIR  # "the supplementary datasets location"


def cache_path_for(subset_name: str) -> Path:
    return CACHE_DIR / f"RawMetricsCache_{subset_name}.npz"


def build_subset_cache(subset_name: str, fiducials_path: Path,
                        sampling_freq: float = 125, n_samples: int = 1250):
    out_path = cache_path_for(subset_name)
    print(f"\n=== {subset_name} ===")
    print("Loading fiducials:", fiducials_path)
    with h5py.File(fiducials_path, "r") as f:
        fiducials = f["PPG_fiducial_points"]["Fiducials"][()]
    print("  shape:", fiducials.shape)

    cache = compute_raw_metrics_cache(fiducials, fs=sampling_freq, n_samples=n_samples,
                                       thresholds=CHECK_THRESHOLDS)
    save_cache(cache, out_path)
    return out_path


if __name__ == "__main__":
    for subset_name, paths in SUBSETS.items():
        if not paths["fiducials"].exists():
            print(f"Skipping {subset_name}: missing {paths['fiducials']}")
            continue
        build_subset_cache(subset_name, paths["fiducials"])
