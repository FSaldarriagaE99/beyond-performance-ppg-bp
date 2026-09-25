"""
Rebuilds Clean_Features_<subset>_<threshold>.h5 directly from a
sweep_thresholds.py dropped-signal JSON, without re-running the checker.

Use case: the pipeline's config changed (e.g. a threshold or a check
range) after a Clean_Features_*.h5 was already produced, so the file on
disk no longer reflects the current config. Rather than re-running the
whole per-signal pipeline (expensive), this reuses the already-computed
dropped_signals/threshold/thres_<value>.json (see sweep_thresholds.py /
README.txt for that schema) - which was built from the current
raw-metrics cache - to reconstruct exactly the same Clean_Features file
the live pipeline would produce, and overwrites it IN PLACE (same
filename, no renaming).

This does NOT recompute anything - if the JSON itself is stale (built
from an old cache), this will faithfully reproduce that staleness. Check
when the cache and sweep JSONs were last built vs. when run_pipeline.py's
config last changed before trusting the result.

Run from inside this folder:
    ..\..\.venv310\Scripts\python.exe rebuild_clean_features.py
"""

import json

import h5py
import numpy as np

from run_pipeline import SUBSETS, DATA_DIR
from sweep_common import load_subject_ids

THRESHOLDS_TO_REBUILD = [90, 80]  # 90 first (the important one), then 80

SWEEP_ROOT = (
    DATA_DIR.parent.parent  # .../9. Experiments
    / "2.LightGBM_SHAP" / "Abblation_Studies" / "Filtering_Sensitivity"
)


def load_features_raw(path) -> dict:
    """Loads every top-level dataset from Features_<subset>.h5 as-is (no transpose)."""
    with h5py.File(path, "r") as f:
        return {name: f[name][()] for name in f}


def keep_mask_from_dropped_json(payload: dict, subjects: np.ndarray) -> np.ndarray:
    """
    Inverts a sweep_thresholds.py JSON payload ("dropped": {subject_id:
    [local_index, ...]}) back into a global boolean keep_mask, using the
    same per-subject local-index enumeration sweep_common.py used to build
    it in the first place (so it's exact even for non-contiguous subjects).
    """
    n_signals = len(subjects)
    assert payload["n_total_signals"] == n_signals, (
        f"JSON was built for {payload['n_total_signals']} signals, "
        f"but the current features file has {n_signals} - stale JSON or wrong subset."
    )

    subject_local_index = np.zeros(n_signals, dtype=np.int64)
    for subject_id in np.unique(subjects):
        mask = subjects == subject_id
        subject_local_index[mask] = np.arange(mask.sum())

    discard = np.zeros(n_signals, dtype=bool)
    for subject_id, local_indices in payload["dropped"].items():
        subject_mask = subjects == subject_id
        target_locals = np.asarray(local_indices)
        discard |= subject_mask & np.isin(subject_local_index, target_locals)

    n_discarded = int(discard.sum())
    assert n_discarded == payload["n_dropped"], (
        f"Reconstructed {n_discarded} dropped signals, JSON says {payload['n_dropped']} - "
        f"local-index mapping mismatch."
    )
    return ~discard


def rebuild_subset_threshold(subset_name: str, threshold: int):
    features_path = SUBSETS[subset_name]["features"]
    json_path = SWEEP_ROOT / subset_name / "dropped_signals" / "threshold" / f"thres_{threshold}.json"
    out_path = DATA_DIR / f"Clean_Features_{subset_name}_{threshold}.h5"

    if not json_path.exists():
        print(f"Skipping {subset_name} @ {threshold}: no JSON at {json_path}")
        return

    print(f"\n=== {subset_name} @ threshold {threshold} ===")
    with open(json_path) as f:
        payload = json.load(f)
    print("  JSON parameters:", payload["parameters"])

    subjects = load_subject_ids(features_path)
    keep_mask = keep_mask_from_dropped_json(payload, subjects)

    features = load_features_raw(features_path)

    old_n = None
    if out_path.exists():
        with h5py.File(out_path, "r") as f:
            old_n = f["PPG_Features"].shape[1]

    with h5py.File(out_path, "w") as f:
        for name, values in features.items():
            if name == "PPG_Features":
                continue
            # values shape is (1, n_signals); keep_mask applies to axis 1.
            f.create_dataset(name, data=values[:, keep_mask])
        ppg_features = features["PPG_Features"][:, keep_mask]  # (28, n_kept)
        f.create_dataset("PPG_Features", data=ppg_features.astype(np.float64))

    new_n = int(keep_mask.sum())
    print(f"  Overwrote: {out_path}")
    print(f"  Signals kept: {old_n if old_n is not None else '(new file)'} -> {new_n} "
          f"(out of {len(subjects)} total)")


if __name__ == "__main__":
    for threshold in THRESHOLDS_TO_REBUILD:
        for subset_name in SUBSETS:
            rebuild_subset_threshold(subset_name, threshold)
