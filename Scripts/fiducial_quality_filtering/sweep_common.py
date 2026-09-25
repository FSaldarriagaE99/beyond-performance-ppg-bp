"""
Shared helpers for sweep_alpha_beta.py and sweep_thresholds.py: turning a
per-signal discard mask into the summary stats and the reusable dropped-
signal JSON described in README.txt.

Both sweep scripts load a raw_metrics_cache.py cache once per subset and
then just call summarize_drop() / dropped_signals_payload() per grid
point - all vectorized numpy, no per-signal Python loop.
"""

import json

import h5py
import numpy as np


def load_subject_ids(features_path) -> np.ndarray:
    """
    Reads only the "Subject" dataset from a Features_<subset>.h5 file,
    decoded to plain strings, in the same signal order as the fiducials
    file (they're joined by column position - see run_pipeline.py).
    """
    with h5py.File(features_path, "r") as f:
        raw = f["Subject"][0]
    return np.array([s.decode() if isinstance(s, bytes) else s for s in raw])


def summarize_drop(discard_mask: np.ndarray, subjects: np.ndarray) -> dict:
    """
    discard_mask : bool array, one per signal, True = dropped.
    subjects : str array, same length, subject id per signal.

    Returns raw counts + percentages for samples and for both subject-drop
    definitions ("subject has >=1 dropped sample" and "subject entirely
    dropped").
    """
    n_total_signals = len(discard_mask)
    n_dropped_signals = int(discard_mask.sum())

    unique_subjects, subject_index = np.unique(subjects, return_inverse=True)
    n_subjects_total = len(unique_subjects)

    total_per_subject = np.bincount(subject_index, minlength=n_subjects_total)
    dropped_per_subject = np.bincount(subject_index[discard_mask], minlength=n_subjects_total)

    n_subjects_any_dropped = int((dropped_per_subject > 0).sum())
    n_subjects_fully_dropped = int((dropped_per_subject == total_per_subject).sum())

    return {
        "n_total_signals": n_total_signals,
        "n_dropped_signals": n_dropped_signals,
        "pct_dropped_signals": 100 * n_dropped_signals / n_total_signals,
        "n_subjects_total": n_subjects_total,
        "n_subjects_any_dropped": n_subjects_any_dropped,
        "pct_subjects_any_dropped": 100 * n_subjects_any_dropped / n_subjects_total,
        "n_subjects_fully_dropped": n_subjects_fully_dropped,
        "pct_subjects_fully_dropped": 100 * n_subjects_fully_dropped / n_subjects_total,
    }


def dropped_signals_payload(subset_name: str, parameters: dict, discard_mask: np.ndarray,
                             subjects: np.ndarray) -> dict:
    """
    Builds the JSON payload (see README.txt for the schema): subject id ->
    list of dropped sample indices (positional index within that subject's
    signals, in the order they appear in Features_<subset>.h5) - directly
    reusable as a keep/drop mask when loading the full dataframe later.
    """
    dropped = {}
    # Position of each signal within its own subject's block of signals,
    # e.g. subject "p000003"'s 5th signal overall gets local index 4.
    subject_local_index = np.zeros(len(subjects), dtype=np.int64)
    for subject_id in np.unique(subjects):
        mask = subjects == subject_id
        subject_local_index[mask] = np.arange(mask.sum())

    dropped_idx = np.where(discard_mask)[0]
    for i in dropped_idx:
        subject_id = subjects[i]
        dropped.setdefault(subject_id, []).append(int(subject_local_index[i]))

    return {
        "subset": subset_name,
        "parameters": parameters,
        "n_total_signals": int(len(discard_mask)),
        "n_dropped": int(discard_mask.sum()),
        "dropped": dropped,
    }


def save_json(payload: dict, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)
