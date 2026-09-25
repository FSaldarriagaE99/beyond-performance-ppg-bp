"""
Cacheable, vectorization-friendly version of the expensive per-signal
metrics loop.

checker.QualityChecker.compute_raw_metrics() (see checker.py) is correct,
but it stores its per-signal output as pandas DataFrames in Python dicts.
That's fine for a single run, but two things make it a poor fit for
sweeping alpha/beta/thresholds across many parameter values:

  1. It can't be saved to disk cheaply (465k small DataFrames pickle
     slowly and bloat on disk).
  2. Re-scoring for a new alpha/beta still means a Python loop over every
     signal (score_and_report() loops per signal_id).

This module computes the SAME numbers (using the exact same metrics.py
functions checker.py uses - no logic is duplicated/reimplemented), but
into plain numpy arrays that can be:
  - saved/loaded as a single .npz file (compute once, reuse forever), and
  - scored for ANY alpha/beta/thresholds combination with a handful of
    vectorized numpy ops over ALL signals at once (see
    vectorized_combined_score / vectorized_report), instead of a
    465,480-iteration Python loop per sweep point.

Padding note
------------
Different subsets have different fixed window counts (Train's fiducial
array has 336 rows = 21 windows before any dropna; the CalFree_Test
subset has 320 rows = 20 windows). alignment/consistency per signal are
therefore padded to `max_windows` (established from data.shape[0]) with
NaN in unused slots. This is safe: np.nansum treats NaN as 0, exactly
matching the skipna=True sum() behaviour in metrics.py/checker.py, and
because rows are copied in their original (dropna-filtered) order, WHICH
window ends up in which padded slot doesn't matter - only the sum over
real values and the true per-signal window count (stored separately in
n_windows_for_scoring) matter for the final score. See checker.py's
score_and_report() docstring for why sum()/n rather than mean() is used.
"""

import numpy as np

from metrics import (
    FIDUCIAL_ORDER,
    FIDUCIALS_BY_DERIVATIVE,
    DEFAULT_THRESHOLDS,
    SignalMetrics,
    windows_from_column,
)

N_FIDUCIALS = len(FIDUCIAL_ORDER)


def compute_raw_metrics_cache(fiducials: np.ndarray, fs: float, n_samples: int,
                               thresholds: dict = None, progress_every: int = None) -> dict:
    """
    Runs the expensive per-signal loop once and returns a dict of plain
    numpy arrays (see RawMetricsCache fields below). This is the ONLY
    expensive step - everything downstream (any alpha/beta/threshold
    combination) is vectorized numpy on the returned arrays.
    """
    n_signals = fiducials.shape[1]
    max_windows = fiducials.shape[0] // N_FIDUCIALS - 1  # -1: consistency_alignment uses a .diff()
    thresholds = thresholds if thresholds else DEFAULT_THRESHOLDS
    progress_every = progress_every or max(1, n_signals // 20)

    alignment = np.full((n_signals, max_windows, N_FIDUCIALS), np.nan, dtype=np.float32)
    consistency = np.full((n_signals, max_windows, N_FIDUCIALS), np.nan, dtype=np.float32)
    n_windows_for_scoring = np.zeros(n_signals, dtype=np.int32)
    hr_flag = np.zeros(n_signals, dtype=bool)
    peak_flag = np.zeros(n_signals, dtype=bool)
    pct_fiducials_detected = np.zeros(n_signals, dtype=np.float32)
    pct_by_derivative = {d: np.zeros(n_signals, dtype=np.float32) for d in FIDUCIALS_BY_DERIVATIVE}
    pct_problem_per_fiducial = np.full((n_signals, N_FIDUCIALS), np.nan, dtype=np.float32)

    for i in range(n_signals):
        if i % progress_every == 0:
            print(f"  compute_raw_metrics_cache: {i}/{n_signals} signals", flush=True)

        windows = windows_from_column(fiducials[:, i])
        sm = SignalMetrics(windows, fs, n_samples, thresholds)

        hr_flag[i] = sm.check_heart_rate()
        peak_flag[i] = sm.check_peak_count()

        _, n_flagged_points, _, pct_flagged = sm.check_order(i)
        for j, fidu in enumerate(FIDUCIAL_ORDER):
            if fidu in pct_flagged:
                pct_problem_per_fiducial[i, j] = pct_flagged[fidu]

        n_windows = windows.shape[0]
        n_fiducial_points = n_windows * N_FIDUCIALS
        pct_fiducials_detected[i] = (1 - n_flagged_points / n_fiducial_points) * 100

        for deriv, deriv_fiducials in FIDUCIALS_BY_DERIVATIVE.items():
            n_deriv_points = n_windows * len(deriv_fiducials)
            pct_by_derivative[deriv][i] = (1 - sm.n_flagged_per_derivative[deriv] / n_deriv_points) * 100

        align_df, consist_df = sm.consistency_alignment()
        rows = align_df.shape[0]
        n_windows_for_scoring[i] = rows
        # rows can't exceed max_windows: max_windows was derived from the
        # same fixed fiducial-array shape every signal in this subset uses.
        alignment[i, :rows, :] = align_df.to_numpy(dtype=np.float32)
        consistency[i, :rows, :] = consist_df.to_numpy(dtype=np.float32)

    return {
        "alignment": alignment,
        "consistency": consistency,
        "n_windows_for_scoring": n_windows_for_scoring,
        "hr_flag": hr_flag,
        "peak_flag": peak_flag,
        "pct_fiducials_detected": pct_fiducials_detected,
        "pct_by_derivative_ppg": pct_by_derivative["ppg"],
        "pct_by_derivative_d1": pct_by_derivative["d1"],
        "pct_by_derivative_d2": pct_by_derivative["d2"],
        "pct_by_derivative_d3": pct_by_derivative["d3"],
        "pct_problem_per_fiducial": pct_problem_per_fiducial,
        "fiducial_order": np.array(FIDUCIAL_ORDER),
    }


def save_cache(cache: dict, path):
    """Saves a compute_raw_metrics_cache() result to a single .npz file."""
    np.savez(path, **cache)
    print("Saved raw metrics cache:", path)


def load_cache(path) -> dict:
    """Loads a cache saved by save_cache(). Returns the same dict shape."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def vectorized_combined_score(cache: dict, w_consistency: float, w_alignment: float) -> np.ndarray:
    """
    Criterion 4 for every signal at once. Exactly reproduces
    checker.QualityChecker.score_and_report()'s combined-score formula
    (sum-over-total-window-count, not pandas .mean() - see that function's
    docstring for why), just vectorized across all signals.
    """
    score = w_alignment * cache["alignment"] + w_consistency * cache["consistency"]  # (n_signals, max_windows, 16)
    sum_per_fiducial = np.nansum(score, axis=1)  # (n_signals, 16); NaN (real or padding) contributes 0
    mean_per_fiducial = sum_per_fiducial / cache["n_windows_for_scoring"][:, None]
    return np.nansum(mean_per_fiducial, axis=1) / mean_per_fiducial.shape[1]


def vectorized_report(cache: dict, w_consistency: float, w_alignment: float,
                       thres_fiducials: float, thres_score: float) -> np.ndarray:
    """
    Boolean array, one per signal: True means "discard" - reproduces
    checker.QualityChecker.score_and_report()'s `report` column exactly.
    """
    combined_score = vectorized_combined_score(cache, w_consistency, w_alignment)
    return (
        cache["hr_flag"]
        | cache["peak_flag"]
        | (cache["pct_fiducials_detected"] < thres_fiducials)
        | (combined_score < thres_score)
    )
