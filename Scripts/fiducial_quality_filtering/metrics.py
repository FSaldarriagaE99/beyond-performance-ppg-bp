"""
Per-signal quality metrics computed from a set of pyPPG fiducial points.

This module does NOT know about pyPPG, patients, or HDF5 files. It only
consumes the fiducial-point indices already extracted for ONE signal (one
PPG beat/segment) and computes a handful of quality-check numbers used to
decide whether that signal should be kept or discarded.

Terminology
-----------
- "Signal": one PPG segment (a fixed-length window containing several beats).
- "Window": one beat inside a signal. Every window has (up to) 16 fiducial
  points, one per column in FIDUCIAL_ORDER below.
- "Fiducial point": a sample index marking a specific landmark in the PPG
  waveform or its 1st/2nd/3rd derivative (e.g. "sp" = systolic peak).

Criteria implemented (matches the 4 checks used downstream in checker.py):
  1. checkHR            -> mean heart rate within a plausible range
  2. checkNumPeaks       -> enough systolic peaks detected for the signal length
  3. checkOrder / NA     -> fiducials present and in the physiologically
                             expected order within each window
  4. combined score       -> weighted blend of "alignment" (how close each
                             fiducial-to-fiducial interval is to the peak
                             period) and "consistency" (how close it is to
                             its own average across windows). The weights
                             (alpha=w_consistency, beta=w_alignment) and the
                             pass/fail threshold are NOT baked into this
                             class - see `combine_scores()` below - so they
                             can be swept cheaply without recomputing
                             anything else in this file.
"""

import numpy as np
import pandas as pd

# Order of the 16 fiducial points pyPPG produces per beat/window, grouped by
# the signal they come from: PPG itself, and its 1st/2nd/3rd derivatives.
FIDUCIAL_ORDER = ["on", "sp", "dn", "dp", "off",  # PPG
                   "u", "v", "w",                  # 1st derivative
                   "a", "b", "c", "d", "e", "f",   # 2nd derivative
                   "p1", "p2"]                      # 3rd derivative

FIDUCIALS_BY_DERIVATIVE = {
    "ppg": ["on", "sp", "dn", "dp", "off"],
    "d1": ["u", "v", "w"],
    "d2": ["a", "b", "c", "d", "e", "f"],
    "d3": ["p1", "p2"],
}

DEFAULT_THRESHOLDS = {
    "sp_limit": 2,       # max allowed shortfall in detected systolic peaks
    "bmin": 50,          # min plausible heart rate (bpm)
    "bmax": 180,         # max plausible heart rate (bpm)
    "w_consistency": 0.25,  # alpha
    "w_alignment": 0.75,    # beta
}


def windows_from_column(fiducial_column: np.ndarray) -> pd.DataFrame:
    """
    Reshape one signal's flat fiducial array into one row per window
    (beat), one column per fiducial point.

    The raw HDF5 storage is flat: 16 fiducial indices per window,
    concatenated window after window, for a single signal (column).

    Rows that are entirely NaN (padding beyond the last real window) are
    dropped.
    """
    n_fiducials = len(FIDUCIAL_ORDER)
    n_windows = len(fiducial_column) // n_fiducials
    trimmed = fiducial_column[: n_windows * n_fiducials]
    as_2d = trimmed.reshape((n_windows, n_fiducials))
    return pd.DataFrame(as_2d, columns=FIDUCIAL_ORDER).dropna(how="all")


class SignalMetrics:
    """
    Computes quality-check numbers for a single signal, given its fiducial
    points reshaped into windows (see `windows_from_column`).

    Everything computed here is INDEPENDENT of alpha/beta and of the
    pass/fail thresholds - those are only applied later, in
    `combine_scores()` and in checker.py's reporting step. That split is
    what makes an alpha/beta/threshold sweep cheap: this class only needs
    to run once per signal.
    """

    def __init__(self, fiducial_windows: pd.DataFrame, fs: int, n_samples: int,
                 thresholds: dict = None):
        """
        fiducial_windows : DataFrame, shape (n_windows, 16)
            One row per beat, columns = FIDUCIAL_ORDER, values = sample index
            of that fiducial point within the signal (NaN if not found).
        fs : sampling frequency in Hz.
        n_samples : number of samples in the signal (its length).
        thresholds : dict, see DEFAULT_THRESHOLDS. Only the non-scoring
            keys (sp_limit, bmin, bmax) are used by this class.
        """
        self.fiducials = fiducial_windows
        self.sp = fiducial_windows["sp"]
        self.fs = fs
        self.n_samples = n_samples
        self.thresholds = thresholds if thresholds else DEFAULT_THRESHOLDS
        self.signal_duration_s = self.n_samples / self.fs

        self.inter_peak_interval_s, self.inter_peak_rate_bpm = self._peak_rate()
        self._fiducial_time_diffs()

        # Filled in by consistency_alignment(); kept as attributes so
        # combine_scores() can be called later without recomputation.
        self.alignment = None
        self.consistency = None

    def _peak_rate(self):
        """Instantaneous heart rate (bpm) from consecutive systolic peaks."""
        peak_period_s = np.round(np.diff(self.sp / self.fs), 6)
        rate_bpm = 60 / peak_period_s
        return peak_period_s, rate_bpm

    def _fiducial_time_diffs(self):
        """
        Time (seconds) of each fiducial point, the window-to-window
        difference of those times, and their per-fiducial mean - used by
        consistency_alignment().
        """
        fiducial_times = self.fiducials / self.fs
        time_diffs = fiducial_times.diff().iloc[1:]
        self.fiducial_times = fiducial_times
        self.fiducial_time_diffs = time_diffs
        self.mean_fiducial_time_diff = time_diffs.mean(axis=0)

    def check_missing(self) -> int:
        """Total count of NaN fiducial values across all windows."""
        return int(self.fiducials.isna().sum().sum())

    def check_peak_count(self) -> bool:
        """
        True (flag) if far fewer systolic peaks were detected than the
        signal duration implies.
        """
        limit = self.thresholds.get("sp_limit", DEFAULT_THRESHOLDS["sp_limit"])
        detected_peaks = len(self.sp)
        expected_peaks = np.mean((self.inter_peak_rate_bpm / 60) * self.signal_duration_s)
        expected_peaks = round(expected_peaks) if expected_peaks > 0 else expected_peaks
        return detected_peaks < expected_peaks - limit

    def check_heart_rate(self) -> bool:
        """True (flag) if mean heart rate falls outside [bmin, bmax]."""
        bmin = self.thresholds.get("bmin", DEFAULT_THRESHOLDS["bmin"])
        bmax = self.thresholds.get("bmax", DEFAULT_THRESHOLDS["bmax"])
        mean_hr = np.mean(self.inter_peak_rate_bpm)
        return not (bmin < mean_hr < bmax)

    def check_order(self, signal_id):
        """
        Checks that fiducials within each derivative group appear in the
        physiologically expected order (e.g. "on" before "sp" before "dn"
        ... within the PPG group), and that none are missing.

        Returns
        -------
        signals_with_problem : dict[fiducial_name -> [signal_id, ...]]
            Which fiducial columns had at least one problem in this signal.
        n_flagged_points : int
            Total count of individual fiducial values (across all windows)
            that were missing or out of order.
        n_flagged_per_window : dict[window_index -> count]
            How many fiducial problems occurred in each window.
        pct_flagged_per_fiducial : dict[fiducial_name -> percent]
            Percent of windows where that specific fiducial was flagged.
        """
        n_flagged_points = 0
        flagged_indices_by_fiducial = {}
        signals_with_problem = {fidu: [] for fidu in FIDUCIAL_ORDER}
        n_flagged_per_derivative = {}

        for group_name, group_fiducials in FIDUCIALS_BY_DERIVATIVE.items():
            group_flag_count = 0
            for i, fidu in enumerate(group_fiducials):
                current = self.fiducials[fidu]
                previous = self.fiducials[group_fiducials[i - 1]] if i > 0 else current
                nxt = self.fiducials[group_fiducials[i + 1]] if i < len(group_fiducials) - 1 else current

                if current.isna().any():
                    bad_idx = np.where(current.isna())[0]
                    n_flagged_points += len(bad_idx)
                    flagged_indices_by_fiducial[fidu] = bad_idx
                    signals_with_problem[fidu].append(signal_id)
                    continue

                # Drop windows where the neighbour needed for the order
                # check is itself missing, so the comparison is well defined.
                if previous.isna().any():
                    drop_idx = previous[previous.isna()].index
                    previous, current, nxt = previous.dropna(), current.drop(drop_idx), nxt.drop(drop_idx)
                if nxt.isna().any():
                    drop_idx = nxt[nxt.isna()].index
                    nxt, previous, current = nxt.dropna(), previous.drop(drop_idx), current.drop(drop_idx)

                if i == 0:
                    in_order = current < nxt
                elif i == len(group_fiducials) - 1:
                    in_order = previous < current
                else:
                    in_order = (previous < current) & (current < nxt)

                bad_idx = np.where(~in_order)[0]
                flagged_indices_by_fiducial[fidu] = bad_idx
                if len(bad_idx) > 0:
                    group_flag_count += len(bad_idx)
                    n_flagged_points += len(bad_idx)
                    signals_with_problem[fidu].append(signal_id)

            n_flagged_per_derivative[group_name] = group_flag_count

        n_windows = self.fiducials.shape[0]
        n_flagged_per_window = {
            f"win{w}": sum((flagged_indices_by_fiducial[f] == w).any() for f in flagged_indices_by_fiducial)
            for w in range(n_windows)
        }
        pct_flagged_per_fiducial = {
            f: (len(idx) / n_windows) * 100 for f, idx in flagged_indices_by_fiducial.items()
        }

        self.n_flagged_per_derivative = n_flagged_per_derivative
        return signals_with_problem, n_flagged_points, n_flagged_per_window, pct_flagged_per_fiducial

    def consistency_alignment(self):
        """
        For every fiducial-to-fiducial time interval:
          - "alignment": how close it is (in %) to that window's systolic
            peak-to-peak period. Penalises intervals that are inconsistent
            with the beat's own rhythm.
          - "consistency": how close it is (in %) to the fiducial's own
            average interval across all windows of this signal. Penalises
            fiducials that jump around window to window.

        Returns (alignment, consistency), both DataFrames shaped like
        fiducial_time_diffs. Also stored as self.alignment / self.consistency
        so combine_scores() can reuse them without recomputation.
        """
        time_diffs = self.fiducial_time_diffs

        alignment_error = abs(time_diffs.sub(self.inter_peak_interval_s, axis=0))
        alignment_error = alignment_error.div(self.inter_peak_interval_s, axis=0)
        self.alignment = (1 - alignment_error) * 100

        consistency_error = abs(time_diffs.sub(self.mean_fiducial_time_diff, axis=1))
        consistency_error = consistency_error.div(self.mean_fiducial_time_diff, axis=1)
        self.consistency = (1 - consistency_error) * 100

        return self.alignment, self.consistency


def combine_scores(alignment: pd.DataFrame, consistency: pd.DataFrame,
                    w_consistency: float, w_alignment: float) -> pd.DataFrame:
    """
    Criterion 4: weighted blend of alignment and consistency into a single
    per-window, per-fiducial score.

    This is deliberately a free function (not a method that recomputes
    alignment/consistency): alignment and consistency don't depend on the
    weights, so when sweeping alpha (w_consistency) / beta (w_alignment)
    you only need to re-run this cheap weighted sum, not the full
    SignalMetrics pipeline.
    """
    return w_alignment * alignment + w_consistency * consistency
