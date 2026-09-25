"""
QualityChecker: runs the per-signal quality metrics (metrics.py) over a
whole group of signals, and turns them into a pass/fail report.

Design note - why this is split into an expensive and a cheap stage
--------------------------------------------------------------------
`compute_raw_metrics()` is the expensive part: it loops over every signal
and does the fiducial-order checks, NA checks, heart-rate check, and the
window-vs-window alignment/consistency computation. NONE of that depends on
alpha/beta (the combined-score weights) or on the pass/fail thresholds.

`score_and_report()` is the cheap part: it only combines already-computed
alignment/consistency numbers with the chosen weights, and applies the
chosen thresholds. It does not touch the fiducial arrays again.

So a script that sweeps alpha/beta/thresholds should call
`compute_raw_metrics()` ONCE per group, then call `score_and_report()` in a
loop with different weights/thresholds - this is orders of magnitude
cheaper than re-running the whole pipeline per parameter combination.
"""

import h5py
import numpy as np
import pandas as pd

from metrics import (
    FIDUCIAL_ORDER,
    FIDUCIALS_BY_DERIVATIVE,
    DEFAULT_THRESHOLDS,
    SignalMetrics,
    windows_from_column,
    combine_scores,
)


class QualityChecker:
    def __init__(self, fiducials: dict, demo_info: dict, n_samples: dict, ids: dict,
                 thresholds: dict = None):
        """
        fiducials : dict[group -> DataFrame], each shaped
            (n_windows_per_signal * 16, n_signals_in_group). One column per
            signal, following FIDUCIAL_ORDER within every stacked window
            (see metrics.windows_from_column).
        demo_info : dict[group -> {"SamplingFrequency": Hz, ...}]
        n_samples : dict[group -> number of samples per signal]
        ids : dict[group -> list of signal ids, same order as the columns
            of `fiducials[group]`]
        thresholds : dict, see metrics.DEFAULT_THRESHOLDS. Only the
            checking thresholds (sp_limit, bmin, bmax) are used here;
            w_consistency/w_alignment/thresFiducials/thresScores are passed
            explicitly to score_and_report() instead, so they're easy to
            sweep.
        """
        self.fiducials = fiducials
        self.demo_info = demo_info
        self.n_samples = n_samples
        self.ids = ids
        self.thresholds = thresholds if thresholds else DEFAULT_THRESHOLDS
        self.raw_metrics = {}   # group -> dict, filled by compute_raw_metrics()
        self.df_results = {}    # group -> DataFrame, filled by score_and_report()

    def compute_raw_metrics(self, group: str) -> dict:
        """
        Runs every alpha/beta/threshold-independent check for every signal
        in `group`. Expensive - see module docstring. Caches its output in
        self.raw_metrics[group] and returns it.
        """
        fs = int(self.demo_info[group]["SamplingFrequency"])
        n_samples = self.n_samples[group]
        signal_ids = self.ids[group]
        # Index straight into the numpy array by position (not through a
        # pandas column lookup) - with hundreds of thousands of signals,
        # per-column label lookups on a DataFrame add up.
        fiducial_array = np.asarray(self.fiducials[group])

        hr_flag, peak_flag, na_count = {}, {}, {}
        pct_fiducials_detected = {}
        pct_by_derivative = {d: {} for d in FIDUCIALS_BY_DERIVATIVE}
        order_problem_ids = {fidu: [] for fidu in FIDUCIAL_ORDER}
        pct_problem_per_fiducial = {fidu: [] for fidu in FIDUCIAL_ORDER}
        alignment_by_id, consistency_by_id = {}, {}

        n_signals = len(signal_ids)
        progress_every = max(1, n_signals // 20)

        for i, signal_id in enumerate(signal_ids):
            if i % progress_every == 0:
                print(f"  compute_raw_metrics: {i}/{n_signals} signals", flush=True)
            windows = windows_from_column(fiducial_array[:, i])
            sm = SignalMetrics(windows, fs, n_samples, self.thresholds)

            na_count[signal_id] = sm.check_missing()
            peak_flag[signal_id] = sm.check_peak_count()
            hr_flag[signal_id] = sm.check_heart_rate()

            problems, n_flagged_points, _, pct_flagged = sm.check_order(signal_id)
            for fidu, pct in pct_flagged.items():
                pct_problem_per_fiducial[fidu].append(pct)
            for fidu, flagged_signals in problems.items():
                order_problem_ids[fidu].extend(flagged_signals)

            n_windows = windows.shape[0]
            n_fiducial_points = n_windows * len(FIDUCIAL_ORDER)
            pct_fiducials_detected[signal_id] = (1 - n_flagged_points / n_fiducial_points) * 100

            for deriv, deriv_fiducials in FIDUCIALS_BY_DERIVATIVE.items():
                n_deriv_points = n_windows * len(deriv_fiducials)
                n_deriv_flagged = sm.n_flagged_per_derivative[deriv]
                pct_by_derivative[deriv][signal_id] = (1 - n_deriv_flagged / n_deriv_points) * 100

            alignment, consistency = sm.consistency_alignment()
            alignment_by_id[signal_id] = alignment
            consistency_by_id[signal_id] = consistency

        raw = {
            "hr_flag": hr_flag,
            "peak_flag": peak_flag,
            "na_count": na_count,
            "pct_fiducials_detected": pct_fiducials_detected,
            "pct_by_derivative": pct_by_derivative,
            "order_problem_ids": order_problem_ids,
            "pct_problem_per_fiducial": pct_problem_per_fiducial,
            "alignment": alignment_by_id,
            "consistency": consistency_by_id,
        }
        self.raw_metrics[group] = raw
        return raw

    def score_and_report(self, group: str, w_consistency: float, w_alignment: float,
                          thres_fiducials: float, thres_score: float) -> pd.DataFrame:
        """
        Cheap stage: combines the cached alignment/consistency into a score
        using (w_consistency, w_alignment) = (alpha, beta), then applies
        (thres_fiducials, thres_score) plus the fixed HR/peak-count checks
        to produce a per-signal report.

        Requires compute_raw_metrics(group) to have been called first.

        Returns a DataFrame (index = signal id) with one column per
        criterion plus a final "report" column: 1 means "discard this
        signal", 0 means "keep it".
        """
        raw = self.raw_metrics[group]
        signal_ids = self.ids[group]

        combined_score = {}
        for signal_id in signal_ids:
            score_matrix = combine_scores(raw["alignment"][signal_id],
                                           raw["consistency"][signal_id],
                                           w_consistency, w_alignment)
            # NOTE: intentionally sum()/n_windows rather than .mean(axis=0).
            # A signal with NaN entries (e.g. a fiducial missing in some
            # windows) has those NaNs treated as 0 and averaged over the
            # FULL window count - not the count of non-NaN windows, which
            # is what pandas .mean() would do. This matches the original
            # checker.py/metrics_functions.py behaviour exactly; using
            # .mean() here silently gives NaN-heavy signals an inflated
            # score by averaging over fewer windows than the original did.
            n_windows = score_matrix.shape[0]
            mean_per_fiducial = score_matrix.sum(axis=0, skipna=True) / n_windows
            combined_score[signal_id] = mean_per_fiducial.sum(skipna=True) / len(mean_per_fiducial)

        df = pd.DataFrame(index=signal_ids)
        df["checkHR"] = [int(raw["hr_flag"][i]) for i in signal_ids]
        df["checkSP"] = [int(raw["peak_flag"][i]) for i in signal_ids]
        df["numberProperFiducials"] = [raw["pct_fiducials_detected"][i] for i in signal_ids]
        df["combinedScore"] = [combined_score[i] for i in signal_ids]
        for deriv, values in raw["pct_by_derivative"].items():
            df[deriv] = [values[i] for i in signal_ids]

        discard = (
            (df["checkHR"] == 1)
            | (df["checkSP"] == 1)
            | (df["numberProperFiducials"] < thres_fiducials)
            | (df["combinedScore"] < thres_score)
        )
        df["report"] = discard.astype(int)

        self.df_results[group] = df
        return df

    def problematic_fiducials_summary(self, group: str) -> pd.Series:
        """
        Mean percent of windows flagged as problematic, per fiducial point,
        across all signals in `group`. Uses cached raw metrics - no drop
        involved, purely descriptive.
        """
        raw = self.raw_metrics[group]
        return pd.Series(
            {fidu: np.mean(pcts) if pcts else np.nan
             for fidu, pcts in raw["pct_problem_per_fiducial"].items()},
            name="Ratio (%)",
        )

    def save_report_h5(self, filename: str):
        """
        Saves self.df_results to an HDF5 file, one group per key in
        self.df_results, matching the format cleaner.py expects: a
        "Metrics" dataset plus a "metrics" attribute listing column names.
        The signal id is stored as the last column ("ids"), which is how
        cleaner.py recovers it as the DataFrame index.
        """
        with h5py.File(filename, "w") as f:
            for group, df in self.df_results.items():
                out = df.copy()
                out["ids"] = list(out.index)
                grp = f.create_group(group)
                grp.create_dataset("Metrics", data=out.to_numpy(dtype=np.float64))
                grp.attrs["metrics"] = np.array(out.columns, dtype="S")
