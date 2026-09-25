"""
SignalCleaner: reads a quality report produced by checker.QualityChecker,
figures out which signal ids to discard, and drops them from any dataset
that shares those ids.

This only supports the in-memory "data_ext" workflow (a dict of pandas
DataFrames already loaded from HDF5), which is what run_pipeline.py uses.
"""

import numpy as np
import pandas as pd
import h5py


class SignalCleaner:
    def __init__(self, report_h5_path: str):
        """
        Reads the quality report saved by QualityChecker.save_report_h5().

        report_h5_path : path to the metrics/report .h5 file, one HDF5
            group per signal group, each containing a "Metrics" dataset
            (rows = signals, last column = signal id) and a "metrics"
            attribute listing the column names.
        """
        self.path = report_h5_path
        self.data = {}   # group -> DataFrame of the report, indexed by signal id
        self.ids = {}     # group -> array of signal ids (same as the index above)
        self.remove = {}  # filled by detect()

        with h5py.File(report_h5_path, "r") as f:
            for group in f.keys():
                grp = f[group]
                values = grp["Metrics"][:]
                columns = [c.decode() for c in grp.attrs["metrics"]]
                signal_ids = values[:, -1]
                self.ids[group] = signal_ids
                self.data[group] = pd.DataFrame(values, columns=columns, index=signal_ids)

    def detect(self) -> dict:
        """
        Returns (and caches) dict[group -> list of signal ids flagged for
        removal], i.e. where the report column == 1.
        """
        self.remove = {
            group: df.index[df["report"].eq(1)].tolist()
            for group, df in self.data.items()
        }
        return self.remove

    def clean(self, data_ext: dict) -> dict:
        """
        Drops the flagged signal ids from every DataFrame in data_ext whose
        top-level key matches a group in self.remove (call detect() first).

        data_ext : dict[group -> DataFrame] or dict[group -> dict[dataset_name -> DataFrame]],
            all indexed by signal id, matching the report's ids.

        Returns a new dict with the same shape as data_ext, signals removed.
        """
        if not self.remove:
            raise RuntimeError("Call detect() before clean().")

        cleaned = {}
        for group, content in data_ext.items():
            ids_to_remove = self.remove.get(group)
            if ids_to_remove is None:
                # Not one of the groups the checker produced a report for
                # (e.g. a side dataset like raw Subject/Age arrays) - leave
                # untouched here; use clean_dataset() explicitly for these.
                cleaned[group] = content
                continue
            if isinstance(content, dict):
                cleaned[group] = {name: self.clean_dataset(df, ids_to_remove)
                                   for name, df in content.items()}
            else:
                cleaned[group] = self.clean_dataset(content, ids_to_remove)
        self.clean_data = cleaned
        return cleaned

    @staticmethod
    def clean_dataset(dataset: pd.DataFrame, ids_to_remove) -> pd.DataFrame:
        """Drops rows whose index is in `ids_to_remove` from any DataFrame."""
        keep_mask = ~dataset.index.isin(ids_to_remove)
        return dataset.loc[keep_mask]

    def csv_report(self, filename: str):
        """
        Saves a per-group summary of all 4 quality criteria as percentages,
        computed from the pre-cleaning report (self.data) - nothing is
        dropped to produce this, it's purely descriptive.

        Columns: total signals, % failing HR check, % failing peak-count
        check, mean/std % fiducials properly detected, mean/std combined
        score, % flagged for removal overall, and the count actually
        removed (available only after detect() has run).
        """
        rows = {}
        for group, df in self.data.items():
            n_signals = len(df)
            n_removed = len(self.remove.get(group, [])) if self.remove else np.nan
            rows[group] = {
                "n_signals": n_signals,
                "pct_fail_HR": 100 * df["checkHR"].sum() / n_signals,
                "pct_fail_peak_count": 100 * df["checkSP"].sum() / n_signals,
                "mean_pct_fiducials_detected": df["numberProperFiducials"].mean(),
                "std_pct_fiducials_detected": df["numberProperFiducials"].std(),
                "mean_combined_score": df["combinedScore"].mean(),
                "std_combined_score": df["combinedScore"].std(),
                "pct_flagged_for_removal": 100 * df["report"].sum() / n_signals,
                "n_removed": n_removed,
            }
        report_df = pd.DataFrame.from_dict(rows, orient="index")
        report_df.to_csv(filename)
        return report_df
