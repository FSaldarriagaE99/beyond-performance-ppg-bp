######## PREPROCESSING SCRIPT ################################################################
#                                                                                            #
# This script handles the splitting, standarization and things like that.                    #
#                                                                                            #
#                                                                                            #
##############################################################################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List
import h5py
from typing import List, Tuple, Union, Optional
from sklearn.model_selection import StratifiedShuffleSplit

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FUNCTIONS FOR SPLITTING         ~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def split_patients_by_signal_share(
    meta_df: pd.DataFrame,
    threshold: float = 0.80,             # target fraction (0–1)
    margin: float = 0.02,                # allowed tolerance
    patient_col: str = "Patient",        # your DF’s column name
    signals_col: str = "Total_signals",  # your DF’s column name
    prefer: str = "desc",                # "desc" = biggest patients first, "asc" = smallest first
) -> Dict[str, object]:
    """
    Greedy splitter: selects patients until cumulative share of signals
    falls within [threshold - margin, threshold + margin].

    If adding a patient would overshoot threshold + margin, that patient is skipped.
    """

    # total number of signals across all patients
    total_signals = int(meta_df[signals_col].sum())
    if total_signals == 0:
        return {
            "selected_ids": [],
            "remaining_ids": meta_df[patient_col].tolist(),
            "selected_share": 0.0,
            "note": "No signals in dataset."
        }

    # sort patients
    df = meta_df[[patient_col, signals_col]].copy()
    df = df.sort_values(signals_col, ascending=(prefer == "asc"))

    # target band
    low = (threshold - margin) * total_signals
    high = (threshold + margin) * total_signals

    selected: List[str] = []
    selected_signals = 0

    for _, row in df.iterrows():
        pid, sigs = row[patient_col], int(row[signals_col])
        tentative = selected_signals + sigs

        if tentative > high:  # overshoot → skip this patient
            continue

        selected.append(pid)
        selected_signals = tentative

        if low <= selected_signals <= high:  # stop once inside band
            break

    selected_share = selected_signals / total_signals
    remaining = [pid for pid in df[patient_col].tolist() if pid not in selected]

    return {
        "selected_ids": selected,
        "remaining_ids": remaining,
        "selected_share": selected_share,
        "selected_signals": selected_signals,
        "remaining_signals": total_signals - selected_signals,
        "total_signals": total_signals,
        "note": (
            "Within margin."
            if low <= selected_signals <= high
            else "Could not hit band exactly; stopped at closest under upper bound."
        ),
    }

def split_train_test(XY: pd.DataFrame, split_results: dict, patient_col: str = "Patient"):
    """
    Split XY dataframe into train/test sets based on split_results dict (from split_patients_by_signal_share).
    
    Args:
        XY: Combined dataframe with a patient column.
        split_results: Dict containing keys:
            - "selected_ids": patient IDs for train/test split
            - "remaining_ids": patient IDs for the other partition
            - "selected_share": share of signals assigned
            - "selected_signals": expected number of signals in selected partition
            - "remaining_signals": expected number of signals in remaining partition
            - "total_signals": total signals across dataset
        patient_col: column name for patient IDs (default="Patient")
    
    Returns:
        XY_selected, XY_remaining (two DataFrames)
    """
    selected_ids = set(split_results["selected_ids"])
    remaining_ids = set(split_results["remaining_ids"])
    
    # --- Split ---
    XY_selected = XY[XY[patient_col].isin(selected_ids)].reset_index(drop=True)
    XY_remaining = XY[XY[patient_col].isin(remaining_ids)].reset_index(drop=True)
    
    # --- Safety checks ---
    # 1) Signal count check
    n_sel, n_rem = len(XY_selected), len(XY_remaining)
    if n_sel != split_results["selected_signals"]:
        raise ValueError(f"❌ Selected signals mismatch: expected {split_results['selected_signals']}, got {n_sel}")
    if n_rem != split_results["remaining_signals"]:
        raise ValueError(f"❌ Remaining signals mismatch: expected {split_results['remaining_signals']}, got {n_rem}")
    
    # 2) Unique patient count check
    n_sel_ids = XY_selected[patient_col].nunique()
    n_rem_ids = XY_remaining[patient_col].nunique()
    if n_sel_ids != len(selected_ids):
        raise ValueError(f"❌ Unique patients mismatch in selected: expected {len(selected_ids)}, got {n_sel_ids}")
    if n_rem_ids != len(remaining_ids):
        raise ValueError(f"❌ Unique patients mismatch in remaining: expected {len(remaining_ids)}, got {n_rem_ids}")
    
    # --- Print summary ---
    print("✅ Safety check passed")
    print(f"Selected partition: {n_sel} signals, {n_sel_ids} patients "
          f"({split_results['selected_share']:.2%} share)")
    print(f"Remaining partition: {n_rem} signals, {n_rem_ids} patients")
    print(f"Total signals: {split_results['total_signals']}")
    print(f"Note: {split_results['note']}")
    
    return XY_selected, XY_remaining

def split_XY(
    XY: pd.DataFrame,
    target_cols: List[str],
    id_cols: List[str] = ["Patient", "segment_ID"],
    drop_cols: List[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split combined dataframe into X (features) and Y (target),
    automatically removing other BP target columns.
    """

    drop_cols = drop_cols or []

    # --- Drop other target columns if present ---
    all_targets = {"SBP", "DBP", "MAP"}
    current_target = target_cols[0]
    other_targets = list(all_targets - {current_target})
    total_drop = list(set(drop_cols + other_targets))

    # --- Build Y and X ---
    ids_in_df = [c for c in id_cols if c in XY.columns]
    Y = XY[ids_in_df + target_cols].copy()

    exclude_cols = set(target_cols + total_drop)
    X_cols = [c for c in XY.columns if c not in exclude_cols]
    X = XY[X_cols].copy()

    return X, Y

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FUNCTIONS FOR STANDARDIZING     ~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def standardize_z_score(df: pd.DataFrame, exclude_cols=None) -> pd.DataFrame:
    """
    Standardize DataFrame features using sklearn's StandardScaler (z-score).
    
    Args:
        df: Input DataFrame.
        exclude_cols: List of columns to exclude from scaling (e.g., 'Patient').
    
    Returns:
        DataFrame with scaled features, same column order.
        Fitted scaler object (to allow inverse transform later).
    """
    if exclude_cols is None:
        exclude_cols = []

    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in exclude_cols]
    
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df_scaled, scaler


def check_standarization(df: pd.DataFrame):

    features_df = df.drop(columns=["Patient"])
    # Column-wise mean and std
    print("Means:")
    print(features_df.mean())

    print("\nStandard deviations:")
    print(features_df.std())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FUNCTIONS FOR DATA IMPUTATION   ~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def fill_missing_bp(Y: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing SBP/DBP values based on MAP column.
    
    Rules:
    - If MAP has any NaN -> raise ValueError immediately.
    - If SBP or DBP is NaN but the other is present:
        Use MAP = DBP + 1/3*(SBP - DBP) to fill the missing one.
    - If both SBP and DBP are NaN:
        Estimate SBP as the mean of the 5 nearest segments
        from the same patient (non-NaN SBP values).
        Then compute DBP with formula: DBP = 1.5*MAP - 0.5*SBP.
    
    Args:
        Y: DataFrame with columns ["Patient", "MAP", "SBP", "DBP"].
    
    Returns:
        DataFrame with SBP and DBP filled in.
    """
    Y_filled = Y.copy()

    # --- Check MAP first ---
    if Y_filled["MAP"].isna().any():
        raise ValueError("❌ MAP column contains NaN values. Cannot proceed.")

    # --- Filter only rows where SBP or DBP are NaN ---
    mask = Y_filled["SBP"].isna() | Y_filled["DBP"].isna()
    missing_rows = Y_filled[mask]

    for idx, row in missing_rows.iterrows():
        patient_id = row["Patient"]
        sbp, dbp, map_val = row["SBP"], row["DBP"], row["MAP"]

        # Case 1: only one missing
        if pd.isna(sbp) and pd.notna(dbp):
            # MAP = DBP + (SBP - DBP)/3  -> SBP = 3*MAP - 2*DBP
            sbp_est = 3 * map_val - 2 * dbp
            Y_filled.at[idx, "SBP"] = sbp_est

        elif pd.isna(dbp) and pd.notna(sbp):
            # MAP = DBP + (SBP - DBP)/3  -> DBP = 1.5*MAP - 0.5*SBP
            dbp_est = 1.5 * map_val - 0.5 * sbp
            Y_filled.at[idx, "DBP"] = dbp_est

        # Case 2: both missing
        elif pd.isna(sbp) and pd.isna(dbp):
            # get same-patient rows (excluding current one)
            patient_rows = Y_filled[Y_filled["Patient"] == patient_id].drop(idx)

            # find 5 closest by index
            diffs = np.abs(patient_rows.index - idx)
            nearest_idx = diffs.nsmallest(5).index

            sbp_est = patient_rows.loc[nearest_idx, "SBP"].dropna().mean()
            if pd.isna(sbp_est):
                raise ValueError(f"❌ Not enough SBP data to estimate for patient {patient_id} at row {idx}")

            Y_filled.at[idx, "SBP"] = sbp_est
            dbp_est = 1.5 * map_val - 0.5 * sbp_est
            Y_filled.at[idx, "DBP"] = dbp_est

    return Y_filled

def median_impute_patientwise(df: pd.DataFrame, patient_col: str = "Patient") -> pd.DataFrame:
    """
    Fill NaN values with the median of the same feature within each patient.
    
    Args:
        df: Input DataFrame.
        patient_col: Column name identifying patients (default="Patient").
    
    Returns:
        DataFrame with NaNs imputed patient-wise.
    """
    df_imputed = df.copy() # This is for safety! This avoids modifications on the original DF in memory.
    # group by patient and apply median imputation
    df_imputed = df_imputed.groupby(patient_col, group_keys=False).apply(
        lambda g: g.fillna(g.median(numeric_only=True)) # The fillna function is native from pandas
    )

    # Note: When doing groupby and then apply a lmbda function, the function is only used over the samll DF that is generated by grouping
    return df_imputed.reset_index(drop=True)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FUNCTIONS FOR DOWNSAMPLING      ~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def downsample_per_patient(
    df: pd.DataFrame,
    patient_col: str = "Subject",
    proportion: float = 0.5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Downsample signals from each patient by a fixed proportion using scikit-learn style.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing a patient column.
    patient_col : str, default="Subject"
        Column identifying patient/group membership.
    proportion : float, default=0.5
        Proportion of signals to keep per patient (0 < proportion <= 1).
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Downsampled dataframe with the same proportion of signals kept per patient.
    """
    if not (0 < proportion <= 1):
        raise ValueError("proportion must be between 0 and 1")

    rng = np.random.default_rng(random_state)
    sampled_frames = []

    for patient_id, group_df in df.groupby(patient_col):
        n_signals = len(group_df)
        n_keep = int(np.floor(proportion * n_signals))

        # sample indices within this group
        keep_idx = rng.choice(group_df.index, size=n_keep, replace=False)
        sampled_frames.append(group_df.loc[keep_idx])

    df_resampled = pd.concat(sampled_frames).reset_index(drop=True)
    
    return df_resampled

def select_subjects_by_target_distribution(
    df: pd.DataFrame,
    subject_col: str,
    target_col: str,
    n_subjects: int,
    n_bins: int = 10,
    random_state: int = 42
) -> list:
    """
    Select a subset of subjects preserving the distribution of a target variable 
    (e.g., MAP or SBP) based on subject-level means.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing at least subject_col and target_col columns.
    subject_col : str
        Column name identifying each subject.
    target_col : str
        Continuous target variable to preserve (e.g., 'MAP' or 'SBP').
    n_subjects : int
        Number of subjects to keep.
    n_bins : int, default=10
        Number of bins to use for stratification (higher = finer matching).
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    selected_subjects : list
        List of selected subject IDs.
    """
    # --- Collapse dataset to subject-level mean of the target ---
    subj_stats = (
        df.groupby(subject_col)[target_col]
          .mean()
          .rename(f"{target_col}_mean")
          .reset_index()
    )

    # --- Create stratification bins on subject-level means ---
    subj_stats["bin"] = pd.qcut(
        subj_stats[f"{target_col}_mean"], 
        q=n_bins, 
        labels=False, 
        duplicates="drop"
    )

    # --- Stratified sampling of subjects ---
    n_total = len(subj_stats)
    if n_subjects > n_total:
        raise ValueError(f"Requested {n_subjects} subjects, but only {n_total} available.")
    if n_subjects >= len(subj_stats):
        return subj_stats[subject_col].values  # no need to split
    else:
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=1 - (n_subjects/n_total),
            random_state=random_state
        )
        train_idx, _ = next(sss.split(subj_stats[[subject_col]], subj_stats["bin"]))
        selected_subjects = subj_stats.iloc[train_idx][subject_col].tolist()

        return selected_subjects

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FUNCTIONS FOR ARREGLAR MACHETAZOS ~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def add_segment_id(X: pd.DataFrame,
                   Y: pd.DataFrame,
                   file_path: str,
                   dataset_type: str = "segments") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add Segment_ID column to X and Y DataFrames based on the first row length
    of the HDF5 'segments' dataset for each patient.

    Args:
        X, Y: pandas DataFrames with a "Patient" column.
        file_path: path to the .h5 file.
        dataset_type: which dataset to use inside each group (default="segments").
    
    Returns:
        (X_new, Y_new): DataFrames with Segment_ID column added.

    Raises:
        ValueError: if number of rows in X/Y for a patient does not match
                    the length of the first row in the HDF5 dataset.
    """
    X_new, Y_new = X.copy(), Y.copy()

    with h5py.File(file_path, "r") as f:
        for patient_id in f.keys():
            group = f[patient_id]
            if dataset_type not in group:
                raise KeyError(f"Dataset '{dataset_type}' not found in group '{patient_id}'")

            ds = group[dataset_type]
            first_row = ds[0]                  # shape = (L,)
            n_expected = len(first_row)        # number of elements in that row

            # rows in X and Y for this patient
            idx_X = X_new.index[X_new["Patient"] == patient_id]
            idx_Y = Y_new.index[Y_new["Patient"] == patient_id]

            if len(idx_X) != n_expected or len(idx_Y) != n_expected:
                raise ValueError(
                    f"Mismatch for patient {patient_id}: "
                    f"HDF5 first row length = {n_expected}, "
                    f"X rows = {len(idx_X)}, Y rows = {len(idx_Y)}"
                )

            # assign Segment_IDs
            segment_ids = [f"{patient_id}_seg{i}" for i in range(n_expected)]
            X_new.loc[idx_X, "Segment_ID"] = segment_ids
            Y_new.loc[idx_Y, "Segment_ID"] = segment_ids

            print(f"✅ Added Segment_ID for {patient_id} ({n_expected} segments)")

    return X_new, Y_new

def drop_abp_features(
    df: pd.DataFrame,
    patient_col: Union[str, List[str]] = "Patient"
) -> pd.DataFrame:
    """
    Drop the last half of rows for each group (defined by patient_col)
    in the DataFrame.

    Args:
        df: Input DataFrame.
        patient_col: Column name or list of column names to group by
                     (default="Patient").

    Returns:
        A new DataFrame with only the first half of rows kept for each group.
    """
    keep_rows = []
    for group_id, group in df.groupby(patient_col):
        n = len(group)
        half = n // 2  # floor division, keeps first half if odd number
        keep_rows.extend(group.index[:half])  # take the first half

    return df.loc[keep_rows].reset_index(drop=True)

import pandas as pd
from typing import Union, List, Optional

def merge_XY(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    on: Optional[Union[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Merge X and Y DataFrames into one.

    Args:
        X: Features DataFrame.
        Y: Targets DataFrame.
        on: Column name or list of column names to merge on. If None:
            - Merge assuming correct order patient-wise.

    Returns:
        Merged DataFrame.
    """
    if on is not None:
        # Merge on the user-specified column(s)
        merged = pd.merge(X, Y, on=on, suffixes=("_X", "_Y"))

    else:
        # Assume order matches within each patient
        if not all(X["Patient"].values == Y["Patient"].values):
            raise ValueError(
                "Patient order mismatch between X and Y. Cannot merge safely without keys."
            )
        merged = pd.concat(
            [X.reset_index(drop=True),
             Y.drop(columns=["Patient", "segment_ID"]).reset_index(drop=True)],
            axis=1
        )

    return merged

def reset_segment_ids(df: pd.DataFrame, patient_col="Patient", segment_col="segment_ID") -> pd.DataFrame:
    """
    Reset segment IDs so they are unique and sequential per patient.

    Args:
        df : DataFrame
            Input DataFrame with patient and segment columns.
        patient_col : str
            Name of the patient identifier column.
        segment_col : str
            Name of the segment identifier column.

    Returns:
        DataFrame with segment IDs reset per patient.
    """
    df = df.copy()
    df[segment_col] = (
        df.groupby(patient_col)
          .cumcount()  # 0,1,2,... per patient
    )
    return df

def find_uncommon_rows(
    df_full: pd.DataFrame,
    df_clean: pd.DataFrame,
    keys=["Patient", "segment_ID"]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Find uncommon rows between df_full and df_clean (row-wise).

    Args:
        df_full : DataFrame
            Original full dataset.
        df_clean : DataFrame
            Cleaned dataset (subset of df_full).

    Returns:
        uncommon : DataFrame
            Rows in df_full that are not present in df_clean.
    """
    # Find uncommon rows (row-wise comparison)
    merged = df_full.merge(df_clean, how="left", indicator=True)
    uncommon = merged.loc[merged["_merge"] == "left_only"].drop(columns="_merge")

    return uncommon

def align_segment_ids(df_ref: pd.DataFrame, df_to_update: pd.DataFrame,
                      seg_col: str = "segment_ID") -> pd.DataFrame:
    """
    Align segment_IDs in df_to_update with df_ref by comparing all other columns.
    
    Parameters
    ----------
    df_ref : DataFrame
        Reference dataframe with the correct segment_IDs.
    df_to_update : DataFrame
        DataFrame whose segment_IDs should be updated.
    seg_col : str
        Column name for segment identifier.
    
    Returns
    -------
    df_aligned : DataFrame
        Copy of df_to_update with segment_IDs updated to match df_ref
        wherever all other columns match.
    """
    # Columns to match on (everything except seg_col)
    match_cols = [c for c in df_ref.columns if c not in [seg_col]]
    
    # Merge on match_cols, but bring in seg_col from df_ref
    merged = df_to_update.merge(
        df_ref[[*match_cols, seg_col]],
        on=match_cols,
        how="left",
        suffixes=("", "_ref")
    )
    
    # Replace segment_ID in df_to_update with the aligned one
    df_aligned = df_to_update.copy()
    df_aligned[seg_col] = merged[f"{seg_col}_ref"].values
    
    return df_aligned

def drop_rows_by_keys(labels_df: pd.DataFrame,
                        df_to_drop: pd.DataFrame,
                        keys: list[str]) -> pd.DataFrame:
    """
    Drop rows from labels_df whose (Patient, segment_ID) pairs appear in uncommon_df.
    
    Parameters
    ----------
    labels_df : DataFrame
        Labels DataFrame that should be cleaned.
    df_to_drop : DataFrame
        DataFrame containing rows to exclude (with Patient + segment_ID keys).
    keys : list of str
        Column names to match on (e.g., ["Patient", "segment_ID"]).
    
    Returns
    -------
    cleaned_labels : DataFrame
        A copy of labels_df with uncommon rows removed.
    """
    # Build set of key tuples to drop
    keys_to_drop = set(map(tuple, df_to_drop[keys].to_numpy()))
    
    # Keep only rows not in keys_to_drop
    mask = [tuple(row) not in keys_to_drop for row in labels_df[keys].to_numpy()]
    cleaned_labels = labels_df[mask].reset_index(drop=True)
    
    return cleaned_labels
