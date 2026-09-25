######## DATA LOADER ################################################################
#                                                                                   #
# This scrip is meant for loading the data from the feature extraction made by Juan #
#                                                                                   #
#####################################################################################

from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from typing import List, Optional, Union

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FUNCTIONS FOR EXPLORING H5 FILE ~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def show_h5_attributes(h5_path, key_path):
    """
    Display the attributes of a dataset or group inside an HDF5 file.

    Args:
        h5_path (str): Path to the .h5 file.
        key_path (str): Path inside the HDF5 file, e.g., 'p000366/segments'.
    
    Returns:
        dict: Attributes as a dictionary {attr_name: value}.
    """
    attrs_dict = {}
    with h5py.File(h5_path, "r") as f:
        if key_path not in f:
            print(f"❌ Path '{key_path}' not found in file.")
            return {}
        
        obj = f[key_path]  # can be group or dataset
        print(f"📂 Inspecting: {key_path}")
        for key, val in obj.attrs.items():
            attrs_dict[key] = val
            print(f"  @{key} = {val}")
    return attrs_dict

def inspect_dataset(file_path, group_name, dataset_name, head=5):
    """
    Inspect a specific dataset inside a group of an HDF5 file.
    
    Args:
        file_path (str | Path): Path to the HDF5 file.
        group_name (str): Name of the group (e.g. "p000366").
        dataset_name (str): Name of the dataset (e.g. "mean_p000366", "segments").
        head (int): Number of rows to preview.
    
    Returns:
        dict: summary report with keys {shape, dtype, attributes, preview}.
    """
    file_path = Path(file_path)
    with h5py.File(file_path, "r") as f:
        if group_name not in f:
            raise KeyError(f"Group '{group_name}' not found in file.")
        
        group = f[group_name]
        if dataset_name not in group:
            raise KeyError(f"Dataset '{dataset_name}' not found in group '{group_name}'. "
                           f"Available: {list(group.keys())}")
        
        ds = group[dataset_name]  # h5py.Dataset
        arr = np.asarray(ds[()])  # load full dataset

        # collect attributes
        attrs = {k: v for k, v in ds.attrs.items()}

        # build preview
        preview = pd.DataFrame(arr).head(head)

        report = {
            "shape": ds.shape,
            "dtype": ds.dtype,
            "attributes": attrs,
            "preview": preview
        }

        # print a small report
        print(f"\n📊 Report for '{group_name}/{dataset_name}':")
        print(f"  shape       : {ds.shape}")
        print(f"  dtype       : {ds.dtype}")
        if attrs:
            print(f"  attributes  :")
            for k, v in attrs.items():
                print(f"    @{k} = {v}")
        else:
            print(f"  attributes  : None")
        print(f"\n  First {head} rows:")
        print(preview)

        return report
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FUNCTIONS FOR LOADING           ~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_patient_dataset(
    file_path: str | Path,
    dataset_type: str | None = None,  # optional now
    column_names: Optional[List[str]] = None     # user-specified names override
) -> pd.DataFrame:
    """
    Load one dataset type across all patients into a single DataFrame.

    Args:
        file_path: path to the .h5 file
        dataset_type: one of {"mean", "median", "segments"} or None.
                      If None and each patient group has only one dataset,
                      that dataset is used automatically.
        column_names: optional list of names for dataset columns.
                      If not provided, tries to use dataset's `features` attribute
                      if available and matches dataset width.
    

    Returns:
        pd.DataFrame with all rows concatenated, patient id included
    """
    file_path = Path(file_path)
    all_data = []

    with h5py.File(file_path, "r") as f:
        for patient_id in f.keys():
            group = f[patient_id]

            # --- dataset selection logic ---
            if dataset_type is None:
                keys = list(group.keys())
                if len(keys) != 1:
                    raise ValueError(
                        f"Patient {patient_id} has {len(keys)} datasets. "
                        f"Please specify dataset_type explicitly."
                    )
                ds_name = keys[0]
            else:    
                ds_name = dataset_type

            if ds_name not in group:
                print(f"⚠️ Skipping {patient_id}, dataset {ds_name} not found")
                continue

            ds = group[ds_name] # Both Group and Dataset objects from the h5py library have attributes 
            arr = np.asarray(ds[()]).T  # transpose: features x samples → samples x features

            # --- column names priority ---
            if column_names is not None:
                # user-specified override
                if len(column_names) != arr.shape[1]:
                    raise ValueError(
                        f"Length of column_names ({len(column_names)}) does not "
                        f"match number of features ({arr.shape[1]}) for patient {patient_id}"
                    )
                cols = column_names
            else:
                # try dataset attribute 'features'
                if "features" in ds.attrs:
                    features = ds.attrs["features"]
                    if len(features) == arr.shape[1]:
                        cols = [f.decode("utf-8") if isinstance(f, bytes) else str(f) for f in features]
                    else:
                        print(f"⚠️ Patient {patient_id}, 'features' length mismatch "
                              f"(attr={len(features)}, data={arr.shape[1]}). Using default.")
                        cols = list(range(arr.shape[1]))
                else:
                    # fallback into default numeric names
                    cols = list(range(arr.shape[1]))

            # wrap into dataframe
            df = pd.DataFrame(arr, columns=cols)
            df.insert(0, "Patient", patient_id)  # patient as first col
            all_data.append(df)

    if not all_data:
        raise RuntimeError(
            f"No data loaded from {file_path} with dataset_type={dataset_type}."
        )

    return pd.concat(all_data, ignore_index=True)

def load_group_attributes(
        file_path: str | Path,
        dataset_for_count: str = "mean"   # e.g. "segments", "mean", "median"
        ) -> pd.DataFrame:
    """
    Collect attributes from each patient group in an HDF5 file and
    return them as a DataFrame (one row per patient).
    Attribute names become column names.
    Adds an extra column 'Total_signals' based on dataset_for_count.
    Adds extra column named BMI.
    """
    file_path = Path(file_path)
    records = []

    with h5py.File(file_path, "r") as f:
        for patient_id in f.keys():
            group = f[patient_id]

            # --- collect attributes ---
            attrs = {k: _stringify(v) for k, v in group.attrs.items()}
            attrs["Patient"] = patient_id

            # --- count signals ---
            if dataset_for_count not in group:
                print(f"⚠️ Skipping {patient_id}, dataset {dataset_for_count} not found")
                total_signals = np.nan
            else:
                ds = group[dataset_for_count]
                # ds.shape = (features, samples)
                total_signals = ds.shape[1]

            attrs["Total_signals"] = total_signals
            records.append(attrs)

    if not records:
        raise RuntimeError(f"No group attributes found in {file_path}")

    df = pd.DataFrame(records)

    # Create the BMI column if Height/Weight exist
    if "Height" in df and "Weight" in df:
        df["Height"] = df["Height"] / 100  # cm → m
        df["BMI"] = (df["Weight"] / df["Height"]**2).round(2)

    # Ensure patient column is first
    cols = ["Patient"] + [c for c in df.columns if c != "Patient"]
    return df[cols]


def _stringify(val):
    """Convert HDF5 attribute to a JSON/pandas-friendly value."""
    if isinstance(val, (bytes, bytearray)):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, np.generic):  # numpy scalar
        return val.item()
    if isinstance(val, (list, tuple, np.ndarray)):
        arr = np.array(val)
        if arr.size == 1:            # unwrap single values
            return arr.item()
        return arr.tolist()
    return val

def load_PulseDB_sup_ds(
    file_path: str | Path,
    feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load flat HDF5 dataset into a tidy DataFrame. This h5 files come from the preprocessing of the Vital DB supplementary material from the Pulse DB

    Structure:
└─ [G] /
     ├─ [D] Age :: shape=(57600, 1), dtype=float32, no-filters, est=225.00 KB
     ├─ [D] BMI :: shape=(57600, 1), dtype=float32, no-filters, est=225.00 KB
     ├─ [D] DBP :: shape=(57600, 1), dtype=float32, no-filters, est=225.00 KB
     ├─ [D] Gender :: shape=(57600, 1), dtype=object, no-filters, est=450.00 KB 
     ├─ [D] Height :: shape=(57600, 1), dtype=float32, no-filters, est=225.00 KB
     ├─ [D] MAP :: shape=(57600, 1), dtype=float32, no-filters, est=225.00 KB   
     ├─ [D] PPG_features :: shape=(28, 57600), dtype=float64, no-filters, est=12.30 MB
     ├─ [D] SBP :: shape=(57600, 1), dtype=float32, no-filters, est=225.00 KB
     ├─ [D] SF :: shape=(57600, 1), dtype=float32, no-filters, est=225.00 KB
     ├─ [D] Subject :: shape=(57600, 1), dtype=object, no-filters, est=450.00 KB
     └─ [D] Weight :: shape=(57600, 1), dtype=float32, no-filters, est=225.00 KB

    Args
    ----
    file_path : str | Path
        Path to the .h5 file
    feature_names : list[str], optional
        Names for the PPG features (length must match number of rows in PPG_features)

    Returns
    -------
    df : pd.DataFrame
        DataFrame with metadata + target columns + PPG feature columns
    """
    file_path = Path(file_path)
    with h5py.File(file_path, "r") as f:
        # Load scalars (all shape (N,1))
        n_samples = f["Age"].shape[0]
        data_dict = {
            "Age": np.array(f["Age"]).reshape(-1),
            "BMI": np.array(f["BMI"]).reshape(-1),
            "DBP": np.array(f["DBP"]).reshape(-1),
            "Gender": np.array(f["Gender"]).astype(str).reshape(-1),
            "Height": np.array(f["Height"]).reshape(-1),
            "MAP": np.array(f["MAP"]).reshape(-1),
            "SBP": np.array(f["SBP"]).reshape(-1),
            "SF": np.array(f["SF"]).reshape(-1),
            "Subject": np.array(f["Subject"]).astype(str).reshape(-1),
            "Weight": np.array(f["Weight"]).reshape(-1),
        }

        # Load and transpose PPG features (28, N) → (N, 28)
        ppg_arr = np.array(f["PPG_Features"]).T

        # Assign names
        if feature_names is None:
            feature_names = [f"PPG_feat{i+1}" for i in range(ppg_arr.shape[1])]
        elif len(feature_names) != ppg_arr.shape[1]:
            raise ValueError(
                f"feature_names length {len(feature_names)} != number of PPG features {ppg_arr.shape[1]}"
            )

        for i, name in enumerate(feature_names):
            data_dict[name] = ppg_arr[:, i]

    # Assemble DataFrame with desired order
    df = pd.DataFrame(data_dict)

    ordered_cols = [
        "Subject", "Age", "Gender", "Height", "Weight",
        "BMI", "SF", "SBP", "DBP", "MAP"
    ] + feature_names

    return df[ordered_cols]

#====================================================================