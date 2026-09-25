############ EXPERIMENT DEMOGRAPHIC STRATIFICATION ############
#                                                             #
# Here, THE ds WILL BE SPLITTED ACCORDING TO SOME STRATA      #
#                                                             #
###############################################################

import numpy as np
import demo_strata_utils as ds
import preprocessing
import eval
from pathlib import Path
from local_paths import PULSE_DB_SUP_DIR, GS_RESULT_PAPER, DEMOG_RESULTS_PAPER
from data import load_PulseDB_sup_ds
from config import ExperimentConfig, load_config, stress_test_params
from models import build_lgbm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone # to clone the pipeline freshh to avoid any cross contamination while looping

if __name__ == "__main__":

    # =========================================================
    # Paths
    #==========================================================
    train_original_path = PULSE_DB_SUP_DIR / "Features_VitalDB_Train_Subset.h5"
    test_original_path = PULSE_DB_SUP_DIR / "Features_VitalDB_CalFree_Test_Subset.h5"

    # =========================================================
    # Load Data
    #==========================================================
    feature_names = ["IPR", "Tsp", "TWRRF25", "TWRRF50", "Tsw25", 
                 "Tsw50", "Tsw75", "Tdw25", "Tdw50", "Tdw75", 
                 "AUCpi", "IPA",  "Av-Au_ratio", "Ab-Aa_ratio", "Ac-Aa_ratio", 
                 "Ad-Aa_ratio", "Ap2-Ap1_ratio", "AGI", "Kurtosis", "Skewness", 
                 "L-H_ratio", "ShannonEntropy", "Tpp", "PRV", "FullKurt", 
                 "FullSkew", "sdPRV", "IQR_PRV"]
    df_train = load_PulseDB_sup_ds(train_original_path, feature_names=feature_names)
    df_test = load_PulseDB_sup_ds(test_original_path, feature_names=feature_names)
    
    print(df_train.head())
    print(df_test.head())

    # =========================================================
    # Check NaNs and fill them
    #==========================================================
    """
    print("************************************************************")
    print("Check NaNs")
    print("************************************************************")
    
    # Check if any NaN at all
    print("Is there any NaN in: df_features_mean?")
    print(df_train.isna().any().any())
    # Count total number of NaNs
    print(df_train.isna().sum())
    """
    # Fill X NaNs
    df_train = preprocessing.median_impute_patientwise(df_train, patient_col= "Subject")
    df_test = preprocessing.median_impute_patientwise(df_test, patient_col= "Subject")

    # =========================================================
    # Segmentation by demo thresholds
    #==========================================================
    thresholds = {
        "Age": [40, 60],
        "BMI": [25],
        "Gender": ["M", "F"]
    }
    dfs_dict_train = ds.segment_thresholds(df_train, rules=thresholds, subject_col_name="Subject", verbose=True)
    dfs_dict_test = ds.segment_thresholds(df_test, rules=thresholds, subject_col_name="Subject", verbose=True)

    # =========================================================
    # Initialize model
    #==========================================================
    # build the model
    
    grid_path = GS_RESULT_PAPER / "Full_Grid_Randomized_search_3targets.json"
    cfg = load_config(grid_path)
    """
    cfg = ExperimentConfig(n_splits=5,
                           random_state=42,
                           experiment_name="Stress_Test_Demographics",
                           verbose = -1,
                           model_params=stress_test_params)
    """
    np.random.seed(cfg.random_state)
    lgbm = build_lgbm(cfg)

    # build the pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", lgbm)
    ])
    # Set the targets
    drop_cols = ["Age", "Gender", "Height", "Weight", "BMI", "SF"]
    targets = ["SBP", "DBP", "MAP"]

    # =========================================================
    # Call the loop for running the stratified analysis
    #==========================================================
    path = DEMOG_RESULTS_PAPER#/ r"Stress_Test_1variable"
    for target in targets:
        df_results = ds.run_analysis_for_target(
            dfs_dict_train,
            dfs_dict_test,
            target=target,
            model_fn=lambda: clone(pipeline),  # <-- creates a fresh copy each loop
            val_split_size=0.1,
            train_subset_size=0.1,
            evaluate_fn=eval.evaluate,
            base_results_dir=path,
            drop_features=drop_cols,
            random_state=cfg.random_state,
        )