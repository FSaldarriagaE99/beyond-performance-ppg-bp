############ EXPERIMENT SHAPLEY VALUES ########################
#                                                             #
# Here, the Shap values experiment will be performed          #
#                                                             #
###############################################################
"""
Note: The implementation of SHAP is done using the shap library.

For the iterative process for the feature ranking is borrowed from:

"""

import numpy as np
import preprocessing
import shap_analysis as sa
import demo_strata_utils as ds
from local_paths import PULSE_DB_SUP_DIR, GS_RESULT_PAPER, SHAP_RESULTS_PAPER
from data import load_PulseDB_sup_ds
from models import build_lgbm
from config import load_config, ExperimentConfig, stress_test_params
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import time

if __name__ == "__main__":

    # =========================================================
    # Paths
    #==========================================================
    train_original_path = PULSE_DB_SUP_DIR / "Features_VitalDB_Train_Subset.h5"

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
    
    # =========================================================
    # Check NaNs and fill them
    #==========================================================
    """
    print("************************************************************")
    print("Check NaNs")
    print("************************************************************")
    
    # Check if any NaN at all
    print("Is there any NaN in: df_features_mean?")
    print(df_test.isna().any().any())
    # Count total number of NaNs
    print(df_test.isna().sum())
    """
    # Fill X NaNs
    df_train = preprocessing.median_impute_patientwise(df_train, patient_col= "Subject")
    """
    print(df_test.isna().any().any())
    """

    # =========================================================
    # Initialize model
    #==========================================================
    
    # build the model
    """
    grid_path = GS_RESULT_PAPER / "Full_Grid_Randomized_search_3targets.json"
    cfg = load_config(grid_path)
    """
    cfg = ExperimentConfig(n_splits=5,
                           random_state=42,
                           experiment_name="Overfitted_Shap_stratified",
                           verbose = -1,
                           model_params=stress_test_params)
    np.random.seed(cfg.random_state)
    lgbm = build_lgbm(cfg)

    # build the pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", lgbm)
    ])

    # =========================================================
    # Segmentation by demo thresholds
    #==========================================================
    thresholds = {
        "Age": [40, 60],
        "BMI": [25],
        "Gender": ["M", "F"]
    }
    dfs_dict_train = ds.segment_thresholds(df_train, rules=thresholds, subject_col_name="Subject", verbose=True)
    # =========================================================
    # Shapley Analysis
    #==========================================================
    strata_root = SHAP_RESULTS_PAPER / "Stratified_1variable_Stress_Test"

    for variable, strata_list in dfs_dict_train.items():
        ds.run_shap_experiment_stratified(
            strata_list=strata_list,
            variable_name=variable,
            output_root=strata_root,
            targets=["SBP", "DBP", "MAP"],
            pipeline=pipeline,
            n_iter=30
    )
