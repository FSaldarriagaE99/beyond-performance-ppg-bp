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
from local_paths import PULSE_DB_SUP_DIR, GS_RESULT_PAPER, SHAP_RESULTS_PAPER
from data import load_PulseDB_sup_ds
from models import build_lgbm
from config import ExperimentConfig, load_config, stress_test_params
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
    
    print(df_train.info())
    """
    df_train = preprocessing.downsample_per_patient(df_train, patient_col="Subject", proportion = 0.1)
    print(df_train.info())
    """
    # build the model
    """
    grid_path = GS_RESULT_PAPER / "Full_Grid_Randomized_search_3targets.json"
    cfg = load_config(grid_path)
    """
    cfg = ExperimentConfig(n_splits=5,
                           random_state=42,
                           experiment_name="Aggressive_fit_shap_Original_ds",
                           verbose = -1,
                           model_params=stress_test_params)
    np.random.seed(cfg.random_state)
    lgbm = build_lgbm(cfg)

    # build the pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", lgbm)
    ])

    # Splitting and dropping
    id_cols = ["Subject", "Age", "Gender", "Height", "Weight", "BMI", "SF"]
    targets = ["MAP"] #["SBP", "DBP", "MAP"]
    id_cols.extend(targets)
    groups = df_train["Subject"]
    X_train = df_train.drop(columns=id_cols)
    
    Y_train = df_train[targets]

    # =========================================================
    # Shapley Analysis
    #==========================================================
    Shap_path = SHAP_RESULTS_PAPER/"StressTest_Original"

    for target in targets:
        Y_t = Y_train[target]
        
        print(f"********** Analizing {target} **********")
        start = time.time()
        avg_rank, rank_matrix, avg_abs_shap, abs_shap_matrix, avg_raw_shap, raw_shap_matrix, rank_diff_matrix, feature_names = sa.shap_rank_stability(pipeline, 
                                                                                                                    X_train, 
                                                                                                                    Y_t,
                                                                                                                    groups = groups, 
                                                                                                                    n_iter=30, 
                                                                                                                    save_path=Shap_path/f"{target}")
        end = time.time()
        print(f"Execution time: {end - start:.3f} seconds")
        print("end", "\n")

    
    