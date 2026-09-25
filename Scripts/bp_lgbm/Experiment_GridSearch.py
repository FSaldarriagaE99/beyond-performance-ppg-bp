############ EXPERIMENT GRID SEARCH ###########################
#                                                             #
# This script is for executing the grid search only.          #
#                                                             #
###############################################################
"""
EXPERIMENT DESCRIPTION:

- Grid Search using Hyperion.
- Original dataset (without cleaning)
- Settings: 5 folds CV - group patient wise splitting, full grid search, optimization for the three targets (SBP, DBP, MAP).
- Preprocessing details: Inputation strategy is meadian based on patient entries! Scaler is fitted only on training set.
- Imputation for labels:
- It appears that MAP is full, so I am going to fill those NaNs with the formula for calculating MAP based on DBP and SBP.
- If by any chance, both SBP and DBP are missing, I am going to impute one of them based on the mean of the adjacent 5 values 
    for SBP and use the method above to calculate the DBP.

"""
import numpy as np
import preprocessing
import gs
from pathlib import Path
from local_paths import PULSE_DB_SUP_DIR
from local_paths import GS_RESULT_PAPER
from data import load_PulseDB_sup_ds
from config import ExperimentConfig, lightGBM_default_params, save_config, lightGBM_full_grid_3target
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from models import build_lgbm


if __name__ == "__main__":

    # =========================================================
    # Paths
    #==========================================================
    #train_original_path = Path(r"data_features") / "Features_VitalDB_Train_Subset.h5"
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
    df = load_PulseDB_sup_ds(train_original_path, feature_names=feature_names)
    
    print(df.info())
    print(df.head())
    

    # =========================================================
    # Check NaNs and fill them
    #==========================================================
    """
    print("************************************************************")
    print("Check NaNs")
    print("************************************************************")

    # Check if any NaN at all
    print("Is there any NaN in: df_features_mean?")
    print(df.isna().any().any())
    # Count total number of NaNs
    print(df.isna().sum())
    """
    
    # Fill X NaNs
    df = preprocessing.median_impute_patientwise(df, patient_col= "Subject")

    # =========================================================
    # Downsample the dataset --> balanced based on the subject
    #==========================================================
    """
    df_downsampled = preprocessing.downsample_per_patient(df, patient_col="Subject", proportion = 0.1)
    print(df_downsampled.info())
    """
    
    # =========================================================
    # Initialize model
    #==========================================================
    # drop ID columns before training
    id_cols = ["Subject", "Age", "Gender", "Height", "Weight", "BMI", "SF"]
    targets = ["SBP", "DBP", "MAP"]
    id_cols.extend(targets)
    print(id_cols)
    groups = df["Subject"]
    Y_train = df[targets]
    X_train = df.drop(columns=id_cols)

    # build the model
    cfg = ExperimentConfig(n_splits=5,
                           random_state=42,
                           experiment_name="Full_Grid_Randomized_search_3targets_Full_DS",
                           verbose = -1,
                           model_params=lightGBM_default_params,
                           n_jobs=1)
    np.random.seed(cfg.random_state)
    lgbm = build_lgbm(cfg)
    model = MultiOutputRegressor(lgbm)

    # build the pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    # =========================================================
    # Grid Search, standarization and CV happens inside.
    #==========================================================
    print("**********************************************************************it is running until here")
    search, results_df = gs.run_grid_search(pipeline=pipeline,
                                            X=X_train,
                                            y=Y_train,
                                            groups=groups,
                                            param_grid=lightGBM_full_grid_3target,
                                            n_splits=cfg.n_splits,
                                            cv_type= "group",
                                            save_results=True,
                                            verbose=0,
                                            search_mode="random",
                                            n_iter=50,
                                            scoring="neg_mean_squared_error",
                                            n_jobs=-1,
                                            random_state=cfg.random_state,
                                            )
    
    print("Best parameters:", search.best_params_)
    print("Best CV score (MSE):", -search.best_score_)
    print("Best CV score (RMSE):", (-search.best_score_)**0.5)

    best_params_clean = {k.replace("model__", ""): v 
                     for k, v in search.best_params_.items()}
    print(best_params_clean)

    # Update config with best hyperparameters
    cfg.model_params.update(best_params_clean)
    save_config(cfg, Path(GS_RESULT_PAPER / f"{cfg.experiment_name}.json"))