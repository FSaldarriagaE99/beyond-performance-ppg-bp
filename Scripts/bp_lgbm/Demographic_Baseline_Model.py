############ BASELINE MODEL ###################################
#                                                             #
# Here, a baseline model based on linear regression will be   #
#  trained for SBP, DBP and MAP as baseline comparison        #
#                                                             #
###############################################################

import preprocessing
import eval
from pathlib import Path
from local_paths import PULSE_DB_SUP_DIR, PERFORMANCE_RESULTS_PAPER
from data import load_PulseDB_sup_ds
from config import ExperimentConfig
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pprint import pprint

# No LightGBM grid-search config for this plain linear baseline - reuse
# ExperimentConfig's own default so the seed still has one source of truth.
RANDOM_STATE = ExperimentConfig().random_state

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
    
    print(df_train.info())
    print(df_test.info())

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
    df_test = preprocessing.median_impute_patientwise(df_test, patient_col= "Subject")

    # =========================================================
    # Initialize model
    #==========================================================
    # --- Preprocessing ---
    num_features = ["Age", "BMI", "Height", "Weight"]
    cat_features = ["Gender"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first"), cat_features),
        ]
    )
    
    # --- Build pipeline ---
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    # drop ID columns before training
    demographics_col = ["Subject", "Age", "Gender", "Height", "Weight", "BMI"]
    targets = ["SBP", "DBP", "MAP"]
    X_train = df_train[demographics_col]
    X_test = df_test[demographics_col]
    Y_train = df_train[targets]
    Y_test = df_test[targets]

    # --- Train/test split ---
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.1, random_state=RANDOM_STATE, shuffle=True, stratify=X_train["Subject"]
    )

    X_train = X_train.drop(columns=["Subject"])
    X_test = X_test.drop(columns=["Subject"])
    X_val = X_val.drop(columns=["Subject"])
    # =========================================================
    # Training and Eval
    #==========================================================
    for target in targets:
        Y_tr = Y_train[target]
        Y_v = Y_val[target]
        Y_t = Y_test[target]

        # --- Fit model ---
        model.fit(X_train, Y_tr)

        # --- Predict ---
        Y_pred_train = model.predict(X_train)
        Y_pred_val = model.predict(X_val)
        Y_pred_test = model.predict(X_test)

        res_path = PERFORMANCE_RESULTS_PAPER / r"Demographic_Baseline"
        metrics_train = eval.evaluate(Y_tr, Y_pred_train,
                                             BA_path=res_path / f"BA_train_subset_{target}.png",
                                             R2_path= res_path / f"R2_train_subset_{target}.png")
        metrics_val = eval.evaluate(Y_v, Y_pred_val, 
                                             BA_path=res_path / f"BA_val_{target}.png",
                                             R2_path= res_path / f"R2_val_{target}.png")
        metrics_test_original = eval.evaluate(Y_t, Y_pred_test, 
                                              BA_path=res_path / f"BA_test_original_{target}.png",
                                              R2_path= res_path / f"R2_test_original_{target}.png")

        print(f"=== Results for {target}, in Training ===")
        print(metrics_train)
        
        print(f"=== Results for {target}, in VAL ===")
        print(metrics_val)
        
        print(f"=== Results for {target}, in TEST ===")
        print(metrics_test_original)

        # Create dict for saving results
        results = {
            "Train": metrics_train,
            "Val": metrics_val,
            "Test_original": metrics_test_original
        }
        eval.save_results_dict(results, res_path/ f"Results_{target}.csv", "Data_Subset")