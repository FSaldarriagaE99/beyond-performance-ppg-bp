############ LGBM DEMOGRAPHIC BASELINE ################################
#                                                                      #
# Demographic-only LightGBM baseline (Age, Gender, Height, Weight,    #
# BMI). Mirrors Demographic_Baseline_Model.py but replaces the linear  #
# regressor with LightGBM. Gender is passed as a pandas Categorical — #
# LightGBM handles it natively without OHE.                           #
#                                                                      #
# NOTE: Uses the same GS config as the PPG model. A dedicated GS for  #
# demographic features was not run.                                    #
########################################################################

import preprocessing
import eval
from pathlib import Path
from local_paths import PULSE_DB_SUP_DIR, PERFORMANCE_RESULTS_PAPER, GS_RESULT_PAPER
from data import load_PulseDB_sup_ds
from config import load_config
from models import build_lgbm
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    # =========================================================
    # Paths
    # =========================================================
    train_path = PULSE_DB_SUP_DIR / "Features_VitalDB_Train_Subset.h5"
    test_path  = PULSE_DB_SUP_DIR / "Features_VitalDB_CalFree_Test_Subset.h5"
    res_path   = PERFORMANCE_RESULTS_PAPER / "LGBM_Demographic_Baseline"
    res_path.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # Load
    # =========================================================
    feature_names = [
        "IPR", "Tsp", "TWRRF25", "TWRRF50", "Tsw25",
        "Tsw50", "Tsw75", "Tdw25", "Tdw50", "Tdw75",
        "AUCpi", "IPA", "Av-Au_ratio", "Ab-Aa_ratio", "Ac-Aa_ratio",
        "Ad-Aa_ratio", "Ap2-Ap1_ratio", "AGI", "Kurtosis", "Skewness",
        "L-H_ratio", "ShannonEntropy", "Tpp", "PRV", "FullKurt",
        "FullSkew", "sdPRV", "IQR_PRV",
    ]
    df_train = load_PulseDB_sup_ds(train_path, feature_names=feature_names)
    df_test  = load_PulseDB_sup_ds(test_path,  feature_names=feature_names)

    # =========================================================
    # Impute
    # =========================================================
    df_train = preprocessing.median_impute_patientwise(df_train, patient_col="Subject")
    df_test  = preprocessing.median_impute_patientwise(df_test,  patient_col="Subject")

    # =========================================================
    # Gender as categorical — LightGBM native support
    # =========================================================
    df_train["Gender"] = df_train["Gender"].astype("category")
    df_test["Gender"]  = df_test["Gender"].astype("category")

    # =========================================================
    # Feature / target selection
    # =========================================================
    demo_features = ["Age", "Gender", "Height", "Weight", "BMI"]
    targets       = ["SBP", "DBP", "MAP"]

    X_train = df_train[demo_features]
    X_test       = df_test[demo_features]
    Y_train = df_train[targets]
    Y_test       = df_test[targets]

    # =========================================================
    # Model — same GS config as PPG model
    # =========================================================
    grid_path = GS_RESULT_PAPER / "Full_Grid_Randomized_search_3targets.json"
    cfg  = load_config(grid_path)
    lgbm = build_lgbm(cfg)

    # =========================================================
    # Train / val split (sample-wise, stratified by subject)
    # =========================================================
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train,
        test_size=0.1, random_state=cfg.random_state, shuffle=True,
        stratify=df_train["Subject"],
    )
    _, X_sub_train, _, Y_sub_train = train_test_split(
        X_train, Y_train,
        test_size=0.1, random_state=cfg.random_state, shuffle=True,
        stratify=df_train.loc[X_train.index, "Subject"],
    )

    # =========================================================
    # Train and evaluate per target
    # =========================================================
    for target in targets:
        lgbm.fit(X_train, Y_train[target])

        Y_tr_sub_pred = lgbm.predict(X_sub_train)
        Y_val_pred    = lgbm.predict(X_val)
        Y_test_pred   = lgbm.predict(X_test)

        metrics_train = eval.evaluate(
            Y_sub_train[target].values, Y_tr_sub_pred,
            BA_path=res_path / f"BA_train_subset_{target}.png",
            R2_path=res_path / f"R2_train_subset_{target}.png",
        )
        metrics_val = eval.evaluate(
            Y_val[target].values, Y_val_pred,
            BA_path=res_path / f"BA_val_{target}.png",
            R2_path=res_path / f"R2_val_{target}.png",
        )
        metrics_test = eval.evaluate(
            Y_test[target].values, Y_test_pred,
            BA_path=res_path / f"BA_test_original_{target}.png",
            R2_path=res_path / f"R2_test_original_{target}.png",
        )

        results = {
            "Train":         metrics_train,
            "Val":           metrics_val,
            "Test_original": metrics_test,
        }
        eval.save_results_dict(results, res_path / f"Results_{target}.csv", "Data_Subset")

        print(f"=== {target} — TRAIN ===");  print(metrics_train)
        print(f"=== {target} — VAL ===");    print(metrics_val)
        print(f"=== {target} — TEST ===");   print(metrics_test)
