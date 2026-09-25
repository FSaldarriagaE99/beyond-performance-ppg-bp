############ BOOTSTRAP ARTIFACT GENERATOR #############################
#                                                                      #
# Trains three LightGBM models on the original training set:          #
#   - ppg      : 28 PPG features only                                  #
#   - demo     : demographics only (Age, Gender, Height, Weight, BMI)  #
#   - ppg_demo : PPG features + demographics                           #
#                                                                      #
# Bootstrap resamples are pre-allocated ONCE (subject-level, with      #
# replacement) and reused across all three models and all 3 targets.  #
#                                                                      #
# Saves per model per target:                                          #
#   Distribution_{target}.csv  — 1000 rows × metrics (raw resamples)  #
#   PointEstimates_{target}.csv — metrics on the full test set         #
#                                                                      #
# CI computation is intentionally separated into                       #
# Experiment_Bootstrap_CI.py so it can be rerun cheaply.              #
########################################################################

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import preprocessing
import eval
from local_paths import PULSE_DB_SUP_DIR, PERFORMANCE_RESULTS_PAPER, GS_RESULT_PAPER
from data import load_PulseDB_sup_ds
from config import load_config
from models import build_lgbm
from sklearn.model_selection import train_test_split

N_RESAMPLES  = 1000
RANDOM_STATE = 42
TARGETS      = ["SBP", "DBP", "MAP"]

FEATURE_NAMES = [
    "IPR", "Tsp", "TWRRF25", "TWRRF50", "Tsw25",
    "Tsw50", "Tsw75", "Tdw25", "Tdw50", "Tdw75",
    "AUCpi", "IPA", "Av-Au_ratio", "Ab-Aa_ratio", "Ac-Aa_ratio",
    "Ad-Aa_ratio", "Ap2-Ap1_ratio", "AGI", "Kurtosis", "Skewness",
    "L-H_ratio", "ShannonEntropy", "Tpp", "PRV", "FullKurt",
    "FullSkew", "sdPRV", "IQR_PRV",
]
DEMO_FEATURES     = ["Age", "Gender", "Height", "Weight", "BMI"]
PPG_DEMO_FEATURES = FEATURE_NAMES + DEMO_FEATURES


if __name__ == "__main__":

    # =========================================================
    # Paths
    # =========================================================
    train_path = PULSE_DB_SUP_DIR / "Features_VitalDB_Train_Subset.h5"
    test_path  = PULSE_DB_SUP_DIR / "Features_VitalDB_CalFree_Test_Subset.h5"
    res_root   = PERFORMANCE_RESULTS_PAPER / "Bootstrap"

    # =========================================================
    # Load
    # =========================================================
    df_train = load_PulseDB_sup_ds(train_path, feature_names=FEATURE_NAMES)
    df_test  = load_PulseDB_sup_ds(test_path,  feature_names=FEATURE_NAMES)

    # =========================================================
    # Impute
    # =========================================================
    df_train = preprocessing.median_impute_patientwise(df_train, patient_col="Subject")
    df_test  = preprocessing.median_impute_patientwise(df_test,  patient_col="Subject")

    # =========================================================
    # Gender as categorical
    # =========================================================
    df_train["Gender"] = df_train["Gender"].astype("category")
    df_test["Gender"]  = df_test["Gender"].astype("category")

    # =========================================================
    # Train / val split
    # =========================================================
    X_train_full = df_train.drop(columns=["SF"] + TARGETS)
    Y_train_full = df_train[TARGETS]

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_full, Y_train_full,
        test_size=0.1, random_state=RANDOM_STATE, shuffle=True,
        stratify=X_train_full["Subject"],
    )
    X_train = X_train.drop(columns=["Subject"])
    X_val   = X_val.drop(columns=["Subject"])

    # =========================================================
    # Full test set — feature slices (Subject kept for bootstrap)
    # =========================================================
    X_test_ppg      = df_test[FEATURE_NAMES]
    X_test_demo     = df_test[DEMO_FEATURES]
    X_test_ppg_demo = df_test[PPG_DEMO_FEATURES]
    Y_test          = df_test[TARGETS]

    # =========================================================
    # Train all models (one per model-type per target = 9 fits)
    # =========================================================
    grid_path = GS_RESULT_PAPER / "Full_Grid_Randomized_search_3targets.json"
    cfg = load_config(grid_path)

    print("Training models...")
    models = {m: {} for m in ("ppg", "demo", "ppg_demo")}
    X_slices = {
        "ppg":      X_train[FEATURE_NAMES],
        "demo":     X_train[DEMO_FEATURES],
        "ppg_demo": X_train[PPG_DEMO_FEATURES],
    }
    for model_key, X_tr in X_slices.items():
        for target in TARGETS:
            lgbm = build_lgbm(cfg)
            lgbm.fit(X_tr, Y_train[target])
            models[model_key][target] = lgbm
            print(f"  Trained {model_key}/{target}")

    # =========================================================
    # Point estimates on full test set — saved as artifact
    # =========================================================
    print("\nComputing point estimates...")
    test_X_map = {
        "ppg":      X_test_ppg,
        "demo":     X_test_demo,
        "ppg_demo": X_test_ppg_demo,
    }
    for model_key, X_te in test_X_map.items():
        for target in TARGETS:
            y_pred = models[model_key][target].predict(X_te)
            pt = eval.compute_metrics(Y_test[target].values, y_pred)
            pt_df = pd.DataFrame([{"metric": k, "value": v} for k, v in pt.items()])
            pt_path = res_root / model_key / f"PointEstimates_{target}.csv"
            pt_path.parent.mkdir(parents=True, exist_ok=True)
            pt_df.to_csv(pt_path, index=False)
            print(f"  Saved point estimates: {model_key}/{target}")

    # =========================================================
    # Bootstrap pre-allocation and subject index (ONCE)
    # =========================================================
    print(f"\nPre-allocating {N_RESAMPLES} bootstrap resamples...")
    resamples     = eval.allocate_bootstrap_resamples(
        df_test["Subject"], n_resamples=N_RESAMPLES, random_state=RANDOM_STATE
    )
    subject_index = eval.build_subject_index(df_test, subject_col="Subject")

    # =========================================================
    # Bootstrap loop — saves raw distributions only
    # =========================================================
    boot_metrics = {m: {t: [] for t in TARGETS} for m in models}

    print("Running bootstrap loop...")
    for resample in tqdm(resamples, total=N_RESAMPLES):
        sample = eval.build_bootstrap_sample(df_test, resample, subject_index)

        X_boot = {
            "ppg":      sample[FEATURE_NAMES],
            "demo":     sample[DEMO_FEATURES],
            "ppg_demo": sample[PPG_DEMO_FEATURES],
        }
        for model_key in models:
            for target in TARGETS:
                y_true = sample[target].values
                y_pred = models[model_key][target].predict(X_boot[model_key])
                boot_metrics[model_key][target].append(
                    eval.compute_metrics(y_true, y_pred)
                )

    # =========================================================
    # Save raw distributions
    # =========================================================
    print("\nSaving distributions...")
    for model_key in models:
        for target in TARGETS:
            dist_df = pd.DataFrame(boot_metrics[model_key][target])
            dist_path = res_root / model_key / f"Distribution_{target}.csv"
            dist_df.to_csv(dist_path, index=False)
            print(f"  Saved: {dist_path}")

    print("\nDone. Run Experiment_Bootstrap_CI.py to compute confidence intervals.")
