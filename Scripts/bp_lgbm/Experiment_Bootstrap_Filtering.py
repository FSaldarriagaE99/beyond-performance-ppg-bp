############ BOOTSTRAP — FILTERING COMPARISON #################################
#                                                                              #
# Compares two PPG-only models evaluated on the FILTERED test set:            #
#                                                                              #
#   ppg_original : trained on original training set                            #
#   ppg_filtered : trained on quality-filtered training set (_90)              #
#                                                                              #
# Both models use the Full_Grid_Randomized_search config.                      #
# Bootstrap resamples are pre-allocated ONCE from the filtered test set        #
# (subject-level, with replacement) and reused for both models, making         #
# the comparison exactly paired.                                               #
#                                                                              #
# Note: filtered test set has imbalanced signals per subject due to           #
# quality filtering. Subject-level bootstrap is kept (preserves within-        #
# subject correlation); resample sizes will vary more than in the balanced     #
# case but point estimates and CIs remain valid.                               #
#                                                                              #
# Saves per model per target:                                                  #
#   Distribution_{target}.csv   — 1000 rows × metrics                         #
#   PointEstimates_{target}.csv — metrics on full filtered test set            #
#                                                                              #
# Bootstrap_Filtering/                                                         #
#   ppg_original/                                                              #
#   ppg_filtered/                                                              #
###############################################################################

import numpy as np
import pandas as pd
from tqdm import tqdm
import preprocessing
import eval
from local_paths import PULSE_DB_SUP_DIR, PERFORMANCE_RESULTS_PAPER, GS_RESULT_PAPER
from data import load_PulseDB_sup_ds
from config import load_config
from models import build_lgbm

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


def prepare_train(df):
    """Impute, encode Gender, return full X_train, Y_train."""
    df = preprocessing.median_impute_patientwise(df, patient_col="Subject")
    df["Gender"] = df["Gender"].astype("category")
    X = df.drop(columns=["Subject", "SF"] + TARGETS)
    Y = df[TARGETS]
    return X[FEATURE_NAMES], Y


if __name__ == "__main__":

    # =========================================================
    # Paths
    # =========================================================
    orig_train_path      = PULSE_DB_SUP_DIR / "Features_VitalDB_Train_Subset.h5"
    filtered_train_path  = PULSE_DB_SUP_DIR / "Clean_Features_VitalDB_Train_Subset_90.h5"
    filtered_test_path   = PULSE_DB_SUP_DIR / "Clean_Features_VitalDB_CalFree_Test_Subset_90.h5"
    res_root             = PERFORMANCE_RESULTS_PAPER / "Bootstrap_Filtering"
    grid_path            = GS_RESULT_PAPER / "Full_Grid_Randomized_search_3targets.json"

    # =========================================================
    # Config
    # =========================================================
    cfg = load_config(grid_path)
    print(f"Loaded config: {grid_path.name}")

    # =========================================================
    # Load & prepare training sets
    # =========================================================
    print("\nLoading training sets...")
    df_orig     = load_PulseDB_sup_ds(orig_train_path,     feature_names=FEATURE_NAMES)
    df_filtered = load_PulseDB_sup_ds(filtered_train_path, feature_names=FEATURE_NAMES)

    X_train_orig,     Y_train_orig     = prepare_train(df_orig)
    X_train_filtered, Y_train_filtered = prepare_train(df_filtered)

    print(f"  Original train:  {len(X_train_orig)} signals")
    print(f"  Filtered train:  {len(X_train_filtered)} signals")

    # =========================================================
    # Load & prepare filtered test set
    # =========================================================
    print("\nLoading filtered test set...")
    df_test = load_PulseDB_sup_ds(filtered_test_path, feature_names=FEATURE_NAMES)
    df_test = preprocessing.median_impute_patientwise(df_test, patient_col="Subject")
    df_test["Gender"] = df_test["Gender"].astype("category")

    X_test = df_test[FEATURE_NAMES]
    Y_test = df_test[TARGETS]
    print(f"  Filtered test:   {len(X_test)} signals, {df_test['Subject'].nunique()} subjects")

    # =========================================================
    # Train models — one per model type per target
    # =========================================================
    print("\nTraining models...")
    models = {"ppg_original": {}, "ppg_filtered": {}}
    train_sets = {
        "ppg_original": (X_train_orig,     Y_train_orig),
        "ppg_filtered": (X_train_filtered, Y_train_filtered),
    }
    for model_key, (X_tr, Y_tr) in train_sets.items():
        for target in TARGETS:
            lgbm = build_lgbm(cfg)
            lgbm.fit(X_tr, Y_tr[target])
            models[model_key][target] = lgbm
            print(f"  Trained {model_key}/{target}")

    # =========================================================
    # Point estimates on full filtered test set
    # =========================================================
    print("\nComputing point estimates...")
    for model_key in models:
        for target in TARGETS:
            y_pred = models[model_key][target].predict(X_test)
            pt     = eval.compute_metrics(Y_test[target].values, y_pred)
            pt_df  = pd.DataFrame([{"metric": k, "value": v} for k, v in pt.items()])
            pt_path = res_root / model_key / f"PointEstimates_{target}.csv"
            pt_path.parent.mkdir(parents=True, exist_ok=True)
            pt_df.to_csv(pt_path, index=False)
            print(f"  Saved: {model_key}/{target}  (R2={pt['R2']:.3f}, MAE={pt['MAE']:.2f})")

    # =========================================================
    # Bootstrap pre-allocation — subject-level, with replacement
    # Allocated ONCE from the filtered test set; reused for both models
    # =========================================================
    print(f"\nPre-allocating {N_RESAMPLES} bootstrap resamples (subject-level)...")
    resamples     = eval.allocate_bootstrap_resamples(
        df_test["Subject"], n_resamples=N_RESAMPLES, random_state=RANDOM_STATE
    )
    subject_index = eval.build_subject_index(df_test, subject_col="Subject")

    # =========================================================
    # Bootstrap loop
    # =========================================================
    boot_metrics = {mk: {t: [] for t in TARGETS} for mk in models}

    print("Running bootstrap loop...")
    for resample in tqdm(resamples, total=N_RESAMPLES):
        sample = eval.build_bootstrap_sample(df_test, resample, subject_index)
        X_boot = sample[FEATURE_NAMES]

        for model_key in models:
            for target in TARGETS:
                y_true = sample[target].values
                y_pred = models[model_key][target].predict(X_boot)
                boot_metrics[model_key][target].append(
                    eval.compute_metrics(y_true, y_pred)
                )

    # =========================================================
    # Save raw distributions
    # =========================================================
    print("\nSaving distributions...")
    for model_key in models:
        for target in TARGETS:
            dist_df   = pd.DataFrame(boot_metrics[model_key][target])
            dist_path = res_root / model_key / f"Distribution_{target}.csv"
            dist_df.to_csv(dist_path, index=False)
            print(f"  Saved: {dist_path.relative_to(res_root.parent.parent)}")

    print("\nDone. Run Experiment_Bootstrap_CI_Filtering.py to compute CIs.")
