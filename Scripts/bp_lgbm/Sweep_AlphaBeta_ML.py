############ ALPHA/BETA SENSITIVITY SWEEP — FILTERED VS UNFILTERED ##################
#                                                                                    #
# Same design as Sweep_Threshold_ML.py, but sweeping the criterion-4 score weights  #
# (alpha = w_consistency, beta = w_alignment = 1 - alpha) at a FIXED discard        #
# threshold of 90 (thres_fiducials = thres_score = 90), instead of sweeping the     #
# threshold at fixed weights. See that script's header for the full rationale -     #
# repeated briefly here:                                                           #
#                                                                                    #
#   - For each alpha in {0, 0.25, 0.5, 0.75, 1}: filter train/test in memory via    #
#     the paired dropped-ID JSONs (Scripts/fiducial_quality_filtering/              #
#     sweep_alpha_beta.py output, already built at thres=90 for these five alphas). #
#   - Train unfiltered-train (once, reused across alphas) and alpha-filtered-train  #
#     models, both evaluated on the SAME alpha-filtered test set.                   #
#   - Bootstrap the paired difference in test R2, resamples re-allocated FRESH per  #
#     alpha (test-subject pool changes with alpha, same as with threshold).         #
#   - Comparison is filtered-vs-unfiltered AT EACH alpha, not comparable ACROSS     #
#     alphas (test population differs) - intentional, not worked around.           #
#                                                                                    #
# Only R2 is tracked. All R2 quantities reported as percentages / percentage        #
# points (_pct / _pp columns).                                                     #
#                                                                                    #
# Output layout (local_paths.ALPHA_BETA_SWEEP_ML_RESULTS):                         #
#   alpha_<a>/Distribution_R2_unfiltered.csv   1000 rows x {SBP,DBP,MAP} (raw R2)  #
#   alpha_<a>/Distribution_R2_filtered.csv     1000 rows x {SBP,DBP,MAP} (raw R2)  #
#   alpha_<a>/PointEstimates_R2_pct.csv        arm x {SBP,DBP,MAP}, R2 as % on     #
#                                               full (non-resampled) test set      #
#   summary.csv   one row per (alpha, target): paired-diff point estimate, SD,     #
#                 MDE, 95%/90% CI, equivalence verdict (bootstrap_stats.py)        #
###############################################################################

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import preprocessing
import eval
from local_paths import PULSE_DB_SUP_DIR, GS_RESULT_PAPER, ALPHA_BETA_SWEEP_ML_RESULTS, FILTERING_SENSITIVITY_DIR
from data import load_PulseDB_sup_ds
from config import load_config
from models import build_lgbm
from bootstrap_stats import paired_diff_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fiducial_quality_filtering"))
from sweep_common import load_subject_ids          # noqa: E402
from rebuild_clean_features import keep_mask_from_dropped_json  # noqa: E402

ALPHAS       = [0.0, 0.25, 0.5, 0.75, 1.0]  # beta = 1 - alpha
FIXED_THRESHOLD = 90                          # thres_fiducials = thres_score, held fixed
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

TRAIN_SUBSET_NAME = "VitalDB_Train_Subset"
TEST_SUBSET_NAME  = "VitalDB_CalFree_Test_Subset"


def prepare_xy(df):
    """Impute, encode Gender, return X, Y. Mirrors Experiment_Bootstrap_Filtering.py."""
    df = preprocessing.median_impute_patientwise(df, patient_col="Subject")
    df["Gender"] = df["Gender"].astype("category")
    X = df.drop(columns=["Subject", "SF"] + TARGETS)
    Y = df[TARGETS]
    return X[FEATURE_NAMES], Y


def load_keep_mask(subset_name: str, features_path: Path, alpha: float) -> np.ndarray:
    json_path = FILTERING_SENSITIVITY_DIR / subset_name / "dropped_signals" / "alpha_beta" / f"alpha_{alpha:.2f}.json"
    with open(json_path) as f:
        payload = json.load(f)
    params = payload["parameters"]
    assert params["thres_fiducials"] == FIXED_THRESHOLD and params["thres_score"] == FIXED_THRESHOLD, (
        f"{json_path} was built at threshold {params['thres_fiducials']}/{params['thres_score']}, "
        f"expected {FIXED_THRESHOLD} - wrong JSON for this sweep."
    )
    subjects = load_subject_ids(features_path)
    return keep_mask_from_dropped_json(payload, subjects)


if __name__ == "__main__":

    # =========================================================
    # Paths / config
    # =========================================================
    train_features_path = PULSE_DB_SUP_DIR / f"Features_{TRAIN_SUBSET_NAME}.h5"
    test_features_path  = PULSE_DB_SUP_DIR / f"Features_{TEST_SUBSET_NAME}.h5"
    grid_path            = GS_RESULT_PAPER / "Full_Grid_Randomized_search_3targets.json"

    cfg = load_config(grid_path)
    print(f"Loaded config: {grid_path.name}")

    # =========================================================
    # Load full (unfiltered) train and test sets once.
    # =========================================================
    print("\nLoading full train/test sets...")
    df_train_full = load_PulseDB_sup_ds(train_features_path, feature_names=FEATURE_NAMES)
    df_test_full  = load_PulseDB_sup_ds(test_features_path,  feature_names=FEATURE_NAMES)
    print(f"  Train: {len(df_train_full)} signals, {df_train_full['Subject'].nunique()} subjects")
    print(f"  Test:  {len(df_test_full)} signals, {df_test_full['Subject'].nunique()} subjects")

    # =========================================================
    # Unfiltered-train model: training data never changes across alphas,
    # so it is trained ONCE and reused, evaluated against each alpha's
    # own filtered test set below.
    # =========================================================
    print("\nTraining unfiltered-train model (once, reused for every alpha)...")
    X_train_full, Y_train_full = prepare_xy(df_train_full)
    models_unfiltered = {}
    for target in TARGETS:
        lgbm = build_lgbm(cfg)
        lgbm.fit(X_train_full, Y_train_full[target])
        models_unfiltered[target] = lgbm
        print(f"  Trained unfiltered/{target}")

    ALPHA_BETA_SWEEP_ML_RESULTS.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for alpha in ALPHAS:
        beta = 1.0 - alpha
        print(f"\n{'=' * 60}\nAlpha {alpha:.2f} (beta {beta:.2f}), threshold {FIXED_THRESHOLD}\n{'=' * 60}")
        alpha_dir = ALPHA_BETA_SWEEP_ML_RESULTS / f"alpha_{alpha:.2f}"
        alpha_dir.mkdir(parents=True, exist_ok=True)

        # -----------------------------------------------------
        # Filter train and test in memory via the dropped-ID JSON masks
        # -----------------------------------------------------
        train_keep_mask = load_keep_mask(TRAIN_SUBSET_NAME, train_features_path, alpha)
        test_keep_mask  = load_keep_mask(TEST_SUBSET_NAME,  test_features_path,  alpha)

        df_train_filtered = df_train_full.loc[train_keep_mask].reset_index(drop=True)
        df_test_filtered  = df_test_full.loc[test_keep_mask].reset_index(drop=True)
        print(f"  Train filtered: {len(df_train_filtered)} / {len(df_train_full)} signals kept")
        print(f"  Test filtered:  {len(df_test_filtered)} / {len(df_test_full)} signals kept, "
              f"{df_test_filtered['Subject'].nunique()} subjects")

        # -----------------------------------------------------
        # Train the alpha-filtered-train model
        # -----------------------------------------------------
        X_train_filt, Y_train_filt = prepare_xy(df_train_filtered)
        models_filtered = {}
        for target in TARGETS:
            lgbm = build_lgbm(cfg)
            lgbm.fit(X_train_filt, Y_train_filt[target])
            models_filtered[target] = lgbm
            print(f"  Trained filtered_alpha{alpha:.2f}/{target}")

        # -----------------------------------------------------
        # Prepare the alpha-filtered test set (same set evaluated by both arms)
        # -----------------------------------------------------
        df_test_eval = preprocessing.median_impute_patientwise(df_test_filtered, patient_col="Subject")
        df_test_eval["Gender"] = df_test_eval["Gender"].astype("category")
        X_test = df_test_eval[FEATURE_NAMES]
        Y_test = df_test_eval[TARGETS]

        # -----------------------------------------------------
        # Point estimates (R2 only) on the full, non-resampled alpha-filtered test set
        # -----------------------------------------------------
        point_r2 = {"unfiltered": {}, "filtered": {}}
        for target in TARGETS:
            point_r2["unfiltered"][target] = float(
                eval.compute_metrics(Y_test[target].values, models_unfiltered[target].predict(X_test))["R2"]
            )
            point_r2["filtered"][target] = float(
                eval.compute_metrics(Y_test[target].values, models_filtered[target].predict(X_test))["R2"]
            )
        pd.DataFrame(point_r2).mul(100.0).to_csv(alpha_dir / "PointEstimates_R2_pct.csv", index_label="target")

        # -----------------------------------------------------
        # Bootstrap: fresh 1000 subject-level resamples for THIS alpha's
        # test-subject pool. Same resample list reused for both arms.
        # -----------------------------------------------------
        resamples = eval.allocate_bootstrap_resamples(
            df_test_eval["Subject"], n_resamples=N_RESAMPLES, random_state=RANDOM_STATE
        )
        subject_index = eval.build_subject_index(df_test_eval, subject_col="Subject")

        r2_dist = {"unfiltered": {t: [] for t in TARGETS}, "filtered": {t: [] for t in TARGETS}}

        for resample in tqdm(resamples, total=N_RESAMPLES, desc=f"alpha_{alpha:.2f} bootstrap"):
            sample = eval.build_bootstrap_sample(df_test_eval, resample, subject_index)
            X_boot = sample[FEATURE_NAMES]
            for target in TARGETS:
                y_true = sample[target].values
                r2_dist["unfiltered"][target].append(
                    float(eval.compute_metrics(y_true, models_unfiltered[target].predict(X_boot))["R2"])
                )
                r2_dist["filtered"][target].append(
                    float(eval.compute_metrics(y_true, models_filtered[target].predict(X_boot))["R2"])
                )

        dist_unfiltered = pd.DataFrame(r2_dist["unfiltered"])
        dist_filtered   = pd.DataFrame(r2_dist["filtered"])
        dist_unfiltered.to_csv(alpha_dir / "Distribution_R2_unfiltered.csv", index=False)
        dist_filtered.to_csv(alpha_dir / "Distribution_R2_filtered.csv", index=False)

        # -----------------------------------------------------
        # Paired diff (filtered - unfiltered), per target, via bootstrap_stats
        # -----------------------------------------------------
        for target in TARGETS:
            diff_pp = (dist_filtered[target] - dist_unfiltered[target]).to_numpy() * 100.0
            observed_diff_pp = (point_r2["filtered"][target] - point_r2["unfiltered"][target]) * 100.0
            stats = paired_diff_stats(diff_pp, observed_diff_pp)
            summary_rows.append({
                "alpha": alpha,
                "beta": beta,
                "threshold": FIXED_THRESHOLD,
                "target": target,
                "n_test_subjects": int(df_test_eval["Subject"].nunique()),
                "n_test_signals": int(len(df_test_eval)),
                "r2_unfiltered_point_pct": point_r2["unfiltered"][target] * 100.0,
                "r2_filtered_point_pct": point_r2["filtered"][target] * 100.0,
                **stats,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = ALPHA_BETA_SWEEP_ML_RESULTS / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")

    print("\n=== Alpha/beta sensitivity - paired diff (filtered - unfiltered), pp of R2 ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.2f}"):
        print(summary_df.to_string(index=False))
